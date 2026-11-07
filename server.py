#!/usr/bin/env python3
"""Local site for Malaysian passport extraction."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from extract import format_stay_fields

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = ROOT / "output"
SAVED = ROOT / "saved" / "test-image"
PYTHON = ROOT / ".venv" / "bin" / "python"
UPLOADS = ROOT / "input" / "uploads"
HOST = "127.0.0.1"
PORT = 8766
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

FILES = {
    "/": (WEB / "index.html", "text/html; charset=utf-8"),
    "/index.html": (WEB / "index.html", "text/html; charset=utf-8"),
    "/styles.css": (WEB / "styles.css", "text/css; charset=utf-8"),
    "/app.js": (WEB / "app.js", "text/javascript; charset=utf-8"),
}

state = {
    "running": False,
    "started": 0.0,
    "phase": "",
    "error": None,
    "source": None,
    "source_name": "",
}
lock = threading.Lock()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def file_mtime(name: str) -> float:
    path = OUT / name
    return path.stat().st_mtime if path.is_file() else 0.0


def snapshot() -> dict:
    with lock:
        running = state["running"]
        started = state["started"]
        phase = state["phase"]
        error = state["error"]
        source_name = state["source_name"]
    layer4 = read_json(OUT / "layer4_stays.json")
    saved = read_json(SAVED / "layer4_stays.json")

    def current(doc: dict | None, mtime: float) -> dict | None:
        if not doc or (running and mtime < started):
            return None
        return doc

    return {
        "running": running,
        "phase": phase,
        "error": error,
        "source": source_name,
        "layer4": current(layer4, file_mtime("layer4_stays.json")),
        "layer1": current(read_json(OUT / "layer1_name.json"), file_mtime("layer1_name.json")),
        "layer2": current(read_json(OUT / "layer2_attributes.json"), file_mtime("layer2_attributes.json")),
        "layer3": current(read_json(OUT / "layer3_image.json"), file_mtime("layer3_image.json")),
        "layer5": current(read_json(OUT / "layer5_dates.json"), file_mtime("layer5_dates.json")),
        "layer6": current(read_json(OUT / "layer6_stays.json"), file_mtime("layer6_stays.json")),
        "saved": saved,
    }


def safe_filename(filename: str) -> str:
    base = Path(filename).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-")
    return (base or "upload")[:80]


def parse_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("Choose a passport PDF or image to upload.")
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    if not message.is_multipart():
        raise ValueError("Choose a passport PDF or image to upload.")
    for part in message.iter_parts():
        filename = part.get_filename()
        if part.get_param("name", header="content-disposition") != "file" or not filename:
            continue
        payload = part.get_payload(decode=True) or b""
        if not payload:
            raise ValueError("That file is empty.")
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("Use a PDF or an image (PNG, JPEG, WEBP, GIF, BMP, or TIFF).")
        return filename, payload
    raise ValueError("Choose a passport PDF or image to upload.")


def save_upload(filename: str, payload: bytes) -> Path:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_filename(filename)}"
    path.write_bytes(payload)
    return path


def run_extract() -> None:
    try:
        if not PYTHON.is_file():
            raise RuntimeError("Missing .venv. Install requirements before starting the site.")
        with lock:
            source = state["source"]
        if not source or not Path(source).is_file():
            raise RuntimeError("Upload a Malaysian passport first.")
        process = subprocess.Popen(
            [str(PYTHON), str(ROOT / "extract.py"), "--source", str(source)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            with lock:
                state["phase"] = text
        code = process.wait()
        with lock:
            if code != 0:
                state["error"] = state["phase"] or f"Extractor exited with status {code}"
    except Exception as exc:
        with lock:
            state["error"] = str(exc)
    finally:
        with lock:
            state["running"] = False


def start_run(source: Path | None = None) -> str | None:
    with lock:
        if state["running"]:
            return "An extract is already running."
        if source is not None:
            state["source"] = source
            state["source_name"] = source.name
        if not state["source"] or not Path(state["source"]).is_file():
            return "Upload a Malaysian passport first."
        state["running"] = True
        state["started"] = time.time()
        state["phase"] = f"Reading {state['source_name']}"
        state["error"] = None
    threading.Thread(target=run_extract, daemon=True).start()
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(snapshot())
            return
        if path == "/portrait.jpg":
            image = OUT / "portrait.jpg"
            if not image.is_file():
                self.send_error(404)
                return
            body = image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/seals/") or path.startswith("/saved/seals/"):
            name = Path(path).name
            folder = SAVED / "seals" if path.startswith("/saved/seals/") else OUT / "seals"
            image = (folder / name).resolve()
            if image.parent != folder.resolve() or not image.is_file():
                self.send_error(404)
                return
            body = image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        item = FILES.get(path)
        if not item or not item[0].is_file():
            self.send_error(404)
            return
        body = item[0].read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", item[1])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/run":
            message = start_run()
            if message:
                self.send_json({"ok": False, "error": message}, 409)
                return
            self.send_json({"ok": True})
            return
        if path == "/api/layer4":
            self.save_layer4()
            return
        if path == "/api/layer6":
            self.save_layer6()
            return
        if path != "/api/upload":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json({"ok": False, "error": "Upload a passport file under 32 MB."}, 400)
            return
        try:
            filename, payload = parse_upload(self.headers.get("Content-Type") or "", self.rfile.read(length))
            saved = save_upload(filename, payload)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)
            return
        message = start_run(saved)
        if message:
            self.send_json({"ok": False, "error": message}, 409)
            return
        self.send_json({"ok": True})

    def save_layer4(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 200_000:
            self.send_json({"ok": False, "error": "That edit is too large."}, 400)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"ok": False, "error": "The edit could not be read."}, 400)
            return
        raw = payload.get("seals") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or len(raw) > 200:
            self.send_json({"ok": False, "error": "Send the list of seals."}, 400)
            return
        edits = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            file = str(item.get("file") or "").strip()
            if not file.startswith("seals/") or "/" in file[6:] or ".." in file:
                continue
            country = str(item.get("country") or "").strip()[:200]
            edits[file] = {
                "text": str(item.get("text") or "").strip()[:4000],
                "country": country or None,
            }
        path = OUT / "layer4_stays.json"
        current = read_json(path)
        if not current or not isinstance(current.get("seals"), list):
            self.send_json({"ok": False, "error": "There is no stamp text to edit."}, 400)
            return
        for seal in current["seals"]:
            if not isinstance(seal, dict):
                continue
            change = edits.get(seal.get("file"))
            if not change:
                continue
            seal["text"] = change["text"]
            seal["country"] = change["country"]
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        self.send_json({"ok": True})

    def save_layer6(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 200_000:
            self.send_json({"ok": False, "error": "That edit is too large."}, 400)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"ok": False, "error": "The edit could not be read."}, 400)
            return
        raw = payload.get("stays") if isinstance(payload, dict) else None
        if not isinstance(raw, list) or len(raw) > 100:
            self.send_json({"ok": False, "error": "Send the list of stays."}, 400)
            return
        drafts = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            departure = str(item.get("departure_country") or "").strip()[:200]
            arrival = str(item.get("arrival_country") or "").strip()[:200]
            if departure.casefold() == "malaysia" and arrival.casefold() == "malaysia":
                departure = ""
            drafts.append({
                "departure_country": departure,
                "arrival_country": arrival,
                "start": str(item.get("start") or ""),
                "end": str(item.get("end") or ""),
            })
        formatted = format_stay_fields(
            [value for draft in drafts for value in (draft["start"], draft["end"])]
        )
        stays = []
        for index, draft in enumerate(drafts):
            stays.append({
                "departure_country": draft["departure_country"],
                "arrival_country": draft["arrival_country"],
                "start": formatted[index * 2][:200],
                "end": formatted[index * 2 + 1][:200],
            })
        current = read_json(OUT / "layer6_stays.json") or {}
        document = {
            "layer": 6,
            "status": "finished",
            "model": current.get("model") or "",
            "source": current.get("source") or "",
            "stays": stays,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "layer6_stays.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.send_json({"ok": True})

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Passport extractor at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
