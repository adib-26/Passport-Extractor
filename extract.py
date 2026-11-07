#!/usr/bin/env python3
"""Extract a Malaysian passport.

A biodata page yields the holder name, printed details, and portrait.
A stamp page yields the seals, the dates on those seals, and the stays.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_SECRETS = ROOT / "secrets.env"
DEFAULT_OUT = ROOT / "output"
SAVED_DIR = ROOT / "saved"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.8-flash"
# Fast vision model for turning, leveling, place names, and the second reading.
VISION_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
QWEN_OCR_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
GLM_OCR_MODEL = "z-ai/glm-5.3-flashx"
OCR_MODEL = GLM_OCR_MODEL
READERS = (QWEN_OCR_MODEL, GLM_OCR_MODEL, MODEL)
STAY_MODEL = "qwen/qwen3-vl-8b-instruct"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SEAL_WORKERS = 6

LAYER1_PROMPT = """You read OCR text from a Malaysian passport scan.
Find the passport holder's name. It is printed on the biodata page, on the line beside Nama / Name, above Nationality and Date of birth.
That page may also show Pasport / Passport No and a machine-readable line containing P<MYS or the name written with < between words.

Ignore the cover, the observations page, visa stamps, and entry stamps.
Do not return a passport number, a date, a place, or any other field.
Copy the name from the OCR. Do not invent a person who is not written there.
Keep bin or binti. Use the machine-readable name only to fix a letter that the visual line already almost matches.

Return only JSON:
{"page": 2, "name": "FULL NAME AS PRINTED"}

If this file has no holder name, return {"page": null, "name": null}.

Pages:
"""

LAYER2_PROMPT = """Layer 1 has finished. The holder's name is already stored. Do not repeat it.
Read only the biodata page below, the same page where that name was printed.
Extract attributes printed on this page. Do not use other pages.

Return only JSON with this object. Use null for a field that is not printed on this page:
{"attributes":{"passport_number":null,"nationality":null,"date_of_birth":null,"place_of_birth":null,"sex":null,"height":null,"identity_number":null,"date_of_issue":null,"date_of_expiry":null,"issuing_office":null,"type":null,"country_code":null}}

Rules:
- Do not include the name, a nickname, or the machine-readable name.
- passport_number is required when No. Pasport / Passport No is printed. It is one letter plus eight digits, such as A16073400. Do not attach MYS or the identity number to it.
- identity_number is the MyKad or Kad Pengenalan number when it is printed.
- nationality is the country printed beside Warganegara / Nationality, such as MALAYSIA.
- sex is "LELAKI- Male" or "WANITA- Female". Do not append M or F.
- height is printed only on some older passports, such as 168 cm. Leave it null when it is absent.
- Keep dates as printed, such as 16 FEB 1936.
- issuing_office is the place or authority that issued the passport when it is printed on this page.
- type is the meaning of Jenis / Type, never the code letter. P means "Ordinary passport". A diplomatic code means "Diplomatic passport". An official code means "Official passport". country_code is MYS when it is printed.

Biodata page OCR:
"""

LAYER1_IMAGE_PROMPT = """This image is one passport biodata page or a description-of-holder page.
Read the holder's name printed beside Nama / Name, or beside Name of Holder / Nama Pembawa.
Copy that name as printed. Keep MR, MRS, bin, or binti when they are printed on that line.
Do not return a passport number, an identity-card number, a date, a place, or a profession.
Do not invent a name that is not printed.
Return only JSON: {"name": "FULL NAME AS PRINTED"}
If no holder name is printed, return {"name": null}.
"""

LAYER2_IMAGE_PROMPT = """This image is the passport biodata page or the description-of-holder page.
The holder's name is already stored. Do not repeat it.
Read only what is printed on this page.

Return only JSON. Use null for a field that is not printed:
{"attributes":{"passport_number":null,"nationality":null,"date_of_birth":null,"place_of_birth":null,"sex":null,"height":null,"identity_number":null,"date_of_issue":null,"date_of_expiry":null,"issuing_office":null,"type":null,"country_code":null}}

Rules:
- passport_number is No. Pasport / Passport No, one letter plus eight digits. Leave it null when this page has no passport number. Do not put the identity-card number there.
- identity_number is No. Kad Pengenalan / Identity Card No.
- date_of_birth is Tarikh lahir / Date of Birth, kept as printed.
- place_of_birth is Tempat lahir / Place of Birth.
- sex is "LELAKI- Male" or "WANITA- Female".
- height is Tinggi / Height as printed. When both a Malay line and an English line are printed, keep the English line, such as 5 feet 6 inches. Leave it null when it is absent.
- nationality is only the country beside a printed Warganegara / Nationality label. Country of residence is not nationality. Leave nationality null when that label is absent.
- date_of_birth is copied as printed. Do not add TH, ST, ND, RD, or a period that is not printed.
- date_of_expiry is the date beside Pasport ini berlaku sa-hingga / The validity of this passport expires, or Tarikh tamat.
- date_of_issue is the date beside Di-keluarkan / Issued at, including the date inside the passport-office issue stamp when that stamp is the issue date.
- issuing_office is the place beside Di-keluarkan di- / Issued at.
- type is the meaning of a printed Jenis / Type label. Leave it null when that label is absent. Do not guess Ordinary passport.
- country_code is the printed code such as MYS. Leave it null when no country code is printed. Do not infer MYS from the word Malaysia.
"""

LAYER2_LABEL_PROMPT = """Look only at the printed field labels on this passport page.
Return only JSON:
{"nationality_label":false,"type_label":false,"country_code_printed":false}

nationality_label is true only when Warganegara or Nationality is printed as a field label.
Country of residence does not count.
type_label is true only when Jenis or Type is printed as a field label.
country_code_printed is true only when a country code such as MYS is printed as a code.
The word Malaysia by itself does not count.
"""

ATTRIBUTE_FIELDS = (
    "passport_number",
    "nationality",
    "date_of_birth",
    "place_of_birth",
    "sex",
    "height",
    "identity_number",
    "date_of_issue",
    "date_of_expiry",
    "issuing_office",
    "type",
    "country_code",
)


def load_api_key(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(
            f"Missing {path.name}. Copy secrets.env.example to secrets.env and set OPENROUTER_API_KEY."
        )
    key = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "OPENROUTER_API_KEY":
            key = value.strip().strip('"').strip("'")
    if not key or key.lower().startswith("paste"):
        raise SystemExit(f"Set OPENROUTER_API_KEY in {path.name}.")
    return key


def _extract_json_block(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Model returned empty content")

    # Remove common markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # If the whole thing is already JSON, use it.
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned

    # Otherwise, try to locate the first plausible JSON block.
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")
    start_candidates = [i for i in (obj_start, arr_start) if i >= 0]
    if not start_candidates:
        raise ValueError(f"Could not find JSON in model output: {cleaned[:200]!r}")
    start = min(start_candidates)

    # Try object/array termination heuristics.
    obj_end = cleaned.rfind("}")
    arr_end = cleaned.rfind("]")
    end_candidates: list[int] = []
    if obj_end > start:
        end_candidates.append(obj_end)
    if arr_end > start:
        end_candidates.append(arr_end)
    if not end_candidates:
        raise ValueError(f"Could not find JSON end in model output: {cleaned[:200]!r}")

    end = max(end_candidates)
    return cleaned[start : end + 1]


def parse_json(text: str) -> dict:
    raw = _extract_json_block(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model JSON parse failed: {exc} | Snippet: {raw[:250]!r}") from exc

    if isinstance(data, list):
        return {"items": data}
    if not isinstance(data, dict):
        raise ValueError("Model JSON was not an object")
    return data


def clean_text(value) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" .,:;|-")
    if not text or text.lower() in {"null", "none", "n/a", "nil", "-"}:
        return None
    return text


def content_text(message: dict) -> str:
    content = message.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        text = "".join(parts)
    if text.strip():
        return text
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = []
        for item in details:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return ""


class OpenRouter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.extra = {
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        self.last_model = MODEL
        self.active_model = MODEL
        self.models_used = []

    def complete(self, prompt: str, max_tokens: int | None = None, images: list | None = None, model: str | None = None) -> dict:
        last_error: Exception | None = None
        chosen = model or self.active_model
        for attempt in range(3):
            try:
                return self._once(prompt, max_tokens, images, chosen)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                body = exc.response.text[:400] if exc.response is not None else ""
                if status == 401:
                    raise SystemExit(
                        "OpenRouter rejected the API key in secrets.env. "
                        "Paste a current key in that file and run the extractor again.\n" + body
                    ) from exc
                if status == 400 and "response_format" in self.extra:
                    self.extra.pop("response_format", None)
                    last_error = exc
                    continue
                last_error = exc
                if status == 429 or status >= 500:
                    time.sleep(3 * (attempt + 1))
                    continue
                break
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if "finish_reason=length" in str(exc):
                    break
                if "response_format" in str(exc) and "response_format" in self.extra:
                    self.extra.pop("response_format", None)
                    continue
                if attempt < 2:
                    time.sleep(2)
                    continue
                break
        raise SystemExit(f"OpenRouter request failed: {last_error}")

    def ask(self, prompt: str, images: list, model: str | None = None) -> dict | None:
        try:
            return self.complete(prompt, images=images, model=model)
        except SystemExit as exc:
            print(f"[Layer 4] the model did not answer: {exc}", flush=True)
            return None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Passport extractor",
        }

    def _once(self, prompt: str, max_tokens: int | None, images: list | None, model: str) -> dict:
        if images:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}})
        else:
            content = prompt
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            **self.extra,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if model in READERS and not model.startswith("deepseek/"):
            payload["reasoning"] = {"effort": "low"}
        response = requests.post(
            OPENROUTER_URL,
            headers=self._headers(),
            json=payload,
            timeout=600,
        )
        if response.status_code >= 400:
            raise requests.HTTPError(response=response)
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        stopped = choice.get("error")
        if stopped or choice.get("finish_reason") == "error":
            detail = stopped.get("message") if isinstance(stopped, dict) else stopped
            raise ValueError(f"Model stopped early: {detail or choice.get('finish_reason')}")
        text = content_text(message)
        if not text.strip():
            reason = choice.get("finish_reason") or "empty"
            raise ValueError(f"Model returned no text (finish_reason={reason})")
        self.last_model = model
        if model not in self.models_used:
            self.models_used.append(model)
        return parse_json(text)


def _jpeg_data_url(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > 1400:
        scale = 1400 / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif min(height, width) < 480:
        scale = 480 / min(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode a stamp page")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def upscale(image: np.ndarray, short_side: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    short = min(height, width)
    if short >= short_side:
        return image
    scale = short_side / short
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _pixmap_bgr(pix: pymupdf.Pixmap) -> np.ndarray | None:
    if pix.n - pix.alpha > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)


def _ink_fraction(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((gray < 210).mean())


def page_bgr(doc: pymupdf.Document, page: pymupdf.Page) -> np.ndarray:
    decoded = []
    for item in page.get_images(full=True):
        image = _pixmap_bgr(pymupdf.Pixmap(doc, item[0]))
        if image is not None:
            decoded.append(image)
    # A scan can store a blank white layer slightly larger than the photograph.
    # Keep the layer that actually shows the page, not the largest white rectangle.
    if decoded:
        largest = max(image.shape[0] * image.shape[1] for image in decoded)
        candidates = [image for image in decoded if image.shape[0] * image.shape[1] >= largest * 0.55]
        return max(candidates, key=_ink_fraction)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    array = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def image_file_bgr(path: Path) -> np.ndarray:
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit(f"Could not read image {path.name}.")
    return decoded


def tesseract_text(image: np.ndarray) -> str:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise SystemExit("Tesseract is not installed. Install it and run the extractor again.")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    result = subprocess.run(
        [tesseract, "stdin", "stdout", "--psm", "6", "-l", "eng"],
        input=encoded.tobytes(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.stdout.decode("utf-8", errors="replace")


def ocr_image(image: np.ndarray) -> str:
    gray = cv2.cvtColor(upscale(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # Malaysian biodata sits in the lower panel on an open page, or across a full data page.
    bands = (
        gray,
        gray[int(height * 0.40) : int(height * 0.90), int(width * 0.02) : int(width * 0.98)],
        gray[int(height * 0.08) : int(height * 0.92), int(width * 0.28) : int(width * 0.98)],
    )
    chunks = []
    for band in bands:
        if band.size == 0 or min(band.shape) < 40:
            continue
        enlarged = cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enlarged = cv2.GaussianBlur(enlarged, (3, 3), 0)
        _, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = " ".join(tesseract_text(binary).split())
        if text and text not in chunks:
            chunks.append(text)
    return "\n".join(chunks)


def read_pages(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    pages: list[dict] = []
    if suffix == ".pdf":
        doc = pymupdf.open(path)
        try:
            for index, page in enumerate(doc, start=1):
                image = page_bgr(doc, page)
                embedded = page.get_text("text")
                ocr = " ".join(ocr_image(image).split())
                text = ocr
                if embedded and embedded not in ocr:
                    text = f"{ocr}\nEmbedded text: {embedded}".strip()
                extras = []
                images = page.get_images(full=True)
                for item in images[1:]:
                    extra = pymupdf.Pixmap(doc, item[0])
                    if extra.n - extra.alpha > 3:
                        extra = pymupdf.Pixmap(pymupdf.csRGB, extra)
                    decoded = cv2.imdecode(np.frombuffer(extra.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None and decoded.shape[0] > 60 and decoded.shape[1] > 60:
                        extras.append(decoded)
                pages.append({
                    "number": index,
                    "text": text,
                    "embedded": embedded,
                    "image": image,
                    "extras": extras,
                })
                print(f"[Read] page {index}: {len(text)} characters", flush=True)
        finally:
            doc.close()
        return pages
    if suffix in IMAGE_SUFFIXES:
        image = image_file_bgr(path)
        text = " ".join(ocr_image(image).split())
        print(f"[Read] image: {len(text)} characters", flush=True)
        return [{"number": 1, "text": text, "embedded": "", "image": image, "extras": []}]
    raise SystemExit("Use a PDF or an image of a Malaysian passport.")


def name_in_text(name: str, text: str) -> bool:
    folded = re.sub(r"[^A-Z]", "", text.upper())
    tokens = [token for token in re.findall(r"[A-Za-z]{3,}", name) if token.lower() not in {"bin", "binti", "binte"}]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token.upper() in folded)
    return hits >= max(1, len(tokens) - 1)


def _biodata_pages(pages: list[dict]) -> list[dict]:
    """Pages that carry the holder's identity, not a visa or entry stamp."""
    found = []
    for page in pages:
        folded = re.sub(r"\s+", " ", str(page.get("text") or "")).upper()
        compact = folded.replace(" ", "")
        signals = 0
        if re.search(r"\bNAMA\b|\bNAME\b", folded):
            signals += 1
        if "PASPORT" in folded or "PASSPORT NO" in folded:
            signals += 1
        if "WARGANEGARA" in folded or "NATIONALITY" in folded:
            signals += 1
        if "P<MYS" in compact:
            signals += 1
        if re.search(r"\bJENIS\b|\bTYPE\b", folded):
            signals += 1
        for phrase in (
            "IDENTITY CARD",
            "KAD PENGENALAN",
            "DESCRIPTION OF HOLDER",
            "PLACE OF BIRTH",
            "COLOUR OF HAIR",
            "COLOR OF HAIR",
            "DISTINGUISHING MARKS",
            "PASSPORT EXPIRES",
            "SIGNATURE OF BEARER",
            "COUNTRY OF RESIDENCE",
        ):
            if phrase in folded:
                signals += 1
        if signals >= 2:
            found.append(page)
    return found


def layer1(client: OpenRouter, pages: list[dict]) -> dict:
    packed = "\n".join(f"--- page {page['number']} ---\n{page['text'] or '(no text)'}" for page in pages)
    data = client.complete(LAYER1_PROMPT + packed)
    name = clean_text(data.get("name"))
    page_no = data.get("page")
    try:
        page_no = int(page_no) if page_no is not None else None
    except (TypeError, ValueError):
        page_no = None
    known = {page["number"] for page in pages}
    if page_no not in known:
        page_no = None
    if name and page_no is None and len(pages) == 1:
        page_no = pages[0]["number"]
    page_text = ""
    if page_no is not None:
        page_text = next(item["text"] for item in pages if item["number"] == page_no)
    if name and page_text and not name_in_text(name, page_text):
        retry = client.complete(
            "The previous name was not printed in the OCR. Copy the holder name from the Nama / Name line only. "
            "Do not invent a name.\n"
            + LAYER1_PROMPT
            + f"--- page {page_no} ---\n{page_text}",
            max_tokens=None,
        )
        retried = clean_text(retry.get("name"))
        if retried and name_in_text(retried, page_text):
            name = retried
    if not name or page_no is None:
        return None
    if page_text and not name_in_text(name, page_text):
        return None
    return {"layer": 1, "status": "finished", "page": page_no, "name": name}


def _person_name(value) -> str | None:
    """A printed holder name, not a title, date, or place standing alone."""
    name = clean_text(value)
    if not name:
        return None
    titles = {"bin", "binti", "binte", "mr", "mrs", "ms", "encik", "puan", "cik"}
    tokens = [token for token in re.findall(r"[A-Za-z]{2,}", name) if token.lower() not in titles]
    if not tokens:
        return None
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return None
    return name


def layer1_from_image(client: OpenRouter, pages: list[dict]) -> dict | None:
    """Read the holder name from the page with Qwen OCR."""
    for page in pages:
        image = page.get("image")
        if image is None:
            continue
        data = client.complete(
            LAYER1_IMAGE_PROMPT,
            images=[image],
            model=QWEN_OCR_MODEL,
        )
        name = _person_name(data.get("name"))
        if not name:
            continue
        return {
            "layer": 1,
            "status": "finished",
            "page": page["number"],
            "name": name,
            "from_image": True,
        }
    return None


def same_name(value: str, name: str) -> bool:
    left = re.sub(r"[^a-z]", "", value.lower())
    right = re.sub(r"[^a-z]", "", name.lower())
    if len(left) < 4 or len(right) < 4:
        return False
    return left == right or left in right or right in left


TYPE_MEANINGS = {
    "P": "Ordinary passport",
    "PD": "Diplomatic passport",
    "D": "Diplomatic passport",
    "PO": "Official passport",
    "O": "Official passport",
    "S": "Service passport",
}


def written_sex(value: str) -> str:
    text = value.lower()
    if any(word in text for word in ("perempuan", "wanita", "female")) or re.search(r"\bf\b", text):
        return "WANITA- Female"
    if any(word in text for word in ("lelaki", "male")) or re.search(r"\b[ml]\b", text):
        return "LELAKI- Male"
    return value


def written_type(value: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", value).upper()
    if letters in TYPE_MEANINGS:
        return TYPE_MEANINGS[letters]
    return value


def layer2(client: OpenRouter, pages: list[dict], layer1_doc: dict, from_image: bool = False) -> dict:
    if layer1_doc.get("status") != "finished" or not layer1_doc.get("name"):
        raise SystemExit("Layer 2 starts only after Layer 1 has finished.")
    page_no = layer1_doc["page"]
    page = next(item for item in pages if item["number"] == page_no)
    if from_image and page.get("image") is not None:
        data = client.complete(LAYER2_IMAGE_PROMPT, images=[page["image"]], model=QWEN_OCR_MODEL)
    else:
        data = client.complete(
            LAYER2_PROMPT + f"--- page {page_no} ---\n{page['text']}",
        )
    raw = data.get("attributes") if isinstance(data.get("attributes"), dict) else data
    if (
        not from_image
        and not clean_text(raw.get("passport_number"))
        and re.search(r"\b[A-Z]\d{8}\b", page["text"])
    ):
        data = client.complete(
            "Include passport_number. It is the letter-and-digits value printed beside No. Pasport / Passport No. "
            "Do not include the holder name.\n" + LAYER2_PROMPT + f"--- page {page_no} ---\n{page['text']}",
        )
        raw = data.get("attributes") if isinstance(data.get("attributes"), dict) else data
    labels = {}
    if from_image and page.get("image") is not None:
        labels = client.complete(LAYER2_LABEL_PROMPT, images=[page["image"]], model=VISION_MODEL)
        if not isinstance(labels, dict):
            labels = {}
    attributes = {}
    for field in ATTRIBUTE_FIELDS:
        value = clean_text(raw.get(field))
        if from_image and field in {"date_of_birth", "date_of_issue", "date_of_expiry"} and value:
            value = re.sub(r"(\d{1,2})(?:ST|ND|RD|TH)\b", r"\1", value, flags=re.IGNORECASE)
            value = re.sub(r"\s+", " ", value.replace(".", " ")).strip(" .")
        if from_image and field == "nationality" and not labels.get("nationality_label"):
            continue
        if from_image and field == "type" and not labels.get("type_label"):
            continue
        if from_image and field == "country_code" and not labels.get("country_code_printed"):
            continue
        if field == "passport_number" and value:
            match = re.search(r"[A-Z]\d{8}", value)
            if match:
                value = match.group(0)
            elif from_image:
                continue
        if field == "sex" and value:
            value = written_sex(value)
        if field == "type" and value:
            value = written_type(value)
        if value and same_name(value, layer1_doc["name"]):
            continue
        if value:
            attributes[field] = value
    return {
        "layer": 2,
        "status": "finished",
        "page": page_no,
        "depends_on": "layer1_name.json",
        "attributes": attributes,
    }


MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
MONTH_NAMES = {number: name for name, number in MONTHS.items()}


def _format_date(day: int, month: int | str, year: int) -> str:
    """Write every date as DD MON YYYY, for example 23 DEC 1981."""
    month_name = MONTH_NAMES[month] if isinstance(month, int) else str(month)[:3].upper()
    return f"{int(day):02d} {month_name} {int(year):04d}"


def _stamp_dates(text: str) -> list[str]:
    cleaned = text.upper()
    cleaned = cleaned.replace("0CT", "OCT").replace("NOY", "NOV").replace("5EP", "SEP")
    cleaned = re.sub(r"\b(\d)\s+(\d)\s+([A-Z]{3})", r"\1\2 \3", cleaned)
    found = []
    for day, month, year in re.findall(r"\b(\d{1,2})\s*([A-Z]{3})\s*(\d{4})\b", cleaned):
        if month not in MONTHS:
            continue
        day_n, year_n = int(day), int(year)
        if not (1 <= day_n <= 31 and 1940 <= year_n <= 2035):
            continue
        stamp = _format_date(day_n, month, year_n)
        if stamp not in found:
            found.append(stamp)
    return found


def _parse_stamp_date(value: str):
    day, month, year = value.split()
    return (int(year), MONTHS[month], int(day))


# Longest Malaysian passport term. A stay further before expiry than this is a misread year.
PASSPORT_LIFE_YEARS = 10
_YEAR_CONFUSIONS = (("6", "8"), ("0", "8"), ("1", "7"), ("5", "6"), ("3", "8"), ("9", "0"), ("4", "9"))


def _passport_expiry(out_dir: Path) -> str | None:
    path = out_dir / "layer2_attributes.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    text = clean_text((document.get("attributes") or {}).get("date_of_expiry"))
    return text or None


def _passport_window(expiry: str | None):
    from datetime import date

    if not expiry:
        return None
    found = _stamp_dates(expiry)
    if not found:
        return None
    year, month, day = _parse_stamp_date(found[0])
    try:
        end = date(year, month, day)
        start = end.replace(year=end.year - PASSPORT_LIFE_YEARS)
    except ValueError:
        return None
    return start, end


def _confused_years(year: int) -> list[int]:
    text = f"{year:04d}"
    found = []
    for index, digit in enumerate(text):
        for left, right in _YEAR_CONFUSIONS:
            replacement = right if digit == left else left if digit == right else ""
            if not replacement:
                continue
            candidate = int(text[:index] + replacement + text[index + 1 :])
            if candidate not in found:
                found.append(candidate)
    return found


def _fit_stay_date(value: str | None, window) -> str | None:
    from datetime import date

    if not value or window is None:
        return value
    start, end = window
    try:
        year, month, day = _parse_stamp_date(value)
        current = date(year, month, day)
    except (KeyError, ValueError):
        return value
    if start <= current <= end:
        return value
    month_name = value.split()[1]
    fixes = []
    for candidate in _confused_years(year):
        try:
            fixed = date(candidate, month, day)
        except ValueError:
            continue
        if start <= fixed <= end and fixed not in fixes:
            fixes.append(fixed)
    if len(fixes) != 1:
        return value
    return _format_date(day, month_name, fixes[0].year)


def _fit_stays_to_passport(stays: list[dict], expiry: str | None) -> list[dict]:
    window = _passport_window(expiry)
    fitted = []
    seen = set()
    for stay in stays:
        visit = _fit_stay_date(stay.get("date_of_visiting"), window)
        leave = _fit_stay_date(stay.get("date_of_leaving"), window)
        if visit and leave:
            try:
                if _days_between(visit, leave) < 0:
                    leave = None
            except (KeyError, ValueError):
                leave = None
        updated = {
            **stay,
            "date_of_visiting": visit,
            "date_of_leaving": leave,
            "days": _model_days(None, visit, leave),
        }
        key = (updated.get("page"), updated.get("country"), visit, leave)
        if key in seen:
            continue
        seen.add(key)
        if visit or leave:
            fitted.append(updated)
    return fitted


def _life_note(expiry: str | None) -> str:
    if not expiry:
        return ""
    return (
        f"\nThis passport expires on {expiry}. "
        f"A Malaysian passport is valid for at most {PASSPORT_LIFE_YEARS} years before that date, not 25 years. "
        "If a seal year falls outside that life, one digit of the year was misread. "
        "Read the year so the date falls within the passport's life."
    )


def _days_between(visit: str, leave: str) -> int:
    from datetime import date

    vy, vm, vd = _parse_stamp_date(visit)
    ly, lm, ld = _parse_stamp_date(leave)
    return (date(ly, lm, ld) - date(vy, vm, vd)).days


LAYER4_PROMPT = """You are reading one passport page image. The page is covered with immigration seals. They differ in color and shape: purple, blue, green, and black rectangles, circles, ovals, and triangles.
Read every seal you can see. Do not return an empty list when seals are visible on the page.
For each stay copy only what is printed:
- country is the country named on that seal
- date_of_visiting is the admitted, arrived, or visit-pass date
- date_of_leaving is the date in the green marker placed beside that visit date, or the date on a seal that says departed, when it belongs to the same stay
- days is how many days passed from the visit date to the leave date
The date of visiting is not later than the date of leaving. If a date or the country is not printed, use null. Do not guess.
Return only JSON:
{"stays":[{"country":null,"date_of_visiting":null,"date_of_leaving":null,"days":null}]}
"""


def _normalize_stay_date(value) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        if month not in MONTH_NAMES or not (1 <= day <= 31 and 1940 <= year <= 2035):
            return None
        return _format_date(day, month, year)
    found = _stamp_dates(text)
    return found[0] if found else None


def _model_days(value, visit: str | None, leave: str | None) -> int | None:
    if not visit or not leave:
        return None
    try:
        gap = _days_between(visit, leave)
    except (KeyError, ValueError):
        return None
    if gap < 0:
        return None
    return gap


def _stays_from_model(data: dict) -> list[dict]:
    raw_stays = data.get("stays")
    if not isinstance(raw_stays, list):
        raw_stays = data.get("items") if isinstance(data.get("items"), list) else []
    stays = []
    seen = set()
    for item in raw_stays:
        if not isinstance(item, dict):
            continue
        country = clean_text(item.get("country"))
        visit = _normalize_stay_date(item.get("date_of_visiting") or item.get("visit"))
        leave = _normalize_stay_date(item.get("date_of_leaving") or item.get("leave"))
        if not visit and not leave:
            continue
        if visit and leave:
            try:
                if _days_between(visit, leave) < 0:
                    leave = None
            except (KeyError, ValueError):
                leave = None
        page_no = item.get("page")
        try:
            page_no = int(page_no) if page_no is not None else None
        except (TypeError, ValueError):
            page_no = None
        stay = {
            "page": page_no,
            "country": country,
            "date_of_visiting": visit,
            "date_of_leaving": leave,
            "days": _model_days(item.get("days"), visit, leave),
        }
        key = (country, visit, leave)
        if key in seen:
            continue
        seen.add(key)
        stays.append(stay)
    merged = []
    leave_only = []
    for stay in stays:
        if stay["date_of_leaving"] and not stay["date_of_visiting"]:
            leave_only.append(stay)
        else:
            merged.append(stay)
    for leave in leave_only:
        match = next(
            (
                stay
                for stay in merged
                if stay["country"] == leave["country"]
                and stay["date_of_visiting"] == leave["date_of_leaving"]
                and not stay["date_of_leaving"]
            ),
            None,
        )
        if match is None:
            merged.append(leave)
            continue
        match["date_of_leaving"] = leave["date_of_leaving"]
        match["days"] = _model_days(None, match["date_of_visiting"], match["date_of_leaving"])
        if match.get("page") is None:
            match["page"] = leave.get("page")
    def _sort_key(stay: dict):
        raw = stay["date_of_visiting"] or stay["date_of_leaving"]
        if not raw:
            return (9999, 12, 31)
        try:
            return _parse_stamp_date(raw)
        except (KeyError, ValueError):
            return (9999, 12, 31)

    merged.sort(key=_sort_key)
    return merged


def _readable(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    short = min(height, width)
    if short < 1400:
        scale = 1400 / short
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def _seal_place(x: int, y: int, width: int, height: int) -> str:
    column = "left" if x < width / 3 else "right" if x >= (2 * width) / 3 else "center"
    row = "top" if y < height / 3 else "bottom" if y >= (2 * height) / 3 else "middle"
    if row == "middle" and column == "center":
        return "center"
    return f"{row} {column}"


def _same_chop(ax: int, ay: int, bx: int, by: int) -> bool:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    if dx * dx + dy * dy < 140 * 140:
        return True
    # Two peaks stacked on one tall chop, such as a triangle.
    return dx < 55 and dy < 260


def _seal_peaks(page: np.ndarray) -> list[tuple[int, int]]:
    """Each seal is a peak in the blurred ink, so overlapping chops stay separate."""
    height, width = page.shape[:2]
    hue, saturation, value = cv2.split(cv2.cvtColor(page, cv2.COLOR_BGR2HSV))
    ink = saturation.astype(np.float32)
    sticky = (hue >= 15) & (hue <= 42) & (value > 140)
    ink[sticky] = 0
    dark_chop = (hue >= 90) & (hue <= 150) & (value < 95) & (saturation >= 12)
    ink[dark_chop] = np.maximum(ink[dark_chop], 48)
    ink[ink < 22] = 0
    blurred = cv2.GaussianBlur(ink, (0, 0), 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))
    peaks = (blurred >= cv2.dilate(blurred, kernel) - 0.01) & (blurred >= 24)
    count, _, _, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
    raw: list[tuple[int, int, float]] = []
    for index in range(1, count):
        center_x = int(round(centroids[index][0]))
        center_y = int(round(centroids[index][1]))
        if center_y < height * 0.04 or center_y > height * 0.96:
            continue
        if center_x < width * 0.04 or center_x > width - 8:
            continue
        y0, y1 = max(0, center_y - 40), min(height, center_y + 40)
        x0, x1 = max(0, center_x - 40), min(width, center_x + 40)
        nearby_sat = saturation[y0:y1, x0:x1]
        nearby_hue = hue[y0:y1, x0:x1]
        if nearby_sat.size == 0:
            continue
        spot_y, spot_x = np.unravel_index(int(nearby_sat.argmax()), nearby_sat.shape)
        spot_hue = int(nearby_hue[spot_y, spot_x])
        spot_sat = int(nearby_sat[spot_y, spot_x])
        # Pale red-brown print is the coat of arms. Blue, purple, and green chops are kept,
        # including a chop whose center sits on the photo edge.
        on_edge = center_x > width * 0.95
        if spot_hue <= 18 or spot_hue >= 168:
            if spot_sat < 110:
                continue
        elif 35 <= spot_hue <= 165:
            if spot_sat < 40:
                continue
        elif spot_sat < 90:
            continue
        if on_edge and not (90 <= spot_hue <= 165):
            continue
        stamp_ink = ((hue >= 35) & (hue <= 165) & (saturation >= 22)).astype(np.uint8)
        radius = max(40, int(min(height, width) * 0.055))
        window = stamp_ink[
            max(0, center_y - radius) : center_y + radius,
            max(0, center_x - radius) : center_x + radius,
        ]
        # A pen scribble has little ink. A chop fills its neighborhood.
        if window.size == 0 or float(window.mean()) < 0.12:
            continue
        raw.append((center_x, center_y, float(blurred[center_y, center_x])))
    raw.sort(key=lambda item: item[2], reverse=True)
    kept: list[tuple[int, int, float]] = []
    for center_x, center_y, strength in raw:
        if any(_same_chop(center_x, center_y, other_x, other_y) for other_x, other_y, _ in kept):
            continue
        kept.append((center_x, center_y, strength))
    kept.sort(key=lambda item: (item[1], item[0]))
    return [(center_x, center_y) for center_x, center_y, _ in kept]


def _owned_ink(page: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    height, width = page.shape[:2]
    peaks = _seal_peaks(page)
    if not peaks:
        return []
    saturation = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)[:, :, 1]
    ink_y, ink_x = np.where(saturation >= 40)
    if len(ink_x) == 0:
        return [((x, y), np.empty((0, 2), np.float32)) for x, y in peaks]
    points = np.stack((ink_x, ink_y), axis=1).astype(np.float32)
    centers = np.array(peaks, dtype=np.float32)
    distance = (points[:, None, 0] - centers[None, :, 0]) ** 2 + (points[:, None, 1] - centers[None, :, 1]) ** 2
    owner = distance.argmin(axis=1)
    nearest = distance.min(axis=1)
    owned = []
    for index, peak in enumerate(peaks):
        chosen = points[(owner == index) & (nearest <= 180 ** 2)]
        owned.append((peak, chosen))
    return owned


def _rotate_white(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the center. Empty corners stay white so the seal is not smeared."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    bound_w = int(height * sine + width * cosine)
    bound_h = int(height * cosine + width * sine)
    matrix[0, 2] += bound_w / 2.0 - center[0]
    matrix[1, 2] += bound_h / 2.0 - center[1]
    fill = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _axis_angle(image: np.ndarray, around: float | None = None) -> float:
    """Degrees to rotate so the straight text in the middle of the seal is horizontal.

    The date is a straight line of dark letters near the center. Curved names on a
    rim, and the stamp's own border, are left out of the measurement. OpenCV treats
    a positive angle as counter-clockwise, which is the same sign used here.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    height, width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    local = cv2.GaussianBlur(gray, (0, 0), max(8.0, 0.03 * min(height, width)))
    ink = (blur < local - 16).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    reach = np.hypot((xx - center_x) / (0.36 * width), (yy - center_y) / (0.36 * height))
    field = ink * np.exp(-0.5 * reach * reach).astype(np.float32)
    ys, xs = np.nonzero(field > 0.15)
    if len(xs) < 40:
        return 0.0
    weights = field[ys, xs].astype(np.float64)
    rel_x = xs.astype(np.float64) - center_x
    rel_y = ys.astype(np.float64) - center_y
    angles = np.arange(-90, 91, dtype=np.float64)
    scores = np.empty(len(angles), dtype=np.float64)
    for index, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        cosine, sine = np.cos(theta), np.sin(theta)
        # Same mapping as getRotationMatrix2D: y' = -x sin + y cos.
        rotated_y = -sine * rel_x + cosine * rel_y
        bins = np.round(rotated_y / 3.0).astype(np.int32)
        counts = np.bincount(bins - int(bins.min()), weights=weights)
        scores[index] = float(np.square(counts).sum())
    smooth = np.convolve(scores, np.array([1.0, 2.0, 3.0, 2.0, 1.0]), mode="same")
    if around is not None:
        window = np.where(np.abs(angles - float(around)) <= 18)[0]
        if len(window) == 0:
            return float(around)
        peak = int(window[int(np.argmax(smooth[window]))])
        return _blended_angle(angles, smooth, peak)
    peak = int(np.argmax(smooth))
    level = float(smooth[90])
    if level <= 0:
        return 0.0
    ratio = float(smooth[peak] / level)
    rival = 0.0
    for index, value in enumerate(smooth):
        # ±90 is the same vertical axis, so the gap wraps around.
        gap = abs(float(angles[index] - angles[peak]))
        gap = min(gap, 180.0 - gap)
        if gap > 20:
            rival = max(rival, float(value))
    # A weak peak is already-level text. Two close peaks are two directions of writing.
    if ratio < 1.20 or rival > float(smooth[peak]) * 0.90:
        return 0.0
    correction = _blended_angle(angles, smooth, peak)
    if abs(correction) < 3.5:
        return 0.0
    return correction


def _blended_angle(angles: np.ndarray, smooth: np.ndarray, peak: int) -> float:
    weight = 0.0
    signed = 0.0
    for offset in range(-3, 4):
        index = peak + offset
        if index < 0 or index >= len(angles):
            continue
        weight += float(smooth[index])
        signed += float(smooth[index]) * float(angles[index])
    if weight <= 0:
        return float(angles[peak])
    return signed / weight


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Colored stamp ink and dark letters. Pale paper is left out."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    paper = float(np.median(gray))
    dark = gray < paper - 16
    if image.ndim != 3:
        return dark
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    return dark | (saturation >= 28)


def _trim_margin(image: np.ndarray, pad: int = 10) -> np.ndarray:
    ys, xs = np.where(_ink_mask(image))
    if len(xs) < 30:
        return image
    height, width = image.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(height, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(width, int(xs.max()) + pad + 1)
    return image[y0:y1, x0:x1]


def _hue_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Separate seals of different colors. A second color inside one stamp stays with that stamp."""
    if image.ndim != 3:
        return []
    hue, saturation, _value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    ink = saturation >= 32
    if int(ink.sum()) < 80:
        return []
    labels = (((hue.astype(np.int16) + 8) % 180) // 15) + 1
    labels = np.where(ink, labels, 0)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, 13):
        ys, xs = np.where(labels == label)
        if len(xs) < 120:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
    if len(boxes) < 2:
        return []
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=_pixel_area, reverse=True):
        if any(_pixel_coverage(box, other) > 0.7 for other in kept):
            continue
        kept.append(box)
    return kept if len(kept) >= 2 else []


def _corner_fragment(image: np.ndarray) -> bool:
    """True when the ink is a corner or a blank margin, not a seal filling the cut."""
    mask = _ink_mask(image)
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return True
    height, width = image.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if float(mask.mean()) < 0.012:
        return True
    span = ((y1 - y0) * (x1 - x0)) / float(height * width)
    center_inside = y0 <= height / 2 <= y1 and x0 <= width / 2 <= x1
    return (not center_inside) and span < 0.55


def _align_seal(image: np.ndarray) -> np.ndarray:
    """Turn the seal so its date and other straight text sit on a horizontal line."""
    angle = _axis_angle(image)
    if abs(angle) < 1.0:
        return image
    print(f"[Layer 4] seal turned {angle:.0f} degrees", flush=True)
    return _trim_margin(_rotate_white(image, angle))


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the center and grow the canvas so the corners stay in frame."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    bound_w = int(height * sine + width * cosine)
    bound_h = int(height * cosine + width * sine)
    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _skew_angle(image: np.ndarray) -> float:
    """Angle, in degrees, that brings the seal's long edges to the horizontal."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=25, maxLineGap=6)
    if lines is None:
        return 0.0
    weighted: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        length = float(np.hypot(x2 - x1, y2 - y1))
        weighted.append((angle, length))
    total = sum(length for _, length in weighted)
    if total <= 0:
        return 0.0
    weighted.sort(key=lambda item: item[0])
    half = total / 2
    running = 0.0
    for angle, length in weighted:
        running += length
        if running >= half:
            return -angle
    return 0.0


def _upright_seal(page: np.ndarray, points: np.ndarray, center: tuple[int, int]) -> np.ndarray | None:
    """Cut out one seal and rotate it when its edges are tilted."""
    height, width = page.shape[:2]
    center_x, center_y = center
    if len(points) >= 30:
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        x0, y0 = max(0, int(x0) - 16), max(0, int(y0) - 16)
        x1, y1 = min(width, int(x1) + 17), min(height, int(y1) + 17)
    else:
        x0, y0 = max(0, center_x - 100), max(0, center_y - 80)
        x1, y1 = min(width, center_x + 100), min(height, center_y + 80)
    crop = page[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    angle = _skew_angle(crop)
    if abs(angle) < 8:
        return crop
    return _rotate_bound(crop, angle)


def seal_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """One box per seal. OpenCV only, so an 8 GB Mac does not load another model."""
    page = _readable(image)
    # A dark cover is ink almost everywhere. It is not a page of seals.
    if _ink_fraction(page) >= 0.97:
        return []
    height, width = page.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for (center_x, center_y), chosen in _owned_ink(page):
        if len(chosen) < 30:
            boxes.append((max(0, center_x - 80), max(0, center_y - 60), 160, 120))
            continue
        x0, y0 = chosen.min(axis=0)
        x1, y1 = chosen.max(axis=0)
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(width, int(x1) + 1)
        y1 = min(height, int(y1) + 1)
        boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return boxes


def _stay_from_seal_text(text: str) -> dict:
    residue = re.sub(r"\b\d{1,2}\s*[A-Za-z]{3}\s*\d{4}\b", " ", text)
    lines = []
    for line in residue.splitlines():
        cleaned = re.sub(r"[^A-Za-z .'-]", " ", line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
        if len(re.sub(r"[^A-Za-z]", "", cleaned)) >= 3:
            lines.append(cleaned)
    return {
        "page": None,
        "country": max(lines, key=len) if lines else None,
    }


def seal_crops(image: np.ndarray) -> list[dict]:
    """Cut each seal out and rotate it upright from its tilted outline."""
    page = _readable(image)
    if _ink_fraction(page) >= 0.97:
        return []
    height, width = page.shape[:2]
    crops = []
    for (center_x, center_y), points in _owned_ink(page):
        upright = _upright_seal(page, points, (center_x, center_y))
        if upright is None or upright.size == 0:
            continue
        crops.append({
            "image": upright,
            "place": _seal_place(center_x, center_y, width, height),
        })
    return crops


LAYER4_CROP_PROMPT = """These images are separate seals cut from one passport page. Each image is one seal, still in its real colors.
Read every image. Copy only what is printed on that seal:
- country is the country named on that seal
- date_of_visiting is the admitted, arrived, or visit-pass date
- date_of_leaving is the date in the green marker on that seal, or the date on a seal that says departed, when it belongs to the same stay
- days is how many days passed from the visit date to the leave date
The date of visiting is not later than the date of leaving. If a date or the country is not printed, use null. Do not guess.
Return only JSON:
{"stays":[{"country":null,"date_of_visiting":null,"date_of_leaving":null,"days":null}]}
"""


def _same_country(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    return left.casefold() == right.casefold()


def _absorb_crop_stays(page_stays: list[dict], crop_stays: list[dict]) -> list[dict]:
    for extra in crop_stays:
        visit = extra.get("date_of_visiting")
        leave = extra.get("date_of_leaving")
        country = extra.get("country")
        filled = False
        if visit:
            for stay in page_stays:
                if stay.get("date_of_visiting") != visit or not _same_country(stay.get("country"), country):
                    continue
                if country and not stay.get("country"):
                    stay["country"] = country
                if leave and not stay.get("date_of_leaving"):
                    try:
                        if _days_between(visit, leave) >= 0:
                            stay["date_of_leaving"] = leave
                            stay["days"] = _model_days(None, visit, leave)
                    except (KeyError, ValueError):
                        pass
                filled = True
                break
        if filled:
            continue
        key = (country, visit, leave)
        if any((stay.get("country"), stay.get("date_of_visiting"), stay.get("date_of_leaving")) == key for stay in page_stays):
            continue
        if visit and not leave and any(
            stay.get("date_of_visiting") == visit and _same_country(stay.get("country"), country) for stay in page_stays
        ):
            continue
        if leave and not visit and any(
            (stay.get("date_of_leaving") == leave or stay.get("date_of_visiting") == leave)
            and _same_country(stay.get("country"), country)
            for stay in page_stays
        ):
            continue
        page_stays.append(extra)
    return page_stays


def _stay_order(stay: dict) -> int:
    try:
        return int(stay.get("page") or 0)
    except (TypeError, ValueError):
        return 0


def _layer4_document(
    stays: list[dict],
    status: str,
    seal_count: int | None = None,
    seal_pages: list[dict] | None = None,
) -> dict:
    document = {
        "layer": 4,
        "status": status,
        "model": MODEL,
        "stays": sorted(stays, key=_stay_order),
    }
    if seal_count is not None:
        document["seal_count"] = seal_count
    if seal_pages is not None:
        document["seal_pages"] = seal_pages
    return document


def _save_layer4(
    path: Path,
    stays: list[dict],
    status: str,
    seal_count: int | None = None,
    seal_pages: list[dict] | None = None,
) -> dict:
    document = _layer4_document(stays, status, seal_count, seal_pages)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


GROUND_PROMPT = """Locate every immigration or visa seal in this passport page: each rubber stamp or chop, whatever its shape, color, or tilt.
Each stamp is a separate object, even when it overlaps another stamp. Include faded stamps and stamps with handwriting inside them.
A visa can also be a large stamped or printed form on the page, with ruled lines, a number, typed or handwritten entries, and a signature. That whole form counts as one seal.
Do not include a glued paper sticker, a printed page number, blank paper, or the printed word VISAS by itself.
The box must hold that one stamp's full outer border, plus a little paper outside the border. A tilted stamp still needs its whole border inside the box. A stamp whose writing runs on a diagonal still needs every letter of that writing, plus paper past the first and last letter. A box that cuts the border, cuts a letter, or keeps only a corner or a strip, is wrong.
Return JSON: {"seals": [{"bbox_2d": [x1, y1, x2, y2]}]} with coordinates normalized to 0-1000 of this image.
If there are none, return {"seals": []}.
"""

MISSING_PROMPT = """Red numbered boxes mark the immigration or visa seals already found on this passport page.
Locate every seal that does not have a red box around its whole border: a rubber stamp or chop of any shape, color, or tilt, including faded stamps and stamps that overlap a boxed one.
A red box that covers only a corner, a few letters, or one side of a stamp does not count. That stamp is still missing: return a box around its whole border.
Do not return a seal whose whole border is already inside a red box. Do not include a sticker, a printed page number, blank paper, or the printed word VISAS by itself.
Return JSON: {"seals": [{"bbox_2d": [x1, y1, x2, y2]}]} with coordinates normalized to 0-1000 of this image.
If none are missing, return {"seals": []}.
"""

TURN_PROMPT = """These 8 numbered pictures are the same passport stamp turned to different angles.
The stamp to level is the one in the center. Other printing may show through behind it.
Pick the picture where that center stamp's own date, or its own main line of text, is upright and reads normally from left to right: not upside down, not sideways, not mirrored.
If that stamp is printed on a diagonal, pick the turn that levels its own words. Leave it diagonal only when every turn still shows it diagonal.
Return JSON: {"pick": 1}
"""

ENDS_PROMPT = """This is one passport stamp, roughly upright. Other printing may show through behind it.
Find this stamp's own date, or its own main straight line of text if there is no date. Do not use a line that belongs to a different stamp behind it.
Box the first character and the last character of that line.
Return JSON: {"first": [x1, y1, x2, y2], "last": [x1, y1, x2, y2], "text": "the line"} with coordinates normalized to 0-1000 of this image.
"""

FRAME_PROMPT = """One passport stamp sits at the center of this picture, already turned upright. Other stamps may overlap it or sit beside it.
Box the stamp at the center. The box must hold its full outer border on every side, plus a thin margin of paper outside that border. Every letter of that stamp must sit inside the box, including a word printed on a diagonal, with paper past the first and last letter.
Do not include a different stamp. Do not stop at a date, one word, or a corner. A box that cuts through the border or through a letter is wrong.
complete: false when any side of that outer border is missing, when a letter is cut off, or when the stamp runs off this picture. true only when the full border, every letter, and a little paper outside them, are inside the picture.
Return JSON: {"bbox_2d": [x1, y1, x2, y2], "complete": true} with coordinates normalized to 0-1000 of this image.
"""

PLACES_PROMPT = """Find every place name written on this passport page: each country, city, airport, port, or border post, whether stamped, printed, typed, or handwritten.
Box each name. Do not box a date, a number, or the words VISA, IMMIGRATION, ARRIVAL, or DEPARTED by themselves.
Return JSON: {"places": [{"name": "", "bbox_2d": [x1, y1, x2, y2]}]} with coordinates normalized to 0-1000 of this image.
If there are none, return {"places": []}.
"""

READ_PROMPT = """Read the passport stamp at the center of this picture.
seal: true when the picture shows an immigration or visa stamp. false when it shows only a sliver of a stamp, blank paper, a sticker, or a printed page number.
text: copy each line of the stamp's writing exactly as the letters appear, one line per line, including the date and any handwriting inside the stamp. Words curved around a round or oval rim count as a line. When stamps overlap, read the stamp whose border frames the center, not the one crossing it.
Only copy letters you can see. When a letter is unclear, keep the letters you can see and leave out the rest. Do not guess a word, do not translate, and do not add a country or a city that is not written there. Use an empty string when there is no writing.
Read each digit from its ink, not from a year that would be familiar. An 8 is two closed loops, one above the other. A 9 is one closed loop with a stem down the right side. When both loops are visible, write 8. When one loop and a right-hand stem are visible, write 9. A 6 has its stem on the left. A 0 is one empty oval. A 1 is a single straight stroke. A 5 has a flat top and a round lower bowl. Do not turn a 1 into a 5, or an 8 into a 9 or a 5.
A star, dash, or dot beside a day is that mark, not a digit. Do not turn it into 0, 1, 2, or 5, and do not place it in front of the day. A day printed as one digit stays one digit. Keep a second digit of the day only when that second mark is itself a digit.
A code printed in parentheses stays in parentheses, and only the marks inside the parentheses belong to the code. A closing parenthesis is not the letter J. Do not rewrite the code as a clock time.
Return only JSON: {"seal": true, "text": ""}
"""

JUDGE_PROMPT = """This picture is one passport stamp. Two readers copied its writing.
Reading A:
{a}

Reading B:
{b}

Write one reading. Use the picture, not the more familiar year. Keep all of A, all of B, or mix lines and digits from both when each reader got a different part right. Every word and every digit you write must be visible on the stamp and present in A or B. Do not invent a third date by joining the wrong digits from each reading.
Check the date one digit at a time. A star beside the day is not a digit: a one-digit day stays one digit, so do not turn * 4 into 04, 05, 14, or 24. An 8 is two closed loops. A 1 is a single stroke. A 5 has a flat top and a round lower bowl. Do not turn 1981 into 1985.
A code in parentheses keeps the parentheses. A closing parenthesis is not the letter J.
Return only JSON: {"text": ""}
"""

DATE_PROMPT = """Look only at the date printed on this stamp. The two readings disagree.
Reading A:
{a}

Reading B:
{b}

Copy the day, the month, and the year that are printed. Ignore every other line.
A star, dash, or dot beside the day is not a digit. Do not turn that mark into 0, 1, 2, or 5, and do not add it in front of the day. If one reading shows a one-digit day and the other added a digit, keep the one-digit day when the extra mark is a star.
An 8 is two closed loops. A 9 is one loop with a stem on the right. A 1 is a single stroke. A 5 has a flat top and a round lower bowl. The last digit of 1981 is a stroke, not a 5.
Return only JSON: {"date": ""}
"""

SETTLE_PROMPT = """This picture is one passport stamp. Two readers copied it, and they do not agree.
Reading A:
{first}
Reading B:
{second}

Copy the stamp again from the picture, one printed line per line.
Where the readings disagree, copy only the mark you can see. A star, dash, dot, or parenthesis is not a digit: do not turn it into a number, and do not add a digit in front of a number. Keep parentheses around a code when they are printed.
Do not translate, and do not add a word that is not on the stamp.
Return only JSON: {"text": ""}
"""

COUNTRY_PROMPT = """This picture is one passport stamp. The OCR text read from it is:
{text}

country: the country or territory that issued this stamp, written as its name. Use the picture and the text together. A place you can see on the stamp decides the country. Use the text when the picture is too faint to read a place. If the stamp names only a city, airport, or port, name the country that place is in. Do not replace a place that is visible on the stamp with a different country. Use null when neither the picture nor the text points to a country. A visa word, a status word, or a date is not a country. Do not answer with a code.
Return only JSON: {"country": null}
"""

STAY_PROMPT = """This is a Malaysian passport. Classify every dated stamp event below.

{events}

For every event id, role is arrival, departure, visa duration, or unknown.
Use the meaning of the complete stamp text, in any language. Do not rely on a fixed list of entry or exit words.
An arrival begins a stay in that event's country. A departure ends a stay in that event's country.
A visa duration is a date that limits how long the holder may stay, such as a visa validity or a permitted-until date, rather than the day they arrived or left.
Use unknown only when the text does not say which of those it is.
Do not change a country or date and do not omit an event.
Return only JSON: {"events": [{"id": 1, "role": "unknown"}]}
"""


def _yes(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"true", "yes", "seal", "1"}


def _fraction_box(value, min_side: float = 0.12) -> tuple[float, float, float, float] | None:
    nums = None
    if isinstance(value, dict):
        if "box" in value:
            return _fraction_box(value["box"], min_side)
        for group in (
            ("x", "y", "w", "h"),
            ("x", "y", "width", "height"),
            ("left", "top", "width", "height"),
        ):
            if all(key in value for key in group):
                nums = [value[key] for key in group]
                break
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        nums = list(value[:4])
    if nums is None:
        return None
    try:
        x, y, width, height = (float(number) for number in nums)
    except (TypeError, ValueError):
        return None
    if max(abs(x), abs(y), abs(width), abs(height)) > 1.5:
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0 - x)
    height = min(max(height, 0.0), 1.0 - y)
    if width < min_side or height < min_side:
        return None
    return (x, y, width, height)


def _box_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - inter
    return inter / union if union else 0.0


def _seal_boxes(data: dict | None, min_side: float = 0.12) -> list[tuple[float, float, float, float]]:
    if not data:
        return []
    raw = data.get("boxes")
    if raw is None:
        raw = data.get("seals")
    if isinstance(raw, dict):
        raw = raw.get("boxes") or raw.get("items")
    if not isinstance(raw, list):
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for item in raw:
        box = _fraction_box(item, min_side)
        if box is None:
            continue
        if any(_box_overlap(box, kept) > 0.6 for kept in boxes):
            continue
        boxes.append(box)
    return boxes


def _seal_count(data: dict) -> int:
    raw = data.get("count")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        boxes = _seal_boxes(data)
        if boxes:
            return len(boxes)
        return 1 if _yes(data.get("seal")) else 0


def _cut_box(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x, y, box_w, box_h = box
    pad = 0.03
    x0 = max(0, int((x - pad) * width))
    y0 = max(0, int((y - pad) * height))
    x1 = min(width, int((x + box_w + pad) * width))
    y1 = min(height, int((y + box_h + pad) * height))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return image[y0:y1, x0:x1]


def _pixel_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2]) * max(0, box[3])


def _pixel_intersection(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[0] + left[2], right[0] + right[2])
    y2 = min(left[1] + left[3], right[1] + right[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _pixel_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    inter = _pixel_intersection(left, right)
    union = _pixel_area(left) + _pixel_area(right) - inter
    return inter / union if union else 0.0


def _pixel_coverage(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    area = _pixel_area(inner)
    return _pixel_intersection(inner, outer) / area if area else 0.0


def _merge_page_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Drop duplicate views of one seal, and drop a box that swallows several seals."""
    cleaned = []
    for box in boxes:
        width, height = box[2], box[3]
        if width < 48 or height < 48:
            continue
        if max(width / height, height / width) > 4.2:
            continue
        cleaned.append(box)
    cleaned.sort(key=_pixel_area)
    kept: list[tuple[int, int, int, int]] = []
    for box in cleaned:
        contained = [other for other in kept if _pixel_coverage(other, box) > 0.62]
        if len(contained) >= 2:
            continue
        if len(contained) == 1:
            inner = contained[0]
            if _pixel_area(box) > _pixel_area(inner) * 1.45:
                kept.remove(inner)
                kept.append(box)
            continue
        if any(_pixel_iou(box, other) > 0.42 or _pixel_coverage(box, other) > 0.62 for other in kept):
            continue
        kept.append(box)
    # A box that sits inside a larger one is the same seal seen twice.
    unique: list[tuple[int, int, int, int]] = []
    for box in sorted(kept, key=_pixel_area, reverse=True):
        if any(_pixel_coverage(box, other) > 0.55 for other in unique):
            continue
        unique.append(box)
    unique.sort(key=lambda item: (item[1], item[0]))
    return unique


def _cluster_boxes(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Join boxes that land on the same stamp. Separate stamps stay apart."""
    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index, left in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            right = boxes[right_index]
            smaller = min(_pixel_area(left), _pixel_area(right))
            if not smaller or _pixel_intersection(left, right) / smaller < 0.45:
                continue
            left_x = left[0] + left[2] / 2
            left_y = left[1] + left[3] / 2
            right_x = right[0] + right[2] / 2
            right_y = right[1] + right[3] / 2
            distance = ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5
            reach = 0.6 * (max(left[2], left[3]) + max(right[2], right[3])) / 2
            if distance <= reach:
                parent[find(left_index)] = find(right_index)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, box in enumerate(boxes):
        groups.setdefault(find(index), []).append(box)
    return list(groups.values())


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _crop_pixels(page: np.ndarray, box: tuple[int, int, int, int], pad: int = 12) -> np.ndarray | None:
    height, width = page.shape[:2]
    x0 = max(0, box[0] - pad)
    y0 = max(0, box[1] - pad)
    x1 = min(width, box[0] + box[2] + pad)
    y1 = min(height, box[1] + box[3] + pad)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return page[y0:y1, x0:x1]


def _has_stamp_ink(crop: np.ndarray) -> bool:
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    sticker = (hue >= 15) & (hue <= 42) & (value > 140) & (saturation > 40)
    if float(sticker.mean()) > 0.22:
        return False
    # Colored ink and dark ink both count. A blank margin does not.
    ink = ((saturation > 18) | ((value < 125) & (value > 28)))
    if float(ink.mean()) < 0.025:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) >= 24


def _point_in_box(x: int, y: int, box: tuple[int, int, int, int], pad: int = 10) -> bool:
    return box[0] - pad <= x <= box[0] + box[2] + pad and box[1] - pad <= y <= box[1] + box[3] + pad


def _missed_peak_boxes(page: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """A colored peak the page scan missed still gets its own window."""
    height, width = page.shape[:2]
    extra = []
    window_w = max(120, int(width * 0.16))
    window_h = max(100, int(height * 0.18))
    for center_x, center_y in _seal_peaks(page):
        if any(_point_in_box(center_x, center_y, box) for box in boxes):
            continue
        x0 = max(0, center_x - window_w // 2)
        y0 = max(0, center_y - window_h // 2)
        box = (x0, y0, min(width, x0 + window_w) - x0, min(height, y0 + window_h) - y0)
        if any(_pixel_iou(box, other) > 0.3 for other in boxes):
            continue
        extra.append(box)
    return extra


def _center_inside(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    return other[0] <= center_x <= other[0] + other[2] and other[1] <= center_y <= other[1] + other[3]


def _grounded_box(value, width: int, height: int, min_side: int = 8) -> tuple[int, int, int, int] | None:
    """A model box, x1 y1 x2 y2 on the 0-1000 scale, in pixels of a width × height picture."""
    if isinstance(value, dict):
        value = value.get("bbox_2d") or value.get("box") or value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(number) for number in value[:4])
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) <= 1.0:
        x1, y1, x2, y2 = x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    left = int(round(max(0.0, x1) / 1000 * width))
    top = int(round(max(0.0, y1) / 1000 * height))
    right = int(round(min(1000.0, x2) / 1000 * width))
    bottom = int(round(min(1000.0, y2) / 1000 * height))
    if right - left < min_side or bottom - top < min_side:
        return None
    return (left, top, right - left, bottom - top)


def _seal_sized(page: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """A seal is more than a speck and less than a quarter of the page."""
    page_h, page_w = page.shape[:2]
    if min(box[2], box[3]) < 0.035 * min(page_h, page_w):
        return False
    if _pixel_area(box) > 0.25 * page_h * page_w:
        return False
    crop = _crop_pixels(page, box, pad=0)
    return crop is not None and _has_stamp_ink(crop)


def _answer_boxes(data: dict | None, page: np.ndarray) -> list[tuple[int, int, int, int]]:
    raw = []
    if isinstance(data, dict):
        raw = data.get("seals") or data.get("items") or data.get("boxes") or []
    if not isinstance(raw, list):
        return []
    page_h, page_w = page.shape[:2]
    boxes = []
    for item in raw:
        box = _grounded_box(item, page_w, page_h)
        if box is not None and _seal_sized(page, box):
            boxes.append(box)
    return boxes


def _distinct_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=_pixel_area, reverse=True):
        if any(_pixel_iou(box, other) > 0.55 or _pixel_coverage(box, other) > 0.85 for other in kept):
            continue
        kept.append(box)
    kept.sort(key=lambda box: (box[1], box[0]))
    return kept


def _found_seals(client: OpenRouter, page: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Qwen boxes every seal on the whole page, then looks again for any it left out."""
    boxes = _distinct_boxes(_answer_boxes(client.ask(GROUND_PROMPT, [page]), page))
    print(f"[Layer 4] {len(boxes)} seals located", flush=True)
    marked = page.copy()
    thickness = max(2, min(page.shape[:2]) // 400)
    for number, (x, y, width, height) in enumerate(boxes, start=1):
        cv2.rectangle(marked, (x, y), (x + width, y + height), (0, 0, 255), thickness)
        cv2.putText(marked, str(number), (x + 6, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), thickness)
    extra = []
    for box in _answer_boxes(client.ask(MISSING_PROMPT, [marked]), page):
        # A box already found may be only a corner of this one. Keep the new box
        # unless most of it is already inside a box we have.
        if any(_pixel_coverage(box, other) > 0.5 for other in boxes + extra):
            continue
        extra.append(box)
    if extra:
        print(f"[Layer 4] {len(extra)} more seals found on the second look", flush=True)
    boxes = _distinct_boxes(boxes + extra)
    beside = _place_seals(client, page, boxes)
    if beside:
        print(f"[Layer 4] {len(beside)} more seals found beside place names", flush=True)
    return _distinct_boxes(boxes + beside)


def _seal_holding(client: OpenRouter, page: np.ndarray, place: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Look closer around one place name. The largest seal that holds the name is the stamp it belongs to."""
    page_h, page_w = page.shape[:2]
    center_x = place[0] + place[2] / 2
    center_y = place[1] + place[3] / 2
    x0 = int(max(0, center_x - 0.25 * page_w))
    x1 = int(min(page_w, center_x + 0.25 * page_w))
    y0 = int(max(0, center_y - 0.35 * page_h))
    y1 = int(min(page_h, center_y + 0.35 * page_h))
    window = page[y0:y1, x0:x1]
    data = client.ask(GROUND_PROMPT, [window], model=VISION_MODEL) or {}
    raw = data.get("seals") if isinstance(data, dict) else None
    best = None
    for item in raw if isinstance(raw, list) else []:
        box = _grounded_box(item, x1 - x0, y1 - y0)
        if box is None:
            continue
        box = (box[0] + x0, box[1] + y0, box[2], box[3])
        if not _center_inside(place, box) or not _seal_sized(page, box):
            continue
        if best is None or _pixel_area(box) > _pixel_area(best):
            best = box
    return best


def _place_seals(client: OpenRouter, page: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """A place name outside every seal box often belongs to a stamp that was missed. Find that stamp."""
    page_h, page_w = page.shape[:2]
    data = client.ask(PLACES_PROMPT, [page], model=VISION_MODEL) or {}
    raw = data.get("places") if isinstance(data, dict) else None
    places = []
    for item in raw if isinstance(raw, list) else []:
        place = _grounded_box(item, page_w, page_h, min_side=2)
        if place is not None and not any(_center_inside(place, box) for box in boxes):
            places.append(place)
    if not places:
        return []
    with ThreadPoolExecutor(max_workers=SEAL_WORKERS) as pool:
        holding = list(pool.map(lambda place: _seal_holding(client, page, place), places[:8]))
    found: list[tuple[int, int, int, int]] = []
    for box in sorted((box for box in holding if box is not None), key=_pixel_area, reverse=True):
        if any(_pixel_coverage(box, other) > 0.5 for other in boxes + found):
            continue
        found.append(box)
    return found


def _window(page: np.ndarray, box: tuple[int, int, int, int], grow: float) -> np.ndarray:
    """The box and a margin around it. Past the page edge is white paper."""
    x, y, width, height = box
    grow_x = int(round(width * grow))
    grow_y = int(round(height * grow))
    pad = max(grow_x, grow_y) + 1
    padded = cv2.copyMakeBorder(page, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    top = y + pad - grow_y
    left = x + pad - grow_x
    return padded[top:top + height + 2 * grow_y, left:left + width + 2 * grow_x].copy()


SEAL_TURNS = (0, 45, 90, 135, 180, 225, 270, 315)


def _turn_sheet(image: np.ndarray, cell: int = 320) -> np.ndarray:
    tiles = []
    for number, angle in enumerate(SEAL_TURNS, start=1):
        turned = _rotate_white(image, angle)
        scale = cell / max(turned.shape[:2])
        turned = cv2.resize(turned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile = np.full((cell + 40, cell, 3), 255, np.uint8)
        top = 40 + (cell - turned.shape[0]) // 2
        left = (cell - turned.shape[1]) // 2
        tile[top:top + turned.shape[0], left:left + turned.shape[1]] = turned
        cv2.putText(tile, str(number), (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 220), 3)
        cv2.rectangle(tile, (0, 0), (cell - 1, cell + 39), (180, 180, 180), 2)
        tiles.append(tile)
    half = len(tiles) // 2
    return np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])])


def _coarse_turn(client: OpenRouter, image: np.ndarray) -> int:
    """Qwen picks which of eight turns shows the date upright. Picking is surer than guessing degrees."""
    data = client.ask(TURN_PROMPT, [_turn_sheet(image)], model=VISION_MODEL)
    try:
        pick = int((data or {}).get("pick"))
    except (TypeError, ValueError):
        return 0
    return SEAL_TURNS[pick - 1] if 1 <= pick <= len(SEAL_TURNS) else 0


def _date_tilt(client: OpenRouter, image: np.ndarray) -> float:
    """Degrees that level the date, from the boxes Qwen draws on its first and last character."""
    height, width = image.shape[:2]
    data = client.ask(ENDS_PROMPT, [image], model=VISION_MODEL) or {}
    first = _grounded_box(data.get("first"), width, height, min_side=2)
    last = _grounded_box(data.get("last"), width, height, min_side=2)
    if first is None or last is None:
        return 0.0
    first_x, first_y = first[0] + first[2] / 2, first[1] + first[3] / 2
    last_x, last_y = last[0] + last[2] / 2, last[1] + last[3] / 2
    if last_x - first_x < 0.15 * width:
        return 0.0
    # OpenCV turns counter-clockwise for a positive angle, and y grows downward.
    tilt = float(np.degrees(np.arctan2(last_y - first_y, last_x - first_x)))
    # A first pick one step off still leaves less than one more step to level.
    return tilt if abs(tilt) <= 50 else 0.0


def _grow_box(page: np.ndarray, box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """The same box grown outward, still inside the page."""
    page_h, page_w = page.shape[:2]
    x, y, width, height = box
    grow_x = int(round(width * scale))
    grow_y = int(round(height * scale))
    x0 = max(0, x - grow_x)
    y0 = max(0, y - grow_y)
    x1 = min(page_w, x + width + grow_x)
    y1 = min(page_h, y + height + grow_y)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _box_ran_off(value) -> bool:
    """True when the model's box reaches the edge of the picture, so the stamp continues past it."""
    if isinstance(value, dict):
        value = value.get("bbox_2d") or value.get("box") or value.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        x1, y1, x2, y2 = (float(number) for number in value[:4])
    except (TypeError, ValueError):
        return False
    if max(x1, y1, x2, y2) <= 1.0:
        x1, y1, x2, y2 = x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return min(x1, y1) < 40 or x2 > 960 or y2 > 960


def _framed_seal(client: OpenRouter, upright: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, bool]:
    """Qwen boxes the centered seal again now that it is upright, so the cut is tight and whole."""
    height, width = upright.shape[:2]
    data = client.ask(FRAME_PROMPT, [upright]) or {}
    framed = _grounded_box(data, width, height)
    whole = _told(data, "complete", True) and not _box_ran_off(data)
    center = (width / 2, height / 2)
    if framed is not None:
        fx, fy, fw, fh = framed
        holds_center = fx <= center[0] <= fx + fw and fy <= center[1] <= fy + fh
        # The first mark is sometimes only a corner. Allow the upright box to grow
        # to the whole stamp, and still reject a box that swallows the window.
        sized = 0.3 * _pixel_area(box) <= _pixel_area(framed) <= 4.0 * _pixel_area(box)
        if holds_center and sized:
            pad = max(16, int(0.06 * min(fw, fh)))
            piece = _crop_pixels(upright, framed, pad=pad)
            if piece is not None:
                if _corner_fragment(piece):
                    whole = False
                return piece, whole
    side_w = min(width, box[2])
    side_h = min(height, box[3])
    left = int(center[0] - side_w / 2)
    top = int(center[1] - side_h / 2)
    fallback = upright[top:top + side_h, left:left + side_w].copy()
    return fallback, False if _corner_fragment(fallback) else whole


def _turned_seal(client: OpenRouter, page: np.ndarray, box: tuple[int, int, int, int], grow: float) -> tuple[np.ndarray, bool]:
    tight = _window(page, box, 0.05)
    turn = _coarse_turn(client, tight)
    tilt = _date_tilt(client, _rotate_white(tight, turn))
    angle = turn + tilt
    if abs(angle) >= 2:
        print(f"[Layer 4] seal turned {angle:.0f} degrees", flush=True)
    upright = _rotate_white(_window(page, box, grow), angle)
    return _framed_seal(client, upright, box)


def _crop_fits(page: np.ndarray, piece: np.ndarray) -> bool:
    """A wider look that takes in most of the page is not one stamp."""
    page_h, page_w = page.shape[:2]
    height, width = piece.shape[:2]
    return height * width <= 0.25 * page_h * page_w


def _cut_upright(client: OpenRouter, page: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    piece, whole = _turned_seal(client, page, box, 0.4)
    if whole:
        return piece
    print("[Layer 4] seal edge is cut, looking wider", flush=True)
    wider, whole = _turned_seal(client, page, _grow_box(page, box, 1.2), 0.2)
    if not _crop_fits(page, wider):
        print("[Layer 4] wider crop took in the page, keeping the tighter cut", flush=True)
        return piece
    if whole:
        return wider
    print("[Layer 4] seal still cut, looking once more", flush=True)
    again, _still = _turned_seal(client, page, _grow_box(page, box, 2.0), 0.15)
    if not _crop_fits(page, again):
        return wider
    return again


def _page_seals(client: OpenRouter, page: np.ndarray) -> list[np.ndarray]:
    boxes = _found_seals(client, page)
    with ThreadPoolExecutor(max_workers=SEAL_WORKERS) as pool:
        return list(pool.map(lambda box: _cut_upright(client, page, box), boxes))


def _told(data: dict | None, key: str, default: bool = True) -> bool:
    if not isinstance(data, dict) or key not in data:
        return default
    return _yes(data.get(key))


def _seal_lines(value) -> str:
    """Keep the lines Qwen read. Drop a placeholder and the extra spaces around it."""
    if value is None:
        return ""
    lines = []
    for line in str(value).splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned and cleaned.lower() not in {"null", "none", "n/a", "nil"}:
            lines.append(cleaned)
    return "\n".join(lines)


def _digit_runs(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def _for_reading(image: np.ndarray) -> np.ndarray:
    """Give the readers a larger copy. A small stamp is where a mark gets read as a digit."""
    height, width = image.shape[:2]
    short = min(height, width)
    if short >= 720:
        return image
    scale = 720 / short
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _settled_text(client: OpenRouter, image: np.ndarray, first_text: str, second_text: str) -> str:
    """Where the two readings disagree on a digit, look at the stamp and copy that mark again."""
    if not first_text or not second_text:
        return first_text or second_text
    if _digit_runs(first_text) == _digit_runs(second_text):
        return first_text
    prompt = SETTLE_PROMPT.replace("{first}", first_text).replace("{second}", second_text)
    settled = _seal_lines((client.ask(prompt, [image], model=VISION_MODEL) or {}).get("text"))
    if not settled:
        return first_text
    longer = max(len(first_text.splitlines()), len(second_text.splitlines()))
    if len(settled.splitlines()) > longer + 2:
        return first_text
    return settled


def _country_from_text(client: OpenRouter, text: str, image: np.ndarray | None = None) -> str | None:
    """Qwen names the country from the seal and the OCR text. It does not rewrite the text."""
    if image is None and not text:
        return None
    prompt = COUNTRY_PROMPT.replace("{text}", text or "(nothing read)")
    judged = client.ask(prompt, [image] if image is not None else [], model=VISION_MODEL) or {}
    country = clean_text(judged.get("country"))
    if country and country.isupper():
        country = country.title()
    return country


def _same_reading(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", left).strip().casefold() == re.sub(r"\s+", " ", right).strip().casefold()


def _star_settled_date(qwen_text: str, glm_text: str) -> str:
    """A star beside the day is not a digit. 24 next to a star is the 4th, not the 24th."""
    if "*" not in f"{qwen_text}\n{glm_text}":
        return ""
    found = []
    for text in (qwen_text, glm_text):
        parts = _date_parts(text)
        if parts:
            found.append(parts[0])
    if len(found) < 2:
        return ""
    single = [item for item in found if item["day"] < 10]
    starred = [item for item in found if 20 <= item["day"] <= 29]
    if not single or not starred or single[0]["month"] != starred[0]["month"]:
        return ""
    year = starred[0]["year_text"]
    if len(year) != 4:
        return ""
    return f"{starred[0]['day'] % 10} {starred[0]['month_name']} {year}"


def _date_label(text: str) -> str:
    parts = _date_parts(text)
    if not parts:
        return ""
    item = parts[0]
    return f"{item['day']} {item['month_name']} {item['year_text']}"


def _replace_date(text: str, date_label: str) -> str:
    if not date_label:
        return text
    pattern = re.compile(
        r"\*?[ \t]*\d{1,2}[ \t]*[A-Za-z]{3,9}[ \t]*\d{2,4}[ \t]*\*?",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return pattern.sub(date_label, text, count=1)
    return f"{text}\n{date_label}" if text else date_label


def _choose_reading(client: OpenRouter, image: np.ndarray, qwen_text: str, glm_text: str) -> str:
    """Qwen 3.8 keeps one reading, or mixes the two, and settles a disputed date from the ink."""
    if not qwen_text:
        return glm_text
    if not glm_text or _same_reading(qwen_text, glm_text):
        return qwen_text
    prompt = JUDGE_PROMPT.replace("{a}", qwen_text).replace("{b}", glm_text)
    mixed = _seal_lines((client.ask(prompt, [image], model=MODEL) or {}).get("text")) or glm_text
    settled = _star_settled_date(qwen_text, glm_text)
    if settled:
        return _replace_date(mixed, settled)
    qwen_date = _date_label(qwen_text)
    glm_date = _date_label(glm_text)
    if qwen_date and glm_date and qwen_date != glm_date:
        date_prompt = DATE_PROMPT.replace("{a}", qwen_text).replace("{b}", glm_text)
        decided = str((client.ask(date_prompt, [image], model=MODEL) or {}).get("date") or "").strip()
        mixed = _replace_date(mixed, _seal_lines(decided) or decided)
    return mixed


def _read_seal(client: OpenRouter, image: np.ndarray) -> tuple[bool, str, str | None]:
    """Qwen reads the stamp, GLM reads it again, and Qwen 3.8 keeps or mixes them."""
    image = _for_reading(image)
    qwen = client.ask(READ_PROMPT, [image], model=QWEN_OCR_MODEL) or {}
    glm = client.ask(READ_PROMPT, [image], model=GLM_OCR_MODEL) or {}
    qwen_text = _seal_lines(qwen.get("text"))
    glm_text = _seal_lines(glm.get("text"))
    text = _choose_reading(client, image, qwen_text, glm_text)
    is_seal = _told(qwen, "seal", False) or _told(glm, "seal", False) or bool(_stamp_dates(text))
    return is_seal, text, _country_from_text(client, text, image)


def _write_seals(path: Path, seals: list[dict], status: str, source: str = "") -> dict:
    _refuse_saved(path.parent)
    document = {
        "layer": 4,
        "status": status,
        "model": MODEL,
        "ocr": OCR_MODEL,
        "source": source,
        "seal_count": len(seals),
        "seals": seals,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def _seal_file(item: dict | str) -> str:
    if isinstance(item, str):
        return item
    return str(item.get("file") or "")


def read_saved_seals(out_dir: Path | None = None) -> dict:
    """Read each saved seal with Qwen and store the text beside it."""
    folder = out_dir or DEFAULT_OUT
    path = folder / "layer4_stays.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    client = OpenRouter(load_api_key(DEFAULT_SECRETS))
    items = list(document.get("seals") or [])

    def one(item: dict | str) -> dict:
        file_name = _seal_file(item)
        image = cv2.imread(str(folder / file_name))
        previous = item if isinstance(item, dict) else {}
        if image is None:
            text, country = previous.get("text") or "", previous.get("country")
        else:
            _kept, text, country = _read_seal(client, image)
            if not text:
                text, country = previous.get("text") or "", previous.get("country")
        print(f"[Layer 4] {file_name}: {country or ''} {text}", flush=True)
        return {"file": file_name, "text": text, "country": country}

    with ThreadPoolExecutor(max_workers=SEAL_WORKERS) as pool:
        seals = list(pool.map(one, items))
    document["seals"] = seals
    document["ocr"] = OCR_MODEL
    document["seal_count"] = len(seals)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def _refuse_saved(folder: Path) -> None:
    """The saved test image is kept. A new passport must not write over it."""
    resolved = folder.resolve()
    saved = SAVED_DIR.resolve()
    if resolved == saved or saved in resolved.parents:
        raise SystemExit("The saved folder is kept. New extracts are written to output/.")


def layer4(pages: list[dict], client: OpenRouter, out_dir: Path | None = None, source: str = "") -> dict:
    folder = out_dir or DEFAULT_OUT
    _refuse_saved(folder)
    stays_path = folder / "layer4_stays.json"
    seal_dir = folder / "seals"
    if seal_dir.is_dir():
        for old in seal_dir.glob("*"):
            if old.is_file():
                old.unlink()
    seal_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    _write_seals(stays_path, images, "running", source)
    for page in pages:
        if page.get("image") is None:
            continue
        page_image = _readable(page["image"])
        if _ink_fraction(page_image) >= 0.97:
            print(f"[Layer 4] page {page['number']}: no seals", flush=True)
            continue
        pieces = _page_seals(client, page_image)
        print(f"[Layer 4] page {page['number']}: {len(pieces)} seals cut", flush=True)
        page_start = len(images)
        for offset, piece in enumerate(pieces, start=1):
            name = f"{page_start + offset}.jpg"
            cv2.imwrite(str(seal_dir / name), piece, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            images.append({"file": f"seals/{name}", "text": "", "country": None})
        _write_seals(stays_path, images, "running", source)
        with ThreadPoolExecutor(max_workers=SEAL_WORKERS) as pool:
            checked = list(pool.map(lambda piece: _read_seal(client, piece), pieces))
        del images[page_start:]
        for piece, (is_seal, text, country) in zip(pieces, checked):
            if not is_seal:
                print("[Layer 4] not a seal, dropped", flush=True)
                continue
            name = f"{len(images) + 1}.jpg"
            cv2.imwrite(str(seal_dir / name), piece, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            images.append({"file": f"seals/{name}", "text": text, "country": country})
            print(f"[Layer 4] {name}: {country or ''} {text}", flush=True)
        for stale in range(len(images) + 1, page_start + len(pieces) + 1):
            (seal_dir / f"{stale}.jpg").unlink(missing_ok=True)
        _write_seals(stays_path, images, "running", source)
    _unify_seal_dates(images)
    document = _write_seals(stays_path, images, "finished", source)
    print(f"[Layer 4] finished: {len(images)} seals", flush=True)
    events, stays = _layer2(client, images)
    _write_dates(folder / "layer5_dates.json", events, source)
    _write_layer6(folder / "layer6_stays.json", stays, source)
    return document


def _stay_field(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()[:200]


def _date_parts(text: str) -> list[dict]:
    """Read every recognizable date without deciding which trip it belongs to."""
    cleaned = re.sub(r"\s+", " ", text.upper())
    cleaned = cleaned.replace("0CT", "OCT").replace("NOY", "NOV").replace("5EP", "SEP")
    parts: list[dict] = []
    occupied: list[tuple[int, int]] = []

    def add(match, day_text: str, month_text: str, year_text: str) -> None:
        month = month_text[:3]
        day = int(day_text)
        if month not in MONTHS or not 1 <= day <= 31:
            return
        span = match.span()
        if any(span[0] < right and span[1] > left for left, right in occupied):
            return
        occupied.append(span)
        parts.append({
            "day": day,
            "month": MONTHS[month],
            "month_name": month,
            "year_text": year_text,
            "raw": match.group(0).strip(),
        })

    for match in re.finditer(r"\b(\d{1,2})\s*([A-Z]{3,9})\s*(\d{2,4})\b", cleaned):
        add(match, match.group(1), match.group(2), match.group(3))
    for match in re.finditer(r"\b([A-Z]{3,9})\s*(\d{1,2})[,.]?\s*(\d{2,4})\b", cleaned):
        year_at = match.start(3)
        if year_at > 0 and cleaned[year_at - 1].isdigit():
            continue
        add(match, match.group(2), match.group(1), match.group(3))

    for match in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", cleaned):
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        else:
            day, month = first, second
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        span = match.span()
        if any(span[0] < right and span[1] > left for left, right in occupied):
            continue
        occupied.append(span)
        parts.append({
            "day": day,
            "month": month,
            "month_name": MONTH_NAMES[month],
            "year_text": match.group(3),
            "raw": match.group(0).strip(),
        })
    return parts


# A stamp year is never older than 1980. When one is, its third digit is taken
# from the other dates: 8 or 9 keeps the 1980s or 1990s, and 0, 1, or 2 means 2000–2029.
_YEAR_THIRD_DIGITS = (8, 9, 0, 1, 2)


def _year_from_text(year_text: str, full_years: set[int]) -> int | None:
    if len(year_text) == 4:
        year = int(year_text)
    elif len(year_text) == 3:
        matches = [value for value in full_years if str(value).startswith(year_text)]
        if len(matches) != 1:
            return None
        year = matches[0]
    elif len(year_text) == 2:
        number = int(year_text)
        year = 1900 + number if number >= 40 else 2000 + number
    else:
        return None
    if 1940 <= year <= 2035:
        return year
    return None


def _third_digit(year: int) -> int:
    return (year // 10) % 10


def _predict_third_digit(years: list[int]) -> int:
    """Pick 8, 9, 0, 1, or 2 from the dates that are already 1980 or later."""
    counts = {digit: 0 for digit in _YEAR_THIRD_DIGITS}
    for year in years:
        if year < 1980:
            continue
        digit = _third_digit(year)
        if digit in counts:
            counts[digit] += 1
    best = max(counts.values())
    if best:
        for digit in _YEAR_THIRD_DIGITS:
            if counts[digit] == best:
                return digit
    return 8


def _lift_year(year: int | None, years: list[int]) -> int | None:
    """Replace a year older than 1980, keeping its last digit."""
    if year is None or year >= 1980:
        return year
    digit = _predict_third_digit(years)
    last = year % 10
    if digit in {8, 9}:
        return 1900 + digit * 10 + last
    return 2000 + digit * 10 + last


def _repair_month_letters(text: str) -> str:
    return (
        text.replace("0CT", "OCT")
        .replace("0ct", "OCT")
        .replace("NOY", "NOV")
        .replace("noy", "NOV")
        .replace("5EP", "SEP")
        .replace("5ep", "SEP")
    )


def _rewrite_dates(text: str, full_years: set[int], page_years: list[int] | None = None) -> str:
    """Replace every recognizable date with DD MON YYYY."""
    if not text:
        return text
    source = _repair_month_letters(text)
    spans: list[tuple[int, int, str]] = []

    def take(start: int, end: int, day, month, year_text: str) -> None:
        if any(start < right and end > left for left, right, _label in spans):
            return
        if isinstance(month, int):
            if month not in MONTH_NAMES:
                return
            month_num = month
            month_name = MONTH_NAMES[month]
        else:
            month_name = str(month)[:3].upper()
            if month_name not in MONTHS:
                return
            month_num = MONTHS[month_name]
        day_n = int(day)
        if not 1 <= day_n <= 31:
            return
        year = _lift_year(_year_from_text(str(year_text), full_years), page_years or [])
        if year is None:
            return
        try:
            date(year, month_num, day_n)
        except ValueError:
            return
        spans.append((start, end, _format_date(day_n, month_name, year)))

    for match in re.finditer(r"\b(\d{1,2})\s*([A-Za-z]{3,9})\s*(\d{2,4})\b", source):
        take(match.start(), match.end(), match.group(1), match.group(2), match.group(3))
    for match in re.finditer(r"\b([A-Za-z]{3,9})\s*(\d{1,2})[,.]?\s*(\d{2,4})\b", source):
        year_at = match.start(3)
        if year_at > 0 and source[year_at - 1].isdigit():
            continue
        take(match.start(), match.end(), match.group(2), match.group(1), match.group(3))
    for match in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", source):
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            day, month = second, first
        else:
            day, month = first, second
        take(match.start(), match.end(), day, month, match.group(3))
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", source):
        take(match.start(), match.end(), match.group(3), int(match.group(2)), match.group(1))

    if not spans:
        return source
    spans.sort()
    pieces: list[str] = []
    cursor = 0
    for start, end, label in spans:
        if start < cursor:
            continue
        pieces.append(source[cursor:start])
        pieces.append(label)
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def _resolved_years(text: str, full_years: set[int] | None = None) -> list[int]:
    known = set(full_years or ())
    if not known:
        known = {
            int(part["year_text"])
            for part in _date_parts(text)
            if len(part["year_text"]) == 4
        }
    found = []
    for part in _date_parts(text):
        year = _year_from_text(part["year_text"], known)
        if year:
            found.append(year)
    return found


def format_stay_date(value, page_years: list[int] | None = None) -> str:
    """Return one stay date as DD MON YYYY, or blank when there is no date."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    years = {
        int(part["year_text"])
        for part in _date_parts(text)
        if len(part["year_text"]) == 4
    }
    if page_years is None:
        page_years = _resolved_years(text, years)
    return re.sub(r"\s+", " ", _rewrite_dates(text, years, page_years)).strip()


def format_stay_fields(values: list[str]) -> list[str]:
    """Format every stay date, using the other dates to repair a year before 1980."""
    page_years: list[int] = []
    for value in values:
        page_years.extend(_resolved_years(str(value or "")))
    return [format_stay_date(value, page_years) for value in values]


def _seal_date_context(seals: list[dict]) -> tuple[dict[str, set[int]], list[int]]:
    years: dict[str, set[int]] = {}
    keyed: list[tuple[str, str]] = []
    for stamp, seal in enumerate(seals, start=1):
        country = _stay_field(seal.get("country"))
        key = country.casefold() or f"unknown-{stamp}"
        text = str(seal.get("text") or "")
        keyed.append((key, text))
        for part in _date_parts(text):
            if len(part["year_text"]) == 4:
                years.setdefault(key, set()).add(int(part["year_text"]))
    page_years: list[int] = []
    for key, text in keyed:
        page_years.extend(_resolved_years(text, years.get(key, set())))
    return years, page_years


def _unify_seal_dates(seals: list[dict]) -> None:
    """Write every date inside a seal reading as DD MON YYYY."""
    years, page_years = _seal_date_context(seals)
    for stamp, seal in enumerate(seals, start=1):
        country = _stay_field(seal.get("country"))
        key = country.casefold() or f"unknown-{stamp}"
        seal["text"] = _rewrite_dates(str(seal.get("text") or ""), years.get(key, set()), page_years)


def _dated_events(seals: list[dict]) -> list[dict]:
    """Keep every readable date and complete only an unambiguous cut-off year."""
    raw: list[dict] = []
    full_years: dict[str, set[int]] = {}
    for stamp, seal in enumerate(seals, start=1):
        country = _stay_field(seal.get("country"))
        key = country.casefold() or f"unknown-{stamp}"
        for part in _date_parts(str(seal.get("text") or "")):
            year_text = part["year_text"]
            if len(year_text) == 4:
                full_years.setdefault(key, set()).add(int(year_text))
            raw.append({
                **part,
                "stamp": stamp,
                "file": str(seal.get("file") or f"seals/{stamp}.jpg"),
                "country": country,
                "key": key,
                "text": str(seal.get("text") or "").strip(),
            })

    resolved = [
        (item, _year_from_text(item["year_text"], full_years.get(item["key"], set())))
        for item in raw
    ]
    page_years = [year for _item, year in resolved if year]
    events = []
    for item, year in resolved:
        year = _lift_year(year, page_years)
        when = None
        if year and 1940 <= year <= 2035:
            try:
                when = date(year, item["month"], item["day"])
            except ValueError:
                pass
        label = _format_date(item["day"], item["month_name"], year) if when is not None else item["raw"]
        events.append({
            "id": len(events) + 1,
            "stamp": item["stamp"],
            "file": item["file"],
            "country": item["country"],
            "key": item["key"],
            "date": label,
            "when": when,
            "text": item["text"],
            "role": "unknown",
        })
    return events


def _pair_events(events: list[dict]) -> list[dict]:
    """Pair only dates from the same country, earlier first, without losing any date."""
    grouped: dict[str, list[dict]] = {}
    for event in events:
        grouped.setdefault(event["key"], []).append(event)

    stays = []
    for group in grouped.values():
        group.sort(key=lambda event: (event["when"] is None, event["when"] or date.max, event["id"]))
        index = 0
        while index < len(group):
            first = group[index]
            second = group[index + 1] if index + 1 < len(group) else None
            pair = False
            if first["when"] is not None and second and second["when"] is not None:
                gap = (second["when"] - first["when"]).days
                # Same-country dates close together are one stay. Chronology wins
                # over a model that accidentally reverses arrival and departure.
                pair = 0 <= gap <= 183
            if pair:
                start, end = first["date"], second["date"]
                index += 2
            elif first["role"] == "departure":
                start, end = "", first["date"]
                index += 1
            else:
                start, end = first["date"], ""
                index += 1

            country = first["country"]
            home = country.casefold() == "malaysia"
            stays.append({
                "departure_country": "" if home else ("Malaysia" if country else ""),
                "arrival_country": country,
                "start": start,
                "end": end,
                "_when": first["when"],
                "_id": first["id"],
            })

    stays.sort(key=lambda item: (item["_when"] is None, item["_when"] or date.max, item["_id"]))
    for stay in stays:
        del stay["_when"]
        del stay["_id"]
    return stays


def _layer2(client: OpenRouter, seals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Qwen classifies each date; deterministic matching keeps all dates valid."""
    events = _dated_events(seals)
    if not events:
        return []
    blocks = [
        f"Event {event['id']}\nStamp: {event['stamp']}\n"
        f"Country: {event['country'] or 'unknown'}\nDate: {event['date']}\n"
        f"Stamp text:\n{event['text'] or '(no text)'}"
        for event in events
    ]
    print("[Layer 5] classifying each date", flush=True)
    data = client.ask(STAY_PROMPT.replace("{events}", "\n\n".join(blocks)), [], model=STAY_MODEL) or {}
    classified = data.get("events") if isinstance(data, dict) else None
    roles = {}
    for item in classified if isinstance(classified, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            event_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role in {"arrival", "departure", "visa duration", "unknown"}:
            roles[event_id] = role
    for event in events:
        event["role"] = roles.get(event["id"], "unknown")
    stays = _pair_events(events)
    print(f"[Layer 6] {len(stays)} trips", flush=True)
    return events, stays


def _date_rows(events: list[dict]) -> list[dict]:
    ordered = sorted(
        events,
        key=lambda event: (event.get("when") is None, event.get("when") or date.max, event.get("id") or 0),
    )
    return [
        {
            "country": event["country"],
            "date": event["date"],
            "type": event["role"],
            "file": event.get("file") or "",
        }
        for event in ordered
    ]


def _write_dates(path: Path, events: list[dict], source: str = "") -> dict:
    _refuse_saved(path.parent)
    document = {
        "layer": 5,
        "status": "finished",
        "model": STAY_MODEL,
        "source": source,
        "dates": _date_rows(events),
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def _write_layer6(path: Path, stays: list[dict], source: str = "") -> dict:
    _refuse_saved(path.parent)
    document = {
        "layer": 6,
        "status": "finished",
        "model": STAY_MODEL,
        "source": source,
        "stays": stays,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document



def portrait_crop(image: np.ndarray) -> np.ndarray | None:
    image = upscale(image, 1200)
    height, width = image.shape[:2]
    model = ROOT / "models" / "face_detection_yunet.onnx"
    if not model.is_file() or image is None or image.size == 0:
        return None
    detector = cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.6, 0.3, 5000)
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
    x, y, face_w, face_h = (int(value) for value in face[:4])
    x0 = max(0, x - int(face_w * 0.25))
    y0 = max(0, y - int(face_h * 0.18))
    x1 = min(width, x + face_w + int(face_w * 0.25))
    y1 = min(height, y + face_h + int(face_h * 0.28))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def layer3(pages: list[dict], layer1_doc: dict, out_dir: Path) -> dict | None:
    page_no = layer1_doc.get("page")
    page = next((item for item in pages if item["number"] == page_no), None)
    if page is None or page.get("image") is None:
        return None
    crop = portrait_crop(page["image"])
    if crop is None:
        return None
    cv2.imwrite(str(out_dir / "portrait.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return {
        "layer": 3,
        "status": "finished",
        "page": page_no,
        "image": "portrait.jpg",
    }


def _write_json(path: Path, document: dict) -> None:
    _refuse_saved(path.parent)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def run(source: Path, secrets_path: Path, out_dir: Path) -> None:
    _refuse_saved(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "layer1_name.json",
        "layer2_attributes.json",
        "layer3_image.json",
        "layer2_stays.json",
        "layer_dates.json",
        "layer5_dates.json",
        "layer6_stays.json",
        "layer4_stays.json",
        "portrait.jpg",
    ):
        path = out_dir / stale
        if path.is_file():
            path.unlink()
    pages = read_pages(source)
    if not any(page.get("image") is not None for page in pages):
        raise SystemExit("No pages could be read from this file.")

    client = OpenRouter(load_api_key(secrets_path))
    biodata_pages = _biodata_pages(pages)
    holder = None
    if biodata_pages:
        print("[Layer 1] reading the name with Qwen OCR", flush=True)
        holder = layer1_from_image(client, biodata_pages)
        if not holder:
            print("[Layer 1] looking for the holder name", flush=True)
            holder = layer1(client, biodata_pages)
    if holder and holder.get("name"):
        from_image = bool(holder.pop("from_image", False))
        _write_json(out_dir / "layer1_name.json", holder)
        print(f"[Layer 1] {holder['name']}", flush=True)
        print("[Layer 2] reading the biodata", flush=True)
        details = layer2(client, pages, holder, from_image=from_image)
        if details.get("attributes"):
            _write_json(out_dir / "layer2_attributes.json", details)
            print(f"[Layer 2] {len(details['attributes'])} fields", flush=True)
        portrait = layer3(pages, holder, out_dir)
        if portrait:
            _write_json(out_dir / "layer3_image.json", portrait)
            print("[Layer 3] portrait saved", flush=True)
    biodata_numbers = {page["number"] for page in biodata_pages}
    seal_pages = [page for page in pages if page.get("number") not in biodata_numbers]
    if not seal_pages:
        print("[Layer 4] biodata page, seals not read", flush=True)
        return
    print("[Layer 4] finding seals", flush=True)
    stays = layer4(seal_pages, client, out_dir, source.name)
    print(f"[Layer 4] finished: {len(stays['seals'])} seals -> layer4_stays.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a Malaysian passport.")
    parser.add_argument("--source", type=Path, required=True, help="Passport PDF or image")
    parser.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS, help="File that holds OPENROUTER_API_KEY")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder for the extraction result")
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"File not found: {args.source}")
    try:
        run(args.source, args.secrets, args.out)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
