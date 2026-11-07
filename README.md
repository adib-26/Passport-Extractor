# Passport Extractor

Local tool for a Malaysian passport. It reads the holder page, or the immigration and visa stamps, and shows the result in a browser on this machine.

A biodata page produces the holder’s name, the printed details, and a portrait. A stamp page produces each seal, the dates on those seals, and the stays built from those dates. The two kinds of page are not mixed: a biodata page is not searched for seals, and a stamp page is not treated as the holder page.

## What you need

- Python 3.10 or newer
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), used to tell a biodata page from a stamp page
- An [OpenRouter](https://openrouter.ai/) API key

The holder page and each seal image are sent to OpenRouter for reading. The site itself listens only on `127.0.0.1`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp secrets.env.example secrets.env
```

On macOS, install Tesseract with `brew install tesseract`. On Debian or Ubuntu, use `sudo apt install tesseract-ocr`.

Open `secrets.env` and set `OPENROUTER_API_KEY`. That file is not part of the repository.

## Run the site

```bash
.venv/bin/python server.py
```

Open http://127.0.0.1:8766, upload a PDF or image, and wait for the extract to finish. Use **Run again** to repeat the last file.

You can also run the extractor on its own:

```bash
.venv/bin/python extract.py --source passport.pdf
```

`--out` chooses the result folder. It defaults to `output/`.

## Results

| Layer | Page | Contents |
| --- | --- | --- |
| 1 | Biodata | Holder name |
| 2 | Biodata | Passport number, nationality, dates, sex, height, identity number, issuing office, type, and country code, when printed |
| 3 | Biodata | Portrait |
| 4 | Stamps | Each seal, with the country and the stamp text |
| 5 | Stamps | Country, date, and whether that date is an arrival, a departure, a visa duration, or unknown |
| 6 | Stamps | Stays: departure country, arrival country, start date, and end date |

Empty layers are omitted. Dates are written as `DD MON YYYY`, for example `16 FEB 1936`.

On a stamp page you can correct a seal or a stay, and you can add a stay. For this passport, departure and arrival cannot both be Malaysia. An end date cannot be earlier than the start date.

## Models

| Task | Model |
| --- | --- |
| Holder name and biodata fields | `qwen/qwen3-vl-235b-a22b-instruct` |
| Finding and framing seals, and choosing between two readings | `qwen/qwen3.8-flash` |
| First reading of a seal | `qwen/qwen3-vl-235b-a22b-instruct` |
| Second reading of a seal | `z-ai/glm-5.3-flashx` |
| Date type and stays | `qwen/qwen3-vl-8b-instruct` |
| Portrait | YuNet, from `models/face_detection_yunet.onnx` |

The portrait is found on this machine. It is not sent as a separate request.

## Layout

```
extract.py          extraction
server.py           local site
web/                page, script, and styles
models/             face detector
secrets.env.example key template
```

Uploads, extraction output, the virtual environment, and `secrets.env` stay on this machine.
