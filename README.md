# Passport Extractor

## About

Passport Extractor reads a Malaysian passport on this computer and shows the result in a browser. A biodata page produces the holder’s name, the printed details, and a portrait. A stamp page produces each immigration or visa seal, the dates on those seals, the stays built from those dates, and a short summary of the travels.

The two kinds of page stay separate. A biodata page is not searched for seals, and a stamp page is not treated as the holder page. Uploads and results remain on this machine. The repository does not contain passport scans or an API key.

## What you need

- Python 3.10 or newer
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), used to tell a biodata page from a stamp page
- An [OpenRouter](https://openrouter.ai/) API key

The holder page and each seal image are sent to OpenRouter for reading. The holder page is the page as the uploaded file shows it. Seal finding uses the photograph with the most ink when a scan stores more than one picture of the same page. The site itself listens only on `127.0.0.1`.

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

Open http://127.0.0.1:8766, upload a PDF or image, and the page shows each layer as soon as that layer finishes. Use **Run again** to repeat the last file.

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
| 7 | Stamps | One paragraph summarizing every stay, using the name from Layer 1 |

Empty layers are omitted. Dates are written as `DD MON YYYY`, for example `16 FEB 1936`. A printed year is kept. If the other dates on the same page agree, and one date is 30 years or more away from that year, that one date is brought into line. A nearby year, such as 1980 beside 1970, is left as read.

On a stamp page you can correct a seal, a date, or a stay, and you can add a stay. A stay edit includes the countries and the start and end dates. **Exit stamp** and **Entry stamp** open the departure and arrival stamps for that stay. **Page** opens the uploaded page with a box around that seal. In Layer 4, a seal card on the left or in the middle opens the page on the right; a card in the right column opens it on the left. Each date and each stay can be marked **Agree**, **Don't agree**, or **Unsure**. For this passport, departure and arrival cannot both be Malaysia. Layer 7 is written on this computer from the stays. It does not call a model.

## Models

| Task | Model |
| --- | --- |
| Holder name and biodata fields | `qwen/qwen3.8-flash` |
| Finding and framing seals, and choosing between two readings | `qwen/qwen3.8-flash` |
| First reading of a seal | `qwen/qwen3-vl-235b-a22b-instruct` |
| Second reading of a seal | `z-ai/glm-5.3-flashx` |
| Date type and stays | `qwen/qwen3-vl-8b-instruct` |
| Portrait | YuNet, from `models/face_detection_yunet.onnx` |
| Travel summary | None. Written on this computer from the stays and the Layer 1 name |

The portrait is found on this machine. It is not sent as a separate request. The box around a seal is also found on this machine, by matching the seal crop to the uploaded page.

## Layout

```
extract.py          extraction
server.py           local site
web/                page, script, and styles
models/             face detector
secrets.env.example key template
```

Uploads, extraction output, the virtual environment, and `secrets.env` stay on this machine.
