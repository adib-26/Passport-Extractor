const statusNode = document.querySelector("#status");
const results = document.querySelector("#results");
const fileInput = document.querySelector("#file");
const runButton = document.querySelector("#run");

function text(value) {
  return document.createTextNode(value == null ? "" : String(value));
}

function card(title, body) {
  const section = document.createElement("section");
  section.className = "card";
  const heading = document.createElement("h2");
  heading.append(text(title));
  section.append(heading, body);
  return section;
}

function sealCount(layer4) {
  const total = Number.isFinite(layer4.seal_count) ? layer4.seal_count : layer4.seals.length;
  return total === 1 ? "1 seal" : `${total} seals`;
}

function galleryFor(layer4, prefix) {
  const gallery = document.createElement("div");
  gallery.className = "seal-gallery";
  if (!layer4.seals.length && layer4.status === "finished") {
    const note = document.createElement("p");
    note.className = "empty";
    note.append(text("No seals were found."));
    gallery.append(note);
    return gallery;
  }
  for (const seal of layer4.seals) {
    const file = typeof seal === "string" ? seal : seal.file;
    const reading = typeof seal === "string" ? "" : String(seal.text || "").trim();
    const country = typeof seal === "string" ? "" : String(seal.country || "").trim();
    const figure = document.createElement("figure");
    const photo = document.createElement("img");
    photo.alt = country || reading || "Extracted seal";
    photo.src = `${prefix}${file}`;
    const caption = document.createElement("figcaption");
    if (country) {
      const place = document.createElement("span");
      place.className = "country";
      place.append(text(country));
      caption.append(place);
    }
    if (reading) {
      const writing = document.createElement("span");
      writing.className = "reading";
      writing.append(text(reading));
      caption.append(writing);
    }
    figure.append(photo, caption);
    if (typeof seal !== "string" && seal.file) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "seal-edit";
      button.textContent = "Edit";
      button.addEventListener("click", () => openSeal(seal.file));
      figure.append(button);
    }
    gallery.append(figure);
  }
  return gallery;
}

const sealDialog = document.querySelector("#seal-editor");
const sealForm = document.querySelector("#seal-form");
const sealCancel = document.querySelector("#seal-cancel");
let seals = [];
let editingSeal = "";

function openSeal(file) {
  const seal = seals.find((item) => item.file === file);
  if (!seal) return;
  editingSeal = file;
  sealForm.elements.country.value = seal.country;
  sealForm.elements.text.value = seal.text;
  sealDialog.showModal();
}

sealCancel.addEventListener("click", () => sealDialog.close());

sealForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!editingSeal) return;
  const next = seals.map((seal) => (
    seal.file === editingSeal
      ? {
        file: seal.file,
        country: sealForm.elements.country.value.trim(),
        text: sealForm.elements.text.value.trim(),
      }
      : { file: seal.file, country: seal.country, text: seal.text }
  ));
  const submit = sealForm.querySelector("button[type='submit']");
  submit.disabled = true;
  const response = await fetch("/api/layer4", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seals: next }),
  });
  submit.disabled = false;
  const data = await response.json();
  if (!data.ok) {
    statusNode.textContent = data.error || "The edit was not saved";
    statusNode.className = "status error";
    return;
  }
  sealDialog.close();
  editingSeal = "";
  refresh();
});

const stayDialog = document.querySelector("#stay-editor");
const stayForm = document.querySelector("#stay-form");
const stayCancel = document.querySelector("#stay-cancel");
let stays = [];
let editing = -1;

function fieldText(stay, key) {
  return String((stay && stay[key]) || "").trim();
}

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function toDateInput(value) {
  const match = String(value || "").trim().match(/^(\d{2}) ([A-Z]{3}) (\d{4})$/);
  if (!match) return "";
  const month = MONTHS.indexOf(match[2]) + 1;
  if (month < 1) return "";
  return `${match[3]}-${String(month).padStart(2, "0")}-${match[1]}`;
}

function fromDateInput(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return "";
  const month = MONTHS[Number(match[2]) - 1];
  if (!month) return "";
  return `${match[3]} ${month} ${match[1]}`;
}

const DATE_TYPES = {
  arrival: "Arrival",
  departure: "Departure",
  "visa duration": "Visa duration",
  unknown: "Unknown",
};

function dateValue(value) {
  const match = String(value || "").trim().match(/^(\d{2}) ([A-Z]{3}) (\d{4})$/);
  if (!match) return "9999-99-99";
  const month = MONTHS.indexOf(match[2]) + 1;
  if (month < 1) return "9999-99-99";
  return `${match[3]}-${String(month).padStart(2, "0")}-${match[1]}`;
}

function datesFor(layer) {
  const list = document.createElement("div");
  list.className = "date-list";
  const rows = (Array.isArray(layer.dates) ? layer.dates : []).slice().sort((left, right) => {
    const order = dateValue(left.date).localeCompare(dateValue(right.date));
    return order;
  });
  if (!rows.length) {
    const note = document.createElement("p");
    note.className = "empty";
    note.append(text("No dates were read from the stamps."));
    list.append(note);
    return list;
  }
  rows.forEach((row) => {
    const article = document.createElement("article");
    article.className = "date-row";
    const values = [
      ["Country", fieldText(row, "country") || "—"],
      ["Date", fieldText(row, "date") || "—"],
      ["Type", DATE_TYPES[fieldText(row, "type").toLowerCase()] || "Unknown"],
    ];
    for (const [label, value] of values) {
      const block = document.createElement("div");
      const name = document.createElement("span");
      name.className = "stay-label";
      name.append(text(label));
      const shown = document.createElement("span");
      shown.append(text(value));
      block.append(name, shown);
      article.append(block);
    }
    const file = fieldText(row, "file");
    if (/^seals\/\d+\.jpg$/.test(file)) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Stamp";
      button.addEventListener("click", () => openStamp(file));
      article.append(button);
    }
    list.append(article);
  });
  return list;
}

const stampDialog = document.querySelector("#stamp-view");
const stampPhoto = document.querySelector("#stamp-photo");
document.querySelector("#stamp-close").addEventListener("click", () => stampDialog.close());

function openStamp(file) {
  stampPhoto.src = `/${file}`;
  stampDialog.showModal();
}

function staysFor(layer2) {
  const list = document.createElement("div");
  list.className = "stay-list";
  stays = Array.isArray(layer2.stays) ? layer2.stays.map((stay) => ({
    departure_country: fieldText(stay, "departure_country"),
    arrival_country: fieldText(stay, "arrival_country"),
    start: fieldText(stay, "start"),
    end: fieldText(stay, "end"),
  })) : [];
  const add = document.createElement("button");
  add.type = "button";
  add.className = "stay-add";
  add.textContent = "Add stay";
  add.addEventListener("click", openNewStay);
  list.append(add);
  if (!stays.length) {
    const note = document.createElement("p");
    note.className = "empty";
    note.append(text("No trips could be read from the stamps."));
    list.append(note);
    return list;
  }
  stays.forEach((stay, index) => {
    const row = document.createElement("article");
    row.className = "stay";
    for (const [label, key] of [
      ["Departure country", "departure_country"],
      ["Arrival country", "arrival_country"],
      ["Start date", "start"],
      ["End date", "end"],
    ]) {
      const block = document.createElement("div");
      const name = document.createElement("span");
      name.className = "stay-label";
      name.append(text(label));
      const value = document.createElement("span");
      value.append(text(stay[key] || "—"));
      block.append(name, value);
      row.append(block);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Edit";
    button.addEventListener("click", () => openStay(index));
    row.append(button);
    list.append(row);
  });
  return list;
}

function openStay(index) {
  const stay = stays[index];
  if (!stay) return;
  editing = index;
  document.querySelector("#stay-title").textContent = "Edit stay";
  stayForm.elements.departure_country.value = stay.departure_country;
  stayForm.elements.arrival_country.value = stay.arrival_country;
  stayForm.elements.start.value = toDateInput(stay.start);
  stayForm.elements.end.value = toDateInput(stay.end);
  stayDialog.showModal();
}

function openNewStay() {
  editing = stays.length;
  document.querySelector("#stay-title").textContent = "Add stay";
  stayForm.reset();
  stayDialog.showModal();
}

stayCancel.addEventListener("click", () => stayDialog.close());

stayForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (editing < 0 || editing > stays.length) return;
  const startValue = stayForm.elements.start.value;
  const endValue = stayForm.elements.end.value;
  if (startValue && endValue && endValue < startValue) {
    stayForm.elements.end.setCustomValidity("The end date cannot be earlier than the start date.");
    stayForm.reportValidity();
    stayForm.elements.end.setCustomValidity("");
    return;
  }
  if (!stayForm.reportValidity()) return;
  const next = stays.map((stay) => ({ ...stay }));
  const draft = {
    departure_country: stayForm.elements.departure_country.value.trim(),
    arrival_country: stayForm.elements.arrival_country.value.trim(),
    start: fromDateInput(startValue),
    end: fromDateInput(endValue),
  };
  if (editing === stays.length) next.push(draft);
  else next[editing] = draft;
  const submit = stayForm.querySelector("button[type='submit']");
  submit.disabled = true;
  const response = await fetch("/api/layer6", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stays: next }),
  });
  submit.disabled = false;
  const data = await response.json();
  if (!data.ok) {
    statusNode.textContent = data.error || "The edit was not saved";
    statusNode.className = "status error";
    return;
  }
  stayDialog.close();
  editing = -1;
  refresh();
});

const ATTRIBUTE_LABELS = {
  passport_number: "Passport number",
  nationality: "Nationality",
  date_of_birth: "Date of birth",
  place_of_birth: "Place of birth",
  sex: "Sex",
  height: "Height",
  identity_number: "Identity number",
  date_of_issue: "Date of issue",
  date_of_expiry: "Date of expiry",
  issuing_office: "Issuing office",
  type: "Type",
  country_code: "Country code",
};

function appendIdentity(data, parent) {
  const name = data.layer1 && String(data.layer1.name || "").trim();
  if (name) {
    const body = document.createElement("p");
    body.append(text(name));
    parent.append(card("Layer 1", body));
  }
  const attributes = data.layer2 && data.layer2.attributes;
  const fields = attributes ? Object.entries(attributes).filter(([, value]) => String(value || "").trim()) : [];
  if (fields.length) {
    const list = document.createElement("dl");
    for (const [key, value] of fields) {
      const term = document.createElement("dt");
      term.append(text(ATTRIBUTE_LABELS[key] || key));
      const detail = document.createElement("dd");
      detail.append(text(value));
      list.append(term, detail);
    }
    parent.append(card("Layer 2", list));
  }
  if (data.layer3 && data.layer3.image === "portrait.jpg") {
    const photo = document.createElement("img");
    photo.className = "portrait";
    photo.alt = "Passport portrait";
    photo.src = "/portrait.jpg";
    parent.append(card("Layer 3", photo));
  }
}

function render(data) {
  const layer4 = data.layer4;
  results.replaceChildren();
  if (data.error) {
    statusNode.textContent = data.error;
    statusNode.className = "status error";
  } else if (data.running) {
    statusNode.textContent = data.phase || "Reading the passport";
    statusNode.className = "status";
  } else if (data.source) {
    statusNode.textContent = data.phase || "Finished";
    statusNode.className = "status";
  } else if (layer4 && layer4.status === "running") {
    statusNode.textContent = "Reading the passport";
    statusNode.className = "status";
  } else if (layer4) {
    statusNode.textContent = "Extract finished.";
    statusNode.className = "status";
  } else {
    statusNode.textContent = "Upload a Malaysian passport PDF.";
    statusNode.className = "status";
  }

  if (!layer4 && !data.layer1 && !data.layer2 && !data.layer3 && !data.running) {
    const note = document.createElement("p");
    note.className = "empty";
    note.append(text("Upload a Malaysian passport. A biodata page shows the holder. A stamp page shows the seals, dates, and stays."));
    results.append(note);
    return;
  }

  appendIdentity(data, results);

  if (layer4 && layer4.status === "running" && !layer4.seals.length) {
    const note = document.createElement("p");
    note.className = "empty";
    note.append(text("Reading the seals."));
    results.append(note);
    return;
  }

  if (layer4 && Array.isArray(layer4.seals) && layer4.seals.length) {
    seals = layer4.seals.filter((seal) => typeof seal !== "string").map((seal) => ({
      file: seal.file,
      text: String(seal.text || ""),
      country: String(seal.country || ""),
    }));
    const name = layer4.source || data.source || "This passport";
    results.append(card(`Layer 4 · ${name} · ${sealCount(layer4)}`, galleryFor(layer4, "/")));
    if (data.layer5 && Array.isArray(data.layer5.dates)) {
      results.append(card("Layer 5", datesFor(data.layer5)));
    }
    if (data.layer6 && Array.isArray(data.layer6.stays)) {
      results.append(card("Layer 6", staysFor(data.layer6)));
    }
  }
}

async function refresh() {
  const response = await fetch("/api/status");
  const data = await response.json();
  render(data);
  if (data.running || (data.layer4 && data.layer4.status === "running")) setTimeout(refresh, 800);
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  statusNode.textContent = "Uploading";
  const response = await fetch("/api/upload", { method: "POST", body });
  const data = await response.json();
  if (!data.ok) {
    statusNode.textContent = data.error || "Upload failed";
    statusNode.className = "status error";
    return;
  }
  refresh();
});

runButton.addEventListener("click", async () => {
  const response = await fetch("/api/run", { method: "POST" });
  const data = await response.json();
  if (!data.ok) {
    statusNode.textContent = data.error || "Could not start";
    statusNode.className = "status error";
    return;
  }
  refresh();
});

refresh();
