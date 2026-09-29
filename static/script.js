const CONFIG = {
  CHAT_URL: "/rag/query",
  UPLOAD_URL: "/api/upload",
};

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const composer = $("composer");
const questionEl = $("question");
const sendBtn = $("sendBtn");
const sourceList = $("sourceList");
const fileList = $("fileList");
const dropzone = $("dropzone");
const fileInput = $("fileInput");
let docCount = 0;

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    docCount = data.indexed_chunks || 0;
    $("docCount").textContent = docCount;
  } catch {
    $("docCount").textContent = "—";
  }
}

refreshHealth();

/* ---------- Settings ---------- */
$("topK").addEventListener("input", (e) => ($("topKOut").textContent = e.target.value));

/* ---------- Chat helpers ---------- */
function addMessage(text, type) {
  const empty = $("chatEmpty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "msg " + type;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function addTyping() {
  const div = addMessage("", "bot");
  div.setAttribute("aria-label", "Generating answer");
  div.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  return div;
}

function renderSources(sources) {
  sourceList.innerHTML = "";
  if (!sources || !sources.length) {
    sourceList.innerHTML = '<li class="muted-note">No passages were retrieved for this answer.</li>';
    return;
  }
  sources.forEach((s) => {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "src-name";
    const name = document.createElement("span");
    name.textContent = s.source || "Document";
    head.appendChild(name);
    if (typeof s.distance === "number") {
      const sc = document.createElement("span");
      sc.textContent = "Distance " + s.distance.toFixed(3);
      head.appendChild(sc);
    }
    const p = document.createElement("p");
    p.textContent = s.text || "";
    li.append(head, p);
    sourceList.appendChild(li);
  });
}

/* ---------- API calls (real or demo) ---------- */
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function askBackend(question) {
  const res = await fetch(CONFIG.CHAT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      model: $("model").value,
      top_k: Number($("topK").value),
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Server returned " + res.status);
  return data;
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(CONFIG.UPLOAD_URL, { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Upload failed (" + res.status + ")");
  return data;
}

/* ---------- Send a question ---------- */
async function send(question) {
  addMessage(question, "user");
  sendBtn.disabled = true;
  const typing = addTyping();
  try {
    const data = await askBackend(question);
    typing.textContent = data.answer || "No answer was returned.";
    if (data.sources && data.sources.length) {
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = "Based on " + data.sources.length + " passage" + (data.sources.length > 1 ? "s" : "");
      typing.appendChild(meta);
    }
    renderSources(data.sources);
  } catch (err) {
    typing.textContent = "Something went wrong: " + err.message;
    typing.classList.add("error");
  } finally {
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
    questionEl.focus();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = questionEl.value.trim();
  if (!q) return;
  questionEl.value = "";
  autoGrow();
  send(q);
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

function autoGrow() {
  questionEl.style.height = "auto";
  questionEl.style.height = Math.min(questionEl.scrollHeight, 140) + "px";
}
questionEl.addEventListener("input", autoGrow);

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => { questionEl.value = chip.textContent; composer.requestSubmit(); })
);

$("clearChat").addEventListener("click", () => {
  messagesEl.innerHTML = '<div class="empty" id="chatEmpty"><p>Upload a document, then ask a question about it.</p></div>';
  sourceList.innerHTML = '<li class="muted-note">Retrieved passages will appear here.</li>';
});

/* ---------- Uploads ---------- */
async function handleFiles(files) {
  for (const file of files) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = file.name;
    const state = document.createElement("span");
    state.className = "state";
    state.textContent = "Indexing…";
    li.append(name, state);
    fileList.appendChild(li);
    try {
      const result = await uploadFile(file);
      state.textContent = "Indexed";
      state.classList.add("ok");
      if (typeof result.collection_count === "number") {
        docCount = result.collection_count;
        $("docCount").textContent = docCount;
      } else {
        await refreshHealth();
      }
    } catch (err) {
      state.textContent = "Failed";
      state.classList.add("fail");
      state.title = err.message;
    }
  }
}

fileInput.addEventListener("change", () => { handleFiles(fileInput.files); fileInput.value = ""; });

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));