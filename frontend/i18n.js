(function () {
  const KEY = "exitnav_lang";
  const EN = {
    "Domů": "Home",
    "Kamera": "Camera",
    "Editor": "Editor",
    "Trénink": "Training",
    "Admin": "Admin",
    "Administrace": "Administration",
    "Spustit kameru": "Start camera",
    "Spustit snímání": "Start scanning",
    "Zastavit snímání": "Stop scanning",
    "Snímky": "Frames",
    "Místnost": "Room",
    "Jistota": "Confidence",
    "Model nenalezen": "Model missing",
    "Model není natrénován": "Model is not trained",
    "Stav modelu": "Model status",
    "Natrénované třídy": "Training classes",
    "Spustit trénink": "Start training",
    "Log tréninku": "Training log",
    "Počet epoch": "Epochs",
    "Batch size": "Batch size",
    "Learning rate": "Learning rate",
    "FPS extrakce": "Extraction FPS",
    "Nahrát video pro třídu": "Upload class video",
    "Nahrát a extrahovat": "Upload and extract",
    "Vyberte existující": "Choose existing",
    "Nebo zadejte nové ID": "Or enter a new ID",
    "Label místnosti": "Room label",
    "Admin přístup": "Admin access",
    "Přihlásit se": "Sign in",
    "Odhlásit": "Sign out",
    "Nastavení": "Settings",
    "Editor grafu": "Graph editor",
    "Změna hesla": "Change password",
    "Aktuální heslo": "Current password",
    "Nové heslo": "New password",
    "Potvrdit nové heslo": "Confirm password",
    "Žádné třídy": "No classes",
    "Backend nedostupný": "Backend unavailable",
    "Probíhá trénink": "Training in progress",
    "Spouštím": "Starting",
    "Dokončuji": "Finishing",
    "Skenování": "Scanning",
    "Čekám na výsledek": "Waiting for result",
    "Odesílám snímek": "Sending frame",
    "Vyhodnocuji snímek": "Analyzing frame",
    "Nejbližší nouzový exit": "Nearest emergency exit",
    "vzdálenost": "distance",
    "Uživatel vidí aktivitu systému": "System activity is visible",
    "Čekám na spuštění": "Waiting to start",
    "Systém po spuštění průběžně odesílá snímky a vyhodnocuje místnost.": "After start, the system continuously sends frames and identifies the room.",
    "Spouštím kameru": "Starting camera",
    "Připravuji živé snímání a spojení se serverem.": "Preparing live capture and server connection.",
    "Připojuji server": "Connecting server",
    "Čekám na WebSocket pro živou inferenci.": "Waiting for the live inference WebSocket.",
    "Spojení je aktivní. Odesílám snímky v intervalu.": "Connection is active. Sending frames on an interval.",
    "Spojení spadlo, zkouším obnovit.": "Connection dropped, trying to reconnect.",
    "Nelze navázat živou inferenci.": "Live inference cannot be started.",
    "Přejděte do administrace a spusťte trénink modelu.": "Open administration and train the model.",
    "Snímek": "Frame",
    "míří do modelu.": "is being sent to the model.",
    "Čekám na predikci místnosti a trasu k exitu.": "Waiting for room prediction and exit route.",
    "Nejistý výsledek": "Uncertain result",
    "Model nemá dostatečný rozdíl mezi nejlepšími třídami.": "The model does not have enough separation between top classes.",
    "Výsledek nalezen": "Result found",
    "Místnost rozpoznána, počítám navigaci.": "Room recognized, calculating navigation.",
    "Pro trénink se nejdřív přihlaste v administraci.": "Sign in to administration before training.",
    "Klikněte na třídu pro zobrazení framů, ✕ smaže celou třídu.": "Click a class to view its frames, ✕ deletes the whole class.",
    "Framy": "Frames",
    "Žádné framy.": "No frames.",
    "Pro úpravu tříd se nejdřív přihlaste v administraci.": "Sign in to administration before editing classes.",
    "Pro prohlížení framů se nejdřív přihlaste v administraci.": "Sign in to administration before viewing frames.",
  };

  const CS = Object.fromEntries(Object.entries(EN).map(([cs, en]) => [en, cs]));

  function getLang() {
    return localStorage.getItem(KEY) === "en" ? "en" : "cs";
  }

  function setLang(lang) {
    localStorage.setItem(KEY, lang === "en" ? "en" : "cs");
    document.documentElement.lang = getLang();
    updateButton();
    translateDocument();
  }

  function dictionary() {
    return getLang() === "en" ? EN : CS;
  }

  function translateText(text) {
    let out = text;
    for (const [from, to] of Object.entries(dictionary())) {
      out = out.replaceAll(from, to);
    }
    return out;
  }

  function translateDocument() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      const next = translateText(node.nodeValue || "");
      if (next !== node.nodeValue) node.nodeValue = next;
    });
  }

  function updateButton() {
    const btn = document.querySelector(".lang-toggle");
    if (!btn) return;
    btn.textContent = getLang() === "en" ? "EN" : "CS";
    btn.title = getLang() === "en" ? "Přepnout do češtiny" : "Switch to English";
  }

  function mountToggle() {
    const host =
      document.querySelector(".header-actions") ||
      document.querySelector(".header-right") ||
      document.querySelector(".nav-links") ||
      document.querySelector("#topbar") ||
      document.querySelector("header") ||
      document.querySelector("#header");
    if (!host || document.querySelector(".lang-toggle")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "lang-toggle";
    btn.addEventListener("click", () => setLang(getLang() === "en" ? "cs" : "en"));
    host.appendChild(btn);
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.documentElement.lang = getLang();
    mountToggle();
    updateButton();
    translateDocument();
  });

  window.ExitNavI18n = {
    lang: getLang,
    setLang,
    t(text) {
      return translateText(text);
    },
  };
})();
