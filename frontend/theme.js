(function () {
  const KEY = "exitnav_theme";

  function preferredTheme() {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  function applyTheme(theme) {
    document.body.dataset.theme = theme;
    localStorage.setItem(KEY, theme);
    const btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.textContent = theme === "light" ? "Light" : "Dark";
      btn.setAttribute("aria-label", theme === "light" ? "Switch to dark mode" : "Switch to light mode");
      btn.title = btn.getAttribute("aria-label") || "";
    }
  }

  function mountToggle() {
    const host =
      document.querySelector(".header-actions") ||
      document.querySelector(".header-right") ||
      document.querySelector(".nav-links") ||
      document.querySelector("#topbar") ||
      document.querySelector("header") ||
      document.querySelector("#header");
    if (!host || document.querySelector(".theme-toggle")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    btn.addEventListener("click", () => {
      applyTheme(document.body.dataset.theme === "light" ? "dark" : "light");
    });
    host.appendChild(btn);
  }

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(preferredTheme());
    mountToggle();
  });
})();
