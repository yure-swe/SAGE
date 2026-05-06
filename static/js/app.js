/* =============================================================================
   SAGE — frontend behaviors
   - Theme toggle (persisted)
   - Sidebar collapse (desktop) and off-canvas drawer (mobile)
   - Form section collapse with field-completion counter
   - AJAX submit with loading state, preserves the existing fetch contract
   - Confidence ring + animated probability bars on result render
   - Lightbox for SHAP/model images
   - Scrollspy for the guide page TOC
   ============================================================================= */
(function () {
  "use strict";

  // ── Theme ────────────────────────────────────────────────────────────────
  const THEME_KEY = "sage-theme";
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
  }
  const stored = localStorage.getItem(THEME_KEY);
  applyTheme(stored || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

  function bindThemeToggle() {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem(THEME_KEY, next);
      });
    });
  }

  // ── Sidebar ──────────────────────────────────────────────────────────────
  const COLLAPSE_KEY = "sage-sidebar-collapsed";
  function bindSidebar() {
    const layout = document.querySelector(".layout");
    const sidebar = document.querySelector(".sidebar");
    const scrim = document.querySelector(".scrim");
    if (!layout || !sidebar) return;

    if (localStorage.getItem(COLLAPSE_KEY) === "1") {
      layout.classList.add("sidebar-collapsed");
    }

    document.querySelectorAll("[data-sidebar-collapse]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const collapsed = layout.classList.toggle("sidebar-collapsed");
        localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
      });
    });

    function openDrawer() {
      sidebar.classList.add("open");
      scrim && scrim.classList.add("open");
    }
    function closeDrawer() {
      sidebar.classList.remove("open");
      scrim && scrim.classList.remove("open");
    }
    document.querySelectorAll("[data-sidebar-open]").forEach((btn) => btn.addEventListener("click", openDrawer));
    scrim && scrim.addEventListener("click", closeDrawer);
    sidebar.querySelectorAll("a").forEach((a) => a.addEventListener("click", closeDrawer));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  }

  // ── Form sections ────────────────────────────────────────────────────────
  function bindFormSections(root) {
    const scope = root || document;
    scope.querySelectorAll(".form-section").forEach((section, idx) => {
      const heading = section.querySelector("h3");
      if (!heading) return;
      // Add chevron if missing
      if (!heading.querySelector(".chev")) {
        const chev = document.createElement("span");
        chev.className = "chev";
        chev.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';
        heading.appendChild(chev);
      }
      // Collapse all but first by default
      if (idx > 0 && !section.dataset.kept) section.classList.add("collapsed");
      heading.addEventListener("click", () => {
        section.classList.toggle("collapsed");
        section.dataset.kept = "1";
      });
    });
  }

  // ── Result animations ────────────────────────────────────────────────────
  function animateResults(root) {
    const scope = root || document;

    // Confidence ring: read either a [data-confidence] hook or fall back to
    // parsing the legacy ".confidence-meter span" text the original template
    // produces, so this works whether or not the template was updated.
    const ring = scope.querySelector(".confidence-ring");
    if (ring) {
      let pct = parseFloat(ring.dataset.confidence);
      if (isNaN(pct)) {
        const meter = scope.querySelector(".confidence-meter span");
        if (meter) {
          const m = meter.textContent.match(/([\d.]+)/);
          if (m) pct = parseFloat(m[1]);
        }
      }
      if (!isNaN(pct)) {
        const fill = ring.querySelector(".fill");
        const label = ring.querySelector(".label");
        const circ = 226.2;
        const offset = circ - (Math.max(0, Math.min(100, pct)) / 100) * circ;
        requestAnimationFrame(() => {
          if (fill) fill.style.strokeDashoffset = offset;
          if (label) label.textContent = pct.toFixed(0) + "%";
        });
      }
    }

    // Stagger probability bars
    scope.querySelectorAll(".probability-bars").forEach((group) => {
      group.querySelectorAll(".prob-bar").forEach((bar, i) => {
        bar.style.animationDelay = (i * 60) + "ms";
      });
    });

    // Highlight predicted class probability bar
    const predictedRange = (() => {
      const t = scope.querySelector(".prediction-card .tier-class");
      return t ? t.textContent.trim().toLowerCase() : null;
    })();
    if (predictedRange) {
      scope.querySelectorAll(".results-panel > .result-card .probability-bars:not(.small) .prob-item").forEach((row) => {
        const lab = row.querySelector(".prob-label");
        if (lab && lab.textContent.trim().toLowerCase() === predictedRange) {
          row.classList.add("is-predicted");
        }
      });
    }
  }

  // ── Form submit (AJAX) ───────────────────────────────────────────────────
  function bindForm() {
    const form = document.getElementById("prediction-form");
    if (!form) return;
    const submitBtn = form.querySelector('[type="submit"]');

    form.addEventListener("submit", function (e) {
      e.preventDefault();

      const priceInput = document.getElementById("price");
      if (priceInput) {
        const price = parseFloat(priceInput.value);
        if (price < 0 || price > 200) {
          showInlineErrors(["Price must be between 0 and 200 USD"], ["price"]);
          return;
        }
      }

      if (submitBtn) submitBtn.classList.add("is-loading");

      const formData = new FormData(form);
      fetch("/", { method: "POST", body: formData })
        .then((r) => r.text().then((html) => ({ ok: r.ok, status: r.status, html })))
        .then(({ ok, html }) => {
          const doc = new DOMParser().parseFromString(html, "text/html");
          const newForm = doc.querySelector(".form-panel");
          const newResults = doc.querySelector(".results-panel");
          const newAlertSlot = doc.querySelector("#validation-alert-slot");

          const formPanel = document.querySelector(".form-panel");
          if (newForm && formPanel) formPanel.innerHTML = newForm.innerHTML;

          // Swap validation alert slot so server-side errors are visible
          const slot = document.querySelector("#validation-alert-slot");
          if (slot && newAlertSlot) slot.innerHTML = newAlertSlot.innerHTML;

          let resultsPanel = document.querySelector(".results-panel");
          if (!resultsPanel) {
            resultsPanel = document.createElement("div");
            resultsPanel.className = "results-panel";
            document.querySelector(".dashboard-container").appendChild(resultsPanel);
          }
          if (newResults) resultsPanel.innerHTML = newResults.innerHTML;

          // Re-bind dynamic content
          bindFormSections(formPanel);
          bindForm();
          bindDetailedDescCoupling();
          bindNumericGuards();
          bindChips(formPanel);
          bindReleaseDateSync(formPanel);
          animateResults(resultsPanel);
          bindLightbox(document);

          if (!ok && slot) {
            slot.scrollIntoView({ behavior: "smooth", block: "start" });
          } else if (window.matchMedia("(max-width: 1024px)").matches) {
            resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        })
        .catch((err) => {
          console.error("Prediction error:", err);
          showInlineErrors(["Something went wrong. Please try again."], []);
        })
        .finally(() => {
          const btn = document.querySelector('#prediction-form [type="submit"]');
          if (btn) btn.classList.remove("is-loading");
        });
    });
  }

  // ── Lightbox ─────────────────────────────────────────────────────────────
  function ensureLightbox() {
    let lb = document.querySelector(".lightbox");
    if (lb) return lb;
    lb = document.createElement("div");
    lb.className = "lightbox";
    lb.innerHTML = '<button class="lightbox-close" aria-label="Close"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button><img alt="">';
    document.body.appendChild(lb);
    function close() { lb.classList.remove("open"); }
    lb.addEventListener("click", (e) => { if (e.target === lb) close(); });
    lb.querySelector(".lightbox-close").addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    return lb;
  }
  function bindLightbox(scope) {
    const root = scope || document;
    root.querySelectorAll(".shap-plot img").forEach((img) => {
      if (img.dataset.lbBound) return;
      img.dataset.lbBound = "1";
      img.addEventListener("click", () => {
        const lb = ensureLightbox();
        lb.querySelector("img").src = img.src;
        lb.classList.add("open");
      });
    });
  }

  // ── Scrollspy (Guide TOC) ────────────────────────────────────────────────
  function bindScrollspy() {
    const links = document.querySelectorAll(".toc a");
    if (!links.length) return;
    const map = new Map();
    links.forEach((a) => {
      const id = a.getAttribute("href").slice(1);
      const el = document.getElementById(id);
      if (el) map.set(el, a);
    });
    if (!map.size) return;
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        const a = map.get(en.target);
        if (!a) return;
        if (en.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          a.classList.add("active");
        }
      });
    }, { rootMargin: "-30% 0px -60% 0px", threshold: 0 });
    map.forEach((_, el) => obs.observe(el));
  }

  // ── Reliability helpers ──────────────────────────────────────────────────
  function showInlineErrors(messages, fieldNames) {
    const slot = document.querySelector("#validation-alert-slot");
    if (slot) {
      slot.innerHTML =
        '<div class="alert alert-danger" role="alert">' +
        '<strong>Please fix the following:</strong>' +
        '<ul style="margin:0.5rem 0 0 1rem">' +
        messages.map((m) => '<li>' + m + '</li>').join("") +
        '</ul></div>';
      slot.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    document.querySelectorAll(".field-error").forEach((el) => el.classList.remove("field-error"));
    (fieldNames || []).forEach((n) => {
      const f = document.getElementById(n);
      if (f) f.classList.add("field-error");
    });
  }

  function bindDetailedDescCoupling() {
    const about = document.getElementById("about_length");
    const detailed = document.getElementById("has_detailed_desc");
    if (!about || !detailed) return;
    const sync = () => {
      const v = parseInt(about.value, 10);
      const on = Number.isFinite(v) && v > 500;
      detailed.checked = on;
      detailed.disabled = true;
      detailed.title = "Auto-derived from Description Length (>500 chars)";
    };
    about.addEventListener("input", sync);
    about.addEventListener("change", sync);
    sync();
  }

  // ── Chip toggle (tag and language checklists) ────────────────────────────
  function bindChips(root) {
    const scope = root || document;
    scope.querySelectorAll("#tag-checklist label, #lang-checklist label").forEach((label) => {
      if (label.dataset.chipBound === "1") return;
      label.dataset.chipBound = "1";
      const cb = label.querySelector("input[type='checkbox']");
      if (!cb) return;
      if (cb.checked) label.classList.add("is-checked");
      cb.addEventListener("change", () => {
        label.classList.toggle("is-checked", cb.checked);
      });
    });
  }

  // ── Release date → release_month sync ───────────────────────────────────
  function bindReleaseDateSync(root) {
    const scope = root || document;
    const dateInput  = scope.querySelector("#release_date");
    const monthSelect = scope.querySelector("#release_month");
    if (!dateInput || !monthSelect) return;
    if (dateInput.dataset.dateSyncBound === "1") return;
    dateInput.dataset.dateSyncBound = "1";
    function syncMonth() {
      const val = dateInput.value;
      if (!val) return;
      const month = new Date(val + "T00:00:00").getMonth() + 1;
      monthSelect.value = String(month);
    }
    dateInput.addEventListener("change", syncMonth);
    if (dateInput.value) syncMonth();
  }

  function bindNumericGuards() {
    document.querySelectorAll('#prediction-form input[type="number"]').forEach((input) => {
      if (input.dataset.numericGuardBound === "1") return;
      input.dataset.numericGuardBound = "1";
      const step = parseFloat(input.getAttribute("step") || "1");
      const allowDecimal = Number.isFinite(step) && step > 0 && step < 1;
      const allowedKeys = new Set([
        "Backspace", "Delete", "Tab", "Escape", "Enter",
        "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End",
      ]);
      input.addEventListener("keydown", (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (allowedKeys.has(e.key)) return;
        if (/^[0-9]$/.test(e.key)) return;
        if (allowDecimal && e.key === "." && !input.value.includes(".")) return;
        e.preventDefault();
      });
      input.addEventListener("paste", (e) => {
        const txt = (e.clipboardData || window.clipboardData).getData("text");
        const re = allowDecimal ? /^\d*\.?\d*$/ : /^\d+$/;
        if (!re.test(txt)) e.preventDefault();
      });
      input.addEventListener("input", () => {
        const re = allowDecimal ? /[^0-9.]/g : /[^0-9]/g;
        const cleaned = input.value.replace(re, "");
        if (cleaned !== input.value) input.value = cleaned;
      });
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", function () {
    bindThemeToggle();
    bindSidebar();
    bindFormSections();
    bindForm();
    bindDetailedDescCoupling();
    bindNumericGuards();
    bindChips();
    bindReleaseDateSync();
    animateResults();
    bindLightbox();
    bindScrollspy();
  });
})();
