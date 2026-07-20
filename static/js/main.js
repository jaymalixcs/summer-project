/**
 * main.js — Customer Segmentation Dashboard
 * Interactive behaviour: Plotly charts, counters, AOS, dark mode, navbar
 */

// ─── Run when DOM is ready ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initAOS();
  initNavbarScroll();
  initCounters();
  initPlotlyCharts();
  initRangeSliders();
  initDeleteButtons();
  initProgressBars();
  initScrollspy();
});


// ════════════════════════════════════════════════════════════════════════════
//  AOS — Animate On Scroll (lightweight custom implementation)
// ════════════════════════════════════════════════════════════════════════════
function initAOS() {
  const elements = document.querySelectorAll("[data-aos]");
  if (!elements.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const delay = entry.target.dataset.aosDelay || 0;
        setTimeout(() => {
          entry.target.classList.add("aos-animate");
        }, parseInt(delay));
        observer.unobserve(entry.target);   // animate once
      }
    });
  }, { threshold: 0.1, rootMargin: "0px 0px -40px 0px" });

  elements.forEach((el) => observer.observe(el));
}


// ════════════════════════════════════════════════════════════════════════════
//  NAVBAR — solid background on scroll
// ════════════════════════════════════════════════════════════════════════════
function initNavbarScroll() {
  const navbar = document.querySelector(".navbar-custom");
  if (!navbar) return;

  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      navbar.style.background = "rgba(5, 7, 20, 0.97)";
    } else {
      navbar.style.background = "rgba(5, 7, 20, 0.85)";
    }
  }, { passive: true });
}


// ════════════════════════════════════════════════════════════════════════════
//  COUNTER ANIMATION
// ════════════════════════════════════════════════════════════════════════════
function initCounters() {
  const counters = document.querySelectorAll("[data-count]");
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el      = entry.target;
      const target  = parseFloat(el.dataset.count);
      const isFloat = el.dataset.count.includes(".");
      const duration = 1400;
      const start    = performance.now();

      const update = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const ease     = 1 - Math.pow(1 - progress, 3);   // ease-out cubic
        const current  = target * ease;

        el.textContent = isFloat
          ? current.toFixed(3)
          : Math.round(current).toLocaleString();

        if (progress < 1) requestAnimationFrame(update);
      };

      requestAnimationFrame(update);
      observer.unobserve(el);
    });
  }, { threshold: 0.3 });

  counters.forEach((c) => observer.observe(c));
}


// ════════════════════════════════════════════════════════════════════════════
//  PLOTLY CHARTS — render all charts from embedded JSON
// ════════════════════════════════════════════════════════════════════════════
function initPlotlyCharts() {
  const chartEls = document.querySelectorAll("[data-chart-json]");
  chartEls.forEach((el) => {
    try {
      const figData = JSON.parse(el.dataset.chartJson);
      Plotly.newPlot(
        el,
        figData.data,
        {
          ...figData.layout
        },
        {
          displayModeBar: true,
          modeBarButtonsToRemove: ["sendDataToCloud", "editInChartStudio"],
          displaylogo: false,
          responsive: true,
        }
      );
    } catch (e) {
      console.error("Chart render error:", e, el.id);
    }
  });

  // Make charts responsive on resize
  window.addEventListener("resize", () => {
    chartEls.forEach((el) => {
      if (el._fullLayout) Plotly.Plots.resize(el);
    });
  });
}


// ════════════════════════════════════════════════════════════════════════════
//  RANGE SLIDER — show live value
// ════════════════════════════════════════════════════════════════════════════
function initRangeSliders() {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const output = document.getElementById(slider.id + "_val");
    if (!output) return;
    output.textContent = slider.value;
    slider.addEventListener("input", () => {
      output.textContent = slider.value;
    });
  });
}


// ════════════════════════════════════════════════════════════════════════════
//  DELETE BUTTONS (customer table)
// ════════════════════════════════════════════════════════════════════════════
function initDeleteButtons() {
  document.querySelectorAll(".btn-delete-customer").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cid = btn.dataset.id;
      if (!confirm("Delete this customer record?")) return;

      try {
        const resp = await fetch(`/delete-customer/${cid}`, { method: "POST" });
        if (resp.ok) {
          const row = btn.closest("tr");
          row.style.transition = "opacity 0.4s, transform 0.4s";
          row.style.opacity  = "0";
          row.style.transform = "translateX(20px)";
          setTimeout(() => row.remove(), 420);
        }
      } catch (e) {
        alert("Error deleting customer.");
      }
    });
  });
}


// ════════════════════════════════════════════════════════════════════════════
//  PROGRESS BARS — animate width on scroll
// ════════════════════════════════════════════════════════════════════════════
function initProgressBars() {
  const bars = document.querySelectorAll(".progress-fill[data-width]");
  if (!bars.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const bar = entry.target;
      bar.style.width = bar.dataset.width + "%";
      observer.unobserve(bar);
    });
  }, { threshold: 0.3 });

  // Start at 0
  bars.forEach((b) => { b.style.width = "0%"; observer.observe(b); });
}


// ════════════════════════════════════════════════════════════════════════════
//  SCROLLSPY — highlight current section in page-nav
// ════════════════════════════════════════════════════════════════════════════
function initScrollspy() {
  const navLinks = document.querySelectorAll(".page-nav-link[href^='#']");
  if (!navLinks.length) return;

  const sectionIds = Array.from(navLinks).map((l) => l.getAttribute("href").slice(1));
  const sections   = sectionIds.map((id) => document.getElementById(id)).filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        navLinks.forEach((l) => l.classList.remove("active"));
        const active = document.querySelector(`.page-nav-link[href="#${id}"]`);
        if (active) active.classList.add("active");
      });
    },
    { threshold: 0.45 }
  );

  sections.forEach((s) => observer.observe(s));
}


// ════════════════════════════════════════════════════════════════════════════
//  TOAST notification
// ════════════════════════════════════════════════════════════════════════════
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  const colors = { info: "#6C63FF", success: "#00D4AA", error: "#FF6B6B", warning: "#FFD166" };
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: #12143A; border: 1px solid ${colors[type]};
    border-left: 4px solid ${colors[type]};
    color: #E2E8F0; border-radius: 10px;
    padding: 14px 20px; font-size: 0.88rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    max-width: 320px; animation: slideUp 0.35s ease;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.4s";
    setTimeout(() => toast.remove(), 400);
  }, 3200);
}
