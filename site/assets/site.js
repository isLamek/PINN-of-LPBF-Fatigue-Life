/* The static edition's behaviour. Every number on the pages was written in by the
   build from the result files; this script only draws, animates and, on the
   validation page, evaluates the three life models the reader drives with sliders. */
(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var CONFIG = { displaylogo: false, responsive: true };

  function json(id) { return JSON.parse(document.getElementById(id).textContent); }

  // ---------------------------------------------------------------- charts
  // A chart with frames plays by itself, loops, and stays where it is once Pause is
  // pressed; a reader who asks for reduced motion sees the first frame.
  function loop(gd, fig) {
    var m = (fig.layout && fig.layout.meta) || {};
    var opts = { frame: { duration: m.frame_ms || 500, redraw: true },
                 transition: { duration: m.trans_ms || 0 }, mode: "immediate" };
    gd._paused = false;
    function run(fromStart) {
      if (reduce || gd._paused) return;
      Plotly.animate(gd, null, Object.assign({ fromcurrent: !fromStart }, opts));
    }
    gd.on("plotly_animated", function () {
      clearTimeout(gd._timer);
      if (!gd._paused) gd._timer = setTimeout(function () { run(true); }, 1400);
    });
    gd.on("plotly_buttonclicked", function (e) {
      gd._paused = !!(e.button && e.button.label === "Pause");
      if (gd._paused) clearTimeout(gd._timer);
    });
    run(true);
  }

  function render(gd, fig) {
    clearTimeout(gd._timer);
    if (gd.data) Plotly.purge(gd);
    return Plotly.newPlot(gd, fig.data, fig.layout, CONFIG).then(function () {
      if (fig.frames && fig.frames.length) {
        return Plotly.addFrames(gd, fig.frames).then(function () { loop(gd, fig); });
      }
    });
  }
  window.LPBF = { render: render, json: json };

  // charts are drawn when they come near the screen, so a long page opens at once
  var pending = new Map();
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      var f = pending.get(e.target);
      pending.delete(e.target);
      if (f) f();
    });
  }, { rootMargin: "300px" });
  function whenVisible(el, f) { pending.set(el, f); io.observe(el); }
  // a second trigger on scroll: intersection events are not delivered in every
  // embedding or background tab, and a chart must never be left blank
  function sweep() {
    pending.forEach(function (f, el) {
      var r = el.getBoundingClientRect();
      if (r.top < window.innerHeight + 300 && r.bottom > -300) {
        pending.delete(el);
        io.unobserve(el);
        f();
      }
    });
  }
  // a timer, not requestAnimationFrame, which also pauses in hidden tabs
  var ticking = false;
  window.addEventListener("scroll", function () {
    if (ticking) return;
    ticking = true;
    setTimeout(function () { ticking = false; sweep(); }, 80);
  }, { passive: true });
  window.addEventListener("resize", sweep);
  window.addEventListener("load", sweep);

  document.querySelectorAll(".chart[data-fig]").forEach(function (gd) {
    whenVisible(gd, function () { render(gd, json(gd.dataset.fig)); });
  });

  // ---------------------------------------------------------------- choosers
  // <select data-chart="divId">: each option's value is the id of a figure's JSON.
  // Elements with data-show-for="selectId" data-key="value" follow the selection.
  document.querySelectorAll("select.chooser").forEach(function (sel) {
    function apply() {
      var target = sel.dataset.chart && document.getElementById(sel.dataset.chart);
      if (target) {
        var fid = sel.value;
        target.dataset.fig = fid;
        if (pending.has(target)) pending.set(target, function () { render(target, json(fid)); });
        else render(target, json(fid));
      }
      document.querySelectorAll('[data-show-for="' + sel.id + '"]').forEach(function (el) {
        el.hidden = el.dataset.key !== sel.value;
      });
      if (sel.dataset.player) players[sel.dataset.player].load(sel.value);
      if (sel.dataset.growth) growth(sel.dataset.growth, sel.value);
    }
    sel.addEventListener("change", apply);
    apply.initial = true;
    document.querySelectorAll('[data-show-for="' + sel.id + '"]').forEach(function (el) {
      el.hidden = el.dataset.key !== sel.value;
    });
  });

  // ---------------------------------------------------------------- image players
  // A sequence of image sets (one or more images per step) with Play/Pause and a
  // slider. Used for the CT stacks; with data-autoplay="0" it is a plain viewer.
  var players = {};
  document.querySelectorAll(".player").forEach(function (el) {
    var imgs = el.querySelectorAll("img[data-slot]");
    var range = el.querySelector("input[type=range]");
    var out = el.querySelector("output");
    var btn = el.querySelector("button[data-act=toggle]");
    var ms = +el.dataset.ms || 250;
    var auto = el.dataset.autoplay !== "0" && !reduce;
    var frames = [], k = 0, timer = null, playing = false;

    function show(i) {
      k = (i + frames.length) % frames.length;
      var f = frames[k];
      for (var s = 0; s < imgs.length; s++) imgs[s].src = f.src[s];
      range.value = k;
      out.textContent = f.label;
      var cap = el.querySelector("[data-frame-caption]");
      if (cap) cap.textContent = f.caption || "";
    }
    function stop() { clearInterval(timer); timer = null; playing = false;
                      if (btn) btn.textContent = "Play"; }
    function play() { if (timer) return; playing = true; if (btn) btn.textContent = "Pause";
                      timer = setInterval(function () { show(k + 1); }, ms); }
    function load(id) {
      stop();
      frames = json(id);
      range.max = frames.length - 1;
      frames.forEach(function (f) { f.src.forEach(function (s) { new Image().src = s; }); });
      show(+el.dataset.start || 0);
      if (auto) play();
    }
    range.addEventListener("input", function () { stop(); show(+range.value); });
    if (btn) btn.addEventListener("click", function () { playing ? stop() : play(); });
    players[el.id] = { load: load };
    whenVisible(el, function () { load(el.dataset.frames); });
  });

  // ---------------------------------------------------------------- crack growth
  // One crack from its governing defect to collapse, drawn from the curve arrays the
  // build wrote for each held-out cylinder (path A's integral, scaled to its life).
  function growth(divId, key) {
    var gd = document.getElementById(divId);
    var D = json(gd.dataset.curves)[key];
    var navy = "#1F4E79", red = "#8B1E1E", grey = "#8A949C", green = "#2E6B3A";
    function traces(j) {
      return [
        { x: D.N.slice(0, j + 1), y: D.a.slice(0, j + 1), mode: "lines",
          line: { color: navy, width: 2.5 }, xaxis: "x", yaxis: "y", hoverinfo: "skip" },
        { x: [D.N[j]], y: [D.a[j]], mode: "markers", xaxis: "x", yaxis: "y",
          marker: { size: 11, color: navy, line: { color: "white", width: 2 } },
          hoverinfo: "skip" },
        { x: D.a.slice(0, j + 1), y: D.dK.slice(0, j + 1), mode: "lines",
          line: { color: red, width: 2.5 }, xaxis: "x2", yaxis: "y2", hoverinfo: "skip" }];
    }
    var lo = Math.log10(D.a0 * 0.8), hi = Math.log10(D.ac * 1.3);
    var fig = {
      data: traces(D.idx[0]),
      frames: D.idx.map(function (j, k) {
        return { name: String(k), data: traces(j), traces: [0, 1, 2] };
      }),
      layout: {
        template: gd.dataset.template ? json(gd.dataset.template) : undefined,
        height: 430, showlegend: false, margin: { l: 60, r: 20, t: 40, b: 110 },
        font: { family: '"Source Sans 3", "Segoe UI", Arial, sans-serif', size: 13, color: "#1E2A36" },
        xaxis: { domain: [0, 0.58], range: [0, Math.max(D.Nf, D.meas) * 1.08],
                 title: { text: "cycles" }, gridcolor: "#E6E9EC" },
        yaxis: { type: "log", range: [lo, hi], title: { text: "crack depth a (mm)" },
                 gridcolor: "#E6E9EC" },
        xaxis2: { domain: [0.68, 1], type: "log", range: [lo, hi], anchor: "y2",
                  title: { text: "crack depth a (mm)" }, gridcolor: "#E6E9EC" },
        yaxis2: { anchor: "x2", range: [0, Math.max.apply(null, D.dK) * 1.1],
                  title: { text: "ΔK (MPa√m)" }, gridcolor: "#E6E9EC" },
        annotations: [
          { text: "crack depth against cycles", x: 0.29, xref: "paper", y: 1.07,
            yref: "paper", showarrow: false },
          { text: "driving force against crack depth", x: 0.84, xref: "paper", y: 1.07,
            yref: "paper", showarrow: false },
          { text: "collapse depth", x: Math.max(D.Nf, D.meas), xref: "x", y: Math.log10(D.ac),
            yref: "y", showarrow: false, xanchor: "right", yanchor: "top",
            font: { color: grey, size: 11 } },
          { text: "measured life", x: D.meas, xref: "x", y: hi, yref: "y", showarrow: false,
            xanchor: "right", yanchor: "top", font: { color: green, size: 11 } }],
        shapes: [
          { type: "line", xref: "x", yref: "y", x0: 0, x1: Math.max(D.Nf, D.meas) * 1.08,
            y0: D.ac, y1: D.ac, line: { color: grey, dash: "dot" } },
          { type: "line", xref: "x", yref: "y", x0: D.meas, x1: D.meas,
            y0: Math.pow(10, lo), y1: Math.pow(10, hi), line: { color: green, dash: "dash" } }],
        updatemenus: [{ type: "buttons", direction: "left", x: 0, y: -0.16, xanchor: "left",
          yanchor: "top", showactive: false, bgcolor: "#FFFFFF", bordercolor: "#D3D8DD",
          buttons: [
            { label: "Play", method: "animate", args: [null, { frame: { duration: 120,
              redraw: true }, transition: { duration: 0 }, fromcurrent: true }] },
            { label: "Pause", method: "animate", args: [[null], { frame: { duration: 0,
              redraw: false }, mode: "immediate" }] }] }],
        meta: { frame_ms: 120, trans_ms: 0 }
      }
    };
    render(gd, fig);
  }
  document.querySelectorAll(".chart[data-curves]").forEach(function (gd) {
    var sel = document.querySelector('select[data-growth="' + gd.id + '"]');
    whenVisible(gd, function () { growth(gd.id, sel.value); });
  });

  // ---------------------------------------------------------------- explorer
  var ex = document.getElementById("explorer");
  if (ex) {
    var P = json(ex.dataset.params);
    var dsIn = document.getElementById("ex-ds"), saIn = document.getElementById("ex-sa");

    function collapse(smax) {
      var r = P.gauge_r_mm, frac = 1 - smax / P.sigma_uts;
      if (frac <= 0) return 0.05;
      return Math.min(Math.max(Math.sqrt(2 * frac) * r, 0.05), r);
    }
    // path A: equation (3.13) with the threshold removed, Simpson's rule on a
    // logarithmic grid, as src/life_integral.life_numeric
    function pathA(sa_um, ds, af) {
      var a0 = sa_um * 1e-3, n = 2001;
      if (af <= a0) return 0;
      var u0 = Math.log(a0), h = (Math.log(af) - u0) / (n - 1), s = 0;
      for (var i = 0; i < n; i++) {
        var a = Math.exp(u0 + i * h);
        var dK = P.Y * ds * Math.sqrt(Math.PI * a * 1e-3);
        var g = a / (P.C * Math.pow(dK, P.m));
        s += g * (i === 0 || i === n - 1 ? 1 : (i % 2 ? 4 : 2));
      }
      return s * h / 3;
    }
    function b3(sa, ds) {
      var c = P.b3;
      return Math.pow(10, c.b0 + c.b_ds * Math.log(ds) + c.b_sa * Math.log(sa));
    }
    function tanh(v) { return v.map(Math.tanh); }
    function dense(W, b, x) {
      return W.map(function (row, i) {
        return row.reduce(function (s, w, j) { return s + w * x[j]; }, b[i]);
      });
    }
    function pinn(sa, ds) {
      var N = P.pinn, x = [Math.log(ds), Math.log(sa), N.flat_med];
      x = x.map(function (v, i) { return (v - N.mean[i]) / N.std[i]; });
      var h = tanh(dense(N.W1, N.b1, x));
      h = tanh(dense(N.W2, N.b2, h));
      return Math.pow(10, dense(N.W3, N.b3, h)[0]);
    }
    window.LPBF.models = { pathA: pathA, b3: b3, pinn: pinn, collapse: collapse };

    function fmt(v) { return isFinite(v) ? Math.round(v).toLocaleString("en-GB") : "run-out"; }
    var gd = document.getElementById("ex-chart");
    function update() {
      var ds = +dsIn.value, sa = +saIn.value, smax = ds / (1 - P.R), af = collapse(smax);
      document.getElementById("ex-ds-out").textContent = ds + " MPa";
      document.getElementById("ex-sa-out").textContent = sa + " µm";
      document.getElementById("ex-a").textContent = fmt(pathA(sa, ds, af));
      document.getElementById("ex-b3").textContent = fmt(b3(sa, ds));
      document.getElementById("ex-pinn").textContent = fmt(pinn(sa, ds));
      document.getElementById("ex-cap").textContent = "σmax = " + Math.round(smax) +
        " MPa; collapse depth " + af.toFixed(2) + " mm (bar radius " +
        P.gauge_r_mm.toFixed(2) + " mm).";
      var grid = [], A = [], B = [], C = [];
      for (var v = 150; v <= 1200; v += 1050 / 59) grid.push(v);
      grid.forEach(function (g) { A.push(pathA(g, ds, af)); B.push(b3(g, ds)); C.push(pinn(g, ds)); });
      var near = P.measured.filter(function (r) { return Math.abs(r.ds - ds) < 60; });
      var data = [
        { x: grid, y: A, mode: "lines", name: "path A", line: { color: "#1F4E79", width: 2.5 } },
        { x: grid, y: B, mode: "lines", name: "B3", line: { color: "#8A949C", dash: "dash" } },
        { x: grid, y: C, mode: "lines", name: "C1", line: { color: "#8B1E1E", dash: "dot" } },
        { x: near.map(function (r) { return r.sa; }), y: near.map(function (r) { return r.Nf; }),
          mode: "markers", name: "measured, Δσ within 60 MPa",
          marker: { size: 9, color: "#1E2A36", symbol: "x" } }];
      var layout = { height: 440, font: { family: '"Source Sans 3", "Segoe UI", Arial, sans-serif',
          size: 13, color: "#1E2A36" }, margin: { l: 70, r: 20, t: 30, b: 50 },
        xaxis: { title: { text: "governing defect √area (µm)" }, gridcolor: "#E6E9EC" },
        yaxis: { type: "log", title: { text: "cycles to failure" }, gridcolor: "#E6E9EC" },
        legend: { orientation: "h", y: 1.12 },
        shapes: [{ type: "line", xref: "x", yref: "paper", x0: sa, x1: sa, y0: 0, y1: 1,
                   line: { color: "#D3D8DD" } }] };
      Plotly.react(gd, data, layout, CONFIG);
    }
    dsIn.addEventListener("input", update);
    saIn.addEventListener("input", update);
    whenVisible(ex, update);
  }

  // ---------------------------------------------------------------- contents rail
  // marks the section being read
  var links = document.querySelectorAll(".toc a");
  if (links.length) {
    var heads = Array.prototype.map.call(links, function (a) {
      return document.getElementById(a.getAttribute("href").slice(1));
    });
    var mark = function () {
      var y = window.scrollY + 120, cur = 0;
      heads.forEach(function (h, i) {
        if (h && h.getBoundingClientRect().top + window.scrollY <= y) cur = i;
      });
      links.forEach(function (a, i) { a.classList.toggle("on", i === cur); });
    };
    window.addEventListener("scroll", mark, { passive: true });
    mark();
  }

  // ---------------------------------------------------------------- equations
  if (window.renderMathInElement) {
    renderMathInElement(document.body, { delimiters: [
      { left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
      throwOnError: false });
  }
})();
