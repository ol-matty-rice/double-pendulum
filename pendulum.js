/**
 * Double Pendulum Simulation
 * Physics: Lagrangian equations of motion, integrated with Runge-Kutta 4
 * Canvas: HTML5 2D Canvas with fade-trail rendering
 */

(function () {
  "use strict";

  /* ── Canvas setup ──────────────────────────────────────── */
  const canvas = document.getElementById("pendulumCanvas");
  const ctx = canvas.getContext("2d");

  function resizeCanvas() {
    const wrap = canvas.parentElement;
    const w = wrap.clientWidth;
    const h = Math.max(500, Math.min(w, 680));
    canvas.width = w;
    canvas.height = h;
  }

  resizeCanvas();
  window.addEventListener("resize", () => { resizeCanvas(); resetSim(); });

  /* ── State ─────────────────────────────────────────────── */
  let state, params, trace, ghost, paused, animId;
  let panX = 0, panY = 0; // view pan offset

  function defaultParams() {
    return {
      g:       parseFloat(document.getElementById("gravity").value),
      damping: parseFloat(document.getElementById("damping").value),
      speed:   parseFloat(document.getElementById("speed").value),
      l1:      parseFloat(document.getElementById("l1").value),
      l2:      parseFloat(document.getElementById("l2").value),
      m1:      parseFloat(document.getElementById("m1").value),
      m2:      parseFloat(document.getElementById("m2").value),
      a1init:  parseFloat(document.getElementById("a1").value) * Math.PI / 180,
      a2init:  parseFloat(document.getElementById("a2").value) * Math.PI / 180,
    };
  }

  function resetSim() {
    cancelAnimationFrame(animId);
    params = defaultParams();

    // Scale so full swing (l1+l2 radius) fits inside canvas with 10% margin
    const maxRadius = Math.min(canvas.width, canvas.height) * 0.44;
    const rawSum = params.l1 + params.l2;
    const scale  = rawSum > maxRadius ? maxRadius / rawSum : 1;
    params.l1s = params.l1 * scale;
    params.l2s = params.l2 * scale;

    state = {
      a1:  params.a1init,
      a2:  params.a2init,
      v1:  0,
      v2:  0,
    };

    trace = [];
    ghost = [];
    paused = false;
    panX = 0;
    panY = 0;
    lastTime = null;
    accumulator = 0;
    document.getElementById("pauseBtn").textContent = "Pause";
    animId = requestAnimationFrame(loop);
  }

  /* ── Physics: Lagrangian double-pendulum EOM ───────────── */
  // Returns [dv1/dt, dv2/dt] given current angles and velocities
  function derivatives(a1, a2, v1, v2, p) {
    const { g, m1, m2, l1s: l1, l2s: l2, damping } = p;
    const dA = a1 - a2;
    const cosDA = Math.cos(dA);
    const sinDA = Math.sin(dA);
    const denom1 = (2 * m1 + m2 - m2 * Math.cos(2 * dA));
    const denom2 = denom1; // same denominator structure

    const num1 =
      -g * (2 * m1 + m2) * Math.sin(a1)
      - m2 * g * Math.sin(a1 - 2 * a2)
      - 2 * sinDA * m2 * (v2 * v2 * l2 + v1 * v1 * l1 * cosDA);
    const a1dd = (num1 / (l1 * denom1)) - damping * v1;

    const num2 =
      2 * sinDA * (
        v1 * v1 * l1 * (m1 + m2)
        + g * (m1 + m2) * Math.cos(a1)
        + v2 * v2 * l2 * m2 * cosDA
      );
    const a2dd = (num2 / (l2 * denom2)) - damping * v2;

    return [a1dd, a2dd];
  }

  // RK4 step
  function rk4(state, p, dt) {
    const { a1, a2, v1, v2 } = state;

    const [k1v1, k1v2] = derivatives(a1, a2, v1, v2, p);
    const [k2v1, k2v2] = derivatives(
      a1 + v1 * dt / 2, a2 + v2 * dt / 2,
      v1 + k1v1 * dt / 2, v2 + k1v2 * dt / 2, p
    );
    const [k3v1, k3v2] = derivatives(
      a1 + v1 * dt / 2, a2 + v2 * dt / 2,
      v1 + k2v1 * dt / 2, v2 + k2v2 * dt / 2, p
    );
    const [k4v1, k4v2] = derivatives(
      a1 + v1 * dt, a2 + v2 * dt,
      v1 + k3v1 * dt, v2 + k3v2 * dt, p
    );

    return {
      a1: a1 + (dt / 6) * (v1 + 2 * (v1 + k2v1 * dt / 2) / (dt / 2 * 2) * (dt / 6)), // use velocity directly
      a2: a2 + dt * v2 + (dt * dt / 6) * (k1v2 + k2v2 + k3v2), // rough but stable
      v1: v1 + (dt / 6) * (k1v1 + 2 * k2v1 + 2 * k3v1 + k4v1),
      v2: v2 + (dt / 6) * (k1v2 + 2 * k2v2 + 2 * k3v2 + k4v2),
    };
  }

  // Cleaner RK4 – integrate positions via velocity too
  function stepRK4(s, p, dt) {
    function f(a1, a2, v1, v2) {
      const [dv1, dv2] = derivatives(a1, a2, v1, v2, p);
      return { da1: v1, da2: v2, dv1, dv2 };
    }

    const k1 = f(s.a1, s.a2, s.v1, s.v2);
    const k2 = f(s.a1 + k1.da1 * dt / 2, s.a2 + k1.da2 * dt / 2,
                 s.v1 + k1.dv1 * dt / 2, s.v2 + k1.dv2 * dt / 2);
    const k3 = f(s.a1 + k2.da1 * dt / 2, s.a2 + k2.da2 * dt / 2,
                 s.v1 + k2.dv1 * dt / 2, s.v2 + k2.dv2 * dt / 2);
    const k4 = f(s.a1 + k3.da1 * dt, s.a2 + k3.da2 * dt,
                 s.v1 + k3.dv1 * dt, s.v2 + k3.dv2 * dt);

    return {
      a1: s.a1 + (dt / 6) * (k1.da1 + 2 * k2.da1 + 2 * k3.da1 + k4.da1),
      a2: s.a2 + (dt / 6) * (k1.da2 + 2 * k2.da2 + 2 * k3.da2 + k4.da2),
      v1: s.v1 + (dt / 6) * (k1.dv1 + 2 * k2.dv1 + 2 * k3.dv1 + k4.dv1),
      v2: s.v2 + (dt / 6) * (k1.dv2 + 2 * k2.dv2 + 2 * k3.dv2 + k4.dv2),
    };
  }

  /* ── Positions ─────────────────────────────────────────── */
  function getPositions(s, p) {
    const cx = canvas.width / 2 + panX;
    const cy = canvas.height / 2 + panY;
    const x1 = cx + p.l1s * Math.sin(s.a1);
    const y1 = cy + p.l1s * Math.cos(s.a1);
    const x2 = x1 + p.l2s * Math.sin(s.a2);
    const y2 = y1 + p.l2s * Math.cos(s.a2);
    return { cx, cy, x1, y1, x2, y2 };
  }

  /* ── Energy ────────────────────────────────────────────── */
  function calcEnergy(s, p) {
    const { g, m1, m2, l1s: l1, l2s: l2 } = p;
    // Reference: pivot at 0
    const y1 = -l1 * Math.cos(s.a1);
    const y2 = y1 - l2 * Math.cos(s.a2);

    const KE =
      0.5 * m1 * (l1 * s.v1) ** 2 +
      0.5 * m2 * (
        (l1 * s.v1) ** 2 +
        (l2 * s.v2) ** 2 +
        2 * l1 * l2 * s.v1 * s.v2 * Math.cos(s.a1 - s.a2)
      );
    const PE = -g * (m1 * y1 + m2 * y2);
    return { KE, PE, TE: KE + PE };
  }

  /* ── Drawing ───────────────────────────────────────────── */
  const COLORS = {
    pivot:  "#7c6af7",
    arm1:   "#9d8fff",
    arm2:   "#c4baff",
    bob1:   "#7c6af7",
    bob2:   "#f76a6a",
    trace:  "#6af7c4",
    ghost:  "rgba(124,106,247,0.18)",
  };

  function drawScene() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background grid (subtle)
    ctx.strokeStyle = "rgba(255,255,255,0.025)";
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < W; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    const { cx, cy, x1, y1, x2, y2 } = getPositions(state, params);

    // Ghost trail
    if (document.getElementById("showGhost").checked && ghost.length > 1) {
      for (let i = 1; i < ghost.length; i++) {
        const alpha = i / ghost.length * 0.5;
        ctx.strokeStyle = `rgba(124,106,247,${alpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ghost[i - 1].x1, ghost[i - 1].y1);
        ctx.lineTo(ghost[i].x1, ghost[i].y1);
        ctx.stroke();
      }
    }

    // Position trace (tip of arm 2)
    if (document.getElementById("showTrace").checked && trace.length > 1) {
      const maxLen = parseInt(document.getElementById("traceLen").value);
      const start = Math.max(0, trace.length - maxLen);
      ctx.beginPath();
      ctx.moveTo(trace[start].x, trace[start].y);
      for (let i = start + 1; i < trace.length; i++) {
        const t = (i - start) / (trace.length - start);
        ctx.strokeStyle = `rgba(106,247,196,${t * 0.85})`;
        ctx.lineWidth = t * 1.5;
        ctx.beginPath();
        ctx.moveTo(trace[i - 1].x, trace[i - 1].y);
        ctx.lineTo(trace[i].x, trace[i].y);
        ctx.stroke();
      }
    }

    // Arm 1
    ctx.strokeStyle = COLORS.arm1;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    // Arm 2
    ctx.strokeStyle = COLORS.arm2;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Pivot
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.pivot;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Bob 1 — size scales with mass
    const r1 = 5 + params.m1 * 0.5;
    drawBob(x1, y1, r1, COLORS.bob1);

    // Bob 2
    const r2 = 5 + params.m2 * 0.5;
    drawBob(x2, y2, r2, COLORS.bob2);
  }

  function drawBob(x, y, r, color) {
    // Glow
    const grd = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
    grd.addColorStop(0, color + "55");
    grd.addColorStop(1, "transparent");
    ctx.beginPath();
    ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Bob
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  /* ── Energy bars ───────────────────────────────────────── */
  let maxEnergy = 1;

  function updateEnergyBars() {
    const { KE, PE, TE } = calcEnergy(state, params);
    const absTE = Math.abs(TE);
    if (absTE > maxEnergy) maxEnergy = absTE * 1.2;

    document.getElementById("keBar").style.width = Math.min(100, KE / maxEnergy * 100) + "%";
    document.getElementById("peBar").style.width = Math.min(100, Math.abs(PE) / maxEnergy * 100) + "%";
    document.getElementById("teBar").style.width = Math.min(100, absTE / maxEnergy * 100) + "%";
  }

  /* ── Main loop ─────────────────────────────────────────── */
  const DT = 1 / 120; // physics timestep
  let lastTime = null;
  let accumulator = 0;

  function loop(ts) {
    animId = requestAnimationFrame(loop);
    if (paused) return;

    if (lastTime === null) { lastTime = ts; return; }
    const elapsed = Math.min((ts - lastTime) / 1000, 0.05); // cap at 50ms
    lastTime = ts;

    const speedMul = parseFloat(document.getElementById("speed").value);
    accumulator += elapsed * speedMul;

    while (accumulator >= DT) {
      // Update live params that don't require reset
      params.g       = parseFloat(document.getElementById("gravity").value);
      params.damping = parseFloat(document.getElementById("damping").value);

      const prev = state;
      state = stepRK4(state, params, DT);

      // Capture ghost arm positions every 3 steps
      ghost.push({ x1: getPositions(prev, params).x1, y1: getPositions(prev, params).y1 });
      if (ghost.length > 80) ghost.shift();

      accumulator -= DT;
    }

    // Record trace position
    const pos = getPositions(state, params);
    trace.push({ x: pos.x2, y: pos.y2 });
    if (trace.length > 2000) trace.shift(); // hard cap

    drawScene();
    updateEnergyBars();
  }

  /* ── Controls wiring ───────────────────────────────────── */
  function wire(id, labelId, decimals) {
    const el = document.getElementById(id);
    const lbl = document.getElementById(labelId);
    el.addEventListener("input", () => {
      lbl.textContent = parseFloat(el.value).toFixed(decimals);
    });
  }

  wire("gravity", "gravVal", 1);
  wire("damping", "dampVal", 3);
  wire("speed",   "speedVal", 1);
  wire("l1",      "l1Val",   0);
  wire("l2",      "l2Val",   0);
  wire("m1",      "m1Val",   0);
  wire("m2",      "m2Val",   0);
  wire("a1",      "a1Val",   0);
  wire("a2",      "a2Val",   0);
  wire("traceLen","traceVal",0);

  // Params that need a reset to take effect
  ["l1","l2","m1","m2","a1","a2"].forEach(id => {
    document.getElementById(id).addEventListener("change", resetSim);
  });

  document.getElementById("resetBtn").addEventListener("click", resetSim);

  document.getElementById("pauseBtn").addEventListener("click", () => {
    paused = !paused;
    document.getElementById("pauseBtn").textContent = paused ? "Resume" : "Pause";
    if (!paused) { lastTime = null; animId = requestAnimationFrame(loop); }
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    trace = [];
    ghost = [];
  });

  /* ── Pan & drag-to-reposition ──────────────────────────── */
  let dragging = null;   // 'bob1' | 'bob2' | null
  let panning  = false;
  let panStart = { x: 0, y: 0 };
  let panOrigin = { x: 0, y: 0 };

  function canvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const src = e.touches ? e.touches[0] : e;
    return {
      x: (src.clientX - rect.left) * scaleX,
      y: (src.clientY - rect.top)  * scaleY,
    };
  }

  function onPointerDown(e) {
    const { x: mx, y: my } = canvasPoint(e);
    const { x1, y1, x2, y2 } = getPositions(state, params);

    if (paused) {
      const d1 = Math.hypot(mx - x1, my - y1);
      const d2 = Math.hypot(mx - x2, my - y2);
      if (d1 < 22) { dragging = "bob1"; return; }
      if (d2 < 22) { dragging = "bob2"; return; }
    }

    // Start pan
    panning = true;
    panStart  = { x: mx, y: my };
    panOrigin = { x: panX, y: panY };
    canvas.style.cursor = "grabbing";
  }

  function onPointerMove(e) {
    const { x: mx, y: my } = canvasPoint(e);

    if (dragging && paused) {
      const { cx, cy, x1, y1 } = getPositions(state, params);
      if (dragging === "bob1") {
        state.a1 = Math.atan2(mx - cx, my - cy);
        state.v1 = 0; state.v2 = 0;
      } else {
        state.a2 = Math.atan2(mx - x1, my - y1);
        state.v1 = 0; state.v2 = 0;
      }
      trace = []; ghost = [];
      drawScene();
      return;
    }

    if (panning) {
      panX = panOrigin.x + (mx - panStart.x);
      panY = panOrigin.y + (my - panStart.y);
      if (!paused) drawScene(); // will redraw on next frame anyway
    }
  }

  function onPointerUp() {
    dragging = null;
    panning  = false;
    canvas.style.cursor = "grab";
  }

  canvas.addEventListener("mousedown",  onPointerDown);
  canvas.addEventListener("mousemove",  onPointerMove);
  canvas.addEventListener("mouseup",    onPointerUp);
  canvas.addEventListener("mouseleave", onPointerUp);
  canvas.addEventListener("touchstart", e => { e.preventDefault(); onPointerDown(e); }, { passive: false });
  canvas.addEventListener("touchmove",  e => { e.preventDefault(); onPointerMove(e); }, { passive: false });
  canvas.addEventListener("touchend",   onPointerUp);

  canvas.style.cursor = "grab";
  canvas.title = "Drag to pan · Pause then drag bobs to reposition";

  /* ── Start ─────────────────────────────────────────────── */
  resetSim();

})();
