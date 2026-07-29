// Static showcase dashboard -- fetches real project data (site/data/*.json,
// produced by scripts/export_site_data.py) and renders it. No backend, no
// live agent calls.

(function initMermaid() {
  const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: true, theme: isDark ? "dark" : "default", securityLevel: "loose" });
  }
})();

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function asPretty(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

// ---------------------------------------------------------------------
// Investigation Replay
// ---------------------------------------------------------------------

let replayState = { traces: [], current: null, stepIndex: 0, autoplayTimer: null };

function renderReplayPicker() {
  const picker = document.getElementById("replay-picker");
  picker.innerHTML = "";

  // Two categories can repeat (e.g. two separate real "oom" investigations)
  // -- number duplicates so the picker doesn't show two identical labels.
  const seenCounts = {};
  replayState.traces.forEach((trace) => {
    seenCounts[trace.category] = (seenCounts[trace.category] || 0) + 1;
  });
  const runningIndex = {};

  replayState.traces.forEach((trace, i) => {
    const btn = document.createElement("button");
    let label = trace.category.replace(/_/g, " ");
    if (seenCounts[trace.category] > 1) {
      runningIndex[trace.category] = (runningIndex[trace.category] || 0) + 1;
      label += ` #${runningIndex[trace.category]}`;
    }
    btn.textContent = label;
    btn.setAttribute("role", "tab");
    btn.addEventListener("click", () => selectTrace(i));
    picker.appendChild(btn);
  });
}

function selectTrace(index) {
  replayState.current = replayState.traces[index];
  replayState.stepIndex = 0;
  stopAutoplay();

  document.querySelectorAll("#replay-picker button").forEach((btn, i) => {
    btn.classList.toggle("active", i === index);
  });

  const trace = replayState.current;
  document.getElementById("replay-category").textContent = trace.category.replace(/_/g, " ");
  document.getElementById("replay-description").textContent = trace.description;
  document.getElementById("replay-final").hidden = true;

  renderStep();
}

function renderStep() {
  const trace = replayState.current;
  const container = document.getElementById("replay-steps");
  const finalBox = document.getElementById("replay-final");
  const progress = document.getElementById("replay-progress");
  const totalSteps = trace.steps.length;

  if (replayState.stepIndex >= totalSteps) {
    // Past the last tool step -- show the Final Answer.
    container.innerHTML = "";
    finalBox.hidden = false;
    document.getElementById("replay-final-json").textContent = asPretty(trace.final_answer);
    progress.textContent = `Step ${totalSteps} / ${totalSteps} (Final Answer)`;
    document.getElementById("replay-next").disabled = true;
    return;
  }

  finalBox.hidden = true;
  document.getElementById("replay-next").disabled = false;

  const step = trace.steps[replayState.stepIndex];
  container.innerHTML = `
    <div class="step-card">
      <div class="step-label">Thought</div>
      <p class="step-thought">${escapeHtml(step.thought)}</p>
      <div class="step-label">Action</div>
      <p class="step-action">${escapeHtml(step.action)}(${escapeHtml(JSON.stringify(step.action_input))})</p>
      <div class="step-label">Observation</div>
      <pre>${escapeHtml(asPretty(step.observation))}</pre>
    </div>
  `;
  progress.textContent = `Step ${replayState.stepIndex + 1} / ${totalSteps}`;
}

function stopAutoplay() {
  if (replayState.autoplayTimer) {
    clearInterval(replayState.autoplayTimer);
    replayState.autoplayTimer = null;
    document.getElementById("replay-autoplay").textContent = "▶ Autoplay";
  }
}

function initReplayControls() {
  document.getElementById("replay-prev").addEventListener("click", () => {
    stopAutoplay();
    if (replayState.stepIndex > 0) {
      replayState.stepIndex -= 1;
      renderStep();
    }
  });

  document.getElementById("replay-next").addEventListener("click", () => {
    stopAutoplay();
    const totalSteps = replayState.current.steps.length;
    if (replayState.stepIndex < totalSteps) {
      replayState.stepIndex += 1;
      renderStep();
    }
  });

  document.getElementById("replay-autoplay").addEventListener("click", (e) => {
    if (replayState.autoplayTimer) {
      stopAutoplay();
      return;
    }
    e.target.textContent = "⏸ Pause";
    replayState.autoplayTimer = setInterval(() => {
      const totalSteps = replayState.current.steps.length;
      if (replayState.stepIndex >= totalSteps) {
        stopAutoplay();
        return;
      }
      replayState.stepIndex += 1;
      renderStep();
    }, 1800);
  });
}

async function loadTraces() {
  const res = await fetch("data/traces.json");
  const data = await res.json();
  replayState.traces = data.traces;
  renderReplayPicker();
  initReplayControls();
  if (replayState.traces.length > 0) selectTrace(0);
}

// ---------------------------------------------------------------------
// Eval Results
// ---------------------------------------------------------------------

async function loadEvalResults() {
  const res = await fetch("data/eval_results.json");
  const data = await res.json();

  const tiles = [
    { label: "Category Accuracy", value: `${Math.round(data.category_accuracy * 100)}%` },
    { label: "Avg Keyword Hit Rate", value: data.avg_keyword_hit_rate.toFixed(2) },
    { label: "Avg Trajectory Score", value: data.avg_trajectory_score.toFixed(2) },
    { label: "Avg Judge Score", value: `${data.avg_judge_score.toFixed(1)} / 10` },
  ];
  document.getElementById("eval-tiles").innerHTML = tiles
    .map((t) => `<div class="stat-tile"><div class="stat-value">${t.value}</div><div class="stat-label">${t.label}</div></div>`)
    .join("");

  document.getElementById("eval-caveat").innerHTML = `<strong>Read this before the number above:</strong> ${escapeHtml(data.caveat)}`;

  const tbody = document.querySelector("#eval-table tbody");
  tbody.innerHTML = data.per_task_results
    .map(
      (r) => `
      <tr>
        <td>${escapeHtml(r.task_id)}</td>
        <td>${escapeHtml(r.expected_category)}</td>
        <td>${escapeHtml(r.predicted_category)}</td>
        <td>${r.category_correct ? "yes" : "no"}</td>
        <td>${r.keyword_hit_rate.toFixed(2)}</td>
        <td>${r.confidence.toFixed(2)}</td>
        <td>${r.trajectory_score.toFixed(2)}</td>
        <td>${r.judge_score.toFixed(1)}</td>
      </tr>`
    )
    .join("");
}

// ---------------------------------------------------------------------
// Phase Status
// ---------------------------------------------------------------------

function statusClass(status) {
  return `status-${status}`;
}

function statusLabel(status) {
  return status.replace(/-/g, " ");
}

async function loadPhases() {
  const res = await fetch("data/phases.json");
  const data = await res.json();

  document.getElementById("phase-strip").innerHTML = data.phases
    .map(
      (p) => `
      <div class="phase-card">
        <div class="phase-num">Phase ${escapeHtml(p.phase)}</div>
        <div class="phase-name">${escapeHtml(p.name)}</div>
        <span class="badge ${statusClass(p.status)}">${escapeHtml(statusLabel(p.status))}</span>
        <div class="phase-note">${escapeHtml(p.note)}</div>
      </div>`
    )
    .join("");
}

// ---------------------------------------------------------------------

loadTraces().catch((e) => console.error("Failed to load traces.json", e));
loadEvalResults().catch((e) => console.error("Failed to load eval_results.json", e));
loadPhases().catch((e) => console.error("Failed to load phases.json", e));
