/* ═══════════════════════════════════════════════
   AutoML — Frontend Application Logic
   ═══════════════════════════════════════════════ */

const API_BASE = (() => {
  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1";
  return isLocal ? "http://localhost:5000" : window.location.origin;
})();
const API = `${API_BASE}/api`;

// ── State ──────────────────────────────────────
let state = {
  token: localStorage.getItem("automl_token") || null,
  user: JSON.parse(localStorage.getItem("automl_user") || "null"),
  datasetId: null,
  uploadData: null,   // full response from /upload
  analysisData: null,
  modelsData: null,
  baselineData: null,
  customData: null,
  distributionData: null,
  chatMessages: [],
  chatSending: false,
  modelCache: JSON.parse(localStorage.getItem("automl_model_cache") || "{}"),
};

// ── DOM Refs ───────────────────────────────────
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

function buildApiUrl(pathOrUrl) {
  if (!pathOrUrl) return API;
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) return pathOrUrl;
  if (pathOrUrl.startsWith("/")) return `${API_BASE}${pathOrUrl}`;
  return `${API}/${pathOrUrl}`;
}

function persistModelCache() {
  localStorage.setItem("automl_model_cache", JSON.stringify(state.modelCache || {}));
}

function cacheCurrentModelState() {
  if (!state.datasetId) return;
  state.modelCache[state.datasetId] = {
    modelsData: state.modelsData || null,
    baselineData: state.baselineData || null,
    customData: state.customData || null,
    updatedAt: Date.now(),
  };
  persistModelCache();
}

function restoreCachedModelState(datasetId) {
  if (!datasetId) return false;
  const cached = state.modelCache?.[datasetId];
  if (!cached) return false;

  state.modelsData = cached.modelsData || null;
  state.baselineData = cached.baselineData || null;
  state.customData = cached.customData || null;

  let restored = false;
  if (state.modelsData) {
    renderModelResults(state.modelsData);
    restored = true;
  }
  if (state.baselineData) {
    renderBaselineResults(state.baselineData);
    restored = true;
  }
  if (state.customData) {
    renderCustomResults(state.customData);
    restored = true;
  }
  return restored;
}

async function parseErrorResponse(res, fallbackMessage) {
  const raw = await res.text();
  try {
    const data = JSON.parse(raw);
    return data.error || fallbackMessage;
  } catch {
    return `${fallbackMessage} (status ${res.status})`;
  }
}

const pageLanding = $("#page-landing");
const pageDashboard = $("#page-dashboard");
const authModal = $("#auth-modal");
const formLogin = $("#form-login");
const formSignup = $("#form-signup");

// Navigation
const navLinksLanding = $("#nav-links-landing");
const navActions = $("#nav-actions");
const navUser = $("#nav-user");
const navUserName = $("#nav-user-name");

// Sidebar buttons
const sidebarBtns = {
  upload: $("#sidebar-upload"),
  overview: $("#sidebar-overview"),
  analysis: $("#sidebar-analysis"),
  models: $("#sidebar-models"),
  chatbot: $("#sidebar-chatbot"),
  profile: $("#sidebar-profile"),
};

// Tab panels
const tabPanels = {
  upload: $("#tab-upload"),
  overview: $("#tab-overview"),
  analysis: $("#tab-analysis"),
  models: $("#tab-models"),
  chatbot: $("#tab-chatbot"),
  profile: $("#tab-profile"),
};

// ── Init ───────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  checkAuth();
  bindEvents();
  initGoogleAuth();
});

// ═══════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════

window.handleCredentialResponse = async (response) => {
  try {
    const res = await fetch(`${API}/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: response.credential }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("automl_token", data.token);
    localStorage.setItem("automl_user", JSON.stringify(data.user));
    closeModal();
    showDashboard();
  } catch (err) {
    alert("Google signin failed: " + err.message);
  }
};

async function initGoogleAuth() {
  const loginContainer = document.getElementById("google-login-button");
  const signupContainer = document.getElementById("google-signup-button");
  if (!loginContainer && !signupContainer) return;

  try {
    const res = await fetch(`${API}/auth/google-config`);
    const cfg = await res.json();
    if (!res.ok || !cfg.enabled || !cfg.client_id) {
      [loginContainer, signupContainer].forEach((el) => {
        if (el) el.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">Google sign-in is not configured.</div>';
      });
      return;
    }

    if (!window.google?.accounts?.id) {
      [loginContainer, signupContainer].forEach((el) => {
        if (el) el.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">Google Identity script not loaded.</div>';
      });
      return;
    }

    window.google.accounts.id.initialize({
      client_id: cfg.client_id,
      callback: window.handleCredentialResponse,
      ux_mode: "popup",
      auto_select: false,
    });

    const commonRenderOpts = {
      type: "standard",
      shape: "rectangular",
      theme: "outline",
      size: "large",
    };

    if (loginContainer) {
      loginContainer.innerHTML = "";
      window.google.accounts.id.renderButton(loginContainer, {
        ...commonRenderOpts,
        text: "signin_with",
      });
    }
    if (signupContainer) {
      signupContainer.innerHTML = "";
      window.google.accounts.id.renderButton(signupContainer, {
        ...commonRenderOpts,
        text: "signup_with",
      });
    }
  } catch {
    [loginContainer, signupContainer].forEach((el) => {
      if (el) el.innerHTML = '<div style="font-size:12px;color:var(--text-muted);">Google sign-in is unavailable.</div>';
    });
  }
}

function checkAuth() {
  if (state.token && state.user) {
    showDashboard();
  } else {
    showLanding();
  }
}

function renderNavAuthState() {
  const isLoggedIn = Boolean(state.token && state.user);
  navActions.classList.toggle("hidden", isLoggedIn);
  navUser.classList.toggle("hidden", !isLoggedIn);
  if (isLoggedIn) {
    navUserName.textContent = state.user?.name || state.user?.email || "User";
  }
}

function showLanding() {
  pageLanding.classList.add("active");
  pageLanding.classList.remove("hidden");
  pageDashboard.classList.remove("active");
  pageDashboard.classList.add("hidden");
  navLinksLanding.classList.remove("hidden");
  renderNavAuthState();
}

function showDashboard() {
  pageLanding.classList.remove("active");
  pageLanding.classList.add("hidden");
  pageDashboard.classList.add("active");
  pageDashboard.classList.remove("hidden");
  navLinksLanding.classList.add("hidden");
  renderNavAuthState();
}

function openModal(form) {
  authModal.classList.remove("hidden");
  if (form === "signup") {
    formLogin.classList.add("hidden");
    formSignup.classList.remove("hidden");
  } else {
    formSignup.classList.add("hidden");
    formLogin.classList.remove("hidden");
  }
  // Re-create icons inside modal
  lucide.createIcons();
}

function closeModal() {
  authModal.classList.add("hidden");
  // Clear errors
  $("#login-error").classList.add("hidden");
  $("#signup-error").classList.add("hidden");
}

async function handleLogin(e) {
  e.preventDefault();
  const email = $("#login-email").value;
  const password = $("#login-password").value;
  const errEl = $("#login-error");
  const btn = $("#login-submit-btn");

  btn.disabled = true;
  try {
    const res = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Login failed");

    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("automl_token", data.token);
    localStorage.setItem("automl_user", JSON.stringify(data.user));
    closeModal();
    showDashboard();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

async function handleSignup(e) {
  e.preventDefault();
  const name = $("#signup-name").value;
  const email = $("#signup-email").value;
  const password = $("#signup-password").value;
  const errEl = $("#signup-error");
  const btn = $("#signup-submit-btn");

  btn.disabled = true;
  try {
    const res = await fetch(`${API}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Signup failed");

    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("automl_token", data.token);
    localStorage.setItem("automl_user", JSON.stringify(data.user));
    closeModal();
    showDashboard();
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

function handleLogout() {
  state.token = null;
  state.user = null;
  state.datasetId = null;
  state.uploadData = null;
  state.analysisData = null;
  state.modelsData = null;
  state.baselineData = null;
  state.customData = null;
  state.chatMessages = [];
  localStorage.removeItem("automl_token");
  localStorage.removeItem("automl_user");
  resetChatbotUI();
  showLanding();
}

// ═══════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════

function switchTab(tabName) {
  // Don't navigate to disabled tabs
  if (sidebarBtns[tabName]?.classList.contains("disabled")) return;

  // Activate sidebar button
  Object.values(sidebarBtns).forEach((b) => b?.classList.remove("active"));
  sidebarBtns[tabName]?.classList.add("active");

  // Show tab panel
  Object.values(tabPanels).forEach((p) => {
    p?.classList.remove("active");
    p?.classList.add("hidden");
  });
  tabPanels[tabName]?.classList.add("active");
  tabPanels[tabName]?.classList.remove("hidden");

  // Load data if needed
  if (tabName === "analysis" && !state.analysisData) loadAnalysis();
  if (tabName === "models" && !state.modelsData) loadModels();
  if (tabName === "chatbot") updateChatbotDatasetBadge();
  if (tabName === "profile") loadProfile();
}

function enableTab(name) {
  sidebarBtns[name]?.classList.remove("disabled");
}

// ═══════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════

function setupUpload() {
  const zone = $("#upload-zone");
  const input = $("#file-input");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
  });
}

async function uploadFile(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please upload a CSV file.");
    return;
  }

  const progress = $("#upload-progress");
  const progressFill = $("#progress-fill");
  const progressText = $("#upload-progress-text");
  const zone = $("#upload-zone");

  progress.classList.remove("hidden");
  zone.style.display = "none";
  progressFill.style.width = "30%";
  progressText.textContent = "Uploading dataset…";

  const formData = new FormData();
  formData.append("file", file);

  try {
    progressFill.style.width = "50%";
    progressText.textContent = "Analyzing columns…";

    const res = await fetch(`${API}/dataset/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${state.token}` },
      body: formData,
    });

    progressFill.style.width = "80%";
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    progressFill.style.width = "100%";
    progressText.textContent = "Analysis complete!";

    state.datasetId = data.dataset_id;
    state.uploadData = data;
    state.modelsData = null;
    state.baselineData = null;
    state.customData = null;
    state.chatMessages = [];
    resetChatbotUI();

    // Update sidebar
    const sidebarInfo = $("#sidebar-dataset-info");
    sidebarInfo.classList.remove("hidden");
    $("#sidebar-filename").textContent = data.filename;
    $("#sidebar-filesize").textContent = `${data.summary.rows} rows × ${data.summary.columns} cols`;

    // Enable tabs
    enableTab("overview");
    enableTab("analysis");
    enableTab("models");
    enableTab("chatbot");

    // Render overview
    renderOverview(data);

    // Switch to overview after delay
    setTimeout(() => {
      switchTab("overview");
      progress.classList.add("hidden");
      zone.style.display = "";
    }, 600);
  } catch (err) {
    progressText.textContent = `Error: ${err.message}`;
    progressFill.style.width = "0%";
    setTimeout(() => {
      progress.classList.add("hidden");
      zone.style.display = "";
    }, 2000);
  }
}

// ═══════════════════════════════════════════════
// RENDER: OVERVIEW
// ═══════════════════════════════════════════════

function renderOverview(data) {
  const s = data.summary;

  // Metric cards
  const metricCards = $("#metric-cards");
  metricCards.innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Rows</div>
      <div class="metric-value">${s.rows.toLocaleString()}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Columns</div>
      <div class="metric-value">${s.columns}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Missing Data</div>
      <div class="metric-value">${s.missing_pct}%</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Memory</div>
      <div class="metric-value">${s.memory_mb} MB</div>
    </div>
  `;

  // Type pills
  const typePills = $("#type-pills");
  const types = [
    { label: "Continuous", count: s.continuous_numeric, color: "#818cf8" },
    { label: "Discrete", count: s.discrete_numeric, color: "#34d399" },
    { label: "Categorical", count: s.categorical, color: "#fbbf24" },
    { label: "DateTime", count: s.datetime, color: "#f87171" },
  ];
  typePills.innerHTML = types
    .map(
      (t) => `
    <div class="type-pill">
      <span class="type-pill-dot" style="background:${t.color}"></span>
      <span>${t.label}</span>
      <span class="type-pill-count">${t.count}</span>
    </div>
  `
    )
    .join("");

  // Classification table
  renderTable("#classification-table", data.classification, [
    { key: "column", label: "Column" },
    {
      key: "type",
      label: "Type",
      render: (v) => `<span class="type-badge ${typeBadgeClass(v)}">${v}</span>`,
    },
    { key: "reasoning", label: "Reasoning" },
    { key: "unique_values", label: "Unique Values" },
  ]);

  // Preview table
  if (data.preview && data.preview.length) {
    const cols = Object.keys(data.preview[0]);
    renderTable(
      "#preview-table",
      data.preview,
      cols.map((c) => ({ key: c, label: c }))
    );
  }

  // Load quality metrics (separate endpoint)
  loadOverviewQuality();
}

async function loadOverviewQuality() {
  if (!state.datasetId) return;
  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/overview`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    const totalRows = data.summary.rows;
    const threshold = totalRows * 0.40;

    const tbodyHtml = data.quality_metrics.map(row => {
      let isHighMissing = row["Null Count"] >= threshold;
      let style = isHighMissing ? "background-color: rgba(239, 68, 68, 0.1); color: #f87171;" : "";

      return `<tr style="${style}">
        <td>
          ${row["Column"]} 
          ${isHighMissing ? '<span class="badge" style="margin-left:8px; background:#ef4444; color:#fff; border:none; padding:2px 6px;">≥40% Missing</span>' : ''}
        </td>
        <td>${row["Null Count"]} <span style="font-size:11px; opacity:0.7; margin-left:4px;">(${((row["Null Count"] / totalRows) * 100).toFixed(1)}%)</span></td>
        <td>${row["Duplicate Rows Involved"]}</td>
        <td>${row["Outlier Count"]}</td>
      </tr>`;
    }).join("");

    const theadHtml = `<tr><th style="text-align:left; padding:12px 18px; border-bottom:1px solid var(--border);">Column</th><th style="padding:12px 18px; text-align:left; border-bottom:1px solid var(--border);">Null Count</th><th style="padding:12px 18px; text-align:left; border-bottom:1px solid var(--border);">Duplicate Rows</th><th style="padding:12px 18px; text-align:left; border-bottom:1px solid var(--border);">Outliers</th></tr>`;
    $("#quality-table").innerHTML = `<table style="width:100%; border-collapse: collapse; font-size:13px;"><thead>${theadHtml}</thead><tbody>${tbodyHtml}</tbody></table>`;

    if (data.cleaned_preview && data.cleaned_preview.length) {
      const cols = Object.keys(data.cleaned_preview[0]);
      renderTable(
        "#preview-table",
        data.cleaned_preview,
        cols.map((c) => ({ key: c, label: c }))
      );
    }
  } catch (err) {
    console.error("Failed to load quality metrics:", err);
  }
}

// ═══════════════════════════════════════════════
// RENDER: ANALYSIS
// ═══════════════════════════════════════════════

async function loadAnalysis() {
  if (!state.datasetId) return;
  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/analysis`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    state.analysisData = data;
    renderAnalysis(data);
  } catch (err) {
    console.error("Analysis load error:", err);
  }
}

function renderAnalysis(data) {
  // Numerical stats
  if (data.numerical_stats?.length) {
    const keys = Object.keys(data.numerical_stats[0]);
    renderTable(
      "#numerical-stats-table",
      data.numerical_stats,
      keys.map((k) => ({ key: k, label: k }))
    );
  } else {
    $("#numerical-stats-table").innerHTML = '<div class="no-data">No numerical columns detected</div>';
  }

  // Categorical stats
  if (data.categorical_stats?.length) {
    const keys = Object.keys(data.categorical_stats[0]);
    renderTable(
      "#categorical-stats-table",
      data.categorical_stats,
      keys.map((k) => ({ key: k, label: k }))
    );
  } else {
    $("#categorical-stats-table").innerHTML = '<div class="no-data">No categorical columns detected</div>';
  }

  // Correlation matrix
  if (data.correlation_matrix) {
    const { columns, values } = data.correlation_matrix;
    const rows_data = values.map((row, i) => {
      const obj = { Column: columns[i] };
      columns.forEach((c, j) => (obj[c] = row[j]));
      return obj;
    });
    renderTable(
      "#correlation-matrix-table",
      rows_data,
      [{ key: "Column", label: "" }, ...columns.map((c) => ({ key: c, label: c }))]
    );

    // Plotly heatmap
    const trace = {
      z: values,
      x: columns,
      y: columns,
      type: 'heatmap',
      colorscale: 'RdBu',
      zmin: -1, zmax: 1
    };
    const layout = {
      margin: { t: 30, b: 80, l: 80, r: 30 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#a1a1aa' }
    };
    Plotly.newPlot('correlation-heatmap-plot', [trace], layout);

  } else {
    $("#correlation-matrix-table").innerHTML = '<div class="no-data">Insufficient continuous columns for correlation</div>';
  }

  // Spearman
  renderTableOrEmpty("#spearman-table", data.spearman, "No strong Spearman correlations detected");
  // Kendall
  renderTableOrEmpty("#kendall-table", data.kendall, "No strong Kendall correlations detected");

  // Prescriptive
  renderTableOrEmpty("#numeric-prescriptive-table", data.numeric_prescriptive, "No numeric prescriptions");
  renderTableOrEmpty("#categorical-prescriptive-table", data.categorical_prescriptive, "No categorical prescriptions");
  renderTableOrEmpty("#correlation-prescriptive-table", data.correlation_prescriptive, "No correlated features to remove");
  renderTableOrEmpty("#dataset-prescriptive-table", data.dataset_prescriptive, "Dataset looks healthy — no actions needed");

  // Show first sub-panel
  showSubPanel("analysis", "descriptive");
}

function showSubPanel(prefix, name) {
  $$(`.${prefix}-sub-panel`).forEach((p) => {
    p.classList.remove("active-sub");
    p.classList.add("hidden");
  });
  const target = $(`#${prefix}-sub-${name}`);
  if (target) {
    target.classList.add("active-sub");
    target.classList.remove("hidden");
  }

  $$(`.${prefix}-sub-tab`).forEach((t) => t.classList.remove("active"));
  $$(`.${prefix}-sub-tab[data-subtab="${name}"]`).forEach((t) => t.classList.add("active"));

  if (prefix === "overview" && name === "distribution" && !state.distributionData) {
    loadOverviewDistribution();
  }
}

async function loadOverviewDistribution() {
  if (!state.datasetId) return;
  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/distribution`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error();
    state.distributionData = data.distributions;
    renderDistribution(data.distributions);
  } catch (err) {
    console.error("Distribution load error:", err);
  }
}

function renderDistribution(dists) {
  const grid = $("#distribution-grid");
  grid.innerHTML = "";
  if (!dists || dists.length === 0) {
    grid.innerHTML = '<div class="no-data" style="grid-column: 1 / -1;">No distributions available</div>';
    return;
  }

  dists.forEach((d, i) => {
    const div = document.createElement("div");
    div.className = "card";

    let insightsHtml = "";
    if (d.insights && d.insights.length) {
      insightsHtml = d.insights.map(ins => `<tr><td style="padding:8px 16px; border-bottom:1px solid var(--border);">${ins.Metric}</td><td style="padding:8px 16px; border-bottom:1px solid var(--border); font-weight:500;">${ins.Value}</td></tr>`).join("");
    } else {
      insightsHtml = `<tr><td style="padding:8px 16px; border-bottom:1px solid var(--border);" colspan="2">No insights</td></tr>`;
    }

    div.innerHTML = `
      <div class="card-header">
        <h3>${d.column}</h3>
        <span class="badge ${typeBadgeClass(d.type)}">${d.type}</span>
      </div>
      <div style="padding:1rem;">
        <div id="${getDistributionPlotId(d, i)}" style="width:100%; height:250px;"></div>
      </div>
      <div class="table-wrapper">
        <table style="width:100%; border-collapse: collapse; font-size:12px;">
          <tbody>
            ${insightsHtml}
          </tbody>
        </table>
      </div>
    `;
    grid.appendChild(div);

    if (d.plot && d.plot.x) {
      const trace = {
        x: d.plot.x,
        y: d.plot.y,
        type: d.plot.type,
        marker: { color: '#818cf8' }
      };

      const layout = {
        margin: { t: 10, b: 40, l: 40, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#a1a1aa' }
      };

      if (d.plot.type === "histogram") {
        trace.type = "bar"; // API returns pre-binned histogram data as bar
      }

      Plotly.newPlot(getDistributionPlotId(d, i), [trace], layout, { displayModeBar: false });
    }
  });
}

function getDistributionPlotId(distItem, index) {
  const safeName = String(distItem?.column || `col_${index}`).replace(/[^a-zA-Z0-9_-]/g, "");
  return `dist-plot-${index}-${safeName}`;
}

// ═══════════════════════════════════════════════
// RENDER: MODELS
// ═══════════════════════════════════════════════

async function loadModels() {
  if (!state.datasetId) return;
  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/models`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    // Populate column selector
    const select = $("#target-select");
    select.innerHTML = '<option value="">Choose a column…</option>';
    (data.columns || []).forEach((col) => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error("Models load error:", err);
  }
}

async function handleTargetChange(e) {
  const target = e.target.value;
  resetBaselineView();
  resetCustomView();
  if (!target || !state.datasetId) {
    setModelsTabsEnabled(false);
    $("#model-results").classList.add("hidden");
    state.modelsData = null;
    return;
  }

  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/models?target=${encodeURIComponent(target)}`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    state.modelsData = data;
    renderModelResults(data);
  } catch (err) {
    console.error("Model results error:", err);
  }
}

function renderModelResults(data) {
  state.modelsData = data;
  const container = $("#model-results");
  container.classList.remove("hidden");
  setModelsTabsEnabled(true);

  // Task type badge
  $("#task-type-badge").textContent = data.target_type;
  populateCustomModelOptions(data.target_type);
  populateFeatureSelectors(data.features || []);

  // Features table
  if (data.features?.length) {
    renderTable("#features-table", data.features, [
      { key: "Feature", label: "Feature" },
      {
        key: "Type",
        label: "Type",
        render: (v) => `<span class="type-badge ${typeBadgeClass(v)}">${v}</span>`,
      },
      { key: "Recommended Preprocessing", label: "Preprocessing" },
    ]);
  } else {
    $("#features-table").innerHTML = '<div class="no-data">No suitable features found</div>';
  }

  // Recommendations table
  if (data.recommendations?.length) {
    renderTable("#recommendations-table", data.recommendations, [
      { key: "Recommended Model", label: "Model" },
      { key: "Why Use It", label: "Rationale" },
    ]);
  } else {
    $("#recommendations-table").innerHTML = '<div class="no-data">No model recommendations available</div>';
  }

  showSubPanel("models", "overview");
  cacheCurrentModelState();
}

function populateFeatureSelectors(features) {
  const baselineList = $("#baseline-feature-list");
  const customList = $("#custom-feature-list");
  const names = features.map((f) => f.Feature);

  [baselineList, customList].forEach((list) => {
    if (!list) return;
    list.innerHTML = "";
    names.forEach((name) => {
      const row = document.createElement("label");
      row.className = "feature-picker-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = name;
      checkbox.checked = true;
      const text = document.createElement("span");
      text.textContent = name;
      row.appendChild(checkbox);
      row.appendChild(text);
      list.appendChild(row);
    });
  });
}

function getSelectedFeatureColumns(listId) {
  const list = $(listId);
  if (!list) return [];
  return Array.from(list.querySelectorAll('input[type="checkbox"]:checked')).map((cb) => cb.value);
}

function setAllSelections(listId, selected) {
  const list = $(listId);
  if (!list) return;
  Array.from(list.querySelectorAll('input[type="checkbox"]')).forEach((cb) => {
    cb.checked = selected;
  });
}

function setModelsTabsEnabled(enabled) {
  const tabs = $("#models-sub-tabs");
  if (!tabs) return;
  tabs.classList.toggle("disabled", !enabled);
}

function populateCustomModelOptions(targetType) {
  const select = $("#custom-model-select");
  if (!select) return;

  const normalizedType = String(targetType || "").toLowerCase().trim();
  let resolvedType = targetType;
  if (normalizedType.includes("continuous") || normalizedType.includes("regression")) {
    resolvedType = "Regression";
  } else if (normalizedType.includes("multiclass") || normalizedType.includes("multi-class")) {
    resolvedType = "Multiclass Classification";
  } else if (normalizedType.includes("binary")) {
    resolvedType = "Binary Classification";
  }

  const optionsByType = {
    Regression: [
      "Linear Regression",
      "Ridge Regression",
      "Lasso Regression",
      "ElasticNet Regression",
      "Decision Tree Regressor",
      "Random Forest Regressor",
      "Gradient Boosting Regressor",
      "AdaBoost Regressor",
      "KNN Regressor",
      "SVR (Support Vector Regressor)",
    ],
    "Binary Classification": [
      "Logistic Regression",
      "Decision Tree Classifier",
      "Random Forest Classifier",
      "Gradient Boosting Classifier",
      "AdaBoost Classifier",
      "KNN Classifier",
      "SVM Classifier",
      "Naive Bayes",
    ],
    "Multiclass Classification": [
      "Logistic Regression",
      "Decision Tree Classifier",
      "Random Forest Classifier",
      "Gradient Boosting Classifier",
      "AdaBoost Classifier",
      "KNN Classifier",
      "SVM Classifier",
      "Naive Bayes",
    ],
  };
  const options = optionsByType[resolvedType] || ["Random Forest Classifier"];

  select.innerHTML = '<option value="">Select model…</option>';
  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    select.appendChild(el);
  });
  select.value = options[0] || "";
}

function resetBaselineView() {
  state.baselineData = null;
  $("#btn-download-pipeline")?.classList.add("hidden");
  [
    "#baseline-fit-card",
    "#baseline-train-test-card",
    "#baseline-metrics-card",
    "#baseline-model-details-card",
    "#baseline-model-graph-card",
    "#baseline-x-columns-card",
    "#baseline-report-card",
  ].forEach((selector) => $(selector)?.classList.add("hidden"));
  $("#baseline-train-test-table").innerHTML = '<div class="no-data">Train the baseline to compare train vs test behavior</div>';
  $("#baseline-metrics-table").innerHTML = '<div class="no-data">No baseline metrics yet</div>';
  $("#baseline-model-details-table").innerHTML = '<div class="no-data">Train the model to view estimator details</div>';
  $("#baseline-x-columns-table").innerHTML = '<div class="no-data">Selected X columns will appear here</div>';
  $("#baseline-report-table").innerHTML = '<div class="no-data">Classification report will appear for classification tasks</div>';
  if (window.Plotly) Plotly.purge("baseline-model-graph");
  $("#baseline-fit-status").textContent = "";
  $("#baseline-fit-reason").textContent = "";
}

function resetCustomView() {
  state.customData = null;
  $("#btn-download-custom-pipeline")?.classList.add("hidden");
  [
    "#custom-fit-card",
    "#custom-train-test-card",
    "#custom-metrics-card",
    "#custom-model-details-card",
    "#custom-model-graph-card",
    "#custom-x-columns-card",
    "#custom-report-card",
  ].forEach((selector) => $(selector)?.classList.add("hidden"));
  $("#custom-train-test-table").innerHTML = '<div class="no-data">Train a custom model to compare train vs test behavior</div>';
  $("#custom-metrics-table").innerHTML = '<div class="no-data">No custom model metrics yet</div>';
  $("#custom-model-details-table").innerHTML = '<div class="no-data">Train the model to view estimator details</div>';
  $("#custom-x-columns-table").innerHTML = '<div class="no-data">Selected X columns will appear here</div>';
  $("#custom-report-table").innerHTML = '<div class="no-data">Classification report will appear for classification tasks</div>';
  if (window.Plotly) Plotly.purge("custom-model-graph");
  $("#custom-fit-status").textContent = "";
  $("#custom-fit-reason").textContent = "";
}

function renderModelInsights(prefix, data) {
  const detailsCard = $(`#${prefix}-model-details-card`);
  const detailsTable = `#${prefix}-model-details-table`;
  const graphCard = $(`#${prefix}-model-graph-card`);
  const graphElId = `${prefix}-model-graph`;

  const detailRows = [];
  const details = data.model_details || {};
  Object.entries(details).forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") return;
    let value;
    if (Array.isArray(v)) {
      value = v.map((x) => (typeof x === "number" ? x.toFixed(6) : String(x))).join(", ");
    } else if (typeof v === "object") {
      value = Object.entries(v)
        .map(([pk, pv]) => `${pk}: ${pv}`)
        .join(" | ");
    } else if (typeof v === "number") {
      value = Number.isInteger(v) ? String(v) : v.toFixed(6);
    } else {
      value = String(v);
    }
    detailRows.push({ Property: k, Value: value });
  });

  if (detailRows.length) {
    detailsCard?.classList.remove("hidden");
    renderTable(detailsTable, detailRows, [
      { key: "Property", label: "Property" },
      { key: "Value", label: "Value" },
    ]);
  }

  const graph = data.model_graph;
  if (graph && Array.isArray(graph.x) && Array.isArray(graph.y) && graph.x.length && graph.y.length && window.Plotly) {
    graphCard?.classList.remove("hidden");
    const trace = {
      type: graph.type || "bar",
      x: graph.x,
      y: graph.y,
      marker: { color: "#818cf8" },
    };
    const layout = {
      title: { text: graph.title || "Model Graph", font: { color: "#a1a1aa", size: 13 } },
      margin: { t: 40, b: 90, l: 60, r: 20 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#a1a1aa" },
      xaxis: { tickangle: -25 },
      yaxis: { title: graph.y_label || "Value" },
    };
    Plotly.newPlot(graphElId, [trace], layout, { displayModeBar: false, responsive: true });
  }
}

function renderBaselineResults(data) {
  state.baselineData = data;

  const fitCard = $("#baseline-fit-card");
  const fitStatus = $("#baseline-fit-status");
  const fitReason = $("#baseline-fit-reason");
  fitCard.classList.remove("hidden");
  fitStatus.textContent = data.fit_assessment?.status || "Unknown";
  fitReason.textContent = data.fit_assessment?.reason || "No fit assessment available";

  const metricsEntries = Object.entries(data.metrics || {}).map(([metric, value]) => ({
    Metric: metric,
    Value: typeof value === "number" ? value : String(value),
  }));
  $("#baseline-metrics-card").classList.remove("hidden");
  renderTable("#baseline-metrics-table", metricsEntries, [
    { key: "Metric", label: "Metric" },
    { key: "Value", label: "Value" },
  ]);
  renderModelInsights("baseline", data);

  const usedColumns = (data.feature_columns || []).map((c) => ({ Column: c }));
  $("#baseline-x-columns-card").classList.remove("hidden");
  $("#baseline-x-columns-table").classList.add("x-columns-compact");
  renderTable("#baseline-x-columns-table", usedColumns, [{ key: "Column", label: "Column" }]);

  const isRegression = data.target_type === "Regression";
  const trainMetric = isRegression ? "Train RMSE" : "Train Accuracy";
  const testMetric = isRegression ? "Test RMSE" : "Test Accuracy";
  const trainValue = data.metrics?.[trainMetric];
  const testValue = data.metrics?.[testMetric];

  if (trainValue !== undefined && testValue !== undefined) {
    const rows = [
      {
        Metric: isRegression ? "RMSE" : "Accuracy",
        Train: trainValue,
        Test: testValue,
        Gap: isRegression ? Math.abs(testValue - trainValue) : Math.abs(trainValue - testValue),
      },
    ];
    $("#baseline-train-test-card").classList.remove("hidden");
    renderTable("#baseline-train-test-table", rows, [
      { key: "Metric", label: "Metric" },
      { key: "Train", label: "Train" },
      { key: "Test", label: "Test" },
      { key: "Gap", label: "Absolute Gap" },
    ]);
  }

  if (data.classification_report) {
    const reportRows = Object.entries(data.classification_report)
      .filter(([, vals]) => vals && typeof vals === "object")
      .map(([label, vals]) => ({
        Label: label,
        Precision: vals.precision,
        Recall: vals.recall,
        "F1 Score": vals["f1-score"],
        Support: vals.support,
      }));
    $("#baseline-report-card").classList.remove("hidden");
    renderTable("#baseline-report-table", reportRows, [
      { key: "Label", label: "Label" },
      { key: "Precision", label: "Precision" },
      { key: "Recall", label: "Recall" },
      { key: "F1 Score", label: "F1 Score" },
      { key: "Support", label: "Support" },
    ]);
  }

  $("#btn-download-pipeline")?.classList.remove("hidden");
  cacheCurrentModelState();
}

async function handleTrainBaseline() {
  const target = $("#target-select")?.value;
  const featureColumns = getSelectedFeatureColumns("#baseline-feature-list");
  if (!target || !state.datasetId) {
    alert("Select a target column first.");
    return;
  }
  if (!featureColumns.length) {
    alert("Select at least one X column for baseline training.");
    return;
  }

  const btn = $("#btn-train-baseline");
  const prev = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Training...`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/models/baseline`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.token}`,
      },
      body: JSON.stringify({ target, feature_columns: featureColumns }),
    });
    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Baseline training failed (server returned non-JSON, status ${res.status})`);
    }
    if (!res.ok) throw new Error(data.error || `Baseline training failed (status ${res.status})`);

    renderBaselineResults(data);
    showSubPanel("models", "baseline");
  } catch (err) {
    alert(`Baseline training failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = prev;
    lucide.createIcons();
  }
}

async function handleDownloadPipeline() {
  if (!state.baselineData?.pipeline_download_url) {
    alert("Train a baseline model first.");
    return;
  }

  try {
    const res = await fetch(buildApiUrl(state.baselineData.pipeline_download_url), {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    if (!res.ok) {
      throw new Error(await parseErrorResponse(res, "Pipeline download failed"));
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = `baseline_pipeline_${state.baselineData.target}.joblib`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    alert(err.message);
  }
}

function renderCustomResults(data) {
  state.customData = data;

  const fitCard = $("#custom-fit-card");
  const fitStatus = $("#custom-fit-status");
  const fitReason = $("#custom-fit-reason");
  fitCard.classList.remove("hidden");
  fitStatus.textContent = data.fit_assessment?.status || "Unknown";
  fitReason.textContent = data.fit_assessment?.reason || "No fit assessment available";

  const metricsEntries = Object.entries(data.metrics || {}).map(([metric, value]) => ({
    Metric: metric,
    Value: typeof value === "number" ? value : String(value),
  }));
  $("#custom-metrics-card").classList.remove("hidden");
  renderTable("#custom-metrics-table", metricsEntries, [
    { key: "Metric", label: "Metric" },
    { key: "Value", label: "Value" },
  ]);
  renderModelInsights("custom", data);

  const usedColumns = (data.feature_columns || []).map((c) => ({ Column: c }));
  $("#custom-x-columns-card").classList.remove("hidden");
  $("#custom-x-columns-table").classList.add("x-columns-compact");
  renderTable("#custom-x-columns-table", usedColumns, [{ key: "Column", label: "Column" }]);

  const isRegression = data.target_type === "Regression";
  const trainMetric = isRegression ? "Train RMSE" : "Train Accuracy";
  const testMetric = isRegression ? "Test RMSE" : "Test Accuracy";
  const trainValue = data.metrics?.[trainMetric];
  const testValue = data.metrics?.[testMetric];

  if (trainValue !== undefined && testValue !== undefined) {
    const rows = [
      {
        Metric: isRegression ? "RMSE" : "Accuracy",
        Train: trainValue,
        Test: testValue,
        Gap: isRegression ? Math.abs(testValue - trainValue) : Math.abs(trainValue - testValue),
      },
    ];
    $("#custom-train-test-card").classList.remove("hidden");
    renderTable("#custom-train-test-table", rows, [
      { key: "Metric", label: "Metric" },
      { key: "Train", label: "Train" },
      { key: "Test", label: "Test" },
      { key: "Gap", label: "Absolute Gap" },
    ]);
  }

  if (data.classification_report) {
    const reportRows = Object.entries(data.classification_report)
      .filter(([, vals]) => vals && typeof vals === "object")
      .map(([label, vals]) => ({
        Label: label,
        Precision: vals.precision,
        Recall: vals.recall,
        "F1 Score": vals["f1-score"],
        Support: vals.support,
      }));
    $("#custom-report-card").classList.remove("hidden");
    renderTable("#custom-report-table", reportRows, [
      { key: "Label", label: "Label" },
      { key: "Precision", label: "Precision" },
      { key: "Recall", label: "Recall" },
      { key: "F1 Score", label: "F1 Score" },
      { key: "Support", label: "Support" },
    ]);
  }

  $("#btn-download-custom-pipeline")?.classList.remove("hidden");
  cacheCurrentModelState();
}

async function handleTrainCustom() {
  const target = $("#target-select")?.value;
  const featureColumns = getSelectedFeatureColumns("#custom-feature-list");
  const modelName = $("#custom-model-select")?.value;
  const scalerName = $("#custom-scaler-select")?.value || "StandardScaler";
  const testSize = Number($("#custom-test-size")?.value || 0.2);
  const randomState = Number($("#custom-random-state")?.value || 42);

  if (!target || !state.datasetId) {
    alert("Select a target column first.");
    return;
  }
  if (!modelName) {
    alert("Select a custom model.");
    return;
  }
  if (!featureColumns.length) {
    alert("Select at least one X column for custom training.");
    return;
  }

  const btn = $("#btn-train-custom");
  const prev = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Training...`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/models/custom`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.token}`,
      },
      body: JSON.stringify({
        target,
        feature_columns: featureColumns,
        model_name: modelName,
        scaler_name: scalerName,
        test_size: testSize,
        random_state: randomState,
      }),
    });
    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Custom training failed (server returned non-JSON, status ${res.status})`);
    }
    if (!res.ok) throw new Error(data.error || `Custom training failed (status ${res.status})`);

    renderCustomResults(data);
    showSubPanel("models", "custom");
  } catch (err) {
    alert(`Custom training failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = prev;
    lucide.createIcons();
  }
}

async function handleDownloadCustomPipeline() {
  if (!state.customData?.pipeline_download_url) {
    alert("Train a custom model first.");
    return;
  }

  try {
    const res = await fetch(buildApiUrl(state.customData.pipeline_download_url), {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    if (!res.ok) {
      throw new Error(await parseErrorResponse(res, "Pipeline download failed"));
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = `custom_pipeline_${state.customData.target}.joblib`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    alert(err.message);
  }
}

// ═══════════════════════════════════════════════
// TABLE UTILITIES
// ═══════════════════════════════════════════════

function renderTable(selector, rows, columns) {
  const container = $(selector);
  if (!rows || !rows.length) {
    container.innerHTML = '<div class="no-data">No data available</div>';
    return;
  }

  const thead = columns
    .map((c) => `<th>${c.label}</th>`)
    .join("");

  const tbody = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((c) => {
            const val = row[c.key];
            const display = c.render ? c.render(val) : formatValue(val);
            return `<td>${display}</td>`;
          })
          .join("")}</tr>`
    )
    .join("");

  container.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderTableOrEmpty(selector, data, emptyMsg) {
  if (data && data.length) {
    const keys = Object.keys(data[0]);
    renderTable(
      selector,
      data,
      keys.map((k) => ({ key: k, label: k }))
    );
  } else {
    $(selector).innerHTML = `<div class="no-data">${emptyMsg}</div>`;
  }
}

function formatValue(val) {
  if (val === null || val === undefined || val === "") return "—";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return val.toLocaleString();
    return val.toFixed(4);
  }
  return String(val);
}

function typeBadgeClass(type) {
  const t = (type || "").toLowerCase();
  if (t.includes("continuous")) return "continuous";
  if (t.includes("discrete")) return "discrete";
  if (t.includes("categor")) return "categorical";
  if (t.includes("date") || t.includes("time")) return "datetime";
  return "unknown";
}

// ═══════════════════════════════════════════════
// CHATBOT
// ═══════════════════════════════════════════════

function updateChatbotDatasetBadge() {
  const badge = $("#chatbot-dataset-badge");
  if (!badge) return;
  if (!state.uploadData?.filename) {
    badge.textContent = "No dataset loaded";
    return;
  }
  badge.textContent = state.uploadData.filename;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatChatInline(text) {
  let out = text;
  out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  return out;
}

function formatAssistantMessage(text) {
  const escaped = escapeHtml(text || "").replace(/\r\n/g, "\n");
  const blocks = escaped.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  if (!blocks.length) return "";

  return blocks.map((block) => {
    const lines = block.split("\n").map((l) => l.trimEnd());
    const hasOnlyBullets = lines.every((line) => /^\s*[*-]\s+/.test(line));

    if (hasOnlyBullets) {
      const items = lines
        .map((line) => line.replace(/^\s*[*-]\s+/, ""))
        .map((line) => `<li>${formatChatInline(line)}</li>`)
        .join("");
      return `<ul class="chatbot-list">${items}</ul>`;
    }

    const withBreaks = lines.map((l) => formatChatInline(l)).join("<br>");
    return `<p>${withBreaks}</p>`;
  }).join("");
}

function appendChatMessage(role, text) {
  const container = $("#chatbot-messages");
  if (!container) return;

  const msg = document.createElement("div");
  msg.className = `chatbot-msg chatbot-msg-${role}`;
  if (role === "assistant") {
    msg.innerHTML = formatAssistantMessage(text);
  } else {
    msg.textContent = text;
  }
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

function resetChatbotUI() {
  const container = $("#chatbot-messages");
  const input = $("#chatbot-input");
  if (container) container.innerHTML = "";
  if (input) input.value = "";
  state.chatSending = false;
  updateChatbotDatasetBadge();
}

async function handleChatbotSubmit(e) {
  e.preventDefault();
  const input = $("#chatbot-input");
  const question = (input?.value || "").trim();

  if (state.chatSending) return;

  if (!state.datasetId || !state.uploadData) {
    alert("Upload a dataset first, then use the chatbot.");
    return;
  }
  if (!question) return;

  appendChatMessage("user", question);
  input.value = "";
  state.chatSending = true;
  input.disabled = true;

  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${state.token}`,
      },
      body: JSON.stringify({ question }),
    });

    const raw = await res.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Chat service returned non-JSON (status ${res.status})`);
    }
    if (!res.ok) throw new Error(data.error || "Chat request failed");

    appendChatMessage("assistant", data.answer || "No answer returned.");
  } catch (err) {
    appendChatMessage("assistant", `Error: ${err.message}`);
  } finally {
    state.chatSending = false;
    input.disabled = false;
    input.focus();
  }
}

// ═══════════════════════════════════════════════
// EVENT BINDING
// ═══════════════════════════════════════════════

function bindEvents() {
  // Auth modal
  $("#btn-open-login").addEventListener("click", () => openModal("login"));
  $("#btn-open-signup").addEventListener("click", () => openModal("signup"));
  $("#hero-cta-btn").addEventListener("click", () => {
    if (state.token && state.user) {
      showDashboard();
      switchTab("upload");
      return;
    }
    openModal("signup");
  });
  $("#modal-close").addEventListener("click", closeModal);
  authModal.addEventListener("click", (e) => {
    if (e.target === authModal) closeModal();
  });

  // Switch forms
  $("#switch-to-signup").addEventListener("click", (e) => {
    e.preventDefault();
    openModal("signup");
  });
  $("#switch-to-login").addEventListener("click", (e) => {
    e.preventDefault();
    openModal("login");
  });

  // Form submissions
  $("#login-form").addEventListener("submit", handleLogin);
  $("#signup-form").addEventListener("submit", handleSignup);

  // Logout
  $("#btn-logout").addEventListener("click", handleLogout);

  // Logo goes home
  $("#nav-logo-link").addEventListener("click", (e) => {
    e.preventDefault();
    showLanding();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Clicking the signed-in user name always opens Profile in dashboard
  navUserName.addEventListener("click", () => {
    if (!state.token) return;
    showDashboard();
    switchTab("profile");
  });

  // Learn more scroll
  $("#hero-learn-btn")?.addEventListener("click", () => {
    document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
  });

  // Sidebar navigation
  Object.entries(sidebarBtns).forEach(([name, btn]) => {
    btn.addEventListener("click", () => switchTab(name));
  });

  // Upload
  setupUpload();

  // Analysis sub-tabs
  $$(".analysis-sub-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      showSubPanel("analysis", tab.dataset.subtab);
    });
  });

  // Overview sub-tabs
  $$(".overview-sub-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      showSubPanel("overview", tab.dataset.subtab);
    });
  });

  // Models target select
  $("#target-select").addEventListener("change", handleTargetChange);

  // Models sub-tabs
  $$(".models-sub-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      if ($("#models-sub-tabs")?.classList.contains("disabled")) return;
      showSubPanel("models", tab.dataset.subtab);
    });
  });

  // Baseline model actions
  $("#btn-train-baseline")?.addEventListener("click", handleTrainBaseline);
  $("#btn-download-pipeline")?.addEventListener("click", handleDownloadPipeline);
  $("#btn-train-custom")?.addEventListener("click", handleTrainCustom);
  $("#btn-download-custom-pipeline")?.addEventListener("click", handleDownloadCustomPipeline);
  $("#baseline-select-all")?.addEventListener("click", () => setAllSelections("#baseline-feature-list", true));
  $("#baseline-clear-all")?.addEventListener("click", () => setAllSelections("#baseline-feature-list", false));
  $("#custom-select-all")?.addEventListener("click", () => setAllSelections("#custom-feature-list", true));
  $("#custom-clear-all")?.addEventListener("click", () => setAllSelections("#custom-feature-list", false));

  // Chatbot
  $("#chatbot-form")?.addEventListener("submit", handleChatbotSubmit);
  $("#chatbot-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#chatbot-form")?.requestSubmit();
    }
  });

  // Clean Dataset CTA
  $("#btn-clean-dataset")?.addEventListener("click", handleCleanDataset);

  // Download Cleaned Dataset CTA
  $("#btn-download-cleaned")?.addEventListener("click", handleDownloadCleaned);

  // Profile Form
  $("#profile-form")?.addEventListener("submit", handleProfileUpdate);

  // Window mousemove for Grid glow effect
  document.addEventListener("mousemove", (e) => {
    document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
    document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
  });

  // Download Report functionality 
  $$(".btn-report").forEach(btn => {
    btn.addEventListener("click", async () => {
      const targetId = btn.getAttribute("data-target");
      const prevHTML = btn.innerHTML;
      btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Generating...`;
      lucide.createIcons();
      btn.disabled = true;

      try {
        await compilePDFReport(targetId);
      } catch (err) {
        alert("Report generation failed: " + err.message);
      } finally {
        btn.innerHTML = prevHTML;
        lucide.createIcons();
        btn.disabled = false;
      }
    });
  });
}

async function handleDownloadCleaned() {
  if (!state.datasetId) return;
  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/download`, {
      method: 'GET',
      headers: { Authorization: `Bearer ${state.token}` }
    });
    if (!res.ok) throw new Error("Download failed");

    // Create blob and trigger download
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `cleaned_dataset.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (err) {
    alert(err.message);
  }
}

async function handleCleanDataset() {
  if (!state.datasetId) return;
  const btn = $("#btn-clean-dataset");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Cleaning...`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API}/dataset/${state.datasetId}/clean`, {
      method: "POST",
      headers: { Authorization: `Bearer ${state.token}` }
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    alert("Dataset successfully cleaned!");

    // Unhide the preview and download button
    $("#preview-card")?.classList.remove("hidden");
    $("#btn-download-cleaned")?.classList.remove("hidden");

    // Reload state logic
    loadOverviewQuality();
    loadAnalysis();
    loadModels();
    if (state.distributionData) {
      loadOverviewDistribution();
    }
  } catch (err) {
    alert("Cleaning failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="wrench" class="icon-sm"></i> Clean Dataset`;
    lucide.createIcons();
  }
}

// ═══════════════════════════════════════════════
// PROFILE
// ═══════════════════════════════════════════════

function loadProfile() {
  $("#profile-name").value = state.user?.name || "";
  loadMyDatasets();
}

async function handleProfileUpdate(e) {
  e.preventDefault();
  const name = $("#profile-name").value;
  const password = $("#profile-password").value;

  const btn = $("#btn-update-profile");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Updating...`;
  lucide.createIcons();

  try {
    const res = await fetch(`${API}/user/profile`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}` },
      body: JSON.stringify({ name, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    state.user = data.user;
    localStorage.setItem("automl_user", JSON.stringify(data.user));
    $("#nav-user-name").textContent = data.user.name;
    alert("Profile updated successfully");
    $("#profile-password").value = "";
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Update Profile";
  }
}

async function loadMyDatasets() {
  try {
    const res = await fetch(`${API}/user/datasets`, {
      headers: { Authorization: `Bearer ${state.token}` },
    });
    const data = await res.json();
    if (!res.ok) throw new Error("");

    const tbody = $("#profile-datasets-table tbody");
    tbody.innerHTML = "";
    if (data.datasets.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:20px;">No datasets uploaded yet</td></tr>`;
      return;
    }

    data.datasets.forEach(d => {
      const tr = document.createElement("tr");
      const historyPayload = encodeURIComponent(JSON.stringify(d.history || []));
      const historyRowId = `history-row-${d.dataset_id}`;
      tr.innerHTML = `
        <td style="padding:12px 18px; border-bottom:1px solid var(--border);">${d.filename}</td>
        <td style="padding:12px 18px; border-bottom:1px solid var(--border);color:var(--text-secondary)">${new Date(d.latest_uploaded_at).toLocaleString()}</td>
        <td style="padding:12px 18px; border-bottom:1px solid var(--border);color:var(--text-secondary)">${d.versions || 1}</td>
        <td style="padding:12px 18px; border-bottom:1px solid var(--border);">
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn btn-outline btn-sm action-profile-use" data-id="${d.dataset_id}">
              <i data-lucide="play" class="icon-sm"></i> Use
            </button>
            <button class="btn btn-outline btn-sm action-profile-dl" data-id="${d.dataset_id}">
              <i data-lucide="download" class="icon-sm"></i> Download
            </button>
            <button class="btn btn-outline btn-sm action-profile-report" data-id="${d.dataset_id}" data-filename="${d.filename}">
              <i data-lucide="file-text" class="icon-sm"></i> Full Report
            </button>
            <button class="btn btn-ghost btn-sm action-profile-history" data-row-id="${historyRowId}" data-filename="${d.filename}" data-history="${historyPayload}">
              <i data-lucide="history" class="icon-sm"></i> History
            </button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);

      const historyRow = document.createElement("tr");
      historyRow.id = historyRowId;
      historyRow.classList.add("hidden");
      historyRow.innerHTML = `
        <td colspan="4" style="padding:0; border-bottom:1px solid var(--border);">
          <div style="padding:14px 18px; background: var(--bg-elevated);">
            <div style="font-size:13px; color: var(--text-primary); margin-bottom:10px; font-weight:600;">Upload History (last 25)</div>
            <div class="table-wrapper" data-history-container></div>
          </div>
        </td>
      `;
      tbody.appendChild(historyRow);
    });
    lucide.createIcons();

    $$(".action-profile-dl").forEach(btn => {
      btn.addEventListener("click", async () => {
        const did = btn.dataset.id;
        try {
          const dRes = await fetch(`${API}/dataset/${did}/download`, { headers: { Authorization: `Bearer ${state.token}` } });
          const blob = await dRes.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        } catch (e) { alert("Download failed"); }
      });
    });

    $$(".action-profile-use").forEach(btn => {
      btn.addEventListener("click", async () => {
        const did = btn.dataset.id;
        try {
          const res = await fetch(`${API}/dataset/${did}/resume`, {
            headers: { Authorization: `Bearer ${state.token}` },
          });
          if (!res.ok) throw new Error(await parseErrorResponse(res, "Failed to load dataset"));

          const data = await res.json();
          state.datasetId = data.dataset_id;
          state.uploadData = data;
          state.analysisData = null;
          state.modelsData = null;
          state.baselineData = null;
          state.customData = null;
          state.distributionData = null;
          state.chatMessages = [];
          resetChatbotUI();

          const sidebarInfo = $("#sidebar-dataset-info");
          sidebarInfo.classList.remove("hidden");
          $("#sidebar-filename").textContent = data.filename;
          $("#sidebar-filesize").textContent = `${data.summary.rows} rows × ${data.summary.columns} cols`;

          enableTab("overview");
          enableTab("analysis");
          enableTab("models");
          enableTab("chatbot");

          renderOverview(data);
          restoreCachedModelState(data.dataset_id);
          switchTab("overview");
          alert("Dataset loaded for analysis");
        } catch (e) {
          alert(`Failed to use dataset: ${e.message}`);
        }
      });
    });

    $$(".action-profile-report").forEach(btn => {
      btn.addEventListener("click", async () => {
        const did = btn.dataset.id;
        const filename = btn.dataset.filename || "dataset";
        const prevHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="spinner icon-sm"></i> Building...`;
        lucide.createIcons();

        try {
          const res = await fetch(`${API}/dataset/${did}/resume`, {
            headers: { Authorization: `Bearer ${state.token}` },
          });
          if (!res.ok) throw new Error(await parseErrorResponse(res, "Failed to load dataset for report"));

          const data = await res.json();
          state.datasetId = data.dataset_id;
          state.uploadData = data;
          state.analysisData = null;
          state.modelsData = null;
          state.baselineData = null;
          state.customData = null;
          state.distributionData = null;

          const sidebarInfo = $("#sidebar-dataset-info");
          sidebarInfo.classList.remove("hidden");
          $("#sidebar-filename").textContent = data.filename;
          $("#sidebar-filesize").textContent = `${data.summary.rows} rows × ${data.summary.columns} cols`;

          renderOverview(data);
          await loadOverviewQuality();
          await loadAnalysis();
          await loadOverviewDistribution();
          restoreCachedModelState(data.dataset_id);
          await new Promise((resolve) => setTimeout(resolve, 150));

          await compilePDFReport("tab-combined", filename);
        } catch (e) {
          alert(`Combined report failed: ${e.message}`);
        } finally {
          btn.disabled = false;
          btn.innerHTML = prevHtml;
          lucide.createIcons();
        }
      });
    });

    $$(".action-profile-history").forEach(btn => {
      btn.addEventListener("click", () => {
        const rowId = btn.dataset.rowId;
        const historyRow = rowId ? document.getElementById(rowId) : null;
        if (!historyRow) return;

        const isHidden = historyRow.classList.contains("hidden");
        // Collapse any other open history rows first
        $$("#profile-datasets-table tbody tr[id^='history-row-']").forEach((r) => {
          if (r.id !== rowId) r.classList.add("hidden");
        });

        historyRow.classList.toggle("hidden", !isHidden);
        if (!isHidden) return;

        const filename = btn.dataset.filename || "Dataset";
        let history = [];
        try {
          history = JSON.parse(decodeURIComponent(btn.dataset.history || "[]"));
        } catch {
          history = [];
        }

        const rows = history.map((h, idx) => ({
          "#": idx + 1,
          Name: filename,
          "Date Time": new Date(h.uploaded_at).toLocaleString(),
        }));

        const container = historyRow.querySelector("[data-history-container]");
        if (container) {
          if (!rows.length) {
            container.innerHTML = '<div class="no-data" style="padding:16px;">No history available</div>';
          } else {
            renderTable(`#${historyRow.id} [data-history-container]`, rows, [
              { key: "#", label: "#" },
              { key: "Name", label: "Name" },
              { key: "Date Time", label: "Date Time" },
            ]);
          }
        }
      });
    });

  } catch (err) { }
}

// ═══════════════════════════════════════════════
// REPORT GENERATION (Programmatic)
// ═══════════════════════════════════════════════

async function compilePDFReport(tabId, overrideFilename = "") {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF('p', 'pt', 'a4');
  const margin = 40;
  let cursorY = margin + 20;

  function printHeader() {
    doc.setFontSize(10);
    doc.setFont("helvetica", "italic");
    doc.setTextColor(100);
    doc.text("AutoML", 595 - margin, margin + 5, { align: "right" });
    doc.setTextColor(0);
  }

  function addTitle(text, size = 18) {
    if (cursorY > 750) { doc.addPage(); printHeader(); cursorY = margin + 20; }
    doc.setFontSize(size);
    doc.setFont("helvetica", "bold");
    doc.text(text, margin, cursorY);
    cursorY += size + 10;
  }

  function addText(text, size = 11) {
    if (cursorY > 750) { doc.addPage(); printHeader(); cursorY = margin + 20; }
    doc.setFontSize(size);
    doc.setFont("helvetica", "normal");
    const splitText = doc.splitTextToSize(String(text), 595 - margin * 2);
    doc.text(splitText, margin, cursorY);
    cursorY += (splitText.length * (size + 4)) + 10;
  }

  async function addPlotlyImage(elementId, titleLabel = "") {
    const el = document.getElementById(elementId);
    if (!el || !el.data) return;
    try {
      const imgData = await Plotly.toImage(el, { format: 'png', width: 600, height: 350 });
      if (cursorY > 580) { doc.addPage(); printHeader(); cursorY = margin + 20; }
      if (titleLabel) {
        doc.setFontSize(11);
        doc.setFont("helvetica", "bold");
        doc.text(titleLabel, margin, cursorY);
        cursorY += 15;
      }
      doc.addImage(imgData, 'PNG', margin, cursorY, 350, 204);
      cursorY += 215; // Tight packing
    } catch (err) {
      console.warn("Plotly toImage failed", err);
    }
  }

  async function addPlotlyImageCompact(elementId, titleLabel = "") {
    const el = document.getElementById(elementId);
    if (!el || !el.data) return;
    try {
      const imgData = await Plotly.toImage(el, { format: 'png', width: 560, height: 260 });
      if (cursorY > 640) { doc.addPage(); printHeader(); cursorY = margin + 20; }
      if (titleLabel) {
        doc.setFontSize(10);
        doc.setFont("helvetica", "bold");
        doc.text(titleLabel, margin, cursorY);
        cursorY += 12;
      }
      doc.addImage(imgData, 'PNG', margin, cursorY, 300, 140);
      cursorY += 150;
    } catch (err) {
      console.warn("Plotly compact toImage failed", err);
    }
  }

  function autoTableSafe(options) {
    doc.autoTable(options);
    if (doc.lastAutoTable) cursorY = doc.lastAutoTable.finalY + 25;
  }

  function addDomTableIfPresent(selector, titleText = "", fontSize = 9) {
    const table = document.getElementById(selector)?.querySelector("table");
    if (!table || table.rows.length === 0) return false;
    if (titleText) addText(titleText, 12);
    autoTableSafe({ html: table, startY: cursorY, margin: { left: margin }, styles: { fontSize } });
    return true;
  }

  printHeader();

  if (tabId === "tab-overview") {
    addTitle("Dataset Overview Report", 24);

    if (!state.uploadData) throw new Error("No dataset loaded");
    const sum = state.uploadData.summary;

    addTitle("1. Basic Information", 16);
    autoTableSafe({
      startY: cursorY,
      margin: { left: margin },
      head: [['Metric', 'Value']],
      body: [
        ['Total Rows', String(sum.rows)],
        ['Total Columns', String(sum.columns)],
        ['Missing Data %', String(sum.missing_pct) + '%'],
        ['Memory Usage', String(sum.memory_mb) + ' MB']
      ],
      styles: { fontSize: 10 }
    });

    addTitle("2. Column Classification", 16);
    const clf = state.uploadData.classification.map(c => [c.column, c.type, String(c.unique_values), c.reasoning]);
    autoTableSafe({
      startY: cursorY,
      margin: { left: margin },
      head: [['Column', 'Type', 'Unique', 'Reasoning']],
      body: clf,
      styles: { fontSize: 9 }
    });

    const qTable = document.getElementById('quality-table')?.querySelector("table");
    if (qTable && qTable.rows.length > 0) {
      addTitle("3. Quality Metrics", 16);
      autoTableSafe({ html: qTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });
    }

    // Ensure distributions are available even if user did not open Distribution sub-tab.
    if (!Array.isArray(state.distributionData) || state.distributionData.length === 0) {
      await loadOverviewDistribution();
      // Allow Plotly to finish DOM updates before converting charts to images.
      await new Promise((resolve) => setTimeout(resolve, 150));
    }

    // Distributions graphs
    if (Array.isArray(state.distributionData) && state.distributionData.length > 0) {
      addTitle("4. Feature Distributions", 16);
      for (const [i, dist] of state.distributionData.entries()) {
        const pid = getDistributionPlotId(dist, i);
        await addPlotlyImage(pid, `Distribution: ${dist.column}`);
      }
    }

  } else if (tabId === "tab-analysis") {
    addTitle("Data Analysis Report", 24);
    if (!state.analysisData) throw new Error("No analysis data loaded");

    addTitle("1. Descriptive Statistics", 16);
    addText("Numerical Columns", 12);
    const nTable = document.getElementById('numerical-stats-table')?.querySelector("table");
    if (nTable) autoTableSafe({ html: nTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 8 } });

    addText("Categorical Columns", 12);
    const catTable = document.getElementById('categorical-stats-table')?.querySelector("table");
    if (catTable && catTable.rows.length > 0) autoTableSafe({ html: catTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 8 } });

    addTitle("2. Diagnostic Analytics", 16);
    await addPlotlyImage("correlation-heatmap-plot", "Pearson Correlation Heatmap");

    addText("Pearson Correlation Matrix", 12);
    const cTable = document.getElementById('correlation-matrix-table')?.querySelector("table");
    if (cTable) autoTableSafe({ html: cTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 8 } });

    addText("Spearman Correlations", 12);
    const sTable = document.getElementById('spearman-table')?.querySelector("table");
    if (sTable) autoTableSafe({ html: sTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 8 } });

    addText("Kendall Correlations", 12);
    const kTable = document.getElementById('kendall-table')?.querySelector("table");
    if (kTable) autoTableSafe({ html: kTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 8 } });

    addTitle("3. Prescriptive Recommendations", 16);
    addText("Numeric Actions:", 12);
    const pTableNumeric = document.getElementById('numeric-prescriptive-table')?.querySelector("table");
    if (pTableNumeric) autoTableSafe({ html: pTableNumeric, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

    addText("Categorical Actions:", 12);
    const pTableCat = document.getElementById('categorical-prescriptive-table')?.querySelector("table");
    if (pTableCat) autoTableSafe({ html: pTableCat, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

    addText("Correlation Removals:", 12);
    const pTableCorr = document.getElementById('correlation-prescriptive-table')?.querySelector("table");
    if (pTableCorr) autoTableSafe({ html: pTableCorr, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

    addText("Dataset-Level Actions:", 12);
    const pTableSet = document.getElementById('dataset-prescriptive-table')?.querySelector("table");
    if (pTableSet) autoTableSafe({ html: pTableSet, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

  } else if (tabId === "tab-models") {
    addTitle("ML Models Report", 24);
    if (!state.modelsData) throw new Error("No target variable selected / Models missing");

    addTitle("1. Target Configuration", 16);
    const selectedTarget = document.getElementById("target-select")?.value || state.baselineData?.target || state.customData?.target || "Not selected";
    addText("Target Column: " + selectedTarget);
    addText("Detected Task Type: " + (state.modelsData.target_type || state.modelsData.task_type || "Unknown"));
    cursorY += 10;

    addTitle("2. Model-Ready Features", 16);
    const fTable = document.getElementById('features-table')?.querySelector("table");
    if (fTable) autoTableSafe({ html: fTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

    addTitle("3. Recommended Algorithms", 16);
    const rTable = document.getElementById('recommendations-table')?.querySelector("table");
    if (rTable) autoTableSafe({ html: rTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });

    if (state.baselineData) {
      addTitle("4. Baseline Training Summary", 16);
      addText(`Fit Assessment: ${state.baselineData.fit_assessment?.status || "Unknown"}`);
      addText(state.baselineData.fit_assessment?.reason || "");

      addDomTableIfPresent("baseline-metrics-table", "Baseline Metrics", 9);
      addDomTableIfPresent("baseline-train-test-table", "Train vs Test Comparison", 9);
      addDomTableIfPresent("baseline-x-columns-table", "X Columns Used", 9);
      addDomTableIfPresent("baseline-model-details-table", "Model Details", 8);
      addDomTableIfPresent("baseline-report-table", "Classification Report", 8);

      await addPlotlyImage("baseline-model-graph", "Baseline Model Graph");
    }

    if (state.customData) {
      addTitle("5. Custom Training Summary", 16);
      addText(`Model: ${state.customData.model_name || "Custom"}`);
      addText(`Scaler: ${state.customData.scaler_name || "StandardScaler"}`);
      addText(`Test Size: ${state.customData.test_size ?? "0.2"}`);
      addText(`Random State: ${state.customData.random_state ?? "42"}`);
      addText(`Fit Assessment: ${state.customData.fit_assessment?.status || "Unknown"}`);
      addText(state.customData.fit_assessment?.reason || "");

      addDomTableIfPresent("custom-metrics-table", "Custom Model Metrics", 9);
      addDomTableIfPresent("custom-train-test-table", "Train vs Test Comparison", 9);
      addDomTableIfPresent("custom-x-columns-table", "X Columns Used", 9);
      addDomTableIfPresent("custom-model-details-table", "Model Details", 8);
      addDomTableIfPresent("custom-report-table", "Classification Report", 8);

      await addPlotlyImage("custom-model-graph", "Custom Model Graph");
    }

    if (!state.baselineData && !state.customData) {
      addText("No baseline or custom model has been trained yet. Train a model to include full model diagnostics.");
    }

  } else if (tabId === "tab-combined") {
    addTitle("Comprehensive Dataset Report", 24);
    if (!state.uploadData) throw new Error("No dataset loaded");

    const sum = state.uploadData.summary;
    addText(`Dataset: ${state.uploadData.filename || overrideFilename || "Current Dataset"}`, 12);
    addText(`Generated: ${new Date().toLocaleString()}`, 10);

    addTitle("1. Overview", 16);
    autoTableSafe({
      startY: cursorY,
      margin: { left: margin },
      head: [['Metric', 'Value']],
      body: [
        ['Total Rows', String(sum.rows)],
        ['Total Columns', String(sum.columns)],
        ['Missing Data %', String(sum.missing_pct) + '%'],
        ['Memory Usage', String(sum.memory_mb) + ' MB']
      ],
      styles: { fontSize: 10 }
    });

    const clf = state.uploadData.classification.map(c => [c.column, c.type, String(c.unique_values), c.reasoning]);
    autoTableSafe({
      startY: cursorY,
      margin: { left: margin },
      head: [['Column', 'Type', 'Unique', 'Reasoning']],
      body: clf,
      styles: { fontSize: 9 }
    });

    const qTable = document.getElementById('quality-table')?.querySelector("table");
    if (qTable && qTable.rows.length > 0) {
      addTitle("2. Data Quality", 16);
      autoTableSafe({ html: qTable, startY: cursorY, margin: { left: margin }, styles: { fontSize: 9 } });
    }

    if (!state.analysisData) {
      await loadAnalysis();
    }

    addTitle("3. Analysis", 16);
    addDomTableIfPresent("numerical-stats-table", "Numerical Statistics", 8);
    addDomTableIfPresent("categorical-stats-table", "Categorical Statistics", 8);
    await addPlotlyImage("correlation-heatmap-plot", "Pearson Correlation Heatmap");
    addDomTableIfPresent("spearman-table", "Spearman Correlations", 8);
    addDomTableIfPresent("kendall-table", "Kendall Correlations", 8);
    addDomTableIfPresent("numeric-prescriptive-table", "Numeric Recommendations", 8);
    addDomTableIfPresent("categorical-prescriptive-table", "Categorical Recommendations", 8);
    addDomTableIfPresent("correlation-prescriptive-table", "Correlation-Based Removal", 8);
    addDomTableIfPresent("dataset-prescriptive-table", "Dataset-Level Actions", 8);

    if (!Array.isArray(state.distributionData) || state.distributionData.length === 0) {
      await loadOverviewDistribution();
      await new Promise((resolve) => setTimeout(resolve, 150));
    }

    if (Array.isArray(state.distributionData) && state.distributionData.length > 0) {
      addTitle("4. Distribution Graphs", 16);
      for (const [i, dist] of state.distributionData.entries()) {
        const pid = getDistributionPlotId(dist, i);
        await addPlotlyImageCompact(pid, `Column: ${dist.column}`);
      }
    }

    addTitle("5. Model Section", 16);
    addText("Model details are included if you have trained Baseline and/or Custom models for this dataset.", 10);
    if (state.modelsData) {
      addText("Detected Task Type: " + (state.modelsData.target_type || "Unknown"), 11);
      addDomTableIfPresent("features-table", "Model-Ready Features", 8);
      addDomTableIfPresent("recommendations-table", "Recommended Algorithms", 8);
    }
    if (state.baselineData) {
      addText("Baseline Model", 12);
      addDomTableIfPresent("baseline-metrics-table", "Baseline Metrics", 8);
      addDomTableIfPresent("baseline-train-test-table", "Baseline Train vs Test", 8);
      addDomTableIfPresent("baseline-model-details-table", "Baseline Model Details", 8);
      addDomTableIfPresent("baseline-report-table", "Baseline Classification Report", 8);
      await addPlotlyImage("baseline-model-graph", "Baseline Model Graph");
    }
    if (state.customData) {
      addText("Custom Model", 12);
      addDomTableIfPresent("custom-metrics-table", "Custom Metrics", 8);
      addDomTableIfPresent("custom-train-test-table", "Custom Train vs Test", 8);
      addDomTableIfPresent("custom-model-details-table", "Custom Model Details", 8);
      addDomTableIfPresent("custom-report-table", "Custom Classification Report", 8);
      await addPlotlyImage("custom-model-graph", "Custom Model Graph");
    }
    if (!state.modelsData && !state.baselineData && !state.customData) {
      addText("No model results found for this dataset in current session. Kindly train the model first.", 10);
    }

  } else {
    addTitle("General Report", 24);
    addText("No specific data found for this view.");
  }

  const safeDataset = (overrideFilename || state.uploadData?.filename || "dataset")
    .replace(/\.[^/.]+$/, "")
    .replace(/[^a-zA-Z0-9_-]+/g, "_");
  const outputName = tabId === "tab-combined"
    ? `${safeDataset}_combined_report.pdf`
    : `${tabId}-report.pdf`;
  doc.save(outputName);
}
