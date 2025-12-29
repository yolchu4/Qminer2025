// ==================================================
// Utilities
// ==================================================

function smoothSeries(y, window = 3) {
  return y.map((_, i) => {
    const start = Math.max(0, i - window);
    const end = Math.min(y.length, i + window + 1);
    const slice = y.slice(start, end);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

console.log("APP.JS VERSION = 2025-01-FINAL-STABLE");

// ==================================================
// CSV Loader (SAFE)
// ==================================================

async function loadCSV(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load " + url);

  const text = await res.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");

  return {
    headers,
    rows: lines.slice(1).map(line => {
      const values = line.split(",");
      return Object.fromEntries(
        headers.map((h, i) => [h, parseFloat(values[i].trim())])
      );
    })
  };
}

// ==================================================
// Global plot cache + chart instance
// ==================================================

let plotCache = {
  pred: null,
  truth: null,
  columns: []
};

let meanTrueChartInstance = null;

// ==================================================
// Run Demo
// ==================================================

async function runDemo() {
  const runBtn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  const spinner = document.getElementById("spinner");
  const filesDiv = document.getElementById("files");

  const deltaBody = document.querySelector("#deltaTable tbody");
  const coverageBody = document.querySelector("#coverageTable tbody");
  const maeBody = document.querySelector("#maeTable tbody");

  const n_regimes = document.getElementById("n_regimes").value;
  const alpha = document.getElementById("alpha").value;

  filesDiv.innerHTML = "";
  deltaBody.innerHTML = "";
  coverageBody.innerHTML = "";
  maeBody.innerHTML = "";

  runBtn.disabled = true;
  spinner.style.display = "block";
  status.textContent = "Status: Running…";

  const formData = new FormData();
  formData.append("n_regimes", n_regimes);
  formData.append("alpha", alpha);

  try {
    const res = await fetch("http://127.0.0.1:8000/run/demo-local", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Server error: " + res.status);

    const json = await res.json();
    console.log("API response:", json);

    if (json.metrics?.delta) fillTable(deltaBody, json.metrics.delta, 3);
    if (json.metrics?.coverage) fillTable(coverageBody, json.metrics.coverage, 3);
    if (json.metrics?.mae) fillTable(maeBody, json.metrics.mae, 3);

    let runName = null;

    if (json.files && json.output_dir) {
      runName = json.output_dir.replace(/\\/g, "/").split("/").pop();

      json.files.forEach(f => {
        const url = `http://127.0.0.1:8000/out/${runName}/${f}`;
        const a = document.createElement("a");
        a.href = url;
        a.textContent = "⬇ Download " + f;
        a.target = "_self";
        filesDiv.appendChild(a);
      });
    }

    status.textContent = "Status: Success";

    if (runName) {
      await plotMeanVsTrue(runName);
    }

  } catch (err) {
    console.error("UI error:", err);
    status.textContent = "Status: Error";
  } finally {
    spinner.style.display = "none";
    runBtn.disabled = false;
  }
}

// ==================================================
// Table helper
// ==================================================

function fillTable(tbody, data, digits = 3) {
  tbody.innerHTML = "";
  Object.entries(data).forEach(([k, v]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${v.toFixed(digits)}</td>`;
    tbody.appendChild(tr);
  });
}

// ==================================================
// Plot init + dropdown
// ==================================================

async function plotMeanVsTrue(runName) {
  const base = `http://127.0.0.1:8000/out/${runName}`;

  const pred = await loadCSV(`${base}/pred_mean.csv`);
  const truth = await loadCSV(`${base}/compare_dataset.csv`);

  const commonCols = pred.headers.filter(h => truth.headers.includes(h));
  if (commonCols.length === 0) return;

  plotCache = { pred, truth, columns: commonCols };

  const select = document.getElementById("paramSelect");
  select.innerHTML = "";

  commonCols.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    select.appendChild(opt);
  });

  drawChartForParam(commonCols[0]);
  select.onchange = () => drawChartForParam(select.value);
}

// ==================================================
// Draw chart (FINAL FIX)
// ==================================================

function drawChartForParam(targetCol) {
  const { pred, truth } = plotCache;
  if (!pred || !truth) return;

  const yPred = smoothSeries(pred.rows.map(r => r[targetCol]), 3);
  const yTrue = smoothSeries(truth.rows.map(r => r[targetCol]), 3);
  const labels = yTrue.map((_, i) => i);

  const canvas = document.getElementById("meanTrueChart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  if (meanTrueChartInstance) {
    meanTrueChartInstance.destroy();
  }

  meanTrueChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "True",
          data: yTrue,
          borderDash: [6, 4],
          borderColor: "#222",
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          label: "Predicted Mean",
          data: yPred,
          borderColor: "#2563eb",
          borderWidth: 2.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        title: {
          display: true,
          text: `Mean Prediction vs True — ${targetCol}`,
          font: { size: 16, weight: "bold" }
        },
        legend: { position: "top" }
      },
      scales: {
        x: { title: { display: true, text: "Sample Index" } },
        y: { title: { display: true, text: targetCol } }
      }
    }
  });
}
