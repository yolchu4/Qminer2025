function $(id) { return document.getElementById(id); }

function setStatus(msg) {
  $("status").textContent = msg;
}

function dictToTable(obj) {
  if (!obj || typeof obj !== "object") return "<em>n/a</em>";
  const keys = Object.keys(obj);
  if (keys.length === 0) return "<em>empty</em>";

  const rows = keys.map(k => {
    const v = obj[k];
    const val = (typeof v === "number") ? v.toFixed(6) : String(v);
    return `<tr><td>${k}</td><td>${val}</td></tr>`;
  }).join("");

  return `
    <table>
      <thead><tr><th>target</th><th>value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderResults(resp, apiBase) {
  const metrics = resp.metrics || {};
  $("tbl_delta").innerHTML = dictToTable(metrics.delta);
  $("tbl_coverage").innerHTML = dictToTable(metrics.coverage);
  $("tbl_mae").innerHTML = dictToTable(metrics.mae);

  const runId = resp.run_id; // IMPORTANT: from backend patch
  const files = resp.files || [];
  if (!runId) {
    $("downloads").innerHTML = `<p><b>run_id missing</b> (update backend response to include run_id)</p>`;
    return;
  }

  const links = files.map(fn => {
    const url = `${apiBase}/download/${encodeURIComponent(runId)}/${encodeURIComponent(fn)}`;
    return `<li><a href="${url}" target="_blank" rel="noreferrer">${fn}</a></li>`;
  }).join("");

  $("downloads").innerHTML = `
    <p><b>output_dir:</b> ${resp.output_dir || ""}<br/>
       <b>run_id:</b> ${runId}</p>
    <ul>${links}</ul>
  `;
}

function getSettings() {
  const n_regimes = parseInt($("n_regimes").value, 10);
  const alpha = parseFloat($("alpha").value);
  const data_dir = $("data_dir").value.trim() || "data";
  const output_dir = $("output_dir").value.trim() || "";
  const api_base = $("api_base").value.trim() || "http://127.0.0.1:8000";

  if (!Number.isFinite(n_regimes) || n_regimes < 1) throw new Error("Invalid n_regimes");
  if (!Number.isFinite(alpha) || alpha < 0.5 || alpha > 0.99) throw new Error("Invalid alpha");

  return { n_regimes, alpha, data_dir, output_dir, api_base };
}

async function runLocal() {
  const { n_regimes, alpha, data_dir, output_dir, api_base } = getSettings();
  setStatus("Running local demo...");

  const form = new FormData();
  form.append("n_regimes", String(n_regimes));
  form.append("alpha", String(alpha));
  form.append("data_dir", data_dir);
  if (output_dir) form.append("output_dir", output_dir);

  const res = await fetch(`${api_base}/run/demo-local`, { method: "POST", body: form });
  const text = await res.text();
  if (!res.ok) throw new Error(text);

  const json = JSON.parse(text);
  setStatus(`Success.\n${JSON.stringify(json, null, 2)}`);
  renderResults(json, api_base);
}

async function runUpload() {
  const { n_regimes, alpha, output_dir, api_base } = getSettings();

  const train = $("train_file").files[0];
  const calib = $("calib_file").files[0];
  const predict = $("predict_file").files[0];
  const compare = $("compare_file").files[0];

  if (!train || !calib || !predict) {
    throw new Error("Please select train, calib, and predict CSV files.");
  }

  setStatus("Uploading files and running demo...");

  const form = new FormData();
  form.append("train_file", train);
  form.append("calib_file", calib);
  form.append("predict_file", predict);
  if (compare) form.append("compare_file", compare);

  form.append("n_regimes", String(n_regimes));
  form.append("alpha", String(alpha));
  if (output_dir) form.append("output_dir", output_dir);

  const res = await fetch(`${api_base}/run/demo`, { method: "POST", body: form });
  const text = await res.text();
  if (!res.ok) throw new Error(text);

  const json = JSON.parse(text);
  setStatus(`Success.\n${JSON.stringify(json, null, 2)}`);
  renderResults(json, api_base);
}

function setBusy(isBusy) {
  $("btn_run_local").disabled = isBusy;
  $("btn_run_upload").disabled = isBusy;
}

$("btn_run_local").addEventListener("click", async () => {
  try {
    setBusy(true);
    await runLocal();
  } catch (e) {
    setStatus(`ERROR:\n${e.message || e}`);
  } finally {
    setBusy(false);
  }
});

$("btn_run_upload").addEventListener("click", async () => {
  try {
    setBusy(true);
    await runUpload();
  } catch (e) {
    setStatus(`ERROR:\n${e.message || e}`);
  } finally {
    setBusy(false);
  }
});
