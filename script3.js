// Constants and configuration
const NUM_FEATURES = 8;
const CAT_FEATURES = 2;
const TOTAL = NUM_FEATURES + CAT_FEATURES;
const CAT_CLASSES = [3, 2];
const SAMPLE_SIZE = 200;

// Sample frequency data for categorical features
const CAT_FREQS = [
  [120, 80, 50],
  [150, 100]
];

// Distribution types for numeric features
const DISTRIBUTION_TYPES = [
  'uniform', 'exponential', 'bimodal', 'lognormal', 'beta', 'gamma', 'chiSquared', 'weibull'
];

// DOM elements
const featuresGrid = document.getElementById('featuresGrid');
const status = document.getElementById('status');

// Data stores
let values = new Array(TOTAL).fill(null);
let uncertainties = new Array(TOTAL).fill(null);
let predictedFlags = new Array(TOTAL).fill(false);
let panelCharts = {};
let barCharts = {};
let chartVisibility = new Array(NUM_FEATURES).fill(true);
let currentFeatureIndex = null;
let featureData = {};

// Load data from generated files
async function loadFeatureData() {
  try {
    // In a real scenario, load from actual files
    // For now, generate sample data
    return generateFallbackData();
  } catch (error) {
    console.error('Error loading feature data:', error);
    return generateFallbackData();
  }
}

// Fallback data generation
function generateFallbackData() {
  console.log('Using fallback data generation');
  const fallbackData = {};
  
  for (let i = 0; i < NUM_FEATURES; i++) {
    const type = DISTRIBUTION_TYPES[i % DISTRIBUTION_TYPES.length];
    let data = [];
    
    // Simple fallback distributions
    switch (type) {
      case 'uniform':
        data = Array.from({length: SAMPLE_SIZE}, () => Math.random() * 100);
        break;
      case 'exponential':
        data = Array.from({length: SAMPLE_SIZE}, () => -Math.log(1 - Math.random()) * 20);
        break;
      case 'bimodal':
        data = Array.from({length: SAMPLE_SIZE}, () => Math.random() < 0.6 ? 
          30 + (Math.random() * 20) : 70 + (Math.random() * 20));
        break;
      case 'lognormal':
        data = Array.from({length: SAMPLE_SIZE}, () => Math.exp(Math.random() * 0.8 + 3.5));
        break;
      default:
        data = Array.from({length: SAMPLE_SIZE}, () => Math.random() * 100);
    }
    
    fallbackData[i] = {
      values: data,
      type: type,
      mean: data.reduce((a, b) => a + b, 0) / data.length,
      std: Math.sqrt(data.map(x => Math.pow(x - data.reduce((a, b) => a + b, 0) / data.length, 2)).reduce((a, b) => a + b, 0) / data.length)
    };
  }
  
  return fallbackData;
}

// Create histogram data from sample values
function createHistogramData(values, bins = 20) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  const binSize = range / bins;
  
  const histogram = new Array(bins).fill(0);
  
  values.forEach(value => {
    let binIndex = Math.floor((value - min) / binSize);
    if (binIndex === bins) binIndex = bins - 1;
    histogram[binIndex]++;
  });
  
  const maxCount = Math.max(...histogram);
  const normalizedHistogram = histogram.map(count => count / maxCount);
  
  const binLabels = [];
  for (let i = 0; i < bins; i++) {
    binLabels.push(min + (i + 0.5) * binSize);
  }
  
  return {
    values: normalizedHistogram,
    labels: binLabels,
    min: min,
    max: max
  };
}

// Create density curve data using kernel density estimation
function createDensityData(values, points = 100) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  
  const n = values.length;
  const std = Math.sqrt(values.map(x => Math.pow(x - values.reduce((a, b) => a + b, 0) / n, 2)).reduce((a, b) => a + b, 0) / n);
  const h = 1.06 * std * Math.pow(n, -0.2);
  
  const step = range / (points - 1);
  const xValues = [];
  const yValues = [];
  
  for (let i = 0; i < points; i++) {
    const x = min + i * step;
    let sum = 0;
    
    for (let j = 0; j < n; j++) {
      const u = (x - values[j]) / h;
      sum += Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
    }
    
    xValues.push(x);
    yValues.push(sum / (n * h));
  }
  
  const maxY = Math.max(...yValues);
  const normalizedY = yValues.map(y => y / maxY);
  
  return {
    x: xValues,
    y: normalizedY
  };
}

// Initialize the application
async function initApp() {
  console.log('Initializing application...');
  
  // Wait for DOM to be fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
      await initializeApp();
    });
  } else {
    await initializeApp();
  }
}

async function initializeApp() {
  try {
    featureData = await loadFeatureData();
    createFeaturePanels();
    setupEventListeners();
    loadModel();
    console.log('Application initialized successfully');
  } catch (error) {
    console.error('Error initializing app:', error);
    status.innerText = 'خطا در بارگذاری برنامه';
  }
}

// Create feature panels
function createFeaturePanels() {
  for (let i = 0; i < NUM_FEATURES; i++) {
    createNumericFeaturePanel(i);
  }
  
  for (let j = 0; j < CAT_FEATURES; j++) {
    createCategoricalFeaturePanel(j);
  }
}

// Create numeric feature panel
function createNumericFeaturePanel(index) {
  const data = featureData[index];
  
  const panel = document.createElement('div');
  panel.className = 'feature-panel';
  panel.id = `featurePanel${index}`;
  
  panel.innerHTML = `
    <div class="panel-header">
      <div class="panel-title">
        <span>ویژگی عددی ${index + 1}</span>
        <span class="feature-type">عددی</span>
        <span class="distribution-type">${data.type}</span>
      </div>
      <div class="panel-actions">
        <button class="toggle-btn" id="toggleChart${index}">−</button>
      </div>
    </div>
    <div class="panel-body">
      <div class="panel-value" id="valueDisplay${index}">
        <span>مقدار: <span id="valueText${index}">خالی</span></span>
        <div class="value-controls">
          <input type="number" class="value-input" id="valueInput${index}" placeholder="مقدار" step="0.1">
          <span class="predicted-badge" id="predictedBadge${index}" style="display: none;">پیش‌بینی شده</span>
        </div>
      </div>
      <div class="panel-chart">
        <canvas id="chart${index}" height="120"></canvas>
      </div>
      <div class="data-info">توزیع ${data.type} با ۲۰۰ نمونه داده</div>
      <input type="range" class="panel-slider" id="slider${index}" min="0" max="100" step="0.1" value="50">
      <div class="panel-footer">
        <div class="uncertainty" id="uncertainty${index}"></div>
        <div>
          <button class="chart-toggle-btn" id="chartSelect${index}">انتخاب روی نمودار</button>
        </div>
      </div>
    </div>
  `;
  
  featuresGrid.appendChild(panel);
  
  initFeatureChart(index, data);
  
  const valueInput = document.getElementById(`valueInput${index}`);
  const slider = document.getElementById(`slider${index}`);
  const chartSelectBtn = document.getElementById(`chartSelect${index}`);
  const toggleChartBtn = document.getElementById(`toggleChart${index}`);
  
  valueInput.addEventListener('input', (e) => {
    const v = e.target.value ? Number(e.target.value) : null;
    updateFeatureValue(index, v);
  });
  
  slider.addEventListener('input', (e) => {
    const v = Number(e.target.value);
    updateFeatureValue(index, v);
    valueInput.value = v;
  });
  
  chartSelectBtn.addEventListener('click', () => {
    openDistributionModal(index);
  });
  
  toggleChartBtn.addEventListener('click', () => {
    toggleChartVisibility(index);
  });
}

// Create categorical feature panel
function createCategoricalFeaturePanel(index) {
  const panelIndex = NUM_FEATURES + index;
  const panel = document.createElement('div');
  panel.className = 'feature-panel';
  panel.id = `featurePanel${panelIndex}`;
  
  let options = '<option value="">-- انتخاب --</option>';
  for (let k = 0; k < CAT_CLASSES[index]; k++) {
    options += `<option value="${k}">کلاس ${k}</option>`;
  }
  
  panel.innerHTML = `
    <div class="panel-header">
      <div class="panel-title">
        <span>ویژگی دسته‌ای ${index + 1}</span>
        <span class="feature-type">دسته‌ای</span>
      </div>
    </div>
    <div class="panel-body">
      <div class="panel-value" id="valueDisplay${panelIndex}">
        <span>مقدار: <span id="valueText${panelIndex}">خالی</span></span>
        <div class="value-controls">
          <span class="predicted-badge" id="predictedBadge${panelIndex}" style="display: none;">پیش‌بینی شده</span>
        </div>
      </div>
      <div class="panel-chart">
        <canvas id="barchart${index}" height="120"></canvas>
      </div>
      <select class="panel-select" id="select${index}">${options}</select>
      <div class="panel-footer">
        <div class="uncertainty" id="uncertainty${panelIndex}"></div>
      </div>
    </div>
  `;
  
  featuresGrid.appendChild(panel);
  
  initBarChart(index);
  
  const select = document.getElementById(`select${index}`);
  
  select.addEventListener('change', (e) => {
    const v = e.target.value;
    updateCategoricalFeatureValue(panelIndex, index, v);
  });
}

// Initialize feature chart with actual data distribution
function initFeatureChart(index, data) {
  const ctx = document.getElementById(`chart${index}`).getContext('2d');
  
  const histogramData = createHistogramData(data.values, 15);
  const densityData = createDensityData(data.values, 100);
  
  panelCharts[index] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: histogramData.labels,
      datasets: [
        {
          label: 'توزیع داده',
          data: histogramData.values,
          backgroundColor: 'rgba(78, 201, 176, 0.6)',
          borderColor: 'rgba(78, 201, 176, 1)',
          borderWidth: 1,
          barPercentage: 0.9,
          categoryPercentage: 1.0
        },
        {
          label: 'چگالی',
          data: densityData.y,
          type: 'line',
          borderColor: '#ff6b6b',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          tension: 0.3
        }
      ]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
      animation: { duration: 0 }
    }
  });
  
  if (values[index] !== null) {
    drawMarkerOnChart(panelCharts[index], values[index], histogramData.min, histogramData.max);
  }
}

// Update feature value
function updateFeatureValue(index, value) {
  values[index] = value;
  predictedFlags[index] = false;
  uncertainties[index] = null;
  
  document.getElementById(`valueText${index}`).innerText = value !== null ? value.toFixed(2) : 'خالی';
  document.getElementById(`valueDisplay${index}`).classList.remove('predicted');
  document.getElementById(`predictedBadge${index}`).style.display = 'none';
  document.getElementById(`slider${index}`).value = value !== null ? value : 50;
  
  if (value !== null) {
    const data = featureData[index];
    const histogramData = createHistogramData(data.values, 15);
    drawMarkerOnChart(panelCharts[index], value, histogramData.min, histogramData.max);
  }
  
  document.getElementById(`uncertainty${index}`).innerText = '';
}

// Update categorical feature value
function updateCategoricalFeatureValue(panelIndex, featureIndex, value) {
  values[panelIndex] = value ? Number(value) : null;
  predictedFlags[panelIndex] = false;
  uncertainties[panelIndex] = null;
  
  document.getElementById(`valueText${panelIndex}`).innerText = value ? value : 'خالی';
  document.getElementById(`valueDisplay${panelIndex}`).classList.remove('predicted');
  document.getElementById(`predictedBadge${panelIndex}`).style.display = 'none';
  updateBarChartSelection(featureIndex, value ? parseInt(value) : -1);
  document.getElementById(`uncertainty${panelIndex}`).innerText = '';
}

// Toggle chart visibility
function toggleChartVisibility(index) {
  const panel = document.getElementById(`featurePanel${index}`);
  const toggleBtn = document.getElementById(`toggleChart${index}`);
  
  if (chartVisibility[index]) {
    panel.classList.add('collapsed');
    toggleBtn.textContent = '+';
    chartVisibility[index] = false;
  } else {
    panel.classList.remove('collapsed');
    toggleBtn.textContent = '−';
    chartVisibility[index] = true;
  }
}

// Initialize bar chart
function initBarChart(j) {
  const ctx = document.getElementById(`barchart${j}`).getContext('2d');
  const frequencies = CAT_FREQS[j];
  const backgroundColors = frequencies.map((_, index) => 
    'rgba(78, 201, 176, 0.6)'
  );
  
  barCharts[j] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: frequencies.map((_, idx) => `クラス ${idx}`),
      datasets: [{
        label: 'تعداد نمونه',
        data: frequencies,
        backgroundColor: backgroundColors,
        borderColor: 'rgba(78, 201, 176, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, display: false }, x: { display: true } }
    }
  });
}

// Update bar chart selection
function updateBarChartSelection(j, selectedClass) {
  const chart = barCharts[j];
  const backgrounds = chart.data.labels.map((_, index) => 
    index === selectedClass ? 'rgba(76, 175, 80, 0.8)' : 'rgba(78, 201, 176, 0.6)'
  );
  
  chart.data.datasets[0].backgroundColor = backgrounds;
  chart.update();
}

// Draw marker on chart
function drawMarkerOnChart(chart, value, min, max) {
  chart.data.datasets = chart.data.datasets.filter(ds => !ds._isMarker);
  
  const xPos = ((value - min) / (max - min)) * (chart.data.labels.length - 1);
  const datasetIndex = Math.floor(xPos);
  const fraction = xPos - datasetIndex;
  
  let yPos = 0;
  if (datasetIndex >= 0 && datasetIndex < chart.data.labels.length - 1) {
    const y1 = chart.data.datasets[0].data[datasetIndex] || 0;
    const y2 = chart.data.datasets[0].data[datasetIndex + 1] || 0;
    yPos = y1 + fraction * (y2 - y1);
  } else {
    yPos = chart.data.datasets[0].data[datasetIndex] || 0;
  }
  
  const marker = {
    label: 'marker',
    data: Array(chart.data.labels.length).fill(null),
    pointBackgroundColor: 'red',
    pointRadius: 6,
    type: 'scatter',
    showLine: false,
    _isMarker: true
  };
  
  marker.data[Math.round(xPos)] = yPos;
  
  chart.data.datasets.push(marker);
  chart.update('none');
}

// Save input values to JSON file
function saveInputToJSON() {
  const inputData = {
    timestamp: new Date().toISOString(),
    numeric_features: {},
    categorical_features: {}
  };
  
  // Collect numeric features
  for (let i = 0; i < NUM_FEATURES; i++) {
    inputData.numeric_features[`feature_${i}`] = values[i];
  }
  
  // Collect categorical features
  for (let j = 0; j < CAT_FEATURES; j++) {
    const idx = NUM_FEATURES + j;
    inputData.categorical_features[`feature_${j}`] = values[idx];
  }
  
  // Convert to JSON string
  const jsonString = JSON.stringify(inputData, null, 2);
  
  // Create and download JSON file
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'input_features.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log('Input features saved to input_features.json');
  return inputData;
}

// Simulate prediction (fake model)
function simulatePrediction() {
  const prediction = {
    timestamp: new Date().toISOString(),
    numeric_predictions: {},
    categorical_predictions: {},
    uncertainties: {},
    probabilities: {}
  };
  
  // Generate predictions for numeric features
  for (let i = 0; i < NUM_FEATURES; i++) {
    const data = featureData[i];
    // Simple prediction based on distribution
    const predictedValue = data.mean + (Math.random() - 0.5) * data.std;
    const uncertainty = data.std * (0.8 + Math.random() * 0.4); // Random uncertainty
    
    prediction.numeric_predictions[`feature_${i}`] = predictedValue;
    prediction.uncertainties[`feature_${i}`] = uncertainty;
  }
  
  // Generate predictions for categorical features
  for (let j = 0; j < CAT_FEATURES; j++) {
    const classes = CAT_CLASSES[j];
    const randomClass = Math.floor(Math.random() * classes);
    const probability = 0.6 + Math.random() * 0.3; // 60-90% probability
    
    prediction.categorical_predictions[`feature_${j}`] = randomClass;
    prediction.probabilities[`feature_${j}`] = probability;
  }
  
  return prediction;
}

// Save prediction to JSON file
function savePredictionToJSON(predictionData) {
  const jsonString = JSON.stringify(predictionData, null, 2);
  
  // Create and download JSON file
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'prediction_results.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  console.log('Prediction results saved to prediction_results.json');
}

// Update UI with prediction results
function updateUIWithPrediction(predictionData) {
  // Update numeric features
  for (let i = 0; i < NUM_FEATURES; i++) {
    const predictedValue = predictionData.numeric_predictions[`feature_${i}`];
    const uncertainty = predictionData.uncertainties[`feature_${i}`];
    
    if (predictedValue !== undefined) {
      values[i] = predictedValue;
      uncertainties[i] = uncertainty;
      predictedFlags[i] = true;
      
      document.getElementById(`valueText${i}`).innerText = predictedValue.toFixed(3);
      document.getElementById(`valueInput${i}`).value = predictedValue.toFixed(3);
      document.getElementById(`slider${i}`).value = predictedValue;
      document.getElementById(`valueDisplay${i}`).classList.add('predicted');
      document.getElementById(`predictedBadge${i}`).style.display = 'inline-block';
      
      // Update chart marker
      const data = featureData[i];
      const histogramData = createHistogramData(data.values, 15);
      drawMarkerOnChart(panelCharts[i], predictedValue, histogramData.min, histogramData.max);
      
      document.getElementById(`uncertainty${i}`).innerText = `عدم قطعیت: ${uncertainty.toFixed(3)}`;
    }
  }
  
  // Update categorical features
  for (let j = 0; j < CAT_FEATURES; j++) {
    const idx = NUM_FEATURES + j;
    const predictedClass = predictionData.categorical_predictions[`feature_${j}`];
    const probability = predictionData.probabilities[`feature_${j}`];
    
    if (predictedClass !== undefined) {
      values[idx] = predictedClass;
      predictedFlags[idx] = true;
      
      document.getElementById(`select${j}`).value = predictedClass;
      document.getElementById(`valueText${idx}`).innerText = predictedClass;
      document.getElementById(`valueDisplay${idx}`).classList.add('predicted');
      document.getElementById(`predictedBadge${idx}`).style.display = 'inline-block';
      updateBarChartSelection(j, predictedClass);
      
      if (probability !== undefined) {
        document.getElementById(`uncertainty${idx}`).innerText = `احتمال: ${(probability * 100).toFixed(1)}%`;
      }
    }
  }
  
  // Update summary chart and table
  updateSummaryWithPrediction(predictionData);
}

// Update summary with prediction
function updateSummaryWithPrediction(predictionData) {
  const mu = [];
  const std = [];
  
  for (let i = 0; i < NUM_FEATURES; i++) {
    mu.push(predictionData.numeric_predictions[`feature_${i}`]);
    std.push(predictionData.uncertainties[`feature_${i}`]);
  }
  
  drawSummaryChart(mu, std);
  
  const table = buildResultTableFromPrediction(predictionData);
  document.getElementById('tableContainer').innerHTML = table;
}

// Build result table from prediction
function buildResultTableFromPrediction(predictionData) {
  let html = `
    <table class="table">
      <thead>
        <tr>
          <th>ویژگی</th>
          <th>مقدار پیش‌بینی</th>
          <th>عدم قطعیت</th>
          <th>دسته‌ای (احتمال)</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  for (let i = 0; i < NUM_FEATURES; i++) {
    const value = predictionData.numeric_predictions[`feature_${i}`];
    const uncertainty = predictionData.uncertainties[`feature_${i}`];
    
    html += `
      <tr>
        <td>num${i + 1} (${featureData[i].type})</td>
        <td>${value.toFixed(4)}</td>
        <td>${uncertainty.toFixed(4)}</td>
        <td>-</td>
      </tr>
    `;
  }
  
  for (let j = 0; j < CAT_FEATURES; j++) {
    const classValue = predictionData.categorical_predictions[`feature_${j}`];
    const probability = predictionData.probabilities[`feature_${j}`];
    
    html += `
      <tr>
        <td>cat${j + 1}</td>
        <td>-</td>
        <td>-</td>
        <td>کلاس ${classValue} (${(probability * 100).toFixed(1)}%)</td>
      </tr>
    `;
  }
  
  html += '</tbody></table>';
  return html;
}

// Setup event listeners
function setupEventListeners() {
  console.log('Setting up event listeners...');
  
  // Fill random values
  const fillRandomBtn = document.getElementById('fillRandom');
  if (fillRandomBtn) {
    fillRandomBtn.addEventListener('click', () => {
      console.log('Fill random clicked');
      const all = Array.from({ length: TOTAL }, (_, i) => i);
      const chosen = all.sort(() => 0.5 - Math.random()).slice(0, 5);
      
      chosen.forEach(i => {
        let v;
        if (i < NUM_FEATURES) {
          v = featureData[i].mean;
          document.getElementById(`slider${i}`).value = v;
          document.getElementById(`valueText${i}`).innerText = v.toFixed(2);
          document.getElementById(`valueInput${i}`).value = v;
          updateFeatureValue(i, v);
        } else {
          const selIdx = i - NUM_FEATURES;
          const sel = document.getElementById(`select${selIdx}`);
          const randomClass = Math.floor(Math.random() * CAT_CLASSES[selIdx]);
          sel.value = randomClass;
          document.getElementById(`valueText${i}`).innerText = randomClass;
          updateCategoricalFeatureValue(i, selIdx, randomClass);
        }
      });
    });
  } else {
    console.error('fillRandom button not found');
  }

  // Clear all values
  const clearAllBtn = document.getElementById('clearAll');
  if (clearAllBtn) {
    clearAllBtn.addEventListener('click', () => {
      console.log('Clear all clicked');
      values = new Array(TOTAL).fill(null);
      uncertainties = new Array(TOTAL).fill(null);
      predictedFlags = new Array(TOTAL).fill(false);
      
      for (let i = 0; i < NUM_FEATURES; i++) {
        const data = featureData[i];
        document.getElementById(`slider${i}`).value = data.mean;
        document.getElementById(`valueText${i}`).innerText = 'خالی';
        document.getElementById(`valueInput${i}`).value = '';
        document.getElementById(`valueDisplay${i}`).classList.remove('predicted');
        document.getElementById(`predictedBadge${i}`).style.display = 'none';
        updateFeatureValue(i, null);
        document.getElementById(`uncertainty${i}`).innerText = '';
        
        document.getElementById(`featurePanel${i}`).classList.remove('collapsed');
        document.getElementById(`toggleChart${i}`).textContent = '−';
        chartVisibility[i] = true;
      }
      
      for (let j = 0; j < CAT_FEATURES; j++) {
        const idx = NUM_FEATURES + j;
        document.getElementById(`select${j}`).value = '';
        document.getElementById(`valueText${idx}`).innerText = 'خالی';
        document.getElementById(`valueDisplay${idx}`).classList.remove('predicted');
        document.getElementById(`predictedBadge${idx}`).style.display = 'none';
        updateBarChartSelection(j, -1);
        document.getElementById(`uncertainty${idx}`).innerText = '';
      }
      
      if (window.summaryChart) window.summaryChart.destroy();
      document.getElementById('tableContainer').innerHTML = '';
    });
  } else {
    console.error('clearAll button not found');
  }

  // Predict values - FIXED VERSION
  const predictBtn = document.getElementById('predict');
  if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
      console.log('Predict button clicked');
      status.innerText = 'در حال پیش‌بینی...';
      
      try {
        // Step 1: Save current input values to JSON
        console.log('Saving input values to JSON...');
        saveInputToJSON();
        
        // Step 2: Simulate model processing
        console.log('Simulating model prediction...');
        setTimeout(() => {
          // Step 3: Generate prediction
          const predictionData = simulatePrediction();
          
          // Step 4: Save prediction to JSON
          savePredictionToJSON(predictionData);
          
          // Step 5: Update UI with prediction results
          updateUIWithPrediction(predictionData);
          
          status.innerText = 'پیش‌بینی تکمیل شد ✅';
          
          console.log('Prediction completed successfully');
        }, 1000); // Simulate 1 second delay for model processing
        
      } catch (error) {
        console.error('Prediction error:', error);
        status.innerText = 'خطا در پیش‌بینی ❌';
      }
    });
  } else {
    console.error('predict button not found');
  }

  // Report buttons
  const reportResultsBtn = document.getElementById('reportResults');
  if (reportResultsBtn) {
    reportResultsBtn.addEventListener('click', () => {
      const reportContent = generateResultsReport();
      showReport('گزارش نتایج با عدم قطعیت', reportContent);
    });
  } else {
    console.error('reportResults button not found');
  }

  const reportDistributionsBtn = document.getElementById('reportDistributions');
  if (reportDistributionsBtn) {
    reportDistributionsBtn.addEventListener('click', () => {
      const reportContent = generateDistributionsReport();
      showReport('گزارش توزیع ویژگی‌ها', reportContent);
    });
  } else {
    console.error('reportDistributions button not found');
  }

  // Close modals
  const closeReportBtn = document.getElementById('closeReport');
  if (closeReportBtn) {
    closeReportBtn.addEventListener('click', () => {
      document.getElementById('reportModal').style.display = 'none';
    });
  } else {
    console.error('closeReport button not found');
  }

  console.log('Event listeners setup completed');
}

// Generate results report
function generateResultsReport() {
  let html = `
    <h3>گزارش نتایج با عدم قطعیت‌ها</h3>
    <div class="report-table-container">
      <table class="table">
        <thead>
          <tr>
            <th>ویژگی</th>
            <th>مقدار</th>
            <th>عدم قطعیت</th>
            <th>وضعیت</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  for (let i = 0; i < NUM_FEATURES; i++) {
    const value = values[i] !== null ? values[i].toFixed(4) : 'خالی';
    const uncertainty = uncertainties[i] !== null ? uncertainties[i].toFixed(4) : '-';
    const status = predictedFlags[i] ? 'پیش‌بینی شده' : 'دستی';
    
    html += `
      <tr>
        <td>ویژگی عددی ${i + 1}</td>
        <td>${value}</td>
        <td>${uncertainty}</td>
        <td>${status}</td>
      </tr>
    `;
  }
  
  for (let j = 0; j < CAT_FEATURES; j++) {
    const idx = NUM_FEATURES + j;
    const value = values[idx] !== null ? values[idx] : 'خالی';
    const uncertainty = uncertainties[idx] !== null ? uncertainties[idx] : '-';
    const status = predictedFlags[idx] ? 'پیش‌بینی شده' : 'دستی';
    
    html += `
      <tr>
        <td>ویژگی دسته‌ای ${j + 1}</td>
        <td>${value}</td>
        <td>${uncertainty}</td>
        <td>${status}</td>
      </tr>
    `;
  }
  
  html += `
        </tbody>
      </table>
    </div>
    <div class="data-info">تاریخ تولید گزارش: ${new Date().toLocaleString('fa-IR')}</div>
  `;
  
  return html;
}

// Generate distributions report
function generateDistributionsReport() {
  let html = `
    <h3>گزارش توزیع ویژگی‌ها</h3>
    <div class="report-table-container">
      <table class="table">
        <thead>
          <tr>
            <th>ویژگی</th>
            <th>نوع توزیع</th>
            <th>میانگین</th>
            <th>انحراف معیار</th>
            <th>تعداد نمونه</th>
          </tr>
        </thead>
        <tbody>
  `;
  
  for (let i = 0; i < NUM_FEATURES; i++) {
    const data = featureData[i];
    html += `
      <tr>
        <td>ویژگی عددی ${i + 1}</td>
        <td>${data.type}</td>
        <td>${data.mean.toFixed(4)}</td>
        <td>${data.std.toFixed(4)}</td>
        <td>${SAMPLE_SIZE}</td>
      </tr>
    `;
  }
  
  for (let j = 0; j < CAT_FEATURES; j++) {
    const total = CAT_FREQS[j].reduce((a, b) => a + b, 0);
    html += `
      <tr>
        <td>ویژگی دسته‌ای ${j + 1}</td>
        <td>طبقه‌ای</td>
        <td>-</td>
        <td>-</td>
        <td>${total}</td>
      </tr>
    `;
  }
  
  html += `
        </tbody>
      </table>
    </div>
    <div class="data-info">تاریخ تولید گزارش: ${new Date().toLocaleString('fa-IR')}</div>
  `;
  
  return html;
}

// Show report modal
function showReport(title, content) {
  document.getElementById('reportModalTitle').textContent = title;
  document.getElementById('reportContainer').innerHTML = content;
  document.getElementById('reportModal').style.display = 'flex';
}

// Draw summary chart
function drawSummaryChart(mu, std) {
  const ctx = document.getElementById('summaryChart').getContext('2d');
  
  if (window.summaryChart) {
    window.summaryChart.destroy();
  }
  
  window.summaryChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: mu.map((_, i) => `ویژگی ${i + 1}`),
      datasets: [{
        label: 'μ',
        data: mu,
        backgroundColor: 'rgba(78, 201, 176, 0.6)',
        errorBars: std.map(s => ({ plus: s, minus: s }))
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const i = ctx.dataIndex;
              return `μ=${mu[i].toFixed(3)} ± σ=${std[i].toFixed(3)}`;
            }
          }
        }
      },
      animation: { duration: 600 }
    }
  });
}

// Load model (simulated)
function loadModel() {
  setTimeout(() => {
    status.innerText = 'مدل: آماده ✅ (مدل دمو)';
  }, 1500);
}

// Initialize the application
window.onload = function() {
  console.log('Window loaded, initializing app...');
  initApp();
};