/**
 * Solar Saathi — Frontend Application Logic
 * Handles form interaction, API calls, and result visualization
 */

const API_BASE = window.location.origin;

// ===================================================================
// DOM Elements
// ===================================================================
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const infoPanel = document.getElementById('infoPanel');
const loadingPanel = document.getElementById('loadingPanel');
const resultsSection = document.getElementById('results');
const tiltSlider = document.getElementById('tiltAngle');
const tiltValue = document.getElementById('tiltValue');

// ===================================================================
// Quick Location Buttons
// ===================================================================
document.querySelectorAll('.quick-loc-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('latitude').value = btn.dataset.lat;
    document.getElementById('longitude').value = btn.dataset.lon;
    
    // Highlight selected
    document.querySelectorAll('.quick-loc-btn').forEach(b => {
      b.style.background = '';
      b.style.borderColor = '';
      b.style.color = '';
    });
    btn.style.background = 'rgba(245, 158, 11, 0.15)';
    btn.style.borderColor = '#f59e0b';
    btn.style.color = '#f59e0b';
  });
});

// ===================================================================
// Tilt Angle Slider
// ===================================================================
tiltSlider.addEventListener('input', () => {
  tiltValue.textContent = tiltSlider.value + '°';
});

// ===================================================================
// Form Submission
// ===================================================================
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const latitude = parseFloat(document.getElementById('latitude').value);
  const longitude = parseFloat(document.getElementById('longitude').value);
  const moduleType = document.getElementById('moduleType').value;
  const mountingType = document.getElementById('mountingType').value;
  const tiltAngle = parseFloat(tiltSlider.value);
  const panelWattage = parseInt(document.getElementById('panelWattage').value);
  
  // Validate
  if (isNaN(latitude) || isNaN(longitude)) {
    showError('Please enter valid coordinates.');
    return;
  }
  
  if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    showError('Coordinates out of range. Latitude: -90 to 90, Longitude: -180 to 180.');
    return;
  }
  
  // Show loading
  infoPanel.style.display = 'none';
  loadingPanel.classList.add('active');
  resultsSection.classList.remove('active');
  predictBtn.disabled = true;
  predictBtn.textContent = '⏳ Analyzing...';
  
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        latitude,
        longitude,
        module_type: moduleType,
        mounting_type: mountingType,
        tilt_angle: tiltAngle,
        panel_wattage: panelWattage
      })
    });
    
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (result.success) {
      displayResults(result.data, latitude, longitude);
    } else {
      throw new Error('Prediction failed. Please try again.');
    }
    
  } catch (error) {
    console.error('Prediction error:', error);
    showError(error.message || 'Failed to connect to the prediction server. Make sure the backend is running on port 8000.');
    loadingPanel.classList.remove('active');
    infoPanel.style.display = 'block';
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = '⚡ Predict Lifespan';
  }
});

// ===================================================================
// Display Results
// ===================================================================
function displayResults(data, lat, lon) {
  loadingPanel.classList.remove('active');
  resultsSection.classList.add('active');
  
  // Scroll to results
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
  
  // Location
  document.getElementById('resultLocation').textContent = `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`;
  
  // Model metrics
  if (data.model_metrics) {
    const r2 = data.model_metrics.hybrid_r2;
    if (r2) {
      document.getElementById('modelR2').textContent = `R² = ${(r2).toFixed(4)}`;
    }
  }
  
  // Animate main metrics
  animateValue('metricLifespan', data.estimated_lifespan_years, 1);
  animateValue('metricDegradation', data.degradation_rate_pct_year, 3);
  animateValue('metricEnergy', data.annual_energy_yield_kwh, 0);
  animateValue('metricConfidence', (data.confidence_score * 100), 1);
  
  // Lifetime energy
  document.getElementById('lifetimeEnergy').textContent = data.lifetime_energy_yield_mwh || '—';
  document.getElementById('lcoeFactor').textContent = data.lcoe_factor || '—';
  document.getElementById('panelConfig').textContent = `${data.panel_wattage_W}W ${data.module_type}`;
  
  // Confidence bar
  document.getElementById('confidenceValue').textContent = (data.confidence_score * 100).toFixed(1) + '%';
  setTimeout(() => {
    document.getElementById('confidenceBar').style.width = (data.confidence_score * 100) + '%';
  }, 300);
  
  // Climate summary
  if (data.climate_summary) {
    renderClimateSummary(data.climate_summary);
  }
  
  // Charts
  if (data.degradation_projection) {
    renderDegradationChart(data.degradation_projection, data.estimated_lifespan_years);
  }
  if (data.monthly_climate) {
    renderClimateChart(data.monthly_climate);
  }
}

// ===================================================================
// Animated Counter
// ===================================================================
function animateValue(elementId, target, decimals) {
  const el = document.getElementById(elementId);
  if (!el) return;
  
  const duration = 1500;
  const start = 0;
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * eased;
    
    el.textContent = current.toFixed(decimals);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  
  requestAnimationFrame(update);
}

// ===================================================================
// Climate Summary Grid
// ===================================================================
function renderClimateSummary(climate) {
  const grid = document.getElementById('climateGrid');
  
  const items = [
    { icon: '🌡️', value: climate.avg_temperature_C + '°C', label: 'Avg Temperature' },
    { icon: '💧', value: climate.avg_humidity_pct + '%', label: 'Avg Humidity' },
    { icon: '☀️', value: climate.avg_ghi_kwh_m2_day, label: 'GHI (kWh/m²/day)' },
    { icon: '🔆', value: climate.avg_uv_index, label: 'UV Index' },
    { icon: '💨', value: climate.avg_wind_speed_ms + ' m/s', label: 'Wind Speed' },
    { icon: '🌧️', value: climate.avg_precipitation_mm_day + ' mm/d', label: 'Precipitation' },
  ];
  
  grid.innerHTML = items.map(item => `
    <div class="climate-item">
      <div class="icon">${item.icon}</div>
      <div class="value">${item.value}</div>
      <div class="label">${item.label}</div>
    </div>
  `).join('');
}

// ===================================================================
// Canvas Charts (No external dependency)
// ===================================================================

function renderDegradationChart(projection, lifespanYears) {
  const canvas = document.getElementById('degradationChart');
  const ctx = canvas.getContext('2d');
  
  // Set canvas size
  const wrapper = canvas.parentElement;
  canvas.width = wrapper.clientWidth * 2;
  canvas.height = wrapper.clientHeight * 2;
  ctx.scale(2, 2);
  
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartW = W - padding.left - padding.right;
  const chartH = H - padding.top - padding.bottom;
  
  // Clear
  ctx.clearRect(0, 0, W, H);
  
  // Data
  const years = projection.map(d => d.year);
  const power = projection.map(d => d.power_remaining_pct);
  const maxYear = Math.max(...years);
  
  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartH / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();
  }
  
  // Y-axis labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const val = 100 - i * 20;
    const y = padding.top + (chartH / 5) * i;
    ctx.fillText(val + '%', padding.left - 8, y + 4);
  }
  
  // X-axis labels
  ctx.textAlign = 'center';
  const xStep = Math.max(1, Math.floor(maxYear / 6));
  for (let yr = 0; yr <= maxYear; yr += xStep) {
    const x = padding.left + (yr / maxYear) * chartW;
    ctx.fillText('Yr ' + yr, x, H - padding.bottom + 20);
  }
  
  // 80% threshold line (end-of-life)
  const thresholdY = padding.top + (1 - 80 / 100) * chartH;
  ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(padding.left, thresholdY);
  ctx.lineTo(W - padding.right, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  
  // Label threshold
  ctx.fillStyle = '#f43f5e';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('80% — End of Life', padding.left + 4, thresholdY - 6);
  
  // Lifespan line
  if (lifespanYears <= maxYear) {
    const lifespanX = padding.left + (lifespanYears / maxYear) * chartW;
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(lifespanX, padding.top);
    ctx.lineTo(lifespanX, H - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`~${lifespanYears.toFixed(1)} yrs`, lifespanX, padding.top - 6);
  }
  
  // Area fill (gradient)
  const gradient = ctx.createLinearGradient(0, padding.top, 0, H - padding.bottom);
  gradient.addColorStop(0, 'rgba(59, 130, 246, 0.15)');
  gradient.addColorStop(1, 'rgba(59, 130, 246, 0.01)');
  
  ctx.beginPath();
  ctx.moveTo(padding.left, H - padding.bottom);
  years.forEach((yr, i) => {
    const x = padding.left + (yr / maxYear) * chartW;
    const y = padding.top + (1 - power[i] / 100) * chartH;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(padding.left + (years[years.length-1] / maxYear) * chartW, H - padding.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();
  
  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  years.forEach((yr, i) => {
    const x = padding.left + (yr / maxYear) * chartW;
    const y = padding.top + (1 - power[i] / 100) * chartH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Dots at key points
  [0, Math.floor(years.length / 4), Math.floor(years.length / 2), Math.floor(3 * years.length / 4), years.length - 1].forEach(i => {
    if (i >= years.length) return;
    const x = padding.left + (years[i] / maxYear) * chartW;
    const y = padding.top + (1 - power[i] / 100) * chartH;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#0a0e1a';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}


function renderClimateChart(monthly) {
  const canvas = document.getElementById('climateChart');
  const ctx = canvas.getContext('2d');
  
  const wrapper = canvas.parentElement;
  canvas.width = wrapper.clientWidth * 2;
  canvas.height = wrapper.clientHeight * 2;
  ctx.scale(2, 2);
  
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartW = W - padding.left - padding.right;
  const chartH = H - padding.top - padding.bottom;
  
  ctx.clearRect(0, 0, W, H);
  
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const temps = monthly.map(d => d.temperature);
  const humidities = monthly.map(d => d.humidity);
  const ghis = monthly.map(d => d.ghi);
  
  const maxTemp = Math.max(...temps) + 5;
  const minTemp = Math.min(...temps) - 5;
  const tempRange = maxTemp - minTemp;
  
  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(W - padding.right, y);
    ctx.stroke();
  }
  
  // Labels
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxTemp - (tempRange / 4) * i;
    const y = padding.top + (chartH / 4) * i;
    ctx.fillText(val.toFixed(0) + '°C', padding.left - 8, y + 4);
  }
  
  // Month labels
  ctx.textAlign = 'center';
  const n = Math.min(monthly.length, 12);
  for (let i = 0; i < n; i++) {
    const x = padding.left + (i / (n - 1)) * chartW;
    ctx.fillText(months[i] || `M${i+1}`, x, H - padding.bottom + 18);
  }
  
  // GHI bars
  const barWidth = chartW / n * 0.5;
  const maxGHI = Math.max(...ghis, 1);
  ghis.forEach((ghi, i) => {
    const x = padding.left + (i / (n - 1)) * chartW - barWidth / 2;
    const barH = (ghi / maxGHI) * chartH * 0.4;
    const y = H - padding.bottom - barH;
    
    const barGradient = ctx.createLinearGradient(x, y, x, H - padding.bottom);
    barGradient.addColorStop(0, 'rgba(245, 158, 11, 0.25)');
    barGradient.addColorStop(1, 'rgba(245, 158, 11, 0.05)');
    
    ctx.fillStyle = barGradient;
    roundRect(ctx, x, y, barWidth, barH, 3);
    ctx.fill();
  });
  
  // Temperature line
  ctx.beginPath();
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  temps.forEach((t, i) => {
    const x = padding.left + (i / (n - 1)) * chartW;
    const y = padding.top + ((maxTemp - t) / tempRange) * chartH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  
  // Temperature dots
  temps.forEach((t, i) => {
    const x = padding.left + (i / (n - 1)) * chartW;
    const y = padding.top + ((maxTemp - t) / tempRange) * chartH;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#f43f5e';
    ctx.fill();
  });
  
  // Legend
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = '#f43f5e';
  ctx.fillRect(W - padding.right - 100, padding.top, 12, 3);
  ctx.fillText('Temperature', W - padding.right - 84, padding.top + 4);
  
  ctx.fillStyle = 'rgba(245, 158, 11, 0.5)';
  ctx.fillRect(W - padding.right - 100, padding.top + 14, 12, 8);
  ctx.fillStyle = '#f59e0b';
  ctx.fillText('GHI', W - padding.right - 84, padding.top + 22);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ===================================================================
// Error Display
// ===================================================================
function showError(message) {
  // Create temporary error toast
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
    background: rgba(244, 63, 94, 0.95); color: #fff; padding: 14px 28px;
    border-radius: 12px; font-size: 0.9rem; font-weight: 500; z-index: 1000;
    box-shadow: 0 8px 32px rgba(244, 63, 94, 0.3);
    animation: fadeInUp 0.3s ease-out;
    max-width: 500px; text-align: center;
  `;
  toast.textContent = '⚠️ ' + message;
  document.body.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}

// ===================================================================
// Reset Prediction
// ===================================================================
function resetPrediction() {
  resultsSection.classList.remove('active');
  infoPanel.style.display = 'block';
  
  // Scroll to form
  document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
}

// ===================================================================
// Navbar scroll effect
// ===================================================================
window.addEventListener('scroll', () => {
  const navbar = document.getElementById('navbar');
  if (window.scrollY > 50) {
    navbar.style.background = 'rgba(10, 14, 26, 0.95)';
    navbar.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
  } else {
    navbar.style.background = 'rgba(10, 14, 26, 0.8)';
    navbar.style.boxShadow = 'none';
  }
});

// ===================================================================
// Smooth scroll for anchor links
// ===================================================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
