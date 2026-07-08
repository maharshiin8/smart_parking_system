const grid = document.getElementById('lot-grid');
const tooltip = document.getElementById('tooltip');
const refreshBtn = document.getElementById('refresh-btn');
const liveDot = document.getElementById('live-dot');

function renderStatus(data) {
  document.getElementById('stat-total').textContent = data.total_spots;
  document.getElementById('stat-empty').textContent = data.empty;
  document.getElementById('stat-occupied').textContent = data.occupied;
  document.getElementById('stat-rate').textContent = Math.round(data.occupancy_rate * 100) + '%';
  document.getElementById('updated-at').textContent = data.updated_at || 'just now';

  // infer grid shape from bbox rows/cols isn't in status.json directly,
  // so just lay spots out in the order given with a reasonable column count
  const cols = Math.ceil(Math.sqrt(data.total_spots * 2));
  grid.style.gridTemplateColumns = `repeat(${Math.min(cols, 6)}, 1fr)`;

  grid.innerHTML = '';
  data.spots.forEach(spot => {
    const el = document.createElement('div');
    el.className = `spot ${spot.prediction}`;
    el.textContent = spot.id;
    el.addEventListener('mousemove', (e) => {
      tooltip.textContent = `${spot.id} · ${spot.prediction} · ${(spot.confidence * 100).toFixed(1)}% conf.`;
      tooltip.style.left = e.clientX + 'px';
      tooltip.style.top = e.clientY + 'px';
      tooltip.classList.add('visible');
    });
    el.addEventListener('mouseleave', () => tooltip.classList.remove('visible'));
    grid.appendChild(el);
  });
}

async function fetchStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  renderStatus(data);
}

async function refreshFrame() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = 'Processing frame…';
  liveDot.style.background = '#f5c518';
  try {
    const res = await fetch('/api/refresh', { method: 'POST' });
    const data = await res.json();
    renderStatus(data);
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = 'Pull new frame';
    liveDot.style.background = '';
  }
}

refreshBtn.addEventListener('click', refreshFrame);

fetchStatus();
setInterval(fetchStatus, 15000);
