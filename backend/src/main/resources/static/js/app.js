/**
 * Traffic Sign Recognition Frontend Application
 * Vanilla JS connecting directly to Spring Boot REST API & Realtime WebSocket
 */

function formatConfidence(val) {
  if (val == null || isNaN(val)) return 0;
  let num = Number(val);
  if (num <= 1.0 && num > 0) num = num * 100;
  return Math.round(num * 10) / 10;
}

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initImageRecognition();
  initVideoRecognition();
  initWebcamRecognition();
  initLogsAndStats();
  initClassCatalog();
});

// ==========================================
// 1. TABS MANAGEMENT
// ==========================================
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(targetTab);
      if (pane) pane.classList.add('active');

      if (targetTab === 'tab-logs') {
        loadStats();
        loadLogs(0);
      } else if (targetTab === 'tab-classes') {
        loadClasses();
      }
    });
  });
}

// ==========================================
// 2. IMAGE RECOGNITION
// ==========================================
function initImageRecognition() {
  const dropZone = document.getElementById('image-drop-zone');
  const fileInput = document.getElementById('image-file-input');
  const previewContainer = document.getElementById('image-preview-container');
  const previewImg = document.getElementById('image-preview');
  const btnCancel = document.getElementById('btn-cancel-image');
  const btnSubmit = document.getElementById('btn-submit-image');

  const emptyResult = document.getElementById('image-empty-result');
  const resultBox = document.getElementById('image-result-box');

  let selectedFile = null;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
      alert('Vui lòng chọn một tệp hình ảnh hợp lệ (PNG, JPEG, v.v.)');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropZone.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      btnSubmit.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  btnCancel.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    btnSubmit.disabled = true;
    emptyResult.classList.remove('hidden');
    resultBox.classList.add('hidden');
  });

  btnSubmit.addEventListener('click', async () => {
    if (!selectedFile) return;

    setLoading(btnSubmit, true);
    emptyResult.classList.add('hidden');
    resultBox.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/recognize/image', {
        method: 'POST',
        body: formData
      });

      const json = await res.json();
      if (!res.ok || !json.success) {
        throw new Error(json.message || 'Lỗi khi nhận diện ảnh');
      }

      renderImageResult(json.data);

    } catch (err) {
      alert('Lỗi: ' + err.message);
      emptyResult.classList.remove('hidden');
    } finally {
      setLoading(btnSubmit, false);
    }
  });

  function renderImageResult(data) {
    document.getElementById('img-sign-vi').textContent = data.signNameVi || 'Không xác định';
    document.getElementById('img-sign-en').textContent = data.signNameEn || '';
    document.getElementById('img-class-id').textContent = 'Class ID: ' + data.classId;

    const badgeCategory = document.getElementById('img-category-badge');
    badgeCategory.textContent = data.category || 'Biển báo';
    badgeCategory.className = 'badge ' + getCategoryBadgeClass(data.category);

    const confPct = formatConfidence(data.confidence);
    document.getElementById('img-confidence-val').textContent = confPct + '%';
    document.getElementById('img-confidence-bar').style.width = Math.min(confPct, 100) + '%';

    document.getElementById('img-latency').textContent = (data.inferenceTimeMs || 0) + ' ms';
    document.getElementById('img-log-id').textContent = '#' + (data.logId || '--');
    document.getElementById('img-filename').textContent = data.fileName || selectedFile.name;

    resultBox.classList.remove('hidden');
  }
}

// ==========================================
// 3. VIDEO RECOGNITION
// ==========================================
function initVideoRecognition() {
  const dropZone = document.getElementById('video-drop-zone');
  const fileInput = document.getElementById('video-file-input');
  const previewContainer = document.getElementById('video-preview-container');
  const previewVideo = document.getElementById('video-preview');
  const btnCancel = document.getElementById('btn-cancel-video');
  const btnSubmit = document.getElementById('btn-submit-video');

  const emptyResult = document.getElementById('video-empty-result');
  const resultsBox = document.getElementById('video-results-box');
  const summaryText = document.getElementById('video-summary-text');
  const tbody = document.getElementById('video-detections-tbody');

  let selectedFile = null;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleVideoSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleVideoSelect(fileInput.files[0]);
    }
  });

  function handleVideoSelect(file) {
    if (!file.type.startsWith('video/')) {
      alert('Vui lòng chọn một tệp video hợp lệ (MP4, AVI, v.v.)');
      return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    dropZone.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    btnSubmit.disabled = false;
  }

  btnCancel.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewVideo.src = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    btnSubmit.disabled = true;
    emptyResult.classList.remove('hidden');
    resultsBox.classList.add('hidden');
    summaryText.textContent = 'Chưa có dữ liệu phân tích video.';
  });

  btnSubmit.addEventListener('click', async () => {
    if (!selectedFile) return;

    setLoading(btnSubmit, true);
    emptyResult.classList.add('hidden');
    resultsBox.classList.add('hidden');

    const sampleInterval = document.getElementById('sample-interval').value || 0.5;
    const minThresholdPct = document.getElementById('min-threshold').value || 40;
    const minThreshold = minThresholdPct / 100.0;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('sampleIntervalSeconds', sampleInterval);
    formData.append('minConfidenceThreshold', minThreshold);

    try {
      const res = await fetch('/api/recognize/video', {
        method: 'POST',
        body: formData
      });

      const json = await res.json();
      if (!res.ok || !json.success) {
        throw new Error(json.message || 'Lỗi khi nhận diện video');
      }

      renderVideoResults(json.data);

    } catch (err) {
      alert('Lỗi phân tích video: ' + err.message);
      emptyResult.classList.remove('hidden');
    } finally {
      setLoading(btnSubmit, false);
    }
  });

  function renderVideoResults(data) {
    document.getElementById('vid-duration').textContent = (data.totalDurationSeconds || 0).toFixed(1) + 's';
    document.getElementById('vid-frames').textContent = data.totalFramesProcessed || 0;
    document.getElementById('vid-time').textContent = (data.processingTimeMs || 0) + 'ms';

    tbody.innerHTML = '';

    const detections = data.detections || [];
    summaryText.textContent = `Phát hiện ${detections.length} lượt biển báo trong video.`;

    if (detections.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-center">Không phát hiện biển báo nào vượt qua ngưỡng tin cậy.</td></tr>';
    } else {
      detections.forEach(d => {
        const pred = d.prediction;
        const tr = document.createElement('tr');
        const confPct = formatConfidence(pred.confidence);

        tr.innerHTML = `
          <td><strong>${(d.timestampSeconds || 0).toFixed(1)}s</strong></td>
          <td><span class="class-id-badge">#${pred.classId}</span></td>
          <td>${pred.signNameVi || '--'}</td>
          <td><span class="badge ${getCategoryBadgeClass(pred.category)}">${pred.category || 'Biển báo'}</span></td>
          <td class="text-success font-weight-bold">${confPct}%</td>
        `;
        tbody.appendChild(tr);
      });
    }

    resultsBox.classList.remove('hidden');
  }
}

// ==========================================
// 4. WEBCAM REALTIME VIA WEBSOCKET
// ==========================================
function initWebcamRecognition() {
  const btnStart = document.getElementById('btn-start-webcam');
  const btnStop = document.getElementById('btn-stop-webcam');
  const video = document.getElementById('webcam-video');
  const canvas = document.getElementById('webcam-canvas');
  const placeholder = document.getElementById('webcam-placeholder');
  const overlay = document.getElementById('webcam-overlay');

  const hudSignVi = document.getElementById('hud-sign-vi');
  const hudSignEn = document.getElementById('hud-sign-en');
  const hudConfBar = document.getElementById('hud-conf-bar');
  const hudConfText = document.getElementById('hud-conf-text');
  const hudFps = document.getElementById('hud-fps');
  const hudLatency = document.getElementById('hud-latency');

  let mediaStream = null;
  let socket = null;
  let streamInterval = null;
  let frameCount = 0;
  let fpsTimer = null;
  let lastSendTime = 0;

  btnStart.addEventListener('click', async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      video.srcObject = mediaStream;

      placeholder.classList.add('hidden');
      overlay.classList.remove('hidden');
      btnStart.classList.add('hidden');
      btnStop.classList.remove('hidden');

      // Initialize WebSocket connection
      connectWebSocket();

    } catch (err) {
      alert('Không thể truy cập Webcam: ' + err.message);
    }
  });

  btnStop.addEventListener('click', () => {
    stopWebcam();
  });

  function stopWebcam() {
    if (streamInterval) clearInterval(streamInterval);
    if (fpsTimer) clearInterval(fpsTimer);
    if (socket) socket.close();

    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }

    video.srcObject = null;
    placeholder.classList.remove('hidden');
    overlay.classList.add('hidden');
    btnStart.classList.remove('hidden');
    btnStop.classList.add('hidden');
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime`;

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      startFrameStreaming();
    };

    socket.onmessage = (event) => {
      const latency = Date.now() - lastSendTime;
      hudLatency.textContent = latency + ' ms';
      frameCount++;

      try {
        const res = JSON.parse(event.data);
        if (res.success && res.data) {
          const data = res.data;
          const conf = formatConfidence(data.confidence);

          if (conf >= 40) {
            hudSignVi.textContent = data.signNameVi || '--';
            hudSignEn.textContent = (data.signNameEn || '') + (data.category ? ' (' + data.category + ')' : '');
            hudConfBar.style.width = Math.min(conf, 100) + '%';
            hudConfText.textContent = 'Độ tin cậy: ' + conf + '%';
          } else {
            hudSignVi.textContent = 'Đang dò tìm biển báo...';
            hudSignEn.textContent = 'Đưa biển báo lại gần camera';
            hudConfBar.style.width = '0%';
            hudConfText.textContent = 'Độ tin cậy: < 40%';
          }
        }
      } catch (e) {
        console.error('Error parsing WS response:', e);
      }
    };

    socket.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    socket.onclose = () => {
      if (streamInterval) clearInterval(streamInterval);
    };
  }

  function startFrameStreaming() {
    const ctx = canvas.getContext('2d');

    // Send frame every 250ms (4 FPS is optimal for real-time traffic sign inspection)
    streamInterval = setInterval(() => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      const size = Math.min(vw, vh);
      const sx = (vw - size) / 2;
      const sy = (vh - size) / 2;

      canvas.width = 128;
      canvas.height = 128;
      ctx.drawImage(video, sx, sy, size, size, 0, 0, 128, 128);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

      lastSendTime = Date.now();
      socket.send(dataUrl);
    }, 250);

    // Track FPS
    frameCount = 0;
    fpsTimer = setInterval(() => {
      hudFps.textContent = (frameCount * 2) + ' FPS';
      frameCount = 0;
    }, 1000);
  }
}

// ==========================================
// 5. LOGS & STATISTICS
// ==========================================
let currentPage = 0;

function initLogsAndStats() {
  const filterSelect = document.getElementById('log-filter-type');
  const btnRefresh = document.getElementById('btn-refresh-logs');
  const btnPrev = document.getElementById('btn-prev-page');
  const btnNext = document.getElementById('btn-next-page');

  btnRefresh.addEventListener('click', () => {
    loadStats();
    loadLogs(currentPage);
  });

  filterSelect.addEventListener('change', () => {
    currentPage = 0;
    loadLogs(0);
  });

  btnPrev.addEventListener('click', () => {
    if (currentPage > 0) {
      currentPage--;
      loadLogs(currentPage);
    }
  });

  btnNext.addEventListener('click', () => {
    currentPage++;
    loadLogs(currentPage);
  });
}

async function loadStats() {
  try {
    const res = await fetch('/api/logs/stats');
    const json = await res.json();
    if (json.success && json.data) {
      const data = json.data;
      document.getElementById('stat-total').textContent = data.totalRecognitions || 0;

      const avgConf = formatConfidence(data.averageConfidence);
      document.getElementById('stat-avg-conf').textContent = avgConf + '%';

      if (data.topDetectedSigns && Object.keys(data.topDetectedSigns).length > 0) {
        const topSignEntry = Object.entries(data.topDetectedSigns)[0];
        document.getElementById('stat-top-sign').textContent = topSignEntry[0];
        document.getElementById('stat-top-count').textContent = topSignEntry[1] + ' lượt phát hiện';
      } else {
        document.getElementById('stat-top-sign').textContent = 'Chưa có';
        document.getElementById('stat-top-count').textContent = '0 lượt phát hiện';
      }
    }
  } catch (err) {
    console.error('Error fetching stats:', err);
  }
}

async function loadLogs(page) {
  const tbody = document.getElementById('logs-tbody');
  const filterType = document.getElementById('log-filter-type').value;

  let url = `/api/logs?page=${page}&size=10`;
  if (filterType) url += `&inputType=${filterType}`;

  try {
    const res = await fetch(url);
    const json = await res.json();
    if (!json.success || !json.data) return;

    const pageData = json.data;
    const content = pageData.content || [];
    tbody.innerHTML = '';

    if (content.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-center">Chưa có bản ghi nhận diện nào.</td></tr>';
    } else {
      content.forEach(log => {
        const tr = document.createElement('tr');
        const confPct = formatConfidence(log.confidenceScore);
        const timeStr = log.createdAt ? log.createdAt.replace('T', ' ').substring(0, 19) : '--';

        tr.innerHTML = `
          <td>#${log.id}</td>
          <td><span class="badge ${getInputTypeBadgeClass(log.inputType)}">${log.inputType}</span></td>
          <td><strong>${log.detectedSignVi || '--'}</strong></td>
          <td>${log.detectedSignEn || '--'}</td>
          <td><span class="badge ${getCategoryBadgeClass(log.category)}">${log.category || 'Khác'}</span></td>
          <td class="text-success font-weight-bold">${confPct}%</td>
          <td>${log.inferenceTimeMs ? log.inferenceTimeMs + ' ms' : '--'}</td>
          <td style="color: var(--text-muted); font-size: 0.8rem;">${timeStr}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Pagination update
    const totalPages = (pageData.page && pageData.page.totalPages) ? pageData.page.totalPages : 1;
    document.getElementById('page-indicator').textContent = `Trang ${page + 1} / ${Math.max(totalPages, 1)}`;
    document.getElementById('btn-prev-page').disabled = (page <= 0);
    document.getElementById('btn-next-page').disabled = (page >= totalPages - 1);

  } catch (err) {
    console.error('Error fetching logs:', err);
    tbody.innerHTML = '<tr><td colspan="8" class="text-center text-danger">Lỗi khi tải nhật ký nhận diện.</td></tr>';
  }
}

// ==========================================
// 6. CLASS CATALOG
// ==========================================
let cachedClasses = null;

function initClassCatalog() {
  loadClasses();
}

async function loadClasses() {
  const grid = document.getElementById('classes-grid');
  const searchInput = document.getElementById('class-search-input');

  if (cachedClasses) {
    renderClassCards(cachedClasses);
    return;
  }

  try {
    const res = await fetch('/api/recognize/classes');
    const json = await res.json();
    if (json.success && json.data) {
      cachedClasses = Object.values(json.data);
      const tabBtn = document.querySelector('[data-tab="tab-classes"]');
      if (tabBtn) tabBtn.textContent = `Danh mục ${cachedClasses.length} Biển báo`;
      const title = document.getElementById('classes-card-title');
      if (title) title.textContent = `Danh mục ${cachedClasses.length} Biển báo Giao thông Việt Nam`;
      renderClassCards(cachedClasses);
    }
  } catch (err) {
    console.error('Error fetching classes:', err);
    grid.innerHTML = '<p class="text-center">Không thể tải danh mục biển báo.</p>';
  }

  searchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase().trim();
    if (!cachedClasses) return;
    const filtered = cachedClasses.filter(c => 
      (c.nameVi && c.nameVi.toLowerCase().includes(term)) ||
      (c.nameEn && c.nameEn.toLowerCase().includes(term)) ||
      (c.category && c.category.toLowerCase().includes(term))
    );
    renderClassCards(filtered);
  });
}

function renderClassCards(classes) {
  const grid = document.getElementById('classes-grid');
  grid.innerHTML = '';

  classes.forEach(c => {
    const card = document.createElement('div');
    card.className = 'class-card';
    card.innerHTML = `
      <div class="class-header">
        <span class="class-id-badge">ID: ${c.id}</span>
        <span class="badge ${getCategoryBadgeClass(c.category)}">${c.category || 'Biển báo'}</span>
      </div>
      <div class="class-name-vi">${c.nameVi}</div>
      <div class="class-name-en">${c.nameEn}</div>
    `;
    grid.appendChild(card);
  });
}

// ==========================================
// HELPERS
// ==========================================
function getCategoryBadgeClass(category) {
  if (!category) return 'badge-id';
  const cat = category.toLowerCase();
  if (cat.includes('danger') || cat.includes('nguy')) return 'badge-danger';
  if (cat.includes('prohibitory') || cat.includes('cấm') || cat.includes('speed')) return 'badge-warning';
  if (cat.includes('mandatory') || cat.includes('hiệu lệnh')) return 'badge-info';
  return 'badge-id';
}

function getInputTypeBadgeClass(inputType) {
  if (inputType === 'IMAGE') return 'badge-info';
  if (inputType === 'VIDEO') return 'badge-warning';
  if (inputType === 'REALTIME') return 'badge-danger';
  return 'badge-id';
}

function setLoading(button, isLoading) {
  button.disabled = isLoading;
  const text = button.querySelector('.btn-text');
  const spinner = button.querySelector('.spinner');
  if (isLoading) {
    if (text) text.classList.add('hidden');
    if (spinner) spinner.classList.remove('hidden');
  } else {
    if (text) text.classList.remove('hidden');
    if (spinner) spinner.classList.add('hidden');
  }
}
