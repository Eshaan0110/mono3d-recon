/**
 * Application controller — connects UI elements to the 3D viewer
 * and handles server communication for reconstruction jobs.
 */

(function () {
    'use strict';

    const viewer = new SceneViewer('canvas');
    let selectedFile = null;
    let currentJobId = null;
    let pollTimer = null;

    // ─── Element refs ───
    const $uploadModal   = document.getElementById('upload-modal');
    const $dropZone      = document.getElementById('drop-zone');
    const $fileInput     = document.getElementById('file-input');
    const $startBtn      = document.getElementById('btn-start-recon');
    const $qualitySelect = document.getElementById('quality-select');
    const $processing    = document.getElementById('processing-overlay');
    const $progressFill  = document.getElementById('progress-fill');
    const $progressMsg   = document.getElementById('progress-message');
    const $emptyState    = document.getElementById('empty-state');
    const $controlsPanel = document.getElementById('controls-panel');
    const $statsDisplay  = document.getElementById('stats-display');

    // ─── Header buttons ───
    document.getElementById('btn-upload').addEventListener('click', () => {
        $uploadModal.classList.remove('hidden');
    });

    document.getElementById('modal-close').addEventListener('click', () => {
        $uploadModal.classList.add('hidden');
    });

    document.getElementById('btn-load-file').addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.glb,.gltf,.ply';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            try {
                await viewer.loadFile(file);
                $emptyState.classList.add('hidden');
                $controlsPanel.classList.remove('hidden');
                showStats({ vertices: '—', triangles: '—', format: file.name.split('.').pop().toUpperCase() });
            } catch (err) {
                alert('Failed to load file: ' + err.message);
            }
        };
        input.click();
    });

    // ─── Drop zone ───
    $dropZone.addEventListener('click', () => $fileInput.click());
    $dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        $dropZone.classList.add('dragover');
    });
    $dropZone.addEventListener('dragleave', () => {
        $dropZone.classList.remove('dragover');
    });
    $dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        $dropZone.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });
    $fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        if (!file) return;
        selectedFile = file;
        $dropZone.querySelector('.drop-zone-content').innerHTML = `
            <div class="drop-icon">✓</div>
            <p>${file.name}</p>
            <p class="drop-hint">${(file.size / 1048576).toFixed(1)} MB</p>
        `;
        $startBtn.disabled = false;
    }

    // ─── Start reconstruction ───
    $startBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        $uploadModal.classList.add('hidden');
        $processing.classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', selectedFile);
        formData.append('quality', $qualitySelect.value);

        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) {
                alert('Upload failed: ' + data.error);
                $processing.classList.add('hidden');
                return;
            }

            currentJobId = data.job_id;
            startPolling();
        } catch (err) {
            alert('Upload error: ' + err.message);
            $processing.classList.add('hidden');
        }
    });

    // ─── Polling ───
    const STAGES = ['extract', 'depth', 'features', 'poses', 'pointcloud', 'mesh'];

    function startPolling() {
        pollTimer = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${currentJobId}`);
                const data = await res.json();
                updateProgress(data);

                if (data.status === 'complete') {
                    clearInterval(pollTimer);
                    await onReconstructionComplete(data);
                } else if (data.status === 'error') {
                    clearInterval(pollTimer);
                    alert('Reconstruction failed: ' + data.error);
                    $processing.classList.add('hidden');
                }
            } catch (err) {
                console.error('Poll error:', err);
            }
        }, 1000);
    }

    function updateProgress(data) {
        const stageEls = document.querySelectorAll('.stage');
        const currentIdx = STAGES.indexOf(data.stage);

        stageEls.forEach((el, i) => {
            el.classList.remove('active', 'complete', 'error');
            if (i < currentIdx) el.classList.add('complete');
            else if (i === currentIdx) el.classList.add('active');
        });

        const overallProgress = Math.max(0, ((currentIdx + data.progress) / STAGES.length) * 100);
        $progressFill.style.width = overallProgress + '%';
        $progressMsg.textContent = data.message || '';
    }

    async function onReconstructionComplete(data) {
        $progressFill.style.width = '100%';
        $progressMsg.textContent = 'Loading 3D model...';

        try {
            const glbUrl = `/api/model/${currentJobId}/scene.glb`;
            await viewer.loadGLB(glbUrl);

            $processing.classList.add('hidden');
            $emptyState.classList.add('hidden');
            $controlsPanel.classList.remove('hidden');

            if (data.stats) {
                showStats(data.stats);
            }
        } catch (err) {
            // Fall back to PLY
            try {
                const plyUrl = `/api/model/${currentJobId}/scene.ply`;
                await viewer.loadPLY(plyUrl);
                $processing.classList.add('hidden');
                $emptyState.classList.add('hidden');
                $controlsPanel.classList.remove('hidden');
            } catch (err2) {
                alert('Failed to load reconstructed model');
                $processing.classList.add('hidden');
            }
        }
    }

    function showStats(stats) {
        let html = '';
        if (stats.mesh) {
            html += `<span class="stat-label">Vertices:</span> <span class="stat-value">${stats.mesh.num_vertices?.toLocaleString() || '—'}</span><br>`;
            html += `<span class="stat-label">Triangles:</span> <span class="stat-value">${stats.mesh.num_triangles?.toLocaleString() || '—'}</span><br>`;
            html += `<span class="stat-label">Watertight:</span> <span class="stat-value">${stats.mesh.is_watertight ? 'Yes' : 'No'}</span><br>`;
        }
        if (stats.num_frames) {
            html += `<span class="stat-label">Frames:</span> <span class="stat-value">${stats.num_frames}</span><br>`;
        }
        if (stats.pointcloud_size) {
            html += `<span class="stat-label">Points:</span> <span class="stat-value">${stats.pointcloud_size?.toLocaleString()}</span><br>`;
        }
        if (stats.elapsed_seconds) {
            html += `<span class="stat-label">Time:</span> <span class="stat-value">${stats.elapsed_seconds}s</span>`;
        }
        if (stats.format) {
            html += `<span class="stat-label">Format:</span> <span class="stat-value">${stats.format}</span>`;
        }
        $statsDisplay.innerHTML = html;
    }

    // ─── Control panel ───

    // Display mode buttons
    document.querySelectorAll('[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-mode]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            viewer.setDisplayMode(btn.dataset.mode);
        });
    });

    // Background buttons
    document.querySelectorAll('[data-bg]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('[data-bg]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            viewer.setBackground(btn.dataset.bg);
        });
    });

    // Point size slider
    const $pointSize = document.getElementById('point-size');
    const $pointSizeVal = document.getElementById('point-size-value');
    $pointSize.addEventListener('input', () => {
        const val = parseFloat($pointSize.value);
        $pointSizeVal.textContent = val.toFixed(1);
        viewer.setPointSize(val);
    });

    // Trajectory toggle
    document.getElementById('show-trajectory').addEventListener('change', (e) => {
        viewer.toggleTrajectory(e.target.checked);
    });

    // Reset camera
    document.getElementById('btn-reset-camera').addEventListener('click', () => {
        viewer.resetCamera();
    });

    // Screenshot
    document.getElementById('btn-screenshot').addEventListener('click', () => {
        viewer.takeScreenshot();
    });

    // Download GLB
    document.getElementById('btn-download').addEventListener('click', () => {
        if (!currentJobId) {
            alert('No reconstruction to download');
            return;
        }
        const link = document.createElement('a');
        link.href = `/api/model/${currentJobId}/scene.glb`;
        link.download = 'scene.glb';
        link.click();
    });

})();
