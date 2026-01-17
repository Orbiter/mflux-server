/**
 * mflux-server Modern Web Client
 * Handles API interactions and UI state management.
 */

class MfluxClient {
    constructor() {
        this.serverUrl = localStorage.getItem('mflux-server') || window.location.origin;
        this.models = [];
        this.isGenerating = false;
        this.initImage = null;

        this.init();
    }

    async init() {
        this.cacheDOM();
        this.bindEvents();
        this.restoreSettings();
        await this.fetchModels();
        this.updateAspectRatio();
    }

    cacheDOM() {
        this.dom = {
            // Inputs
            prompt: document.getElementById('prompt'),
            modelSelect: document.getElementById('model-select'),
            width: document.getElementById('width'),
            height: document.getElementById('height'),
            steps: document.getElementById('steps'),
            seed: document.getElementById('seed'),
            count: document.getElementById('count'),
            quality: document.getElementById('quality'),
            mfluxServer: document.getElementById('mflux-server'),

            // Value Displays
            widthVal: document.getElementById('width-value'),
            heightVal: document.getElementById('height-value'),
            stepsVal: document.getElementById('steps-value'),
            countVal: document.getElementById('count-value'),
            qualityVal: document.getElementById('quality-value'),
            aspectRatio: document.getElementById('aspect-ratio'),

            // Init Image
            fileInput: document.getElementById('init-image'),
            fileStatus: document.getElementById('init-image-status'),
            uploadBtn: document.getElementById('upload-btn'),
            deleteImgBtn: document.getElementById('delete-img-btn'),

            // Actions & Output
            generateBtn: document.getElementById('generate-btn'),
            gallery: document.getElementById('image-gallery'),
            statusBar: document.getElementById('status-bar'),
            statusText: document.getElementById('status-text'),
            emptyState: document.getElementById('empty-state')
        };
    }

    bindEvents() {
        // Generate
        this.dom.generateBtn.addEventListener('click', () => this.generate());

        // Sliders (Update value display & aspect ratio)
        ['width', 'height', 'steps', 'count', 'quality'].forEach(id => {
            this.dom[id].addEventListener('input', (e) => {
                this.dom[id + 'Val'].textContent = e.target.value;
                if (id === 'width' || id === 'height') this.updateAspectRatio();
            });
        });

        // Server URL
        this.dom.mfluxServer.addEventListener('input', (e) => {
            this.serverUrl = e.target.value;
            localStorage.setItem('mflux-server', this.serverUrl);
        });

        this.dom.mfluxServer.value = this.serverUrl;

        // Init Image
        this.dom.uploadBtn.addEventListener('click', () => this.dom.fileInput.click());

        this.dom.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.initImage = e.target.files[0];
                this.dom.fileStatus.textContent = this.initImage.name;
                this.dom.deleteImgBtn.style.display = 'inline-block';
            }
        });

        this.dom.deleteImgBtn.addEventListener('click', () => {
            this.initImage = null;
            this.dom.fileInput.value = '';
            this.dom.fileStatus.textContent = 'No image selected';
            this.dom.deleteImgBtn.style.display = 'none';
        });
    }

    restoreSettings() {
        // Could restore other settings from localStorage here if desired
    }

    async fetchModels() {
        try {
            const response = await fetch(`${this.serverUrl}/v1/models`);
            if (!response.ok) throw new Error('Failed to fetch models');

            const data = await response.json();
            this.dom.modelSelect.innerHTML = '';

            (data.data || []).forEach(model => {
                const opt = document.createElement('option');
                opt.value = model.id;
                opt.textContent = model.id;
                this.dom.modelSelect.appendChild(opt);
            });

            // Try to set current loaded model from health check
            this.checkHealth();
        } catch (err) {
            console.error('Model fetch error:', err);
            this.showStatus('Error connecting to server', true);
        }
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.serverUrl}/health`);
            const data = await response.json();
            if (data.model) {
                this.dom.modelSelect.value = data.model;
            }
        } catch (e) { /* ignore */ }
    }

    updateAspectRatio() {
        const w = parseInt(this.dom.width.value);
        const h = parseInt(this.dom.height.value);
        const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
        const divisor = gcd(w, h);
        const ratio = `${w / divisor}:${h / divisor}`;

        this.dom.aspectRatio.textContent = ratio;

        // Highlight common ratios
        const common = ["1:1", "5:4", "4:3", "3:2", "16:9", "2:1", "9:16"];
        this.dom.aspectRatio.style.color = common.includes(ratio) ? 'var(--text-primary)' : 'var(--text-muted)';
    }

    async generate() {
        if (this.isGenerating) return;

        const prompt = this.dom.prompt.value.trim();
        if (!prompt) {
            this.showStatus('Please enter a prompt', true);
            return;
        }

        this.setGenerating(true);
        this.dom.emptyState.style.display = 'none';

        const count = parseInt(this.dom.count.value);
        const total = count;

        for (let i = 0; i < count; i++) {
            this.showStatus(`Generating image ${i + 1} of ${total}...`);

            try {
                const imageUrl = await this.callApi();
                this.addImageToGallery(imageUrl);
            } catch (err) {
                this.showStatus(`Error: ${err.message}`, true);
                break;
            }
        }

        if (this.isGenerating) { // If not aborted by error
            this.showStatus('Generation complete');
            setTimeout(() => this.hideStatus(), 3000);
        }

        this.setGenerating(false);
    }

    async callApi() {
        let endpoint = `${this.serverUrl}/v1/images/generations`;
        let body;
        let headers = {};

        const params = {
            prompt: this.dom.prompt.value,
            model: this.dom.modelSelect.value,
            n: 1,
            size: `${this.dom.width.value}x${this.dom.height.value}`,
            response_format: "b64_json",
            steps: parseInt(this.dom.steps.value)
        };

        const seed = this.dom.seed.value.trim();
        if (seed) params.seed = parseInt(seed);

        if (this.initImage) {
            endpoint = `${this.serverUrl}/v1/images/edits`;
            const formData = new FormData();
            formData.append('image', this.initImage);
            Object.keys(params).forEach(key => formData.append(key, params[key]));
            body = formData;
        } else {
            headers['Content-Type'] = 'application/json';
            body = JSON.stringify(params);
        }

        const response = await fetch(endpoint, { method: 'POST', headers, body });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error?.message || 'Generation failed');
        }

        const data = await response.json();
        return `data:image/png;base64,${data.data[0].b64_json}`;
    }

    addImageToGallery(url) {
        const div = document.createElement('div');
        div.className = 'image-card';

        const img = document.createElement('img');
        img.src = url;
        img.onclick = () => {
            const win = window.open();
            if (win) {
                win.document.write(`
                    <html>
                        <head><title>Generated Image</title></head>
                        <body style="margin:0; display:flex; justify-content:center; align-items:center; background:#0f0f10; height:100vh;">
                            <img src="${url}" style="max-width:100%; max-height:100%; object-fit:contain;">
                        </body>
                    </html>
                `);
                win.document.close(); // Important for browser to finish loading
            } else {
                console.error("Popup blocked");
                window.location.href = url;
            }
        };

        const actions = document.createElement('div');
        actions.className = 'image-actions';

        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'action-btn';
        downloadBtn.textContent = 'Download';
        downloadBtn.onclick = (e) => {
            e.stopPropagation();
            const a = document.createElement('a');
            a.href = url;
            a.download = `flux-${Date.now()}.png`;
            a.click();
        };

        actions.appendChild(downloadBtn);
        div.appendChild(img);
        div.appendChild(actions);

        this.dom.gallery.insertBefore(div, this.dom.gallery.firstChild);
    }

    setGenerating(bool) {
        this.isGenerating = bool;
        this.dom.generateBtn.disabled = bool;
        this.dom.generateBtn.textContent = bool ? 'Generating...' : 'Generate';
        this.dom.generateBtn.style.opacity = bool ? '0.7' : '1';
    }

    showStatus(msg, isError = false) {
        this.dom.statusText.textContent = msg;
        this.dom.statusText.style.color = isError ? '#ff6b6b' : '#fff';
        this.dom.statusBar.classList.add('active');

        // Show/hide spinner based on error
        const spinner = this.dom.statusBar.querySelector('.spinner');
        if (spinner) spinner.style.display = isError ? 'none' : 'block';
    }

    hideStatus() {
        this.dom.statusBar.classList.remove('active');
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    new MfluxClient();
});
