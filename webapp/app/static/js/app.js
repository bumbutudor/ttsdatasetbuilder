/**
 * TTS/STT Dataset Builder - Frontend JavaScript
 */

// ============== API Helper ==============
window.api = {
    token: localStorage.getItem('token'),
    
    async request(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        const response = await fetch(endpoint, {
            ...options,
            headers
        });
        
        if (response.status === 401) {
            this.logout();
            window.location.href = '/login';
            throw new Error('Unauthorized');
        }
        
        return response;
    },
    
    async get(endpoint) {
        return this.request(endpoint);
    },
    
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    },
    
    async uploadFiles(endpoint, files, extraData = {}) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        
        Object.keys(extraData).forEach(key => {
            formData.append(key, extraData[key]);
        });
        
        const headers = {};
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        return fetch(endpoint, {
            method: 'POST',
            headers,
            body: formData
        });
    },
    
    setToken(token) {
        this.token = token;
        localStorage.setItem('token', token);
    },
    
    logout() {
        this.token = null;
        localStorage.removeItem('token');
        localStorage.removeItem('currentProjectId');
    },
    
    isAuthenticated() {
        return !!this.token;
    }
};

// ============== UI Helpers ==============
window.ui = {
    showAlert(message, type = 'success') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} fade-in`;
        alertDiv.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                ${type === 'success' 
                    ? '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>'
                    : type === 'error'
                    ? '<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>'
                    : '<path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>'}
            </svg>
            <span>${message}</span>
        `;
        
        const container = document.querySelector('.alert-container') || document.body;
        container.prepend(alertDiv);
        
        setTimeout(() => alertDiv.remove(), 5000);
    },
    
    showModal(id) {
        const modal = document.getElementById(id);
        console.log('showModal called with id:', id, 'element found:', modal);
        if (modal) {
            modal.classList.add('active');
            console.log('Modal classes after add:', modal.classList.toString());
        } else {
            console.error('Modal element not found:', id);
        }
    },
    
    hideModal(id) {
        const modal = document.getElementById(id);
        if (modal) {
            modal.classList.remove('active');
        }
    },
    
    showLoading(button) {
        const originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Loading...';
        return () => {
            button.disabled = false;
            button.innerHTML = originalText;
        };
    }
};

// ============== Project Selector ==============
function getCurrentProjectId() {
    return localStorage.getItem('currentProjectId');
}

function setCurrentProjectId(projectId) {
    localStorage.setItem('currentProjectId', projectId);
}

window.loadProjectSelector = async function() {
    const selector = document.getElementById('project-selector');
    if (!selector) return;
    
    try {
        const response = await api.get('/api/projects');
        if (!response.ok) throw new Error('Failed to load projects');
        
        const data = await response.json();
        
        selector.innerHTML = '<option value="">Select a project...</option>' +
            data.projects.map(p => 
                `<option value="${p.id}" ${getCurrentProjectId() == p.id ? 'selected' : ''}>
                    ${escapeHtml(p.name)} (${p.dataset_type})
                </option>`
            ).join('');
        
        // If there's a saved project, trigger load
        const currentId = getCurrentProjectId();
        if (currentId && data.projects.some(p => p.id == currentId)) {
            // Project page-specific loaders will pick this up
            if (typeof loadUploadedFiles === 'function') loadUploadedFiles();
            if (typeof loadFilesForProcessing === 'function') loadFilesForProcessing();
            if (typeof loadVideosForProcessing === 'function') loadVideosForProcessing();
            if (typeof loadDatasetPreview === 'function') loadDatasetPreview();
            if (typeof loadSentences === 'function') loadSentences();
            if (typeof loadDatasetStats === 'function') loadDatasetStats();
            if (typeof loadMissingFiles === 'function') loadMissingFiles();
            if (typeof loadSettings === 'function') loadSettings();
            if (typeof loadDataset === 'function') loadDataset();
        }
        
    } catch (error) {
        console.error('Error loading projects:', error);
        selector.innerHTML = '<option value="">Error loading projects</option>';
    }
}

window.selectProject = async function(projectId) {
    if (!projectId) return;
    
    setCurrentProjectId(projectId);
    
    // Reload page-specific data
    if (typeof loadUploadedFiles === 'function') loadUploadedFiles();
    if (typeof loadFilesForProcessing === 'function') loadFilesForProcessing();
    if (typeof loadVideosForProcessing === 'function') loadVideosForProcessing();
    if (typeof loadDatasetPreview === 'function') loadDatasetPreview();
    if (typeof loadSentences === 'function') loadSentences();
    if (typeof loadDatasetStats === 'function') loadDatasetStats();
    if (typeof loadMissingFiles === 'function') loadMissingFiles();
    if (typeof loadSettings === 'function') loadSettings();
    if (typeof loadDataset === 'function') loadDataset();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============== Auth Functions ==============
async function handleLogin(event) {
    event.preventDefault();
    
    const form = event.target;
    const button = form.querySelector('button[type="submit"]');
    const stopLoading = ui.showLoading(button);
    
    try {
        const formData = new FormData(form);
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Login failed');
        }
        
        const data = await response.json();
        api.setToken(data.access_token);
        window.location.href = '/dashboard';
    } catch (error) {
        ui.showAlert(error.message, 'error');
    } finally {
        stopLoading();
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const form = event.target;
    const button = form.querySelector('button[type="submit"]');
    const stopLoading = ui.showLoading(button);
    
    const password = form.password.value;
    const confirmPassword = form.confirm_password.value;
    
    if (password !== confirmPassword) {
        ui.showAlert('Passwords do not match', 'error');
        stopLoading();
        return;
    }
    
    try {
        const response = await api.post('/api/auth/register', {
            username: form.username.value,
            email: form.email.value,
            password: password,
            full_name: form.full_name.value
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }
        
        ui.showAlert('Registration successful! Please login.');
        setTimeout(() => window.location.href = '/login', 1500);
    } catch (error) {
        ui.showAlert(error.message, 'error');
    } finally {
        stopLoading();
    }
}

function logout() {
    api.logout();
    window.location.href = '/login';
}

// ============== File Upload ==============
function setupUploadZone(zoneId, inputId, onFiles) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    
    if (!zone || !input) return;
    
    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        onFiles(Array.from(e.dataTransfer.files));
    });
    
    input.addEventListener('change', () => {
        onFiles(Array.from(input.files));
        input.value = ''; // Reset to allow re-upload of same file
    });
}

// ============== Job Polling ==============
async function pollJobStatus(jobId, onComplete, onError) {
    const poll = async () => {
        try {
            const response = await api.get(`/api/jobs/${jobId}`);
            if (!response.ok) return;
            
            const job = await response.json();
            
            if (job.status === 'completed') {
                if (onComplete) onComplete(job);
                return;
            }
            
            if (job.status === 'failed') {
                if (onError) onError(job.error_message || 'Job failed');
                return;
            }
            
            // Continue polling
            setTimeout(poll, 2000);
        } catch (error) {
            console.error('Poll error:', error);
            if (onError) onError(error.message);
        }
    };
    
    poll();
}

// ============== Utility Functions ==============
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}
