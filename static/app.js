// Frontend JavaScript for AI Equity Research Platform
class AIEquityResearchApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.websocket = null;
        this.isConnected = false;
        this.messageHistory = [];
        this.isLoading = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadMarketData();
        this.autoResizeTextarea();
        
        // Load chat history from localStorage
        this.loadChatHistory();
        
        // Update market data every 5 minutes
        setInterval(() => this.loadMarketData(), 5 * 60 * 1000);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupEventListeners() {
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const llmProvider = document.getElementById('llm-provider');
        
        if (chatInput) {
            chatInput.addEventListener('input', this.autoResizeTextarea);
            chatInput.addEventListener('keydown', (e) => this.handleKeyPress(e));
        }
        
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        if (llmProvider) {
            llmProvider.addEventListener('change', () => this.handleLLMProviderChange());
        }
        
        // Handle window resize
        window.addEventListener('resize', this.handleWindowResize.bind(this));
    }
    
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.isConnected = false;
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIndicator = document.querySelector('.status-indicator span:last-child');
        const statusDot = document.querySelector('.status-dot');
        
        if (statusIndicator && statusDot) {
            if (connected) {
                statusIndicator.textContent = 'System Active';
                statusDot.style.background = 'var(--success-color)';
            } else {
                statusIndicator.textContent = 'Reconnecting...';
                statusDot.style.background = 'var(--warning-color)';
            }
        }
    }
    
    handleWebSocketMessage(data) {
        if (data.response) {
            this.hideTypingIndicator();
            this.addMessage(data.response, 'assistant', data.timestamp);
            this.setLoading(false);
        }
    }
    
    async sendMessage(message = null) {
        const chatInput = document.getElementById('chat-input');
        const messageText = message || chatInput.value.trim();
        
        if (!messageText || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage(messageText, 'user');
        
        // Clear input
        if (!message) {
            chatInput.value = '';
            this.autoResizeTextarea();
        }
        
        // Set loading state
        this.setLoading(true);
        this.showTypingIndicator();
        
        try {
            if (this.isConnected && this.websocket) {
                // Send via WebSocket
                this.websocket.send(JSON.stringify({
                    message: messageText,
                    session_id: this.sessionId,
                    llm_provider: document.getElementById('llm-provider').value
                }));
            } else {
                // Fallback to HTTP API
                await this.sendMessageHTTP(messageText);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
            this.setLoading(false);
        }
    }
    
    async sendMessageHTTP(messageText) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: messageText,
                    session_id: this.sessionId,
                    llm_provider: document.getElementById('llm-provider').value
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            this.hideTypingIndicator();
            this.addMessage(data.response, 'assistant', data.timestamp);
            this.setLoading(false);
            
        } catch (error) {
            console.error('HTTP API error:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
            this.setLoading(false);
        }
    }
    
    addMessage(text, sender, timestamp = null) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = `message-avatar ${sender}-avatar`;
        
        if (sender === 'user') {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.innerHTML = this.formatMessageText(text);
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = timestamp ? this.formatTime(timestamp) : this.formatTime(new Date().toISOString());
        
        contentDiv.appendChild(textDiv);
        contentDiv.appendChild(timeDiv);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        messagesContainer.appendChild(messageDiv);
        
        // Store in history
        this.messageHistory.push({
            text: text,
            sender: sender,
            timestamp: timestamp || new Date().toISOString()
        });
        
        // Save to localStorage
        this.saveChatHistory();
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    formatMessageText(text) {
        // Convert markdown-style formatting to HTML
        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        // Format financial data tables if present
        if (text.includes('Sector') && text.includes('%')) {
            formatted = this.formatFinancialData(formatted);
        }
        
        return formatted;
    }
    
    formatFinancialData(text) {
        // This would format tabular financial data into HTML tables
        // For now, return as-is with some basic formatting
        return text.replace(/(\+?\-?\d+\.\d+%)/g, '<span class="metric-value">$1</span>');
    }
    
    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
    }
    
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-avatar assistant-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>Analyzing...</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        const sendBtn = document.getElementById('send-btn');
        const chatInput = document.getElementById('chat-input');
        
        if (sendBtn) {
            sendBtn.disabled = loading;
            sendBtn.innerHTML = loading ? 
                '<div class="spinner"></div>' : 
                '<i class="fas fa-paper-plane"></i>';
        }
        
        if (chatInput) {
            chatInput.disabled = loading;
        }
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
    
    autoResizeTextarea() {
        const textarea = document.getElementById('chat-input');
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    handleLLMProviderChange() {
        const provider = document.getElementById('llm-provider').value;
        console.log('LLM Provider changed to:', provider);
        
        // Show a brief notification
        this.showNotification(`Switched to ${provider.toUpperCase()}`, 'info');
    }
    
    showNotification(message, type = 'info') {
        // Create and show notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: var(--primary-blue);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px var(--shadow-medium);
            z-index: 1001;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    async loadMarketData() {
        try {
            // Load sector data
            const sectorResponse = await fetch('/api/sector-data');
            if (sectorResponse.ok) {
                const sectorData = await sectorResponse.json();
                this.updateSectorMetrics(sectorData.data);
            }
            
            // Load economic indicators
            const economicResponse = await fetch('/api/economic-indicators');
            if (economicResponse.ok) {
                const economicData = await economicResponse.json();
                this.updateMarketMetrics(economicData.data);
            }
            
        } catch (error) {
            console.error('Error loading market data:', error);
        }
    }
    
    updateMarketMetrics(data) {
        const metricsContainer = document.getElementById('market-metrics');
        if (!metricsContainer || !data) return;
        
        const metrics = [
            { label: 'Fed Funds Rate', value: data.Fed_Funds_Rate, format: 'percentage' },
            { label: 'VIX', value: data.VIX, format: 'number' },
            { label: '10Y Yield', value: data.Yield_10Y, format: 'percentage' }
        ];
        
        metricsContainer.innerHTML = metrics.map(metric => `
            <div class="metric-item">
                <span>${metric.label}</span>
                <span class="metric-value">
                    ${this.formatMetricValue(metric.value, metric.format)}
                </span>
            </div>
        `).join('');
    }
    
    updateSectorMetrics(data) {
        const sectorContainer = document.getElementById('sector-metrics');
        if (!sectorContainer || !data) return;
        
        // Sort by performance and take top 5
        const topSectors = data
            .sort((a, b) => b['Performance_%'] - a['Performance_%'])
            .slice(0, 5);
        
        sectorContainer.innerHTML = topSectors.map(sector => `
            <div class="metric-item">
                <span>${sector.Sector.replace(' Services', '')}</span>
                <span class="metric-value ${sector['Performance_%'] >= 0 ? 'positive' : 'negative'}">
                    ${sector['Performance_%'] >= 0 ? '+' : ''}${sector['Performance_%']}%
                </span>
            </div>
        `).join('');
    }
    
    formatMetricValue(value, format) {
        if (value === null || value === undefined) return 'N/A';
        
        switch (format) {
            case 'percentage':
                return `${value.toFixed(2)}%`;
            case 'number':
                return value.toFixed(2);
            default:
                return value.toString();
        }
    }
    
    async generateFullReport() {
        const reportBtn = document.getElementById('report-btn');
        if (!reportBtn || this.isLoading) return;
        
        // Set loading state
        reportBtn.disabled = true;
        reportBtn.innerHTML = '<div class="spinner"></div> Generating...';
        
        try {
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    llm_provider: document.getElementById('llm-provider').value,
                    include_sentiment: true,
                    include_options: true
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add report to chat
            this.addMessage(
                `📊 **Comprehensive Research Report Generated**\n\n${data.report}`,
                'assistant'
            );
            
            this.showNotification('Research report generated successfully!', 'success');
            
        } catch (error) {
            console.error('Error generating report:', error);
            this.addMessage(
                'Sorry, I encountered an error generating the research report. Please try again later.',
                'assistant'
            );
            this.showNotification('Error generating report. Please try again.', 'error');
        } finally {
            // Reset button state
            reportBtn.disabled = false;
            reportBtn.innerHTML = '<i class="fas fa-file-alt"></i> Generate Full Report';
        }
    }
    
    startNewChat() {
        // Clear current chat
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            // Keep only the welcome message
            const welcomeMessage = messagesContainer.querySelector('.message');
            messagesContainer.innerHTML = '';
            if (welcomeMessage) {
                messagesContainer.appendChild(welcomeMessage);
            }
        }
        
        // Generate new session ID
        this.sessionId = this.generateSessionId();
        this.messageHistory = [];
        
        // Clear localStorage
        localStorage.removeItem(`chat_history_${this.sessionId}`);
        
        // Reconnect WebSocket with new session
        if (this.websocket) {
            this.websocket.close();
        }
        setTimeout(() => this.connectWebSocket(), 100);
        
        this.showNotification('Started new analysis session', 'info');
    }
    
    sendQuickMessage(message) {
        this.sendMessage(message);
    }
    
    saveChatHistory() {
        try {
            localStorage.setItem(`chat_history_${this.sessionId}`, JSON.stringify(this.messageHistory));
        } catch (error) {
            console.error('Error saving chat history:', error);
        }
    }
    
    loadChatHistory() {
        try {
            const history = localStorage.getItem(`chat_history_${this.sessionId}`);
            if (history) {
                this.messageHistory = JSON.parse(history);
                // Optionally restore messages to UI
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    handleWindowResize() {
        // Adjust layout for mobile
        if (window.innerWidth <= 768) {
            document.body.classList.add('mobile');
        } else {
            document.body.classList.remove('mobile');
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiApp = new AIEquityResearchApp();
});

// Global functions for HTML onclick handlers
function sendMessage() {
    window.aiApp?.sendMessage();
}

function generateFullReport() {
    window.aiApp?.generateFullReport();
}

function startNewChat() {
    window.aiApp?.startNewChat();
}

function sendQuickMessage(message) {
    window.aiApp?.sendQuickMessage(message);
}

function handleKeyPress(event) {
    window.aiApp?.handleKeyPress(event);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);