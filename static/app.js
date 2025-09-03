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

    this.loadChatHistory();

    setInterval(() => this.loadMarketData(), 5 * 60 * 1000);
  }

  generateSessionId() {
    return "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
  }

  setupEventListeners() {
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const llmProvider = document.getElementById("llm-provider");

    if (chatInput) {
      chatInput.addEventListener("input", () => this.autoResizeTextarea());
      chatInput.addEventListener("keydown", (e) => this.handleKeyPress(e));
    }

    if (sendBtn) {
      sendBtn.addEventListener("click", () => this.sendMessage());
    }

    if (llmProvider) {
      llmProvider.addEventListener("change", () => this.handleLLMProviderChange());
    }

    window.addEventListener("resize", this.handleWindowResize.bind(this));
  }

  connectWebSocket() {
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        this.isConnected = true;
        this.updateConnectionStatus(true);
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      };

      this.websocket.onclose = () => {
        this.isConnected = false;
        this.updateConnectionStatus(false);
        setTimeout(() => this.connectWebSocket(), 3000);
      };

      this.websocket.onerror = () => {
        this.isConnected = false;
        this.updateConnectionStatus(false);
      };
    } catch {
      this.isConnected = false;
      this.updateConnectionStatus(false);
    }
  }

  updateConnectionStatus(connected) {
    const statusIndicator = document.querySelector(".status-indicator span:last-child");
    const statusDot = document.querySelector(".status-dot");

    if (statusIndicator && statusDot) {
      if (connected) {
        statusIndicator.textContent = "System Active";
        statusDot.style.background = "var(--success-color)";
      } else {
        statusIndicator.textContent = "Reconnecting...";
        statusDot.style.background = "var(--warning-color)";
      }
    }
  }

  handleWebSocketMessage(data) {
    if (data.response) {
      this.hideTypingIndicator();
      this.addMessage(data.response, "assistant", data.timestamp);
      this.setLoading(false);
    }
  }

  async sendMessage(message = null) {
    const chatInput = document.getElementById("chat-input");
    const messageText = message || chatInput.value.trim();
    if (!messageText || this.isLoading) return;

    this.addMessage(messageText, "user");

    if (!message) {
      chatInput.value = "";
      this.autoResizeTextarea();
    }

    this.setLoading(true);
    this.showTypingIndicator();

    try {
      if (this.isConnected && this.websocket) {
        this.websocket.send(
          JSON.stringify({
            message: messageText,
            session_id: this.sessionId,
            llm_provider: document.getElementById("llm-provider")?.value || "openai",
          })
        );
      } else {
        await this.sendMessageHTTP(messageText);
      }
    } catch {
      this.hideTypingIndicator();
      this.addMessage("Sorry, I encountered an error processing your request. Please try again.", "assistant");
      this.setLoading(false);
    }
  }

  async sendMessageHTTP(messageText) {
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: messageText,
          session_id: this.sessionId,
          llm_provider: document.getElementById("llm-provider")?.value || "openai",
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      this.hideTypingIndicator();
      this.addMessage(data.response, "assistant", data.timestamp);
      this.setLoading(false);
    } catch (error) {
      console.error("HTTP API error:", error);
      this.hideTypingIndicator();
      this.addMessage("Sorry, I encountered an error processing your request. Please try again.", "assistant");
      this.setLoading(false);
    }
  }

  addMessage(text, sender, timestamp = null) {
    const messagesContainer = document.getElementById("chat-messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = "message";

    const avatarDiv = document.createElement("div");
    avatarDiv.className = `message-avatar ${sender}-avatar`;
    avatarDiv.innerHTML = sender === "user" ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const textDiv = document.createElement("div");
    textDiv.className = "message-text";
    textDiv.innerHTML = this.formatMessageText(text);

    const timeDiv = document.createElement("div");
    timeDiv.className = "message-time";
    timeDiv.textContent = this.formatTime(timestamp || new Date().toISOString());

    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);

    this.messageHistory.push({ text, sender, timestamp: timestamp || new Date().toISOString() });
    this.saveChatHistory();
    this.scrollToBottom();
  }

  formatMessageText(text) {
    let formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");

    if (text.includes("Sector") && text.includes("%")) {
      formatted = this.formatFinancialData(formatted);
    }
    return formatted;
  }

  formatFinancialData(text) {
    return text.replace(/(\+?\-?\d+\.\d+%)/g, '<span class="metric-value">$1</span>');
  }

  formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: true });
  }

  showTypingIndicator() {
    const messagesContainer = document.getElementById("chat-messages");
    const typingDiv = document.createElement("div");
    typingDiv.id = "typing-indicator";
    typingDiv.className = "typing-indicator";
    typingDiv.innerHTML = `
      <div class="message-avatar assistant-avatar">
        <i class="fas fa-robot"></i>
      </div>
      <div style="display:flex;align-items:center;gap:.5rem;">
        <span>Analyzing...</span>
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>`;
    messagesContainer.appendChild(typingDiv);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    document.getElementById("typing-indicator")?.remove();
  }

  setLoading(loading) {
    this.isLoading = loading;
    const sendBtn = document.getElementById("send-btn");
    const chatInput = document.getElementById("chat-input");

    if (sendBtn) {
      sendBtn.disabled = loading;
      sendBtn.innerHTML = loading ? '<div class="spinner"></div>' : '<i class="fas fa-paper-plane"></i>';
    }
    if (chatInput) chatInput.disabled = loading;
  }

  scrollToBottom() {
    const messagesContainer = document.getElementById("chat-messages");
    if (messagesContainer) messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  autoResizeTextarea() {
    const textarea = document.getElementById("chat-input");
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    }
  }

  handleKeyPress(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  handleLLMProviderChange() {
    const provider = document.getElementById("llm-provider")?.value || "openai";
    this.showNotification(`Switched to ${provider.toUpperCase()}`, "info");
  }

  showNotification(message, type = "info") {
    const notification = document.createElement("div");
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
      position: fixed; top: 100px; right: 20px; background: var(--primary-blue);
      color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 12px var(--shadow-medium);
      z-index: 1001; animation: slideInRight .3s ease-out;`;
    document.body.appendChild(notification);
    setTimeout(() => {
      notification.style.animation = "slideOutRight .3s ease-out";
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }

  async loadMarketData() {
    try {
      const sectorResponse = await fetch("/api/sector-data");
      if (sectorResponse.ok) {
        const sectorData = await sectorResponse.json();
        this.updateSectorMetrics(sectorData.data || []);
      }

      const economicResponse = await fetch("/api/economic-indicators");
      if (economicResponse.ok) {
        const econ = await economicResponse.json();
        const list = econ.data || econ.list || [];
        const kv = econ.kv || this.listToKV(list);
        this.updateMarketMetrics(kv);
      }
    } catch (err) {
      console.error("Error loading market data:", err);
    }
  }

  listToKV(list) {
    const kv = {};
    for (const row of list) {
      const key = (row.Indicator || "")
        .replace(/\s+/g, "_")
        .replace(/[()%]/g, "")
        .replace(/__/g, "_");
      kv[key] = row.Latest;
    }
    return kv;
  }

  updateMarketMetrics(kv) {
    const metricsContainer = document.getElementById("market-metrics");
    if (!metricsContainer || !kv) return;

    const fed = kv.Fed_Funds_Rate || kv["Fed_Funds_Rate_%"] || kv["Fed_Funds_Rate"];
    const tenY = kv["10Y_Treasury_Yield"] || kv["10Y_Treasury_Yield_%"] || kv["DGS10"];
    const vix = kv.VIX || kv.CBOE_Volatility_Index_VIX;

    const metrics = [
      { label: "Fed Funds Rate", value: fed, format: "percentage" },
      { label: "VIX", value: vix, format: "number" },
      { label: "10Y Yield", value: tenY, format: "percentage" },
    ];

    metricsContainer.innerHTML = metrics
      .map(
        (m) => `
        <div class="metric-item">
          <span>${m.label}</span>
          <span class="metric-value">${this.formatMetricValue(m.value, m.format)}</span>
        </div>`
      )
      .join("");
  }

  updateSectorMetrics(data) {
    const sectorContainer = document.getElementById("sector-metrics");
    if (!sectorContainer || !data) return;

    const topSectors = [...data].sort((a, b) => (b["Performance_%"] ?? 0) - (a["Performance_%"] ?? 0)).slice(0, 5);

    sectorContainer.innerHTML = topSectors
      .map(
        (s) => `
        <div class="metric-item">
          <span>${(s.Sector || "").replace(" Services", "")}</span>
          <span class="metric-value ${s["Performance_%"] >= 0 ? "positive" : "negative"}">
            ${s["Performance_%"] >= 0 ? "+" : ""}${(s["Performance_%"] ?? 0).toFixed(2)}%
          </span>
        </div>`
      )
      .join("");
  }

  formatMetricValue(value, format) {
    if (value === null || value === undefined || isNaN(value)) return "N/A";
    if (format === "percentage") return `${Number(value).toFixed(2)}%`;
    if (format === "number") return Number(value).toFixed(2);
    return String(value);
  }

  async generateFullReport() {
    const reportBtn = document.getElementById("report-btn");
    if (!reportBtn || this.isLoading) return;

    reportBtn.disabled = true;
    reportBtn.innerHTML = '<div class="spinner"></div> Generating...';

    try {
      const response = await fetch("/api/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm_provider: document.getElementById("llm-provider")?.value || "openai",
          include_sentiment: true,
          include_options: true,
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      this.addMessage(`📊 <strong>Comprehensive Research Report Generated</strong><br><br>${data.report}`, "assistant");
      this.showNotification("Research report generated successfully!", "success");
    } catch (error) {
      console.error("Error generating report:", error);
      this.addMessage("Sorry, I encountered an error generating the research report. Please try again later.", "assistant");
      this.showNotification("Error generating report. Please try again.", "error");
    } finally {
      reportBtn.disabled = false;
      reportBtn.innerHTML = '<i class="fas fa-file-alt"></i> Generate Full Report';
    }
  }

  startNewChat() {
    const messagesContainer = document.getElementById("chat-messages");
    if (messagesContainer) messagesContainer.innerHTML = "";

    const oldSession = this.sessionId;
    this.sessionId = this.generateSessionId();
    this.messageHistory = [];
    localStorage.removeItem(`chat_history_${oldSession}`);

    if (this.websocket) this.websocket.close();
    setTimeout(() => this.connectWebSocket(), 100);

    this.showNotification("Started new analysis session", "info");
  }

  sendQuickMessage(message) {
    this.sendMessage(message);
  }

  saveChatHistory() {
    try {
      localStorage.setItem(`chat_history_${this.sessionId}`, JSON.stringify(this.messageHistory));
    } catch (e) {
      console.error("Error saving chat history:", e);
    }
  }

  loadChatHistory() {
    try {
      const history = localStorage.getItem(`chat_history_${this.sessionId}`);
      if (history) this.messageHistory = JSON.parse(history);
    } catch (e) {
      console.error("Error loading chat history:", e);
    }
  }

  handleWindowResize() {
    if (window.innerWidth <= 768) document.body.classList.add("mobile");
    else document.body.classList.remove("mobile");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.aiApp = new AIEquityResearchApp();
});

// Expose for onclick handlers
function sendMessage() { window.aiApp?.sendMessage(); }
function generateFullReport() { window.aiApp?.generateFullReport(); }
function startNewChat() { window.aiApp?.startNewChat(); }
function sendQuickMessage(message) { window.aiApp?.sendQuickMessage(message); }
function handleKeyPress(event) { window.aiApp?.handleKeyPress(event); }

// CSS animations
const style = document.createElement("style");
style.textContent = `
@keyframes slideInRight { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes slideOutRight { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
`;
document.head.appendChild(style);
