# 🚀 AI Equity Research Analyst Platform

A sophisticated AI-powered equity research platform featuring multi-analyst framework with real-time market data integration and institutional-quality research reports.

## 📊 Features

### Multi-Analyst Framework
- **Macro Economist**: Historical business cycle expert with deep economic philosophy
- **Quantitative Analyst**: Technical analysis specialist with statistical modeling
- **Fundamental Analyst**: CA/CFA with strategic insights from legendary investors

### Advanced Data Integration
- Real-time sector rotation analysis
- Fund flow and institutional positioning tracking
- Social media sentiment analysis
- Options flow and derivatives data
- Economic indicators from FRED API
- News sentiment analysis
- Earnings transcripts processing

### Professional Features
- Goldman Sachs/JP Morgan style research reports
- Interactive chat interface similar to Claude
- WebSocket real-time communication
- Multiple LLM provider support (OpenAI, Claude, Grok, Gemini, Perplexity)
- Professional financial UI/UX design
- Comprehensive API with full documentation

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11.9
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Database**: SQLite (development), PostgreSQL (production)
- **Caching**: Redis (optional)
- **Deployment**: Docker, Render.com
- **APIs**: OpenAI, Claude, Yahoo Finance, FRED, NewsAPI, Alpha Vantage, Polygon

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-equity-research.git
cd ai-equity-research
```

### 2. Set Up Environment

Create a `.env` file in the root directory:

```env
# Required - At least one LLM provider
OPENAI_KEY=sk-your-openai-api-key
CLAUDE_KEY=your-claude-api-key
GROK_KEY=your-grok-api-key
GEMINI_KEY=your-gemini-api-key
PERPLEXITY_KEY=your-perplexity-api-key

# Financial Data APIs
ALPHA_VANTAGE_KEY=your-alpha-vantage-key
FRED_KEY=your-fred-api-key
NEWS_API_KEY=your-news-api-key
POLYGON_KEY=your-polygon-api-key
NASDAQ_KEY=your-nasdaq-api-key

# Social Media APIs (Optional)
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
TWITTER_BEARER_TOKEN=your-twitter-bearer-token

# Application Settings
ENVIRONMENT=development
PORT=8000
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Set Up Project Structure

```bash
# Create required directories
mkdir -p templates static reports logs data

# Copy HTML template
# Copy the content from chat_interface_html artifact to templates/index.html

# Copy JavaScript
# Copy the content from frontend_javascript artifact to static/app.js
```

### 5. Run the Application

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` to access the application.

## 📁 Project Structure

```
ai-equity-research/
├── main.py                     # Application entry point
├── enhanced_equity_system.py   # Core research system
├── fastapi_server.py          # FastAPI web server
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── render.yaml               # Render deployment config
├── .env                      # Environment variables
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── templates/
│   └── index.html           # Main chat interface
├── static/
│   ├── app.js              # Frontend JavaScript
│   ├── favicon.ico         # Site icon
│   └── styles.css          # Additional styles (optional)
├── reports/                # Generated research reports
├── logs/                   # Application logs
├── data/                   # Cached data files
└── tests/                  # Unit tests (optional)
    ├── test_api.py
    ├── test_analysts.py
    └── test_data_collection.py
```

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the Docker image
docker build -t ai-equity-research .

# Run the container
docker run -p 8000:8000 --env-file .env ai-equity-research
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./reports:/app/reports
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: equity_research
      POSTGRES_USER: equity_user
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

Run with: `docker-compose up -d`

## ☁️ Render.com Deployment

### 1. Prepare Repository

1. Push your code to GitHub
2. Ensure all files are committed:
   - `main.py`
   - `enhanced_equity_system.py`
   - `fastapi_server.py`
   - `requirements.txt`
   - `render.yaml`
   - `templates/index.html`
   - `static/app.js`

### 2. Deploy on Render

1. Go to [Render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Set environment variables in Render dashboard:

```
OPENAI_KEY=sk-your-key
CLAUDE_KEY=your-key
ALPHA_VANTAGE_KEY=your-key
FRED_KEY=your-key
NEWS_API_KEY=your-key
POLYGON_KEY=your-key
```

6. Deploy!

### 3. Custom Domain (Optional)

1. In Render dashboard, go to Settings
2. Add your custom domain
3. Configure DNS records as instructed

## 🔑 API Keys Setup Guide

### Required APIs

1. **OpenAI API** (Required)
   - Go to [OpenAI API](https://platform.openai.com/api-keys)
   - Create new secret key
   - Add to environment as `OPENAI_KEY`

2. **Claude API** (Optional but recommended)
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Generate API key
   - Add as `CLAUDE_KEY`

### Financial Data APIs

3. **Alpha Vantage** (Stock data)
   - Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Get free API key
   - Add as `ALPHA_VANTAGE_KEY`

4. **FRED API** (Economic data)
   - Go to [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Create account and request API key
   - Add as `FRED_KEY`

5. **News API** (News data)
   - Go to [NewsAPI](https://newsapi.org/register)
   - Register for free API key
   - Add as `NEWS_API_KEY`

6. **Polygon.io** (Options data)
   - Go to [Polygon.io](https://polygon.io/)
   - Sign up for API access
   - Add as `POLYGON_KEY`

## 🎯 Usage Examples

### Chat Interface

1. **Sector Analysis**: "What sectors are outperforming today?"
2. **Macro Questions**: "What's the current economic outlook?"
3. **Technical Analysis**: "Show me momentum indicators for tech stocks"
4. **Research Reports**: Click "Generate Full Report" for comprehensive analysis

### API Endpoints

```python
# Get current sector performance
GET /api/sector-data

# Generate research report  
POST /api/generate-report
{
    "llm_provider": "openai",
    "include_sentiment": true,
    "include_options": true
}

# Chat endpoint
POST /api/chat
{
    "message": "Analyze current market conditions",
    "session_id": "session_123",
    "llm_provider": "claude"
}
```

### WebSocket Usage

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session_id');
ws.send(JSON.stringify({
    message: "What's the macro outlook?",
    llm_provider: "openai"
}));
```

## 🔧 Configuration Options

### LLM Providers

The system supports multiple LLM providers. Configure in the UI or via API:

- **OpenAI**: GPT-4 Turbo (default)
- **Claude**: Claude 3 Sonnet
- **Grok**: Grok Beta
- **Gemini**: Gemini Pro
- **Perplexity**: Perplexity AI

### Data Sources

- **Yahoo Finance**: Real-time stock data (free)
- **FRED**: Economic indicators (free)
- **Alpha Vantage**: Enhanced stock data
- **NewsAPI**: Financial news
- **Polygon**: Options flow data
- **Social Media**: Sentiment analysis

## 📈 Monitoring and Maintenance

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/health-detailed
```

### Logs

```bash
# View application logs
tail -f logs/application.log

# Docker logs
docker logs ai-equity-research
```

### Performance Monitoring

The application includes built-in monitoring:
- API response times
- Memory usage
- Active sessions
- External API status

## 🛡️ Security Considerations

### Production Security

1. **Environment Variables**: Never commit API keys to Git
2. **HTTPS**: Use HTTPS in production
3. **Rate Limiting**: Implement rate limiting for public APIs
4. **Input Validation**: All user inputs are validated
5. **CORS**: Configure CORS for production domains

### API Security

```python
# Example rate limiting (implement as needed)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, message: ChatMessage):
    # Implementation
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all files are in correct locations
   # Check Python path
   export PYTHONPATH=/path/to/project:$PYTHONPATH
   ```

2. **API Key Errors**
   ```bash
   # Verify environment variables are set
   echo $OPENAI_KEY
   
   # Check .env file is loaded
   cat .env
   ```

3. **Port Issues**
   ```bash
   # Check if port is in use
   lsof -i :8000
   
   # Use different port
   export PORT=8001
   ```

4. **WebSocket Connection Issues**
   - Check firewall settings
   - Verify WebSocket support
   - Check browser console for errors

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints
- Write unit tests
- Update documentation
- Test on multiple Python versions

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Anthropic for Claude API
- Yahoo Finance for market data
- Federal Reserve for economic data
- All the legendary investors whose wisdom inspired the fundamental analysis framework

## 📞 Support

- **Documentation**: Check this README and API docs at `/api/docs`
- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [Your email for support]

---

**Built with ❤️ for the financial research community**

*Empowering investment decisions through AI-driven multi-analyst research framework*