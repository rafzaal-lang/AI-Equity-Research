from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime
import logging
from pathlib import Path

# Import our enhanced research system
from enhanced_equity_system import EnhancedEquityResearchSystem, APIConfig, get_research_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Equity Research Analyst",
    description="Professional AI-powered equity research platform with multi-analyst framework",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_provider: Optional[str] = "openai"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    
class ReportRequest(BaseModel):
    llm_provider: Optional[str] = "openai"
    include_sentiment: bool = True
    include_options: bool = True
    
class ReportResponse(BaseModel):
    report: str
    sector_data: List[Dict[str, Any]]
    economic_data: Dict[str, Any]
    metadata: Dict[str, Any]
    report_id: str

class SystemStatus(BaseModel):
    status: str
    available_llm_providers: List[str]
    data_sources_status: Dict[str, str]
    last_updated: datetime

# In-memory storage for chat sessions
chat_sessions: Dict[str, List[Dict]] = {}
active_websockets: Dict[str, WebSocket] = {}

# Utility functions
def generate_session_id() -> str:
    return str(uuid.uuid4())

def get_available_llm_providers() -> List[str]:
    """Check which LLM providers have API keys configured"""
    config = APIConfig()
    providers = []
    
    if config.openai_key:
        providers.append("openai")
    if config.claude_key:
        providers.append("claude")
    if config.grok_key:
        providers.append("grok")
    if config.gemini_key:
        providers.append("gemini")
    if config.perplexity_key:
        providers.append("perplexity")
    
    return providers

def check_data_sources_status() -> Dict[str, str]:
    """Check status of data source APIs"""
    config = APIConfig()
    status = {}
    
    status["alpha_vantage"] = "active" if config.alpha_vantage_key else "inactive"
    status["fred"] = "active" if config.fred_key else "inactive"
    status["news_api"] = "active" if config.news_key else "inactive"
    status["polygon"] = "active" if config.polygon_key else "inactive"
    status["nasdaq"] = "active" if config.nasdaq_key else "inactive"
    
    return status

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface"""
    html_file = Path("templates/index.html")
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Equity Research Platform</title>
        </head>
        <body>
            <h1>AI Equity Research Platform</h1>
            <p>Please ensure templates/index.html exists</p>
            <p><a href="/api/docs">API Documentation</a></p>
        </body>
        </html>
        """, status_code=200)

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and available providers"""
    return SystemStatus(
        status="active",
        available_llm_providers=get_available_llm_providers(),
        data_sources_status=check_data_sources_status(),
        last_updated=datetime.now()
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Handle chat messages"""
    try:
        # Get or create session
        session_id = message.session_id or generate_session_id()
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        # Get research system
        research_system = get_research_system()
        
        # Process the query
        response = await research_system.process_chat_query(
            message.message, 
            chat_sessions[session_id]
        )
        
        # Store in session history
        chat_sessions[session_id].append({
            "user": message.message,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit session history to last 20 exchanges
        if len(chat_sessions[session_id]) > 20:
            chat_sessions[session_id] = chat_sessions[session_id][-20:]
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-report", response_model=ReportResponse)
async def generate_research_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """Generate comprehensive research report"""
    try:
        # Initialize research system with specified LLM provider
        config = APIConfig()
        research_system = EnhancedEquityResearchSystem(config, llm_choice=request.llm_provider)
        
        # Generate comprehensive report
        result = await research_system.generate_comprehensive_research_report()
        
        # Generate unique report ID
        report_id = f"EQR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Store report (in production, save to database)
        background_tasks.add_task(save_report_to_storage, report_id, result)
        
        return ReportResponse(
            report=result['report'],
            sector_data=result['sector_data'],
            economic_data=result['economic_data'],
            metadata=result['metadata'],
            report_id=report_id
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sector-data")
async def get_sector_data():
    """Get current sector performance data"""
    try:
        research_system = get_research_system()
        sector_data = await research_system.data_collector.get_enhanced_sector_performance()
        return JSONResponse(content={
            "data": sector_data.to_dict('records'),
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Sector data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/economic-indicators")
async def get_economic_indicators():
    """Get current economic indicators"""
    try:
        research_system = get_research_system()
        economic_data = await research_system.data_collector.get_economic_indicators()
        return JSONResponse(content={
            "data": economic_data,
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Economic indicators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_websockets[session_id] = websocket
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            research_system = get_research_system()
            response = await research_system.process_chat_query(
                message_data.get("message", ""),
                chat_sessions.get(session_id, [])
            )
            
            # Store in session
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            
            chat_sessions[session_id].append({
                "user": message_data.get("message", ""),
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send response back to client
            await websocket.send_text(json.dumps({
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }))
            
    except WebSocketDisconnect:
        if session_id in active_websockets:
            del active_websockets[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session_id in active_websockets:
            del active_websockets[session_id]

@app.get("/api/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    history = chat_sessions.get(session_id, [])
    return JSONResponse(content={"history": history, "session_id": session_id})

@app.delete("/api/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return JSONResponse(content={"message": "Chat history cleared", "session_id": session_id})

@app.get("/api/reports")
async def list_reports():
    """List available reports"""
    # In production, this would query a database
    reports_dir = Path("reports")
    reports = []
    
    if reports_dir.exists():
        for file_path in reports_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f).get('metadata', {})
                    reports.append({
                        'report_id': file_path.stem,
                        'created_at': metadata.get('generation_time'),
                        'data_sources': metadata.get('data_sources', [])
                    })
            except Exception as e:
                logger.error(f"Error reading report {file_path}: {e}")
    
    return JSONResponse(content={"reports": reports})

@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Get specific report by ID"""
    report_file = Path(f"reports/{report_id}.json")
    
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        return JSONResponse(content=report_data)
    except Exception as e:
        logger.error(f"Error reading report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading report")

# Background tasks
async def save_report_to_storage(report_id: str, report_data: Dict[str, Any]):
    """Save report to persistent storage"""
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        report_file = reports_dir / f"{report_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=serialize_datetime)
        
        logger.info(f"Report {report_id} saved to storage")
        
    except Exception as e:
        logger.error(f"Error saving report {report_id}: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info"
    )