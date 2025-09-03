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

from enhanced_equity_system import APIConfig, get_research_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Equity Research Analyst",
    description="Professional AI-powered equity research platform with multi-analyst framework",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

chat_sessions: Dict[str, List[Dict]] = {}
active_websockets: Dict[str, WebSocket] = {}

def generate_session_id() -> str:
    return str(uuid.uuid4())

def get_available_llm_providers() -> List[str]:
    cfg = APIConfig()
    providers = []
    if cfg.openai_key: providers.append("openai")
    if cfg.claude_key: providers.append("claude")
    if cfg.grok_key: providers.append("grok")
    if cfg.gemini_key: providers.append("gemini")
    if cfg.perplexity_key: providers.append("perplexity")
    return providers

def check_data_sources_status() -> Dict[str, str]:
    cfg = APIConfig()
    return {
        "alpha_vantage": "active" if cfg.alpha_vantage_key else "inactive",
        "fred": "active" if cfg.fred_key else "inactive",
        "news_api": "active" if cfg.news_key else "inactive",
        "polygon": "active" if cfg.polygon_key else "inactive",
        "nasdaq": "active" if cfg.nasdaq_key else "inactive",
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path("templates/index.html")
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    return HTMLResponse(
        content="""
        <!DOCTYPE html><html><head><title>AI Equity Research Platform</title></head>
        <body><h1>AI Equity Research Platform</h1>
        <p>Please ensure templates/index.html exists</p>
        <p><a href="/api/docs">API Documentation</a></p></body></html>""",
        status_code=200,
    )

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    return SystemStatus(
        status="active",
        available_llm_providers=get_available_llm_providers(),
        data_sources_status=check_data_sources_status(),
        last_updated=datetime.now(),
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    try:
        session_id = message.session_id or generate_session_id()
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        system = get_research_system(provider=message.llm_provider)
        # process_chat_query now only requires message; history optional in impl
        response = await system.process_chat_query(message.message)

        chat_sessions[session_id].append(
            {"user": message.message, "assistant": response, "timestamp": datetime.now().isoformat()}
        )
        if len(chat_sessions[session_id]) > 20:
            chat_sessions[session_id] = chat_sessions[session_id][-20:]

        return ChatResponse(response=response, session_id=session_id, timestamp=datetime.now())
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-report", response_model=ReportResponse)
async def generate_research_report(request: ReportRequest, background_tasks: BackgroundTasks):
    try:
        system = get_research_system(provider=request.llm_provider)
        result = await system.generate_comprehensive_research_report(
            include_sentiment=request.include_sentiment, include_options=request.include_options
        )

        report_id = f"EQR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        background_tasks.add_task(save_report_to_storage, report_id, result)

        return ReportResponse(
            report=result["report"],
            sector_data=result["sector_data"],
            economic_data=result["economic_data"],
            metadata=result["metadata"],
            report_id=report_id,
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sector-data")
async def get_sector_data():
    try:
        system = get_research_system()
        df = await system.build_sector_rotation_table(days=30)
        return JSONResponse(content={"data": df.to_dict("records"), "last_updated": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Sector data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/economic-indicators")
async def get_economic_indicators():
    try:
        system = get_research_system()
        econ = await system.get_economic_indicators()  # returns {"list":[...], "kv":{...}}
        return JSONResponse(content={"data": econ.get("list", []), "kv": econ.get("kv", {}), "last_updated": datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Economic indicators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_websockets[session_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            system = get_research_system(provider=payload.get("llm_provider", "openai"))
            response = await system.process_chat_query(payload.get("message", ""))

            chat_sessions.setdefault(session_id, []).append(
                {"user": payload.get("message", ""), "assistant": response, "timestamp": datetime.now().isoformat()}
            )
            await websocket.send_text(json.dumps({"response": response, "timestamp": datetime.now().isoformat(), "session_id": session_id}))
    except WebSocketDisconnect:
        active_websockets.pop(session_id, None)
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        active_websockets.pop(session_id, None)
        logger.error(f"WebSocket error: {e}")

@app.get("/api/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    history = chat_sessions.get(session_id, [])
    return JSONResponse(content={"history": history, "session_id": session_id})

@app.delete("/api/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return JSONResponse(content={"message": "Chat history cleared", "session_id": session_id})

@app.get("/api/reports")
async def list_reports():
    reports_dir = Path("reports")
    reports = []
    if reports_dir.exists():
        for fp in reports_dir.glob("*.json"):
            try:
                with open(fp, "r") as f:
                    metadata = json.load(f).get("metadata", {})
                reports.append({"report_id": fp.stem, "created_at": metadata.get("generation_time"), "data_sources": metadata.get("data_sources", [])})
            except Exception as e:
                logger.error(f"Error reading report {fp}: {e}")
    return JSONResponse(content={"reports": reports})

@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    rp = Path(f"reports/{report_id}.json")
    if not rp.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    try:
        with open(rp, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error reading report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading report")

async def save_report_to_storage(report_id: str, report_data: Dict[str, Any]):
    try:
        Path("reports").mkdir(exist_ok=True)
        def _default(o):
            if isinstance(o, datetime): return o.isoformat()
            raise TypeError()
        with open(Path("reports") / f"{report_id}.json", "w") as f:
            json.dump(report_data, f, indent=2, default=_default)
        logger.info(f"Report {report_id} saved to storage")
    except Exception as e:
        logger.error(f"Error saving report {report_id}: {e}")

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"})

app.mount("/static", StaticFiles(directory="static"), name="static")
