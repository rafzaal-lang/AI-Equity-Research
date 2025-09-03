#!/usr/bin/env python3
"""
AI Equity Research Analyst Platform
Main application entry point for deployment
"""
import os
import sys
import logging
from pathlib import Path

from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Import FastAPI app from our server module
try:
    from fastapi_server import app as fastapi_app
except ImportError:
    print("Error: Could not import fastapi_server module")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = fastapi_app
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # tighten in prod
app.add_middleware(GZipMiddleware, minimum_size=1000)

def ensure_directories():
    for d in ["templates", "static", "reports", "logs", "data"]:
        Path(d).mkdir(exist_ok=True)

def create_default_template():
    template_file = Path("templates/index.html")
    if not template_file.exists():
        template_file.write_text("""<!DOCTYPE html>
<html><head><title>AI Equity Research Platform</title><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body><h1>AI Equity Research Platform</h1><p>Copy your UI HTML to templates/index.html</p><p><a href="/api/docs">API Docs</a></p></body></html>""", encoding="utf-8")

def create_static_files():
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    (static_dir / "app.js").touch(exist_ok=True)
    (static_dir / "favicon.ico").touch(exist_ok=True)

def validate_environment():
    required = ["OPENAI_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        return False
    for v in ["ALPHA_VANTAGE_KEY","FRED_KEY","NEWS_API_KEY","POLYGON_KEY","CLAUDE_KEY","GROK_KEY","GEMINI_KEY"]:
        logger.info(f"{v}: {'✓ Set' if os.getenv(v) else '✗ Not set'}")
    return True

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Equity Research Platform...")
    if not validate_environment():
        logger.warning("Continuing without optional providers...")
    ensure_directories()
    create_default_template()
    create_static_files()
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Equity Research Platform...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=os.getenv("ENVIRONMENT") != "production")
