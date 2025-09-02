#!/usr/bin/env python3
"""
AI Equity Research Analyst Platform
Main application entry point for deployment
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Import our application modules
try:
    from fastapi_server import app as fastapi_app
except ImportError:
    print("Error: Could not import fastapi_server module")
    print("Make sure all required files are present")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the main application
app = fastapi_app

# Add production middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, specify your actual domain
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        "templates",
        "static",
        "reports",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# Create templates directory and index.html if it doesn't exist
def create_default_template():
    """Create default HTML template if it doesn't exist"""
    template_dir = Path("templates")
    template_file = template_dir / "index.html"
    
    if not template_file.exists():
        # Read the HTML content from our artifacts
        # In a real deployment, you'd copy the HTML file directly
        template_content = """
        <!-- This would be populated with the HTML content from the chat_interface_html artifact -->
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Equity Research Platform</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
            <h1>AI Equity Research Platform</h1>
            <p>Application is running. Please copy the HTML content from the artifacts to templates/index.html</p>
            <p><a href="/api/docs">API Documentation</a></p>
        </body>
        </html>
        """
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"Created default template: {template_file}")

# Create static directory and files
def create_static_files():
    """Create static files directory and basic files"""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Create basic app.js if it doesn't exist
    app_js = static_dir / "app.js"
    if not app_js.exists():
        with open(app_js, 'w', encoding='utf-8') as f:
            f.write("// JavaScript content from frontend_javascript artifact should go here\n")
        logger.info(f"Created default app.js: {app_js}")
    
    # Create basic favicon
    favicon_path = static_dir / "favicon.ico"
    if not favicon_path.exists():
        # Create a simple placeholder favicon
        with open(favicon_path, 'wb') as f:
            f.write(b'')  # Empty file for now
        logger.info(f"Created placeholder favicon: {favicon_path}")

# Environment validation
def validate_environment():
    """Validate that required environment variables are set"""
    required_env_vars = [
        'OPENAI_KEY',  # At least one LLM provider is required
    ]
    
    optional_env_vars = [
        'ALPHA_VANTAGE_KEY',
        'FRED_KEY',
        'NEWS_API_KEY',
        'POLYGON_KEY',
        'CLAUDE_KEY',
        'GROK_KEY',
        'GEMINI_KEY'
    ]
    
    missing_required = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        logger.error("Please set these environment variables before starting the application")
        return False
    
    # Log optional variables status
    for var in optional_env_vars:
        status = "✓ Set" if os.getenv(var) else "✗ Not set"
        logger.info(f"{var}: {status}")
    
    return True

# Health check with detailed system info
@app.get("/api/health-detailed")
async def detailed_health_check():
    """Detailed health check for monitoring"""
    try:
        # Check database connection
        db_status = "healthy"  # Would check actual DB connection
        
        # Check external APIs
        api_status = {
            "openai": "healthy" if os.getenv('OPENAI_KEY') else "unavailable",
            "fred": "healthy" if os.getenv('FRED_KEY') else "unavailable",
            "alpha_vantage": "healthy" if os.getenv('ALPHA_VANTAGE_KEY') else "unavailable"
        }
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "version": "1.0.0",
            "database": db_status,
            "external_apis": api_status,
            "memory_usage": "normal",  # Would calculate actual usage
            "active_sessions": 0  # Would count actual sessions
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting AI Equity Research Platform...")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories()
    
    # Create default files
    create_default_template()
    create_static_files()
    
    logger.info("Application startup completed successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down AI Equity Research Platform...")
    # Add any cleanup code here
    logger.info("Application shutdown completed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Custom 500 handler"""
    logger.error(f"Internal server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# Development server function
def run_development_server():
    """Run the development server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
        access_log=True
    )

# Production server configuration
def get_production_config():
    """Get production server configuration"""
    return {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "workers": int(os.getenv("WORKERS", 1)),
        "worker_class": "uvicorn.workers.UvicornWorker",
        "worker_connections": int(os.getenv("WORKER_CONNECTIONS", 1000)),
        "max_requests": int(os.getenv("MAX_REQUESTS", 10000)),
        "max_requests_jitter": int(os.getenv("MAX_REQUESTS_JITTER", 1000)),
        "timeout": int(os.getenv("TIMEOUT", 120)),
        "keepalive": int(os.getenv("KEEPALIVE", 5)),
        "preload_app": True,
        "log_level": "info"
    }

if __name__ == "__main__":
    # Check if running in production mode
    if os.getenv("ENVIRONMENT") == "production":
        # In production, use gunicorn (configured in Dockerfile)
        import gunicorn.app.wsgiapp as wsgi
        
        config = get_production_config()
        logger.info(f"Starting production server with config: {config}")
        
        # This would be handled by gunicorn in production
        uvicorn.run(
            "main:app",
            host=config["host"],
            port=config["port"],
            log_level=config["log_level"],
            access_log=True
        )
    else:
        # Development mode
        logger.info("Starting development server...")
        run_development_server()