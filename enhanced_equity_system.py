import os
import asyncio
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import openai
from anthropic import Anthropic
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
import warnings
import aiohttp
import sqlite3
from pathlib import Path
import logging
from functools import lru_cache
from textblob import TextBlob

warnings.filterwarnings('ignore')

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for all API keys and endpoints"""
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_KEY', '')
    fred_key: str = os.getenv('FRED_KEY', '')
    nasdaq_key: str = os.getenv('NASDAQ_KEY', '')
    news_key: str = os.getenv('NEWS_API_KEY', '')
    polygon_key: str = os.getenv('POLYGON_KEY', '')
    openai_key: str = os.getenv('OPENAI_KEY', '')
    claude_key: str = os.getenv('CLAUDE_KEY', '')
    grok_key: str = os.getenv('GROK_KEY', '')
    gemini_key: str = os.getenv('GEMINI_KEY', '')
    perplexity_key: str = os.getenv('PERPLEXITY_KEY', '')
    custom_llm_key: str = os.getenv('CUSTOM_LLM_KEY', '')
    custom_llm_endpoint: str = os.getenv('CUSTOM_LLM_ENDPOINT', '')
    reddit_client_id: str = os.getenv('REDDIT_CLIENT_ID', '')
    reddit_client_secret: str = os.getenv('REDDIT_CLIENT_SECRET', '')
    twitter_bearer_token: str = os.getenv('TWITTER_BEARER_TOKEN', '')

class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return f"Error generating response: {str(e)}"

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude error: {e}")
            return f"Error generating response: {str(e)}"

class GrokProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"
        
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", 
                                      headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        return f"Grok API error: {response.status}"
        except Exception as e:
            logger.error(f"Grok error: {e}")
            return f"Error generating response: {str(e)}"

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        headers = {"Content-Type": "application/json"}
        
        data = {
            "contents": [{
                "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}",
                    headers=headers, json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        return f"Gemini API error: {response.status}"
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"Error generating response: {str(e)}"

class PerplexityProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/chat/completions", 
                                      headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        return f"Perplexity API error: {response.status}"
        except Exception as e:
            logger.error(f"Perplexity error: {e}")
            return f"Error generating response: {str(e)}"

class EnhancedDataCollector:
    def __init__(self, config: APIConfig):
        self.config = config
        self.fred = Fred(api_key=config.fred_key) if config.fred_key else None
        self.db_path = Path("research_data.db")
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for caching"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sector_data (
                    date TEXT,
                    sector TEXT,
                    data TEXT,
                    PRIMARY KEY (date, sector)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    date TEXT,
                    source TEXT,
                    sector TEXT,
                    sentiment REAL,
                    data TEXT,
                    PRIMARY KEY (date, source, sector)
                )
            ''')
    
    @lru_cache(maxsize=100)
    def get_sector_etfs(self) -> Dict[str, str]:
        """Map of sectors to their representative ETFs"""
        return {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Financial": "XLF",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
            "Industrials": "XLI",
            "Communication Services": "XLC"
        }
    
    async def get_enhanced_sector_performance(self, days: int = 30) -> pd.DataFrame:
        """Get comprehensive sector performance data"""
        sectors = self.get_sector_etfs()
        performance_data = []
        
        for sector, ticker in sectors.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=f"{days}d")
                
                if len(hist) > 0:
                    # Basic performance metrics
                    current_price = hist['Close'].iloc[-1]
                    start_price = hist['Close'].iloc[0]
                    performance = ((current_price - start_price) / start_price) * 100
                    
                    # Volume analysis
                    avg_volume = hist['Volume'].mean()
                    recent_volume = hist['Volume'].tail(5).mean()
                    volume_trend = ((recent_volume - avg_volume) / avg_volume) * 100
                    
                    # Volatility metrics
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                    
                    # Technical indicators
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]
                    rsi = self._calculate_rsi(hist['Close'])
                    
                    # Options flow (if available)
                    options_data = await self._get_options_flow(ticker)
                    
                    performance_data.append({
                        'Sector': sector,
                        'Ticker': ticker,
                        'Performance_%': round(performance, 2),
                        'Current_Price': round(current_price, 2),
                        'Volume_Trend_%': round(volume_trend, 2),
                        'Volatility_%': round(volatility, 2),
                        'Sharpe_Ratio': round(sharpe_ratio, 2),
                        'RSI': round(rsi, 2),
                        'Price_vs_SMA20_%': round(((current_price - sma_20) / sma_20) * 100, 2),
                        'Price_vs_SMA50_%': round(((current_price - sma_50) / sma_50) * 100, 2),
                        'Put_Call_Ratio': options_data.get('put_call_ratio', 'N/A'),
                        'Options_Volume': options_data.get('total_volume', 'N/A')
                    })
            except Exception as e:
                logger.error(f"Error fetching data for {sector} ({ticker}): {e}")
                
        return pd.DataFrame(performance_data).sort_values('Performance_%', ascending=False)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    async def _get_options_flow(self, ticker: str) -> Dict:
        """Get options flow data"""
        try:
            if self.config.polygon_key:
                url = f"https://api.polygon.io/v3/reference/options/contracts"
                params = {
                    'underlying_ticker': ticker,
                    'limit': 100,
                    'apikey': self.config.polygon_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Process options data
                            return {'put_call_ratio': 0.8, 'total_volume': 100000}  # Placeholder
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
        
        return {'put_call_ratio': 'N/A', 'total_volume': 'N/A'}
    
    async def get_social_sentiment(self, sector: str) -> Dict[str, Any]:
        """Get social media sentiment for sector"""
        sentiment_data = {
            'reddit_sentiment': await self._get_reddit_sentiment(sector),
            'twitter_sentiment': await self._get_twitter_sentiment(sector),
            'news_sentiment': await self._get_news_sentiment(sector)
        }
        
        return sentiment_data
    
    async def _get_reddit_sentiment(self, sector: str) -> Dict:
        """Get Reddit sentiment"""
        try:
            # Reddit API implementation
            sentiment_score