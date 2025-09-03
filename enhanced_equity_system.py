import os
import asyncio
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from anthropic import Anthropic
import yfinance as yf
from fredapi import Fred
import warnings
import aiohttp
import sqlite3
from pathlib import Path
import logging
from functools import lru_cache
from textblob import TextBlob

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class APIConfig:
    """
    Configuration for API keys and endpoints.
    Supports both *_KEY and *_API_KEY variants so env naming is flexible.
    """
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", ""))
    fred_key: str = os.getenv("FRED_KEY", os.getenv("FRED_API_KEY", ""))
    nasdaq_key: str = os.getenv("NASDAQ_KEY", os.getenv("NASDAQ_API_KEY", ""))
    news_key: str = os.getenv("NEWS_API_KEY", os.getenv("NEWS_KEY", ""))
    polygon_key: str = os.getenv("POLYGON_KEY", os.getenv("POLYGON_API_KEY", ""))
    openai_key: str = os.getenv("OPENAI_KEY", os.getenv("OPENAI_API_KEY", ""))
    claude_key: str = os.getenv("CLAUDE_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    grok_key: str = os.getenv("GROK_KEY", "")
    gemini_key: str = os.getenv("GEMINI_KEY", os.getenv("GOOGLE_API_KEY", ""))
    perplexity_key: str = os.getenv("PERPLEXITY_KEY", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    default_llm: str = os.getenv("DEFAULT_LLM", "openai")  # openai|claude|grok|gemini|perplexity


# -----------------------------------------------------------------------------
# LLM Providers
# -----------------------------------------------------------------------------
class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # adjust if you prefer a different model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.exception("OpenAI error")
            return f"[OpenAI error] {e}"


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            # Anthropic SDK is sync; run in thread
            resp = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.exception("Claude error")
            return f"[Claude error] {e}"


class GrokProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"

    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "grok-beta",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.base_url}/chat/completions", headers=headers, json=data) as r:
                    if r.status == 200:
                        j = await r.json()
                        return j["choices"][0]["message"]["content"]
                    return f"[Grok API error] HTTP {r.status}"
        except Exception as e:
            logger.exception("Grok error")
            return f"[Grok error] {e}"


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": max_tokens},
        }
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as r:
                    if r.status == 200:
                        j = await r.json()
                        return j["candidates"][0]["content"]["parts"][0]["text"]
                    return f"[Gemini API error] HTTP {r.status}"
        except Exception as e:
            logger.exception("Gemini error")
            return f"[Gemini error] {e}"


class PerplexityProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"

    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.base_url}/chat/completions", headers=headers, json=data) as r:
                    if r.status == 200:
                        j = await r.json()
                        return j["choices"][0]["message"]["content"]
                    return f"[Perplexity API error] HTTP {r.status}"
        except Exception as e:
            logger.exception("Perplexity error")
            return f"[Perplexity error] {e}"


# -----------------------------------------------------------------------------
# Data collector (prices, options flow, sentiment, macro snapshot)
# -----------------------------------------------------------------------------
class EnhancedDataCollector:
    def __init__(self, config: APIConfig):
        self.config = config
        self.fred = Fred(api_key=config.fred_key) if config.fred_key else None
        self.db_path = Path("research_data.db")
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sector_data (
                    date TEXT,
                    sector TEXT,
                    data TEXT,
                    PRIMARY KEY (date, sector)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    date TEXT,
                    source TEXT,
                    sector TEXT,
                    sentiment REAL,
                    data TEXT,
                    PRIMARY KEY (date, source, sector)
                )
                """
            )

    @lru_cache(maxsize=100)
    def get_sector_etfs(self) -> Dict[str, str]:
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
            "Communication Services": "XLC",
        }

    async def get_enhanced_sector_performance(self, days: int = 30) -> pd.DataFrame:
        sectors = self.get_sector_etfs()
        rows: List[Dict[str, Any]] = []

        for sector, ticker in sectors.items():
            try:
                hist = yf.Ticker(ticker).history(period=f"{days}d")
                if hist.empty:
                    continue

                current_price = float(hist["Close"].iloc[-1])
                start_price = float(hist["Close"].iloc[0])
                performance = (current_price - start_price) / start_price * 100.0

                avg_volume = float(hist["Volume"].mean())
                recent_volume = float(hist["Volume"].tail(5).mean())
                volume_trend = (recent_volume - avg_volume) / max(avg_volume, 1.0) * 100.0

                returns = hist["Close"].pct_change().dropna()
                volatility = float(returns.std() * np.sqrt(252) * 100.0)
                sharpe_ratio = float((returns.mean() * 252) / (returns.std() * np.sqrt(252))) if returns.std() > 0 else 0.0

                sma20 = float(hist["Close"].rolling(window=min(20, len(hist))).mean().iloc[-1])
                sma50 = float(hist["Close"].rolling(window=min(50, len(hist))).mean().iloc[-1])

                rsi = self._calculate_rsi(hist["Close"])
                options_data = await self._get_options_flow(ticker)

                rows.append(
                    {
                        "Sector": sector,
                        "Ticker": ticker,
                        "Performance_%": round(performance, 2),
                        "Current_Price": round(current_price, 2),
                        "Volume_Trend_%": round(volume_trend, 2),
                        "Volatility_%": round(volatility, 2),
                        "Sharpe_Ratio": round(sharpe_ratio, 2),
                        "RSI": round(rsi, 2),
                        "Price_vs_SMA20_%": round(((current_price - sma20) / max(sma20, 1e-9)) * 100.0, 2),
                        "Price_vs_SMA50_%": round(((current_price - sma50) / max(sma50, 1e-9)) * 100.0, 2),
                        "Put_Call_Ratio": options_data.get("put_call_ratio", "N/A"),
                        "Options_Volume": options_data.get("total_volume", "N/A"),
                    }
                )
            except Exception as e:
                logger.error(f"Data fetch error for {sector} ({ticker}): {e}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Performance_%", ascending=False).reset_index(drop=True)
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    async def _get_options_flow(self, ticker: str) -> Dict[str, Any]:
        """
        Polygon example placeholder. If POLYGON_KEY is present, you can extend this to parse real flow.
        """
        try:
            if self.config.polygon_key:
                url = "https://api.polygon.io/v3/reference/options/contracts"
                params = {"underlying_ticker": ticker, "limit": 25, "apikey": self.config.polygon_key}
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, params=params, timeout=20) as r:
                        if r.status == 200:
                            _ = await r.json()
                            # TODO: parse for real put/call metrics
                            return {"put_call_ratio": 0.8, "total_volume": 100000}
        except Exception as e:
            logger.warning(f"Options flow error for {ticker}: {e}")
        return {"put_call_ratio": "N/A", "total_volume": "N/A"}

    # ---------------------------- Macro snapshot ------------------------------
    async def get_economic_indicators(self) -> pd.DataFrame:
        """
        Async wrapper so route handlers can 'await' this.
        Runs the sync computation in a thread.
        """
        return await asyncio.to_thread(self._get_economic_indicators_sync)

    def _get_economic_indicators_sync(self) -> pd.DataFrame:
        """
        Pull a small macro snapshot from FRED.
        Returns a DataFrame with latest levels and simple deltas.
        Safe no-op if no FRED key.
        """
        if not self.fred:
            return pd.DataFrame(
                columns=["Indicator", "Latest", "Date", "1m_change", "3m_change", "YoY"]
            )

        series = [
            ("Inflation (CPI, Index)", "CPIAUCSL"),
            ("Unemployment Rate (%)", "UNRATE"),
            ("Fed Funds Rate (%)", "FEDFUNDS"),
            ("10Y Treasury Yield (%)", "DGS10"),
            ("ISM Mfg PMI", "NAPM"),
        ]

        rows = []
        for name, sid in series:
            try:
                s = self.fred.get_series(sid).dropna()
                last_date = s.index[-1]
                last = float(s.iloc[-1])

                def at_months_ago(m: int) -> float:
                    target = last_date - pd.DateOffset(months=m)
                    # nearest index value in case of business-day calendar
                    idx = s.index.get_indexer([target], method="nearest")[0]
                    return float(s.iloc[idx])

                one_m = at_months_ago(1)
                three_m = at_months_ago(3)
                twelve_m = at_months_ago(12)

                rows.append({
                    "Indicator": name,
                    "Latest": round(last, 2),
                    "Date": str(pd.to_datetime(last_date).date()),
                    "1m_change": round(last - one_m, 2),
                    "3m_change": round(last - three_m, 2),
                    "YoY": round(last - twelve_m, 2),
                })
            except Exception as e:
                logger.warning(f"FRED series {sid} error: {e}")

        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Research System
# -----------------------------------------------------------------------------
class EnhancedEquityResearchSystem:
    """
    Orchestrates data collection + LLM analysis for sector rotation.
    """
    def __init__(self, config: APIConfig, llm: LLMProvider):
        self.config = config
        self.llm = llm
        self.collector = EnhancedDataCollector(config)
        self.data_collector = self.collector  # ← back-compat alias for older route handlers

    async def build_sector_rotation_table(self, days: int = 30) -> pd.DataFrame:
        return await self.collector.get_enhanced_sector_performance(days=days)

    async def get_economic_indicators(self) -> pd.DataFrame:
        return await self.collector.get_economic_indicators()

    async def write_research_note(self, df: pd.DataFrame, days: int = 30) -> str:
        if df is None or df.empty:
            return "No sector data available."

        # Compose sentiment enrichments for the top 3 sectors to keep latency reasonable
        top = df.head(3)["Sector"].tolist()
        sentiments: Dict[str, Dict[str, Any]] = {}
        for sct in top:
            sentiments[sct] = await self.collector.get_social_sentiment(sct)

        system_prompt = (
            "You are an AI Equity Research Analyst with three voices: "
            "(1) Macro economist (cycle-aware, concise), "
            "(2) Quant technician (signals, momentum/RSI/volatility, sector rotation), "
            "(3) Fundamental analyst (business quality, catalysts, risks). "
            "Synthesize them into a clear, balanced note with sector overweights/underweights."
        )

        # Build a compact table string
        table = df.to_string(index=False)
        sent_json = json.dumps(sentiments, indent=2)

        user_prompt = f"""
Timeframe: last {days} days.
Sector performance table:
{table}

Supplementary social/news sentiment (top sectors):
{sent_json}

Task: Provide a sector-rotation view: which sectors are gaining/losing traction and why.
Include 2-3 actionable observations and a list of overweight/market-weight/underweight calls with brief rationale.
"""

        return await self.llm.generate_response(user_prompt, system_prompt, max_tokens=1200)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def get_research_system(config: Optional[APIConfig] = None, provider: Optional[str] = None) -> EnhancedEquityResearchSystem:
    cfg = config or APIConfig()
    name = (provider or cfg.default_llm or "openai").lower()

    if name == "openai" and cfg.openai_key:
        llm = OpenAIProvider(cfg.openai_key)
    elif name == "claude" and cfg.claude_key:
        llm = ClaudeProvider(cfg.claude_key)
    elif name == "grok" and cfg.grok_key:
        llm = GrokProvider(cfg.grok_key)
    elif name == "gemini" and cfg.gemini_key:
        llm = GeminiProvider(cfg.gemini_key)
    elif name == "perplexity" and cfg.perplexity_key:
        llm = PerplexityProvider(cfg.perplexity_key)
    else:
        # Fallback to OpenAI if available, else a stub that just echoes
        llm = OpenAIProvider(cfg.openai_key) if cfg.openai_key else _EchoProvider()

    return EnhancedEquityResearchSystem(cfg, llm)


class _EchoProvider(LLMProvider):
    async def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        return f"[LLM not configured]\n\nSYSTEM:\n{system_prompt}\n\nPROMPT:\n{prompt[:2000]}"


# -----------------------------------------------------------------------------
# Local smoke test (won't run on import)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo():
        sys = get_research_system()
        df = await sys.build_sector_rotation_table(days=30)
        print(df.head(5))
        econ = await sys.get_economic_indicators()
        print("\n====== ECON SNAPSHOT ======\n")
        print(econ.to_string(index=False))
        note = await sys.write_research_note(df, days=30)
        print("\n====== RESEARCH NOTE ======\n")
        print(note)

    asyncio.run(_demo())
