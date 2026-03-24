# Prediction Market Strategy Blueprint
## CBB Edge Analyzer → Quant Trading Infrastructure

**Prepared for:** CBB Edge Analyzer Development Team  
**Date:** March 24, 2026  
**Classification:** Internal Strategy Document  
**Version:** 1.0

---

## Executive Summary

Your current CBB Edge Analyzer represents a sophisticated predictive modeling system that generates +EV (positive expected value) sports predictions. The architecture—Python/FastAPI, PostgreSQL with SQLAlchemy async support, and deployment on Railway—is well-suited for **signal generation** but faces fundamental limitations for **exchange operation**.

This blueprint presents two strategic pivots:

1. **Primary Strategy (Automated Quant):** Transform your predictive models into an automated market-making and arbitrage system that plugs into existing prediction market APIs (Kalshi/Polymarket)
2. **Secondary Strategy (Private Syndicate):** Deploy a lightweight, PostgreSQL-based matching engine for your Discord community using virtual points

---

## 1. The Architectural Gap Analysis

### 1.1 How P2P Event Contract Exchanges Actually Work

#### The Three-Layer Architecture

A production prediction market exchange operates across three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LAYER 1: CLIENT INTERFACE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Web App   │  │  Mobile App │  │  API/WebSocket │  │  Institutional FIX │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 2: MATCHING ENGINE (CLOB)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CENTRAL LIMIT ORDER BOOK                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Order Ingest│  │ Price-Time  │  │   Trade     │  │   Market    │ │   │
│  │  │   Queue     │  │  Priority   │  │  Execution  │  │   Data      │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                                     │   │
│  │  Latency Requirement: <10μs (microseconds) for matching             │   │
│  │  Throughput: 100K-1M orders/second                                  │   │
│  │  State: In-memory (Redis/Custom) with WAL to durable storage        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 3: SETTLEMENT & CLEARING                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │  Collateral │  │   Oracle    │  │   P&L       │  │   Regulatory│ │   │
│  │  │ Management  │  │  Resolution │  │ Calculation │  │   Reporting │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                                     │   │
│  │  Kalshi: Centralized USD custody, CFTC-regulated oracle             │   │
│  │  Polymarket: USDC on Polygon, decentralized conditional tokens      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### The Matching Engine Deep Dive

The CLOB (Central Limit Order Book) is the heart of any prediction market exchange:

**Order Book Structure (Binary Market):**
```
Market: "Will Duke beat UNC by >5.5 points?"

BIDS (YES side)                    ASKS (YES side)
─────────────                      ─────────────
Price    Size    User              Price    Size    User
0.62     $500    @alice            0.63     $300    @bob
0.61     $800    @charlie          0.64     $600    @dave
0.60     $1200   @eve              0.65     $400    @frank

Spread: 1 cent (0.62 bid / 0.63 ask)
Midpoint: 0.625 (62.5% implied probability)
```

**The Matching Algorithm (Price-Time Priority):**
1. **Price Priority:** Higher bids match before lower bids; lower asks match before higher asks
2. **Time Priority:** At the same price, earlier orders fill first
3. **Pro-Rata (optional):** Large orders may fill against multiple resting orders

**Trade Execution Flow:**
```python
# Simplified matching logic
class MatchingEngine:
    def match_order(self, incoming_order):
        # Buy YES at 0.65 → match against asks <= 0.65
        if incoming_order.side == BUY:
            for ask in self.asks.ascending_by_price():
                if ask.price <= incoming_order.price:
                    fill_qty = min(ask.size, incoming_order.remaining)
                    self.execute_trade(fill_qty, ask.price, ask.user, incoming_order.user)
                    
                    if incoming_order.remaining == 0:
                        return  # Fully filled
                else:
                    break  # No more matches possible
            
            # Remaining quantity goes to order book as resting bid
            if incoming_order.remaining > 0:
                self.bids.insert(incoming_order)
```

**Latency Requirements by Use Case:**

| Use Case | Latency Budget | Technology |
|----------|----------------|------------|
| Retail web trading | <100ms | Standard REST/WebSocket |
| Mobile app | <50ms | Optimized REST + push notifications |
| API trading | <10ms | WebSocket + colocation |
| Market making | <1ms | FIX protocol + FPGA/hardware |
| Cross-exchange arb | <100μs | Microwave networks, kernel bypass |

### 1.2 Your Current Stack: The Brutally Honest Assessment

#### Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR CURRENT STACK                         │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI (async/sync hybrid)                                    │
│  ├── SQLAlchemy ORM (sync path) + asyncpg (hot path)           │
│  ├── APScheduler for nightly analysis                          │
│  └── Pydantic for request/response validation                    │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL (Railway)                                           │
│  ├── Models: Game, Prediction, BetLog, PerformanceSnapshot     │
│  ├── Connection pooling: 20 pool + 40 overflow (sync)          │
│  └── Async pool: 10 pool + 20 overflow                          │
├─────────────────────────────────────────────────────────────────┤
│  Current Throughput Characteristics:                            │
│  ├── Nightly batch: ~300 games processed over 5-10 minutes     │
│  ├── API response time: 50-200ms typical                        │
│  └── Peak concurrent: ~50 connections estimated                 │
└─────────────────────────────────────────────────────────────────┘
```

#### The Concurrency Wall

Your current PostgreSQL configuration on Railway:

```python
# Current models.py configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # ← This is your hard ceiling
    max_overflow=40,        # ← Temporary burst capacity
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_engine = create_async_engine(
    _ASYNC_DATABASE_URL,
    pool_size=10,           # ← Even lower for async
    max_overflow=20,
)
```

**The Math Problem:**
- A public exchange needs to handle **10,000-100,000 concurrent connections**
- Each market needs constant order book updates (WebSocket subscriptions)
- Each trade requires multiple database writes (order state, position, balance)
- Your 60 total connections (20+40 sync, 10+20 async) would saturate at ~100 concurrent users

**Connection Pool Exhaustion Scenario:**
```
User A places order → Takes 1 DB connection
User B places order → Takes 1 DB connection  
WebSocket broadcasts to 1000 subscribers → 1000 potential connection requests
→ POOL EXHAUSTION → Connection refused errors
```

#### The Latency Reality

Your current latency profile:

| Operation | Current Latency | Exchange Requirement | Gap |
|-----------|-----------------|---------------------|-----|
| Simple SELECT | 5-20ms | <1ms | 5-20x |
| Complex JOIN (predictions + games) | 50-150ms | <5ms | 10-30x |
| INSERT (new bet log) | 10-30ms | <1ms | 10-30x |
| Transaction commit | 20-50ms | <1ms | 20-50x |
| API round-trip | 100-300ms | <10ms | 10-30x |

**Why PostgreSQL Can't Match:**
1. **Disk I/O:** Even with SSDs, WAL (Write-Ahead Log) commits add 5-20ms
2. **MVCC Overhead:** PostgreSQL's concurrency model creates tuple versions
3. **Lock Contention:** Row-level locks on hot rows (popular markets) create queues
4. **Network Round-trip:** Railway's managed Postgres adds 1-5ms network latency

#### State Management: The ACID vs Speed Tradeoff

Your current state model (from `models.py`):

```python
class BetLog(Base):
    """Represents a bet placed by a user"""
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    odds_taken = Column(Float, nullable=False)
    outcome = Column(Integer)  # 1=win, 0=loss
    profit_loss_dollars = Column(Float)
    # ... 20+ more fields
```

**The Exchange State Problem:**

A single trade on an exchange updates:
1. Order book state (bids/asks)
2. User's open orders
3. User's position
4. User's available balance
5. Trade history
6. Market statistics (volume, OHLC)

**ACID guarantees across all these = Slow.**

Production exchanges use **event sourcing + CQRS**:
- **Write Path:** Events appended to immutable log (Kafka/Pulsar) → <1ms
- **Read Path:** Materialized views rebuilt from events (Redis/ClickHouse) → <1ms
- **State Reconstruction:** Replay events to rebuild any state

### 1.3 What Building a Public Exchange Would Actually Require

#### The Infrastructure Investment

| Component | Current | Exchange-Grade | Cost |
|-----------|---------|----------------|------|
| **Matching Engine** | N/A | Custom C++ or FPGA | $500K-2M dev |
| **Database** | Postgres (single) | CockroachDB/TiDB cluster | $5K-20K/mo |
| **Cache Layer** | N/A | Redis Cluster (10+ nodes) | $2K-5K/mo |
| **Message Queue** | N/A | Apache Kafka/Pulsar | $3K-8K/mo |
| **WebSocket** | N/A | Custom + load balancers | $2K-5K/mo |
| **Compliance** | N/A | Legal + regulatory tech | $500K+/year |
| **Security Audit** | N/A | Penetration testing | $100K-500K |

#### The Talent Requirement

Your current stack is maintainable by a small team. A production exchange requires:

- **Low-latency systems engineers** ($300K-500K salary)
- **Distributed systems architects** ($250K-400K)
- **Security engineers** (crypto/financial) ($250K-400K)
- **Compliance officers** ($150K-300K)
- **DevOps/SRE** for 99.99% uptime ($200K-350K)

**Bottom Line:** Building a public P2P exchange from your current stack is a 2-3 year, $5-10M investment with significant regulatory risk. It's not the right path.

---

## 2. The "Automated Quant" Pivot (Primary Strategy)

### 2.1 Strategic Overview

Instead of building an exchange, **become the smartest trader on existing exchanges.**

Your edge:
- Sophisticated predictive models (KenPom + BartTorvik + EvanMiya ensemble)
- Fatigue/rest adjustments
- Kelly criterion sizing
- CLV (Closing Line Value) tracking

The opportunity:
- Prediction markets have **inefficient pricing** on niche sports
- Retail order flow creates **predictable mispricings**
- You can provide **liquidity** (market making) while capturing edge

### 2.2 Technical Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOUR PREDICTION ENGINE                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CBB Edge Model (Existing)                                          │   │
│  │  ├── KenPom/BartTorvik/EvanMiya ratings ingestion                   │   │
│  │  ├── Fatigue model (rest, travel, altitude)                         │   │
│  │  ├── Possession simulator (Markov/Gaussian)                         │   │
│  │  └── Output: Projected margin + win probability                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Signal Translation Layer (NEW)                                     │   │
│  │  ├── Convert spread predictions → binary contract prices            │   │
│  │  ├── Account for market fees (Polymarket: 0%, Kalshi: 0%)           │   │
│  │  ├── Apply risk-adjusted Kelly sizing                               │   │
│  │  └── Generate: Target entry price, max position, exit triggers      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXCHANGE CONNECTIVITY LAYER                            │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │      KALSHI INTEGRATION     │  │      POLYMARKET INTEGRATION         │   │
│  │  ┌─────────────────────┐    │  │  ┌─────────────────────────────┐    │   │
│  │  │ kalshi-python SDK   │    │  │  │ py-clob-client              │    │   │
│  │  │ RSA-PSS signing     │    │  │  │ EIP-712 order signing       │    │   │
│  │  │ JWT auth (30min)    │    │  │  │ HMAC L2 headers             │    │   │
│  │  │ REST + WebSocket    │    │  │  │ Gamma + CLOB + Data APIs    │    │   │
│  │  └─────────────────────┘    │  │  └─────────────────────────────┘    │   │
│  │                             │  │                                     │   │
│  │  Rate limits: 10 req/s      │  │  Rate limits: ~100/min              │   │
│  │  Settlement: USD (fiat)     │  │  Settlement: USDC on Polygon        │   │
│  │  Regulatory: CFTC-approved  │  │  Regulatory: Offshore/unclear       │   │
│  │  Markets: Sports, Econ      │  │  Markets: Sports, Politics, Crypto  │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION ORCHESTRATION (NEW)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Async Trading Engine                                               │   │
│  │  ├── Market scanner (continuous order book monitoring)              │   │
│  │  ├── Signal generator (price → model divergence detection)          │   │
│  │  ├── Order manager (place/cancel/amend)                             │   │
│  │  ├── Position tracker (P&L, exposure, risk limits)                  │   │
│  │  └── Settlement handler (exercise at expiration)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Converting Your Predictions to Binary Contracts

This is the critical translation layer. Your model outputs **point spreads and probabilities**; prediction markets trade **binary yes/no contracts at 0.01-0.99**.

#### The Mathematics of Translation

**Step 1: Spread → Binary Probability**

Your model outputs: `Projected Margin = Home - Away = +4.5` (home favored by 4.5)

The market asks: "Will Home win by more than 5.5 points?" (binary: YES/NO)

```python
import numpy as np
from scipy import stats

def spread_to_binary_probability(
    projected_margin: float,
    historical_std: float,  # ~10-12 points for CBB
    market_spread: float    # The line being offered
) -> float:
    """
    Convert your projected spread to probability of covering market spread.
    
    Example:
    - Your model: Duke +4.5 over UNC
    - Market contract: "Duke wins by >5.5?" 
    - You need P(Duke margin > 5.5)
    """
    # Distance from your projection to market line
    distance = projected_margin - market_spread  # 4.5 - 5.5 = -1.0
    
    # Assume normal distribution of outcomes
    # P(margin > market_spread) = 1 - CDF((market_spread - projected_margin) / std)
    z_score = (market_spread - projected_margin) / historical_std
    probability = 1 - stats.norm.cdf(z_score)
    
    return probability  # 0.0 to 1.0

# Example
model_margin = 4.5
market_spread = 5.5
historical_std = 11.0  # CBB typical

prob = spread_to_binary_probability(model_margin, historical_std, market_spread)
print(f"P(Duke covers -5.5) = {prob:.3f}")  # ~0.464 (46.4%)
```

**Step 2: Account for Juice/Vig**

Prediction markets typically don't charge spread juice like sportsbooks, but Kalshi prices include an implied spread:

```python
def remove_vig(yes_price: float, no_price: float) -> float:
    """
    Kalshi shows YES at 63¢ and NO at 40¢ (sum = 103¢)
    Remove the 3% vig to get true probability.
    """
    total = yes_price + no_price
    true_prob = yes_price / total
    return true_prob

# Example
kalshi_yes = 0.63  # 63 cents
kalshi_no = 0.40   # 40 cents

true_prob = remove_vig(kalshi_yes, kalshi_no)
print(f"True implied probability: {true_prob:.3f}")  # 0.612 (61.2%)
```

**Step 3: Calculate Edge**

```python
def calculate_edge(model_prob: float, market_prob: float, fees: float = 0.0) -> dict:
    """
    Calculate expected value of betting YES at market price.
    """
    # Market price = implied probability + vig
    # EV = (model_prob * payout) + ((1 - model_prob) * -loss)
    # Binary contract: Pay P, win $1 (profit = 1-P) or lose P
    
    payout_if_win = (1 - market_prob) * (1 - fees)
    loss_if_lose = market_prob
    
    ev = (model_prob * payout_if_win) - ((1 - model_prob) * loss_if_lose)
    
    # Edge as percentage
    edge_pct = ev / market_prob if market_prob > 0 else 0
    
    # Kelly criterion fraction (for binary bets)
    # f* = (bp - q) / b
    # where b = odds received (decimal), p = win prob, q = lose prob
    b = payout_if_win / market_prob  # Decimal odds
    kelly = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0
    
    return {
        "expected_value": ev,
        "edge_percent": edge_pct * 100,
        "kelly_fraction": kelly,
        "recommendation": "BET" if edge_pct > 0.05 else "PASS"  # 5% edge threshold
    }

# Example
model_prob = 0.55
market_prob = 0.46

result = calculate_edge(model_prob, market_prob)
print(f"EV: ${result['expected_value']:.3f}")
print(f"Edge: {result['edge_percent']:.1f}%")
print(f"Kelly: {result['kelly_fraction']:.3f}")
```

**Step 4: Your Existing Kelly Integration**

Your current `BetLog` model already tracks Kelly sizing:

```python
class BetLog(Base):
    kelly_full = Column(Float)
    kelly_fractional = Column(Float)  # Probably 0.25 or 0.5 of full Kelly
    bet_size_units = Column(Float)
```

Extend this for prediction markets:

```python
class PredictionMarketSignal(Base):
    """NEW: Maps your model outputs to prediction market opportunities"""
    __tablename__ = "pm_signals"
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    
    # Market identification
    exchange = Column(String)  # "kalshi" or "polymarket"
    market_ticker = Column(String)  # "CBB-DUKE-20250325" or token_id
    contract_type = Column(String)  # "spread", "moneyline", "total"
    
    # Your model's view
    model_probability = Column(Float)  # Your calculated P(win)
    model_confidence = Column(Float)   # SNR or similar
    
    # Market state at signal generation
    market_yes_price = Column(Float)   # Current YES price (0.01-0.99)
    market_implied_prob = Column(Float)  # Vig-adjusted
    
    # Edge calculation
    edge_percent = Column(Float)
    kelly_fraction = Column(Float)
    
    # Execution plan
    target_entry = Column(Float)       # Don't pay more than this
    max_position_dollars = Column(Float)  # Kelly-derived size
    exit_trigger_win = Column(Float)   # Take profit level
    exit_trigger_loss = Column(Float)  # Stop loss level
    
    # Execution tracking
    status = Column(String)  # "PENDING", "ENTERED", "CLOSED", "EXPIRED"
    entry_price = Column(Float)
    entry_time = Column(DateTime)
    exit_price = Column(Float)
    pnl = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 2.4 Async Worker Architecture for Market Monitoring

This is where your FastAPI/asyncPostgreSQL stack can excel—**as a trading bot, not an exchange**.

#### System Components

```python
# /backend/trading_engine/
├── __init__.py
├── config.py              # Exchange API keys, risk limits
├── exchange_clients/      # Kalshi & Polymarket wrappers
│   ├── base.py           # Abstract exchange interface
│   ├── kalshi_client.py  # Kalshi implementation
│   └── polymarket_client.py  # Polymarket implementation
├── models.py             # Trading-specific SQLAlchemy models
├── scanners.py           # Market scanning workers
├── signal_generator.py   # Edge detection logic
├── order_manager.py      # Execution logic
├── position_tracker.py   # P&L and exposure monitoring
├── risk_manager.py       # Circuit breakers, limits
└── main.py               # Async worker entry point
```

#### The Scanner Worker

```python
# backend/trading_engine/scanners.py
import asyncio
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import httpx

@dataclass
class MarketQuote:
    """Normalized quote across exchanges"""
    exchange: str
    ticker: str
    yes_bid: float      # Best bid for YES
    yes_ask: float      # Best ask for YES
    no_bid: float       # Best bid for NO
    no_ask: float       # Best ask for NO
    last_trade: float
    volume_24h: float
    timestamp: datetime

class KalshiScanner:
    """
    Continuously scans Kalshi sports markets for CBB opportunities.
    
    Rate limit: 10 requests/second (conservative: 8/s)
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.client = KalshiClient(api_key, api_secret)
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self._semaphore = asyncio.Semaphore(8)  # Rate limiting
        
    async def scan_cbb_markets(self) -> List[MarketQuote]:
        """
        Fetch all active CBB markets and their order books.
        
        Kalshi series for CBB: "KXSCOLLEGE" or similar
        """
        async with self._semaphore:
            markets = await self._fetch_series_markets("KXSCOLLEGE")
            
        quotes = []
        for market in markets:
            if not self._is_tradable(market):
                continue
                
            async with self._semaphore:
                orderbook = await self._fetch_orderbook(market['ticker'])
                
            quote = self._normalize_quote(market, orderbook)
            quotes.append(quote)
            
        return quotes
    
    async def _fetch_series_markets(self, series_ticker: str) -> List[dict]:
        """Fetch all markets in a series (e.g., all CBB games)"""
        url = f"{self.base_url}/series/{series_ticker}/markets"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._auth_headers())
            response.raise_for_status()
            data = response.json()
            return data.get('markets', [])
    
    async def _fetch_orderbook(self, ticker: str) -> dict:
        """Fetch order book for a specific market"""
        url = f"{self.base_url}/markets/{ticker}/orderbook"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._auth_headers())
            response.raise_for_status()
            return response.json()
    
    def _normalize_quote(self, market: dict, orderbook: dict) -> MarketQuote:
        """Convert Kalshi format to standardized MarketQuote"""
        # Kalshi orderbook: yes bids + no bids
        # YES ask = (100 - best NO bid) / 100
        yes_bids = orderbook.get('yes_bids', [])
        no_bids = orderbook.get('no_bids', [])
        
        best_yes_bid = yes_bids[0]['price'] / 100 if yes_bids else 0
        best_no_bid = no_bids[0]['price'] / 100 if no_bids else 0
        best_yes_ask = (100 - best_no_bid) / 100 if best_no_bid else 1.0
        best_no_ask = (100 - best_yes_bid * 100) / 100 if best_yes_bid else 1.0
        
        return MarketQuote(
            exchange="kalshi",
            ticker=market['ticker'],
            yes_bid=best_yes_bid,
            yes_ask=best_yes_ask,
            no_bid=best_no_bid,
            no_ask=best_no_ask,
            last_trade=market.get('last_price', 0) / 100,
            volume_24h=market.get('volume', 0),
            timestamp=datetime.utcnow()
        )

class PolymarketScanner:
    """
    Scans Polymarket sports markets.
    
    Rate limit: ~100 requests/minute
    Uses Gamma API for discovery, CLOB API for prices
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.clob_url = "https://clob.polymarket.com"
        self._semaphore = asyncio.Semaphore(80)  # Conservative under 100/min
        
    async def scan_cbb_markets(self) -> List[MarketQuote]:
        """
        Fetch all active CBB markets from Polymarket.
        
        Polymarket uses 'slugs' and 'condition_ids' for market identification.
        Sports markets are under specific tags.
        """
        async with self._semaphore:
            # Search for CBB markets
            markets = await self._search_markets(
                tag="sports",
                active=True,
                closed=False
            )
        
        # Filter to CBB specifically
        cbb_markets = [
            m for m in markets 
            if 'college basketball' in m.get('description', '').lower() 
            or 'ncaa' in m.get('description', '').lower()
        ]
        
        quotes = []
        for market in cbb_markets:
            # Polymarket markets have multiple tokens (outcomes)
            for token in market.get('tokens', []):
                async with self._semaphore:
                    book = await self._fetch_orderbook(token['token_id'])
                    
                quote = self._normalize_quote(market, token, book)
                quotes.append(quote)
                
        return quotes
    
    async def _search_markets(self, **filters) -> List[dict]:
        """Query Gamma API for markets"""
        url = f"{self.gamma_url}/markets"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=filters)
            response.raise_for_status()
            return response.json()
    
    async def _fetch_orderbook(self, token_id: str) -> dict:
        """Fetch CLOB order book for token"""
        url = f"{self.clob_url}/book"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"token_id": token_id})
            response.raise_for_status()
            return response.json()
```

#### The Signal Generator

```python
# backend/trading_engine/signal_generator.py
from typing import Optional, List
from decimal import Decimal
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Prediction, PredictionMarketSignal
from backend.trading_engine.scanners import MarketQuote

class SignalGenerator:
    """
    Compares your model predictions against market prices to generate trading signals.
    
    This bridges your existing Prediction model with prediction market opportunities.
    """
    
    def __init__(self, db_session: AsyncSession, min_edge_pct: float = 0.05):
        self.db = db_session
        self.min_edge_pct = min_edge_pct  # 5% minimum edge
        
    async def generate_signals(
        self, 
        quotes: List[MarketQuote]
    ) -> List[PredictionMarketSignal]:
        """
        Main entry: Scan all quotes, find edges against your predictions.
        """
        signals = []
        
        for quote in quotes:
            # Match market to your prediction
            prediction = await self._match_prediction(quote)
            if not prediction:
                continue
                
            signal = await self._calculate_signal(prediction, quote)
            if signal and signal.edge_percent >= self.min_edge_pct:
                signals.append(signal)
                
        return signals
    
    async def _match_prediction(self, quote: MarketQuote) -> Optional[Prediction]:
        """
        Map market ticker to your Prediction record.
        
        This requires maintaining a mapping between:
        - Kalshi tickers (e.g., "CBB-DUKE-20250325") 
        - Your game records (home_team="Duke", away_team="UNC", game_date=...)
        """
        # Parse ticker to extract teams and date
        # This is exchange-specific parsing
        parsed = self._parse_ticker(quote.ticker, quote.exchange)
        
        if not parsed:
            return None
            
        # Query your existing predictions
        from sqlalchemy import select
        from backend.models import Game
        
        stmt = (
            select(Prediction)
            .join(Game)
            .where(
                Game.home_team.ilike(f"%{parsed['home']}%"),
                Game.away_team.ilike(f"%{parsed['away']}%"),
                Game.game_date >= parsed['date'],
                Prediction.run_tier == "nightly"
            )
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _calculate_signal(
        self, 
        prediction: Prediction, 
        quote: MarketQuote
    ) -> Optional[PredictionMarketSignal]:
        """
        Calculate edge between your model and market price.
        """
        # Convert your prediction to binary probability
        # This depends on what the specific market asks
        market_question = self._fetch_market_question(quote.ticker)
        
        if "spread" in market_question.lower():
            model_prob = self._spread_to_prob(
                projected_margin=prediction.projected_margin,
                market_line=self._extract_spread(market_question),
                confidence=prediction.adjusted_sd
            )
        elif "moneyline" in market_question.lower() or "win" in market_question.lower():
            model_prob = self._moneyline_prob(prediction)
        else:
            return None
            
        # Calculate market's implied probability (remove vig)
        market_prob = self._remove_vig(quote.yes_bid, quote.yes_ask)
        
        # Edge calculation
        edge = model_prob - market_prob
        
        if edge <= self.min_edge_pct:
            return None
            
        # Kelly sizing (using your existing Kelly calculations)
        # For binary contracts: f* = (bp - q) / b
        # b = (1 - price) / price (decimal odds - 1)
        price_to_pay = quote.yes_ask  # Worst case: take the ask
        b = (1 - price_to_pay) / price_to_pay
        p = model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b if b > 0 else 0
        fractional_kelly = kelly * 0.25  # Your existing fractional approach
        
        # Create signal record
        signal = PredictionMarketSignal(
            prediction_id=prediction.id,
            exchange=quote.exchange,
            market_ticker=quote.ticker,
            contract_type=self._classify_contract(market_question),
            model_probability=model_prob,
            model_confidence=prediction.snr,
            market_yes_price=quote.yes_ask,
            market_implied_prob=market_prob,
            edge_percent=edge * 100,
            kelly_fraction=fractional_kelly,
            target_entry=market_prob + 0.01,  # Try to get slightly better than current
            max_position_dollars=self._calculate_position_size(fractional_kelly),
            status="PENDING"
        )
        
        return signal
    
    def _spread_to_prob(
        self, 
        projected_margin: float, 
        market_line: float,
        confidence: float
    ) -> float:
        """
        Convert projected margin to probability of covering market line.
        
        Uses normal distribution assumption.
        """
        from scipy import stats
        
        # Distance from projection to market line
        distance = projected_margin - market_line
        
        # Standard deviation (confidence is your adjusted_sd)
        std = confidence if confidence else 11.0  # Default CBB std
        
        # P(cover) = P(actual_margin > market_line)
        z = -distance / std  # Negative because we want P(X > line)
        prob = 1 - stats.norm.cdf(z)
        
        return prob
    
    def _calculate_position_size(self, kelly_fraction: float) -> float:
        """
        Calculate dollar position size based on bankroll and Kelly.
        
        This should reference your existing bankroll management.
        """
        # Your existing BetLog tracks bankroll_at_bet
        # Use a configured bankroll or track separately for prediction markets
        
        bankroll = self._get_pm_bankroll()  # Configured PM trading bankroll
        return bankroll * kelly_fraction
```

#### The Order Manager

```python
# backend/trading_engine/order_manager.py
import asyncio
from typing import Optional
from datetime import datetime
from decimal import Decimal

from backend.trading_engine.exchange_clients.base import ExchangeClient
from backend.trading_engine.models import Order, OrderStatus

class OrderManager:
    """
    Handles execution of trading signals.
    
    Responsibilities:
    - Convert signals to exchange orders
    - Handle order lifecycle (place, fill, cancel)
    - Track fills and update positions
    - Implement execution algos (TWAP, iceberg, etc.)
    """
    
    def __init__(self, exchange_clients: dict, db_session):
        self.clients = exchange_clients  # {"kalshi": client, "polymarket": client}
        self.db = db_session
        self.active_orders = {}  # order_id -> Order
        
    async def execute_signal(self, signal: PredictionMarketSignal) -> Optional[Order]:
        """
        Execute a trading signal by placing an order.
        """
        client = self.clients.get(signal.exchange)
        if not client:
            raise ValueError(f"No client configured for {signal.exchange}")
            
        # Check if we're already in this market
        existing_position = await self._get_position(signal.market_ticker)
        if existing_position:
            # Adjust sizing or skip
            return None
            
        # Determine order type based on urgency
        # For prediction markets, FAK (Fill and Kill) is often best
        order_type = self._select_order_type(signal)
        
        # Calculate size in contracts
        # Binary contracts: $1 payout, so cost = price * contracts
        contracts = int(signal.max_position_dollars / signal.target_entry)
        
        try:
            # Place order
            exchange_order = await client.create_order(
                ticker=signal.market_ticker,
                side="YES",
                size=contracts,
                price=signal.target_entry,  # Limit order
                order_type=order_type
            )
            
            # Record in database
            order = Order(
                signal_id=signal.id,
                exchange=signal.exchange,
                exchange_order_id=exchange_order['id'],
                ticker=signal.market_ticker,
                side="YES",
                size=contracts,
                price=signal.target_entry,
                status=OrderStatus.PENDING,
                created_at=datetime.utcnow()
            )
            
            self.db.add(order)
            await self.db.commit()
            
            self.active_orders[exchange_order['id']] = order
            
            # Update signal status
            signal.status = "ENTERED"
            signal.entry_price = exchange_order.get('avg_fill_price')
            signal.entry_time = datetime.utcnow()
            await self.db.commit()
            
            return order
            
        except Exception as e:
            # Log failure, potentially retry
            signal.status = "FAILED"
            signal.notes = str(e)
            await self.db.commit()
            return None
    
    async def monitor_fills(self):
        """
        Background task: Continuously poll for order fills.
        """
        while True:
            for order_id, order in list(self.active_orders.items()):
                client = self.clients[order.exchange]
                
                try:
                    status = await client.get_order_status(order_id)
                    
                    if status['state'] == 'FILLED':
                        order.status = OrderStatus.FILLED
                        order.filled_size = status['filled_size']
                        order.avg_fill_price = status['avg_fill_price']
                        order.filled_at = datetime.utcnow()
                        
                        # Update position tracking
                        await self._update_position(order)
                        
                        del self.active_orders[order_id]
                        
                    elif status['state'] == 'CANCELLED':
                        order.status = OrderStatus.CANCELLED
                        del self.active_orders[order_id]
                        
                except Exception as e:
                    # Log error, don't crash the loop
                    print(f"Error checking order {order_id}: {e}")
                    
            await asyncio.sleep(5)  # Poll every 5 seconds
```

### 2.5 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAILWAY DEPLOYMENT                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FASTAPI WEB SERVER                              │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Routes:                                                      │ │   │
│  │  │  ├── GET /api/predictions (existing)                          │ │   │
│  │  │  ├── GET /api/bets (existing)                                 │ │   │
│  │  │  ├── GET /api/pm/signals (NEW: view active signals)           │ │   │
│  │  │  ├── GET /api/pm/positions (NEW: current positions)           │ │   │
│  │  │  └── POST /api/pm/panic (NEW: emergency close all)            │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │  Scale: 1-2 instances (standard web load)                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  TRADING WORKER (NEW)                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Asyncio event loop running:                                  │ │   │
│  │  │  ├── Market scanner (every 30s)                               │ │   │
│  │  │  ├── Signal generator (on new predictions)                    │ │   │
│  │  │  ├── Order manager (on signals)                               │ │   │
│  │  │  ├── Position tracker (continuous)                            │ │   │
│  │  │  └── Risk manager (continuous)                                │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │  Scale: 1 dedicated instance (CPU-optimized)                        │   │
│  │  Note: Not horizontally scalable (stateful)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              POSTGRESQL (Railway managed)                           │   │
│  │  Existing tables + new PM tables:                                   │   │
│  │  ├── pm_signals                                                     │   │
│  │  ├── pm_orders                                                      │   │
│  │  ├── pm_positions                                                   │   │
│  │  └── pm_trades                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The "Private Syndicate" Pivot (Secondary Strategy)

### 3.1 Concept

Build a **prediction market experience** for your Discord community without the regulatory complexity:

- Users trade with **virtual points** (not real money)
- Markets based on **your model's predictions**
- Community competes on **prediction accuracy** and **trading P&L**
- Leaderboards, achievements, bragging rights

### 3.2 PostgreSQL-Based Matching Engine

This can run entirely on your existing stack.

#### Schema Extensions

```sql
-- Virtual currency system
CREATE TABLE syndicate_users (
    id SERIAL PRIMARY KEY,
    discord_id VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    points_balance DECIMAL(12,2) DEFAULT 10000.00,  -- Starting bankroll
    lifetime_pnl DECIMAL(12,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Internal prediction markets
CREATE TABLE syndicate_markets (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    question TEXT NOT NULL,
    market_type VARCHAR(20) CHECK (market_type IN ('SPREAD', 'MONEYLINE', 'TOTAL')),
    
    -- The line being offered
    line_value DECIMAL(6,1),  -- e.g., -5.5 for spread
    
    -- Your model's view (displayed to users)
    model_probability DECIMAL(5,4),  -- 0.0000 to 1.0000
    model_confidence VARCHAR(10) CHECK (model_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    
    -- Market state
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'SETTLED')),
    
    -- Settlement
    actual_result BOOLEAN,  -- TRUE = YES, FALSE = NO
    settled_at TIMESTAMP,
    
    opens_at TIMESTAMP,
    closes_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Central Limit Order Book (simplified)
CREATE TABLE syndicate_orders (
    id SERIAL PRIMARY KEY,
    market_id INTEGER REFERENCES syndicate_markets(id),
    user_id INTEGER REFERENCES syndicate_users(id),
    
    side VARCHAR(4) CHECK (side IN ('YES', 'NO')),
    price DECIMAL(4,2) CHECK (price >= 0.01 AND price <= 0.99),  -- 1¢ to 99¢
    size INTEGER CHECK (size > 0),  -- Number of contracts
    filled INTEGER DEFAULT 0,
    
    order_type VARCHAR(10) DEFAULT 'LIMIT' CHECK (order_type IN ('LIMIT', 'MARKET')),
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'FILLED', 'CANCELLED')),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for efficient order book queries
CREATE INDEX idx_orders_market_side_price ON syndicate_orders(market_id, side, price DESC) 
    WHERE status = 'OPEN';

-- Trades (fills)
CREATE TABLE syndicate_trades (
    id SERIAL PRIMARY KEY,
    market_id INTEGER REFERENCES syndicate_markets(id),
    
    -- Matched orders
    bid_order_id INTEGER REFERENCES syndicate_orders(id),
    ask_order_id INTEGER REFERENCES syndicate_orders(id),
    
    -- Trade details
    price DECIMAL(4,2),  -- Execution price
    size INTEGER,
    
    -- Both parties
    buyer_id INTEGER REFERENCES syndicate_users(id),
    seller_id INTEGER REFERENCES syndicate_users(id),
    
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Positions (holdings)
CREATE TABLE syndicate_positions (
    id SERIAL PRIMARY KEY,
    market_id INTEGER REFERENCES syndicate_markets(id),
    user_id INTEGER REFERENCES syndicate_users(id),
    
    side VARCHAR(4),  -- YES or NO
    contracts_held INTEGER DEFAULT 0,
    avg_entry_price DECIMAL(4,2),
    
    realized_pnl DECIMAL(12,2) DEFAULT 0.00,
    unrealized_pnl DECIMAL(12,2) DEFAULT 0.00,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(market_id, user_id, side)
);
```

#### Matching Engine (Python)

```python
# backend/syndicate/matching_engine.py
from typing import List, Optional, Tuple
from decimal import Decimal
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.syndicate.models import (
    SyndicateMarket, SyndicateOrder, SyndicateTrade, 
    SyndicatePosition, SyndicateUser
)

class SyndicateMatchingEngine:
    """
    Simple CLOB matching engine using PostgreSQL.
    
    NOT production-exchange grade, but sufficient for:
    - <100 concurrent users
    - <1000 open orders
    - <100 trades per hour
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        
    async def place_order(
        self,
        market_id: int,
        user_id: int,
        side: str,  # 'YES' or 'NO'
        price: Decimal,
        size: int,
        order_type: str = 'LIMIT'
    ) -> SyndicateOrder:
        """
        Place a new order and attempt to match immediately.
        """
        # Validate user has sufficient points
        user = await self.db.get(SyndicateUser, user_id)
        required = price * size
        
        if user.points_balance < required:
            raise ValueError(f"Insufficient balance: {user.points_balance} < {required}")
        
        # Reserve the funds
        user.points_balance -= required
        
        # Create order
        order = SyndicateOrder(
            market_id=market_id,
            user_id=user_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            status='OPEN'
        )
        self.db.add(order)
        await self.db.flush()  # Get order.id
        
        # Attempt matching
        fills = await self._match_order(order)
        
        await self.db.commit()
        return order
    
    async def _match_order(self, incoming: SyndicateOrder) -> List[SyndicateTrade]:
        """
        Price-time priority matching algorithm.
        """
        fills = []
        
        # Find matching orders
        # YES buy matches against YES sell (or NO buy, depending on design)
        # For binary markets: YES bid at P = NO ask at (1-P)
        
        if incoming.side == 'YES':
            # Match against resting YES asks (sells)
            # Or equivalently: NO bids (buying NO = selling YES)
            matches = await self._find_matches(
                market_id=incoming.market_id,
                opposite_side='YES',  # Looking for YES sellers
                max_price=incoming.price,
                exclude_user=incoming.user_id
            )
        else:
            matches = await self._find_matches(
                market_id=incoming.market_id,
                opposite_side='NO',
                max_price=incoming.price,
                exclude_user=incoming.user_id
            )
        
        remaining = incoming.size - incoming.filled
        
        for match in matches:
            if remaining <= 0:
                break
                
            fill_size = min(remaining, match.size - match.filled)
            fill_price = match.price  # Price of resting order
            
            # Execute fill
            trade = await self._execute_fill(
                market_id=incoming.market_id,
                bid_order=incoming if incoming.side == 'YES' else match,
                ask_order=match if incoming.side == 'YES' else incoming,
                price=fill_price,
                size=fill_size
            )
            
            fills.append(trade)
            remaining -= fill_size
            
        return fills
    
    async def _find_matches(
        self,
        market_id: int,
        opposite_side: str,
        max_price: Decimal,
        exclude_user: int
    ) -> List[SyndicateOrder]:
        """
        Find resting orders that match the incoming order.
        
        Sorted by price (best first), then time.
        """
        from sqlalchemy import asc
        
        stmt = (
            select(SyndicateOrder)
            .where(
                SyndicateOrder.market_id == market_id,
                SyndicateOrder.side == opposite_side,
                SyndicateOrder.price <= max_price,
                SyndicateOrder.user_id != exclude_user,
                SyndicateOrder.status == 'OPEN',
                SyndicateOrder.filled < SyndicateOrder.size
            )
            .order_by(asc(SyndicateOrder.price), asc(SyndicateOrder.created_at))
            .limit(10)  # Batch size
        )
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def _execute_fill(
        self,
        market_id: int,
        bid_order: SyndicateOrder,
        ask_order: SyndicateOrder,
        price: Decimal,
        size: int
    ) -> SyndicateTrade:
        """
        Record a trade and update both parties' positions.
        """
        # Create trade record
        trade = SyndicateTrade(
            market_id=market_id,
            bid_order_id=bid_order.id,
            ask_order_id=ask_order.id,
            price=price,
            size=size,
            buyer_id=bid_order.user_id,
            seller_id=ask_order.user_id
        )
        self.db.add(trade)
        
        # Update orders
        bid_order.filled += size
        if bid_order.filled >= bid_order.size:
            bid_order.status = 'FILLED'
            
        ask_order.filled += size
        if ask_order.filled >= ask_order.size:
            ask_order.status = 'FILLED'
        
        # Update or create positions
        await self._update_position(market_id, bid_order.user_id, 'YES', size, price)
        await self._update_position(market_id, ask_order.user_id, 'NO', size, price)
        
        # Handle payment
        # Buyer pays: price * size
        # Seller receives: price * size (minus any fees)
        buyer = await self.db.get(SyndicateUser, bid_order.user_id)
        # Already reserved, now deduct
        
        seller = await self.db.get(SyndicateUser, ask_order.user_id)
        seller.points_balance += price * size
        
        await self.db.flush()
        return trade
    
    async def settle_market(self, market_id: int, result: bool):
        """
        Settle all positions when market resolves.
        
        result=True: YES pays $1, NO pays $0
        result=False: YES pays $0, NO pays $1
        """
        market = await self.db.get(SyndicateMarket, market_id)
        market.status = 'SETTLED'
        market.actual_result = result
        market.settled_at = datetime.utcnow()
        
        # Get all positions
        stmt = select(SyndicatePosition).where(
            SyndicatePosition.market_id == market_id
        )
        result_set = await self.db.execute(stmt)
        positions = result_set.scalars().all()
        
        for pos in positions:
            user = await self.db.get(SyndicateUser, pos.user_id)
            
            # Calculate payout
            if result and pos.side == 'YES':
                # Won: Receive $1 per contract
                payout = pos.contracts_held * Decimal('1.00')
                pnl = payout - (pos.avg_entry_price * pos.contracts_held)
            elif not result and pos.side == 'NO':
                payout = pos.contracts_held * Decimal('1.00')
                pnl = payout - (pos.avg_entry_price * pos.contracts_held)
            else:
                # Lost: Receive $0
                payout = Decimal('0')
                pnl = -pos.avg_entry_price * pos.contracts_held
            
            user.points_balance += payout
            user.lifetime_pnl += pnl
            pos.realized_pnl = pnl
            
        await self.db.commit()
```

#### Discord Bot Integration

```python
# backend/discord/syndicate_bot.py
import discord
from discord.ext import commands
from sqlalchemy.ext.asyncio import AsyncSession

from backend.syndicate.matching_engine import SyndicateMatchingEngine
from backend.syndicate.models import SyndicateMarket

class SyndicateCog(commands.Cog):
    """
    Discord bot commands for the private prediction market.
    """
    
    def __init__(self, bot: commands.Bot, db_session: AsyncSession):
        self.bot = bot
        self.db = db_session
        self.engine = SyndicateMatchingEngine(db_session)
        
    @commands.command()
    async def markets(self, ctx: commands.Context):
        """List all open markets"""
        stmt = select(SyndicateMarket).where(SyndicateMarket.status == 'OPEN')
        result = await self.db.execute(stmt)
        markets = result.scalars().all()
        
        embed = discord.Embed(title="🎯 Open Prediction Markets", color=0x00ff00)
        
        for market in markets:
            embed.add_field(
                name=f"#{market.id}: {market.question}",
                value=f"Model says: {market.model_probability:.0%} ({market.model_confidence})",
                inline=False
            )
            
        await ctx.send(embed=embed)
    
    @commands.command()
    async def book(self, ctx: commands.Context, market_id: int):
        """Show order book for a market"""
        # Fetch order book
        stmt = select(SyndicateOrder).where(
            SyndicateOrder.market_id == market_id,
            SyndicateOrder.status == 'OPEN'
        )
        result = await self.db.execute(stmt)
        orders = result.scalars().all()
        
        yes_bids = [o for o in orders if o.side == 'YES']
        yes_asks = [o for o in orders if o.side == 'YES' and o.filled < o.size]
        
        embed = discord.Embed(title=f"📊 Order Book - Market #{market_id}")
        
        # Bids ( buyers)
        bid_text = "\n".join([
            f"{o.price:.0%} | {o.size - o.filled} contracts"
            for o in sorted(yes_bids, key=lambda x: x.price, reverse=True)[:5]
        ]) or "No bids"
        
        embed.add_field(name="YES Bids (Buying)", value=bid_text, inline=True)
        
        # Asks
        ask_text = "\n".join([
            f"{o.price:.0%} | {o.size - o.filled} contracts"
            for o in sorted(yes_asks, key=lambda x: x.price)[:5]
        ]) or "No asks"
        
        embed.add_field(name="YES Asks (Selling)", value=ask_text, inline=True)
        
        await ctx.send(embed=embed)
    
    @commands.command()
    async def buy(self, ctx: commands.Context, market_id: int, price: float, size: int):
        """Place a bid to buy YES contracts"""
        # Get or create user
        user = await self._get_or_create_user(ctx.author.id, ctx.author.name)
        
        try:
            order = await self.engine.place_order(
                market_id=market_id,
                user_id=user.id,
                side='YES',
                price=Decimal(str(price)),
                size=size
            )
            
            await ctx.send(
                f"✅ Order placed! Buying {size} YES @ {price:.0%}\n"
                f"Order ID: {order.id}"
            )
        except ValueError as e:
            await ctx.send(f"❌ Error: {e}")
    
    @commands.command()
    async def balance(self, ctx: commands.Context):
        """Check your points balance and P&L"""
        user = await self._get_or_create_user(ctx.author.id, ctx.author.name)
        
        embed = discord.Embed(title=f"💰 {ctx.author.name}'s Balance")
        embed.add_field(name="Available Points", value=f"{user.points_balance:,.0f}")
        embed.add_field(name="Lifetime P&L", value=f"{user.lifetime_pnl:+,.0f}")
        
        await ctx.send(embed=embed)
```

### 3.3 Why This Works

| Aspect | Public Exchange | Your Private Syndicate |
|--------|----------------|----------------------|
| **Regulatory** | CFTC/state licensing required | Virtual points = not gambling |
| **Capital** | $5-10M+ required | $0 (existing infrastructure) |
| **Users** | Need thousands for liquidity | Works with 10-100 |
| **Tech** | Custom matching engine | PostgreSQL-based (sufficient) |
| **Revenue** | Trading fees | Community engagement, potential subscription |
| **Risk** | High (operational, legal) | Low (community building) |

---

## 4. Regulatory & API Constraints

### 4.1 Platform Comparison Matrix

| Dimension | Kalshi | Polymarket | Robinhood Event Contracts |
|-----------|--------|------------|---------------------------|
| **Regulatory Status** | CFTC DCM (Designated Contract Market) | Offshore, CFTC fined $1.4M in 2022 (now has QCX LLC DCM) | State-regulated (varies) |
| **US Access** | 42 states (excludes AZ, IL, MA, MD, MI, MT, NJ, OH) | Gray area (no explicit block, no US license) | Licensed per state |
| **Settlement** | USD (fiat) | USDC on Polygon | USD (fiat) |
| **Trading API** | REST + WebSocket + FIX 4.4 | Gamma + CLOB + Data APIs + WebSocket | No public trading API |
| **Rate Limits** | 10 req/s (standard), higher tiers available | ~100/min CLOB, 15K/10s Gamma | N/A |
| **Auth Method** | RSA-PSS signed requests + JWT (30min expiry) | HMAC L2 headers + EIP-712 signing | OAuth 2.0 (consumer only) |
| **Python SDK** | kalshi-python (official) | py-clob-client (official) | None |
| **Webhooks** | ❌ No | ❌ No | ❌ No |
| **Real-time Data** | WebSocket (orderbook, trades) | WebSocket (multiple channels) | None |
| **Fees** | 0% maker, 0% taker (promotional) | 0% (gasless via relayer) | Built into spread |
| **Sports Markets** | Yes (CBB, NBA, NFL, etc.) | Yes (limited) | Yes (growing) |

### 4.2 Kalshi Integration Details

**Authentication Flow:**
```python
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import httpx

class KalshiAuth:
    """
    Kalshi requires RSA-PSS signed requests.
    """
    
    def __init__(self, key_id: str, private_key_pem: str):
        self.key_id = key_id
        self.private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        
    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        """
        Create signed headers for Kalshi API.
        """
        timestamp = str(int(time.time()))
        msg_string = timestamp + method.upper() + path + body
        
        signature = self.private_key.sign(
            msg_string.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256()
        )
        
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }
    
    async def place_order(self, ticker: str, side: str, count: int, price: int):
        """
        Place an order on Kalshi.
        
        Price in cents: 0-100 (e.g., 63 = 63¢ = 63% implied probability)
        """
        path = "/orders"
        body = json.dumps({
            "ticker": ticker,
            "side": side,  # "yes" or "no"
            "count": count,
            "price": price,
            "type": "limit"
        })
        
        headers = self._sign_request("POST", path, body)
        headers["Content-Type"] = "application/json"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{path}",
                headers=headers,
                content=body
            )
            return response.json()
```

**Geographic Restrictions:**
```python
KALSHI_BLOCKED_STATES = {
    'AZ',  # Arizona
    'IL',  # Illinois  
    'MA',  # Massachusetts
    'MD',  # Maryland
    'MI',  # Michigan
    'MT',  # Montana
    'NJ',  # New Jersey
    'OH',  # Ohio
}

def can_trade_kalshi(ip_address: str) -> bool:
    """
    Check if user IP is in allowed jurisdiction.
    Note: Kalshi enforces this server-side; this is for UX only.
    """
    # Use GeoIP service to check state
    location = geolocate_ip(ip_address)
    return location['state'] not in KALSHI_BLOCKED_STATES
```

### 4.3 Polymarket Integration Details

**Architecture Complexity:**
Polymarket uses THREE separate APIs:

1. **Gamma API** (`gamma-api.polymarket.com`): Market discovery, metadata
2. **CLOB API** (`clob.polymarket.com`): Trading, order book
3. **Data API** (`data-api.polymarket.com`): Historical trades, positions

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

class PolymarketTrader:
    """
    Polymarket requires understanding of:
    - Proxy wallets (Gnosis Safe)
    - EIP-712 typed data signing
    - Token IDs (each outcome has unique token)
    - Conditional Tokens Framework (CTF)
    """
    
    def __init__(self, private_key: str, proxy_address: str):
        # Private key controls EOA (Externally Owned Account)
        # But funds are in proxy (Gnosis Safe)
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,  # Polygon
            signature_type=2,  # Gnosis Safe proxy
            funder=proxy_address  # Where the USDC lives
        )
        
        # Set API credentials (HMAC for L2 authentication)
        creds = ApiCreds(
            api_key=os.getenv("POLYMARKET_API_KEY"),
            api_secret=os.getenv("POLYMARKET_API_SECRET"),
            api_passphrase=os.getenv("POLYMARKET_PASSPHRASE")
        )
        self.client.set_api_creds(creds)
    
    async def get_order_book(self, token_id: str):
        """Fetch CLOB order book for a specific outcome token."""
        return self.client.get_order_book(token_id)
    
    async def place_limit_order(
        self, 
        token_id: str, 
        side: str,  # "BUY" or "SELL"
        size: float,  # In USDC
        price: float  # 0.00 to 1.00
    ):
        """
        Place a limit order on Polymarket.
        
        Note: Size is in USDC, not share count.
        Price is 0.00-1.00 representing 0-99¢ per share.
        """
        # Convert to ticks (Polymarket uses 0.01 increments)
        from py_clob_client.utilities import price_to_tick
        
        order_args = OrderArgs(
            token_id=token_id,
            side=side,
            size=size,
            price=price_to_tick(price),  # Converts to proper tick format
        )
        
        # Create and sign order (EIP-712)
        signed_order = self.client.create_order(order_args)
        
        # Submit to CLOB
        response = self.client.post_order(
            signed_order,
            order_type=OrderType.GTC  # Good Till Cancelled
        )
        
        return response
```

**Rate Limits (Polymarket):**
```
Gamma API:      4,000 requests / 10 seconds
CLOB API (GET): 15,000 requests / 10 seconds  
CLOB API (POST /order): 3,500 / 10 seconds burst, 36,000 / 10 min sustained
Data API:       1,000 requests / 10 seconds
WebSocket:      20 simultaneous subscriptions
```

### 4.4 Robinhood Event Contracts

**Current Status:**
- **No public trading API** — Consumer mobile/web only
- **OAuth 2.0** for read-only portfolio access
- **Limited automation potential** — would require browser automation ( fragile)

**Recommendation:** Monitor for API launch, but don't build around it yet.

### 4.5 Compliance Checklist

If pursuing the Automated Quant strategy:

| Requirement | Kalshi | Polymarket | Action Required |
|-------------|--------|------------|-----------------|
| KYC/AML | ✅ Required | ⚠️ Minimal | Complete identity verification |
| Accreditation | ❌ No | ❌ No | None |
| Tax Reporting | ✅ 1099 issued | ⚠️ Self-reported | Track all transactions |
| IP Geoblocking | ✅ Enforced | ⚠️ Honor system | Respect restrictions |
| Trading Limits | ✅ Position limits | ⚠️ None | Monitor exposure |
| API TOS | ✅ Read and accept | ✅ Read and accept | Legal review recommended |

**Recommended Legal Structure for Quant Operation:**
```
Option A: Individual Trading
- Trade in personal account
- Report as Schedule C business income
- Deduct data/software costs

Option B: LLC (Recommended at scale)
- Form single-member LLC
- Elect S-corp taxation if >$50K profit
- Separate business bank account
- Professional liability insurance

Option C: Fund Structure (>$1M AUM)
- Form Delaware LLC/LP
- Engage fund administrator
- SEC registration (if >$150M)
- Qualified purchaser requirements
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Kalshi and Polymarket testnet/sandbox accounts
- [ ] Implement exchange client abstractions (`exchange_clients/`)
- [ ] Build market scanner (read-only)
- [ ] Create signal translation layer (spread → binary probability)

### Phase 2: Paper Trading (Weeks 3-4)
- [ ] Build signal generator with paper trade tracking
- [ ] Implement order manager (simulate execution)
- [ ] Create monitoring dashboard
- [ ] Backtest signals against historical market data

### Phase 3: Live Trading (Weeks 5-6)
- [ ] Enable live order placement (small size: $10-50 per trade)
- [ ] Implement risk manager (circuit breakers, max loss)
- [ ] Add alerting (Discord/Slack webhooks)
- [ ] Daily performance reconciliation

### Phase 4: Scale (Weeks 7-8)
- [ ] Increase position sizing based on track record
- [ ] Add market making strategies (provide liquidity)
- [ ] Implement cross-exchange arbitrage
- [ ] Consider private syndicate launch

### Phase 5: Private Syndicate (Parallel Track)
- [ ] Build PostgreSQL matching engine
- [ ] Create Discord bot commands
- [ ] Launch beta with 10-20 community members
- [ ] Iterate on UX based on feedback

---

## 6. Risk Management Framework

### Trading Limits (Hard Stops)

```python
# backend/trading_engine/risk_manager.py

RISK_LIMITS = {
    # Position limits
    "max_position_per_market": 500.0,      # $500 max in any single market
    "max_correlated_exposure": 2000.0,      # $2000 across correlated markets
    "max_total_exposure": 10000.0,          # $10K total at risk
    
    # Daily limits
    "max_daily_loss": 1000.0,               # Stop trading after -$1K day
    "max_daily_trades": 50,                 # Prevent overtrading
    "max_daily_volume": 10000.0,            # $10K turnover limit
    
    # Edge requirements
    "min_edge_percent": 0.05,               # 5% minimum edge
    "min_model_confidence": 0.3,            # SNR threshold from your model
    
    # Market restrictions
    "min_market_liquidity": 1000.0,         # $1K daily volume minimum
    "max_market_age_hours": 48,             # Don't trade markets >48h old
    "banned_markets": [                     # Manual exclusions
        "politics",                         # Avoid regulatory scrutiny
        "crypto",                           # Too volatile
    ]
}

class RiskManager:
    """
    Circuit breakers and risk checks before any trade.
    """
    
    async def check_trade_allowed(self, signal: PredictionMarketSignal) -> tuple[bool, str]:
        """
        Pre-trade risk check. Returns (allowed, reason).
        """
        # Check edge
        if signal.edge_percent < RISK_LIMITS["min_edge_percent"] * 100:
            return False, f"Edge {signal.edge_percent:.1f}% below minimum"
            
        # Check position limit
        current_position = await self.get_position(signal.market_ticker)
        new_exposure = current_position + signal.max_position_dollars
        if new_exposure > RISK_LIMITS["max_position_per_market"]:
            return False, f"Would exceed ${RISK_LIMITS['max_position_per_market']} market limit"
            
        # Check daily loss
        daily_pnl = await self.get_daily_pnl()
        if daily_pnl < -RISK_LIMITS["max_daily_loss"]:
            return False, f"Daily loss limit hit: ${daily_pnl:.2f}"
            
        # Check market liquidity
        market_data = await self.get_market_data(signal.market_ticker)
        if market_data["volume_24h"] < RISK_LIMITS["min_market_liquidity"]:
            return False, f"Insufficient liquidity: ${market_data['volume_24h']:.0f}"
            
        return True, "OK"
```

### Circuit Breakers

```python
async def global_circuit_breaker():
    """
    Emergency stop for entire trading system.
    Triggers on:
    - API authentication failure
    - Unexpected balance changes
    - Connectivity issues
    - Manual kill switch
    """
    while True:
        # Check health
        health = await check_exchange_health()
        
        if not health["kalshi_connected"] and not health["polymarket_connected"]:
            await emergency_close_all_positions()
            await alert_ops("CRITICAL: All exchange connections lost")
            
        if health["daily_loss"] > EMERGENCY_LOSS_THRESHOLD:
            await emergency_close_all_positions()
            await alert_ops(f"CRITICAL: Daily loss ${health['daily_loss']:.2f}")
            
        await asyncio.sleep(30)
```

---

## 7. Conclusion

### Summary

| Strategy | Investment | Timeline | Expected Outcome |
|----------|-----------|----------|------------------|
| **Build Public Exchange** | $5-10M, 2-3 years | 24-36 months | Regulatory nightmare, don't do this |
| **Automated Quant** | $50K-100K, 2-3 months | 2-3 months | Monetize existing models, scalable |
| **Private Syndicate** | $10K-20K, 1 month | 4-6 weeks | Community engagement, zero regulatory risk |

### Recommendation

**Primary:** Pursue the **Automated Quant** strategy immediately. Your existing models have demonstrated edge in CBB; prediction markets are less efficient than traditional sportsbooks and offer:
- No vig (Kalshi/Polymarket charge 0%)
- Transparent order books
- API access for automation
- Growing liquidity in sports markets

**Secondary:** Launch the **Private Syndicate** as a community engagement tool. It:
- Builds brand loyalty
- Generates training data for model improvement
- Creates a funnel for potential subscribers
- Zero regulatory risk

**Avoid:** Building a public exchange. Your current stack cannot support it, and the regulatory/compliance burden far exceeds any potential return.

### Next Steps

1. **This Week:** Create sandbox accounts on Kalshi and Polymarket
2. **Week 2:** Implement read-only market scanning
3. **Week 3:** Build signal translation layer
4. **Week 4:** Begin paper trading with $100/week "virtual" bankroll
5. **Month 2:** Evaluate results, decide on live trading capital

---

## Appendices

### A. Key Libraries

```
# Core trading
pip install py-clob-client          # Polymarket
pip install kalshi-python           # Kalshi official
pip install web3                    # Blockchain interactions

# Async infrastructure  
pip install httpx                   # Async HTTP
pip install websockets              # WebSocket client
pip install aiohttp                 # Alternative async HTTP

# Data & analysis
pip install pandas                  # Data manipulation
pip install numpy                   # Numerical computing
pip install scipy                   # Statistical functions

# Security
pip install cryptography            # RSA-PSS signing for Kalshi
```

### B. Useful Resources

- [Kalshi API Documentation](https://trading-api.readme.io/reference/)
- [Polymarket CLOB Client](https://github.com/Polymarket/py-clob-client)
- [Prediction Market Trading Layer Guide](https://agentbets.ai/guides/prediction-market-trading-layer/)
- [CFTC Event Contracts Regulation](https://www.cftc.gov/sites/default/files/2024-12/24-08%20Event%20Contracts%20Federal%20Register.pdf)

### C. Database Migration Script

```sql
-- Run this to add prediction market tables to your existing database

BEGIN;

-- Prediction market signals
CREATE TABLE pm_signals (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    exchange VARCHAR(20) NOT NULL,
    market_ticker VARCHAR(100) NOT NULL,
    contract_type VARCHAR(20),
    model_probability DECIMAL(5,4),
    model_confidence DECIMAL(5,2),
    market_yes_price DECIMAL(4,2),
    market_implied_prob DECIMAL(5,4),
    edge_percent DECIMAL(5,2),
    kelly_fraction DECIMAL(6,4),
    target_entry DECIMAL(4,2),
    max_position_dollars DECIMAL(10,2),
    exit_trigger_win DECIMAL(4,2),
    exit_trigger_loss DECIMAL(4,2),
    status VARCHAR(20) DEFAULT 'PENDING',
    entry_price DECIMAL(4,2),
    entry_time TIMESTAMP,
    exit_price DECIMAL(4,2),
    pnl DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Orders placed
CREATE TABLE pm_orders (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES pm_signals(id),
    exchange VARCHAR(20) NOT NULL,
    exchange_order_id VARCHAR(100),
    ticker VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size INTEGER NOT NULL,
    price DECIMAL(4,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    filled_size INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(4,2),
    created_at TIMESTAMP DEFAULT NOW(),
    filled_at TIMESTAMP
);

-- Positions held
CREATE TABLE pm_positions (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    ticker VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    contracts INTEGER DEFAULT 0,
    avg_entry_price DECIMAL(4,2),
    unrealized_pnl DECIMAL(10,2) DEFAULT 0,
    realized_pnl DECIMAL(10,2) DEFAULT 0,
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    UNIQUE(exchange, ticker, side)
);

-- Trades executed
CREATE TABLE pm_trades (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES pm_orders(id),
    exchange VARCHAR(20) NOT NULL,
    ticker VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size INTEGER NOT NULL,
    price DECIMAL(4,2) NOT NULL,
    fee DECIMAL(10,4) DEFAULT 0,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_pm_signals_status ON pm_signals(status);
CREATE INDEX idx_pm_signals_prediction ON pm_signals(prediction_id);
CREATE INDEX idx_pm_orders_status ON pm_orders(status);
CREATE INDEX idx_pm_positions_open ON pm_positions(closed_at) WHERE closed_at IS NULL;

COMMIT;
```

---

*Document Version: 1.0*  
*Prepared for: CBB Edge Analyzer Development Team*  
*Classification: Internal Use Only*
