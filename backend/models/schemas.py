from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class StrategyType(str, Enum):
    momentum = "momentum"
    sentiment = "sentiment"
    hybrid_ml = "hybrid_ml"


class BacktestRequest(BaseModel):
    strategy: StrategyType
    initial_capital: float = Field(default=100000, ge=1000)
    hold_days: int = Field(default=5, ge=1, le=60)
    position_size_pct: float = Field(default=0.02, ge=0.001, le=0.5)
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.05)
    train_pct: float = Field(default=0.70, ge=0.5, le=0.9)
    momentum_threshold: float = Field(default=0.5)
    sentiment_threshold: float = Field(default=0.1)
    fused_filename: Optional[str] = None
    price_filename: Optional[str] = None


class MetricsResponse(BaseModel):
    CAGR: float
    Sharpe: float
    Max_Drawdown: float
    Win_Pct: float
    Total_Trades: int
    Total_PnL: float
    Final_Capital: float
    Initial_Capital: float


class EquityPoint(BaseModel):
    date: str
    value: float


class TradeRecord(BaseModel):
    entry_date: str
    exit_date: str
    stock: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    win: int


class BacktestResponse(BaseModel):
    strategy_name: str
    metrics: MetricsResponse
    equity_curve: List[EquityPoint]
    drawdown: List[EquityPoint]
    trades: List[TradeRecord]
    run_id: str


class CompareRequest(BaseModel):
    strategies: List[StrategyType]
    initial_capital: float = Field(default=100000, ge=1000)
    hold_days: int = Field(default=5, ge=1, le=60)
    position_size_pct: float = Field(default=0.02, ge=0.001, le=0.5)
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.05)
    train_pct: float = Field(default=0.70, ge=0.5, le=0.9)
    momentum_threshold: float = Field(default=0.5)
    sentiment_threshold: float = Field(default=0.1)


class CompareResponse(BaseModel):
    results: Dict[str, Any]
    comparison_table: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    filename: str
    rows: int
    columns: List[str]
    message: str


class ErrorResponse(BaseModel):
    detail: str
    code: str
