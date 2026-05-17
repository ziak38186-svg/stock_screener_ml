from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.schemas import BacktestRequest, BacktestResponse, CompareRequest, CompareResponse
from services.backtest_service import execute_backtest, execute_compare

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    try:
        result = execute_backtest(request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing column in dataset: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.post("/compare", response_model=CompareResponse)
async def compare_strategies(request: CompareRequest):
    if not request.strategies:
        raise HTTPException(status_code=422, detail="At least one strategy required")
    try:
        result = execute_compare(request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/strategies")
async def list_strategies():
    return {
        "strategies": [
            {"id": "momentum", "name": "Momentum Only", "description": "Trades based on Momentum_score threshold"},
            {"id": "sentiment", "name": "Sentiment Only", "description": "Trades based on Sentiment_score threshold"},
            {"id": "hybrid_ml", "name": "Hybrid ML", "description": "Logistic Regression ML signal (BUY column)"},
        ]
    }
