"""functionSPACE Trading Protocol - FastAPI Backend

Exposes REST API endpoints for the simulation core.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
from core import Market, Position

# Initialize FastAPI app
app = FastAPI(
    title="functionSPACE Trading Protocol API",
    description="High-fidelity cryptoeconomic simulation backend",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global market instance (stateful)
market = Market()


# Pydantic models for request/response validation
class CreateMarketRequest(BaseModel):
    L: float = Field(..., description="Lower bound of outcome space")
    H: float = Field(..., description="Upper bound of outcome space")
    K: int = Field(..., description="Number of discretization points", ge=1)
    P0: float = Field(..., description="Initial pseudocount mass", gt=0)
    mu: float = Field(..., description="Minting ratio", gt=0)
    eps_alpha: float = Field(..., description="Minimum alpha floor", gt=0)
    title: str = Field(..., description="Semantic title for the market")
    tau: float = Field(0.01, description="Settlement eligibility threshold", gt=0)
    gamma: float = Field(1.0, description="Settlement temperature parameter", gt=0)
    lambda_s: float = Field(0.5, description="Settlement claim share weight", ge=0, le=1)
    lambda_d: float = Field(0.5, description="Settlement accuracy share weight", ge=0, le=1)
    overwrite: bool = Field(False, description="Allow overwriting existing market")


class BuyRequest(BaseModel):
    C: float = Field(..., description="Input collateral", gt=0)
    p_vector: List[float] = Field(..., description="Belief vector (must sum to 1)")


class StateFileRequest(BaseModel):
    filepath: str = Field(..., description="Path to state file")


class BuyWithParamsRequest(BaseModel):
    C: float = Field(..., description="Input collateral", gt=0)
    mean: float = Field(..., description="Mean of normal distribution")
    std_dev: float = Field(..., description="Standard deviation", gt=0)


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - serve the frontend."""
    return FileResponse("index.html")


@app.post("/market/create")
async def create_market(req: CreateMarketRequest):
    """Create a new market with specified parameters."""
    try:
        market.create_market(
            L=req.L,
            H=req.H,
            K=req.K,
            P0=req.P0,
            mu=req.mu,
            eps_alpha=req.eps_alpha,
            title=req.title,
            tau=req.tau,
            gamma=req.gamma,
            lambda_s=req.lambda_s,
            lambda_d=req.lambda_d,
            overwrite=req.overwrite
        )
        return {
            "success": True,
            "message": "Market created successfully",
            "params": market.market_params
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/market/state")
async def get_market_state():
    """Get current market state."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    return {
        "alpha_vector": market.alpha_vector.tolist(),
        "market_params": market.market_params,
        "is_settled": market.is_settled,
        "settlement_outcome": market.settlement_outcome,
        "num_positions": len(market.positions_db),
        "title": market.title,
        "current_pool": market.get_current_pool()
    }


@app.get("/market/positions")
async def get_positions():
    """Get all active positions."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    positions = []
    for pos_id, position in market.positions_db.items():
        positions.append({
            "position_id": position.position_id,
            "belief_p": position.belief_p,
            "minted_claims_m": position.minted_claims_m,
            "input_collateral_C": position.input_collateral_C,
            "status": position.status,
            "sold_price": position.sold_price
        })
    
    return {"positions": positions}


@app.get("/market/consensus_pdf")
async def get_consensus_pdf(num_points: int = 100):
    """Get public consensus PDF for visualization."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        x_values, y_values = market.get_consensus_pdf(num_points)
        return {
            "x_values": x_values,
            "y_values": y_values,
            "market_params": market.market_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/position/pdf/{position_id}")
async def get_position_pdf(position_id: str, num_points: int = 100):
    """Get a position's belief PDF for visualization."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    if position_id not in market.positions_db:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    try:
        x_values, y_values = market.get_position_pdf(position_id, num_points)
        return {
            "position_id": position_id,
            "x_values": x_values,
            "y_values": y_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/market/buy")
async def buy_position(req: BuyRequest):
    """Execute a buy operation."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        position = market.buy(req.C, req.p_vector)
        return {
            "success": True,
            "message": "Position created successfully",
            "position": {
                "position_id": position.position_id,
                "belief_p": position.belief_p,
                "minted_claims_m": position.minted_claims_m,
                "input_collateral_C": position.input_collateral_C
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/market/buy_with_params")
async def buy_with_params(req: BuyWithParamsRequest):
    """Execute a buy operation using normal distribution parameters."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        # Generate belief vector from normal distribution
        p_vector = market._generate_p_from_normal(req.mean, req.std_dev)
        
        # Execute buy with generated vector
        position = market.buy(req.C, p_vector)
        
        return {
            "success": True,
            "message": "Position created successfully",
            "position": {
                "position_id": position.position_id,
                "belief_p": position.belief_p,
                "minted_claims_m": position.minted_claims_m,
                "input_collateral_C": position.input_collateral_C
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sell/simulate/{position_id}")
async def simulate_sell(position_id: str):
    """Simulate a sell operation (READ-ONLY)."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    if position_id not in market.positions_db:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    try:
        result = market.simulate_sell(position_id)
        return {
            "position_id": position_id,
            "current_value_t_star": round(result['t_star'], 18),
            "iterations": result['iterations']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sell/execute/{position_id}")
async def execute_sell(position_id: str):
    """Execute a sell operation (MUTATES STATE)."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    if position_id not in market.positions_db:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    
    try:
        t_star = market.execute_sell(position_id)
        return {
            "success": True,
            "message": "Position sold successfully",
            "position_id": position_id,
            "collateral_paid_t_star": round(t_star, 18)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/settle/simulate")
async def simulate_settlement(outcome_x: float):
    """Simulate final settlement (READ-ONLY)."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        payouts = market.simulate_settle(outcome_x)
        # Only sum non-None payouts (closed positions return None)
        total_payout = sum(p for p in payouts.values() if p is not None)
        return {
            "outcome_x": outcome_x,
            "payout_map": payouts,
            "total_payout": total_payout
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/settle/execute")
async def execute_settlement(outcome_x: float = Body(..., embed=True)):
    """Execute final settlement (MUTATES STATE)."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        payouts = market.execute_settle(outcome_x)
        # Only sum non-None payouts (closed positions return None)
        total_payout = sum(p for p in payouts.values() if p is not None)
        return {
            "success": True,
            "message": "Market settled successfully",
            "outcome_x": outcome_x,
            "payout_map": payouts,
            "total_payout": total_payout
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/state/save")
async def save_state(req: StateFileRequest):
    """Save market state to file."""
    if market.alpha_vector is None:
        raise HTTPException(status_code=400, detail="Market not initialized")
    
    try:
        market.save_state(req.filepath)
        return {
            "success": True,
            "message": f"State saved to {req.filepath}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state/load")
async def load_state(req: StateFileRequest):
    """Load market state from file."""
    try:
        market.load_state(req.filepath)
        return {
            "success": True,
            "message": f"State loaded from {req.filepath}",
            "market_params": market.market_params,
            "num_positions": len(market.positions_db)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "market_initialized": market.alpha_vector is not None,
        "market_settled": market.is_settled
    }


if __name__ == "__main__":
    print("=" * 60)
    print("functionSPACE Trading Protocol Simulation")
    print("=" * 60)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend Dashboard: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
