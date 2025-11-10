# functionSPACE Trading Protocol Simulation

A **high-fidelity cryptoeconomic simulation** of the functionSPACE trading protocol, implementing the full trading lifecycle (Create, Buy, Sell, Settle) with mathematical precision.

## 🎯 Features

- **Complete Trading Lifecycle**: Create markets, buy/sell positions, and settle outcomes
- **Mathematical Fidelity**: Implements all 5 numerical directives for production-ready accuracy
  - ✅ Difference form for potential change calculations (prevents subtractive cancellation)
  - ✅ Log-sum-exp (LSE) pattern for settlement (prevents overflow/underflow)
  - ✅ Python f64 floats as fixed-point stand-in
  - ✅ Hard solver boundaries with eps_alpha floor enforcement
  - ✅ Full precision intermediate calculations (18-decimal final rounding)
- **Interactive Dashboard**: Real-time visualization with Chart.js
- **Stateful API**: FastAPI backend with full REST endpoints
- **State Persistence**: Save/load market state to JSON files

## 📁 Project Structure

```
FS one shot/
├── core.py              # Core simulation engine (Market & Position classes)
├── api.py               # FastAPI backend with REST endpoints
├── index.html           # Interactive frontend dashboard
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── Trading - Working Spec v2.pdf  # Original specification
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python api.py
```

The server will start on `http://localhost:8000`

### 3. Open the Dashboard

Navigate to `http://localhost:8000` in your web browser.

## 📖 Usage Guide

### Creating a Market

1. Fill in the market parameters in the **Create Market** section:
   - **L**: Lower bound of outcome space (e.g., 0)
   - **H**: Upper bound of outcome space (e.g., 100)
   - **K**: Number of discretization points (e.g., 10 creates 11 buckets)
   - **P₀**: Initial pseudocount mass (e.g., 100)
   - **μ**: Minting ratio for claims (e.g., 1.0)
   - **ε_α**: Minimum alpha floor for numerical stability (e.g., 0.01)

2. Click **Create Market**

3. The consensus PDF will appear in the chart

### Buying a Position

1. Enter the **Collateral (C)** amount (e.g., 10)

2. Enter your **Belief Vector (p)** as comma-separated values:
   - Must have K+1 values
   - Must sum to 1.0
   - Example for K=10: `0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091`

3. Click **Buy Position**

4. Your new position will appear in the **Active Positions** table

### Visualizing Positions

- Check the **☑** box next to any position to overlay its belief PDF on the chart
- The position's PDF will appear as a dashed line
- Uncheck to remove the overlay

### Selling a Position

1. The **Current Value (t\*)** column shows the real-time redemption value
2. Click the **Sell** button to execute the sale
3. You'll receive collateral equal to t* back

### Simulating Settlement

1. Enter the **Final Outcome (x)** value in the Settlement section
2. Click **Simulate** to see potential payouts
3. Payouts will populate in the **Settlement Payout** column

### Executing Settlement

1. Click **Execute** to finalize settlement
2. ⚠️ **This freezes the market** - no more trades allowed
3. All positions are paid out according to their accuracy

### State Management

- **Save**: Export current market state to a JSON file
- **Load**: Import a previously saved market state

## 🔬 Mathematical Implementation Details

### Directive 1: Difference Form (Buy & Sell)

Instead of computing `ΔA = A(α_new) - A(α_old)`, we use:

```
ΔA = [ln(Γ(P_new)) - ln(Γ(P_old))] - Σ[ln(Γ(α_k,new)) - ln(Γ(α_k,old))]
```

This prevents catastrophic cancellation in floating-point arithmetic.

**Implementation**: `_compute_delta_A_difference_form()` in `core.py`

### Directive 2: Log-Sum-Exp (Settlement)

When interpolating log-densities during settlement:

```python
log_f_x = logsumexp([log(w_lower) + log_f_lower, log(w_upper) + log_f_upper])
```

This prevents overflow/underflow when working in log-space.

**Implementation**: `simulate_settle()` in `core.py`

### Directive 3: Python f64 Precision

All calculations use Python's native 64-bit floats (`np.float64`), leveraging SciPy's special functions:
- `scipy.special.gammaln` - Log-gamma function
- `scipy.special.digamma` - Digamma function (for derivatives)
- `scipy.special.logsumexp` - Numerically stable log-sum-exp

### Directive 4: Hard Solver Boundaries

The sell solver enforces:
- **eps_alpha floor**: `α_k ≥ ε_α` for all k
- **t_max ceiling**: `t ≤ min_k((α_k - ε_α) / p_k)`

Invalid t values are rejected during root-finding.

**Implementation**: `simulate_sell()` in `core.py`

### Directive 5: Full Precision Intermediates

- No rounding in intermediate calculations
- Only final outputs rounded to 18 decimal places
- Prevents path-dependence and rounding drift

## 🔌 API Endpoints

### Market Management

- `POST /market/create` - Create a new market
- `GET /market/state` - Get current market state
- `GET /market/positions` - Get all active positions
- `GET /market/consensus_pdf` - Get consensus PDF data

### Trading

- `POST /market/buy` - Buy a new position
- `GET /sell/simulate/{position_id}` - Simulate sell (read-only)
- `POST /sell/execute/{position_id}` - Execute sell (mutates state)

### Settlement

- `GET /settle/simulate?outcome_x={x}` - Simulate settlement (read-only)
- `POST /settle/execute` - Execute settlement (freezes market)

### State Persistence

- `POST /state/save` - Save market state to file
- `POST /state/load` - Load market state from file

### Health Check

- `GET /health` - API health status

## 📊 Example Workflow

### 1. Create a Simple Market

```python
# Market parameters
L = 0.0
H = 100.0
K = 10
P0 = 100.0
mu = 1.0
eps_alpha = 0.01
```

### 2. Buy a Bullish Position

```python
# Bullish belief (expect high outcomes)
C = 10.0
p_vector = [0.01, 0.02, 0.03, 0.05, 0.09, 0.15, 0.2, 0.2, 0.15, 0.08, 0.02]
```

### 3. Buy a Bearish Position

```python
# Bearish belief (expect low outcomes)
C = 10.0
p_vector = [0.2, 0.2, 0.15, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]
```

### 4. Watch the Market Evolve

The consensus PDF will shift toward the aggregate belief of all traders.

### 5. Simulate Different Outcomes

```python
# Try different settlement values
outcome_x = 30.0  # Bearish traders profit
outcome_x = 70.0  # Bullish traders profit
```

### 6. Execute Final Settlement

```python
# Lock in the final outcome
outcome_x = 50.0
```

## 🧪 Testing Numerical Directives

### Test Directive 1 (Difference Form)

Create two positions with very similar beliefs to test subtraction stability:

```python
# Position 1
p1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]

# Position 2 (nearly identical)
p2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.099, 0.001]
```

Both should produce stable ΔA calculations without precision loss.

### Test Directive 4 (Solver Boundaries)

Try to sell a position that would violate eps_alpha:

1. Create a market with high eps_alpha (e.g., 5.0)
2. Buy a position with C=10
3. Try to sell immediately

The solver should correctly limit t* to respect the alpha floor.

## 🐛 Troubleshooting

### Server won't start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8000 is already in use
- Try running with explicit host: `uvicorn api:app --host 0.0.0.0 --port 8000`

### "Market not initialized" error

- You must create a market first before any trading operations
- Click **Create Market** in the dashboard

### Belief vector doesn't sum to 1

- Ensure your comma-separated values add up to exactly 1.0
- Use the pre-filled example as a template
- For K=10, you need 11 values

### Position overlay not showing

- Ensure the position still exists (hasn't been sold)
- Try refreshing the positions table
- Check the browser console for errors

### Settlement fails

- Verify the outcome_x is within [L, H]
- Ensure the market isn't already settled
- Check that positions exist to settle

## 📚 Technical Stack

- **Backend**: Python 3.8+, FastAPI, SciPy, NumPy
- **Frontend**: Vanilla JavaScript, HTML5, CSS3, Chart.js
- **Data Format**: JSON for state persistence
- **Precision**: 64-bit floating point (f64)

## 🔐 Security Notes

This is a **simulation environment** for testing and research:
- No authentication/authorization
- No input sanitization for production
- Stateful in-memory storage (no database)
- Single-user design

For production deployment, add:
- User authentication
- Input validation and rate limiting
- Database integration
- Multi-user session management
- HTTPS/TLS

## 📄 License

This simulation implements the functionSPACE Trading Protocol specification.

## 🤝 Contributing

This is a research simulation. Suggested improvements:
- Multi-market support
- Historical state snapshots
- Advanced visualization modes
- Batch trade simulation
- Performance benchmarking suite

## 📞 Support

For issues with:
- **Mathematical fidelity**: Check the Trading Spec v2 PDF
- **API errors**: See the FastAPI docs at `/docs`
- **Frontend bugs**: Check browser console for errors

---

**Built with mathematical precision for cryptoeconomic research** 🚀
