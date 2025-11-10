# Iteration 2 - Testing Guide

Quick guide to test all new features and verify regression safety.

---

## 🚀 Quick Start

```bash
# Install dependencies (if not done)
pip install -r requirements.txt

# Start server
python api.py
```

Navigate to: **http://localhost:8000**

---

## ✅ Test Sequence

### Test 1: Market Title Feature

1. **Create Market**
   - Change title to: "BTC Price End of 2025"
   - L=0, H=100000, K=20, P0=100, μ=1.0, ε_α=0.01
   - Click "Create Market"

2. **Verify**
   - Market Information section shows: "BTC Price End of 2025"
   - ✅ **PASS**: Title displays correctly

---

### Test 2: Simple Buy Tab

1. **Switch to Simple Tab** (should be default)
   
2. **Configure Trade**
   - Collateral: 100
   - Outcome: 75000
   - Confidence: High (Narrow)

3. **Verify Preview**
   - Yellow dashed line appears on chart
   - Sharp peak around x=75000
   - ✅ **PASS**: Preview shows narrow distribution

4. **Execute Buy**
   - Click "Buy Position"
   - Position appears in table
   - ✅ **PASS**: Trade executes successfully

---

### Test 3: Moderate Buy Tab

1. **Switch to Moderate Tab**

2. **Use Sliders**
   - Set Mean slider to 50000
   - Set Std Dev slider to 10
   - Verify number inputs update automatically

3. **Use Number Inputs**
   - Type Mean: 60000
   - Type Std Dev: 15
   - Verify sliders update automatically

4. **Verify Preview**
   - Preview updates as you adjust sliders
   - Distribution shape changes with std dev
   - ✅ **PASS**: Sliders and preview work

5. **Execute Buy**
   - Collateral: 50
   - Click "Buy Position"
   - ✅ **PASS**: Position created

---

### Test 4: Advanced Buy Tab

1. **Switch to Advanced Tab**

2. **Enter Manual p_vector**
   ```
   0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
   ```
   (21 values for K=20)

3. **Verify**
   - Preview disappears (Advanced mode doesn't show preview)
   - Buy button enabled if vector sums to 1.0
   - ✅ **PASS**: Manual entry works (original behavior preserved)

4. **Execute Buy**
   - Collateral: 25
   - Click "Buy Position"
   - ✅ **PASS**: Position created

---

### Test 5: Sell Iterations Display

1. **View Positions Table**
   - Note "Sell Iterations" column
   - Should show numbers like 5, 7, 12, etc.

2. **Click Refresh** 
   - Iterations may change slightly (solver behavior)
   - ✅ **PASS**: Iterations displayed

3. **Sell a Position**
   - Click "Sell" on any position
   - Confirm
   - Position removed from table
   - ✅ **PASS**: Sell still works correctly

---

### Test 6: Trade Preview Validation

1. **Simple Tab**
   - Set Outcome to -1000 (outside market range)
   - Buy button should be **disabled**
   - Preview should disappear

2. **Moderate Tab**
   - Set Std Dev to 0 (invalid)
   - Buy button should be **disabled**
   - Preview should disappear

3. **Fix Inputs**
   - Set valid values
   - Buy button re-enabled
   - Preview reappears
   - ✅ **PASS**: Input validation works

---

### Test 7: State Persistence

1. **Save State**
   - Enter filename: "iteration2_test.json"
   - Click "Save"

2. **Reload Page**
   - Refresh browser

3. **Load State**
   - Enter filename: "iteration2_test.json"
   - Click "Load"

4. **Verify**
   - Market title restored
   - All positions restored
   - Iterations column populated
   - ✅ **PASS**: State persistence works

---

### Test 8: Regression Tests (Core Math)

#### Test 8A: Buy Logic
1. **Create New Market**
   - L=0, H=10, K=10, P0=100, μ=1.0
   
2. **Buy Position (Advanced Tab)**
   - C=10
   - p_vector: `0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091, 0.09091`

3. **Verify**
   - Position minted_claims_m ≈ 23.03 (ln(110/100) * 100)
   - ✅ **PASS**: Buy math unchanged

#### Test 8B: Sell Logic
1. **Simulate Sell** (click Refresh)
   - Current Value (t*) should be close to input collateral (10)
   - Iterations: typically 5-15

2. **Execute Sell**
   - Collateral returned ≈ 10 (symmetric redemption)
   - ✅ **PASS**: Sell math unchanged

#### Test 8C: Settle Logic
1. **Create Market and Positions**
   - Use Simple tab for quick testing

2. **Simulate Settlement**
   - Outcome: 5.0 (middle of range)
   - Check settlement payouts

3. **Execute Settlement**
   - Market freezes
   - Payouts match simulation
   - ✅ **PASS**: Settle math unchanged

---

## 🎯 Edge Case Tests

### Edge Case 1: Very High Confidence (Near-Zero Std Dev)

1. **Moderate Tab**
   - Mean: 50000
   - Std Dev: 0.05 (very small)

2. **Verify**
   - Preview shows extremely narrow spike
   - p_vector should be near one-hot
   - ✅ **PASS**: Edge case handled

### Edge Case 2: Mean at Boundary

1. **Simple Tab**
   - Outcome: 0 (lower bound)
   - Confidence: High

2. **Verify**
   - Preview shows spike at left edge
   - Buy executes successfully
   - ✅ **PASS**: Boundary case works

### Edge Case 3: Wide Distribution

1. **Moderate Tab**
   - Mean: 50000
   - Std Dev: 20 (maximum)

2. **Verify**
   - Preview shows very flat, wide distribution
   - Still sums to 1.0
   - ✅ **PASS**: Wide distribution works

---

## 📊 Performance Tests

### Test 1: Preview Responsiveness

1. **Moderate Tab**
   - Rapidly move Mean slider back and forth
   - Chart should update smoothly (no lag)
   - ✅ **PASS**: Preview is responsive

### Test 2: Large Number of Positions

1. **Create 10+ positions** using Simple tab
   - Mix of different means and confidences

2. **Click Refresh**
   - All positions load within ~2 seconds
   - Iterations displayed for all
   - ✅ **PASS**: Scales well

---

## 🐛 Common Issues & Solutions

### Issue 1: "Market not initialized" Error
**Solution**: Create a market first before buying positions

### Issue 2: Buy Button Stays Disabled
**Solution**: 
- Check mean is within [L, H]
- Check std_dev > 0
- For advanced tab, ensure p_vector sums to 1.0

### Issue 3: Preview Not Updating
**Solution**: 
- Ensure you're on Simple or Moderate tab
- Advanced tab doesn't show preview
- Refresh page if stuck

### Issue 4: Sliders Not Syncing
**Solution**: 
- Check that market bounds were initialized (L and H values)
- Try typing in number input directly

---

## ✅ Acceptance Criteria

All tests should pass with these results:

- [x] Market title saved and displayed
- [x] Simple tab creates valid positions
- [x] Moderate tab sliders work correctly
- [x] Advanced tab preserves original behavior
- [x] Trade preview updates in real-time
- [x] Sell iterations displayed in table
- [x] Buy button validation works
- [x] State save/load includes new features
- [x] Core math functions unchanged (no regression)
- [x] All edge cases handled gracefully

---

## 📝 Test Results Template

Copy this to document your testing:

```
## Test Session: [Date]

### Environment
- Browser: [Chrome/Firefox/Edge]
- OS: [Windows/Mac/Linux]
- Python Version: [3.x.x]

### Test Results

| Test | Status | Notes |
|------|--------|-------|
| Market Title | ✅/❌ | |
| Simple Buy Tab | ✅/❌ | |
| Moderate Buy Tab | ✅/❌ | |
| Advanced Buy Tab | ✅/❌ | |
| Sell Iterations | ✅/❌ | |
| Trade Preview | ✅/❌ | |
| State Persistence | ✅/❌ | |
| Regression Tests | ✅/❌ | |

### Issues Found
[List any bugs or unexpected behavior]

### Performance Notes
[Any lag, slow operations, or optimization opportunities]
```

---

## 🎓 Advanced Testing

For thorough validation:

1. **API Testing** (use browser DevTools or Postman)
   - POST /market/create with title
   - POST /market/buy_with_params
   - GET /sell/simulate/{id} (check iterations)

2. **Browser Console**
   - Open DevTools → Console
   - Watch for JavaScript errors
   - Verify API responses

3. **Network Tab**
   - Monitor API calls
   - Check response times
   - Verify data payloads

---

## 🏆 Success Criteria

**Iteration 2 is successful if:**

1. All 8 main tests pass
2. All 3 edge case tests pass
3. No JavaScript console errors
4. No regression in core math
5. UX feels smooth and responsive
6. Documentation matches implementation

---

**Happy Testing!** 🚀
