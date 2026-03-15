# Next-Level Implementation Plan

## Priority 1: LRU Cache + State Hash Optimization ✅ COMPLETE
- [x] Create `circuit_lm/cache.py` - LRU cache for common transitions
- [x] Added integration functions for circuit caching
- [x] Batch operations for SIMD-like performance

## Priority 2: Hierarchical FSM (Multi-Level) ✅ COMPLETE
- [x] Create `circuit_lm/hierarchical.py` - Two-level FSM
- [x] Global circuit for topic tracking
- [x] Local circuits per topic
- [x] Serialization support (to_dict/from_dict)

## Priority 3: Enhanced PDA Memory ⚠️ Partially Complete
- [x] Context keys support in symbolic.py
- [ ] Add attention-like gating mechanism to pda.py
- [ ] Update training to learn gating
- Note: Stack already handles context; gating can be added later

## Priority 4: Neural Corrector Enhancement ✅ COMPLETE
- [x] Modify `src/hybrid.py` - Add residual connections
- [x] Added ResidualCorrector class
- [x] Added QuantizedCorrector for int8 weights
- [x] Added train_residual_hybrid() function
- [x] Added ResidualHybridModel class

## Priority 5: Symbolic Reasoning ✅ COMPLETE
- [x] Create `circuit_lm/symbolic.py` - Constraint solver
- [x] Temporal, Logical, Sequence, Exclusion constraints
- [x] Integration with inference (ConstraintAwareGenerator)
- [x] Fast validation with caching + CP-SAT for complex cases

## Performance Targets
- State space: 1M+ virtual states (via hierarchical)
- Cache hit rate: >80% for common queries (LRU cache implemented)
- Accuracy improvement: +10-20% over baseline (residual corrector)

## Files Created/Modified
- circuit_lm/cache.py (NEW) - LRU caching for transitions
- circuit_lm/hierarchical.py (NEW) - Multi-level FSM
- circuit_lm/symbolic.py (NEW) - Constraint-based generation
- src/hybrid.py (MODIFIED) - Residual corrector + Quantized MLP
