# Troubleshooting

Common issues and solutions for CircuitLM.

## Installation Issues

### OR-Tools Installation Fails

```
ERROR: Could not find a version that satisfies the requirement ortools>=9.8
```

**Solution**: Ensure you have a compatible Python version (3.10+) and pip is up to date:
```bash
pip install --upgrade pip
pip install ortools
```

### Import Errors

```
ModuleNotFoundError: No module named 'circuit_lm'
```

**Solution**: Install the package in editable mode:
```bash
pip install -e ".[dev]"
```

## Training Issues

### No Sequences Found

```
[train] ERROR: no sequences found – is the file long enough?
```

**Solution**: Ensure your input file is large enough (at least 2 tokens after tokenization). The tokenizer needs at least 2 characters for char mode.

### CP-SAT Timeout

Training takes very long or times out.

**Solution**: Reduce the CP-SAT time budget:
```bash
circuit-lm train --data data.txt --out model.json --steps 5
```

For PDA, reduce stack and transition budgets:
```bash
circuit-lm train --data data.txt --out model.json --automaton pda --steps 5
```

### Out of Memory

Large models consume too much memory.

**Solution**: 
- Reduce `state_bits` (fewer states)
- Reduce `vocab_size` (smaller vocabulary)
- For PDA: reduce `stack_depth`
- For PPM: reduce `order`

### Joint PDA Can't Find Stack

Joint PDA doesn't discover push/pop operations.

**Solution**:
- Increase CP-SAT budget (try 60-120 seconds)
- Reduce training set size
- Check that training data has clear bracket patterns

## Evaluation Issues

### Low Accuracy

Model accuracy is lower than expected.

**Solution**:
- Increase `state_bits` for FSM
- Increase `stack_depth` for PDA
- Increase `order` for PPM
- Increase CP-SAT optimization time
- Try different tokenizer (BPE vs char)

### Per-token Breakdown Empty

```
per-token breakdown (top 0 by frequency)
```

**Solution**: Make sure you're using `--per_token` flag:
```bash
circuit-lm eval --data data.txt --model model.json --per_token
```

## Sampling Issues

### Repetitive Output

Generated text repeats the same tokens.

**Solution**: Use repetition penalty:
```bash
circuit-lm sample --prompt "hello" --model model.json --repeat_penalty_div 2 --repeat_window 64
```

### Empty Output

Sampling produces no output.

**Solution**: 
- Check the prompt encoding works
- Ensure model was trained successfully

## Serialization Issues

### Load Fails

```
JSONDecodeError: Expecting value
```

**Solution**: Check the model file exists and is valid JSON:
```bash
cat model.json | head
```

### MessagePack Not Available

```
RuntimeError: MessagePack support requires the 'msgpack' package
```

**Solution**: Install msgpack:
```bash
pip install msgpack
```

## Testing Issues

### Float Detection Fails

Tests fail with float detection.

**Solution**: Ensure no floats in code:
- Use integer division `//` not `/`
- Avoid float literals like `0.0`
- Don't use `math` module functions

### Forbidden Import Fails

Tests fail due to forbidden imports.

**Solution**: Remove any imports of:
- numpy
- torch
- jax
- scipy
- tensorflow

## Performance Issues

### Slow Training

Training is slower than expected.

**Solution**:
- Reduce `refinement_rounds`
- Reduce `context_len`
- Use smaller vocab_size
- Consider using PPM (no CP-SAT needed)

### Slow Inference

Generation is slow.

**Solution**: 
- Use greedy decoding instead of sampling
- Reduce model complexity

## Getting Help

If you encounter issues not covered here:

1. Check the test suite: `pytest`
2. Run a benchmark: `python scripts/benchmark_small.py`
3. Check the status: `cat STATUS.md`
4. Review the README: `cat README.md`
