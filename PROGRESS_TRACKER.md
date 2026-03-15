# Hybrid Circuit-Neural Model Progress Tracker

## Experiment Log

| Date | Circuit | Vocab | Data | Neural | Epochs | Accuracy | Notes |
|------|---------|-------|------|--------|--------|----------|-------|
| 2026-03-14 | test_pda.json | 64 | train_data.txt (synthetic) | small | 7 | 94.93% | Synthetic data - overfits |
| 2026-03-14 | test_pda.json | 64 | chat_data.txt (50k) | small | 10 | 34.78% | Real dialogue data |
| 2026-03-14 | test_pda.json | 64 | chat_data.txt (100k) | big | 15 | 56.53% | Best on small vocab |
| 2026-03-14 | circuit_1k.json | 102 | chat_data_full.txt | - | - | 25.23% | Circuit only baseline |
| 2026-03-14 | circuit_large.json | 198 | combined_data.txt | - | - | 25.35% | Circuit only baseline |
| 2026-03-14 | circuit_large.json | 198 | combined_data.txt (100k) | large | 15 | ~35% | Full hybrid |

---

## Models

### Circuits
| Model File | Vocab | States | Stack | Training Data |
|------------|-------|--------|-------|---------------|
| test_pda.json | 64 | 16 | yes | chat.txt |
| circuit_1k.json | 102 | 64 | yes | chat_data_full.txt |
| circuit_large.json | 198 | 128 | yes | combined_data.txt |

### Correctors
| File | Vocab | Params | Architecture |
|------|-------|--------|--------------|
| corrector.pt | 64 | ~50K | Small MLP |
| big_corrector.pt | 64 | ~200K | Bigger CNN+MLP |
| large_corrector.txt | 198 | ~160K | FC-only |

---

## Generation Tests

### Test 1: "Hello"
- **Date:** 2026-03-14
- **Model:** test_pda.json + big_corrector.pt
- **Output:** ```
�ello, how are you�r urtolete if �roly wewshals to dera ole pececich rents is oortart oter lives.
a
```
- **Notes:** Garbled but some structure

### Test 2: "Hello" (larger vocab)
- **Date:** 2026-03-14
- **Model:** circuit_large.json + large_corrector.pt  
- **Output:** ```
Hellosssstanranserter: ers aw  od bious tecfeyyonarotocecp.  iso

usersuser: Hi
assistant:He Hello  ce ho
```
- **Notes:** More structure emerging (user:, assistant:)

---

## Key Findings

1. **Circuit helps:** PDA baseline 16-25%, hybrid gets 35-56%
2. **Vocabulary bottleneck:** 64-198 chars too small for coherent English
3. **Structure learning:** Model learns chat format (user:/assistant:)
4. **CPU training:** Works! No GPU needed

---

## Next Steps

- [ ] Increase vocab to 1000+ (BPE tokenizer)
- [ ] More training data (millions of tokens)
- [ ] Larger neural network (1M+ params)
- [ ] Try on code data (PDA should shine)
- [ ] Make demo/blog post

---

## Hardware

- **CPU:** AMD Ryzen 7 7730U
- **RAM:** 14GB
- **GPU:** 2GB (not used - CPU training)
- **Storage:** ~20GB free

---

*Last updated: 2026-03-14*
