cd "C:\Users\Zwmar\.openclaw\workspace\projects\circuit_lm"
$env:PYTHONPATH = "C:\Users\Zwmar\.openclaw\workspace\projects\circuit_lm\src;$env:PYTHONPATH"
python -3.12 scripts/train_circuit_lm.py --data research_evolver_data.txt --vocab-size 4096 --max-train-lines 3000 --epochs 5 --automaton pda --state-bits 6 --steps 60 --out-circuit circuit_4k.json --out-corrector corrector_4k.pt 2>&1
