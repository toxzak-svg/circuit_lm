# Reproduce Hybrid (Circuit + Neural Corrector) pipeline
# Run from repo root: .\scripts\reproduce_hybrid.ps1
# Requires: circuit-lm installed (pip install -e .), PyTorch for hybrid-train/chat

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoRoot

$DataFile = "data.txt"
$ModelFile = "model.json"
$CorrectorFile = "corrector.pt"

# Create minimal data if missing
if (-not (Test-Path $DataFile)) {
    @"
User: Hello
Assistant: Hi there! How can I help?
User: What is 2+2?
Assistant: 2+2 equals 4.
User: Thanks
Assistant: You're welcome.
"@ | Set-Content -Path $DataFile -Encoding utf8
    Write-Host "Created $DataFile with sample chat text."
}

Write-Host "=== 1. Train circuit (PDA) ==="
circuit-lm train --data $DataFile --out $ModelFile --vocab_size 128 --state_bits 4 --automaton pda --transition_steps 5 --emission_steps 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 2. Train neural corrector ==="
circuit-lm hybrid-train --circuit $ModelFile --data $DataFile --out $CorrectorFile --epochs 3 --max-examples 5000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== 3. Eval (circuit only) ==="
circuit-lm eval --data $DataFile --model $ModelFile

Write-Host "`n=== 4. Sample (circuit only) ==="
circuit-lm sample --prompt "User: Hi`nAssistant: " --model $ModelFile --max_tokens 20

Write-Host "`n=== 5. Trace (interpretability) ==="
circuit-lm trace --prompt "User: Hi" --model $ModelFile --top_k 3

Write-Host "`nDone. Interactive hybrid chat: circuit-lm chat --model $ModelFile --corrector $CorrectorFile"
