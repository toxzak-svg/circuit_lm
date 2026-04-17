# circuit_lm full pipeline reproducer
# Run from circuit_lm repo root:
#   .\scripts\reproduce.ps1

param(
    [string]$DataFile = "training_data.txt",
    [string]$CircuitOut = "circuit.json",
    [string]$CorrectorOut = "corrector.pt",
    [string]$Automaton = "pda",
    [int]$VocabSize = 1024,
    [int]$BpeMerges = 512,
    [int]$StateBits = 5,
    [int]$StackDepth = 6,
    [int]$Steps = 60,
    [int]$Epochs = 5,
    [int]$MaxExamples = 100000,
    [switch]$UseGPU
)

$ErrorActionPreference = "Stop"

# Check data file exists
if (-not (Test-Path $DataFile)) {
    Write-Error "[reproduce] Data file not found: $DataFile"
    exit 1
}

Write-Host "==================================================" -ForegroundColor cyan
Write-Host "circuit_lm — Full Pipeline Reproduction Script" -ForegroundColor cyan
Write-Host "==================================================" -ForegroundColor cyan
Write-Host ""

# Step 1: Train circuit
Write-Host "[STEP 1] Training $Automaton circuit on $DataFile" -ForegroundColor yellow
$train_cmd = @(
    "python", "-m", "circuit_lm.cli", "train",
    "--data", $DataFile,
    "--out", $CircuitOut,
    "--automaton", $Automaton,
    "--vocab_size", $VocabSize,
    "--state_bits", $StateBits,
    "--stack_depth", $StackDepth,
    "--tokenizer", "bpe",
    "--bpe_merges", $BpeMerges,
    "--steps", $Steps
)
Write-Host ">>> $($train_cmd -join ' ')"
$start = Get-Date
& python -m circuit_lm.cli train `
    --data $DataFile `
    --out $CircuitOut `
    --automaton $Automaton `
    --vocab_size $VocabSize `
    --state_bits $StateBits `
    --stack_depth $StackDepth `
    --tokenizer bpe `
    --bpe_merges $BpeMerges `
    --steps $Steps
if ($LASTEXITCODE -ne 0) { Write-Error "[STEP 1] Circuit training failed"; exit 1 }
Write-Host "[STEP 1] Done in $((Get-Date).Subtract($start).TotalSeconds)s" -ForegroundColor green
Write-Host ""

# Step 2: Train corrector
Write-Host "[STEP 2] Training neural corrector" -ForegroundColor yellow
$correct_cmd = @(
    "python", "scripts\train_bpe_hybrid.py",
    "--data", $DataFile,
    "--circuit-out", $CircuitOut,
    "--corrector-out", $CorrectorOut,
    "--automaton", $Automaton,
    "--vocab-size", $VocabSize,
    "--bpe-merges", $BpeMerges,
    "--state-bits", $StateBits,
    "--stack-depth", $StackDepth,
    "--epochs", $Epochs,
    "--batch-size", 128,
    "--max-examples", $MaxExamples,
    "--embed-dim", 256,
    "--hidden-dim", 512,
    "--num-layers", 3,
    "--streaming"
)
Write-Host ">>> $($correct_cmd -join ' ')"
$start = Get-Date
& python scripts\train_bpe_hybrid.py `
    --data $DataFile `
    --circuit-out $CircuitOut `
    --corrector-out $CorrectorOut `
    --automaton $Automaton `
    --vocab-size $VocabSize `
    --bpe-merges $BpeMerges `
    --state-bits $StateBits `
    --stack-depth $StackDepth `
    --epochs $Epochs `
    --batch-size 128 `
    --max-examples $MaxExamples `
    --embed-dim 256 `
    --hidden-dim 512 `
    --num-layers 3 `
    --streaming
if ($LASTEXITCODE -ne 0) { Write-Error "[STEP 2] Corrector training failed"; exit 1 }
Write-Host "[STEP 2] Done in $((Get-Date).Subtract($start).TotalSeconds)s" -ForegroundColor green
Write-Host ""

# Step 3: Trace a sample
Write-Host "[STEP 3] Running trace on sample prompt" -ForegroundColor yellow
$sample = "User: hello, how are you?`nAssistant:"
Write-Host ">>> python -m circuit_lm.cli trace --model $CircuitOut --prompt `"$sample`" --top_k 5"
python -m circuit_lm.cli trace --model $CircuitOut --prompt $sample --top_k 5
Write-Host ""

# Step 4: Evaluate on held-out data
Write-Host "[STEP 4] Evaluating accuracy" -ForegroundColor yellow
& python -m circuit_lm.cli eval --model $CircuitOut --data $DataFile
Write-Host ""

Write-Host "==================================================" -ForegroundColor cyan
Write-Host "DONE! Files:" -ForegroundColor green
Write-Host "  Circuit:  $CircuitOut"
Write-Host "  Corrector: $CorrectorOut"
Write-Host ""
Write-Host "To chat:" -ForegroundColor cyan
Write-Host "  py -3.12 -m circuit_lm.cli chat --model $CircuitOut --corrector $CorrectorOut" -ForegroundColor White
Write-Host ""
Write-Host "To trace:" -ForegroundColor cyan
Write-Host "  py -3.12 -m circuit_lm.cli trace --model $CircuitOut --prompt `"your text here`" --top_k 5" -ForegroundColor White
Write-Host "==================================================" -ForegroundColor cyan
