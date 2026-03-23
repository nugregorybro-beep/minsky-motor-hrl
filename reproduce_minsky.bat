@echo off
title Minsky Motor Framework - Automated Reproduction
echo ======================================================
echo    MINSKY MOTOR ABSTRACTIONS: PHASE 4 REPRODUCTION
echo ======================================================

:: 0. Path & Environment Setup
set PYTHONPATH=%CD%
echo Setting Project Root to: %CD%

:: 1. Environment Check
echo [STEP 1/4] Checking Python Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b
)

:: 2. Pre-training Phase (Roadmap Phase 2)
echo [STEP 2/4] Validating Low-Level Primitives...
:: Ensure the models directory exists and contains the experts
if not exist "models\reach_expert.pth" (
    echo [WARNING] Low-level weights not found. Training experts now...
    python src/training/train_primitives.py
) else (
    echo Primitives verified in 'models/' directory.
)

:: 3. Hierarchical Execution (Roadmap Phase 4)
echo [STEP 3/4] Starting Minsky Engine (50 Trials)...
:: This runs the final validation loop
python run_minsky.py
if %errorlevel% neq 0 (
    echo Error: Engine execution failed.
    pause
    exit /b
)

:: 4. Data Analysis
echo [STEP 4/4] Generating Performance Graphs...
python src/evaluation/plot_results.py
echo Analysis complete.

echo ======================================================
echo SUCCESS: Results saved to data/results/minsky_phase4_results.csv
echo Graph saved as phase4_performance_graph.png
echo ======================================================
pause