# Minsky Motor Abstractions

[![DSc Research](https://img.shields.io/badge/DSc_Research-Marymount_University-1F4E79.svg)](#)
[![Conference](https://img.shields.io/badge/Accepted-ICCWS_2026-2E74B5.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

A Hierarchical Reinforcement Learning framework for robotic manipulation, implementing Marvin Minsky's Society of Mind principle. A high-level manager sequences five pre-trained motor primitives (REACH, GRASP, PUSH, ROTATE, RELEASE) to perform complex manipulation tasks in a PyBullet simulation.

**Phase 4 Validation Results — 50 Trials:**

| Metric | Result |
|---|---|
| Task Completion Rate | **100.0%** |
| Mean Cumulative Reward | +59.7 |
| Spatial Noise Applied | ±10 cm |
| Skill Sequence | REACH → GRASP → LIFT |

---

## Repository Structure

```
minsky-motor-hrl/
├── config/
│   └── hrl_config.yaml          # Hyperparameters for all phases
├── data/
│   └── results/
│       └── minsky_phase4_results.csv  # Validation output log
├── models/
│   ├── reach_expert.pth         # PPO-trained primitive (REACH)
│   ├── grasp_expert.pth         # PPO-trained primitive (GRASP)
│   ├── push_expert.pth          # PPO-trained primitive (PUSH)
│   ├── rotate_expert.pth        # PPO-trained primitive (ROTATE)
│   ├── release_expert.pth       # PPO-trained primitive (RELEASE)
│   └── minsky_mgr_final.pth     # High-level manager weights (archived)
├── src/
│   ├── environment/
│   │   ├── robotic_manipulation_env.py  # PyBullet Gymnasium environment
│   │   └── task_generator.py            # Spatial noise task generation
│   ├── evaluation/
│   │   ├── metrics.py           # Evaluation runner and statistics
│   │   └── plot_results.py      # Performance graph generation
│   ├── policies/
│   │   └── high_level_policy.py # HighLevelManager MLP architecture
│   ├── primitives/
│   │   ├── primitive_base.py    # Option base class (I, π, β)
│   │   ├── reach.py             # REACH primitive
│   │   ├── grasp.py             # GRASP primitive
│   │   ├── push.py              # PUSH primitive
│   │   ├── rotate.py            # ROTATE primitive
│   │   └── release.py           # RELEASE primitive
│   └── training/
│       ├── train_primitives.py  # Phase 2: PPO expert training
│       └── train_hierarchy.py   # Phase 3: Manager training
├── run_minsky.py                # Phase 4: Validation engine (main entry point)
├── reproduce_minsky.bat         # Windows one-click reproduction
├── setup_project.py             # Directory initialisation utility
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Windows, macOS, or Linux
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/minsky-motor-hrl.git
cd minsky-motor-hrl
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv minsky_env
minsky_env\Scripts\activate

# macOS / Linux
python -m venv minsky_env
source minsky_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the validation suite

```bash
python run_minsky.py
```

This executes 50 randomised trials and logs results to `data/results/minsky_phase4_results.csv`.

**Windows users:** you can also double-click `reproduce_minsky.bat` — it handles dependency installation and runs the full pipeline automatically.

---

## Reproducing Results

To fully reproduce the Phase 4 100% success rate:

```bash
# Step 1 — Run validation
python run_minsky.py

# Step 2 — Generate performance graph
python src/evaluation/plot_results.py
```

Expected output in `data/results/minsky_phase4_results.csv`:
- 50 rows, all `Outcome = SUCCESS`
- Total_Reward between +58 and +60 per trial

---

## Training from Scratch

If you want to retrain the primitives and manager rather than using the pre-trained weights:

### Phase 2 — Train motor primitives

```bash
python src/training/train_primitives.py
```

Trains all five PPO experts (100,000 timesteps each) and saves weights to `models/`.

### Phase 3 — Train the hierarchical manager

```bash
python src/training/train_hierarchy.py
```

Trains the REINFORCE manager over 3,000 episodes. Checkpoints saved to `models/hierarchy/` every 100 episodes.

> **Note:** See `KNOWN_ISSUES.md` for documented limitations in the training pipeline that affect manager performance. The validated Phase 4 results use a direct execution strategy rather than the trained manager weights.

---

## Architecture Overview

```
Observation (16-D)
       │
       ▼
┌─────────────────────┐
│   HighLevelManager  │  Linear(16,128)→ReLU→Linear(128,64)→ReLU→Linear(64,5)→Softmax
│   (REINFORCE)       │  Selects primitive index every 80 env steps
└─────────┬───────────┘
          │  primitive index (0–4)
          ▼
┌─────────────────────────────────────────────────────┐
│                 Primitive Library                   │
│  REACH  │  GRASP  │  PUSH  │  ROTATE  │  RELEASE   │
│  PPO    │  PPO    │  PPO   │  PPO     │  PPO        │
└─────────────────────────────────────────────────────┘
          │  action vector [dx, dy, gripper]
          ▼
┌─────────────────────┐
│  RoboticManipEnv    │  PyBullet · Franka Panda · 16-D obs · ±10cm noise
└─────────────────────┘
```

---

## Environment Details

| Property | Value |
|---|---|
| Simulator | PyBullet 3.2.5 |
| Robot | Franka Panda (franka_panda/panda.urdf) |
| Observation space | 16-D continuous (gripper XYZ, quaternion, 6 joints, block XYZ) |
| Action space | 3-D continuous [-1, 1] (delta-X, delta-Y, gripper cmd) |
| Block spawn noise | X ∈ [0.30, 0.45], Y ∈ [-0.10, 0.10] |
| Episode length | 200 steps max |
| Success condition | attached == True and block_z > 0.10 m |

---

## Team

This project is part of the DSc Information Technology research program at Marymount University, submitted for the AI era - IT797-C

**Team Members:**
- Gregory Broomfield
- Kwabena Akyeampong
- Jilian Angel
- Barry Humphrey
- Ayende Ibere
- Kayode Ibitoye
- Oladimeji Oyegunle

Professor: Dr. Donna Schaeffer


**To contribute or access this repository:**
Share your GitHub username with Gregory and you will be added as a collaborator.

---

## Citation

If you use this codebase in your research, please cite:

```
Team-Minsky (2026). Minsky Motor Abstractions: A Hierarchical Reinforcement
Learning Framework for Robotic Manipulation via Motor Primitive Composition.
Marymount University DSc Program. 
```

---

## License

MIT License — see `LICENSE` for details.
