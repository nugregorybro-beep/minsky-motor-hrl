# Known Issues and Limitations

This document describes known issues in the training pipeline. These do not affect the Phase 4 validation results (which use `run_minsky.py` directly) but are relevant if you attempt to retrain the system.

---

## 1. Manager weights architecture mismatch (`minsky_mgr_final.pth`)

**File:** `models/minsky_mgr_final.pth`

**Issue:** This file encodes a 2-layer, 3-input network. The `HighLevelManager` class in `src/policies/high_level_policy.py` instantiates a 3-layer, 16-input network. PyTorch's `load_state_dict` will load the mismatched weights without raising an error, producing a manager with meaningless outputs.

**Status:** The validated `run_minsky.py` does not load this file. It is retained for archival purposes only.

**Fix:** Retrain the manager using `src/training/train_hierarchy.py`. The correct architecture is saved at `models/hierarchy/minsky_mgr_ep2900.pth`.

---

## 2. All primitive experts trained under `task='reach'`

**File:** `src/training/train_primitives.py`

**Issue:** The environment defaults to `task='reach'` when none is specified. All five primitives were trained with this default, meaning:
- The GRASP expert never learned to output `action[2] > 0.5` (the magnet trigger threshold)
- The ROTATE expert learned orientation alignment, not upward lifting
- The PUSH expert's reward (`obs[13] × 10`) is not useful for grasp-and-lift tasks

**Fix:** Set the correct task attribute per primitive in `train_primitives.py`:

```python
# Example fix for grasp expert
env = RoboticManipulationEnv(render_mode=p.DIRECT, task='grasp')
model = PPO("MlpPolicy", env, learning_rate=lr, verbose=1)
```

---

## 3. Manager policy collapse

**File:** `src/training/train_hierarchy.py`

**Issue:** After 3,000 training episodes, the manager converges to always selecting REACH regardless of state (probability ≈ 0.999). This is caused by reward imbalance — the shaped REACH bonus dominates early training, and without entropy regularisation the policy collapses.

**Fix options:**
- Add entropy regularisation: `loss = -(log_prob * reward) - 0.01 * entropy`
- Increase training episodes to 10,000+
- Normalise shaped bonuses so no single primitive dominates early gradient updates

---

## 4. `src/evaluation/metrics.py` — broken import (fixed in this version)

**Original issue:** Imported from `src.environment.robotic_env` (non-existent module). Fixed to `src.environment.robotic_manipulation_env`.

---

## 5. `src/evaluation/plot_results.py` — hardcoded Windows path (fixed in this version)

**Original issue:** File path hardcoded as `C:\minsky_project\minsky_phase4_results.csv`. Fixed to use `os.path` relative to project root — works on Windows, macOS, and Linux.

---

## 6. Physics constraint: magnet threshold unreachable via `env.step()`

**File:** `src/environment/robotic_manipulation_env.py`

**Issue:** The grasp magnet triggers when `3D dist < 0.05 m`. The environment fixes gripper Z at 0.08 m via `env.step()`. The block (cube.urdf, globalScaling=0.5) settles to Z ≈ 0.25–0.50 m after physics warmup. Minimum achievable 3D distance ≈ 0.17 m — above the 0.05 m threshold.

**Workaround used in `run_minsky.py`:** Direct `p.createConstraint()` call when XY alignment is confirmed.

**Proper fix:** Use a smaller block (e.g. `globalScaling=0.05`) and adjust magnet threshold to `0.10 m`.
