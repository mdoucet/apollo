# Apollo Lander

A Python simulation of the Apollo Lunar Module descent, based on the real Apollo 11 [Luminary099](https://github.com/chrislgarry/Apollo-11/tree/master/Luminary099) AGC code. Fly the spacecraft manually using authentic Program 66 controls, or train a reinforcement learning agent to land autonomously.

## Overview

The simulation ports the mathematical logic from the Apollo Guidance Computer into a modern Python environment:

- **RK4 physics engine** with lunar orbital mechanics, thrust, and mass depletion
- **P66 guidance system** replicating the astronaut's actual control interface — Rotational Hand Controller (attitude rates) and Rate of Descent switch (1 ft/s sink rate increments)
- **Gymnasium environment** (`ApolloLander-v0`) for RL training and evaluation
- **Browser-based UI** (Flask + HTML5 Canvas) with 2D side-profile view, HUD, and DSKY-style display

## Quick Start

### Install

```bash
# Clone the repo
git clone https://github.com/yourusername/apollo.git
cd apollo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core package
pip install -e "."

# Install with all extras (rendering + RL training + dev tools)
pip install -e ".[all]"
```

### Play manually

```bash
apollo play
```

This opens the simulation in your web browser at `http://127.0.0.1:5050`.

**Controls:**
| Key | Action |
|-----|--------|
| Arrow keys | Pitch / Roll (Rotational Hand Controller) |
| Q / E | Yaw |
| Page Up | Decrease sink rate (ROD up, −1 ft/s) |
| Page Down | Increase sink rate (ROD down, +1 ft/s) |
| Escape | Quit |

### Watch the AGC autopilot land

```bash
apollo autopilot
```

This runs a classical (non-RL) autopilot that replicates the real Apollo P66 landing procedure. The throttle is controlled by the AGC's RODCOMP algorithm (faithfully ported from the [Luminary099](https://github.com/chrislgarry/Apollo-11/blob/master/Luminary099/LUNAR_LANDING_GUIDANCE_EQUATIONS.agc) source code), while horizontal velocity is nulled by a simulated Commander on the RHC — just as in the real mission.

Run headless evaluation:

```bash
apollo autopilot --episodes 20
```

### Train an RL agent

```bash
pip install -e ".[rl]"
apollo train --algo ppo --timesteps 500000
apollo evaluate --model models/apollo_lander --episodes 50
```

### Use as a Gymnasium environment

```python
import apollo_lander.envs
import gymnasium as gym

env = gym.make("ApolloLander-v0")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Architecture

```
Agent/Human Input (action)
        │
        ▼
┌─────────────────────────┐
│  Gymnasium Environment  │  ← ApolloLander-v0
│  ┌───────────────────┐  │
│  │ P66 Guidance/DAP  │  │  ← Translates actions → thrust vectors
│  └────────┬──────────┘  │
│  ┌────────▼──────────┐  │
│  │ RK4 Physics Engine│  │  ← Lunar orbital mechanics + mass depletion
│  └───────────────────┘  │
│  → obs, reward, done    │
└─────────────────────────┘
        │
        ▼
  Web UI (Flask)            ← Browser-based Canvas renderer
```

## Project Structure

```
src/apollo_lander/
├── constants.py          # Physical constants (lunar gravity, DPS specs, LM masses)
├── physics.py            # RK4 integrator and equations of motion (7D state vector)
├── transforms.py         # Body ↔ World frame coordinate rotations
├── guidance.py           # P66 guidance: RHC attitude, ROD descent rate, AGC throttle
├── wrappers.py           # FlatActionWrapper for Stable Baselines3 compatibility
├── webapp.py             # Flask web app with REST API
├── renderer.py           # Backward-compat entry point (launches webapp)
├── templates/
│   └── index.html        # HTML5 Canvas game with DSKY panel
├── autopilot.py          # Classical AGC autopilot (non-RL P66 controller)
├── manual.py             # Manual play mode (launches Flask server)
├── train.py              # RL training with PPO/SAC
├── evaluate.py           # Trained model evaluation
├── cli.py                # CLI entry points (apollo play/train/evaluate)
└── envs/
    ├── __init__.py       # Registers ApolloLander-v0
    └── apollo_lander_env.py  # Gymnasium environment
tests/
├── test_physics.py       # Physics engine and RK4 tests
├── test_guidance.py      # P66 controller and coordinate transform tests
├── test_env.py           # Gymnasium environment tests
└── test_cli.py           # CLI tests
```

## How It Works

In the real Apollo 11 landing, the astronaut didn't fly the Lunar Module like a helicopter. They flew **through the computer** using Program 66 (P66):

1. **Rotational Hand Controller (RHC):** A 3-axis joystick that commanded attitude *rates* (not positions). Push forward → the AGC pitches the craft forward at up to 20°/s. Release → the Digital Autopilot fires opposite RCS jets to hold that attitude.

2. **Rate of Descent (ROD) switch:** A spring-loaded toggle. Each click adjusted the AGC's target sink rate by exactly 1 ft/s (0.3048 m/s). The AGC then automatically throttled the Descent Propulsion System to maintain that rate.

3. **Descent Propulsion System (DPS):** The main engine, throttleable from 10% to 100% (4,504 N to 45,040 N). The AGC controlled the throttle — the astronaut never touched it directly.

This simulation replicates exactly that control scheme.

## Running Tests

```bash
pytest                    # Run all 35 tests
pytest -v                 # Verbose output
pytest tests/test_physics.py  # Physics tests only
```

## Dependencies

| Group | Packages |
|-------|----------|
| Core | numpy, scipy, gymnasium, click |
| Web UI | flask |
| RL Training | stable-baselines3, tensorboard |
| Development | pytest, pytest-cov, ruff, black, mypy |

## Documentation

- [docs/project.md](docs/project.md) — Project overview and specifications
- [docs/ground_truths.md](docs/ground_truths.md) — Key findings and design decisions
- [docs/getting-started.md](docs/getting-started.md) — Detailed setup guide

## License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.
