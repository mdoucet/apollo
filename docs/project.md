# Project Overview

## Project Name
Apollo Lander

## Purpose
A Python simulation of the Apollo Lunar Module descent, based on the real Apollo 11 Luminary099 AGC code. It provides a Gymnasium reinforcement learning environment and a manual play mode where users fly the spacecraft using the same controls the Apollo astronauts used (Rotational Hand Controller + Rate of Descent switch in Program 66 mode).

## Target Users
- Space enthusiasts wanting an authentic Apollo landing experience
- RL researchers looking for a realistic continuous-control environment
- Educators teaching orbital mechanics or control systems

## Core Functionality
1. Physics-accurate lunar descent simulation (RK4 integrator, MCI frame, mass depletion)
2. Ported Apollo P66 guidance logic (ROD switch, RHC attitude control, AGC throttle management)
3. Gymnasium environment (ApolloLander-v0) for RL agent training and evaluation
4. Manual play mode with Pygame rendering and DSKY-style display
5. RL training pipeline with Stable Baselines3 (PPO/SAC)

## Input/Output
**Input:** Player keyboard commands (arrow keys, PgUp/PgDn) or RL agent actions (4D continuous vector)

**Output:** Real-time simulation state (position, velocity, attitude, fuel), rendered 2D view, training metrics

## Example Usage

```python
# RL training
import gymnasium as gym
import apollo_lander.envs

env = gym.make("ApolloLander-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

```bash
# CLI usage
apollo play          # manual mode with Pygame
apollo train         # train RL agent
apollo evaluate      # evaluate trained agent
```

## Dependencies
- Core: numpy, scipy, gymnasium, click
- RL: stable-baselines3, tensorboard
- Visualization: pygame

## Technical Notes
- Coordinate system: Moon-Centered Inertial (MCI), not flat-moon approximation
- Physics dt: 0.1s (DAP rate), guidance recalculation every 2s
- Starting phase: P66 terminal descent (~150m altitude)
- AGC math ported to floating-point; no full AGC emulation
- Euler angles (ZYX) with scipy Rotation (quaternion-based internally)
