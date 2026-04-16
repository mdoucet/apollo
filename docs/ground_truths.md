# Ground Truths

This file captures key findings, decisions, and verified facts discovered during development. It serves as a persistent knowledge base that AI assistants (like GitHub Copilot) and developers can reference across sessions.

**Why this matters:** AI assistants don't remember previous conversations. By recording important discoveries here, you ensure that context isn't lost between sessions. When Copilot reads this file, it can make better suggestions based on what's already been learned about your project.

## How to Use This File

- **Add entries as you discover important facts** — things like API quirks, configuration requirements, performance constraints, or design decisions.
- **Include the date and context** so future-you (or Copilot) understands why something was noted.
- **Link to relevant code or docs** when helpful.
- Copilot is instructed to update this file automatically when it discovers key findings during development.

## Findings

### 2026-04-14: AGC Fixed-Point to Float Conversion Strategy

The Apollo Guidance Computer used 15-bit 1's complement fixed-point arithmetic with scale factors. Rather than emulating this directly, we port the mathematical logic to standard Python float64. The AGC represented numbers as fractions in (-1, 1) with scale factors mapping to physical SI units.

**Decision**: Port math only, not the fixed-point representation. Scale factors are implicit in our SI-unit constants.

### 2026-04-14: P66 Control Scheme Is Key to Authenticity

In Program 66 (the manual landing phase), the astronaut did NOT directly control thrust or thruster firings. Instead:
- **Rotational Hand Controller (RHC)**: Commands attitude *rates*, not positions. Releasing the stick causes the DAP to hold the current attitude.
- **Rate of Descent (ROD) switch**: Each click adjusts the AGC's target sink rate by exactly 1 ft/s (0.3048 m/s). The AGC handles throttle to match.

This is the control scheme we replicate in `guidance.py`.

### 2026-04-14: DPS Throttle Floor at 10%

The Descent Propulsion System could not throttle below ~10% (4504 N) due to injector instability. This is enforced in our guidance controller with `np.clip(thrust, MIN_THRUST, MAX_THRUST)`.

### 2026-04-14: Gymnasium Dict Action Space vs RL Agents

Stable Baselines3 PPO/SAC work best with flat Box spaces. Our native action space is `Dict(rhc=Box(3), rod=Discrete(3))`. We created `FlatActionWrapper` in `wrappers.py` to flatten this to `Box(4)` for RL training, mapping the 4th continuous dimension to discrete ROD clicks via thresholds (>0.33 → up, <-0.33 → down).

### 2026-04-14: Physics Timing

- **DAP cycle**: 0.1s (10 Hz) — this is our `env.step()` dt
- **Guidance recalculation**: 2.0s (0.5 Hz) — full guidance loop period
- **RK4 integrator**: Uses the 0.1s DAP dt, which gives good accuracy for the descent phase

### 2026-04-14: MCI Coordinate Frame Choice

Using Moon-Centered Inertial frame (not flat-moon approximation) ensures accurate gravity modeling via `a = -mu/|r|^3 * r`. This matters for longer descent trajectories and is more physically correct than assuming constant g.

### 2026-04-15: Landing Termination States

The simulation ends in one of four outcomes:
- **Landing (success)**: Touched surface with vertical vel ≤ 1.2 m/s AND horizontal vel ≤ 0.6 m/s.
- **Crash**: Touched surface exceeding velocity limits. The LM landing gear was designed for up to 2.1 m/s (7 ft/s) vertical impact; our 1.2 m/s limit provides operational margin matching Apollo 11's actual ~0.9 m/s touchdown.
- **Fuel exhaustion**: DPS propellant depleted before touchdown.
- **Timeout**: Exceeded `max_steps` (3000 steps = 300s at 10 Hz).

Previously all failure modes showed "ABORT". The UI now distinguishes these cases and explains which velocity limits were exceeded on crash. See `apollo_lander_env.py` `step()` for the `termination_reason` field.
