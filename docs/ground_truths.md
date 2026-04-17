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

### 2026-04-15: RHC-to-Attitude Coordinate Mapping

The RHC input flows through the guidance and environment as:
- `rhc[0]` → `guidance.target_attitude[0]` → `env._attitude[0]` = **roll** (X-axis rotation, produces Y-thrust component)
- `rhc[1]` → `guidance.target_attitude[1]` → `env._attitude[1]` = **pitch** (Y-axis rotation, produces X-thrust component)
- `rhc[2]` → `guidance.target_attitude[2]` → `env._attitude[2]` = **yaw**

The `body_to_world` transform uses ZYX Euler order: `Rotation.from_euler("ZYX", [yaw, pitch, roll])`.

**Implication**: To cancel horizontal X-velocity, command `rhc[1]` (physical pitch). To cancel Y-velocity, command `rhc[0]` (physical roll). The browser UI key labels ("pitch forward"/"roll left") are functionally swapped relative to the physics.

### 2026-04-15: AGC Autopilot Controller

`autopilot.py` implements a classical-style P66 autopilot as an alternative to the RL agent. Key design:
- PD controller for horizontal velocity nulling (kp=0.04, kd=0.3; tighter below 20m)
- ROD switch to follow a descent-rate schedule: −5/−3/−2/−1 ft/s by altitude band
- ROD cooldown of 10 ticks (1s) prevents switch bouncing
- MAX_TILT = 0.25 rad (~14°) limits pitch/roll excursion
- Achieves 100% landing success rate across 20-episode tests
- `predict(obs)` returns `(action_dict, None)` matching SB3's interface

### 2026-04-15: Faithful AGC RODCOMP Throttle Controller

The `guidance.py` throttle controller was rewritten to faithfully implement the real AGC P66 RODCOMP algorithm from `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` (pp. 816-819). Key changes:

**Before (non-authentic):** PI controller with kp=0.5, ki=0.1, anti-windup ±10.

**After (faithful to AGC source code):**
- `a_cmd = (VDGVERT - HDOTDISP) / TAUROD`
- **Proportional only** — the real AGC had NO integral term. P66 was intentionally simple.
- **TAUROD = 2.0s** — time constant (pad-loaded erasable, `ERASABLE_ASSIGNMENTS.agc` p.122)
- **Gravity compensation** — added separately: `total_accel = a_cmd + gravity`
- **Throttle lag compensation** — lead-lag filter: `FC += LAG/TAU * (FC - FCOLD)` where LAG/TAU = 0.1 (THROTLAG/TAUROD = 0.2s/2.0s)
- **FCOLD** — tracks previous commanded force (AGC erasable for lag filter state)

The proportional gain 1/TAUROD = 0.5 happens to match our previous kp=0.5, so performance is unchanged (20/20 landings, 100% success).

New constants added to `constants.py`:
- `TAUROD = 2.0` — ROD time constant (pad-loaded, `ERASABLE_ASSIGNMENTS.agc` p.122)
- `LAG_OVER_TAU = 0.1` — throttle lag ratio (pad-loaded, `ERASABLE_ASSIGNMENTS.agc` p.122)
- `THROTLAG = 0.2` — DPS actuator lag (`CONTROLLED_CONSTANTS.agc` p.40: `THROTLAG DEC +20`)

### 2026-04-15: AGC vs Crew Procedures Separation

The autopilot clearly separates what the **AGC computer** did from what the **astronaut crew** did:

**AGC RODCOMP (automated by computer):** Throttle control via the proportional law above. The autopilot sends ROD clicks → guidance.py processes them exactly as the real RODCOMP did (accumulate in VDGVERT, compute throttle). The AGC also computed VHORIZ for display but did NOT control it.

**Crew Procedures (simulated CDR):** Horizontal velocity nulling was 100% manual by the Commander using the RHC and LPDT cross-pointer display. Our autopilot simulates this with a PD controller — this is clearly labeled as a crew procedure approximation, not AGC code.

Source verification: Fetched and analyzed the real Luminary099 AGC source files:
- `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` — RODCOMP algorithm (pp. 816-819)
- `CONTROLLED_CONSTANTS.agc` — THROTLAG, SCALEFAC, FDPS values
- `ERASABLE_ASSIGNMENTS.agc` — TAUROD, LAG/TAU, RODSCALE, MINFORCE, MAXFORCE locations

### 2026-04-16: RHC Rate Command / Attitude Hold (RCAH) Mode

The RHC operated in **RCAH mode** during P66, which has two distinct states:

1. **Out of Detent (Rate Command):** Stick deflection → commanded rotation rate (proportional, max 20°/s at full deflection). The DAP fires RCS jets to maintain that rotational velocity.

2. **In Detent (Attitude Hold):** When the stick centers (springs snap it back), the AGC instantly captures the **current actual attitude** as the new hold target and fires opposing jets to kill residual rotation.

**Bug fixed:** Our original code always integrated `rhc * max_rate * dt` into `target_attitude`. When the stick centered, `target_attitude` could be ahead of actual `_attitude` (due to 0.3 tracking lag), causing the craft to keep rotating to catch up — the opposite of RCAH "freeze where you are" behavior. Fix: when `|rhc| < 0.01`, snap `target_attitude = attitude.copy()`.

The ROD switch was confirmed correct: `Discrete(3)` maps to edge-triggered clicks (each click = ±0.3048 m/s). Holding the switch has no additional effect, matching the spring-loaded toggle behavior.

### 2026-04-15: Altitude Reference — CG vs Footpads

The real AGC computed altitude as CG-to-surface (`HCALC = ABVAL(R1S) - /LAND/`) per SERVICER.agc. The LM footpads sit ~4.2 m below the CG. Our `compute_altitude()` returns footpad altitude = `r_norm - R_MOON - LM_CG_TO_FOOTPAD` so landing detection triggers when feet touch the surface, not the CG.

### 2026-04-15: Terrain Roughness

The real AGC assumed a perfectly spherical Moon (`HCALC = |R| - R_LUNAR`) and corrected the altitude estimate via Landing Radar feedback (Kalman-style position update in SERVICER.agc pp.884-885: `DELTAH = LR_altitude - HCALC`, weighted correction `WH*(1-H/HMAX)`).

Our simulation adds terrain roughness directly:
- **Terrain model**: Sum of 5 low-frequency sinusoids with random amplitude (0.3–1.5 m), wavelength (40–200 m), and phase. Landing pad (x=0) is pinned to zero elevation: `h(x) = Σ amp * [sin(2π*x/λ + φ) - sin(φ)]`.
- **Altitude adjustment**: `terrain_altitude = compute_altitude(state) - terrain_height(x_horiz)`. Landing detection uses terrain-adjusted altitude.
- **Terrain data flow**: Generated at `env.reset()` → stored as `_terrain_components` → passed via info dict → webapp sends to client → JavaScript `terrainHeight(x)` mirrors the server formula → canvas draws filled polygon surface.
- **Impact**: Autopilot maintains 100% landing success (20/20). Terrain variation is small (a few meters) compared to typical approach altitude (150 m), consistent with Apollo mare landing sites.

### 2026-04-16: Three Play Modes — Crew vs. Computer Division of Labor

The simulation offers three modes that map to the real crew/computer division of labor in P66:

| Mode | CLI Command | Player controls | Computer controls |
|---|---|---|---|
| **Manual** | `apollo play` | RHC + ROD (full astronaut experience) | Throttle (RODCOMP) + attitude hold (RCAH) |
| **Assisted** | `apollo assisted` | RHC only (horizontal steering) | Throttle + attitude hold + ROD scheduling |
| **Autopilot** | `apollo autopilot` | Nothing (observe) | Everything (RHC + ROD + throttle + attitude hold) |

**Manual** is the authentic P66 CDR experience — the player does exactly what the real astronaut did (steer with RHC, schedule descent rate with ROD clicks), while the AGC handles throttle and attitude hold automatically.

**Assisted** is a training mode: the autopilot handles ROD scheduling (the more procedural part), letting the player focus on horizontal velocity nulling with the RHC (the skill-intensive part).

**Autopilot** automates everything, including crew procedures (horizontal steering and ROD scheduling) that the real AGC did NOT do. These crew procedure approximations use a PD controller for horizontal nulling and altitude-based ROD scheduling — clearly separated from the authentic AGC code.
