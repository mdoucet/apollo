# How the Apollo Guidance Computer Landed on the Moon

*A deep dive into the real AGC source code and how we faithfully reimplemented it*

---

## Table of Contents

1. [Introduction](#introduction)
2. [The AGC at a Glance](#the-agc-at-a-glance)
3. [Program 66: The Final Descent](#program-66-the-final-descent)
4. [RODCOMP: The Throttle Controller](#rodcomp-the-throttle-controller)
5. [The Digital Autopilot (DAP)](#the-digital-autopilot-dap)
6. [Navigation: SERVICER and the Landing Radar](#navigation-servicer-and-the-landing-radar)
7. [The Descent Propulsion System (DPS)](#the-descent-propulsion-system-dps)
8. [Lunar Module Mass Properties](#lunar-module-mass-properties)
9. [Coordinate Frames and Transforms](#coordinate-frames-and-transforms)
10. [Altitude: CG vs. Footpads vs. Contact Probes](#altitude-cg-vs-footpads-vs-contact-probes)
11. [Terrain and the Landing Radar](#terrain-and-the-landing-radar)
12. [What the Computer Did vs. What the Astronaut Did](#what-the-computer-did-vs-what-the-astronaut-did)
13. [Fixed-Point Arithmetic](#fixed-point-arithmetic)
14. [AGC Source Files Reference](#agc-source-files-reference)
15. [Our Implementation: Mapping AGC to Python](#our-implementation-mapping-agc-to-python)
16. [Glossary](#glossary)

---

## Introduction

On July 20, 1969, Apollo 11's Lunar Module *Eagle* landed on the Moon. The final
approach was controlled by the **Apollo Guidance Computer (AGC)** — a 72 KB,
2 MHz machine with less processing power than a modern doorbell camera.

The AGC source code, written in a custom assembly language called **AGC4**, was
preserved by MIT and is now [publicly available on GitHub](https://github.com/chrislgarry/Apollo-11/tree/master/Luminary099). We studied this code — specifically
the **Luminary099** version (Apollo 11 flight software) — and rebuilt its
landing algorithms in Python.

This document explains how the real AGC code worked, what each piece did, and
how our simulation faithfully reproduces it. Every AGC source reference includes
the specific file and page number so you can read the original yourself.

---

## The AGC at a Glance

| Property | Value |
|---|---|
| **Word size** | 15 bits + parity (1's complement) |
| **Memory** | 2K words erasable (RAM), 36K words fixed (ROM) |
| **Clock speed** | 2.048 MHz |
| **Instruction set** | ~30 instructions |
| **Language** | AGC4 assembly (macro assembler: YUL) |
| **Missions** | Block II AGC used on Apollo 7–17 |
| **Flight software** | Luminary (LM), Colossus (CM) |

The AGC stored numbers as **15-bit 1's complement fixed-point fractions** in the
range (-1, +1), with implicit scale factors mapping to physical units. For
example, a velocity might be stored as a fraction with a scale factor of 2^14
meters per centisecond.

---

## Program 66: The Final Descent

The Apollo descent was managed by a series of **programs** (think: flight modes):

| Program | Phase | Altitude | Control |
|---|---|---|---|
| **P63** | Braking | ~15,000 m → ~2,400 m | Fully automatic |
| **P64** | Approach | ~2,400 m → ~150 m | Auto w/ Landing Point Designator |
| **P66** | Manual landing | ~150 m → touchdown | Semi-manual |

**P66** is where our simulation begins. The Commander (CDR) typically switched to
P66 when the LPD (Landing Point Designator) angle reached zero, at about 500
feet (152 m) altitude.

### What "semi-manual" means

P66 was not fully manual and not fully automatic. The astronaut had two controls:

1. **Rotational Hand Controller (RHC)** — a 3-axis joystick with heavy spring
   centering, operating in **Rate Command / Attitude Hold (RCAH)** mode. When
   pushed out of its center detent, the stick deflection commands a rotation
   *rate* proportional to deflection (max 20°/s at full throw). The DAP fires
   RCS jets to maintain that rotational velocity. The instant the astronaut
   releases the stick, the springs snap it back to center and the AGC
   immediately captures the spacecraft's **current attitude** as the new hold
   target — firing opposing jets to kill any residual rotation and freezing the
   craft at that angle indefinitely.

2. **Rate of Descent (ROD) switch** — a spring-loaded toggle switch on the
   commander's panel. It rests in a neutral center position. Pushing it up or
   down completes a circuit, but the spring snaps it back as soon as the
   astronaut lets go. The AGC registers only the **rising edge** of the signal —
   holding the switch has no additional effect. Each click adjusts the target
   sink rate by exactly **±1 ft/s (±0.3048 m/s)**. To change the rate by 3 ft/s,
   the astronaut had to physically actuate the switch three distinct times.

The astronaut never directly controlled thrust level. The CDR set a sink rate
with ROD clicks and steered laterally with the RHC. The AGC did everything else.

```
┌──────────────────────────────────────────────────────────────┐
│                    P66 Control Flow                          │
│                                                              │
│  Astronaut                     AGC Computer                  │
│  ─────────                     ────────────                  │
│  RHC deflection ──────────────► DAP attitude hold            │
│  ROD switch click ────────────► RODCOMP throttle control     │
│  (eyes out the window,         (VDGVERT accumulator,         │
│   LPDT cross-pointers)          proportional law,            │
│                                  gravity compensation,       │
│                                  lag compensation,           │
│                                  DPS throttle commands)      │
└──────────────────────────────────────────────────────────────┘
```

**Source:** `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` pp. 798–828

### Mapping to a Gymnasium action space

These physical controls map naturally to the RL environment:

```python
action_space = Dict({
    "rhc": Box(low=-1.0, high=1.0, shape=(3,)),  # RHC stick deflection
    "rod": Discrete(3),                            # ROD switch: 0=none, 1=up, 2=down
})
```

**RHC → `Box(3, [-1, 1])`:** A continuous value maps perfectly to RCAH. The
agent outputs `[0.5, 0.0, 0.0]` → the DAP commands 10°/s rotation (half of
20°/s max). The agent outputs `[0.0, 0.0, 0.0]` → the DAP captures the current
attitude and holds it. The RCAH logic means the agent doesn't need to learn
high-frequency jet firing patterns — just that pushing the stick changes heading
and releasing it locks the heading.

**ROD → `Discrete(3)`:** Each non-zero action is a single edge-triggered click.
The action returns to "neutral" (0) automatically on the next step, matching the
spring-loaded toggle behavior. The agent must output `1` or `2` on separate
steps to get multiple clicks.

---

## RODCOMP: The Throttle Controller

This is the heart of P66 — the algorithm that kept the LM descending at the
rate the astronaut commanded. It is called **RODCOMP** in the AGC source code.

### The algorithm in the original AGC assembly

From `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` (pp. 816–819):

```
RODCOMP:    XCH   RODCOUNT        # Read accumulated ROD switch clicks
            MP    RODSCAL1        # Scale: clicks → velocity change
            DAS   VDGVERT         # VDGVERT += scaled clicks

            CA    VDGVERT         # Load desired vertical velocity
            EXTEND
            MSU   HDOTDISP        # Subtract measured vertical velocity
            EXTEND
            MP    RODSCALE        # Divide by TAUROD (multiply by 1/TAUROD)
            # Result = (VDGVERT - HDOTDISP) / TAUROD = commanded acceleration
```

### The algorithm in plain math

$$a_{\text{cmd}} = \frac{V_{\text{DGVERT}} - \dot{H}_{\text{DISP}}}{\tau_{\text{ROD}}}$$

Where:
- $V_{\text{DGVERT}}$ = desired vertical velocity (set by ROD switch clicks)
- $\dot{H}_{\text{DISP}}$ = measured vertical velocity from navigation
- $\tau_{\text{ROD}}$ = time constant ≈ 2.0 seconds

This is a **proportional controller** — the commanded acceleration is proportional
to the velocity error. If you're sinking too fast, thrust up harder. If you're
sinking too slowly, ease off.

### Key insight: No integral term

The real AGC had **no integral term** in P66. Modern control engineers might
expect a PI or PID controller, but MIT's Instrumentation Lab deliberately kept
P66 simple:

- Proportional-only is inherently stable (no wind-up, no oscillation risk)
- The astronaut was in the loop, making corrections via ROD clicks
- P66 was only active for ~2 minutes — not long enough for steady-state error to matter
- Gravity compensation handles the constant offset that an integral term would address

### Gravity compensation

The commanded acceleration alone would let the LM fall. The AGC adds gravity:

$$F_{\text{cmd}} = (a_{\text{cmd}} + g) \times m$$

Where $g = \mu_{\text{Moon}} / r^2 \approx 1.625$ m/s² at the surface.

### Throttle lag compensation

The DPS throttle actuator wasn't instantaneous — it had a time constant of
**0.2 seconds** (the pintle valve needed time to move). The AGC compensated with
a **lead-lag filter**:

$$F_C = F_C + \frac{\text{LAG}}{\text{TAU}} \times (F_C - F_{C,\text{old}})$$

Where:
- $F_{C,\text{old}}$ = previous commanded force (AGC erasable `FCOLD`)
- LAG/TAU = THROTLAG / TAUROD = 0.2s / 2.0s = **0.1** (dimensionless)

This effectively adds a derivative kick to the force command, making the throttle
respond faster than the actuator's natural lag would allow.

### Constants

| Constant | AGC Name | Value | Source |
|---|---|---|---|
| Time constant | `TAUROD` | 2.0 s | `ERASABLE_ASSIGNMENTS.agc` p.122 (pad-loaded) |
| Lag/tau ratio | `LAG/TAU` | 0.1 | `ERASABLE_ASSIGNMENTS.agc` p.122 (pad-loaded) |
| Throttle lag | `THROTLAG` | 0.2 s (20 cs) | `CONTROLLED_CONSTANTS.agc` p.40 |
| ROD increment | `RODSCAL1` | 0.3048 m/s (1 ft/s) | `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` p.816 |

Note: TAUROD and LAG/TAU were **pad-loaded erasable** — meaning they were
uploaded to the AGC's RAM before each mission and could be adjusted per-flight.
They were not burned into the ROM.

### Our Python implementation

```python
# guidance.py — RODCOMP faithful implementation

# RODCOMP: commanded acceleration
a_cmd = (self.target_descent_rate - current_vz) / TAUROD

# Gravity compensation
total_accel = a_cmd + gravity_accel

# Convert to force
commanded_force = total_accel * mass

# Throttle lag compensation: FC += LAG/TAU * (FC - FCOLD)
commanded_force += LAG_OVER_TAU * (commanded_force - self._fcold)
self._fcold = commanded_force

# Clip to DPS operating range
commanded_thrust = np.clip(commanded_force, MIN_THRUST, MAX_THRUST)
```

---

## The Digital Autopilot (DAP)

The DAP was a separate piece of software from the guidance equations. It ran at
**10 Hz** (every 100 ms) via the T6 interrupt, implementing the **Rate Command /
Attitude Hold (RCAH)** paradigm:

### RCAH: Two modes in one controller

**State 1 — Out of Detent (Rate Command):**
When the astronaut pushed the stick out of its center detent, transducers
converted the physical angle into an electrical signal. The AGC read this signal
as a *commanded rate of rotation*:

- A slight push might command 5°/s
- Full deflection (hard stop) commanded the maximum: **20°/s**
- The DAP fired RCS jets as needed to maintain that specific rotational velocity
- The spacecraft's attitude changed continuously at the commanded rate

**State 2 — In Detent (Attitude Hold):**
When the astronaut released the stick, the physical springs snapped it to the
absolute center (the detent). At that instant:

1. The AGC read zero deflection → set commanded rate to 0°/s
2. **Critically:** the AGC recorded the spacecraft's *current spatial attitude*
   as the new hold target
3. The DAP fired opposing RCS jets to kill whatever rotational momentum remained
4. The craft was frozen at that angle and held there indefinitely

This RCAH logic is elegant: the astronaut doesn't need to think about
limit-cycle jet firing patterns. Push the stick → the spacecraft turns. Let go →
it stops and stays. The AGC handles all the high-frequency thruster control
automatically.

### Timing architecture

### Timing architecture

```
┌─────────────────────────────────────────────────┐
│             AGC Task Schedule                    │
│                                                  │
│  Every 100 ms (10 Hz):  DAP cycle (T6 RUPT)     │
│       ├── Read RHC inputs                        │
│       ├── Compute attitude errors                │
│       ├── Fire RCS jets for attitude hold         │
│       └── Apply thrust vector                    │
│                                                  │
│  Every 2.0 s (0.5 Hz):  SERVICER/Guidance cycle  │
│       ├── Read accelerometers (READACCS)         │
│       ├── Update navigation state                │
│       ├── Process Landing Radar data             │
│       ├── Run guidance equations (LUNLAND)       │
│       └── Compute new thrust commands            │
│                                                  │
│  Source: SERVICER.agc — 2SECS DEC 200 (p.883)   │
└─────────────────────────────────────────────────┘
```

### Our implementation

We model the DAP as the `env.step()` loop running at 10 Hz (dt = 0.1 s). The
guidance equations in `guidance.py` run at 2.0 s intervals within this loop.

The RCAH logic is implemented in `guidance.py`:

```python
if rhc_magnitude > 0.01:
    # Rate Command: integrate commanded rate into target attitude
    commanded_rates = rhc_input * self.max_rate  # max_rate = 20°/s
    self.target_attitude += commanded_rates * dt
else:
    # Attitude Hold: capture current attitude as new target
    # This is the key RCAH behavior — freeze at current orientation
    self.target_attitude = attitude.copy()
```

The 0.01 deadband corresponds to the physical detent — the spring centering
mechanism ensured the RHC couldn't rest at tiny non-zero deflections.

---

## Navigation: SERVICER and the Landing Radar

The **SERVICER** routine was the AGC's navigation engine. Every 2 seconds, it:

1. Read the IMU accelerometers (`READACCS` task)
2. Integrated acceleration to update the state vector (position + velocity)
3. Computed a "dead reckoning" altitude
4. Incorporated Landing Radar corrections
5. Passed the updated state to the guidance equations

### Dead reckoning altitude

The AGC computed altitude by assuming a perfectly **spherical Moon**:

$$H_{\text{CALC}} = |R| - R_{\text{LUNAR}}$$

From `SERVICER.agc` (p. 876):
```
HCALC:  VLOAD   R1S             # Load position vector
        ABVAL                    # Compute magnitude |R|
        DSU     /LAND/           # Subtract lunar radius
        STORE   HCALC            # Store altitude
```

This simple calculation ignores terrain roughness entirely — the Moon is treated
as a perfect sphere.

### Landing Radar correction

The Landing Radar measured the actual slant range (and velocity) to the surface
below. The SERVICER used this to correct its dead-reckoning estimate via a
**Kalman-style position update**:

From `SERVICER.agc` (pp. 884–885):
```
# Compute altitude discrepancy
DELTAH = LR_altitude - HCALC

# Weight decreases with altitude (trust LR more when close)
W = WH * (1 - H/HMAX)

# Correct position vector
R_corrected = R + W * DELTAH * R_hat
```

This is elegant: the AGC **didn't try to model terrain**. Instead, it measured
the real surface via radar and nudged its state estimate to match. The weighting
function `(1 - H/HMAX)` means the correction was strongest close to the surface
(where terrain matters most) and negligible at high altitude (where the sphere
approximation is fine).

### Implications for our simulation

The navigation architecture means that the downstream code (RODCOMP, the DAP,
and the crew procedures) **never knew about terrain**. They just received an
altitude number from the navigation filter and trusted it.

Our simulation mirrors this correctly: the environment computes the true
terrain-adjusted altitude and passes it in the observation vector. The guidance
and autopilot code see an accurate altitude directly — equivalent to having a
perfect Landing Radar and navigation filter.

---

## The Descent Propulsion System (DPS)

The DPS was a single, gimbaled, throttleable engine built by TRW. It had unusual
operating constraints that the AGC had to respect.

### Throttle range and the "dead zone"

```
                    DPS Throttle Map
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  0%──10%───────────57%────────94%──────100%      │
  │  │   │              │         │         │        │
  │  │   MIN_THRUST     │         FMAXPOS   FMAXODD  │
  │  │   (4,504 N)      │         (43,455 N)(48,145 N)│
  │  │                  │                            │
  │  │   Throttleable   │ NON-THROTTLEABLE  Saturated│
  │  │   range          │ ZONE (unstable)   (FTO)    │
  │  │   10%–57%        │ 57%–94%                    │
  │  └──────────────────┘                            │
  │                                                  │
  │  For P66 (near touchdown), the engine operated   │
  │  in the lower throttleable range (10%–57%).      │
  └──────────────────────────────────────────────────┘
```

The DPS used a **pintle injector** (like a SpaceX Merlin engine today) that could
not maintain stable combustion below about 10% thrust. The AGC enforced this
floor:

| AGC Constant | Value | Description |
|---|---|---|
| `FDPS` | 43,670 N (9,817 lbf) | Nominal DPS force for guidance |
| `FMAXPOS` | 43,455 N (3467 counts) | Upper throttleable boundary |
| `FMAXODD` | 48,145 N (3841 counts) | Absolute max ("flat out") |
| `MINFORCE` | ~4,504 N | Pad-loaded minimum (erasable) |
| `THROTLAG` | 0.2 s | Actuator time constant |

**Sources:**
- `CONTROLLED_CONSTANTS.agc` p.38: `FDPS 2DEC 4.3670 B-7`
- `THROTTLE_CONTROL_ROUTINES.agc` p.797: `FMAXPOS DEC +3467`, `FMAXODD DEC +3841`

### Specific impulse

```
AGC:  DPSVEX  DEC* -2.95588868 E+1 B-05*    [CONTROLLED_CONSTANTS.agc p.39]
      Comment: "VE (DPS) +2.95588868E+3"
      = 2,955.89 m/s exhaust velocity
      = ISP of 301.4 s (effective, for mass computation)

NASA Operations Handbook:
      ISP_vacuum = 311 s (thermodynamic, TRW specification)
```

The AGC's `DPSVEX` (exhaust velocity) was used for onboard mass tracking, not
for engine performance prediction. The slight difference from the spec ISP
(301 vs 311 s) reflects the effective vs. thermodynamic values.

---

## Lunar Module Mass Properties

The AGC tracked mass carefully because thrust-to-weight ratio directly affected
the throttle commands. The key mass constants from `CONTROLLED_CONSTANTS.agc`
(p.43):

| AGC Constant | Value | Description |
|---|---|---|
| `HIDESCNT` | 15,300 kg | Maximum descent LM mass |
| `LODESCNT` | 1,750 kg | Minimum descent stage mass (structural) |
| `FULLAPS` | 5,050 kg | Full ascent stage mass |
| `MINMINLM` | 2,200 kg | Minimum ascent stage (dry) |
| `MINLMD` | 2,850 kg | Min descent mass (w/ residual propellant) |
| `MDOTDPS` | 14.80 kg/s | DPS mass flow rate (0.1480 kg/cs) |

### Our implementation values (from Apollo 11 LM-5 specs)

| Parameter | Value | AGC Equivalent |
|---|---|---|
| Descent stage dry mass | 2,034 kg | Between LODESCNT and MINLMD |
| Ascent stage mass | 4,670 kg | Close to FULLAPS (5,050) |
| DPS propellant | 8,200 kg | HIDESCNT − LODESCNT − FULLAPS |
| Total mass at PDI | 14,904 kg | Close to HIDESCNT (15,300) |

---

## Coordinate Frames and Transforms

### Moon-Centered Inertial (MCI) frame

The AGC maintained its state vector in an inertial reference frame centered on
the Moon. We use the same approach:

- **Origin**: Center of the Moon
- **Axes**: Fixed relative to the stars (non-rotating)
- **Gravity**: Computed exactly as $\vec{a} = -\frac{\mu}{|\vec{r}|^3} \vec{r}$

This is more physically correct than a "flat Moon" approximation with constant
gravity, and it's what the AGC actually used.

### Body-to-inertial transform

The DPS engine fires along the spacecraft's body +Z axis. To get the thrust
vector in MCI coordinates, we apply the attitude rotation:

$$\vec{F}_{\text{MCI}} = R_Z(\psi) \cdot R_Y(\theta) \cdot R_X(\phi) \cdot \begin{pmatrix} 0 \\ 0 \\ F \end{pmatrix}$$

Where $\phi$ = roll, $\theta$ = pitch, $\psi$ = yaw (ZYX Euler convention).

We use `scipy.spatial.transform.Rotation` (quaternion-based internally) to avoid
gimbal lock issues that plagued earlier flight software.

### RHC-to-attitude mapping

The Rotational Hand Controller input maps through the system as:

```
RHC axis    → Guidance target  → Env attitude  → Thrust effect
─────────     ───────────────    ─────────────    ────────────
rhc[0]      → attitude[0]      → roll (φ)      → Y-force component
rhc[1]      → attitude[1]      → pitch (θ)     → X-force component
rhc[2]      → attitude[2]      → yaw (ψ)       → rotation only
```

This means to cancel X-velocity, you command pitch (rhc[1]). To cancel
Y-velocity, you command roll (rhc[0]). This matches the real LM where tilting
the spacecraft redirected the DPS thrust vector to create horizontal forces.

---

## Altitude: CG vs. Footpads vs. Contact Probes

The Apollo program used three different altitude reference points, which caused
real confusion during development and operations:

### 1. Center of Gravity (CG) altitude — what the AGC computed

```
AGC:  HCALC = ABVAL(R1S) - /LAND/     [SERVICER.agc p.876]
```

The AGC's state vector tracked the LM's center of mass. HCALC was the distance
from the CG to the assumed spherical surface.

### 2. Footpad altitude — what matters for landing

The landing gear footpads sit **4.2 meters below the CG** (at typical landing
mass). This offset matters: when the AGC reports altitude zero, the footpads are
actually 4.2 m underground.

```
LM Vertical Layout (landing configuration)
┌──────┐
│ Asc. │  Ascent stage CG ≈ 5.1 m above footpads
│ Stage│
├──────┤  Interface ≈ 3.23 m above footpads
│ Desc.│
│ Stage│  Descent stage CG ≈ 2.3 m above footpads
├──/\──┤
│ /  \ │  Landing gear (deployed)
└/────\┘
▼      ▼  Footpads: 0 m (reference)
═══════════════════  Lunar surface

Combined CG ≈ (2034×2.3 + 4670×5.1) / 6704 ≈ 4.2 m above footpads
```

### 3. Contact probes — what triggered engine shutdown

The real LM had **contact probes** extending 67 inches (1.7 m) below the
footpads. When a probe touched the surface, the "CONTACT LIGHT" illuminated and
the crew was supposed to shut down the engine. (Armstrong actually flew past the
contact probe signal on Apollo 11.)

### Our implementation

We reference altitude to the **footpads** for intuitive display:
```python
altitude = r_norm - R_MOON - LM_CG_TO_FOOTPAD  # 0 = feet on ground
```

---

## Terrain and the Landing Radar

### How the real AGC handled terrain

The AGC assumed a **perfectly spherical Moon**. It had no terrain model, no
topographic database, no concept of hills or craters. The altitude was simply
the distance from the CG to the center of the Moon, minus the lunar radius.

To handle the reality that the Moon is *not* a smooth sphere, the system relied
on the **Landing Radar (LR)**:

```
┌─────────────────────────────────────────────────────────────┐
│                Landing Radar Data Flow                       │
│                                                             │
│  LR antenna ──► Slant range measurement                     │
│                      │                                      │
│  SERVICER ──────────►├─► DELTAH = LR_alt - HCALC            │
│  (dead reckoning)    │                                      │
│                      ├─► Weight = WH × (1 - H/HMAX)         │
│                      │                                      │
│                      └─► R_corrected = R + W × DELTAH × R̂   │
│                                │                            │
│                                ▼                            │
│                     Corrected state vector                   │
│                                │                            │
│                                ▼                            │
│                     RODCOMP (just sees altitude)             │
│                     Crew (just sees altitude)                │
└─────────────────────────────────────────────────────────────┘
```

The Kalman-style weighting `(1 - H/HMAX)` is clever: at high altitude the LR
beam footprint is large and the measurement is noisy, so the weight is small. As
the LM descends, the LR becomes more accurate and the weight increases toward
WH (maximum correction gain).

**Source:** `SERVICER.agc` pp. 884–885

### Our terrain model

Since we don't simulate a Landing Radar instrument, we add terrain roughness
directly to the environment and give the guidance code the true altitude:

- **Generation**: Sum of 5 random sinusoids per episode
  - Amplitude: 0.3–1.5 m (consistent with Apollo mare landing sites)
  - Wavelength: 40–200 m (gentle undulations, not sharp craters)
  - Phase: random

- **Mathematical model**:
$$h(x) = \sum_{i=1}^{5} A_i \left[ \sin\!\left(\frac{2\pi x}{\lambda_i} + \phi_i\right) - \sin(\phi_i) \right]$$

  The $-\sin(\phi_i)$ term ensures the landing pad (x=0) is always at zero
  elevation.

- **Effect on altitude**: `terrain_altitude = compute_altitude(state) - terrain_height(x)`

This is physically equivalent to having a perfect Landing Radar: the guidance
code sees the true distance to the actual surface, just as it would if the
navigation filter perfectly incorporated LR corrections.

---

## What the Computer Did vs. What the Astronaut Did

This distinction is crucial for understanding P66. Many accounts describe the
landing as "manual," but the AGC was doing most of the work:

### AGC (automated)

| Task | How | Source |
|---|---|---|
| **Throttle control** | RODCOMP proportional law | `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` pp. 816–819 |
| **Attitude hold** | DAP with RCS jet firing | T6 interrupt, 10 Hz |
| **Navigation** | IMU integration + LR corrections | `SERVICER.agc` |
| **Displays** | DSKY (V06N60), cross-pointers | `R10/R11` display routines |
| **ROD processing** | Accumulate clicks into VDGVERT | RODCOMP: `DAS VDGVERT` |

### Astronaut (manual)

| Task | How | Why not automated? |
|---|---|---|
| **Horizontal steering** | RHC pitch/roll while watching LPDT | CDR needed to choose landing spot visually |
| **Descent rate** | ROD toggle switch clicks | CDR judged safe sink rate by eye |
| **Landing site** | Look out window, avoid rocks | No terrain database in AGC |
| **Engine shutdown** | STOP button after contact light | Safety: crew confirms gear on ground |
| **P66 selection** | DSKY PRO key from P64 | CDR decides when to take manual control |

The AGC computed horizontal velocity (VHORIZ) and displayed it on the
cross-pointers, but it **did not attempt to control it**. The
`FLASHVHZ`/`FLASHVHY` routines in the AGC drove the LPDT display needles, but
the RHC commands that nulled horizontal velocity came entirely from the CDR's
hands.

### Our autopilot's approach

```python
# AGC RODCOMP (faithful to real code):
#   - guidance.py handles throttle via the proportional law
#   - autopilot.py sends ROD clicks to adjust VDGVERT
#   → This is exactly how the real hardware worked

# Crew procedures (simulated CDR):
#   - PD controller for horizontal velocity nulling
#   - Descent-rate schedule matching Armstrong's profile
#   → Clearly labeled as crew procedure approximation
```

---

## Fixed-Point Arithmetic

The AGC used **15-bit 1's complement fixed-point** arithmetic. Understanding this
is essential for reading the source code.

### How it worked

Every number was stored as a fraction in the range (-1, +1) with an implicit
**scale factor** (called the "B-N" notation):

```
2DEC  4.9027780 E8  B-30
         │              │
         │              └── Scale factor: multiply by 2^(-30)
         └── Decimal value to store

Stored value = 4.9027780e8 × 2^(-30)
             = 0.456... (fits in [-1, +1])

To recover: stored_value × 2^30 = 4.9027780e8 m³/cs²
```

### Common patterns in AGC code

```
# Multiplication with scale factor adjustment
MP    RODSCALE     # Multiply accumulator by RODSCALE
                   # Product has combined scale factor

# Double-precision add
DAS   VDGVERT      # Add accumulator to VDGVERT (double-precision)

# Conditional branch
BZMF  LABEL        # Branch if accumulator is zero or negative
```

### Our approach: math only, not representation

We port the **mathematical algorithms** from the AGC to Python float64, not the
fixed-point representation. The scale factors become implicit in our SI-unit
constants:

```
AGC:    MUM  2DEC* 4.9027780 E8 B-30*     → Python: MU_MOON = 4.9048695e12 (m³/s²)
AGC:    RSUBM  2DEC 1738090 B-29          → Python: R_MOON = 1737400.0 (m)
AGC:    THROTLAG  DEC +20                 → Python: THROTLAG = 0.2 (s)
```

This preserves the algorithm's behavior while avoiding the numerical precision
limitations of 15-bit arithmetic.

---

## AGC Source Files Reference

All references are to the [Luminary099](https://github.com/chrislgarry/Apollo-11/tree/master/Luminary099) repository (Apollo 11 LM flight software):

| File | Contents | Key Pages |
|---|---|---|
| `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` | P63/P64/P66 guidance, RODCOMP | pp. 798–828 |
| `CONTROLLED_CONSTANTS.agc` | Engine params, masses, mu, radii | pp. 38–53 |
| `ERASABLE_ASSIGNMENTS.agc` | RAM variable locations (TAUROD, etc.) | p. 122 |
| `THROTTLE_CONTROL_ROUTINES.agc` | FMAXPOS, FMAXODD, throttle commanding | pp. 793–797 |
| `SERVICER.agc` | Navigation, READACCS, LR processing, HCALC | pp. 857–897 |
| `THE_LUNAR_LANDING.agc` | P63/P64 automated descent | — |
| `R10,R11.agc` | DSKY/LPDT display routines | — |

### How to read AGC source

```
# Comments start with '#' (added by the digitizers)
# Labels are left-justified, instructions are indented

RODCOMP    CA     VDGVERT         # CA = "Clear and Add" (load A register)
           EXTEND                  # Next instruction is "extended" set
           MSU    HDOTDISP        # MSU = "Modular Subtract"
           EXTEND
           MP     RODSCALE        # MP = "Multiply"
           DXCH   THRUST_CMD      # DXCH = "Double Exchange" (store double-precision)
```

Key instructions:
- `CA` — Load accumulator
- `MP` — Multiply
- `DAS` — Double-precision add to storage
- `MSU` — Modular subtract
- `BZMF` — Branch if zero or minus
- `TC` — Transfer control (function call)
- `EXTEND` — Next instruction is from the extended instruction set

---

## Our Implementation: Mapping AGC to Python

### Architecture mapping

```
Real Apollo System          Our Simulation
──────────────────          ──────────────
AGC P66 Software       →    guidance.py (RODCOMP throttle controller)
DAP (attitude hold)    →    guidance.py (PD attitude controller)
SERVICER (navigation)  →    physics.py + env (altitude, velocity computation)
Landing Radar          →    Terrain-adjusted altitude (perfect LR equivalent)
RHC Hardware           →    action["rhc"] from keyboard/autopilot
ROD Switch             →    action["rod"] from keyboard/autopilot
DPS Engine             →    physics.py (thrust → RK4 integration)
DSKY Display           →    index.html canvas (PROG/VERB/NOUN/R1-R3)
Crew Procedures        →    autopilot.py (PD horizontal nulling)
```

### File-by-file mapping

| Our File | AGC Equivalent | What It Does |
|---|---|---|
| `constants.py` | `CONTROLLED_CONSTANTS.agc` + `ERASABLE_ASSIGNMENTS.agc` | Physical constants with AGC source annotations |
| `guidance.py` | `LUNAR_LANDING_GUIDANCE_EQUATIONS.agc` (RODCOMP) | Throttle control + attitude DAP |
| `physics.py` | `SERVICER.agc` (state integration) | RK4 integrator, altitude/velocity computation |
| `transforms.py` | IMU → inertial transform | Body-to-world rotation |
| `autopilot.py` | Crew procedures (not AGC code) | Simulated CDR with PD controller |
| `apollo_lander_env.py` | Complete avionics stack | Ties everything together as Gymnasium env |

### Verification

Our implementation achieves **100% landing success** across 20-episode evaluations
(20/20), with touchdown velocities consistent with Apollo 11's actual performance:

| Metric | Our Simulation | Apollo 11 Actual |
|---|---|---|
| Vertical velocity at landing | ~0.30 m/s | ~0.9 m/s (3 ft/s) |
| Horizontal velocity at landing | 0.03–0.32 m/s | ~0.2 m/s |
| Descent time (P66 phase) | ~120–180 s | ~130 s |

---

## Glossary

| Term | Meaning |
|---|---|
| **AGC** | Apollo Guidance Computer |
| **CDR** | Commander (flew the LM during landing) |
| **DAP** | Digital Autopilot (attitude control software) |
| **DSKY** | Display and Keyboard (AGC user interface) |
| **DPS** | Descent Propulsion System (main landing engine) |
| **HDOTDISP** | Displayed vertical velocity (navigation output) |
| **IMU** | Inertial Measurement Unit (gyros + accelerometers) |
| **LM** | Lunar Module |
| **LPD/LPDT** | Landing Point Designator / LPD cross-pointer display |
| **LR** | Landing Radar |
| **MCI** | Moon-Centered Inertial (coordinate frame) |
| **P63/P64/P66** | AGC Programs for braking/approach/manual landing |
| **PDI** | Powered Descent Initiation |
| **RCS** | Reaction Control System (attitude thrusters) |
| **RHC** | Rotational Hand Controller (attitude joystick) |
| **ROD** | Rate of Descent (toggle switch) |
| **RODCOMP** | ROD Computer — AGC throttle control algorithm |
| **SERVICER** | AGC navigation task (state integration + LR fusion) |
| **TAUROD** | Time constant for RODCOMP proportional law (2.0 s) |
| **VDGVERT** | Desired vertical velocity (AGC erasable variable) |

---

*Document generated from analysis of the [Apollo 11 Luminary099 AGC source code](https://github.com/chrislgarry/Apollo-11/tree/master/Luminary099) and our faithful Python reimplementation.*
