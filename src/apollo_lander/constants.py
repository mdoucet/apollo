"""
Physical constants for the Apollo Lunar Module descent simulation.

All values are in SI units (meters, seconds, kilograms, Newtons).
Constants are derived from Apollo program documentation and the
Luminary099 AGC source code (Apollo 11 flight software).

AGC Source Repository:
    github.com/chrislgarry/Apollo-11/tree/master/Luminary099

Key AGC Source Files Referenced:
    CONTROLLED_CONSTANTS.agc        — Engine params, masses, mu, radii  (pp. 38-53)
    THROTTLE_CONTROL_ROUTINES.agc   — Throttle limits FMAXPOS/FMAXODD  (pp. 793-797)
    LUNAR_LANDING_GUIDANCE_EQUATIONS.agc — HIGHESTF, GSCALE, P66 logic  (pp. 798-828)
    SERVICER.agc                    — DPSVEX/APSVEX, mass computation   (pp. 857-897)

AGC Fixed-Point Notation:
    ``2DEC X B-N`` stores the decimal value X with scale factor 2^(-N).
    ``DEC X``      stores a single-precision decimal value.
    AGC time unit: centiseconds (cs);  1 cs  = 0.01 s.
    AGC velocity unit: m/cs;           1 m/cs = 100 m/s.
"""

# =============================================================================
# Celestial Body Constants
# =============================================================================

MU_MOON = 4.9048695e12
"""Moon's standard gravitational parameter (m^3/s^2).

AGC: MUM  2DEC* 4.9027780 E8 B-30*       [CONTROLLED_CONSTANTS.agc p.45]
     = 4.9027780e8 m^3/cs^2 = 4.9027780e12 m^3/s^2
     Modern value 4.9048695e12 used here (0.04% larger than AGC).
"""

R_MOON = 1737400.0
"""Mean radius of the Moon (m).

AGC: RSUBM  2DEC 1738090 B-29            [CONTROLLED_CONSTANTS.agc p.45]
     504RM  2DEC 1738090 B-29  # METERS B-29 (EQUATORIAL MOON RADIUS)
     = 1,738,090 m equatorial radius.
     Modern mean radius 1,737,400 m used here (690 m smaller than AGC).
"""

G0 = 9.80665
"""Standard gravitational acceleration at Earth's surface (m/s^2).
Used for specific impulse calculations (Vex = ISP * g0).
Not stored directly in AGC; implied in DPSVEX computation.
"""

# =============================================================================
# Descent Propulsion System (DPS)
# =============================================================================

DPS_ISP = 311.0
"""Specific impulse of the Descent Propulsion System (s).

AGC: DPSVEX  DEC* -2.95588868 E+1 B-05*  [CONTROLLED_CONSTANTS.agc p.39]
     Comment: "VE (DPS) +2.95588868E+3"
     = -29.5589 m/cs exhaust velocity = 2955.89 m/s
     ISP_from_Vex = 2955.89 / 9.80665 = 301.4 s

     MDOTDPS  2DEC 0.1480 B-3             [CONTROLLED_CONSTANTS.agc p.38]
     Comment: "32.62 LBS/SEC IN KGS/CS"
     = 0.1480 kg/cs = 14.80 kg/s mass flow rate

     Note: Vacuum ISP = 311 s from NASA Apollo Operations Handbook
     (LMA790-3-LM). AGC's DPSVEX (equivalent to 301.4 s) is the
     effective exhaust velocity used for onboard mass computation,
     which differs from the thermodynamic vacuum ISP.
"""

MAX_THRUST = 45040.0
"""Maximum DPS thrust (N). Nominal full throttle.

AGC: FDPS  2DEC 4.3670 B-7               [CONTROLLED_CONSTANTS.agc p.38]
     Comment: "9817.5 LBS FORCE IN NEWTONS"
     = 43,670 N (9817.5 lbf) — nominal DPS force for guidance calcs

     FMAXPOS  DEC +3467                   [THROTTLE_CONTROL_ROUTINES.agc p.797]
     Comment: "FMAX +4.34546769 E+4"
     = 43,455 N — upper boundary of throttleable range (~94%)

     FMAXODD  DEC +3841                   [THROTTLE_CONTROL_ROUTINES.agc p.797]
     Comment: "FSAT +4.81454413 E+4"
     = 48,145 N — absolute max thrust ("flat out" / BIT13)

     HIGHESTF  2DEC 4.34546769 B-12       [LUNAR_LANDING_GUIDANCE_EQUATIONS.agc p.827]
     = Max acceleration (FMAX/mass) used in guidance limiting.

     Note: 45,040 N (10,125 lbf) is the commonly cited nominal max
     from TRW DPS specifications. The AGC had three distinct levels:
     FDPS (nominal), FMAXPOS (throttleable ceiling), FMAXODD (flat-out).
     The DPS had a non-throttleable zone between ~57% and ~94%.
"""

MIN_THRUST = 4504.0
"""Minimum DPS thrust (N). Throttle floor at ~10%.

AGC: MAXFORCE / MINFORCE are referenced in P66 guidance code
     (LUNAR_LANDING_GUIDANCE_EQUATIONS.agc p.819) but stored in
     erasable memory — pad-loaded for each mission, not in fixed ROM.

     The DPS pintle injector could not maintain stable combustion
     below ~10% thrust. Minimum = 10% x 45,040 N = 4,504 N.
     Some sources cite 1,050 lbf (4,671 N) as the TRW spec floor.
"""

# =============================================================================
# Lunar Module Mass Properties
# =============================================================================

LM_DRY_MASS = 2034.0
"""LM descent stage dry mass (kg), excluding fuel.

AGC: LODESCNT  DEC 1750 B-16             [CONTROLLED_CONSTANTS.agc p.43]
     = 1,750 kg — minimum descent stage mass (structural only)
     MINLMD  DEC -2850 B-16
     = 2,850 kg — minimum descent stage mass (with residual propellant)

     Note: 2,034 kg from Apollo 11 Mission Report (NASA SP-238).
     LODESCNT is the structural minimum; MINLMD includes residual
     propellant margin required for abort capability.
"""

LM_ASCENT_MASS = 4670.0
"""LM ascent stage mass (kg). Treated as fixed payload during descent.

AGC: FULLAPS  DEC 5050 B-16              [CONTROLLED_CONSTANTS.agc p.43]
     = 5,050 kg — nominal full ascent stage mass (crew + consumables)
     MINMINLM  DEC -2200 B-16
     = 2,200 kg — minimum ascent stage mass (dry)

     Note: 4,670 kg from Apollo 11 LM-5 specifications.
     FULLAPS (5,050 kg) is the nominal fully loaded ascent stage.
"""

LM_FUEL_MASS = 8200.0
"""DPS propellant mass at Powered Descent Initiation (kg).

AGC: HIDESCNT  DEC 15300 B-16            [CONTROLLED_CONSTANTS.agc p.43]
     = 15,300 kg — maximum descent LM mass (ascent + descent + fuel)
     Propellant = HIDESCNT - LODESCNT - ascent ~= 15300 - 1750 - 5050 = 8500 kg

     Note: ~18,000 lbs (8,165 kg) usable propellant from LM-5 specs.
     8,200 kg is the loaded propellant mass including reserves.
"""

LM_TOTAL_MASS = LM_DRY_MASS + LM_ASCENT_MASS + LM_FUEL_MASS
"""Total LM mass at start of powered descent (kg).

AGC: HIDESCNT = 15,300 kg (max descent LM mass for DAP config).
     Our computed total: 2034 + 4670 + 8200 = 14,904 kg.
"""

# =============================================================================
# Timing
# =============================================================================

GUIDANCE_DT = 2.0
"""AGC guidance loop period (s). The guidance equations recalculated
the target thrust vector every 2 seconds.

AGC: 2SECS  DEC 200  (centiseconds)      [SERVICER.agc p.883]
     4SECS  DEC 400  (2 x guidance period, used in P66 FWEIGHT calc)
     The SERVICER READACCS task ran every 2 seconds, triggering the
     guidance cycle (LUNLAND → guidance equations → THROTTLE).
"""

DAP_DT = 0.1
"""Digital Autopilot cycle time (s). The DAP ran at 10 Hz,
handling attitude control and thrust commands.

AGC: The T6 RUPT (DAP interrupt) fired at ~100 ms intervals.
     READACCS synchronization: OCT37771 offset vs TIME5
     (SERVICER.agc p.858) to avoid conflicts with R10/R11 display
     interrupts.
"""

# =============================================================================
# P66 Manual Mode Constants
# =============================================================================

ROD_INCREMENT = 0.3048
"""Rate of Descent switch increment (m/s per click).
Each click of the ROD switch adjusted the target sink rate by 1 ft/s.

AGC: RODSCAL1 — loaded into erasable at P66 init, converts raw
     switch counts to velocity changes. Each ROD discrete (Channel 16
     bit 6 or 7) adds or subtracts one count from RODCOUNT.
     Source: LUNAR_LANDING_GUIDANCE_EQUATIONS.agc pp.800,816-818,822
     Code:   RODCOMP:  XCH RODCOUNT / MP RODSCAL1 / DAS VDGVERT
     Related: BIASFACT  2DEC 655.36 B-28  (P66 IMU bias factor, p.820)

     1 click = 1 ft/s = 0.3048 m/s (standard Apollo ROD increment).
"""

MAX_ROTATION_RATE = 0.3490659  # radians, ~20 deg/s
"""Maximum rotation rate commandable via the Rotational Hand Controller (rad/s).

AGC: RHC produced proportional rate commands scaled by the DAP.
     The max rate depended on DAP configuration (typically 20 deg/s).
     AZEACH  DEC .03491  (2 degrees, redesignation increment, p.822)
     ELEACH  DEC .00873  (0.5 degree, redesignation increment, p.822)
"""

# =============================================================================
# Initial Conditions for P66 Phase
# =============================================================================

P66_INITIAL_ALTITUDE = 150.0
"""Approximate altitude above landing site when P66 engages (m).

AGC: P66 was selected by the astronaut via the auto-mode monitor
     (GUILDENSTERN, LUNAR_LANDING_GUIDANCE_EQUATIONS.agc p.800).
     The commander switched from P64 (approach phase) to P66 when
     the LPD angle reached zero, typically at ~500 ft (152 m).
"""

P66_INITIAL_DESCENT_RATE = -1.524
"""Initial target descent rate at P66 start (m/s). ~5 ft/s downward.

AGC: VDGVERT initialized from current rate at P66 switchover:
     STARTP66:  DCA HDOTDISP / DXCH VDGVERT  (p.800)
     "SET DESIRED ALTITUDE RATE = CURRENT ALTITUDE RATE"
     Typical value at engagement: ~5 ft/s (1.524 m/s) downward.
"""

P66_INITIAL_HORIZONTAL_VEL = 5.0
"""Approximate residual horizontal velocity at P66 start (m/s)."""

# =============================================================================
# Landing Safety Criteria
# =============================================================================

MAX_LANDING_VERTICAL_VEL = 1.2
"""Maximum vertical velocity for a safe landing (m/s).

LM landing gear designed for up to 7 ft/s (2.1 m/s) vertical impact.
Apollo 11 actual touchdown: ~0.9 m/s (3 ft/s). 1.2 m/s provides
operational margin.
"""

MAX_LANDING_HORIZONTAL_VEL = 0.6
"""Maximum horizontal velocity for a safe landing (m/s)."""
