/// Initial morphology pool for Generation 0.
///
/// Each agent is assigned a random morphology from this pool, providing
/// structural diversity for evolution to explore. The NEAT brain is sized
/// dynamically to match each morphology's sensor/joint topology.
///
/// All morphologies share the same L-System conventions:
///   - `#define` constants are the primary targets of parametric mutation
///   - Production rules define repeating structures (legs, segments)
///   - `Si` = IMU sensor, `St` = touch sensor, `Jl(...)` = limited hinge joint
pub const MORPHOLOGY_POOL: &[&str] = &[
    // ── Quadruped ──────────────────────────────────────────────────────
    // 4 legs at 90° intervals, 2 DOF per leg (hip + knee).
    // 9 modules, 8 joints, 1 IMU, 4 touch.
    "\
#define hip_min -1.57
#define hip_max 1.57
#define hip_eff 50
#define hip_vel 5
#define knee_min -2.0
#define knee_max 0.2
#define knee_eff 30
#define knee_vel 5
#define lw 0.15
#define seg 0.5
omega: !(0.6) K(0.2, 0.3) Si [ &(90) L ] [ \\(90) &(90) L ] [ |(1) &(90) L ] [ /(90) &(90) L ]
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) B(seg) Jl(knee_min, knee_max, knee_eff, knee_vel) J B(seg) St",
    // ── Centipede ──────────────────────────────────────────────────────
    // Recursive spine: each derivation adds a segment with 2 lateral legs.
    // 3 derivations → 3 spine segments + 6 legs. 10 modules, 9 joints.
    "\
#define jmin -1.2
#define jmax 1.2
#define eff 40
#define vel 5
#define lw 0.12
#define seg 0.4
omega: !(0.4) K(0.15, 0.25) Si S
S -> Jl(jmin, jmax, eff, vel) J B(seg) [ \\(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) B(seg) St ] [ /(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) B(seg) St ] S",
    // ── Hexapod ────────────────────────────────────────────────────────
    // 6 legs at 60° intervals, 2 DOF per leg.
    // 13 modules, 12 joints, 1 IMU, 6 touch.
    "\
#define jmin -1.57
#define jmax 1.57
#define eff 40
#define vel 5
#define lw 0.12
#define seg 0.4
omega: !(0.5) K(0.2, 0.35) Si [ &(90) L ] [ \\(60) &(90) L ] [ \\(120) &(90) L ] [ |(1) &(90) L ] [ /(120) &(90) L ] [ /(60) &(90) L ]
L -> Jl(jmin, jmax, eff, vel) J !(lw) B(seg) Jl(jmin, jmax, eff, vel) J B(seg) St",
    // ── Snake ──────────────────────────────────────────────────────────
    // Legless chain of segments. Relies on body undulation for locomotion.
    // 3 derivations → 4 modules, 3 joints, 1 IMU, 4 touch.
    "\
#define jmin -0.8
#define jmax 0.8
#define eff 30
#define vel 8
#define seg 0.35
omega: !(0.3) K(0.15, 0.2) Si St S
S -> Jl(jmin, jmax, eff, vel) J B(seg) St S",
    // ── Biped ──────────────────────────────────────────────────────────
    // 2 legs with hip + knee, tall body. Challenging balance problem.
    // 5 modules, 4 joints, 1 IMU, 2 touch.
    "\
#define hip_min -1.57
#define hip_max 1.57
#define eff 60
#define vel 5
#define knee_min -2.0
#define knee_max 0.2
#define keff 40
#define kvel 5
#define lw 0.18
#define seg 0.6
omega: !(0.5) K(0.25, 0.4) Si [ &(90) L ] [ |(1) &(90) L ]
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) B(seg) Jl(knee_min, knee_max, keff, kvel) J B(seg) St",
    // ── Starfish ───────────────────────────────────────────────────────
    // 5 single-segment arms at 72° intervals. Radial symmetry.
    // 6 modules, 5 joints, 1 IMU, 5 touch.
    "\
#define jmin -1.0
#define jmax 1.0
#define eff 35
#define vel 5
#define seg 0.5
omega: !(0.5) K(0.2, 0.2) Si [ &(90) A ] [ \\(72) &(90) A ] [ \\(144) &(90) A ] [ /(144) &(90) A ] [ /(72) &(90) A ]
A -> Jl(jmin, jmax, eff, vel) J B(seg) St",
];
