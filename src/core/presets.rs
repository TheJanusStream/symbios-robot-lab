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
    // 4 legs at 90° intervals, 1-3 DOF per leg.
    "\
#define hip_min -1.57
#define hip_max 1.57
#define hip_eff 500
#define hip_vel 5
#define knee_min -2.0
#define knee_max 2.0
#define knee_eff 300
#define knee_vel 5
#define lw 0.15
#define seg 1.5
omega: !(0.6) O Si !(0.2) [ &(90)  C(0.5) ^(60) L] [ \\(90) &(90) C(0.5) ^(60) L] [ \\(180) &(90) C(0.5) ^(60) L] [ /(90) &(90) C(0.5) ^(60) L]
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) St
0.4 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/2.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/2.0) St
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) St",
    // ── Centipede ──────────────────────────────────────────────────────
    // Recursive spine: each derivation adds a segment, with 0-2 lateral legs.
    "\
#define jmin -1.2
#define jmax 1.2
#define eff 400
#define vel 5
#define lw 0.12
#define seg 1.0
omega: !(0.4) O Si L
0.3 : L -> Jl(jmin, jmax, eff, vel) J K(seg) S
0.15 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ /(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L
0.15 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ \\(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L
0.4 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ \\(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] [ /(90) &(90) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L",
    // ── Hexapod ────────────────────────────────────────────────────────
    // 6 legs at 60° intervals, 1-3 DOF per leg.
    "\
#define jmin -1.57
#define jmax 1.57
#define eff 400
#define vel 5
#define lw 0.12
#define seg 1.0
omega: !(0.5) O Si [ &(90) L ] [ \\(60) &(90) L ] [ \\(120) &(90) L ] [ |(1) &(90) L ] [ /(120) &(90) L ] [ /(60) &(90) L ]
0.3 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St
0.4 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg/2.0) Jl(jmin, jmax, eff, vel) J K(seg/2.0) St
0.3 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg/3.0) Jl(jmin, jmax, eff, vel) J K(seg/3.0) Jl(jmin, jmax, eff, vel) J K(seg/3.0) St",
    // ── Snake ──────────────────────────────────────────────────────────
    // Legless chain of segments. Relies on body undulation for locomotion.
    // 3 derivations → 4 modules, 3 joints, 1 IMU, 4 touch.
    "\
#define jmin -1.8
#define jmax 1.8
#define eff 300
#define vel 8
#define seg 0.35
omega: !(0.3) O Si St L
L -> Jl(jmin, jmax, eff, vel) J K(seg) St A",
    // ── Biped ──────────────────────────────────────────────────────────
    // 2 legs with hip + knee, tall body. Challenging balance problem.
    // 5 modules, 4 joints, 1 IMU, 2 touch.
    "\
#define hip_min -1.57
#define hip_max 1.57
#define eff 600
#define vel 5
#define knee_min -2.0
#define knee_max 2.0
#define keff 400
#define kvel 5
#define lw 0.18
#define seg 0.6
omega: !(0.5) O Si [ &(45) L ] [ ^(45) L ]
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St",
    // ── Starfish ───────────────────────────────────────────────────────
    // 5 single-segment arms at 72° intervals. Radial symmetry.
    "\
#define jmin -2.0
#define jmax 2.0
#define eff 350
#define vel 5
#define seg 0.5
omega: !(0.5) O Si [ &(60) L  St] [ \\(72) &(60) L  St] [ \\(144) &(60) L  St] [ /(144) &(60) L  St] [ /(72) &(60) L  St]
0.1 : L -> Jl(jmin, jmax, eff, vel) J K(seg) L
0.9 : L -> Jl(jmin, jmax, eff, vel) J K(seg)",
];
