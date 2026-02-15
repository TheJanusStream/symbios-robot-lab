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
#define hip_min -0.57
#define hip_max 0.57
#define hip_eff 5000
#define hip_vel 5
#define hiph_min -0.57
#define hiph_max 0.57
#define hiph_eff 5000
#define hiph_vel 5
#define knee_min -0.57
#define knee_max 0.57
#define knee_eff 30000
#define knee_vel 5
#define lw 0.15
#define seg 1.5
#define la 45
#define lb 90
#define ll 0.5
omega: !(0.6) O Si !(0.2) [ &(lb) M ^(la) L] [ \\(90) &(lb) M ^(la) L] [ \\(180) &(lb) M ^(la) L] [ /(90) &(lb) M ^(la) L]
M -> /(90) Jl(hiph_min, hiph_max, hiph_eff, hiph_vel) J C(ll) \\(90)
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) St
0.4 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/2.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/2.0) St
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) St",
    // ── Centipede ──────────────────────────────────────────────────────
    // Recursive spine: each derivation adds a segment, with 0-2 lateral legs.
    "\
#define jmin -0.57
#define jmax 0.57
#define eff 4000
#define vel 5
#define lw 0.12
#define seg 1.0
#define ll 0.5
#define la 90
#define lb 80
omega: !(0.4) O Si L
0.3 : L -> Jl(jmin, jmax, eff, vel) J K(seg) S
0.15 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ /(90) &(la) C(ll) ^(lb) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L
0.15 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ \\(90) &(la) C(ll) ^(lb) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L
0.4 : L -> Jl(jmin, jmax, eff, vel) J K(seg) [ \\(90) &(la) C(ll) ^(lb) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] [ /(90) &(la) C(ll) ^(lb) Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St ] L",
    // ── Hexapod ────────────────────────────────────────────────────────
    // 6 legs at 60° intervals, 1-3 DOF per leg.
    "\
#define jmin -0.57
#define jmax 0.57
#define eff 4000
#define vel 5
#define lw 0.12
#define seg 1.0
#define ll 0.5
#define la 90
#define lb 80
omega: !(0.5) O Si [ &(la) C(ll) ^(lb) L ] [ \\(60) &(la) C(ll) ^(lb) L ] [ \\(120) &(la) C(ll) ^(lb) L ] [ \\(180) &(la) C(ll) ^(lb) L ] [ /(120) &(la) C(ll) ^(lb) L ] [ /(60) &(la) C(ll) ^(lb) L ]
0.3 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St
0.4 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg/2.0) Jl(jmin, jmax, eff, vel) J K(seg/2.0) St
0.3 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg/3.0) Jl(jmin, jmax, eff, vel) J K(seg/3.0) Jl(jmin, jmax, eff, vel) J K(seg/3.0) St",
    // ── Snake ──────────────────────────────────────────────────────────
    // Legless chain of segments. Relies on body undulation for locomotion.
    // 3 derivations → 4 modules, 3 joints, 1 IMU, 4 touch.
    "\
#define jmin -0.57
#define jmax 0.57
#define eff 3000
#define vel 8
#define seg 0.35
omega: !(0.3) O Si St L
L -> Jl(jmin, jmax, eff, vel) J K(seg) St A",
    // ── Biped ──────────────────────────────────────────────────────────
    // 2 legs with hip + knee, tall body. Challenging balance problem.
    // 5 modules, 4 joints, 1 IMU, 2 touch.
    "\
#define hip_min -0.57
#define hip_max 0.57
#define eff 6000
#define vel 5
#define knee_min -0.57
#define knee_max 0.57
#define keff 4000
#define kvel 5
#define lw 0.18
#define seg 0.6
omega: !(0.5) O Si [ &(45) L ] [ ^(45) L ]
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St",
    // ── Starfish ───────────────────────────────────────────────────────
    // 5 single-segment arms at 72° intervals. Radial symmetry.
    "\
#define jmin -0.57
#define jmax 0.57
#define eff 3500
#define vel 5
#define seg 0.5
#define la 60
omega: !(0.5) O Si [ &(la) L  St] [ \\(72) &(la) L  St] [ \\(144) &(la) L  St] [ /(144) &(la) L  St] [ /(72) &(la) L  St]
0.1 : L -> Jl(jmin, jmax, eff, vel) J K(seg) L
0.9 : L -> Jl(jmin, jmax, eff, vel) J K(seg)",
];
