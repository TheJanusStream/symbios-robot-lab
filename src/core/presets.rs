/*
    START OF FILE: src/core/presets.rs
*/
/// Initial morphology pool for Generation 0.
///
/// Each agent is assigned a random morphology from this pool, providing
/// structural diversity for evolution to explore. The NEAT brain is sized
/// dynamically to match each morphology's sensor/joint topology.
///
/// Key Improvements (v2):
///   - Capsules (K) everywhere to prevent physics snagging.
///   - High Torque (4000+) to lift body weight.
///   - Tighter joint limits (+/- 0.6) to prevent self-collision folding.
///   - Alternating Axes (Pitch/Yaw) for snake/worm topologies.
pub const MORPHOLOGY_POOL: &[&str] = &[
    // ── The Dominator (Millipede) ──────────────────────────────────────
    // The current champion. High redundancy, low center of gravity, capsules.
    "\
#define headradius 0.6
#define bodylength 1.6
#define bodyradius 0.4
#define bodylengthfactor 1.0
#define bodyradiusfactor 1.0
#define hipangle 90
#define hipanglefactor 1.0
#define hip_min -0.57
#define hip_max 0.57
#define hip_eff 5000
#define hip_vel 5
#define leg_min -0.57
#define leg_max 0.57
#define leg_eff 5000
#define leg_vel 5
#define leglength 0.5
#define legradius 0.3
#define p1 0.5
#define p2 0.5
#define p3 0.5
#define p4 0.5
omega: O(headradius) Si +(90) LP(hipangle) Jl(hip_min, hip_max, hip_eff, hip_vel) J M(bodylength, bodyradius, hipangle*hipanglefactor)
p1 : M(bl, br, ha) -> K(bl, br) LP(ha * hipanglefactor) M(bl * bodylengthfactor, br * bodyradiusfactor, ha * hipanglefactor)
p2 : M(bl, br, ha) -> M(bl, br, ha)
p3 : LP(ha) -> LP(ha)
p4 : LP(ha) -> [ Jf  &(90) C(1.0) ^(90) -(90) Jl(leg_min, leg_max, leg_eff, leg_vel) J K(leglength, legradius) K(leglength, legradius) St][ Jf  \\(180) &(90) C(1.0) ^(90) +(90) Jl(leg_min, leg_max, leg_eff, leg_vel) J K(leglength, legradius) K(leglength, legradius) St]",

    // ── Quadruped (Upgraded) ───────────────────────────────────────────
    // 4 legs. Upgraded to Capsules (K) and higher torque.
    "\
#define hip_min -0.6
#define hip_max 0.6
#define hip_eff 5000
#define hip_vel 5
#define knee_min -0.8
#define knee_max 0.8
#define knee_eff 5000
#define knee_vel 5
#define lw 0.2
#define seg 1.2
#define la 45
#define lb 90
omega: !(0.7) O Si !(0.3) [ &(lb) M ^(la) L] [ \\(90) &(lb) M ^(la) L] [ \\(180) &(lb) M ^(la) L] [ /(90) &(lb) M ^(la) L]
M -> /(90) Jl(hip_min, hip_max, hip_eff, hip_vel) J C(0.5) \\(90)
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) St
0.4 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/2.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/2.0) St
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) St",

    // ── Hexapod (Sprawler) ─────────────────────────────────────────────
    // 6 legs. Wide stance ("Sprawler") to prevent rolling.
    "\
#define jmin -0.6
#define jmax 0.6
#define eff 4000
#define vel 6
#define lw 0.15
#define seg 1.0
#define spread 70
omega: !(0.6) O Si [ &(spread) L ] [ \\(60) &(spread) L ] [ \\(120) &(spread) L ] [ \\(180) &(spread) L ] [ /(120) &(spread) L ] [ /(60) &(spread) L ]
0.3 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg) St
0.7 : L -> Jl(jmin, jmax, eff, vel) J !(lw) K(seg*0.6) Jl(jmin, jmax, eff, vel) J K(seg*0.6) St",

    // ── Sidewinder (Snake V2) ──────────────────────────────────────────
    // IMPROVEMENT: Alternating Pitch (&) and Yaw (+) joints.
    // A 1D hinge chain can only flop. A 2D hinge chain (Pitch+Yaw) can slither.
    // Added 'Ribs' (Fixed spurs) to provide friction anisotropy.
    "\
#define jmin -0.8
#define jmax 0.8
#define eff 4000
#define vel 8
#define seg 0.5
#define rad 0.35
omega: !(rad) O Si St Spine
// Recursive spine with alternating axes
Spine -> Jl(jmin, jmax, eff, vel) J K(seg) Ribs +(90) Jl(jmin, jmax, eff, vel) J K(seg) Ribs -(90) Spine
// Spurs to catch the ground
Ribs -> [ Jf +(90) K(0.2, 0.1) ] [ Jf -(90) K(0.2, 0.1) ]",

    // ── Biped (Snowshoe) ───────────────────────────────────────────────
    // IMPROVEMENT: Massive feet (Foot) to lower the balance difficulty.
    // High torque (6000) to lift the heavy upper body.
    "\
#define hip_min -0.8
#define hip_max 0.8
#define eff 6000
#define vel 4
#define knee_min -1.0
#define knee_max 0.2
#define keff 6000
#define kvel 4
#define lw 0.25
#define seg 0.7
// Wide hips (45 deg spread)
omega: !(0.6) O Si [ &(45) Leg ] [ ^(45) Leg ]
Leg -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) Foot
// Large flat foot for stability
Foot -> Jf !(0.1) [ +(90) K(0.6) ] [ -(90) K(0.6) ] St",

    // ── Starfish (Crawler) ─────────────────────────────────────────────
    // 5 arms with 'Elbows' to allow crawling motion (pulling) rather than just flapping.
    "\
#define jmin -0.7
#define jmax 0.7
#define eff 4000
#define vel 5
#define seg 0.6
omega: !(0.8) O Si [ &(90) Arm ] [ \\(72) &(90) Arm ] [ \\(144) &(90) Arm ] [ /(144) &(90) Arm ] [ /(72) &(90) Arm ]
// Two-segment arm allows 'pulling' the body forward
Arm -> Jl(jmin, jmax, eff, vel) J K(seg) Jl(jmin, jmax, eff, vel) J K(seg) St",
];