/*
    START OF FILE: src/core/presets.rs
*/
/// Initial morphology pool for Generation 0.
///
/// Every entry is a credible legged walker: capsules to avoid snagging,
/// touch sensors (`St`) on every foot (required for `FootAirTimeFactor` and
/// `ComBounceReward` to score real step cycles), one IMU on the body
/// (`Si`), and bounded recursion. Hinge limits keep joints from folding
/// onto themselves; motor effort is sized to lift body weight.
///
/// The five entries cover a deliberate range of body plans:
///   1. Millipede   — long redundant body, many short legs (proven champion).
///   2. Quadruped   — sphere body, four legs with stochastic leg-segment count.
///   3. Hexapod     — sphere body, six deterministic hip+knee legs at 60°.
///   4. Insect      — elongated rigid spine, three deterministic leg pairs.
///   5. Crawler     — low-CoG sphere body, four short hip+knee legs.
///
/// Removed in this overhaul: Sidewinder (open-ended spine, unwalkable),
/// Biped (no foot sensors, 2-leg balance too hard), Starfish (5-fold
/// symmetric, no forward axis), Inchworm (prismatic demo, flat slide
/// gated by `ComBounceReward`), Sentinel (stationary by design).
pub const MORPHOLOGY_POOL: &[&str] = &[
    // ── 1. Millipede ───────────────────────────────────────────────────
    // The current champion. High redundancy, low center of gravity,
    // capsules everywhere. Stochastic recursion (p1/p2/p3/p4 each 0.5)
    // produces a chain of 4-6 body segments with paired side legs.
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
p4 : LP(ha) -> [ Jf  &(90) C(1.0) ^(90) -(90) /(90) Jl(leg_min, leg_max, leg_eff, leg_vel) J K(leglength, legradius) K(leglength, legradius) St][ Jf  \\(180) &(90) C(1.0) ^(90) +(90) /(90) Jl(leg_min, leg_max, leg_eff, leg_vel) J K(leglength, legradius) K(leglength, legradius) St]",

    // ── 2. Quadruped ───────────────────────────────────────────────────
    // Sphere body, four legs at 90° spacing. Stochastic leg-segment
    // count (1/2/3 segments) gives intra-morphology variation while
    // every leg ends with a touch sensor on the foot.
    "\
#define hip_min -0.6
#define hip_max 0.6
#define hip_eff 3000
#define hip_vel 5
#define knee_min -0.8
#define knee_max 0.8
#define knee_eff 3000
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

    // ── 3. Hexapod (Sprawler v2) ───────────────────────────────────────
    // Six deterministic legs at 60° spacing around the sphere body,
    // each leg = hip stub + thigh + shin + touch foot. Anti-stochastic
    // rewrite of the old Hexapod (which mixed 1- and 2-segment legs
    // per draw and starved NEAT of consistent topology).
    "\
#define hip_min -0.6
#define hip_max 0.6
#define hip_eff 3000
#define hip_vel 5
#define knee_min -0.8
#define knee_max 0.8
#define knee_eff 3000
#define knee_vel 5
#define lw 0.18
#define seg 0.55
#define la 45
#define lb 90
omega: !(0.5) O Si !(lw) [ &(lb) M ^(la) L ] [ \\(60) &(lb) M ^(la) L ] [ \\(120) &(lb) M ^(la) L ] [ \\(180) &(lb) M ^(la) L ] [ /(120) &(lb) M ^(la) L ] [ /(60) &(lb) M ^(la) L ]
M -> /(90) Jl(hip_min, hip_max, hip_eff, hip_vel) J C(0.35) \\(90)
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg) St",

    // ── 4. Insect (Tripod Walker) ──────────────────────────────────────
    // Three-segment horizontal body with three deterministic leg pairs
    // along its length (6 legs total). Body axis is set horizontal by
    // `+(90)` in omega (same trick the Millipede uses); the LP rule
    // uses the Millipede's hip-rotation incantation so legs splay
    // perpendicular to the body and bend downward. Each leg has a
    // knee (hip + thigh + shin) for a 2-DOF substrate — meaningfully
    // distinct from Millipede's fixed-stack 1-DOF legs.
    "\
#define rad 0.3
#define seg 0.7
#define hip 0.3
#define lw 0.18
#define leg 0.45
#define hipmin -0.6
#define hipmax 0.6
#define knee_min -0.7
#define knee_max 0.3
#define eff 3000
#define vel 5
omega: O(rad) Si +(90) LP Jl(hipmin, hipmax, eff, vel) J K(seg, rad) LP Jl(hipmin, hipmax, eff, vel) J K(seg, rad) LP
LP -> [ Jf &(90) C(hip) ^(90) -(90) /(90) Jl(hipmin, hipmax, eff, vel) J K(leg, lw) Jl(knee_min, knee_max, eff, vel) J K(leg, lw) St ] [ Jf \\(180) &(90) C(hip) ^(90) +(90) /(90) Jl(hipmin, hipmax, eff, vel) J K(leg, lw) Jl(knee_min, knee_max, eff, vel) J K(leg, lw) St ]",

    // ── 5. Crawler (Low Rider) ─────────────────────────────────────────
    // Quadruped body plan with short legs and a small body — the
    // lowest center-of-gravity walker in the pool. Knees bend mostly
    // inward (knee_max small, knee_min large negative) so the foot
    // settles under the hip. Easy for NEAT to evolve because wide
    // stance + low CoG keeps the body stable through early random
    // gaits.
    "\
#define hip_min -0.5
#define hip_max 0.5
#define hip_eff 3000
#define hip_vel 5
#define knee_min -1.0
#define knee_max 0.2
#define keff 3000
#define kvel 5
#define lw 0.2
#define seg 0.4
#define la 55
#define lb 90
omega: !(0.45) O Si !(lw) [ &(lb) M ^(la) L ] [ \\(90) &(lb) M ^(la) L ] [ \\(180) &(lb) M ^(la) L ] [ /(90) &(lb) M ^(la) L ]
M -> /(90) Jl(hip_min, hip_max, hip_eff, hip_vel) J C(0.3) \\(90)
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St",
];
