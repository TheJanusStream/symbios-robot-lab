/// Initial morphology pool for Generation 0.
///
/// Every entry is a credible legged walker: capsules to avoid snagging,
/// touch sensors (`St`) on every foot (required for `FootAirTimeFactor` and
/// `ComBounceReward` to score real step cycles), one IMU on the body
/// (`Si`), and bounded recursion. Hinge limits keep joints from folding
/// onto themselves; motor effort is sized to lift body weight.
///
/// The eight entries cover a deliberate range of body plans:
///   1. Millipede - long redundant body, many short legs (proven champion).
///   2. Quadruped - sphere body, four legs with stochastic leg-segment count.
///   3. Hexapod - sphere body, six deterministic hip+knee legs at 60°.
///   4. Insect - elongated rigid spine, three deterministic leg pairs.
///   5. Crawler - low-CoG sphere body, four short hip+knee legs.
///   6. Tripod - three legs whose shins slide (prismatic joints).
///   7. Strider - quadruped on flat feet with limited ball-joint ankles.
///   8. Scout - the Crawler with an ultrasonic eye on a neck, looking at the
///      ground ahead.
///
/// The last three bring in what the lab's other joints and sensors can do,
/// so evolution meets sliders, ball joints and distance sensing from the
/// start rather than only by mutation.
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

    // ── 6. Tripod ──────────────────────────────────────────────────────
    // Three legs at 120°, each a hip, a thigh and a shin that slides along
    // the leg (Ja(0, 1, 0) turns the joint axis to the leg's growth
    // direction before Jp makes it prismatic), so a leg can push off the
    // ground as well as swing. 300 N per slider lifts the ~22 kg robot at
    // about four times its weight with all three legs (the lab caps any one
    // slider at 5000 N, 23 times the robot's weight).
    "\
#define hip_min -0.5
#define hip_max 0.5
#define eff 3000
#define vel 5
#define slide_min -0.15
#define slide_max 0.15
#define slide_eff 300
#define lw 0.2
#define seg 0.55
#define la 45
#define lb 90
omega: !(0.5) O Si !(lw) [ &(lb) M ^(la) L ] [ \\(120) &(lb) M ^(la) L ] [ \\(240) &(lb) M ^(la) L ]
M -> /(90) Jl(hip_min, hip_max, eff, vel) J C(0.3) \\(90)
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Ja(0, 1, 0) Jl(slide_min, slide_max, slide_eff, vel) Jp K(seg) St",

    // ── 7. Strider ─────────────────────────────────────────────────────
    // Quadruped with hip+knee legs standing on flat feet. Each ankle is a
    // passive ball joint (Jb) with its own limits (Jlc drops the leg's
    // hinge limits first): it twists at most tw_max about the shin and
    // swings in a cone of sw_max, so the foot settles flat on the ground.
    "\
#define hip_min -0.6
#define hip_max 0.6
#define knee_min -0.8
#define knee_max 0.8
#define eff 3000
#define vel 5
#define tw_min -0.3
#define tw_max 0.3
#define sw_min -0.45
#define sw_max 0.45
#define lw 0.2
#define seg 0.6
#define la 45
#define lb 90
omega: !(0.6) O Si !(lw) [ &(lb) M ^(la) L ] [ \\(90) &(lb) M ^(la) L ] [ \\(180) &(lb) M ^(la) L ] [ /(90) &(lb) M ^(la) L ]
M -> /(90) Jl(hip_min, hip_max, eff, vel) J C(0.4) \\(90)
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Jl(knee_min, knee_max, eff, vel) J K(seg) A
A -> Jlc Jla(0, 1, 0, tw_min, tw_max, eff, vel) Jla(1, 0, 0, sw_min, sw_max, eff, vel) Jb B(0.08, 0.35, 0.35) St",

    // ── 8. Scout ───────────────────────────────────────────────────────
    // The Crawler with a welded neck reaching forward and up (-(135) from
    // the body's growth axis, which points down once the robot is turned
    // over) and an ultrasonic eye at its tip turned to the ground ahead
    // (+(90)): it reads how high and how pitched the body rides.
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
#define neck 0.4
omega: !(0.45) O Si !(lw) [ -(135) C(neck) +(90) Su ] [ &(lb) M ^(la) L ] [ \\(90) &(lb) M ^(la) L ] [ \\(180) &(lb) M ^(la) L ] [ /(90) &(lb) M ^(la) L ]
M -> /(90) Jl(hip_min, hip_max, hip_eff, hip_vel) J C(0.3) \\(90)
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St",
];
