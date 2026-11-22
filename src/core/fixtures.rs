//! Bodies the tests exercise the lab's machinery with: entries of the former
//! catalogue, kept for what they contain that the body plans do not
//! (three-joint legs, knees with their own range, stochastic rules, sensors
//! far from joints, sliding shins, ball joints, an eye on a neck).
//!
//! The unit tests reach this module as `crate::core::fixtures`; the
//! integration tests include the file by path.

/// A sphere on four legs a quarter turn apart, each rolling one, two or three
/// segments: stochastic rules, so its seed decides its body.
pub const QUADRUPED: &str = "\
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
0.3 : L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg/3.0) St";

/// A sphere on six legs of a hip stub, a thigh and a shin: 18 driven joints,
/// three to a leg.
pub const HEXAPOD: &str = "\
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
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, knee_eff, knee_vel) J K(seg) St";

/// A spine of three parts with a leg pair at each joint, the left and the
/// right leg written out: sensors and joints far apart along the body.
pub const INSECT: &str = "\
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
LP -> [ Jf &(90) C(hip) ^(90) -(90) /(90) Jl(hipmin, hipmax, eff, vel) J K(leg, lw) Jl(knee_min, knee_max, eff, vel) J K(leg, lw) St ] [ Jf \\(180) &(90) C(hip) ^(90) +(90) /(90) Jl(hipmin, hipmax, eff, vel) J K(leg, lw) Jl(knee_min, knee_max, eff, vel) J K(leg, lw) St ]";

/// A low quadruped whose knees have their own range (-1.0 to 0.2) after their
/// hips' (-0.5 to 0.5).
pub const CRAWLER: &str = "\
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
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St";

/// Three legs whose shins slide (prismatic joints), each within its own limit.
pub const TRIPOD: &str = "\
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
L -> Jl(hip_min, hip_max, eff, vel) J !(lw) K(seg) Ja(0, 1, 0) Jl(slide_min, slide_max, slide_eff, vel) Jp K(seg) St";

/// A quadruped on flat feet, its ankles ball joints limited in twist and swing.
pub const STRIDER: &str = "\
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
A -> Jlc Jla(0, 1, 0, tw_min, tw_max, eff, vel) Jla(1, 0, 0, sw_min, sw_max, eff, vel) Jb B(0.08, 0.35, 0.35) St";

/// The crawler with an ultrasonic eye on a welded neck, looking at the ground
/// ahead.
pub const SCOUT: &str = "\
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
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) K(seg) Jl(knee_min, knee_max, keff, kvel) J K(seg) St";

/// Every fixture, for tests that hold for any body.
pub const ALL: &[&str] = &[QUADRUPED, HEXAPOD, INSECT, CRAWLER, TRIPOD, STRIDER, SCOUT];
