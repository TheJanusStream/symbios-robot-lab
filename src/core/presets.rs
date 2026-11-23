//! The body plans Generation 0 and each generation's fresh robots are drawn
//! from: three relatives of one simple body, written so that evolution can
//! build on them.
//!
//! # Why so few, and so plain
//!
//! The former catalogue held eight hand-built walkers of 9 to 18 motors.
//! Bred alone, two of them reached a fitness of 1 in twelve generations, and
//! what evolution did with the catalogue was to cut bodies down: sixteen runs
//! of 100 generations ended on the millipede fourteen times, nine of those
//! stripped to one leg pair and a tail on five or six motors (the lab scores
//! no body with fewer than five). Two things decided the choice, and neither
//! was how well a plan walks at first:
//!
//! - **The fitness of the time paid for few motors.** It taxed the speed of
//!   every joint (see [`crate::simulation::fitness`]): a leg with a knee or
//!   a slider near its foot sensor is steered by that sensor in steps, and
//!   kept 3 to 8 percent of its distance in the plans tried, while a body
//!   with one motor per leg learned to walk in a few generations. So the
//!   plans have stiff legs, and what evolution adds is legs and trunk. The
//!   fitness that replaced it does not: when it came, the drafts with kneed
//!   and with sliding legs scored 5.9 and 6.1 after twelve generations where
//!   these plans scored 8.2 (0.26 and 0.24 against 1.6 before), so such legs
//!   are worth another look.
//! - **Plans share a gene pool, and a fast starter takes it over.** A crab
//!   (six stiff legs in a ring) scored twice the skink's fitness after ten
//!   generations, won all sixteen runs it shared with the skink, and then
//!   stalled at half the fitness the skink reaches alone. A seal (two legs
//!   with knees) won fifteen of sixteen such runs and ended them no better
//!   than the former catalogue. So the plans are one family: every one grows
//!   the same legs and trunk from the same rules, and a child of any two is
//!   one of them.
//!
//! # How a plan is written
//!
//! The rates below are those of the default mutation rate, 0.3.
//!
//! - **Every number evolution should tune is a constant the axiom names.**
//!   The L-system copies a constant's value into the rules when it reads
//!   them, so afterwards a mutated constant reaches the body only through
//!   the axiom, which hands each one down as a module parameter. A constant
//!   the axiom does not name is dead weight: of the eleven to twenty-one
//!   each former plan defined, two to six still did anything. A given
//!   constant changes in about one body mutation in seven, by up to a fifth.
//! - **The axiom holds no legs and no trunk.** Mutation never edits the
//!   axiom, so what is laid out there can never be added to or lost (the
//!   scout's eye is the one thing that is). Legs and trunk segments come
//!   from rules.
//! - **Both legs of a pair come from one rule**, called once for each side,
//!   so whatever mutation does to a leg it does to both. Rules that spell
//!   out a left and a right leg drift apart, and a lopsided body walks in
//!   circles. (A pair can still come apart: mutation may duplicate the leg
//!   rule or edit one of the two calls.)
//! - **Growth is a threshold on a constant.** The trunk gains a segment each
//!   time `trunk / link` passes 2.5, 3.5 and so on, and each plan starts a
//!   tenth below such a threshold. The new segment's legs need one more
//!   derivation step than a fresh body gets; that count changes in one body
//!   mutation in 33, up as often as down, and one step fewer leaves every
//!   plan with four motors, which the lab does not score.
//! - **Rules compute nothing but the trunk's count-down.** Mutation flips
//!   `*` to `/`, `+` to `-` and `>` to `<` wherever a rule has one, so the
//!   ratios the thresholds compare are worked out in the axiom. The one
//!   subtraction left, flipped, makes a trunk that grows with every step.
//! - **No plan uses a turn and its opposite.** Mutation swaps `+` with `-`,
//!   `&` with `^` and `\` with `/`, but only in a plan that contains both
//!   of a pair. A flipped `-(90)` would grow every leg upward, so the plans
//!   turn with `-`, `&` and `/` alone (`-(270)` where `+(90)` would read
//!   better, and the leg's way back from its reach passed in as a second
//!   side).
//! - **Numbers that must hold are large, numbers that should drift are
//!   small.** Mutation nudges a number in a rule by about 0.09 whatever its
//!   size (each one in about one body mutation in seven): nothing to 90
//!   degrees, a 4000 N m motor or a threshold of 150, a real change to a
//!   joint's range in radians. It also sets a number, now and then, to the
//!   first constant by name within 0.5 of it: here a joint range to the
//!   trunk's radius.
//! - **Bodies are thick for their legs.** The physics tears a body apart
//!   when its trunk or head is light against the legs hanging from it, or
//!   its legs thin against their motors: with a trunk radius of 0.25 m
//!   instead of 0.3 a draft lost twelve robots a generation. These
//!   proportions keep a full mutation of margin.
//!
//! `examples/bodyplans.rs` measures all of this for any plan, without or
//! with physics.
//!
//! # The rules
//!
//! `G(s, t, a, b, w)` is a leg: out sideways by the reach `w` (a quarter
//! turn `s` to the side, a quarter turn `t` back), then down, one stiff part
//! of length `a` and radius `b` on a hip that swings fore and aft, ending in
//! a touch sensor.
//!
//! `T(n, m, r, a, b, w)` is the trunk, growing backward from the head: a leg
//! pair, then a segment of length `m` and radius `r` on a hinge that turns it
//! sideways. While `n` (a hundred times `trunk / link`) is above 150 another
//! follows with `n` a hundred less; the last one is the tail.

/// The leg rule every plan shares.
macro_rules! leg_rule {
    () => {
        "G(s, t, a, b, w) -> &(s) f(w) &(t) -(90) /(90) Jl(-0.8, 0.8, 4000, 5) J K(a, b) St"
    };
}

/// A leg pair: the same rule called for either side.
macro_rules! leg_pair {
    () => {
        "[ G(90, -90, a, b, w) ] [ G(-90, 90, a, b, w) ]"
    };
}

/// A trunk segment that is followed by another: a leg pair, the segment on
/// its hinge, the rest of the trunk.
macro_rules! trunk_rule {
    () => {
        concat!(
            "T(n, m, r, a, b, w) : n > 150 -> ",
            leg_pair!(),
            " Jl(-0.5, 0.5, 4000, 5) J K(m, r) T(n - 100, m, r, a, b, w)"
        )
    };
}

/// The trunk's last leg pair and the tail behind it, on a hinge.
macro_rules! tail_rule {
    () => {
        concat!(
            "T(n, m, r, a, b, w) : n <= 150 -> ",
            leg_pair!(),
            " Jl(-0.5, 0.5, 4000, 5) J K(m, r)"
        )
    };
}

/// The same, with a tail that slides in and out along the trunk instead of
/// turning: `Ja(0, 1, 0)` turns the joint's axis to the growth direction
/// before `Jp` makes the joint prismatic.
macro_rules! sliding_tail_rule {
    () => {
        concat!(
            "T(n, m, r, a, b, w) : n <= 150 -> ",
            leg_pair!(),
            " Ja(0, 1, 0) Jl(-0.2, 0.2, 1500, 5) Jp K(m, r)"
        )
    };
}

/// What the axiom hands the trunk after the head: back to the head's centre,
/// a quarter turn to lie level, then how many links the trunk spans (times a
/// hundred), a link's length and radius, and the legs' length, radius and
/// reach.
macro_rules! trunk_start {
    () => {
        "f(0 - head) -(270) T(trunk / link * 100, link, girth, leg, shank, reach)"
    };
}

/// The skink's measures, the scout's too: a trunk of 2.3 links, so that a
/// tenth more of it brings a third segment.
macro_rules! skink_constants {
    () => {
        "#define head 0.52\n\
#define trunk 2.75\n\
#define link 1.2\n\
#define girth 0.36\n\
#define leg 0.8\n\
#define shank 0.2\n\
#define reach 0.8\n"
    };
}

/// The names of the plans in [`MORPHOLOGY_POOL`], in its order.
pub const PLAN_NAMES: &[&str] = &["Skink", "Scout", "Eel"];

/// The body plans, as L-system sources, for the lab's starting three
/// derivation steps.
pub const MORPHOLOGY_POOL: &[&str] = &[
    // -- Skink ---------------------------------------------------------
    // A head, two pairs of stiff legs and a tail of two hinged segments:
    // six motors.
    concat!(
        skink_constants!(),
        "omega: O(head) Si ",
        trunk_start!(),
        "\n",
        trunk_rule!(),
        "\n",
        tail_rule!(),
        "\n",
        leg_rule!()
    ),
    // -- Scout ---------------------------------------------------------
    // The skink with an ultrasonic eye on a welded stalk, reaching out past
    // the head and looking at the ground ahead (-(135) from the growth axis,
    // which points down once the robot is turned over, then a quarter turn):
    // it reads how high and how pitched the head rides. The same measures as
    // the skink, so evolution decides only whether the eye is worth having.
    concat!(
        skink_constants!(),
        "omega: O(head) Si [ Jf -(135) C(head * 2) -(270) Su ] ",
        trunk_start!(),
        "\n",
        trunk_rule!(),
        "\n",
        tail_rule!(),
        "\n",
        leg_rule!()
    ),
    // -- Eel -----------------------------------------------------------
    // A long trunk of short links on short legs, whose tail slides in and
    // out instead of turning: seven motors, one of them the lab's slider.
    // In three steps it grows two leg pairs on three segments; its third
    // pair is laid down already and grows with a fourth step.
    concat!(
        "#define head 0.5\n\
#define trunk 2.6\n\
#define link 0.8\n\
#define girth 0.36\n\
#define leg 0.7\n\
#define shank 0.2\n\
#define reach 0.7\n\
omega: O(head) Si ",
        trunk_start!(),
        "\n",
        trunk_rule!(),
        "\n",
        sliding_tail_rule!(),
        "\n",
        leg_rule!()
    ),
];
