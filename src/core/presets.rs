//! The body plans Generation 0 and each generation's fresh robots are drawn
//! from: three relatives of one simple body, written so that evolution can
//! build on them.
//!
//! # One rule for every limb
//!
//! A body is a limb that bears limbs. One rule, `G`, grows every segment of
//! every limb: the pair of side limbs the segment bears, the hinge it hangs
//! on, the segment itself, and the rest of the limb. The trunk is a limb
//! whose segments each bear a pair of legs. A leg is a limb whose own side
//! limbs are told to have no segments, and so never grow. A second rule,
//! `T`, gives the last segment of a leg its foot sensor.
//!
//! One rule, because a rule no body uses decays: mutation changes it freely
//! while it shows nowhere, and by the time a body needs it it is broken. An
//! earlier draft kept a rule of its own for a leg with a knee, and after 100
//! generations that rule was broken in nearly every champion. Here the rule
//! that grows a limb of several segments is the one every trunk uses, so a
//! leg can always become such a limb.
//!
//! # What one mutation can grow
//!
//! The axiom works out how many segments each limb has, and one mutated
//! constant can change the answer:
//!
//! - **A knee.** A leg has `round(segs)` segments, one to three, each
//!   `leg / that` long. `segs` starts at 1.35; past 1.5 every leg is split
//!   into two halves of its old length with a hinge between them, parallel
//!   to the hip's, that turns `bend` either way. The leg is as long as it
//!   was, hip to foot: a 0.8 m leg becomes two parts of 0.4 m, the knee in
//!   the middle. 3.4 percent of the mutations of a body do it.
//!   (`bend` shows in no body until then, and drifts meanwhile.)
//! - **A pair of legs.** The trunk has `round(trunk / link)` segments, one
//!   to six, each bearing a pair: 7 percent of the mutations of a skink or
//!   a scout grow a third, 9 percent of an eel's.
//!
//! A leg gains parts by splitting, and a split leg is as long as it was:
//! its length is `leg`, which one mutation changes by a fifth at most. The
//! rules are written so that mutation has no other way to more segments
//! (see "The rules compute nothing" and "A limb ends on a number" below).
//! Of 60 000 mutants of each plan, 15 to 24 have a chain of more parts than
//! a leg that is longer than one: where mutation set a part of its own into
//! a rule (the smallest the lab builds) or swapped two of a rule's modules.
//! With the count-down these plans replaced, 870 to 900 had.
//!
//! Two things outside the plans let such a growth live. A fresh body is
//! derived with a step more than it needs (see `INITIAL_LSYSTEM_ITERATIONS`),
//! so what grew is whole at once. And a child takes each constant whole from
//! one parent or the other (see `AgentGenotype::crossover`): averaged, as
//! they once were, a count that had just passed a threshold fell back under
//! it in the first child.
//!
//! What evolution makes of it, over sixteen runs of 100 generations on the
//! flat floor and sixteen on rough ground: little yet. In one of the
//! sixteen runs on the flat floor the champions had grown a third pair of
//! legs, on rough ground in none, and none had a knee. Bred from
//! a plan that starts with knees (measured with a count-down that
//! subtracted), fourteen runs in sixteen had dropped them, and the two that
//! had not walked 2 and 4 m where the others walked 20. A leg split by a
//! driven hinge is, with these brains, a worse leg. The grammar offers the
//! growth; it cannot make it pay. Closing the ways to longer legs cost no
//! distance (24.8 m on the flat floor and 24.5 m on rough ground, against
//! 25.7 and 22.0 with the count-down that could flip, p = 0.75 and 0.22,
//! all with the hips' reach still unbounded); bounding the reach did (see
//! "Nothing a body is made of is free" below).
//!
//! A cross can grow too: a child of a skink and an eel that takes the one's
//! `trunk` and the other's `link` has a trunk of three segments.
//!
//! # How a plan is written
//!
//! The rates below are those of the default mutation rate, 0.3.
//!
//! - **Every number evolution should tune is a constant the axiom names.**
//!   The L-system copies a constant's value into the rules when it reads
//!   them, so afterwards a mutated constant reaches the body only through
//!   the axiom, which hands each one down as a module parameter. That goes
//!   for the joints' ranges too: a draft that wrote them into the axiom as
//!   plain numbers, where nothing mutates, walked a quarter less far. A
//!   given constant changes in about one body mutation in seven, by up to a
//!   fifth.
//! - **The axiom does the arithmetic and holds the bounds.** Mutation never
//!   edits the axiom, and it flips `*` to `/`, `+` to `-` and `>` to `<`
//!   wherever a rule has one, but leaves `min`, `max`, `round`, `sqrt`, a
//!   power and a unary minus alone. So counts are rounded and clamped there,
//!   lengths divided there, and the measures the physics cannot bear are
//!   bounded there: the trunk no thinner than 0.3 m (thinner, it tears), a
//!   hip turning no further than 0.75 rad either way (unbounded, 11 robots a
//!   generation were culled late in a run; bounded, 6), the trunk's joints
//!   1.2 rad.
//! - **Nothing a body is made of is free.** A leg is set out from the trunk
//!   by a move that builds no part, so the way from trunk to hip weighs
//!   nothing and cannot be seen. Unbounded, that `reach` grew from 0.8 m to
//!   a median of 1.5 m on the flat floor and 1.7 m on rough ground in 100
//!   generations, 2.3 m at most, and the legs hung about a metre of empty
//!   air from the body. (A likely reason, which no measurement isolates:
//!   the trunk's joints swing each hip through an arc of that radius.) The
//!   axiom now holds a hip within 0.25 m of where its leg would touch the
//!   trunk's flank, which is where the plans start. Champions then walked
//!   20.3 m on the flat floor and 14.6 m on rough ground, where they had
//!   walked 24.8 and 24.5, broke in 1.4 and 1.2 percent of their walks
//!   where they had in 9.0 and 7.2, and one robot a generation was culled
//!   where five had been. Why they break less is not settled: among the
//!   unbounded champions a longer reach did not go with more broken walks.
//!   A hip set on the flank itself walked less far on rough ground (11.9 m;
//!   19.0 m on the flat floor), and a welded bar in place of the empty
//!   reach far less (9.9 m on the flat floor). The bound does not know the
//!   head, which the first pair of legs hangs from, and mutation can still
//!   set a move of its own into a rule and lengthen it; no champion had.
//! - **The rules compute nothing, not even the count-down.** A limb counts
//!   its segments down by taking a square root, which mutation cannot flip.
//!   An earlier count-down subtracted (`n - 100`). With the sign flipped,
//!   which 1.5 percent of the mutations of a body did, every limb grew a
//!   whole segment with every step: legs of two, three and four parts, each
//!   as long as the leg had been, on a body of 26 parts.
//! - **A limb ends on a number, not below one.** The limb rule grows while
//!   its count is above the end mark `z`, a parameter the axiom hands down,
//!   and a finished limb's count is the mark itself. The L-system calls two
//!   numbers equal within a part in a hundred million, so neither `n > z`
//!   nor the `n < z` mutation makes of it holds for a finished limb, and
//!   nothing can wake one. The limbs that are told to have no segments
//!   count `z` from the start. An earlier draft ended a limb below a
//!   threshold (`n > 30`). Mutation also copies rules, and a copy whose
//!   comparison then flipped matched every finished limb and every limb
//!   that was to have no segments: 41 parts at four steps, more than the
//!   lab takes at five. With the mark, such a copy matches nothing and the
//!   body is the one it was.
//! - **A condition holds one comparison.** The L-system reads `a & b` and
//!   writes it back as `a && b`, which it cannot read again, and the lab
//!   stores bodies as written back. So "this limb is finished and is a leg"
//!   is `max(n, k) <= 300`.
//! - **The foot has a rule of its own symbol.** On the limb's symbol a
//!   flipped comparison would let both rules match one module, and which
//!   applied would be a draw from the body's seed. On its own, the flip
//!   moves the sensors from the feet to the trunk and changes no part.
//! - **No plan uses a turn and its opposite.** Mutation swaps `+` with `-`,
//!   `&` with `^` and `\` with `/`, but only in a plan that contains both
//!   of a pair. A flipped `-(90)` would grow every leg upward, so the plans
//!   turn with `-`, `&` and `/` alone (`-(270)` where `+(90)` would read
//!   better, `&(-90)` for the way back from a leg's reach).
//! - **Numbers that must hold are large, or are no numbers.** Mutation
//!   nudges a number in a rule by about 0.09 whatever its size: nothing to
//!   90 degrees, a 4000 N m motor, the 300 between a last segment's 100 and
//!   the 1000 that marks a limb as no leg. The limb rule's own condition
//!   compares two parameters and holds no number at all.
//! - **New rules get new names.** A child takes each rule from one parent
//!   or the other by the rule's symbol. The rules were `L` and `Y` while
//!   limbs counted in hundreds; an axiom that counts in powers, crossed
//!   with such a rule, grows a trunk without end. Under names of their own,
//!   `G` and `T`, the old rules ride along unused in a child of a robot
//!   kept from before and one of these, and the child is a body of its
//!   first parent's kind.
//!
//! `examples/bodyplans.rs` measures all of this for any plan, without or
//! with physics, and `evolve --plans DIR` breeds from the plans of a folder
//! in place of these.
//!
//! # The rules
//!
//! `G(n, z, m, r, dl, dh, fl, fh, k, c, a, b, w, el, eh, gl, gh, k2)` is
//! what is left of a limb. `n` counts its segments: 10 to the power of 2 to
//! the power of their number, so 100 for one segment, 10 000 for two and
//! 100 000 000 for three, and its square root is the count of a limb one
//! segment shorter. `z` is the end mark, 10: the count of a limb with no
//! segment left. The segments are each `m` long and `r` in radius, the next
//! one on a hinge that turns `dl` one way and `dh` the other, those after
//! it `fl` and `fh`. While `n` is above `z` the rule grows one segment and
//! hands the square root of `n` on to the limb's remainder. Before the
//! segment, to either side and `w` out, it sets a side limb counted by `c`,
//! of segments `a` long and `b` in radius, on hinges of `el`, `eh` and then
//! `gl`, `gh`, a limb of the kind `k2` whose own side limbs count `z`, no
//! segments. After the segment, `T`.
//!
//! `T(n, k)` is a foot in waiting: where the segment before it was the
//! limb's last (`n` is 100, at most 300) and the limb is a leg (`k` at most
//! 300: the axiom gives the trunk 1000 and its side limbs 0) it becomes a
//! touch sensor.
//!
//! The turns before a side limb are the legs': out sideways (a quarter turn
//! to the side, the reach, a quarter turn back), then down, so that the
//! hinges swing fore and aft.

/// The parameters of the limb rule.
macro_rules! limb {
    () => {
        "G(n, z, m, r, dl, dh, fl, fh, k, c, a, b, w, el, eh, gl, gh, k2)"
    };
}

/// A side limb, a quarter turn `$out` to the side and `$back` again: a limb
/// of the measures its parent was handed for its side limbs, whose own side
/// limbs count the end mark, no segments.
macro_rules! side_limb {
    ($out:literal, $back:literal) => {
        concat!(
            "[ &(",
            $out,
            ") f(w) &(",
            $back,
            ") -(90) /(90) G(c, z, a, b, el, eh, gl, gh, k2, z, 0, 0, 0, 0, 0, 0, 0, 1000) ]"
        )
    };
}

/// The rule every limb grows by, a segment a step: its pair of side limbs,
/// its hinge, the segment, the foot it may become, the rest of the limb.
macro_rules! segment_rule {
    () => {
        concat!(
            limb!(),
            " : n > z -> ",
            side_limb!("90", "-90"),
            " ",
            side_limb!("-90", "90"),
            " Jl(-dl, dh, 4000, 5) J K(m, r) T(n, k)",
            " G(sqrt(n), z, m, r, fl, fh, fl, fh, k, c, a, b, w, el, eh, gl, gh, k2)"
        )
    };
}

/// The foot: a touch sensor on the last segment of a leg.
macro_rules! foot_rule {
    () => {
        "T(n, k) : max(n, k) <= 300 -> St"
    };
}

/// How many segments a leg has: `segs` rounded, one to three.
macro_rules! leg_segments {
    () => {
        "min(3, max(1, round(segs)))"
    };
}

/// How many segments the trunk has: its length in links, rounded, one to
/// six (a body gets six derivation steps at most).
macro_rules! trunk_segments {
    () => {
        "min(6, max(1, round(trunk / link)))"
    };
}

/// How far out from the trunk's axis a leg hangs: `reach`, but no further
/// than a quarter of a metre beyond where the leg would touch the trunk's
/// flank. The way out builds no part, and unbounded it grew to two metres
/// and more.
macro_rules! hip_reach {
    () => {
        "min(reach, max(girth, 0.3) + shank + 0.25)"
    };
}

/// What the axiom hands the trunk after the head: back to the head's centre,
/// a quarter turn to lie level, then the trunk as a limb (its segments, the
/// end mark, a segment's length and radius, its joints' ranges, 1000 for
/// "no leg") and what its side limbs, the legs, are to be (their segments,
/// a segment's length and radius, their reach, the hips' ranges and the
/// knees', 0 for "a leg"). Counts, lengths and bounds are worked out here, where mutation
/// does not reach: a count of segments is handed down as 10 to the power of
/// 2 to the power of it, which the rule counts down by square roots, and a
/// leg's segments are each the leg's length divided by their number.
macro_rules! trunk_start {
    () => {
        concat!(
            "f(0 - head) -(270) G(10 ^ (2 ^ ",
            trunk_segments!(),
            "), 10, link, max(girth, 0.3),",
            " min(yawl, 1.2), min(yawr, 1.2), min(yawl, 1.2), min(yawr, 1.2), 1000, ",
            "10 ^ (2 ^ ",
            leg_segments!(),
            "), leg / ",
            leg_segments!(),
            ", shank, ",
            hip_reach!(),
            ", min(fore, 0.75), min(aft, 0.75), min(bend, 1), min(bend, 1), 0)"
        )
    };
}

/// The two rules every plan shares.
macro_rules! rules {
    () => {
        concat!("\n", segment_rule!(), "\n", foot_rule!())
    };
}

/// The joints' ranges every plan starts with (rad): the trunk's to either
/// side, the hips' fore and aft, and a knee's, should a leg split.
macro_rules! ranges {
    () => {
        "#define yawl 0.5\n\
#define yawr 0.5\n\
#define fore 0.7\n\
#define aft 0.7\n\
#define bend 0.1\n"
    };
}

/// The skink's measures, the scout's too: a trunk of 2.3 links, so that a
/// tenth more of it brings a third segment, and legs of 1.35 segments, so
/// that a ninth more splits them.
macro_rules! skink_constants {
    () => {
        concat!(
            "#define head 0.52\n\
#define trunk 2.75\n\
#define link 1.2\n\
#define girth 0.36\n\
#define leg 0.8\n\
#define segs 1.35\n\
#define shank 0.2\n\
#define reach 0.8\n",
            ranges!()
        )
    };
}

/// The names of the plans in [`MORPHOLOGY_POOL`], in its order.
pub const PLAN_NAMES: &[&str] = &["Skink", "Scout", "Eel"];

/// The body plans, as L-system sources, for the lab's starting five
/// derivation steps (four build them; the fifth is in hand).
pub const MORPHOLOGY_POOL: &[&str] = &[
    // -- Skink ---------------------------------------------------------
    // A head, two pairs of stiff legs and a tail: a trunk of two segments,
    // six motors.
    concat!(
        skink_constants!(),
        "omega: O(head) Si ",
        trunk_start!(),
        rules!()
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
        rules!()
    ),
    // -- Eel -----------------------------------------------------------
    // A trunk of short links on short legs with a tail tip that slides in
    // and out instead of turning: seven motors, one of them the lab's
    // slider. The tip is set in the axiom, after the trunk (`Ja(0, 1, 0)`
    // turns the joint's axis to the growth direction before `Jp` makes it
    // prismatic), so like the scout's eye it cannot be lost.
    concat!(
        "#define head 0.5\n\
#define trunk 1.9\n\
#define link 0.8\n\
#define girth 0.36\n\
#define leg 0.7\n\
#define segs 1.35\n\
#define shank 0.2\n\
#define reach 0.7\n",
        ranges!(),
        "omega: O(head) Si ",
        trunk_start!(),
        " Ja(0, 1, 0) Jl(-0.2, 0.2, 1500, 5) Jp K(link, max(girth, 0.3))",
        rules!()
    ),
];
