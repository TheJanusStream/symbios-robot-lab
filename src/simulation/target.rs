//! The target task: each robot walks toward targets, reads their bearing,
//! and scores the ground it gains toward them.
//!
//! While it settles, a robot heads for a target [`TARGET_DISTANCE`] from its
//! spawn point in the generation's direction. The measured window is then
//! split into [`TARGET_LEGS`] legs, and at the start of each a new target
//! appears [`TARGET_DISTANCE`] away at an angle to the robot's own heading,
//! on alternating sides, so every evaluation asks for turning both ways (a
//! single fixed target let robots score by walking whichever way they
//! happened to face). The angles are drawn per generation from the run's
//! seed, so the robots of a generation turn alike and compete fairly. In
//! the windowed lab the user can place one target for every robot instead.

use crate::core::population::Population;
use crate::simulation::clock::GenerationClock;
use crate::simulation::rng::{LabRng, generation_seed};
use avian3d::prelude::{Position, Rotation};
use bevy::prelude::*;

/// How far (m) a robot's target lies from where it stands when it gets it.
pub const TARGET_DISTANCE: f32 = 10.0;
/// Targets in the measured window, one per leg.
pub const TARGET_LEGS: usize = 2;
/// The smallest and largest turn (rad) a leg's target asks for.
const MIN_TURN: f32 = std::f32::consts::FRAC_PI_4;
const MAX_TURN: f32 = 3.0 * std::f32::consts::FRAC_PI_4;
/// Tells the targets' seeds apart from the grounds'.
const TARGET_STREAM: u64 = 0x7A26_E7D1_5EED_0001;

/// A robot's target (world x and z), where its brain reads the bearing, and
/// its progress through the legs of its life. Lives on the body part, next
/// to the brain.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct WalkTarget {
    pub point: Vec2,
    pub input_offset: usize,
    /// The leg it is on: 0 while settling, then 1 to [`TARGET_LEGS`].
    pub leg: usize,
    /// Where the robot stood when it got this target.
    pub start: Vec3,
    /// Ground gained toward the targets of earlier legs (m).
    pub gained: f32,
}

impl WalkTarget {
    /// A robot's first target, which it settles toward.
    pub fn first(point: Vec2, input_offset: usize) -> Self {
        Self {
            point,
            input_offset,
            leg: 0,
            start: Vec3::ZERO,
            gained: 0.0,
        }
    }

    /// The ground gained over the measured window so far: toward earlier
    /// legs' targets, and toward this one since it appeared (none while
    /// settling).
    pub fn ground_gained(&self, now: Vec3) -> f32 {
        if self.leg == 0 {
            return 0.0;
        }
        self.gained + progress(self.start, now, self.point)
    }
}

/// The signed turn (rad, positive toward the side where the bearing's sine
/// is positive) that leg `leg` of a generation asks for: a size drawn per
/// generation and leg between 45 and 135 degrees, and sides alternating
/// from one drawn per generation.
pub fn leg_turn(run_seed: u64, generation: usize, leg: usize) -> f32 {
    let leg_stream = (leg as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let size = generation_seed(run_seed, generation, TARGET_STREAM ^ leg_stream);
    let size = (size >> 40) as f32 / (1u64 << 24) as f32;
    let turn = MIN_TURN + (MAX_TURN - MIN_TURN) * size;
    let first_positive = generation_seed(run_seed, generation, !TARGET_STREAM) & 1 == 0;
    if first_positive == (leg % 2 == 1) {
        turn
    } else {
        -turn
    }
}

/// The point `turn` (rad) off a robot's heading (its body's x axis over the
/// ground), [`TARGET_DISTANCE`] from where it stands.
pub fn target_off_heading(position: Vec3, rotation: Quat, turn: f32) -> Vec2 {
    let heading = rotation * Vec3::X;
    let heading = Vec2::new(heading.x, heading.z)
        .try_normalize()
        .unwrap_or(Vec2::X);
    Vec2::new(position.x, position.z) + Vec2::from_angle(turn).rotate(heading) * TARGET_DISTANCE
}

/// The leg the generation clock is on: 0 while settling, then 1 to
/// [`TARGET_LEGS`], the measured window split evenly.
pub fn leg_of(clock: &GenerationClock) -> usize {
    if !clock.is_settled() {
        return 0;
    }
    let window = clock
        .total_steps()
        .saturating_sub(clock.settle_step())
        .max(1);
    let into = clock.step().saturating_sub(clock.settle_step());
    (1 + into as usize * TARGET_LEGS / window as usize).min(TARGET_LEGS)
}

/// Starts each robot's next leg when the clock reaches it: the ground it
/// gained toward the old target is banked, and its new target appears at the
/// leg's turn off its heading, or at the user's target if one is placed. A
/// robot that outlives generations (a showcased one) starts its legs again
/// with each generation, still heading for its last target meanwhile.
pub fn advance_targets(
    clock: Res<GenerationClock>,
    population: Res<Population>,
    rng: Res<LabRng>,
    user: Res<UserTarget>,
    mut targets: Query<(&mut WalkTarget, &Position, &Rotation)>,
) {
    let leg = leg_of(&clock);
    for (mut target, position, rotation) in &mut targets {
        if target.leg > leg {
            target.leg = leg;
            target.gained = 0.0;
            target.start = position.0;
            continue;
        }
        if target.leg == leg {
            continue;
        }
        target.gained = target.ground_gained(position.0);
        target.leg = leg;
        target.start = position.0;
        target.point = user.0.unwrap_or_else(|| {
            let turn = leg_turn(rng.seed(), population.generation, leg);
            target_off_heading(position.0, rotation.0, turn)
        });
    }
}

/// A target the user placed for every robot (world x and z), in place of
/// the generation's direction until cleared.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq)]
pub struct UserTarget(pub Option<Vec2>);

/// The direction (a unit vector over world x and z) in which a generation's
/// targets lie.
pub fn target_direction(run_seed: u64, generation: usize) -> Vec2 {
    let seed = generation_seed(run_seed, generation, TARGET_STREAM);
    let turn = (seed >> 40) as f32 / (1u64 << 24) as f32;
    Vec2::from_angle(turn * std::f32::consts::TAU)
}

/// A robot's target: the user's if one is placed, otherwise
/// [`TARGET_DISTANCE`] from its spawn point (world x and z) in the
/// generation's direction.
pub fn target_for(spawn: Vec2, run_seed: u64, generation: usize, user: UserTarget) -> Vec2 {
    user.0
        .unwrap_or_else(|| spawn + target_direction(run_seed, generation) * TARGET_DISTANCE)
}

/// The sine and cosine of the target's angle from the robot's heading (its
/// body's x axis) on the ground plane: `[0, 1]` straight ahead, `[0, -1]`
/// behind, `[±1, 0]` to either side. Straight ahead when the robot points
/// up or stands on its target.
pub fn bearing(position: Vec3, rotation: Quat, target: Vec2) -> [f32; 2] {
    let heading = rotation * Vec3::X;
    let heading = Vec2::new(heading.x, heading.z);
    let to_target = target - Vec2::new(position.x, position.z);
    if heading.length_squared() < 1e-6 || to_target.length_squared() < 1e-6 {
        return [0.0, 1.0];
    }
    let angle = heading.angle_to(to_target);
    [angle.sin(), angle.cos()]
}

/// How much nearer (m, over the ground plane) `target` is from `now` than it
/// was from `start`; 0 if no nearer.
pub fn progress(start: Vec3, now: Vec3, target: Vec2) -> f32 {
    let distance = |p: Vec3| (target - Vec2::new(p.x, p.z)).length();
    (distance(start) - distance(now)).max(0.0)
}

/// Points every robot at the user's target when the user places or moves
/// it. Clearing it leaves this generation's targets where they are; the next
/// generation's lie in its own direction.
pub fn follow_user_target(user: Res<UserTarget>, mut targets: Query<&mut WalkTarget>) {
    if !user.is_changed() {
        return;
    }
    if let Some(point) = user.0 {
        for mut target in &mut targets {
            target.point = point;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bearing_says_where_the_target_lies() {
        let at = Vec3::new(1.0, 0.5, 1.0);
        let close =
            |[s, c]: [f32; 2], [es, ec]: [f32; 2]| (s - es).abs() < 1e-5 && (c - ec).abs() < 1e-5;
        // Facing +x: ahead, behind, then either side.
        assert!(close(
            bearing(at, Quat::IDENTITY, Vec2::new(5.0, 1.0)),
            [0.0, 1.0]
        ));
        assert!(close(
            bearing(at, Quat::IDENTITY, Vec2::new(-5.0, 1.0)),
            [0.0, -1.0]
        ));
        let left = bearing(at, Quat::IDENTITY, Vec2::new(1.0, 5.0));
        let right = bearing(at, Quat::IDENTITY, Vec2::new(1.0, -5.0));
        assert!(
            close(left, [1.0, 0.0]) && close(right, [-1.0, 0.0]),
            "{left:?} {right:?}"
        );
        // Turned to face +z, the target on +z is ahead.
        let facing_z = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        assert!(close(
            bearing(at, facing_z, Vec2::new(1.0, 5.0)),
            [0.0, 1.0]
        ));
        // Standing on the target, or pointing straight up: ahead.
        assert_eq!(bearing(at, Quat::IDENTITY, Vec2::new(1.0, 1.0)), [0.0, 1.0]);
        let up = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert_eq!(bearing(at, up, Vec2::new(9.0, 9.0)), [0.0, 1.0]);
    }

    /// Each generation's legs turn to alternating sides, by 45 to 135
    /// degrees, and a seed repeats them.
    #[test]
    fn legs_turn_to_alternating_sides() {
        let mut first_sides = std::collections::BTreeSet::new();
        for generation in 0..30 {
            let turns: Vec<f32> = (1..=TARGET_LEGS)
                .map(|leg| leg_turn(9, generation, leg))
                .collect();
            for turn in &turns {
                assert!((MIN_TURN..=MAX_TURN).contains(&turn.abs()), "{turn}");
            }
            assert!(turns.windows(2).all(|w| w[0].signum() != w[1].signum()));
            first_sides.insert(turns[0] > 0.0);
            assert_eq!(turns[0], leg_turn(9, generation, 1));
        }
        assert_eq!(first_sides.len(), 2, "the first leg always turned one way");
    }

    /// A leg's target lies the leg's turn off the robot's heading, at the
    /// target distance, wherever the robot stands and faces.
    #[test]
    fn a_new_target_lies_off_the_heading() {
        let at = Vec3::new(3.0, 0.4, -2.0);
        let facing_z = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
        let point = target_off_heading(at, facing_z, std::f32::consts::FRAC_PI_2);
        assert!(((point - Vec2::new(3.0, -2.0)).length() - TARGET_DISTANCE).abs() < 1e-4);
        let [sin, cos] = bearing(at, facing_z, point);
        assert!((sin - 1.0).abs() < 1e-5 && cos.abs() < 1e-5, "{sin} {cos}");
    }

    /// The clock's measured window splits evenly into legs; settling is leg
    /// 0.
    #[test]
    fn the_clock_walks_through_the_legs() {
        let mut clock = GenerationClock::default();
        clock.start(15.0, 1.0 / 64.0);
        let mut legs = Vec::new();
        while !clock.is_finished() {
            legs.push(leg_of(&clock));
            clock.advance();
        }
        legs.push(leg_of(&clock));
        let mut seen = legs.clone();
        seen.dedup();
        assert_eq!(seen, (0..=TARGET_LEGS).collect::<Vec<_>>());
        let measured = legs.iter().filter(|&&leg| leg > 0).count();
        let first = legs.iter().filter(|&&leg| leg == 1).count();
        assert!((first as f32 / measured as f32 - 1.0 / TARGET_LEGS as f32).abs() < 0.02);
    }

    /// Ground gained toward earlier legs' targets is kept; none counts while
    /// settling.
    #[test]
    fn ground_gained_adds_up_over_the_legs() {
        let mut target = WalkTarget::first(Vec2::new(10.0, 0.0), 4);
        assert_eq!(target.ground_gained(Vec3::new(5.0, 0.0, 0.0)), 0.0);
        target.leg = 1;
        target.start = Vec3::ZERO;
        assert_eq!(target.ground_gained(Vec3::new(4.0, 0.0, 0.0)), 4.0);
        target.gained = target.ground_gained(Vec3::new(4.0, 0.0, 0.0));
        target.leg = 2;
        target.start = Vec3::new(4.0, 0.0, 0.0);
        target.point = Vec2::new(4.0, 10.0);
        assert_eq!(target.ground_gained(Vec3::new(4.0, 0.0, 3.0)), 7.0);
        // Walking away from the new target gains nothing on that leg.
        assert_eq!(target.ground_gained(Vec3::new(4.0, 0.0, -3.0)), 4.0);
    }

    #[test]
    fn progress_counts_only_the_ground_gained() {
        let target = Vec2::new(10.0, 0.0);
        assert_eq!(progress(Vec3::ZERO, Vec3::new(6.0, 2.0, 0.0), target), 6.0);
        assert_eq!(progress(Vec3::ZERO, Vec3::new(-3.0, 0.0, 0.0), target), 0.0);
    }

    #[test]
    fn each_generation_aims_its_own_way_and_a_seed_repeats_it() {
        let directions: Vec<Vec2> = (0..40).map(|g| target_direction(5, g)).collect();
        assert!(directions.iter().all(|d| (d.length() - 1.0).abs() < 1e-5));
        assert_eq!(
            directions,
            (0..40).map(|g| target_direction(5, g)).collect::<Vec<_>>()
        );
        // They spread around the compass.
        let quadrants = directions
            .iter()
            .map(|d| (d.x > 0.0, d.y > 0.0))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(quadrants.len(), 4);
        let spawn = Vec2::new(-20.0, 30.0);
        let target = target_for(spawn, 5, 3, UserTarget(None));
        assert!(((target - spawn).length() - TARGET_DISTANCE).abs() < 1e-4);
        let user = Vec2::new(7.0, -2.0);
        assert_eq!(target_for(spawn, 5, 3, UserTarget(Some(user))), user);
    }
}
