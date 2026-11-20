//! The target task: each robot walks toward a target, reads its bearing, and
//! scores the progress it makes toward it. Each generation draws one
//! direction from the run's seed and the generation, and every robot's
//! target lies [`TARGET_DISTANCE`] from its spawn point that way: the robots
//! of a generation compete on equal footing, and successive generations
//! steer every way. In the windowed lab the user can place one target for
//! every robot instead.

use crate::simulation::rng::generation_seed;
use bevy::prelude::*;

/// How far (m) from its spawn point a robot's target lies.
pub const TARGET_DISTANCE: f32 = 10.0;
/// Tells the targets' seeds apart from the grounds'.
const TARGET_STREAM: u64 = 0x7A26_E7D1_5EED_0001;

/// A robot's target (world x and z), and where its brain reads the bearing.
/// Lives on the body part, next to the brain.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct WalkTarget {
    pub point: Vec2,
    pub input_offset: usize,
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
