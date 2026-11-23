//! Sensors: what each robot feels every fixed step, written into its brain's
//! input buffer. Physics state is read from avian's `Position`, `Rotation`
//! and velocities, which are current at every fixed step (a `Transform` or
//! `GlobalTransform` read here would be up to a frame stale).

use crate::core::config::Senses;
use crate::simulation::brain::RobotBrain;
use crate::simulation::clock::GenerationClock;
use crate::simulation::collision::GameLayer;
use crate::simulation::target::{WalkTarget, bearing};
use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_robot::{JointType, RobotBlueprint, SensorType};

/// Clock inputs: `sin` and `cos` of an oscillation at the robot's own
/// evolved frequency ([`RobotBrain::rhythm_hz`]), phase 0 when the
/// generation's robots spawn.
pub const NUM_CLOCK_INPUTS: usize = 2;
/// Each IMU contributes 6 values: the part's linear and angular acceleration
/// (world frame), see [`AgentImu`].
pub const INPUTS_PER_IMU: usize = 6;
/// Each touch sensor contributes 1 value (contact boolean).
pub const INPUTS_PER_TOUCH: usize = 1;
/// Each ultrasonic sensor contributes 1 value (proximity in [0, 1]).
pub const INPUTS_PER_ULTRASONIC: usize = 1;
/// Each hinge joint contributes 1 value: its angle across its range, -1 at
/// the bottom and +1 at the top (the brain's output scale for that joint).
pub const INPUTS_PER_JOINT: usize = 1;
/// Each prismatic joint contributes 1 value: its extension across its
/// range, on the same scale as a hinge's angle.
pub const INPUTS_PER_SLIDER: usize = 1;
/// In the target task, two more inputs: the sine and cosine of the target's
/// angle from the robot's heading (see [`bearing`]).
pub const NUM_BEARING_INPUTS: usize = 2;
/// With the tilt sense, two more: how far the body's forward and sideways
/// axes lean from level (see [`tilt`]).
pub const NUM_TILT_INPUTS: usize = 2;
/// Maximum range of the ultrasonic sensor (m). Nothing in range reads 0.
pub const ULTRASONIC_MAX_RANGE: f32 = 8.0;

/// How a body's sensors map onto its brain's inputs. Layout:
/// `[clock (2), imu (6 each), touch (1 each), ultrasonic (1 each),
/// hinge angle (1 each), slider extension (1 each), target bearing (2, in
/// the target task), tilt (2, with the tilt sense)]`. The senses beyond the
/// body's sensors come last, so that a direct brain, resized by adding or
/// dropping inputs at the end, keeps its sensors' meanings when they come
/// and go. Between the two senses it cannot: with tilt on, a task switch
/// hands the tilt's weights to the bearing or the bearing's to the tilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputLayout {
    pub imus: usize,
    pub touches: usize,
    pub ultrasonics: usize,
    pub hinges: usize,
    pub sliders: usize,
    /// Whether the brain reads a target's bearing.
    pub bearing: bool,
    /// Whether the brain reads its body's tilt.
    pub tilt: bool,
}

impl InputLayout {
    /// The layout a blueprint's sensors and hinges produce.
    pub fn of_blueprint(blueprint: &RobotBlueprint) -> Self {
        let mut layout = Self::default();
        for module in blueprint.modules.values() {
            for sensor in &module.sensors {
                match sensor.sensor_type {
                    SensorType::IMU => layout.imus += 1,
                    SensorType::Touch => layout.touches += 1,
                    SensorType::Ultrasonic => layout.ultrasonics += 1,
                    SensorType::Camera | SensorType::Lidar => {}
                }
            }
        }
        let count = |is: fn(&JointType) -> bool| {
            blueprint
                .joints
                .iter()
                .filter(|j| is(&j.joint_type))
                .count()
        };
        layout.hinges = count(|t| matches!(t, JointType::Hinge { .. }));
        layout.sliders = count(|t| matches!(t, JointType::Prismatic { .. }));
        layout
    }

    /// This layout with `senses`' inputs.
    pub fn with_senses(self, senses: Senses) -> Self {
        Self {
            bearing: senses.bearing,
            tilt: senses.tilt,
            ..self
        }
    }

    /// Total number of brain inputs.
    pub fn total(&self) -> usize {
        let tilt = if self.tilt { NUM_TILT_INPUTS } else { 0 };
        self.tilt_offset() + tilt
    }

    /// Where the tilt inputs start (if the brain reads them): after the
    /// bearing's.
    pub fn tilt_offset(&self) -> usize {
        let bearing = if self.bearing { NUM_BEARING_INPUTS } else { 0 };
        self.bearing_offset() + bearing
    }

    /// Where the bearing inputs start (if the brain reads them): after every
    /// sensor's.
    pub fn bearing_offset(&self) -> usize {
        self.slider_offset(self.sliders)
    }

    pub fn imu_offset(&self, index: usize) -> usize {
        NUM_CLOCK_INPUTS + index * INPUTS_PER_IMU
    }

    pub fn touch_offset(&self, index: usize) -> usize {
        self.imu_offset(self.imus) + index * INPUTS_PER_TOUCH
    }

    pub fn ultrasonic_offset(&self, index: usize) -> usize {
        self.touch_offset(self.touches) + index * INPUTS_PER_ULTRASONIC
    }

    pub fn joint_offset(&self, index: usize) -> usize {
        self.ultrasonic_offset(self.ultrasonics) + index * INPUTS_PER_JOINT
    }

    pub fn slider_offset(&self, index: usize) -> usize {
        self.joint_offset(self.hinges) + index * INPUTS_PER_SLIDER
    }
}

// Each sensor is its own entity (a child of the robot's root) naming the part
// it is mounted on, so a part can carry any number of sensors of one kind.

/// IMU on a part: its linear and angular acceleration in the world frame,
/// from the change in its velocities over one fixed step. It reads 0 while
/// the part moves steadily, carries no tilt, and spikes when the part is
/// jolted, as on every footfall.
///
/// It stays because it measured best. Over 24 to 32 seeds of 30
/// generations on flat ground (headless runs, NSGA-II, 64 robots), replacing it with the part's tilt and
/// angular velocity in its own frame lost 20 to 40% of the final fitness,
/// and feeding no IMU at all lost about 20%, so the jolts help. A likely
/// reason: fresh brains that react to them move jerkily and score near
/// zero, which favours decisive, clock-driven gaits from the start. (All of
/// this was measured under the lab's earlier fitness, which taxed every
/// change of a command, and not again since
/// [`crate::simulation::fitness`] stopped doing so.) Tilt
/// added beside it, as two more inputs ([`LabConfig::tilt_sense`]), did not
/// help on hills either (32 seeds).
///
/// [`LabConfig::tilt_sense`]: crate::core::config::LabConfig::tilt_sense
#[derive(Component)]
pub struct AgentImu {
    pub brain_entity: Entity,
    pub input_offset: usize,
    pub part: Entity,
    pub prev_linear_vel: Vec3,
    pub prev_angular_vel: Vec3,
}

/// Touch sensor on a part: 1.0 while the part touches the arena, else 0.0.
/// The part carries `CollidingEntities`. `prev_in_contact` lets the metrics
/// count contact/air transitions per foot.
#[derive(Component)]
pub struct AgentTouch {
    pub brain_entity: Entity,
    pub input_offset: usize,
    pub part: Entity,
    pub prev_in_contact: bool,
}

/// Ultrasonic distance sensor on a part. Casts a ray from its mount point
/// along the turtle's heading when it was mounted (the mount's local +Y) and
/// reads `1 - distance / ULTRASONIC_MAX_RANGE`, so nearer reads larger. Only
/// the arena is seen: robots ghost through each other and their own parts.
#[derive(Component)]
pub struct AgentUltrasonic {
    pub brain_entity: Entity,
    pub input_offset: usize,
    pub part: Entity,
    pub mount_position: Vec3,
    pub mount_rotation: Quat,
}

/// Whether a touch sensor's part is touching the arena.
pub fn in_contact(touch: &AgentTouch, contacts: &Query<&CollidingEntities>) -> bool {
    contacts.get(touch.part).is_ok_and(|c| !c.is_empty())
}

/// Joint-angle sensor on a hinge joint entity, reading the angle across the
/// joint's range `[min, max]`.
#[derive(Component)]
pub struct AgentJoint {
    pub brain_entity: Entity,
    pub input_offset: usize,
    pub min: f32,
    pub max: f32,
}

impl AgentJoint {
    /// `angle` mapped from `[min, max]` onto `[-1, 1]` (beyond the range when
    /// the joint is pushed past its limit).
    pub fn normalized(&self, angle: f32) -> f32 {
        let span = (self.max - self.min).max(f32::EPSILON);
        2.0 * (angle - self.min) / span - 1.0
    }
}

/// Extension sensor on a prismatic joint entity, reading how far the joint
/// has slid across its range `[min, max]`.
#[derive(Component)]
pub struct AgentSlider {
    pub brain_entity: Entity,
    pub input_offset: usize,
    pub min: f32,
    pub max: f32,
}

impl AgentSlider {
    /// `extension` mapped from `[min, max]` onto `[-1, 1]`.
    pub fn normalized(&self, extension: f32) -> f32 {
        let span = (self.max - self.min).max(f32::EPSILON);
        2.0 * (extension - self.min) / span - 1.0
    }
}

/// How far a prismatic joint has slid along its axis (m): the offset
/// between its two anchors along the axis on the first body, as avian's
/// motor and limits measure it.
pub fn slider_extension(
    joint: &PrismaticJoint,
    (position1, rotation1): (Vec3, Quat),
    (position2, rotation2): (Vec3, Quat),
) -> f32 {
    let anchor1 = position1 + rotation1 * joint.local_anchor1().unwrap_or(Vec3::ZERO);
    let anchor2 = position2 + rotation2 * joint.local_anchor2().unwrap_or(Vec3::ZERO);
    let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
    (anchor2 - anchor1).dot(rotation1 * basis1 * joint.slider_axis)
}

/// Signed angle (rad) of a revolute joint about its hinge axis, from the
/// rotations of its two bodies and the joint's local frames.
pub fn hinge_angle(joint: &RevoluteJoint, rot1: Quat, rot2: Quat) -> f32 {
    let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
    let basis2 = joint.local_basis2().unwrap_or(Quat::IDENTITY);
    let axis = joint.hinge_axis;
    let perp = if axis.x.abs() < 0.9 {
        axis.cross(Vec3::X).normalize()
    } else {
        axis.cross(Vec3::Y).normalize()
    };
    let a1 = (rot1 * basis1) * axis;
    let b1 = (rot1 * basis1) * perp;
    let b2 = (rot2 * basis2) * perp;
    b1.cross(b2).dot(a1).atan2(b1.dot(b2))
}

/// Reads every sensor into its brain's input buffer. Iterates sensor
/// archetypes linearly and writes by offset, rather than having each brain
/// look its sensors up.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn gather_sensor_inputs(
    mut brains: Query<&mut RobotBrain>,
    mut imus: Query<&mut AgentImu>,
    touches: Query<&AgentTouch>,
    ultrasonics: Query<&AgentUltrasonic>,
    hinges: Query<(&AgentJoint, &RevoluteJoint)>,
    sliders: Query<(&AgentSlider, &PrismaticJoint)>,
    parts: Query<(&Position, &Rotation, &LinearVelocity, &AngularVelocity)>,
    contacts: Query<&CollidingEntities>,
    targets: Query<(Entity, &WalkTarget)>,
    spatial_query: SpatialQuery,
    clock: Res<GenerationClock>,
    time: Res<Time>,
) {
    let elapsed = clock.elapsed_secs();
    let dt = time.delta_secs();
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };
    for mut brain in &mut brains {
        let phase = clock_phase(brain.rhythm_hz, elapsed);
        brain.sensor_inputs.fill(0.0);
        if brain.sensor_inputs.len() >= NUM_CLOCK_INPUTS {
            brain.sensor_inputs[0] = phase.sin();
            brain.sensor_inputs[1] = phase.cos();
        }
    }

    for mut imu in &mut imus {
        let Ok((_, _, lin_vel, ang_vel)) = parts.get(imu.part) else {
            continue;
        };
        let lin_accel = (lin_vel.0 - imu.prev_linear_vel) * inv_dt;
        let ang_accel = (ang_vel.0 - imu.prev_angular_vel) * inv_dt;
        imu.prev_linear_vel = lin_vel.0;
        imu.prev_angular_vel = ang_vel.0;
        if let Ok(mut brain) = brains.get_mut(imu.brain_entity)
            && let Some(slot) = brain
                .sensor_inputs
                .get_mut(imu.input_offset..imu.input_offset + INPUTS_PER_IMU)
        {
            slot.copy_from_slice(&[
                lin_accel.x,
                lin_accel.y,
                lin_accel.z,
                ang_accel.x,
                ang_accel.y,
                ang_accel.z,
            ]);
        }
    }

    for touch in &touches {
        let contact = if in_contact(touch, &contacts) {
            1.0
        } else {
            0.0
        };
        write_input(&mut brains, touch.brain_entity, touch.input_offset, contact);
    }

    let arena_only = SpatialQueryFilter::from_mask(GameLayer::Arena);
    for sensor in &ultrasonics {
        let Ok((position, rotation, _, _)) = parts.get(sensor.part) else {
            continue;
        };
        let origin = position.0 + rotation.0 * sensor.mount_position;
        let Ok(direction) = Dir3::new(rotation.0 * sensor.mount_rotation * Vec3::Y) else {
            continue;
        };
        let proximity = match spatial_query.cast_ray(
            origin,
            direction,
            ULTRASONIC_MAX_RANGE,
            true,
            &arena_only,
        ) {
            Some(hit) => (1.0 - hit.distance / ULTRASONIC_MAX_RANGE).clamp(0.0, 1.0),
            None => 0.0,
        };
        write_input(
            &mut brains,
            sensor.brain_entity,
            sensor.input_offset,
            proximity,
        );
    }

    for (sensor, joint) in &hinges {
        let (Ok((_, rot1, _, _)), Ok((_, rot2, _, _))) =
            (parts.get(joint.body1), parts.get(joint.body2))
        else {
            continue;
        };
        let angle = hinge_angle(joint, rot1.0, rot2.0);
        write_input(
            &mut brains,
            sensor.brain_entity,
            sensor.input_offset,
            sensor.normalized(angle),
        );
    }

    for (sensor, joint) in &sliders {
        let (Ok((p1, r1, _, _)), Ok((p2, r2, _, _))) =
            (parts.get(joint.body1), parts.get(joint.body2))
        else {
            continue;
        };
        let extension = slider_extension(joint, (p1.0, r1.0), (p2.0, r2.0));
        write_input(
            &mut brains,
            sensor.brain_entity,
            sensor.input_offset,
            sensor.normalized(extension),
        );
    }

    // The target task: the bearing from the brain's own part.
    for (body, target) in &targets {
        let Ok((position, rotation, _, _)) = parts.get(body) else {
            continue;
        };
        let reading = bearing(position.0, rotation.0, target.point);
        let Ok(mut brain) = brains.get_mut(body) else {
            continue;
        };
        if let Some(steering) = brain.steering.as_mut() {
            steering.turn = reading[0].atan2(reading[1]) / std::f32::consts::PI;
        }
        if let Some(slot) = brain
            .sensor_inputs
            .get_mut(target.input_offset..target.input_offset + NUM_BEARING_INPUTS)
        {
            slot.copy_from_slice(&reading);
        }
    }
}

/// How far the body leans: the vertical parts of its forward and sideways
/// axes (its x and z), 0 when level and 1 when standing on end.
pub fn tilt(rotation: Quat) -> [f32; 2] {
    [(rotation * Vec3::X).y, (rotation * Vec3::Z).y]
}

/// Writes each brain's tilt with the tilt sense, from the part that carries
/// it. Runs after `gather_sensor_inputs`, which clears the inputs.
pub fn sense_tilt(mut brains: Query<(&mut RobotBrain, &Rotation)>) {
    for (mut brain, rotation) in &mut brains {
        let Some(at) = brain.tilt_input else {
            continue;
        };
        if let Some(slot) = brain.sensor_inputs.get_mut(at..at + NUM_TILT_INPUTS) {
            slot.copy_from_slice(&tilt(rotation.0));
        }
    }
}

/// The clock's phase (rad) after `elapsed_secs` at `rhythm_hz`.
pub fn clock_phase(rhythm_hz: f32, elapsed_secs: f32) -> f32 {
    std::f32::consts::TAU * rhythm_hz * elapsed_secs
}

fn write_input(brains: &mut Query<&mut RobotBrain>, brain: Entity, offset: usize, value: f32) {
    if let Ok(mut brain) = brains.get_mut(brain)
        && let Some(slot) = brain.sensor_inputs.get_mut(offset)
    {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_turns_once_per_period_of_the_robots_rhythm() {
        // At 2 Hz the clock completes a turn every half second.
        assert!((clock_phase(2.0, 0.5) - std::f32::consts::TAU).abs() < 1e-5);
        assert!((clock_phase(2.0, 0.125).sin() - 1.0).abs() < 1e-5);
        assert_eq!(clock_phase(0.5, 0.0), 0.0);
    }

    #[test]
    fn tilt_reads_how_far_the_body_leans() {
        assert_eq!(tilt(Quat::IDENTITY), [0.0, 0.0]);
        // The spawn turn (over, about x) leaves a standing robot level.
        let standing = Quat::from_rotation_x(-std::f32::consts::PI);
        let [forward, side] = tilt(standing);
        assert!(forward.abs() < 1e-6 && side.abs() < 1e-6);
        // Nose up by 30 degrees: forward reads sin(30).
        let [forward, side] = tilt(Quat::from_rotation_z(std::f32::consts::FRAC_PI_6));
        assert!((forward - 0.5).abs() < 1e-5 && side.abs() < 1e-5);
    }

    #[test]
    fn the_layout_packs_sensor_kinds_in_order() {
        let layout = InputLayout {
            imus: 1,
            touches: 4,
            ultrasonics: 2,
            hinges: 8,
            sliders: 3,
            bearing: false,
            tilt: false,
        };
        assert_eq!(layout.imu_offset(0), 2);
        assert_eq!(layout.touch_offset(0), 8);
        assert_eq!(layout.ultrasonic_offset(0), 12);
        assert_eq!(layout.joint_offset(0), 14);
        assert_eq!(layout.slider_offset(0), 22);
        assert_eq!(layout.total(), 25);
        // The target task appends the bearing, leaving the rest in place.
        let steering = layout.with_senses(Senses::BEARING);
        assert_eq!(steering.bearing_offset(), 25);
        assert_eq!(steering.slider_offset(0), 22);
        assert_eq!(steering.total(), 27);
        assert_eq!(steering.with_senses(Senses::NONE), layout);
        // Tilt follows the bearing, or the sensors when there is none.
        let both = layout.with_senses(Senses {
            bearing: true,
            tilt: true,
        });
        assert_eq!((both.tilt_offset(), both.total()), (27, 29));
        let tilt = layout.with_senses(Senses {
            bearing: false,
            tilt: true,
        });
        assert_eq!((tilt.tilt_offset(), tilt.total()), (25, 27));
    }

    #[test]
    fn slider_extension_reads_the_offset_along_the_axis() {
        let joint = PrismaticJoint::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER)
            .with_slider_axis(Vec3::Y)
            .with_local_anchor1(Vec3::new(0.0, 0.5, 0.0))
            .with_local_anchor2(Vec3::new(0.0, -0.5, 0.0));
        let body1 = (Vec3::ZERO, Quat::IDENTITY);
        // Bodies placed so the anchors meet: no extension.
        assert!(
            slider_extension(&joint, body1, (Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY)).abs()
                < 1e-6
        );
        // Pulled 0.2 along the axis, and moved sideways (which does not count).
        let pulled = (Vec3::new(0.3, 1.2, 0.0), Quat::IDENTITY);
        assert!((slider_extension(&joint, body1, pulled) - 0.2).abs() < 1e-6);
        // The axis turns with the first body.
        let turned = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let along_turned_axis = (turned * Vec3::new(0.0, 1.3, 0.0), Quat::IDENTITY);
        let extension =
            slider_extension(&joint, (Vec3::ZERO, turned), (along_turned_axis.0, turned));
        assert!((extension - 0.3).abs() < 1e-5, "{extension}");
        let sensor = AgentSlider {
            brain_entity: Entity::PLACEHOLDER,
            input_offset: 0,
            min: -0.1,
            max: 0.3,
        };
        assert!((sensor.normalized(0.1)).abs() < 1e-6);
        assert!((sensor.normalized(0.3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hinge_angle_reads_a_rotation_about_the_hinge_axis() {
        let joint =
            RevoluteJoint::new(Entity::PLACEHOLDER, Entity::PLACEHOLDER).with_hinge_axis(Vec3::Z);
        let angle = hinge_angle(&joint, Quat::IDENTITY, Quat::from_rotation_z(0.4));
        assert!((angle.abs() - 0.4).abs() < 1e-5, "angle {angle}");
        assert!(hinge_angle(&joint, Quat::IDENTITY, Quat::IDENTITY).abs() < 1e-6);
    }
}
