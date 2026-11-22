//! Spawns a robot's physics body from its blueprint: one dynamic rigid body
//! per module, one avian joint per blueprint joint, and the sensor mounts.
//!
//! Nothing here draws anything. Each part carries a [`RobotPart`] naming its
//! shape and material, and the windowed app attaches meshes to it, so the
//! same body runs in a window and headless.

use crate::simulation::collision::robot_layers;
use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_robot::{
    AxisLimit, EndEffector, JointType, MaterialId, ModuleId, RobotBlueprint, RobotModule,
    SensorMount, SensorType, ShapePrimitive,
};

/// Root of one robot's entity tree; despawning it removes the whole robot.
#[derive(Component, Debug, Clone, Copy)]
pub struct RobotRoot {
    pub agent_index: usize,
}

/// One robot part (a blueprint module) and what the app should draw for it.
#[derive(Component, Debug, Clone, Copy)]
pub struct RobotPart {
    pub agent_index: usize,
    pub module_id: ModuleId,
    pub shape: ShapePrimitive,
    pub material: MaterialId,
}

/// The robot's end-effector frames, kept on its root for the app to mark.
#[derive(Component, Debug, Clone)]
pub struct EndEffectors(pub Vec<EndEffector>);

/// A spawned joint and the blueprint data the brain wiring needs.
#[derive(Debug, Clone)]
pub struct SpawnedJoint {
    pub entity: Entity,
    pub joint_type: JointType,
    /// The limit that configured the joint's range and motor, if any.
    pub limit: Option<AxisLimit>,
    /// The part the joint carries (each part hangs from one joint).
    pub child_module: ModuleId,
}

/// A sensor mounted on a spawned part.
#[derive(Debug, Clone)]
pub struct SpawnedSensor {
    pub part: Entity,
    pub mount: SensorMount,
}

/// Everything [`spawn_body`] created, for the caller to wire a brain to.
#[derive(Debug)]
pub struct SpawnedBody {
    pub root: Entity,
    /// Parts in `ModuleId` order; the first is the body that carries the brain.
    pub parts: Vec<(ModuleId, Entity)>,
    /// Joints in blueprint order.
    pub joints: Vec<SpawnedJoint>,
    /// Sensors in `ModuleId` order, then mount order within a part.
    pub sensors: Vec<SpawnedSensor>,
    /// Sum of the parts' masses (kg).
    pub total_mass: f32,
}

impl SpawnedBody {
    /// The part that carries the brain: the blueprint's first module.
    pub fn body(&self) -> Option<Entity> {
        self.parts.first().map(|&(_, entity)| entity)
    }

    pub fn sensors_of(&self, kind: SensorType) -> impl Iterator<Item = &SpawnedSensor> {
        self.sensors
            .iter()
            .filter(move |s| s.mount.sensor_type == kind)
    }
}

/// Spawns `blueprint` as agent `agent_index`, placed by `placement` (applied
/// on top of each module's rest-pose transform).
pub fn spawn_body(
    commands: &mut Commands,
    blueprint: &RobotBlueprint,
    agent_index: usize,
    placement: Transform,
) -> SpawnedBody {
    let root = commands
        .spawn((
            RobotRoot { agent_index },
            EndEffectors(blueprint.end_effectors.clone()),
            Name::new(format!("Robot {agent_index}")),
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();

    let modules = modules_in_order(blueprint);
    let mut parts = Vec::with_capacity(modules.len());
    let mut rest_rotations = bevy::platform::collections::HashMap::new();
    let mut sensors = Vec::new();
    let mut total_mass = 0.0;
    for (id, module) in modules {
        let (position, rotation) = module.transform;
        let world = placement * Transform::from_translation(position).with_rotation(rotation);
        let entity = commands
            .spawn((
                RobotPart {
                    agent_index,
                    module_id: id,
                    shape: module.shape,
                    material: module.material_id,
                },
                world,
                RigidBody::Dynamic,
                collider_for(&module.shape),
                MassPropertiesBundle::from_shape(&module.shape.to_bevy_primitive(), module.density),
                robot_layers(),
                ChildOf(root),
            ))
            .id();
        for mount in &module.sensors {
            sensors.push(SpawnedSensor {
                part: entity,
                mount: mount.clone(),
            });
        }
        total_mass += module.mass;
        rest_rotations.insert(id, (entity, rotation));
        parts.push((id, entity));
    }

    let mut joints = Vec::with_capacity(blueprint.joints.len());
    for def in &blueprint.joints {
        let (Some(&(parent, parent_rot)), Some(&(child, child_rot))) = (
            rest_rotations.get(&def.parent_id),
            rest_rotations.get(&def.child_id),
        ) else {
            warn!(
                "Agent {agent_index}: joint {} -> {} names a missing module",
                def.parent_id, def.child_id
            );
            continue;
        };
        // The joint keeps the two parts at their rest-pose relative rotation;
        // without it every joint would pull its parts to one world orientation.
        let rest_basis = child_rot.inverse() * parent_rot;
        let limit = def
            .joint_type
            .axis()
            .and_then(|axis| pick_limit(&def.limits, axis));
        // The child grows along its local +Y; a ball joint twists about that
        // direction, which the blueprint's limit axes share a frame with
        // (the parent's).
        let twist_axis = parent_rot.inverse() * child_rot * Vec3::Y;
        let entity = spawn_joint(
            commands,
            JointSpec {
                parent,
                child,
                anchor_parent: def.anchor_parent,
                anchor_child: def.anchor_child,
                rest_basis,
                joint_type: def.joint_type,
                limit,
                ball: ball_limits(&def.limits, twist_axis),
            },
        );
        commands.entity(entity).insert(ChildOf(root));
        joints.push(SpawnedJoint {
            entity,
            joint_type: def.joint_type,
            limit: limit.copied(),
            child_module: def.child_id,
        });
    }

    SpawnedBody {
        root,
        parts,
        joints,
        sensors,
        total_mass,
    }
}

/// The body's legs: the leg number of every part on one. A leg is the chain
/// of parts from a foot (a part nothing hangs from) up to, not including, the
/// first part that carries more than one (or the root): the skink's four
/// legs and its tail but not the trunk segment between its leg pairs. Legs
/// are numbered in the order of their feet's module ids. A joint is on the leg
/// of the part it carries.
pub fn legs(blueprint: &RobotBlueprint) -> bevy::platform::collections::HashMap<ModuleId, usize> {
    use bevy::platform::collections::HashMap;
    let mut children: HashMap<ModuleId, usize> = HashMap::default();
    let mut parent: HashMap<ModuleId, ModuleId> = HashMap::default();
    for joint in &blueprint.joints {
        *children.entry(joint.parent_id).or_default() += 1;
        parent.insert(joint.child_id, joint.parent_id);
    }
    let mut feet: Vec<ModuleId> = blueprint
        .modules
        .keys()
        .filter(|id| !children.contains_key(*id) && Some(**id) != blueprint.root_module)
        .copied()
        .collect();
    feet.sort_unstable();
    let mut leg_of = HashMap::default();
    for (leg, &foot) in feet.iter().enumerate() {
        let mut part = foot;
        loop {
            leg_of.insert(part, leg);
            let Some(&up) = parent.get(&part) else {
                break;
            };
            if Some(up) == blueprint.root_module || children.get(&up).copied().unwrap_or(0) > 1 {
                break;
            }
            part = up;
        }
    }
    leg_of
}

/// The blueprint's modules in `ModuleId` order: the order parts spawn in,
/// and so the order a brain's sensor inputs list their sensors.
pub fn modules_in_order(blueprint: &RobotBlueprint) -> Vec<(ModuleId, &RobotModule)> {
    let mut modules: Vec<(ModuleId, &RobotModule)> =
        blueprint.modules.iter().map(|(&id, m)| (id, m)).collect();
    modules.sort_unstable_by_key(|&(id, _)| id);
    modules
}

/// The collider matching a blueprint shape (box half-extents become full
/// extents; cylinders and capsules grow along local Y, as the turtle does).
pub fn collider_for(shape: &ShapePrimitive) -> Collider {
    match *shape {
        ShapePrimitive::Box(half) => Collider::cuboid(half.x * 2.0, half.y * 2.0, half.z * 2.0),
        ShapePrimitive::Cylinder { radius, height } => Collider::cylinder(radius, height),
        ShapePrimitive::Sphere(radius) => Collider::sphere(radius),
        ShapePrimitive::Capsule { radius, height } => Collider::capsule(radius, height),
    }
}

/// The per-axis limit that configures a single-axis joint: the LAST one whose
/// axis is parallel to the drive axis, else the last one at all.
///
/// `Jl` appends to a list the turtle carries down each branch, so a knee's
/// list holds its hip's limit first and its own last. `bevy_symbios` takes
/// the first, which gave every knee its hip's range: a range written for a
/// knee after its hip's never applied. The most recent `Jl` is the one the
/// morphology meant for the joint it precedes.
fn pick_limit(limits: &[AxisLimit], drive_axis: Vec3) -> Option<&AxisLimit> {
    let drive = drive_axis.normalize_or_zero();
    limits
        .iter()
        .rev()
        .find(|l| l.axis.normalize_or_zero().dot(drive).abs() > 0.999)
        .or_else(|| limits.last())
}

/// A ball joint's limits, from its blueprint limits.
#[derive(Debug, Clone, Copy, PartialEq)]
struct BallLimits {
    /// The axis it twists about, in the parent's frame.
    twist_axis: Vec3,
    /// How far it may twist, `(min, max)` (rad).
    twist: Option<(f32, f32)>,
    /// The half-angle of the cone it may swing in (rad).
    swing: Option<f32>,
}

/// A ball joint twisting about `twist_axis`: the last blueprint limit about
/// that axis bounds its twist, and the last about any other axis sets the
/// half-angle of its swing cone (the larger end of the limit's range). With
/// no limits it turns freely, as before.
fn ball_limits(limits: &[AxisLimit], twist_axis: Vec3) -> BallLimits {
    let twist_axis = twist_axis.normalize_or(Vec3::Y);
    let about_twist = |l: &&AxisLimit| l.axis.normalize_or_zero().dot(twist_axis).abs() > 0.999;
    BallLimits {
        twist_axis,
        twist: limits
            .iter()
            .rev()
            .find(about_twist)
            .map(|l| (l.min, l.max)),
        swing: limits
            .iter()
            .rev()
            .find(|l| !about_twist(l))
            .map(|l| l.min.abs().max(l.max.abs())),
    }
}

/// What [`spawn_joint`] needs to build one avian constraint.
struct JointSpec<'a> {
    parent: Entity,
    child: Entity,
    anchor_parent: Vec3,
    anchor_child: Vec3,
    rest_basis: Quat,
    joint_type: JointType,
    limit: Option<&'a AxisLimit>,
    ball: BallLimits,
}

/// Spawns the avian constraint for one blueprint joint. Hinges and sliders
/// with a limit get its range and a motor whose torque or force is capped at
/// the limit's effort; the brain sets the motor's targets every step.
fn spawn_joint(commands: &mut Commands, spec: JointSpec) -> Entity {
    let JointSpec {
        parent,
        child,
        anchor_parent,
        anchor_child,
        rest_basis,
        joint_type,
        limit,
        ball,
    } = spec;
    match joint_type {
        JointType::Hinge { axis } => {
            let mut joint = RevoluteJoint::new(parent, child)
                .with_local_anchor1(anchor_parent)
                .with_local_anchor2(anchor_child)
                .with_local_basis2(rest_basis)
                .with_hinge_axis(axis);
            if let Some(limit) = limit {
                joint = joint.with_angle_limits(limit.min, limit.max);
                joint.motor = AngularMotor::default().with_max_torque(limit.effort);
            }
            commands.spawn(joint).id()
        }
        JointType::Prismatic { axis } => {
            let mut joint = PrismaticJoint::new(parent, child)
                .with_local_anchor1(anchor_parent)
                .with_local_anchor2(anchor_child)
                .with_local_basis2(rest_basis)
                .with_slider_axis(axis);
            if let Some(limit) = limit {
                joint = joint.with_limits(limit.min, limit.max);
                joint.motor = LinearMotor::default().with_max_force(limit.effort);
            }
            commands.spawn(joint).id()
        }
        JointType::Ball => {
            let mut joint = SphericalJoint::new(parent, child)
                .with_local_anchor1(anchor_parent)
                .with_local_anchor2(anchor_child)
                .with_local_basis2(rest_basis);
            joint.twist_axis = ball.twist_axis;
            if let Some((min, max)) = ball.twist {
                joint = joint.with_twist_limits(min, max);
            }
            // avian measures swing as an unsigned angle from rest.
            if let Some(half_angle) = ball.swing {
                joint = joint.with_swing_limits(0.0, half_angle);
            }
            commands.spawn(joint).id()
        }
        // avian has no helical constraint, so a screw is welded like a fixed
        // joint; the morphology still spawns in one piece.
        JointType::Fixed | JointType::Screw { .. } => commands
            .spawn(
                FixedJoint::new(parent, child)
                    .with_local_anchor1(anchor_parent)
                    .with_local_anchor2(anchor_child)
                    .with_local_basis2(rest_basis),
            )
            .id(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::{self, CRAWLER, HEXAPOD};
    use crate::core::presets::MORPHOLOGY_POOL;
    use crate::simulation::lifecycle::derive_blueprint;
    use symbios_robot::SensorType;

    /// Driven joints (hinges and sliders) per leg.
    fn driven_per_leg(blueprint: &RobotBlueprint) -> Vec<usize> {
        let leg_of = legs(blueprint);
        let mut counts = vec![0; leg_of.values().max().map_or(0, |&n| n + 1)];
        for joint in &blueprint.joints {
            let driven = matches!(
                joint.joint_type,
                JointType::Hinge { .. } | JointType::Prismatic { .. }
            );
            if let (true, Some(&leg)) = (driven, leg_of.get(&joint.child_id)) {
                counts[leg] += 1;
            }
        }
        counts
    }

    /// The hexapod's 18 driven joints are its six legs' three each.
    #[test]
    fn a_hexapod_has_six_legs_of_three_joints() {
        let hexapod = derive_blueprint(HEXAPOD, 1, 3).expect("derives");
        assert_eq!(hexapod.num_outputs, 18);
        assert_eq!(driven_per_leg(&hexapod.blueprint), vec![3; 6]);
    }

    /// Every body's feet (its touch sensors) end legs that a driven joint
    /// moves, so breaking any of them lames the robot.
    #[test]
    fn every_foot_ends_a_driven_leg() {
        for (i, source) in MORPHOLOGY_POOL.iter().chain(fixtures::ALL).enumerate() {
            let body = derive_blueprint(source, 1, 3).expect("derives").blueprint;
            let leg_of = legs(&body);
            let driven = driven_per_leg(&body);
            for (id, module) in &body.modules {
                if module
                    .sensors
                    .iter()
                    .any(|s| s.sensor_type == SensorType::Touch)
                {
                    let leg = leg_of
                        .get(id)
                        .unwrap_or_else(|| panic!("body {i}: foot {id} on no leg"));
                    assert!(driven[*leg] > 0, "body {i}: foot {id}'s leg drives nothing");
                }
            }
        }
    }

    fn limit(axis: Vec3, min: f32) -> AxisLimit {
        AxisLimit {
            axis,
            min,
            max: 1.0,
            effort: 100.0,
            velocity: 2.0,
        }
    }

    #[test]
    fn pick_limit_takes_the_last_parallel_limit() {
        // A knee's list: its hip's limit first, its own last.
        let limits = [
            limit(Vec3::X, -0.6),
            limit(Vec3::Y, -0.1),
            limit(Vec3::X, -0.8),
        ];
        assert_eq!(pick_limit(&limits, Vec3::X).map(|l| l.min), Some(-0.8));
        assert_eq!(pick_limit(&limits, Vec3::NEG_X).map(|l| l.min), Some(-0.8));
    }

    #[test]
    fn a_ball_joint_twists_about_its_growth_axis_and_swings_in_a_cone() {
        let limits = [
            limit(Vec3::X, -0.2),
            limit(Vec3::Y, -0.3),
            AxisLimit {
                max: 0.5,
                ..limit(Vec3::Z, -0.4)
            },
        ];
        let ball = ball_limits(&limits, Vec3::Y);
        assert_eq!(ball.twist, Some((-0.3, 1.0)));
        // The last limit off the twist axis, its larger end.
        assert_eq!(ball.swing, Some(0.5));
        assert_eq!(ball_limits(&limits[..1], Vec3::Y).twist, None);
        let free = ball_limits(&[], Vec3::Z);
        assert_eq!(
            (free.twist, free.swing, free.twist_axis),
            (None, None, Vec3::Z)
        );
    }

    #[test]
    fn pick_limit_falls_back_to_the_last_limit() {
        let limits = [limit(Vec3::Y, -0.1), limit(Vec3::Y, -0.2)];
        assert_eq!(pick_limit(&limits, Vec3::Z).map(|l| l.min), Some(-0.2));
        assert!(pick_limit(&[], Vec3::Z).is_none());
    }

    #[test]
    fn a_knee_gets_its_own_range() {
        // The Crawler: hips -0.5..0.5, knees -1.0..0.2.
        let derived = derive_blueprint(CRAWLER, 1, 3).expect("crawler derives");
        let mut ranges: Vec<(f32, f32)> = derived
            .blueprint
            .joints
            .iter()
            .filter_map(|j| j.joint_type.axis().and_then(|a| pick_limit(&j.limits, a)))
            .map(|l| (l.min, l.max))
            .collect();
        ranges.sort_by(|a, b| a.0.total_cmp(&b.0));
        ranges.dedup();
        assert!(
            ranges.contains(&(-1.0, 0.2)),
            "no knee range among {ranges:?}"
        );
        assert!(
            ranges.contains(&(-0.5, 0.5)),
            "no hip range among {ranges:?}"
        );
    }
}
