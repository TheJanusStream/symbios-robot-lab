//! Turns the population's agents into robots: a physics body per agent, its
//! sensors and joints wired to a compiled brain, and the generation clock
//! started.

use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::body::{SpawnedBody, SpawnedJoint, spawn_body};
use crate::simulation::brain::{
    DEFAULT_HINGE_RANGE, DEFAULT_SLIDER_RANGE, DrivenJoint, MAX_MOTOR_TORQUE, MAX_PRISMATIC_FORCE,
    RobotBrain, compile_brain,
};
use crate::simulation::clock::GenerationClock;
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::metrics::{AgentMetrics, FitnessBaseline};
use crate::simulation::sensors::{
    AgentImu, AgentJoint, AgentSlider, AgentTouch, AgentUltrasonic, InputLayout,
};
use avian3d::prelude::CollidingEntities;
use bevy::prelude::*;
use symbios_robot::{JointType, RobotBlueprint, SensorType};

/// Distance between neighbouring robots on the spawn grid (m).
const SPAWN_SPACING: f32 = 10.0;

/// Spawns a robot for every agent that has none, whenever the population is
/// replaced, and starts the generation clock.
pub fn spawn_agents(
    mut commands: Commands,
    mut population: ResMut<Population>,
    mut clock: ResMut<GenerationClock>,
    config: Res<LabConfig>,
    time: Res<Time<Fixed>>,
) {
    if !population.is_changed() || population.agents.is_empty() {
        return;
    }
    let count = population.agents.len();
    let cols = (count as f32).sqrt().ceil() as usize;
    let offset = cols as f32 * SPAWN_SPACING / 2.0;

    for (index, agent) in population.agents.iter_mut().enumerate() {
        if agent.root_entity.is_some() || agent.exploded {
            continue;
        }
        let blueprint = match agent.cached_blueprint.take() {
            Some(blueprint) => blueprint,
            None => match derive_blueprint(
                &agent.genotype.morphology,
                agent.genotype.lsystem_seed,
                agent.genotype.lsystem_iterations,
            ) {
                Some(derived) => derived.blueprint,
                None => {
                    warn!("Agent {index}: its morphology builds no body");
                    continue;
                }
            },
        };
        let network = match compile_brain(&agent.genotype, &blueprint) {
            Ok(network) => network,
            Err(err) => {
                warn!("Agent {index}: failed to compile its brain: {err}");
                continue;
            }
        };

        let grid = Vec2::new(
            (index % cols) as f32 * SPAWN_SPACING - offset,
            (index / cols) as f32 * SPAWN_SPACING - offset,
        );
        let body = spawn_body(
            &mut commands,
            &blueprint,
            index,
            placement(&blueprint, grid),
        );
        let Some(body_entity) = body.body() else {
            commands.entity(body.root).despawn();
            continue;
        };
        let wiring = wire_sensors_and_joints(&mut commands, &body, &blueprint, body_entity);
        debug!(
            "Spawning agent {index}: {} parts, {} hinges, {} sliders, {:?}",
            body.parts.len(),
            wiring.hinge_joints.len(),
            wiring.prismatic_joints.len(),
            InputLayout::of_blueprint(&blueprint),
        );

        agent.num_motors = wiring.hinge_joints.len() + wiring.prismatic_joints.len();
        agent.mass = body.total_mass;
        agent.root_entity = Some(body.root);
        agent.body_entity = Some(body_entity);

        let metrics = AgentMetrics::new(
            body.total_mass,
            wiring.hinge_joints.len(),
            agent.num_motors,
            body.sensors_of(SensorType::Touch).count(),
        );
        let num_inputs = network.num_inputs();
        let parts = body.parts.iter().skip(1).map(|&(_, e)| e).collect();
        commands.entity(body_entity).insert((
            RobotBrain {
                scratchpad: network.create_scratchpad(),
                outputs: vec![0.0; network.num_outputs()],
                network,
                agent_index: index,
                sensor_inputs: vec![0.0; num_inputs],
                hinge_joints: wiring.hinge_joints,
                prismatic_joints: wiring.prismatic_joints,
                parts,
            },
            metrics,
            FitnessBaseline::default(),
        ));
    }

    if !clock.is_running() {
        clock.start(config.generation_duration, time.timestep().as_secs_f32());
        info!(
            "Generation {} started ({:.0}s)",
            population.generation, config.generation_duration
        );
    }
}

/// How every robot is turned when it spawns: over, so its legs point down
/// (the turtle grows bodies along +Y).
pub fn spawn_rotation() -> Quat {
    Quat::from_rotation_x(-std::f32::consts::PI)
}

/// Where a robot is placed: turned by [`spawn_rotation`], lifted so its
/// lowest point rests on the floor, at `grid` on the floor plane.
fn placement(blueprint: &RobotBlueprint, grid: Vec2) -> Transform {
    let rotation = spawn_rotation();
    let lowest = blueprint.aabb(rotation).min.y;
    Transform::from_xyz(grid.x, -lowest, grid.y).with_rotation(rotation)
}

/// The driven joints, in the order the brain's outputs drive them.
struct Wiring {
    hinge_joints: Vec<DrivenJoint>,
    prismatic_joints: Vec<DrivenJoint>,
}

/// A driven joint over its limit's range, or `default_range` either side of
/// rest when the blueprint gave it none, with its motor held to the
/// limit's effort up to `max_effort`. The effort is a mutable constant in
/// the morphology, so motor strength evolves; the cap keeps a mutation from
/// building rocket legs.
fn driven(joint: &SpawnedJoint, default_range: f32, max_effort: f32) -> DrivenJoint {
    let (min, max) = joint
        .limit
        .map_or((-default_range, default_range), |l| (l.min, l.max));
    DrivenJoint {
        entity: joint.entity,
        min,
        max,
        max_effort: joint
            .limit
            .map_or(max_effort, |l| l.effort.clamp(0.0, max_effort)),
    }
}

/// Attaches each sensor and driven joint's component, pointing at the brain
/// on `brain` and at the input it fills.
fn wire_sensors_and_joints(
    commands: &mut Commands,
    body: &SpawnedBody,
    blueprint: &RobotBlueprint,
    brain: Entity,
) -> Wiring {
    let layout = InputLayout::of_blueprint(blueprint);
    // Each sensor is its own entity under the robot's root, naming its part,
    // so a part can carry several sensors of one kind.
    for (i, sensor) in body.sensors_of(SensorType::IMU).enumerate() {
        commands.spawn((
            AgentImu {
                brain_entity: brain,
                input_offset: layout.imu_offset(i),
                part: sensor.part,
                prev_linear_vel: Vec3::ZERO,
                prev_angular_vel: Vec3::ZERO,
            },
            ChildOf(body.root),
        ));
    }
    for (i, sensor) in body.sensors_of(SensorType::Touch).enumerate() {
        commands
            .entity(sensor.part)
            .insert(CollidingEntities::default());
        commands.spawn((
            AgentTouch {
                brain_entity: brain,
                input_offset: layout.touch_offset(i),
                part: sensor.part,
                prev_in_contact: false,
            },
            ChildOf(body.root),
        ));
    }
    for (i, sensor) in body.sensors_of(SensorType::Ultrasonic).enumerate() {
        commands.spawn((
            AgentUltrasonic {
                brain_entity: brain,
                input_offset: layout.ultrasonic_offset(i),
                part: sensor.part,
                mount_position: sensor.mount.local_position,
                mount_rotation: sensor.mount.local_rotation,
            },
            ChildOf(body.root),
        ));
    }

    let mut wiring = Wiring {
        hinge_joints: Vec::new(),
        prismatic_joints: Vec::new(),
    };
    for joint in &body.joints {
        match joint.joint_type {
            JointType::Hinge { .. } => {
                let hinge = driven(joint, DEFAULT_HINGE_RANGE, MAX_MOTOR_TORQUE);
                commands.entity(joint.entity).insert(AgentJoint {
                    brain_entity: brain,
                    input_offset: layout.joint_offset(wiring.hinge_joints.len()),
                    min: hinge.min,
                    max: hinge.max,
                });
                wiring.hinge_joints.push(hinge);
            }
            JointType::Prismatic { .. } => {
                let slider = driven(joint, DEFAULT_SLIDER_RANGE, MAX_PRISMATIC_FORCE);
                commands.entity(joint.entity).insert(AgentSlider {
                    brain_entity: brain,
                    input_offset: layout.slider_offset(wiring.prismatic_joints.len()),
                    min: slider.min,
                    max: slider.max,
                });
                wiring.prismatic_joints.push(slider);
            }
            JointType::Fixed | JointType::Ball | JointType::Screw { .. } => {}
        }
    }
    wiring
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbios_robot::AxisLimit;

    fn joint(limit: Option<AxisLimit>) -> SpawnedJoint {
        SpawnedJoint {
            entity: Entity::PLACEHOLDER,
            joint_type: JointType::Hinge { axis: Vec3::X },
            limit,
        }
    }

    fn limit(effort: f32) -> AxisLimit {
        AxisLimit {
            axis: Vec3::X,
            min: -0.4,
            max: 0.6,
            effort,
            velocity: 5.0,
        }
    }

    #[test]
    fn a_driven_joint_takes_its_limits_range_and_effort_up_to_the_cap() {
        let weak = driven(&joint(Some(limit(300.0))), 0.1, 5000.0);
        assert_eq!((weak.min, weak.max, weak.max_effort), (-0.4, 0.6, 300.0));
        let mutated = driven(&joint(Some(limit(9e6))), 0.1, 5000.0);
        assert_eq!(mutated.max_effort, 5000.0);
        let negative = driven(&joint(Some(limit(-20.0))), 0.1, 5000.0);
        assert_eq!(negative.max_effort, 0.0);
        let unlimited = driven(&joint(None), 0.1, 5000.0);
        assert_eq!(
            (unlimited.min, unlimited.max, unlimited.max_effort),
            (-0.1, 0.1, 5000.0)
        );
    }
}
