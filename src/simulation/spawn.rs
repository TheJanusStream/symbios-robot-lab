//! Turns the population's agents into robots: a physics body per agent, its
//! sensors and joints wired to a compiled brain, and the generation clock
//! started.

use crate::core::config::{LabConfig, Task};
use crate::core::descriptor::BodyTraits;
use crate::core::genome::AgentGenotype;
use crate::core::population::Population;
use crate::simulation::body::{SpawnedBody, SpawnedJoint, legs, spawn_body};
use crate::simulation::brain::{
    DEFAULT_HINGE_RANGE, DEFAULT_SLIDER_RANGE, DrivenJoint, MAX_MOTOR_TORQUE, MAX_PRISMATIC_FORCE,
    RobotBrain, compile_brain,
};
use crate::simulation::clock::GenerationClock;
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::metrics::{AgentMetrics, FitnessBaseline};
use crate::simulation::rng::{LabRng, generation_seed};
use crate::simulation::sensors::{
    AgentImu, AgentJoint, AgentSlider, AgentTouch, AgentUltrasonic, InputLayout,
};
use crate::simulation::target::{UserTarget, WalkTarget, target_for};
use crate::simulation::terrain::Ground;
use avian3d::prelude::CollidingEntities;
use bevy::prelude::*;
use symbios_robot::{JointType, RobotBlueprint, SensorType};

/// Distance between neighbouring robots on the spawn grid (m).
const SPAWN_SPACING: f32 = 10.0;

/// Tells leg damage's draws apart from the grounds' and targets'.
const DAMAGE_STREAM: u64 = 0xDA3A_6E00_0000_0001;

/// The draw that picks which of robot `index`'s legs breaks in a generation,
/// under leg damage.
pub fn damage_draw(run_seed: u64, generation: usize, index: usize) -> u64 {
    let robot = (index as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407);
    generation_seed(run_seed ^ robot, generation, DAMAGE_STREAM)
}

/// The brain index of a robot walking outside the population, such as the
/// hall of fame's showcase: no agent has it, so nothing scores the robot.
pub const SHOWCASE_INDEX: usize = usize::MAX;

/// A robot to spawn.
pub struct RobotSpawn<'a> {
    pub genotype: &'a AgentGenotype,
    pub blueprint: &'a RobotBlueprint,
    /// The agent's index, or [`SHOWCASE_INDEX`].
    pub index: usize,
    /// Where on the ground (world x and z).
    pub grid: Vec2,
    /// The task its brain is compiled for.
    pub task: Task,
    /// Its target (world x and z), in the target task.
    pub target: Option<Vec2>,
    /// A draw that picks one of its legs to break, under leg damage.
    pub damage: Option<u64>,
}

/// What spawning a robot made.
pub struct SpawnedRobot {
    pub root: Entity,
    /// The part that carries the brain.
    pub body: Entity,
    pub num_motors: usize,
    pub hinges: usize,
    pub touches: usize,
    pub total_mass: f32,
}

/// Spawns a robot standing on `ground`: its body, its sensors and driven
/// joints wired to a brain compiled for the task, and its target if it has
/// one. `extra` adds components to the brain's part along with the brain.
/// Fails, spawning nothing, if the brain does not compile or the body has
/// no parts.
pub fn spawn_robot<B: Bundle>(
    commands: &mut Commands,
    robot: RobotSpawn,
    ground: &Ground,
    extra: impl FnOnce(&SpawnedRobot) -> B,
) -> Result<SpawnedRobot, String> {
    let network = compile_brain(robot.genotype, robot.blueprint, robot.task)
        .map_err(|err| format!("failed to compile its brain: {err}"))?;
    let body = spawn_body(
        commands,
        robot.blueprint,
        robot.index,
        placement(robot.blueprint, robot.grid, ground),
    );
    let Some(body_entity) = body.body() else {
        commands.entity(body.root).despawn();
        return Err("its body has no parts".to_string());
    };
    let mut wiring = wire_sensors_and_joints(commands, &body, robot.blueprint, body_entity);
    if let Some(draw) = robot.damage {
        wiring.break_a_leg(draw);
    }
    debug!(
        "Spawning robot {}: {} parts, {} hinges, {} sliders, {:?}",
        robot.index,
        body.parts.len(),
        wiring.hinge_joints.len(),
        wiring.prismatic_joints.len(),
        InputLayout::of_blueprint(robot.blueprint),
    );
    let spawned = SpawnedRobot {
        root: body.root,
        body: body_entity,
        num_motors: wiring.hinge_joints.len() + wiring.prismatic_joints.len(),
        hinges: wiring.hinge_joints.len(),
        touches: body.sensors_of(SensorType::Touch).count(),
        total_mass: body.total_mass,
    };
    let num_inputs = network.num_inputs();
    let parts = body.parts.iter().skip(1).map(|&(_, e)| e).collect();
    commands.entity(body_entity).insert((
        RobotBrain {
            scratchpad: network.create_scratchpad(),
            outputs: vec![0.0; network.num_outputs()],
            network,
            agent_index: robot.index,
            sensor_inputs: vec![0.0; num_inputs],
            hinge_joints: wiring.hinge_joints,
            prismatic_joints: wiring.prismatic_joints,
            parts,
            rhythm_hz: robot.genotype.rhythm_hz,
        },
        extra(&spawned),
    ));
    if let Some(point) = robot.target {
        commands.entity(body_entity).insert(WalkTarget {
            point,
            input_offset: InputLayout::of_blueprint(robot.blueprint).bearing_offset(),
        });
    }
    Ok(spawned)
}

/// Spawns a robot for every agent that has none, whenever the population is
/// replaced, and starts the generation clock.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn spawn_agents(
    mut commands: Commands,
    mut population: ResMut<Population>,
    mut clock: ResMut<GenerationClock>,
    config: Res<LabConfig>,
    time: Res<Time<Fixed>>,
    ground: Res<Ground>,
    rng: Res<LabRng>,
    user_target: Res<UserTarget>,
) {
    if !population.is_changed() || population.agents.is_empty() {
        return;
    }
    let count = population.agents.len();
    let cols = (count as f32).sqrt().ceil() as usize;
    let offset = cols as f32 * SPAWN_SPACING / 2.0;
    let generation = population.generation;

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
                Some(derived) => {
                    agent.body_traits = Some(BodyTraits::of(&derived.blueprint));
                    derived.blueprint
                }
                None => {
                    warn!("Agent {index}: its morphology builds no body");
                    continue;
                }
            },
        };
        let grid = Vec2::new(
            (index % cols) as f32 * SPAWN_SPACING - offset,
            (index / cols) as f32 * SPAWN_SPACING - offset,
        );
        let robot = RobotSpawn {
            genotype: &agent.genotype,
            blueprint: &blueprint,
            index,
            grid,
            task: config.task,
            target: config
                .task
                .steers()
                .then(|| target_for(grid, rng.seed(), generation, *user_target)),
            damage: config
                .leg_damage
                .then(|| damage_draw(rng.seed(), generation, index)),
        };
        let spawned = spawn_robot(&mut commands, robot, &ground, |robot| {
            let metrics = AgentMetrics::new(
                robot.total_mass,
                robot.hinges,
                robot.num_motors,
                robot.touches,
            );
            (metrics, FitnessBaseline::default())
        });
        match spawned {
            Ok(robot) => {
                agent.num_motors = robot.num_motors;
                agent.mass = robot.total_mass;
                agent.root_entity = Some(robot.root);
                agent.body_entity = Some(robot.body);
            }
            Err(err) => warn!("Agent {index}: {err}"),
        }
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

/// Where a robot is placed: turned by [`spawn_rotation`], at `grid` on the
/// floor plane, lifted so its lowest point rests on the highest ground under
/// it (the floor, or a patch of terrain).
fn placement(blueprint: &RobotBlueprint, grid: Vec2, ground: &Ground) -> Transform {
    let rotation = spawn_rotation();
    let aabb = blueprint.aabb(rotation);
    let footprint_min = grid + Vec2::new(aabb.min.x, aabb.min.z);
    let footprint_max = grid + Vec2::new(aabb.max.x, aabb.max.z);
    let floor = ground.highest_under(footprint_min, footprint_max);
    Transform::from_xyz(grid.x, floor - aabb.min.y, grid.y).with_rotation(rotation)
}

/// The driven joints, in the order the brain's outputs drive them.
struct Wiring {
    hinge_joints: Vec<DrivenJoint>,
    prismatic_joints: Vec<DrivenJoint>,
}

impl Wiring {
    /// Breaks the leg `draw` picks among those with a driven joint (none if
    /// the body has no legs).
    fn break_a_leg(&mut self, draw: u64) {
        let mut legs: Vec<usize> = self
            .hinge_joints
            .iter()
            .chain(&self.prismatic_joints)
            .filter_map(|joint| joint.leg)
            .collect();
        legs.sort_unstable();
        legs.dedup();
        if legs.is_empty() {
            return;
        }
        let leg = legs[(draw % legs.len() as u64) as usize];
        for joint in self
            .hinge_joints
            .iter_mut()
            .chain(&mut self.prismatic_joints)
        {
            joint.broken |= joint.leg == Some(leg);
        }
    }
}

/// A driven joint over its limit's range, or `default_range` either side of
/// rest when the blueprint gave it none, with its motor held to the
/// limit's effort up to `max_effort`. The effort is a mutable constant in
/// the morphology, so motor strength evolves; the cap keeps a mutation from
/// building rocket legs.
fn driven(
    joint: &SpawnedJoint,
    default_range: f32,
    max_effort: f32,
    leg: Option<usize>,
) -> DrivenJoint {
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
        leg,
        broken: false,
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
    let leg_of = legs(blueprint);
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
                let leg = leg_of.get(&joint.child_module).copied();
                let hinge = driven(joint, DEFAULT_HINGE_RANGE, MAX_MOTOR_TORQUE, leg);
                commands.entity(joint.entity).insert(AgentJoint {
                    brain_entity: brain,
                    input_offset: layout.joint_offset(wiring.hinge_joints.len()),
                    min: hinge.min,
                    max: hinge.max,
                });
                wiring.hinge_joints.push(hinge);
            }
            JointType::Prismatic { .. } => {
                let leg = leg_of.get(&joint.child_module).copied();
                let slider = driven(joint, DEFAULT_SLIDER_RANGE, MAX_PRISMATIC_FORCE, leg);
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
            child_module: 1,
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
        let driven = |limit, leg| driven(&joint(limit), 0.1, 5000.0, leg);
        let weak = driven(Some(limit(300.0)), Some(2));
        assert_eq!((weak.min, weak.max, weak.max_effort), (-0.4, 0.6, 300.0));
        assert_eq!((weak.leg, weak.broken), (Some(2), false));
        let mutated = driven(Some(limit(9e6)), None);
        assert_eq!(mutated.max_effort, 5000.0);
        let negative = driven(Some(limit(-20.0)), None);
        assert_eq!(negative.max_effort, 0.0);
        let unlimited = driven(None, None);
        assert_eq!(
            (unlimited.min, unlimited.max, unlimited.max_effort),
            (-0.1, 0.1, 5000.0)
        );
    }

    #[test]
    fn leg_damage_breaks_one_whole_leg() {
        let joint_on = |leg| driven(&joint(Some(limit(300.0))), 0.1, 5000.0, leg);
        let mut wiring = Wiring {
            hinge_joints: vec![
                joint_on(Some(0)),
                joint_on(Some(0)),
                joint_on(Some(1)),
                joint_on(None),
            ],
            prismatic_joints: vec![joint_on(Some(1))],
        };
        // Draw 3 picks leg 3 % 2 = 1 of legs [0, 1]: its hinge and its slider.
        wiring.break_a_leg(3);
        let broken: Vec<bool> = wiring.hinge_joints.iter().map(|j| j.broken).collect();
        assert_eq!(broken, vec![false, false, true, false]);
        assert!(wiring.prismatic_joints[0].broken);
        let mut legless = Wiring {
            hinge_joints: vec![joint_on(None)],
            prismatic_joints: Vec::new(),
        };
        legless.break_a_leg(7);
        assert!(!legless.hinge_joints[0].broken);
        // Each robot of a generation draws its own leg.
        let draws: std::collections::BTreeSet<u64> =
            (0..64).map(|i| damage_draw(5, 2, i) % 6).collect();
        assert!(draws.len() > 3);
        assert_eq!(damage_draw(5, 2, 9), damage_draw(5, 2, 9));
    }
}
