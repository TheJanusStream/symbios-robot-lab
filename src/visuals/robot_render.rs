use crate::core::population::Population;
use crate::simulation::brain::{AgentImu, AgentJoint, AgentModule, AgentTouch, RobotBrain};
use crate::simulation::brain::{INPUTS_PER_IMU, NUM_CLOCK_INPUTS};
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::scoring::FitnessBaseline;
use bevy::prelude::*;
use bevy_symbios::materials::{MaterialPalette, setup_material_assets};
use bevy_symbios::robot::spawn_robot;
use symbios_neat::CppnEvaluator;
use symbios_robot::{JointType, RobotBlueprint, SensorType};

pub fn setup_materials(
    commands: Commands,
    materials: ResMut<Assets<StandardMaterial>>,
    images: ResMut<Assets<Image>>,
) {
    setup_material_assets(commands, materials, images);
}

/// Reactively spawns robots when the Population resource changes.
/// Attaches RobotBrain (with entity refs) and sensor/joint components.
/// Uses cached blueprints when available to avoid redundant L-System parsing.
pub fn spawn_agents(
    mut commands: Commands,
    mut population: ResMut<Population>,
    palette: Res<MaterialPalette>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !population.is_changed() || population.agents.is_empty() {
        return;
    }

    let pop_size = population.agents.len();
    let cols = (pop_size as f32).sqrt().ceil() as usize;
    let spacing = 5.0;
    let offset = (cols as f32 * spacing) / 2.0;

    for (i, agent) in population.agents.iter_mut().enumerate() {
        if agent.root_entity.is_some() {
            continue;
        }

        // Use cached blueprint or derive fresh from L-System morphology.
        let blueprint = match agent.cached_blueprint.take() {
            Some(bp) => bp,
            None => match derive_blueprint(&agent.genotype.morphology) {
                Some((bp, _, _)) => bp,
                None => {
                    warn!("Agent {}: failed to parse morphology", i);
                    continue;
                }
            },
        };

        let (num_modules, num_hinges, num_imus, num_touches) = count_blueprint_stats(&blueprint);

        agent.num_motors = num_hinges;

        info!(
            "Spawning Agent {}: {} modules, {} joints, {} IMU, {} touch",
            i, num_modules, num_hinges, num_imus, num_touches,
        );

        // Calculate spawn position
        let row = i / cols;
        let col = i % cols;
        let x = col as f32 * spacing - offset;
        let z = row as f32 * spacing - offset;
        let spawn_pos = Vec3::new(x, 2.0, z);
        let spawn_transform = Transform::from_translation(spawn_pos)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI));

        let mut genome = agent.genotype.controller.clone();
        genome.break_cycles();
        let evaluator = CppnEvaluator::new(&genome);
        let scratchpad = evaluator.create_scratchpad();

        let root = commands
            .spawn((
                Transform::from_translation(spawn_pos),
                Visibility::default(),
                Name::new(format!("Agent_{}", i)),
            ))
            .id();

        // Spawn physics hierarchy
        let spawned = spawn_robot(
            &mut commands,
            &blueprint,
            &palette,
            &mut meshes,
            spawn_transform,
        );

        // Tag all modules with AgentModule for explosion detection
        for &module_entity in &spawned.modules {
            commands.entity(module_entity).insert(AgentModule);
        }

        // Parent all spawned entities under root for proper cleanup on despawn
        {
            let mut root_cmds = commands.entity(root);
            for &module_entity in &spawned.modules {
                root_cmds.add_child(module_entity);
            }
            for &(joint_entity, _) in &spawned.joints {
                root_cmds.add_child(joint_entity);
            }
        }

        let num_inputs = genome.input_ids.len();

        // The body entity (first module) will hold the RobotBrain.
        let body_entity = match spawned.modules.first() {
            Some(&e) => e,
            None => continue,
        };

        // Attach sensor components with brain_entity and input_offset so
        // gather_sensor_inputs can iterate sensors linearly and write
        // directly to the brain's input buffer.
        let mut imu_index = 0usize;
        let mut touch_index = 0usize;
        for &(sensor_entity, sensor_type) in &spawned.sensors {
            match sensor_type {
                SensorType::IMU => {
                    let input_offset = NUM_CLOCK_INPUTS + imu_index * INPUTS_PER_IMU;
                    commands.entity(sensor_entity).insert(AgentImu {
                        prev_linear_vel: Vec3::ZERO,
                        prev_angular_vel: Vec3::ZERO,
                        brain_entity: body_entity,
                        input_offset,
                    });
                    imu_index += 1;
                }
                SensorType::Touch => {
                    let input_offset = NUM_CLOCK_INPUTS + num_imus * INPUTS_PER_IMU + touch_index;
                    commands.entity(sensor_entity).insert(AgentTouch {
                        brain_entity: body_entity,
                        input_offset,
                    });
                    touch_index += 1;
                }
                _ => {}
            }
        }

        let mut joint_entities = Vec::with_capacity(num_hinges);
        for &(joint_entity, joint_type) in &spawned.joints {
            if joint_type == JointType::Hinge {
                commands.entity(joint_entity).insert(AgentJoint);
                joint_entities.push(joint_entity);
            }
        }

        // Module entities excluding the body (first module) for explosion checks.
        let module_entities: Vec<Entity> = spawned.modules.iter().skip(1).copied().collect();

        // Attach brain to the body entity with pre-allocated sensor input buffer.
        commands.entity(body_entity).insert((
            RobotBrain {
                evaluator,
                scratchpad,
                agent_index: i,
                sensor_inputs: vec![0.0; num_inputs],
                joint_entities,
                module_entities,
            },
            FitnessBaseline(None),
        ));

        agent.root_entity = Some(root);
    }
}

/// Counts modules, hinge joints, IMU sensors, and touch sensors from a blueprint.
fn count_blueprint_stats(blueprint: &RobotBlueprint) -> (usize, usize, usize, usize) {
    let num_modules = blueprint.modules.len();
    let num_hinges = blueprint
        .joints
        .iter()
        .filter(|j| j.joint_type == JointType::Hinge)
        .count();
    let mut num_imus = 0usize;
    let mut num_touches = 0usize;
    for module in blueprint.modules.values() {
        for sensor in &module.sensors {
            match sensor.sensor_type {
                SensorType::IMU => num_imus += 1,
                SensorType::Touch => num_touches += 1,
                _ => {}
            }
        }
    }
    (num_modules, num_hinges, num_imus, num_touches)
}
