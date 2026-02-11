use crate::core::population::Population;
use crate::simulation::brain::{AgentImu, AgentJoint, AgentTouch, RobotBrain};
use crate::simulation::scoring::FitnessBaseline;
use bevy::prelude::*;
use bevy_symbios::materials::{MaterialPalette, setup_material_assets};
use bevy_symbios::robot::spawn_robot;
use symbios::System;
use symbios_neat::CppnEvaluator;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter, SensorType};

pub fn setup_materials(
    commands: Commands,
    materials: ResMut<Assets<StandardMaterial>>,
    images: ResMut<Assets<Image>>,
) {
    setup_material_assets(commands, materials, images);
}

/// Reactively spawns robots when the Population resource changes.
/// Attaches RobotBrain and AgentJoint components for neural control.
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

        // Parse L-System morphology
        let mut sys = System::new();
        if !agent.genotype.morphology.contains("omega:") {
            let _ = sys.set_axiom("A(4)");
        }
        for line in agent.genotype.morphology.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("omega:") {
                let axiom = trimmed.trim_start_matches("omega:").trim();
                let _ = sys.set_axiom(axiom);
            } else if trimmed.contains("->") {
                let _ = sys.add_rule(trimmed);
            }
        }
        let _ = sys.derive(3);

        // Build blueprint
        let mut interpreter = RobotInterpreter::new(RobotConfig::default());
        interpreter.populate_standard_symbols(&sys.interner);
        let blueprint = interpreter.build_blueprint(&sys.state);

        info!(
            "Spawning Agent {}: {} modules, {} joints",
            i,
            blueprint.modules.len(),
            blueprint.joints.len()
        );

        // Calculate spawn position
        let row = i / cols;
        let col = i % cols;
        let x = col as f32 * spacing - offset;
        let z = row as f32 * spacing - offset;
        let spawn_pos = Vec3::new(x, 2.0, z);
        // Rotate -180° around X
        // maps to world -Y (downward), placing legs beneath the body.
        let spawn_transform = Transform::from_translation(spawn_pos)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::PI));

        // Spawn root entity (container for cleanup via recursive despawn)
        let mut genome = agent.genotype.controller.clone();
        genome.break_cycles();
        let evaluator = CppnEvaluator::new(&genome);

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

        // Attach brain and spawn tracker to the first module (body) entity,
        // which is an actual physics body whose Transform updates under simulation.
        // This is needed for fitness scoring (displacement) and brain sensing (pitch/roll).
        if let Some(&body_entity) = spawned.modules.first() {
            commands.entity(body_entity).insert((
                RobotBrain {
                    evaluator,
                    agent_index: i,
                },
                FitnessBaseline(None),
            ));
        }

        // Attach AgentJoint markers directly to hinge joint entities
        // so drive_motors() can query (&AgentJoint, &mut RevoluteJoint) on the same entity
        let mut joint_index = 0;
        for &(joint_entity, joint_type) in &spawned.joints {
            if joint_type == JointType::Hinge {
                commands.entity(joint_entity).insert(AgentJoint {
                    agent_index: i,
                    joint_index,
                });
                joint_index += 1;
            }
        }

        // Attach AgentImu / AgentTouch markers to sensor-bearing module entities
        let mut imu_index = 0;
        let mut touch_index = 0;
        for &(sensor_entity, sensor_type) in &spawned.sensors {
            match sensor_type {
                SensorType::IMU => {
                    commands.entity(sensor_entity).insert(AgentImu {
                        agent_index: i,
                        sensor_index: imu_index,
                    });
                    imu_index += 1;
                }
                SensorType::Touch => {
                    commands.entity(sensor_entity).insert(AgentTouch {
                        agent_index: i,
                        sensor_index: touch_index,
                    });
                    touch_index += 1;
                }
                _ => {}
            }
        }

        agent.root_entity = Some(root);
    }
}
