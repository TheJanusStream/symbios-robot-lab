use crate::core::population::Population;
use crate::simulation::brain::{AgentJoint, RobotBrain};
use crate::simulation::scoring::SpawnPosition;
use bevy::prelude::*;
use bevy_symbios::materials::{MaterialPalette, setup_material_assets};
use bevy_symbios::robot::spawn_robot;
use symbios::System;
use symbios_neat::CppnEvaluator;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter};

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
        let spawn_transform = Transform::from_translation(spawn_pos);

        // Spawn root entity with brain and spawn position tracker
        let mut genome = agent.genotype.controller.clone();
        genome.break_cycles();
        let evaluator = CppnEvaluator::new(&genome);

        let root = commands
            .spawn((
                Transform::from_translation(spawn_pos),
                Visibility::default(),
                Name::new(format!("Agent_{}", i)),
                RobotBrain {
                    evaluator,
                    agent_index: i,
                },
                SpawnPosition(spawn_pos),
            ))
            .id();

        // Spawn physics hierarchy
        spawn_robot(
            &mut commands,
            &blueprint,
            &palette,
            &mut meshes,
            spawn_transform,
        );

        // Attach AgentJoint markers to hinge joints
        // Count only hinge joints to match NEAT output indices
        let mut joint_index = 0;
        for joint_def in &blueprint.joints {
            if joint_def.joint_type == JointType::Hinge {
                commands.spawn(AgentJoint {
                    agent_index: i,
                    joint_index,
                });
                joint_index += 1;
            }
        }

        agent.root_entity = Some(root);
    }
}
