use crate::core::config::LabConfig;
use crate::core::population::Population;
use bevy::prelude::*;
use bevy_symbios::materials::{
    MaterialPalette, MaterialSettingsMap, setup_material_assets, sync_material_properties,
};
use bevy_symbios::robot::spawn_robot;
use symbios::System;
use symbios_robot::{RobotConfig, RobotInterpreter}; // Needed for parsing standard symbols

pub fn setup_materials(
    commands: Commands,
    materials: ResMut<Assets<StandardMaterial>>,
    images: ResMut<Assets<Image>>,
) {
    // Reuse the bevy_symbios helper to set up the standard palette
    setup_material_assets(commands, materials, images);
}

/// Reactively spawns robots when the Population resource changes
pub fn spawn_agents(
    mut commands: Commands,
    mut population: ResMut<Population>,
    lab_config: Res<LabConfig>,
    palette: Res<MaterialPalette>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !population.is_changed() || population.agents.is_empty() {
        return;
    }

    // 1. Calculate grid layout for the arena
    let pop_size = population.agents.len();
    let cols = (pop_size as f32).sqrt().ceil() as usize;
    let spacing = 5.0; // Meters between robots
    let offset = (cols as f32 * spacing) / 2.0;

    for (i, agent) in population.agents.iter_mut().enumerate() {
        // Skip if already spawned
        if agent.root_entity.is_some() {
            continue;
        }

        // 2. Parse L-System (Morphology)
        // We use a temporary system to derive the source code
        let mut sys = System::new();
        // Simple "Axiom" assumption if not in source:
        if !agent.genotype.morphology.contains("omega:") {
            let _ = sys.set_axiom("A(4)");
        }

        // Load the source
        for line in agent.genotype.morphology.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("omega:") {
                let axiom = trimmed.trim_start_matches("omega:").trim();
                let _ = sys.set_axiom(axiom);
            } else if trimmed.contains("->") {
                let _ = sys.add_rule(trimmed);
            }
        }

        // Derive (Growth Phase)
        let _ = sys.derive(3); // Default iterations

        // 3. Interpret into RobotBlueprint
        // We need a Symbios SymbolTable to resolve string keys to IDs
        let mut interpreter = RobotInterpreter::new(RobotConfig::default());
        interpreter.populate_standard_symbols(&sys.interner);

        let blueprint = interpreter.build_blueprint(&sys.state);

        // Debug log to verify generation
        info!(
            "Spawning Agent {}: {} modules, {} joints",
            i,
            blueprint.modules.len(),
            blueprint.joints.len()
        );

        // 4. Calculate Spawn Position
        let row = i / cols;
        let col = i % cols;
        let x = col as f32 * spacing - offset;
        let z = row as f32 * spacing - offset;
        let spawn_transform = Transform::from_xyz(x, 2.0, z); // Drop them from height 2.0

        // 5. Spawn via bevy_symbios
        // We create a root entity to hold the robot parts
        let root = commands
            .spawn((
                Transform::IDENTITY,
                Visibility::default(),
                Name::new(format!("Agent_{}", i)),
            ))
            .id();

        // Spawn the physics hierarchy
        spawn_robot(
            &mut commands,
            &blueprint,
            &palette,
            &mut meshes,
            spawn_transform,
        );

        // Store the root (Note: spawn_robot currently spawns loose parts;
        // in V2 we should parent them to a root, but for now we track the ID)
        agent.root_entity = Some(root);
    }
}

pub fn sync_transforms() {
    // Placeholder: Avian physics handles transform propagation automatically.
    // We keep this function signature to satisfy main.rs if needed, or remove it.
}
