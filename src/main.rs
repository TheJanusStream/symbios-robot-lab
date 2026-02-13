use avian3d::PhysicsPlugins;
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::PanOrbitCameraPlugin;

mod core;
mod simulation;
mod ui;
mod visuals;
use crate::simulation::lifecycle::SpawnRequest;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Symbios Robot Lab".into(),
                    resolution: WindowResolution::new(1600, 900),
                    ..default()
                }),
                ..default()
            }),
            EguiPlugin::default(),
            PanOrbitCameraPlugin,
            PhysicsPlugins::default(),
        ))
        .add_message::<SpawnRequest>()
        // Resources
        .init_resource::<core::config::LabConfig>()
        .init_resource::<core::population::Population>()
        .init_resource::<bevy_symbios::materials::MaterialSettingsMap>()
        .init_resource::<simulation::generation::GenerationTimer>()
        .init_resource::<simulation::generation::ParentPool>()
        .init_resource::<ui::lab_panel::Telemetry>()
        .init_resource::<visuals::ui::CameraTarget>()
        // Startup
        .add_systems(
            Startup,
            (
                simulation::arena::setup_arena,
                simulation::physics::setup_physics,
                visuals::robot_render::setup_materials,
            ),
        )
        // UI (egui pass)
        .add_systems(EguiPrimaryContextPass, ui::lab_panel::render_ui)
        // Update — ordered pipeline for population lifecycle and spawning.
        // Neural evaluation and scoring run in FixedUpdate (see below).
        .add_systems(
            Update,
            (
                // 1. Population lifecycle: despawn old / create new population
                (
                    simulation::lifecycle::manage_generations,
                    simulation::generation::tick_generation,
                ),
                // 2. React to population changes: spawn entities, start timer
                (
                    visuals::robot_render::spawn_agents,
                    simulation::generation::activate_timer,
                ),
                // 3. Tag newly spawned entities
                simulation::collision::assign_collision_layers,
            )
                .chain(),
        )
        // FixedUpdate — neural evaluation and scoring locked to the physics
        // timestep. Prevents wasted CPU on frames the physics engine skips,
        // and gives IMU sensors a stable dt for acceleration computation.
        .add_systems(
            FixedUpdate,
            (
                simulation::brain::gather_sensor_inputs,
                simulation::brain::drive_motors,
                simulation::scoring::update_fitness_scores,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                simulation::physics::sync_time_scale,
                visuals::ui::click_to_follow,
                visuals::ui::follow_target,
                bevy_symbios::materials::sync_material_properties,
            ),
        )
        .run();
}
