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
        // Update
        .add_systems(
            Update,
            (
                simulation::physics::sync_time_scale,
                simulation::lifecycle::manage_generations,
                visuals::robot_render::spawn_agents,
                simulation::collision::assign_collision_layers,
                simulation::brain::drive_motors,
                simulation::scoring::update_fitness_scores,
                simulation::generation::activate_timer,
                simulation::generation::tick_generation,
                visuals::ui::click_to_follow,
                visuals::ui::follow_target,
                bevy_symbios::materials::sync_material_properties,
            ),
        )
        .run();
}
