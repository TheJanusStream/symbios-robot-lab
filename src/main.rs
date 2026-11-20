//! The windowed lab: the simulation from the library, drawn, with the egui
//! control panel and robot inspector.

use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::PanOrbitCameraPlugin;
use symbios_robot_lab::simulation::lifecycle::SpawnRequest;
use symbios_robot_lab::simulation::physics::sync_time_scale;
use symbios_robot_lab::simulation::{EvolutionSystems, SimulationPlugin};

mod ui;
mod visuals;

fn main() {
    let mut app = App::new();
    app.add_plugins((
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
        SimulationPlugin,
        visuals::VisualsPlugin,
    ))
    .add_systems(
        EguiPrimaryContextPass,
        (
            ui::lab_panel::render_ui,
            ui::robot_inspector::render_robot_inspector,
            ui::hall_panel::render_hall_of_fame,
        ),
    )
    .add_systems(Update, sync_time_scale)
    .add_systems(FixedUpdate, start_evolving.before(EvolutionSystems));

    // Dropping `.lsys` files into assets/morphologies/ needs a folder listing,
    // which a browser cannot give.
    #[cfg(not(target_arch = "wasm32"))]
    app.add_plugins(symbios_robot_lab::core::hotload::MorphologyHotloadPlugin)
        .add_systems(Update, ui::hall_panel::import_dropped_files);

    app.run();
}

/// Starts the first run on the first fixed step, so the lab opens on robots
/// already evolving rather than an empty arena. Sent from the fixed-step
/// schedule, before the loop that reads it: a request sent at startup could
/// expire before the first fixed step runs.
fn start_evolving(mut started: Local<bool>, mut requests: MessageWriter<SpawnRequest>) {
    if !*started {
        *started = true;
        requests.write(SpawnRequest);
    }
}
