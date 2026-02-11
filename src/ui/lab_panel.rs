use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::lifecycle::SpawnRequest;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui}; // We will create this event

pub fn render_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<LabConfig>,
    population: Res<Population>,
    mut spawn_writer: MessageWriter<SpawnRequest>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Symbios Lab Control").show(ctx, |ui| {
        ui.heading("Experiment Control");

        ui.label(format!("Generation: {}", population.generation));
        ui.label(format!("Population: {}", population.agents.len()));

        ui.separator();

        ui.add(egui::Slider::new(&mut config.time_scale, 0.0..=10.0).text("Time Scale"));

        if ui.button("Initialize Population").clicked() {
            spawn_writer.write(SpawnRequest);
        }
    });
}
