use crate::core::config::LabConfig;
use crate::core::population::Population;
use crate::simulation::generation::GenerationTimer;
use crate::simulation::lifecycle::SpawnRequest;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

/// Persistent telemetry state across generations.
#[derive(Resource, Default)]
pub struct Telemetry {
    pub best_all_time: f32,
    pub best_per_generation: Vec<f32>,
}

pub fn render_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<LabConfig>,
    population: Res<Population>,
    gen_timer: Res<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
    mut spawn_writer: MessageWriter<SpawnRequest>,
) {
    // Track best-of-all-time
    let current_best = population
        .agents
        .iter()
        .map(|a| a.fitness)
        .fold(0.0_f32, f32::max);
    if current_best > telemetry.best_all_time {
        telemetry.best_all_time = current_best;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Symbios Lab Control").show(ctx, |ui| {
        ui.heading("Experiment Control");

        ui.separator();

        // Generation info
        ui.label(format!("Generation: {}", population.generation));
        ui.label(format!("Population: {}", population.agents.len()));

        // Timer
        if gen_timer.active {
            let remaining = gen_timer.timer.remaining_secs();
            let duration = gen_timer.timer.duration().as_secs_f32();
            ui.label(format!(
                "Time remaining: {:.1}s / {:.1}s",
                remaining, duration
            ));
            let progress = 1.0 - (remaining / duration);
            ui.add(egui::ProgressBar::new(progress).text("Generation progress"));
        }

        ui.separator();

        // Fitness telemetry
        ui.heading("Fitness");
        ui.label(format!("Current best: {:.5}", current_best));
        ui.label(format!("All-time best: {:.5}", telemetry.best_all_time));

        // Population fitness heatmap (sorted bar visualization)
        if !population.agents.is_empty() {
            ui.separator();
            ui.label("Population fitness:");
            let max_fitness = current_best.max(0.01);
            let mut fitnesses: Vec<f32> = population.agents.iter().map(|a| a.fitness).collect();
            fitnesses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            let available_width = ui.available_width();
            let bar_height = 8.0;
            let (rect, _response) = ui.allocate_exact_size(
                egui::vec2(available_width, bar_height),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            let bar_width = available_width / fitnesses.len() as f32;

            for (i, &fitness) in fitnesses.iter().enumerate() {
                let t = (fitness / max_fitness).clamp(0.0, 1.0);
                let color =
                    egui::Color32::from_rgb((255.0 * (1.0 - t)) as u8, (255.0 * t) as u8, 50);
                let bar_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(i as f32 * bar_width, 0.0),
                    egui::vec2(bar_width, bar_height),
                );
                painter.rect_filled(bar_rect, 0.0, color);
            }
        }

        // Generation history
        if !telemetry.best_per_generation.is_empty() {
            ui.separator();
            ui.label("Best fitness per generation:");
            let history: String = telemetry
                .best_per_generation
                .iter()
                .enumerate()
                .map(|(i, f)| format!("G{}: {:.1}", i, f))
                .collect::<Vec<_>>()
                .join(" | ");
            ui.label(history);
        }

        ui.separator();

        // Controls
        ui.heading("Controls");
        ui.add(egui::Slider::new(&mut config.time_scale, 0.0..=10.0).text("Time Scale"));
        ui.add(
            egui::Slider::new(&mut config.generation_duration, 5.0..=60.0).text("Gen Duration (s)"),
        );
        ui.add(egui::Slider::new(&mut config.population_size, 2..=100).text("Population Size"));
        ui.checkbox(&mut config.auto_restart, "Auto-advance generations");

        if ui.button("Initialize Population").clicked() {
            telemetry.best_all_time = 0.0;
            telemetry.best_per_generation.clear();
            spawn_writer.write(SpawnRequest);
        }
    });
}
