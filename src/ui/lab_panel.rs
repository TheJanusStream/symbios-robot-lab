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
            fitnesses.sort_by(|a, b| b.total_cmp(a));

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

        // Generation history graph
        if telemetry.best_per_generation.len() >= 2 {
            ui.separator();
            ui.label("Best fitness per generation:");

            let data = &telemetry.best_per_generation;
            let max_val = data.iter().copied().fold(0.01_f32, f32::max);
            let graph_width = ui.available_width();
            let graph_height = 80.0;
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(graph_width, graph_height),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);

            // Background
            painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));

            // Horizontal grid lines (25%, 50%, 75%)
            for frac in [0.25, 0.5, 0.75] {
                let y = rect.max.y - frac * graph_height;
                painter.line_segment(
                    [egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)],
                    egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
                );
            }

            // Plot line
            let n = data.len();
            let x_step = graph_width / (n - 1).max(1) as f32;
            let points: Vec<egui::Pos2> = data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let x = rect.min.x + i as f32 * x_step;
                    let y = rect.max.y - (v / max_val) * graph_height;
                    egui::pos2(x, y)
                })
                .collect();

            for window in points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 200, 120)),
                );
            }

            // Tooltip on hover
            if let Some(hover_pos) = response.hover_pos() {
                let rel_x = hover_pos.x - rect.min.x;
                let hovered_gen = ((rel_x / x_step).round() as usize).min(n - 1);
                let val = data[hovered_gen];
                response.on_hover_ui(|ui| {
                    ui.label(format!("Gen {}: {:.2}", hovered_gen, val));
                });
            }

            // Axis labels
            painter.text(
                rect.min + egui::vec2(2.0, 2.0),
                egui::Align2::LEFT_TOP,
                format!("{:.1}", max_val),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );
            painter.text(
                egui::pos2(rect.min.x + 2.0, rect.max.y - 2.0),
                egui::Align2::LEFT_BOTTOM,
                "0",
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );
            painter.text(
                egui::pos2(rect.max.x - 2.0, rect.max.y - 2.0),
                egui::Align2::RIGHT_BOTTOM,
                format!("G{}", n - 1),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );
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
