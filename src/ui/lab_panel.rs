use crate::core::config::{GeneticAlgorithm, LabConfig};
use crate::core::population::Population;
use crate::simulation::generation::{GenerationTimer, ParentPool};
use crate::simulation::lifecycle::SpawnRequest;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

/// Persistent telemetry state across generations.
#[derive(Resource, Default)]
pub struct Telemetry {
    pub best_all_time: f32,
    pub best_per_generation: Vec<f32>,
    /// Number of individuals on the NSGA-II Pareto front (rank 0).
    pub pareto_front_size: usize,
}

pub fn render_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<LabConfig>,
    population: Res<Population>,
    gen_timer: Res<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
    mut parent_pool: ResMut<ParentPool>,
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
        // Genetic algorithm selector
        egui::ComboBox::from_label("Algorithm")
            .selected_text(config.algorithm.label())
            .show_ui(ui, |ui| {
                for alg in GeneticAlgorithm::ALL {
                    ui.selectable_value(&mut config.algorithm, alg, alg.label());
                }
            });
        match config.algorithm {
            GeneticAlgorithm::MapElites => {
                ui.label(format!("Archive cells: {}", parent_pool.archive_size()));
            }
            _ => {}
        }

        ui.add(egui::Slider::new(&mut config.time_scale, 0.0..=10.0).text("Time Scale"));
        ui.add(
            egui::Slider::new(&mut config.generation_duration, 5.0..=60.0).text("Gen Duration (s)"),
        );
        ui.add(egui::Slider::new(&mut config.population_size, 2..=100).text("Population Size"));
        ui.checkbox(&mut config.auto_restart, "Auto-advance generations");

        if ui.button("Initialize Population").clicked() {
            telemetry.best_all_time = 0.0;
            telemetry.best_per_generation.clear();
            telemetry.pareto_front_size = 0;
            parent_pool.clear();
            spawn_writer.write(SpawnRequest);
        }

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
        ui.label(format!(
            "Pareto front: {} individuals",
            telemetry.pareto_front_size
        ));

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
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(graph_width, graph_height), egui::Sense::hover());
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

        // Pareto front scatter plot
        let obj_data = parent_pool.objective_data();
        if obj_data.len() >= 2 {
            ui.separator();
            ui.label("Pareto front (fitness vs efficiency):");

            let graph_width = ui.available_width();
            let graph_height = 120.0;
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(graph_width, graph_height), egui::Sense::hover());
            let painter = ui.painter_at(rect);

            // Background
            painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));

            // Compute axis ranges from finite data
            let (mut x_min, mut x_max) = (f32::MAX, f32::MIN);
            let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);
            for &(o0, o1, _) in &obj_data {
                x_min = x_min.min(o0);
                x_max = x_max.max(o0);
                y_min = y_min.min(o1);
                y_max = y_max.max(o1);
            }

            // Pad axes to avoid zero-range
            let x_range = (x_max - x_min).max(0.01);
            let y_range = (y_max - y_min).max(0.01);
            let pad = 0.05;
            x_min -= x_range * pad;
            x_max += x_range * pad;
            y_min -= y_range * pad;
            y_max += y_range * pad;
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;

            let margin = 4.0;
            let plot_rect = rect.shrink(margin);

            // Map data to screen coordinates
            let to_screen = |o0: f32, o1: f32| -> egui::Pos2 {
                let sx = plot_rect.min.x + ((o0 - x_min) / x_range) * plot_rect.width();
                let sy = plot_rect.max.y - ((o1 - y_min) / y_range) * plot_rect.height();
                egui::pos2(sx, sy)
            };

            // Grid lines
            for frac in [0.25, 0.5, 0.75] {
                let y = plot_rect.max.y - frac * plot_rect.height();
                painter.line_segment(
                    [
                        egui::pos2(plot_rect.min.x, y),
                        egui::pos2(plot_rect.max.x, y),
                    ],
                    egui::Stroke::new(0.5, egui::Color32::from_gray(45)),
                );
                let x = plot_rect.min.x + frac * plot_rect.width();
                painter.line_segment(
                    [
                        egui::pos2(x, plot_rect.min.y),
                        egui::pos2(x, plot_rect.max.y),
                    ],
                    egui::Stroke::new(0.5, egui::Color32::from_gray(45)),
                );
            }

            // Collect and sort front points (rank 0) by objective 0 to draw a connected line
            let mut front_points: Vec<egui::Pos2> = obj_data
                .iter()
                .filter(|d| d.2 == 0)
                .map(|&(o0, o1, _)| to_screen(o0, o1))
                .collect();
            front_points.sort_by(|a, b| a.x.total_cmp(&b.x));

            // Draw front line
            for window in front_points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    egui::Stroke::new(
                        1.5,
                        egui::Color32::from_rgba_premultiplied(80, 200, 120, 100),
                    ),
                );
            }

            // Draw dominated points first (behind), then front points on top
            let dot_radius = 3.0;
            let dominated_color = egui::Color32::from_rgb(100, 100, 120);
            let front_color = egui::Color32::from_rgb(80, 200, 120);

            for &(o0, o1, rank) in &obj_data {
                if rank > 0 {
                    painter.circle_filled(to_screen(o0, o1), dot_radius * 0.7, dominated_color);
                }
            }
            for &(o0, o1, rank) in &obj_data {
                if rank == 0 {
                    painter.circle_filled(to_screen(o0, o1), dot_radius, front_color);
                }
            }

            // Axis labels
            painter.text(
                plot_rect.left_bottom() + egui::vec2(2.0, -2.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.1}", y_min),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );
            painter.text(
                plot_rect.left_top() + egui::vec2(2.0, 2.0),
                egui::Align2::LEFT_TOP,
                format!("{:.1}", y_max),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );
            painter.text(
                plot_rect.right_bottom() + egui::vec2(-2.0, -2.0),
                egui::Align2::RIGHT_BOTTOM,
                format!("{:.1}", x_max),
                egui::FontId::proportional(9.0),
                egui::Color32::from_gray(120),
            );

            // Hover tooltip
            if let Some(hover_pos) = response.hover_pos() {
                let closest = obj_data
                    .iter()
                    .map(|&(o0, o1, rank)| {
                        let sp = to_screen(o0, o1);
                        let dist = sp.distance(hover_pos);
                        (dist, o0, o1, rank)
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0));

                if let Some((dist, o0, o1, rank)) = closest
                    && dist < 15.0
                {
                    response.on_hover_ui(|ui| {
                        ui.label(format!("Fitness: {:.3}", o0));
                        ui.label(format!("Efficiency: {:.3}", o1));
                        ui.label(format!("Rank: {}", rank));
                    });
                }
            }
        }
    });
}
