use crate::visuals::targets::TargetPlacement;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use symbios_robot_lab::core::config::{BrainEncoding, GeneticAlgorithm, LabConfig, Task, Terrain};
use symbios_robot_lab::core::descriptor::QdDescriptor;
use symbios_robot_lab::core::population::Population;
use symbios_robot_lab::simulation::clock::GenerationClock;
use symbios_robot_lab::simulation::lifecycle::SpawnRequest;
use symbios_robot_lab::simulation::parent_pool::ParentPool;
use symbios_robot_lab::simulation::target::{TARGET_DISTANCE, UserTarget};
use symbios_robot_lab::simulation::telemetry::Telemetry;

/// Renders the main "Symbios Lab Control" egui window with algorithm selection,
/// parameter sliders, generation status, fitness telemetry, and algorithm-specific
/// visualisations (Pareto scatter, MAP-Elites heatmap, average fitness graph).
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn render_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<LabConfig>,
    population: Res<Population>,
    clock: Res<GenerationClock>,
    telemetry: Res<Telemetry>,
    parent_pool: Res<ParentPool>,
    mut spawn_writer: MessageWriter<SpawnRequest>,
    mut placement: ResMut<TargetPlacement>,
    mut user_target: ResMut<UserTarget>,
) {
    // The robots' fitness so far this generation. It swings while the
    // measured window is short, so it is shown as live and kept out of the
    // run's statistics, which hold only finished evaluations.
    let current_best = population
        .agents
        .iter()
        .map(|a| a.fitness)
        .fold(0.0_f32, f32::max);

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Symbios Lab Control").show(ctx, |ui| {
        egui::CollapsingHeader::new("What is this?")
            .default_open(true)
            .show(ui, |ui| {
                let goal = match config.task {
                    Task::Forward => "walk forward",
                    Task::Target => "walk to their targets (the glowing posts)",
                };
                ui.label(format!(
                    "Robots evolve their bodies and brains to {goal}. Each generation lasts \
                     {:.0} s; the best walkers breed the next. Right-drag to orbit, \
                     middle-drag to pan, scroll to zoom, click a robot to inspect it.",
                    config.generation_duration
                ));
            });

        // Genetic algorithm selector
        egui::ComboBox::from_label("Algorithm")
            .selected_text(config.algorithm.label())
            .show_ui(ui, |ui| {
                for alg in GeneticAlgorithm::ALL {
                    ui.selectable_value(&mut config.algorithm, alg, alg.label());
                }
            });
        if config.algorithm.uses_descriptor() {
            egui::ComboBox::from_label("Diversity over")
                .selected_text(config.descriptor.label())
                .show_ui(ui, |ui| {
                    for space in QdDescriptor::ALL {
                        let [x, y] = space.axis_names();
                        let text = format!("{} ({x}, {y})", space.label());
                        ui.selectable_value(&mut config.descriptor, space, text);
                    }
                })
                .response
                .on_hover_text(
                    "The two traits the archive spreads robots over: how they walk (duty \
                     factor, action intensity), what they are (feet, length), or how many \
                     feet and how much of the time they stand. Switching re-files the \
                     archive.",
                );
        }
        egui::ComboBox::from_label("Brain")
            .selected_text(config.brain.label())
            .show_ui(ui, |ui| {
                for brain in BrainEncoding::ALL {
                    ui.selectable_value(&mut config.brain, brain, brain.label());
                }
            })
            .response
            .on_hover_text(
                "Direct NEAT evolves the network itself, one input per sensor value and one \
                 output per joint. HyperNEAT evolves a pattern that paints the network onto \
                 the body: neurons sit at the sensors and joints, and each link's weight \
                 depends on where its two neurons sit, so one genome fits any body. New \
                 robots get the chosen brain; restart evolution to switch them all.",
            );
        egui::ComboBox::from_label("Terrain")
            .selected_text(config.terrain.label())
            .show_ui(ui, |ui| {
                for terrain in Terrain::ALL {
                    ui.selectable_value(&mut config.terrain, terrain, terrain.label());
                }
            })
            .response
            .on_hover_text(
                "Flat floor, or rough (bumps up to 20 cm) or hilly (hills up to 3 m) ground, \
                 new each generation so no gait wins by fitting one patch. Takes effect from \
                 the next generation.",
            );
        egui::ComboBox::from_label("Task")
            .selected_text(config.task.label())
            .show_ui(ui, |ui| {
                for task in Task::ALL {
                    ui.selectable_value(&mut config.task, task, task.label());
                }
            })
            .response
            .on_hover_text(
                "Walk forward, or walk to a target: each brain reads the target's bearing \
                 and the robot scores the ground it gains toward it. Robots bred from now on \
                 are wired for the chosen task.",
            );
        if config.task == Task::Target {
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut placement.active,
                    "Click the ground to place the target",
                );
                if user_target.0.is_some() && ui.button("Clear").clicked() {
                    user_target.0 = None;
                }
            });
            ui.label(match user_target.0 {
                Some(point) => format!("Every robot walks to ({:.1}, {:.1})", point.x, point.y),
                None => format!(
                    "Each robot's target lies {TARGET_DISTANCE:.0} m from it, in a direction \
                     drawn anew each generation"
                ),
            });
        } else {
            placement.active = false;
        }
        ui.add(egui::Slider::new(&mut config.time_scale, 0.0..=10.0).text("Time Scale"));
        ui.add(
            egui::Slider::new(&mut config.generation_duration, 5.0..=60.0).text("Gen Duration (s)"),
        );
        ui.add(egui::Slider::new(&mut config.population_size, 2..=100).text("Population Size"));
        ui.add(egui::Slider::new(&mut config.mutation_rate, 0.0..=1.0).text("Mutation Rate"));
        ui.add_enabled(
            !config.protects_new_bodies(),
            egui::Checkbox::new(&mut config.vary_body_or_brain, "Vary body or brain"),
        )
        .on_hover_text(
            "Each child changes either its body (keeping a parent's brain) or its brain \
             (keeping that parent's body), never both: a new body is first tried with a \
             brain that walks. Fitter and steadier than changing both.",
        );
        if config.algorithm == GeneticAlgorithm::Nsga2 {
            ui.checkbox(&mut config.protect_new_bodies, "Protect new bodies")
                .on_hover_text(
                    "Selection also favours young bodies: a new body has a few generations to \
                     tune its brain before it must outwalk bodies whose brains have been tuned \
                     for many (Cheney et al. 2018). Keeps more kinds of body alive; slower to \
                     start, it pays off over long runs. Implies varying body or brain.",
                );
        }
        ui.add(
            egui::Slider::new(&mut config.perturbation_strength, 0.0..=200.0)
                .text("Perturbation (N\u{00b7}s)"),
        )
        .on_hover_text(
            "One sideways impulse applied to each robot at 60% of its post-settling window. \
             0 = off. Higher values stress-test gait stability; winners will be \
             perturbation-robust.",
        );
        ui.checkbox(&mut config.leg_damage, "Leg damage")
            .on_hover_text(
                "Every robot walks with one leg broken (its joints limp), a different leg each \
             generation: evolution favours gaits that survive losing any leg. Takes effect \
             from the next generation.",
            );
        ui.checkbox(&mut config.auto_restart, "Auto-advance generations");

        // A new run also clears the parent pool and the statistics.
        if ui
            .button("Restart evolution")
            .on_hover_text("Start over from a fresh population of preset bodies")
            .clicked()
        {
            spawn_writer.write(SpawnRequest);
        }

        ui.separator();

        // Generation info
        ui.label(format!("Generation: {}", population.generation));
        ui.label(format!("Population: {}", population.agents.len()));

        if clock.is_running() {
            ui.label(format!(
                "Time remaining: {:.1}s / {:.1}s",
                clock.remaining_secs(),
                clock.duration_secs()
            ));
            ui.add(egui::ProgressBar::new(clock.progress()).text("Generation progress"));
        }

        ui.separator();

        // Fitness telemetry
        ui.heading("Fitness");
        ui.label(format!("Live best (this generation): {current_best:.5}"))
            .on_hover_text("Fitness measured so far; it settles as the generation runs");
        ui.label(format!("Best evaluated: {:.5}", telemetry.best_all_time))
            .on_hover_text("The best score any finished generation has given a robot");
        match config.algorithm {
            GeneticAlgorithm::Nsga2 => {
                ui.label(format!(
                    "Pareto front: {} individuals",
                    telemetry.pareto_front_size
                ));
            }
            GeneticAlgorithm::MapElites => {
                ui.label(format!("Archive cells: {}", parent_pool.archive_size()));
            }
            GeneticAlgorithm::CvtMapElites => {
                ui.label(format!(
                    "CVT cells: {} ({:.1}% coverage)",
                    parent_pool.cvt_archive_size(),
                    parent_pool.cvt_coverage() * 100.0
                ));
            }
            GeneticAlgorithm::NoveltySearch => {
                ui.label(format!(
                    "Behaviour archive: {} descriptors",
                    parent_pool.novelty_archive_size()
                ));
            }
            GeneticAlgorithm::DominatedNovelty => {
                ui.label(format!("Parents kept: {}", parent_pool.len()))
                    .on_hover_text(
                        "Each robot is kept for being the best of its region, or far from \
                         anyone better",
                    );
            }
            GeneticAlgorithm::SimpleGA => {
                if let Some(last) = telemetry.history.last() {
                    ui.label(format!("Avg fitness: {:.5}", last.mean_fitness));
                }
            }
        }

        if let Some(last) = telemetry.history.last() {
            ui.label(format!(
                "Bodies in the pool: {} different, oldest {} generations",
                last.distinct_bodies, last.oldest_body
            ))
            .on_hover_text(
                "How many different bodies the parents have, and for how many generations \
                 the longest-lived body has been among them",
            );
        }

        // Archive CSV export (grid MAP-Elites or CVT)
        if matches!(
            config.algorithm,
            GeneticAlgorithm::MapElites | GeneticAlgorithm::CvtMapElites
        ) && ui.button("Export archive CSV").clicked()
        {
            export_archive(&parent_pool, config.algorithm);
        }

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
        let best_series = telemetry.best_series();
        if best_series.len() >= 2 {
            ui.separator();
            ui.label("Best fitness per generation:");

            let data = &best_series;
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

        // Algorithm-specific visualizations
        if config.algorithm == GeneticAlgorithm::Nsga2 {
            // Pareto front scatter plot
            let obj_data = parent_pool.objective_data();
            if obj_data.len() >= 2 {
                ui.separator();
                ui.label("Pareto front (walker_fitness vs 1/(1+CoT)):");

                let graph_width = ui.available_width();
                let graph_height = 120.0;
                let (rect, response) = ui.allocate_exact_size(
                    egui::vec2(graph_width, graph_height),
                    egui::Sense::hover(),
                );
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
        } // end Nsga2

        // MAP-Elites archive grid heatmap
        if config.algorithm == GeneticAlgorithm::MapElites {
            let archive = parent_pool.archive_data();
            let res = parent_pool.archive_resolution().max(1);
            if !archive.is_empty() {
                ui.separator();
                let [x_axis, y_axis] = parent_pool.descriptor_space().axis_names();
                ui.label(format!("Archive grid ({x_axis} vs {y_axis}):"));

                let graph_size = ui.available_width().min(200.0);
                let (rect, response) = ui
                    .allocate_exact_size(egui::vec2(graph_size, graph_size), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));

                let max_fit = archive.iter().map(|&(_, _, f)| f).fold(0.01_f32, f32::max);
                let cell_w = graph_size / res as f32;
                let cell_h = graph_size / res as f32;

                for &(cx, cy, fitness) in &archive {
                    let t = (fitness / max_fit).clamp(0.0, 1.0);
                    let color =
                        egui::Color32::from_rgb((255.0 * (1.0 - t)) as u8, (255.0 * t) as u8, 50);
                    // y=0 at bottom
                    let cell_rect = egui::Rect::from_min_size(
                        rect.min + egui::vec2(cx as f32 * cell_w, (res - 1 - cy) as f32 * cell_h),
                        egui::vec2(cell_w, cell_h),
                    );
                    painter.rect_filled(cell_rect, 0.0, color);
                }

                painter.text(
                    rect.left_bottom() + egui::vec2(2.0, -2.0),
                    egui::Align2::LEFT_BOTTOM,
                    "0",
                    egui::FontId::proportional(9.0),
                    egui::Color32::from_gray(120),
                );
                axis_labels(&painter, rect, [x_axis, y_axis]);

                // Hover tooltip
                if let Some(hover_pos) = response.hover_pos() {
                    let rel = hover_pos - rect.min;
                    let hx = (rel.x / cell_w) as usize;
                    let hy = res.saturating_sub(1) - (rel.y / cell_h) as usize;
                    if let Some(&(_, _, f)) = archive.iter().find(|&&(x, y, _)| x == hx && y == hy)
                    {
                        response.on_hover_ui(|ui| {
                            ui.label(format!("Cell ({}, {})", hx, hy));
                            ui.label(format!("Fitness: {:.3}", f));
                        });
                    }
                }
            }
        }

        // CVT-MAP-Elites: scatter centroids in descriptor space. Empty centroids
        // are drawn dim grey; occupied centroids are coloured by their elite's
        // fitness (red→green).
        if config.algorithm == GeneticAlgorithm::CvtMapElites
            && let Some(centroids) = parent_pool.cvt_centroids()
            && !centroids.is_empty()
        {
            ui.separator();
            let [x_axis, y_axis] = parent_pool.descriptor_space().axis_names();
            ui.label(format!("CVT centroids ({x_axis} vs {y_axis}):"));
            let archive: std::collections::HashMap<usize, f32> =
                parent_pool.cvt_archive_data().into_iter().collect();

            let graph_size = ui.available_width().min(200.0);
            let (rect, _resp) =
                ui.allocate_exact_size(egui::vec2(graph_size, graph_size), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));

            let max_fit = archive.values().copied().fold(0.01_f32, f32::max).max(0.01);

            for (idx, c) in centroids.iter().enumerate() {
                let cx = c.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let cy = c.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                // y=0 at bottom
                let sx = rect.min.x + cx * graph_size;
                let sy = rect.max.y - cy * graph_size;
                let (color, radius) = match archive.get(&idx).copied() {
                    Some(f) => {
                        let t = (f / max_fit).clamp(0.0, 1.0);
                        (
                            egui::Color32::from_rgb(
                                (255.0 * (1.0 - t)) as u8,
                                (255.0 * t) as u8,
                                50,
                            ),
                            4.0,
                        )
                    }
                    None => (egui::Color32::from_gray(60), 2.0),
                };
                painter.circle_filled(egui::pos2(sx, sy), radius, color);
            }
            axis_labels(&painter, rect, [x_axis, y_axis]);
        }

        // Novelty and Dominated Novelty Search: the parents' points in the
        // descriptor space, coloured by fitness (red to green).
        if matches!(
            config.algorithm,
            GeneticAlgorithm::NoveltySearch | GeneticAlgorithm::DominatedNovelty
        ) {
            let points = parent_pool.points();
            if !points.is_empty() {
                ui.separator();
                let [x_axis, y_axis] = parent_pool.descriptor_space().axis_names();
                ui.label(format!("Parents ({x_axis} vs {y_axis}):"));
                let graph_size = ui.available_width().min(200.0);
                let (rect, _resp) = ui
                    .allocate_exact_size(egui::vec2(graph_size, graph_size), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));
                let max_fit = points.iter().map(|&(_, f)| f).fold(0.01_f32, f32::max);
                for &([x, y], fitness) in &points {
                    let t = (fitness / max_fit).clamp(0.0, 1.0);
                    let color =
                        egui::Color32::from_rgb((255.0 * (1.0 - t)) as u8, (255.0 * t) as u8, 50);
                    let position = egui::pos2(
                        rect.min.x + x.clamp(0.0, 1.0) * graph_size,
                        rect.max.y - y.clamp(0.0, 1.0) * graph_size,
                    );
                    painter.circle_filled(position, 3.0, color);
                }
                axis_labels(&painter, rect, [x_axis, y_axis]);
            }
        }

        // SimpleGA: average fitness per generation graph
        let mean_series = telemetry.mean_series();
        if config.algorithm == GeneticAlgorithm::SimpleGA && mean_series.len() >= 2 {
            ui.separator();
            ui.label("Avg fitness per generation:");

            let data = &mean_series;
            let max_val = data.iter().copied().fold(0.01_f32, f32::max);
            let graph_width = ui.available_width();
            let graph_height = 80.0;
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(graph_width, graph_height), egui::Sense::hover());
            let painter = ui.painter_at(rect);

            painter.rect_filled(rect, 4.0, egui::Color32::from_gray(30));

            for frac in [0.25, 0.5, 0.75] {
                let y = rect.max.y - frac * graph_height;
                painter.line_segment(
                    [egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)],
                    egui::Stroke::new(0.5, egui::Color32::from_gray(50)),
                );
            }

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
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(200, 160, 60)),
                );
            }

            if let Some(hover_pos) = response.hover_pos() {
                let rel_x = hover_pos.x - rect.min.x;
                let hovered_gen = ((rel_x / x_step).round() as usize).min(n - 1);
                let val = data[hovered_gen];
                response.on_hover_ui(|ui| {
                    ui.label(format!("Gen {}: {:.2}", hovered_gen, val));
                });
            }

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
    });
}

/// Saves the current quality-diversity archive (grid MAP-Elites or CVT) as a
/// timestamped CSV: a file in the working directory on the desktop, a
/// download in the browser. Invoked by the "Export archive CSV" button.
/// Names a square plot's axes: x along the bottom right, y at the top left.
fn axis_labels(painter: &egui::Painter, rect: egui::Rect, [x_axis, y_axis]: [&str; 2]) {
    painter.text(
        rect.right_bottom() + egui::vec2(-2.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        x_axis,
        egui::FontId::proportional(9.0),
        egui::Color32::from_gray(120),
    );
    painter.text(
        rect.left_top() + egui::vec2(2.0, 2.0),
        egui::Align2::LEFT_TOP,
        y_axis,
        egui::FontId::proportional(9.0),
        egui::Color32::from_gray(120),
    );
}

fn export_archive(parent_pool: &ParentPool, algorithm: GeneticAlgorithm) {
    let suffix = match algorithm {
        GeneticAlgorithm::MapElites => "grid",
        GeneticAlgorithm::CvtMapElites => "cvt",
        _ => return,
    };
    let mut csv = Vec::new();
    let rows = match parent_pool.export_archive_csv(algorithm, &mut csv) {
        Ok(rows) => rows,
        Err(e) => {
            warn!("Archive CSV could not be written: {e}");
            return;
        }
    };
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let file_name = format!("archive_{suffix}_{timestamp}.csv");
    match super::download::save_file(&file_name, "text/csv", &csv) {
        Ok(()) => info!("Exported {rows} archive cells to {file_name}"),
        Err(e) => warn!("Could not save {file_name}: {e}"),
    }
}
