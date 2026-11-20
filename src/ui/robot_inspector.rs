use crate::visuals::ui::CameraTarget;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use symbios_robot_lab::core::config::BrainEncoding;
use symbios_robot_lab::core::population::Population;
use symbios_robot_lab::simulation::brain::RobotBrain;
use symbios_robot_lab::simulation::metrics::AgentMetrics;
use symbios_robot_lab::simulation::sensors::{
    AgentImu, AgentJoint, AgentSlider, AgentTouch, AgentUltrasonic, INPUTS_PER_IMU, InputLayout,
    NUM_CLOCK_INPUTS,
};

/// Renders an inspector window for the currently selected robot.
/// Only visible when `CameraTarget.agent_index` is `Some`.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn render_robot_inspector(
    mut contexts: EguiContexts,
    target: Res<CameraTarget>,
    brains: Query<(Entity, &RobotBrain, &AgentMetrics)>,
    population: Res<Population>,
    imus: Query<&AgentImu>,
    touches: Query<&AgentTouch>,
    ultrasonics: Query<&AgentUltrasonic>,
    joints: Query<&AgentJoint>,
    sliders: Query<&AgentSlider>,
) {
    let Some(agent_index) = target.agent_index else {
        return;
    };

    let Some((brain_entity, brain, live)) =
        brains.iter().find(|(_, b, _)| b.agent_index == agent_index)
    else {
        return;
    };

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new(format!("Agent {agent_index}"))
        .default_width(260.0)
        .show(ctx, |ui| {
            // --- Fitness & Metrics ---
            if let Some(agent) = population.agents.get(agent_index) {
                // Use population-level data (updated by scoring each generation)
                ui.heading("Metrics");
                egui::Grid::new("metrics_grid")
                    .num_columns(2)
                    .spacing([8.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("Fitness:");
                        ui.label(format!("{:.5}", agent.fitness));
                        ui.end_row();
                        ui.label("Displacement:");
                        ui.label(format!("{:.3}", agent.displacement));
                        ui.end_row();
                        ui.label("Avg up-align:");
                        ui.label(format!("{:.3}", agent.avg_up_alignment));
                        ui.end_row();
                        ui.label("Up-align stddev:");
                        ui.label(format!("{:.3}", agent.up_alignment_stddev));
                        ui.end_row();
                        ui.label("Avg height:");
                        ui.label(format!("{:.3}", agent.avg_height));
                        ui.end_row();
                        ui.label("CoM bounce (\u{03c3}_y):");
                        ui.label(format!("{:.3}", agent.com_bounce));
                        ui.end_row();
                        ui.label("Avg velocity:");
                        ui.label(format!("{:.3}", agent.avg_velocity));
                        ui.end_row();
                        ui.label("Action smoothness:");
                        ui.label(format!("{:.3}", agent.action_smoothness));
                        ui.end_row();
                        ui.label("Control jerk:");
                        ui.label(format!("{:.4}", agent.control_jerk));
                        ui.end_row();
                        ui.label("Body jerk:");
                        ui.label(format!("{:.3}", agent.body_jerk));
                        ui.end_row();
                        ui.label("Support duty:");
                        ui.label(format!("{:.3}", agent.support_duty_factor));
                        ui.end_row();
                        ui.label("Foot transitions/sensor:");
                        ui.label(format!("{:.2}", agent.foot_transitions_per_sensor));
                        ui.end_row();
                        ui.label("Action intensity:");
                        ui.label(format!("{:.3}", agent.action_intensity));
                        ui.end_row();
                        ui.label("Energy (cmd):");
                        ui.label(format!("{:.1}", agent.energy_consumed));
                        ui.end_row();
                        ui.label("Mech work:");
                        ui.label(format!("{:.1}", agent.mechanical_work));
                        ui.end_row();
                        ui.label("CoT:");
                        let cot = agent.cost_of_transport;
                        ui.label(if cot.is_finite() {
                            format!("{cot:.2}")
                        } else {
                            "\u{221e}".to_string()
                        });
                        ui.end_row();
                    });
            }

            // --- Live brain metrics ---
            ui.separator();
            ui.heading("Live");
            let ticks = live.metric_ticks.max(1) as f32;
            egui::Grid::new("live_grid")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    ui.label("Energy:");
                    ui.label(format!("{:.1}", live.energy_consumed));
                    ui.end_row();
                    ui.label("Up-alignment:");
                    ui.label(format!("{:.3}", live.up_alignment_sum / ticks));
                    ui.end_row();
                    ui.label("Height:");
                    ui.label(format!("{:.3}", live.height_sum / ticks));
                    ui.end_row();
                });

            // --- Topology ---
            let layout = InputLayout {
                imus: imus
                    .iter()
                    .filter(|i| i.brain_entity == brain_entity)
                    .count(),
                touches: touches
                    .iter()
                    .filter(|t| t.brain_entity == brain_entity)
                    .count(),
                ultrasonics: ultrasonics
                    .iter()
                    .filter(|u| u.brain_entity == brain_entity)
                    .count(),
                hinges: joints
                    .iter()
                    .filter(|j| j.brain_entity == brain_entity)
                    .count(),
                sliders: sliders
                    .iter()
                    .filter(|s| s.brain_entity == brain_entity)
                    .count(),
            };
            let (num_imus, num_touches, num_ultrasonics, num_joints) = (
                layout.imus,
                layout.touches,
                layout.ultrasonics,
                layout.hinges,
            );
            let num_prismatics = brain.prismatic_joints.len();

            let genotype = population.agents.get(agent_index).map(|a| &a.genotype);
            ui.separator();
            ui.heading("Topology");
            egui::Grid::new("topology_grid")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    if let Some(genotype) = genotype {
                        ui.label("Brain:");
                        ui.label(genotype.encoding.label());
                        ui.end_row();
                        if genotype.encoding == BrainEncoding::HyperNeat {
                            ui.label("CPPN:").on_hover_text(
                                "The evolved pattern generator that paints this robot's network",
                            );
                            ui.label(format!(
                                "{} nodes, {} links",
                                genotype.controller.nodes.len(),
                                genotype.controller.num_enabled_connections()
                            ));
                            ui.end_row();
                        }
                    }
                    ui.label("Inputs:");
                    ui.label(format!("{}", brain.sensor_inputs.len()));
                    ui.end_row();
                    ui.label("Outputs:");
                    ui.label(format!("{}", num_joints + num_prismatics));
                    ui.end_row();
                    ui.label("Links:").on_hover_text(
                        "Weighted connections in the network (for HyperNEAT, how many of the \
                         candidate links its CPPN expressed)",
                    );
                    ui.label(format!("{}", brain.network.num_connections()));
                    ui.end_row();
                    ui.label("IMUs:");
                    ui.label(format!("{num_imus}"));
                    ui.end_row();
                    ui.label("Touch:");
                    ui.label(format!("{num_touches}"));
                    ui.end_row();
                    ui.label("Ultrasonic:");
                    ui.label(format!("{num_ultrasonics}"));
                    ui.end_row();
                    ui.label("Hinge joints:");
                    ui.label(format!("{num_joints}"));
                    ui.end_row();
                    ui.label("Prismatic joints:");
                    ui.label(format!("{num_prismatics}"));
                    ui.end_row();
                });

            // --- Brain Inputs ---
            ui.separator();
            ui.heading("Brain Inputs");

            let inputs = &brain.sensor_inputs;
            if inputs.is_empty() {
                ui.label("(no inputs)");
                return;
            }

            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    egui::Grid::new("inputs_grid")
                        .num_columns(2)
                        .spacing([8.0, 1.0])
                        .striped(true)
                        .show(ui, |ui| {
                            // Clock inputs
                            if inputs.len() >= NUM_CLOCK_INPUTS {
                                ui.label("sin(t):");
                                ui.label(format!("{:+.3}", inputs[0]));
                                ui.end_row();
                                ui.label("cos(t):");
                                ui.label(format!("{:+.3}", inputs[1]));
                                ui.end_row();
                            }

                            let row = |ui: &mut egui::Ui, label: String, idx: usize| {
                                if let Some(value) = inputs.get(idx) {
                                    ui.label(label);
                                    ui.label(format!("{value:+.3}"));
                                    ui.end_row();
                                }
                            };
                            let imu_labels =
                                ["lin_ax", "lin_ay", "lin_az", "ang_ax", "ang_ay", "ang_az"];
                            for i in 0..num_imus {
                                for (j, label) in imu_labels.iter().enumerate().take(INPUTS_PER_IMU)
                                {
                                    row(ui, format!("imu{i}.{label}:"), layout.imu_offset(i) + j);
                                }
                            }
                            for i in 0..num_touches {
                                row(ui, format!("touch{i}:"), layout.touch_offset(i));
                            }
                            for i in 0..num_ultrasonics {
                                row(ui, format!("ultra{i}:"), layout.ultrasonic_offset(i));
                            }
                            for i in 0..num_joints {
                                row(ui, format!("joint{i}.angle:"), layout.joint_offset(i));
                            }
                            for i in 0..layout.sliders {
                                row(ui, format!("slider{i}.extension:"), layout.slider_offset(i));
                            }
                        });
                });
        });
}
