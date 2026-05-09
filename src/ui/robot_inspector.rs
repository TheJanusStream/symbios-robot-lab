use crate::core::population::Population;
use crate::simulation::brain::{
    AgentImu, AgentJoint, AgentTouch, AgentUltrasonic, INPUTS_PER_IMU, INPUTS_PER_JOINT,
    INPUTS_PER_TOUCH, INPUTS_PER_ULTRASONIC, NUM_CLOCK_INPUTS, RobotBrain,
};
use crate::visuals::ui::CameraTarget;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

/// Renders an inspector window for the currently selected robot.
/// Only visible when `CameraTarget.agent_index` is `Some`.
pub fn render_robot_inspector(
    mut contexts: EguiContexts,
    target: Res<CameraTarget>,
    brains: Query<(Entity, &RobotBrain)>,
    population: Res<Population>,
    imus: Query<&AgentImu>,
    touches: Query<&AgentTouch>,
    ultrasonics: Query<&AgentUltrasonic>,
    joints: Query<&AgentJoint>,
) {
    let Some(agent_index) = target.agent_index else {
        return;
    };

    let Some((brain_entity, brain)) = brains.iter().find(|(_, b)| b.agent_index == agent_index)
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
                        ui.label("Energy:");
                        ui.label(format!("{:.1}", agent.energy_consumed));
                        ui.end_row();
                        ui.label("Avg up-align:");
                        ui.label(format!("{:.3}", agent.avg_up_alignment));
                        ui.end_row();
                        ui.label("Avg height:");
                        ui.label(format!("{:.3}", agent.avg_height));
                        ui.end_row();
                    });
            }

            // --- Live brain metrics ---
            ui.separator();
            ui.heading("Live");
            let ticks = brain.metric_ticks.max(1) as f32;
            egui::Grid::new("live_grid")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    ui.label("Energy:");
                    ui.label(format!("{:.1}", brain.energy_consumed));
                    ui.end_row();
                    ui.label("Up-alignment:");
                    ui.label(format!("{:.3}", brain.up_alignment_sum / ticks));
                    ui.end_row();
                    ui.label("Height:");
                    ui.label(format!("{:.3}", brain.height_sum / ticks));
                    ui.end_row();
                });

            // --- Topology ---
            let num_imus = imus
                .iter()
                .filter(|i| i.brain_entity == brain_entity)
                .count();
            let num_touches = touches
                .iter()
                .filter(|t| t.brain_entity == brain_entity)
                .count();
            let num_ultrasonics = ultrasonics
                .iter()
                .filter(|u| u.brain_entity == brain_entity)
                .count();
            let num_joints = joints
                .iter()
                .filter(|j| j.brain_entity == brain_entity)
                .count();
            let num_prismatics = brain.prismatic_entities.len();

            ui.separator();
            ui.heading("Topology");
            egui::Grid::new("topology_grid")
                .num_columns(2)
                .spacing([8.0, 2.0])
                .show(ui, |ui| {
                    ui.label("Inputs:");
                    ui.label(format!("{}", brain.sensor_inputs.len()));
                    ui.end_row();
                    ui.label("Outputs:");
                    ui.label(format!("{}", num_joints + num_prismatics));
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

                            // IMU inputs
                            let imu_start = NUM_CLOCK_INPUTS;
                            for i in 0..num_imus {
                                let base = imu_start + i * INPUTS_PER_IMU;
                                let labels =
                                    ["lin_ax", "lin_ay", "lin_az", "ang_ax", "ang_ay", "ang_az"];
                                for (j, label) in labels.iter().enumerate() {
                                    let idx = base + j;
                                    if idx < inputs.len() {
                                        ui.label(format!("imu{i}.{label}:"));
                                        ui.label(format!("{:+.3}", inputs[idx]));
                                        ui.end_row();
                                    }
                                }
                            }

                            // Touch inputs
                            let touch_start = imu_start + num_imus * INPUTS_PER_IMU;
                            for i in 0..num_touches {
                                let idx = touch_start + i * INPUTS_PER_TOUCH;
                                if idx < inputs.len() {
                                    ui.label(format!("touch{i}:"));
                                    ui.label(format!("{:.0}", inputs[idx]));
                                    ui.end_row();
                                }
                            }

                            // Ultrasonic inputs
                            let ultra_start = touch_start + num_touches * INPUTS_PER_TOUCH;
                            for i in 0..num_ultrasonics {
                                let idx = ultra_start + i * INPUTS_PER_ULTRASONIC;
                                if idx < inputs.len() {
                                    ui.label(format!("ultra{i}:"));
                                    ui.label(format!("{:.3}", inputs[idx]));
                                    ui.end_row();
                                }
                            }

                            // Joint angle inputs
                            let joint_start = ultra_start + num_ultrasonics * INPUTS_PER_ULTRASONIC;
                            for i in 0..num_joints {
                                let idx = joint_start + i * INPUTS_PER_JOINT;
                                if idx < inputs.len() {
                                    ui.label(format!("joint{i}.angle:"));
                                    ui.label(format!("{:+.3}", inputs[idx]));
                                    ui.end_row();
                                }
                            }
                        });
                });
        });
}
