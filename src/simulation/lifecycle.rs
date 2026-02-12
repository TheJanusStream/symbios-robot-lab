use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::simulation::brain::compute_num_inputs;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios::System;
use symbios_neat::NeatConfig;
use symbios_robot::{JointType, RobotConfig, RobotInterpreter, SensorType};

/// Default quadruped morphology for Generation 0.
///
/// Structure: flat central body with 4 legs arranged at 90° intervals.
/// Each leg has a hip hinge and a knee hinge for 2 DOF per leg.
///
/// Layout (top view):
/// ```text
///       [leg]
///         |
///  [leg]--[body]--[leg]
///         |
///       [leg]
/// ```
///
/// Sensors:
///   - `Si` (IMU) on the central body — provides pitch & roll (2 inputs)
///   - `St` (Touch) on each foot — ground contact detection (4 inputs)
///
/// L-System structure:
///   - `#define` constants parametrise joint limits, leg width, and segment length.
///     These are the primary targets of parametric mutation.
///   - `L` production rule defines a two-segment leg with hip + knee joints.
///     Subject to structural mutation (insert/delete/swap modules).
///   - 1 derivation step expands `L` → full leg sequence.
///
/// Total: 9 modules (1 body + 4×2 leg segments), 8 hinge joints,
///        1 IMU sensor, 4 touch sensors.
pub const DEFAULT_MORPHOLOGY: &str = "\
#define hip_min -1.57
#define hip_max 1.57
#define hip_eff 50
#define hip_vel 5
#define knee_min -2.0
#define knee_max 0.2
#define knee_eff 30
#define knee_vel 5
#define lw 0.15
#define seg 0.5
omega: !(0.6) K(0.2, 0.3) Si [ &(90) L ] [ \\(90) &(90) L ] [ |(1) &(90) L ] [ /(90) &(90) L ]
L -> Jl(hip_min, hip_max, hip_eff, hip_vel) J !(lw) B(seg) Jl(knee_min, knee_max, knee_eff, knee_vel) J B(seg) St";

/// Derives NEAT input/output sizes from an L-System morphology string.
///
/// Parses the morphology, builds a blueprint, and counts IMU sensors,
/// touch sensors, and hinge joints to determine brain topology.
///
/// Returns `(num_inputs, num_outputs)` or `None` if the morphology fails to parse.
pub fn derive_brain_topology(morphology: &str) -> Option<(usize, usize)> {
    let mut sys = System::from_source(morphology).ok()?;
    sys.derive(3).ok()?;

    let mut interpreter = RobotInterpreter::new(RobotConfig::default());
    interpreter.populate_standard_symbols(&sys.interner);
    let blueprint = interpreter.build_blueprint(&sys.state);

    let mut num_imus = 0usize;
    let mut num_touches = 0usize;
    for module in blueprint.modules.values() {
        for sensor in &module.sensors {
            match sensor.sensor_type {
                SensorType::IMU => num_imus += 1,
                SensorType::Touch => num_touches += 1,
                _ => {}
            }
        }
    }

    let num_outputs = blueprint
        .joints
        .iter()
        .filter(|j| j.joint_type == JointType::Hinge)
        .count();

    Some((compute_num_inputs(num_imus, num_touches), num_outputs))
}

#[derive(Message)]
pub struct SpawnRequest;

pub fn manage_generations(
    mut commands: Commands,
    mut population: ResMut<Population>,
    config: Res<LabConfig>,
    mut spawn_events: MessageReader<SpawnRequest>,
    mut gen_timer: ResMut<crate::simulation::generation::GenerationTimer>,
) {
    for _ in spawn_events.read() {
        info!("Initializing Generation 0...");

        // Despawn any existing robot hierarchies from a previous run
        for agent in &population.agents {
            if let Some(root) = agent.root_entity {
                commands.entity(root).try_despawn();
            }
        }

        // Reset generation timer
        gen_timer.active = false;

        let mut rng = Pcg64::from_rng(&mut rand::rng());

        let (num_inputs, num_outputs) =
            derive_brain_topology(DEFAULT_MORPHOLOGY).unwrap_or_else(|| {
                warn!("Failed to derive brain topology from default morphology, using fallback");
                (5, 4)
            });

        let neat_config = NeatConfig {
            num_inputs,
            num_outputs,
            output_activation: symbios_neat::Activation::Tanh,
            ..NeatConfig::default()
        };

        let new_agents: Vec<Agent> = (0..config.population_size)
            .map(|_| Agent {
                genotype: AgentGenotype::new(
                    DEFAULT_MORPHOLOGY.to_string(),
                    &neat_config,
                    &mut rng,
                ),
                fitness: 0.0,
                root_entity: None,
            })
            .collect();

        *population = Population::new(new_agents);
    }
}
