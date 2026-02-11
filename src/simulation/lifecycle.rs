use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::simulation::brain::NUM_SENSOR_INPUTS;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_neat::NeatConfig;

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
/// Total: 9 modules (1 body + 4×2 leg segments), 8 hinge joints.
pub const DEFAULT_MORPHOLOGY: &str = "\
omega: !(0.6) B(0.3, 0.6, 0.6) \
[ &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ \\(90) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ |(1) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ] \
[ /(90) &(90) Jl(-1.57, 1.57, 50, 5) J !(0.15) B(0.5) Jl(-2.0, 0.2, 30, 5) J B(0.5) ]";

/// The number of hinge joints in the default quadruped morphology.
/// 4 legs × 2 joints (hip + knee) = 8.
pub const DEFAULT_NUM_JOINTS: usize = 8;

#[derive(Message)]
pub struct SpawnRequest;

pub fn manage_generations(
    mut population: ResMut<Population>,
    config: Res<LabConfig>,
    mut spawn_events: MessageReader<SpawnRequest>,
) {
    for _ in spawn_events.read() {
        info!("Initializing Generation 0...");

        let mut rng = Pcg64::from_rng(&mut rand::rng());

        let neat_config = NeatConfig {
            num_inputs: NUM_SENSOR_INPUTS,
            num_outputs: DEFAULT_NUM_JOINTS,
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
