use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::simulation::brain::NUM_SENSOR_INPUTS;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_neat::NeatConfig;

/// The number of hinge joints in our default morphology.
/// "B J B J B" has 2 hinge joints → 2 motor outputs.
pub const DEFAULT_NUM_JOINTS: usize = 2;

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

        let morphology = "omega: B(1) J(Hinge) B(1) J(Hinge) B(1)".to_string();

        let new_agents: Vec<Agent> = (0..config.population_size)
            .map(|_| Agent {
                genotype: AgentGenotype::new(morphology.clone(), &neat_config, &mut rng),
                fitness: 0.0,
                root_entity: None,
            })
            .collect();

        *population = Population::new(new_agents);
    }
}
