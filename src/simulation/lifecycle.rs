use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use bevy::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_neat::NeatConfig;

#[derive(Message)]
pub struct SpawnRequest;

pub fn manage_generations(
    mut commands: Commands,
    mut population: ResMut<Population>,
    config: Res<LabConfig>,
    mut spawn_events: MessageReader<SpawnRequest>,
) {
    for _ in spawn_events.read() {
        info!("Initializing Generation 0...");

        let mut rng = Pcg64::from_rng(&mut rand::rng());
        let neat_config = NeatConfig::default(); // Uses default NEAT settings

        // Define a basic morphology for Gen 0 (Snake-like)
        // A simple chain: Body segment, Hinge Joint, repeat.
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
