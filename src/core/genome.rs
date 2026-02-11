use crate::core::config::LabConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios_genetics::Genotype;
use symbios_neat::NeatGenome;

#[derive(Clone, Serialize, Deserialize)]
pub struct AgentGenotype {
    /// The Body: L-System source code that generates the RobotBlueprint
    pub morphology: String,

    /// The Brain: NEAT Genome that controls the joints
    pub controller: NeatGenome,

    /// Shared metrics
    pub generation: usize,
}

impl AgentGenotype {
    pub fn new(
        morphology: String,
        neat_config: &symbios_neat::NeatConfig,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            morphology,
            controller: NeatGenome::fully_connected(neat_config.clone(), rng),
            generation: 0,
        }
    }
}

impl Genotype for AgentGenotype {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // 1. Mutate Body (L-System Source)
        // TODO: Use symbios::System::from_source -> mutate -> to_source() logic here
        // similar to lsystem-explorer's PlantGenotype

        // 2. Mutate Brain
        self.controller.mutate(rng, rate);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // Crossover both body and brain
        Self {
            morphology: if rng.random_bool(0.5) {
                self.morphology.clone()
            } else {
                other.morphology.clone()
            },
            controller: self.controller.crossover(&other.controller, rng),
            generation: self.generation.max(other.generation) + 1,
        }
    }
}
