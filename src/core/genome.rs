use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios::System;
use symbios::system::mutate::MutationConfig;
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
        if let Ok(mut system) = System::from_source(&self.morphology) {
            let config = MutationConfig {
                constant_rate: rate as f64 * 0.5,
                constant_strength: 0.2,
                gaussian_jitter_scale: rate as f64 * 0.3,
                gaussian_jitter_rate: rate as f64 * 0.5,
                ..MutationConfig::default()
            };
            system.mutate_with_rng(rng, &config);
            self.morphology = system.to_source();
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios_neat::NeatConfig;

    fn test_neat_config() -> NeatConfig {
        NeatConfig {
            num_inputs: 4,
            num_outputs: 2,
            output_activation: symbios_neat::Activation::Tanh,
            ..NeatConfig::default()
        }
    }

    fn make_genotype(rng: &mut Pcg64) -> AgentGenotype {
        AgentGenotype::new("omega: B(1) J B(1)".to_string(), &test_neat_config(), rng)
    }

    #[test]
    fn new_genotype_has_correct_fields() {
        let mut rng = Pcg64::seed_from_u64(42);
        let g = make_genotype(&mut rng);

        assert_eq!(g.generation, 0);
        assert!(g.morphology.contains("B(1)"));
        assert_eq!(g.controller.input_ids.len(), 4);
        assert_eq!(g.controller.output_ids.len(), 2);
    }

    #[test]
    fn mutation_modifies_controller() {
        let mut rng = Pcg64::seed_from_u64(42);
        let mut g = make_genotype(&mut rng);
        let original_weights: Vec<f32> = g
            .controller
            .connections
            .values()
            .map(|c| c.weight)
            .collect();

        g.mutate(&mut rng, 1.0);

        let mutated_weights: Vec<f32> = g
            .controller
            .connections
            .values()
            .map(|c| c.weight)
            .collect();
        assert_ne!(
            original_weights, mutated_weights,
            "Mutation should change weights"
        );
    }

    #[test]
    fn crossover_produces_valid_offspring() {
        let mut rng = Pcg64::seed_from_u64(42);
        let a = AgentGenotype {
            generation: 3,
            ..make_genotype(&mut rng)
        };
        let b = AgentGenotype {
            generation: 5,
            ..make_genotype(&mut rng)
        };

        let child = a.crossover(&b, &mut rng);

        assert_eq!(child.generation, 6, "Generation should be max(3,5) + 1");
        assert!(
            child.morphology == a.morphology || child.morphology == b.morphology,
            "Child morphology should come from one parent"
        );
        assert_eq!(child.controller.input_ids.len(), 4);
        assert_eq!(child.controller.output_ids.len(), 2);
    }

    #[test]
    fn morphology_survives_mutation_roundtrip() {
        let mut rng = Pcg64::seed_from_u64(99);
        let mut g = AgentGenotype::new(
            "omega: B(1) J B(1) J B(1)".to_string(),
            &test_neat_config(),
            &mut rng,
        );

        g.mutate(&mut rng, 0.5);

        // Mutated morphology should still be parseable
        let system = System::from_source(&g.morphology);
        assert!(
            system.is_ok(),
            "Mutated morphology should remain valid L-System source"
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let mut rng = Pcg64::seed_from_u64(42);
        let g = make_genotype(&mut rng);

        let json = serde_json::to_string(&g).expect("serialize");
        let g2: AgentGenotype = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(g.morphology, g2.morphology);
        assert_eq!(g.generation, g2.generation);
        assert_eq!(g.controller.input_ids.len(), g2.controller.input_ids.len());
    }
}
