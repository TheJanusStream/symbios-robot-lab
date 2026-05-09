use rand::Rng;
use serde::{Deserialize, Serialize};
use symbios::System;
use symbios::system::crossover::CrossoverConfig;
use symbios::system::mutate::{
    LiteralPromotionConfig, MutationConfig, OperatorFlipConfig, RuleDuplicationConfig,
    StructuralMutationConfig, TopologicalMutationConfig,
};
use symbios_genetics::Genotype;
use symbios_neat::NeatGenome;

/// Combined genotype for body (L-System morphology) and brain (NEAT network).
///
/// Implements `Genotype` for mutation and crossover. The L-System is mutated
/// through parametric, structural, topological, and operator-flip operators;
/// the NEAT genome is mutated and crossed over independently.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentGenotype {
    /// The Body: L-System source code that generates the RobotBlueprint
    pub morphology: String,

    /// The Brain: NEAT Genome that controls the joints
    pub controller: NeatGenome,

    /// Shared metrics
    pub generation: usize,

    /// Seed for the L-System's internal RNG (affects stochastic rule selection).
    pub lsystem_seed: u64,

    /// Number of L-System derivation steps (controls morphological complexity).
    pub lsystem_iterations: usize,

    /// Cached brain topology `(num_inputs, num_outputs)` derived from the
    /// morphology's L-System. Avoids re-parsing the L-System every time the
    /// counts are needed (e.g. at generation boundaries and spawn).
    /// `None` means "not yet computed" — will be derived on first use.
    #[serde(skip)]
    pub cached_topology: Option<(usize, usize)>,
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
            lsystem_seed: rng.random(),
            lsystem_iterations: 3,
            cached_topology: Some((neat_config.num_inputs, neat_config.num_outputs)),
        }
    }
}

impl Genotype for AgentGenotype {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        // 1. Mutate Body (L-System Source)
        if let Ok(mut system) = System::from_source(&self.morphology) {
            let r = rate as f64;

            // Parametric: jitter #define constants and rule probability weights
            system.mutate_with_rng(
                rng,
                &MutationConfig {
                    rule_probability_rate: r * 0.3,
                    rule_probability_strength: 0.15,
                    constant_rate: r * 0.5,
                    constant_strength: 0.2,
                    gaussian_jitter_scale: r * 0.3,
                    gaussian_jitter_rate: r * 0.5,
                },
            );

            // Structural: insert/delete/swap modules in rule successors + bytecode
            system.structural_mutate_with_rng(
                rng,
                &StructuralMutationConfig {
                    successor_rate: r * 0.1,
                    insert_rate: 0.15,
                    delete_rate: 0.05,
                    swap_rate: 0.1,
                    bytecode_rate: r * 0.1,
                    op_rate: 0.05,
                    push_perturbation: 0.3,
                },
            );

            // Operator flip: swap Add<->Sub, Mul<->Div, Gt<->Lt, etc.
            system.operator_flip_mutate_with_rng(
                rng,
                &OperatorFlipConfig {
                    arithmetic_flip_rate: r * 0.05,
                    relational_flip_rate: r * 0.05,
                },
            );

            // Rule duplication: clone rules with perturbed probabilities
            system.rule_duplication_mutate_with_rng(
                rng,
                &RuleDuplicationConfig {
                    duplication_rate: r * 0.03,
                    condition_perturbation: 0.1,
                },
            );

            // Topological: swap spatial counterparts (+<->-, &<->^, \<->/)
            system.topological_mutate_with_rng(
                rng,
                &TopologicalMutationConfig {
                    swap_rate: r * 0.05,
                },
            );

            // Literal promotion: replace Push literals with matching #define refs
            system.literal_to_constant_promote_with_rng(
                rng,
                &LiteralPromotionConfig {
                    promotion_rate: r * 0.1,
                    match_tolerance: 0.5,
                },
            );

            self.morphology = system.to_source();
        }

        // 2. Mutate L-System seed (re-roll changes stochastic rule selection)
        if rng.random_bool((rate * 0.2) as f64) {
            self.lsystem_seed = rng.random();
        }

        // 3. Mutate L-System iteration count (±1, clamped to [1, 6])
        if rng.random_bool((rate * 0.1) as f64) {
            let delta: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
            self.lsystem_iterations = (self.lsystem_iterations as i32 + delta).clamp(1, 6) as usize;
        }

        // 4. Mutate Brain
        self.controller.mutate(rng, rate);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        // Crossover body via L-System rule selection + constant blending
        let morphology = match (
            System::from_source(&self.morphology),
            System::from_source(&other.morphology),
        ) {
            (Ok(sys_a), Ok(sys_b)) => {
                let config = CrossoverConfig::default();
                match sys_a.crossover_with_rng(&sys_b, rng, &config) {
                    Ok(offspring) => offspring.to_source(),
                    Err(_) => self.morphology.clone(),
                }
            }
            _ => self.morphology.clone(),
        };

        Self {
            morphology,
            controller: self.controller.crossover(&other.controller, rng),
            generation: self.generation.max(other.generation) + 1,
            lsystem_seed: if rng.random_bool(0.5) {
                self.lsystem_seed
            } else {
                other.lsystem_seed
            },
            lsystem_iterations: if rng.random_bool(0.5) {
                self.lsystem_iterations
            } else {
                other.lsystem_iterations
            },
            cached_topology: None,
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

    fn test_morphology() -> String {
        "#define len 1.0\nomega: B(len) L\nL -> J B(len)".to_string()
    }

    fn make_genotype(rng: &mut Pcg64) -> AgentGenotype {
        AgentGenotype::new(test_morphology(), &test_neat_config(), rng)
    }

    #[test]
    fn new_genotype_has_correct_fields() {
        let mut rng = Pcg64::seed_from_u64(42);
        let g = make_genotype(&mut rng);

        assert_eq!(g.generation, 0);
        assert!(g.morphology.contains("omega:"));
        assert!(g.morphology.contains("->"));
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
        // L-System crossover blends constants and selects rules from parents,
        // so the child morphology should still be a valid L-System.
        let sys = System::from_source(&child.morphology);
        assert!(sys.is_ok(), "Child morphology should be a valid L-System");
        assert_eq!(child.controller.input_ids.len(), 4);
        assert_eq!(child.controller.output_ids.len(), 2);
    }

    #[test]
    fn morphology_survives_mutation_roundtrip() {
        let mut rng = Pcg64::seed_from_u64(99);
        let mut g = AgentGenotype::new(test_morphology(), &test_neat_config(), &mut rng);

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
