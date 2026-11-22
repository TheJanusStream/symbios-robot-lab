use crate::core::config::BrainEncoding;
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

/// Combined genotype for body (L-System morphology) and brain (NEAT genome).
///
/// Implements `Genotype` for mutation and crossover. The L-System is mutated
/// through parametric, structural, topological, and operator-flip operators;
/// the NEAT genome is mutated and crossed over independently.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentGenotype {
    /// The Body: L-System source code that generates the RobotBlueprint
    pub morphology: String,

    /// The Brain: a NEAT genome that is either the network that drives the
    /// joints or a CPPN that paints that network onto the body, as
    /// `encoding` says.
    pub controller: NeatGenome,

    /// How `controller` becomes a brain. Archives written before HyperNEAT
    /// hold direct genomes.
    #[serde(default = "BrainEncoding::legacy")]
    pub encoding: BrainEncoding,

    /// Shared metrics
    pub generation: usize,

    /// Seed for the L-System's internal RNG (affects stochastic rule selection).
    pub lsystem_seed: u64,

    /// Number of L-System derivation steps (controls morphological complexity).
    pub lsystem_iterations: usize,

    /// Frequency (Hz) of the robot's clock, the sine and cosine its brain
    /// reads. Evolved per robot: a body's natural stride rate depends on its
    /// size and legs. Archives written before it read as the lab's former
    /// fixed clock, [`DEFAULT_RHYTHM_HZ`].
    #[serde(default = "default_rhythm_hz")]
    pub rhythm_hz: f32,

    /// Generations this body has been in the gene pool: 0 while a new body
    /// is evaluated for the first time, one more each generation it survives
    /// in the parent pool, and inherited by children that keep it. Protecting
    /// new bodies selects for youth, so a new body has a few generations to
    /// tune its brain before it competes with bodies whose brains have been
    /// tuned for many.
    #[serde(default)]
    pub body_age: u32,

    /// The robot's identity within a session, so that a parent walking again
    /// is matched with its own scores (see
    /// [`crate::core::population::RobotIds`]). 0 until one is issued:
    /// children, fresh robots and imported ones get theirs when they join a
    /// generation.
    #[serde(default)]
    pub id: u64,

    /// How strongly the robot's steering reflex turns it toward a target
    /// (the target task): each driven joint's swing is scaled by `1 + gain x
    /// side x turn`, so the joints on one side of the body step shorter (Lan
    /// et al. 2020). Evolved; its sign picks which side shortens. 0 for a
    /// robot bred to walk forward, which has no reflex.
    #[serde(default)]
    pub steering_gain: f32,
}

/// The strongest steering reflex a robot may evolve.
pub const MAX_STEERING_GAIN: f32 = 2.0;
/// A steering mutation adds a uniform step of up to this size.
const STEERING_STEP: f32 = 0.25;

/// L-System derivation steps a fresh genotype starts with.
pub const INITIAL_LSYSTEM_ITERATIONS: usize = 3;
/// The most derivation steps mutation lets a body grow with.
pub const MAX_LSYSTEM_ITERATIONS: usize = 6;

/// The clock frequency (Hz) the lab used for every robot before robots
/// evolved their own, near a legged animal's stride rate.
pub const DEFAULT_RHYTHM_HZ: f32 = 1.0;
/// The slowest and fastest clock a robot may evolve (Hz).
pub const MIN_RHYTHM_HZ: f32 = 0.25;
pub const MAX_RHYTHM_HZ: f32 = 4.0;
/// A rhythm mutation multiplies the frequency by `exp(u)`, `u` uniform in
/// `[-RHYTHM_STEP, RHYTHM_STEP]` (up to about +-25%).
const RHYTHM_STEP: f32 = 0.25;

fn default_rhythm_hz() -> f32 {
    DEFAULT_RHYTHM_HZ
}

impl AgentGenotype {
    /// A fresh genotype whose body is `morphology` derived with
    /// `lsystem_seed` for [`INITIAL_LSYSTEM_ITERATIONS`] steps, and whose
    /// brain is `controller`, read as `encoding` says.
    ///
    /// The seed is the caller's because the caller has usually derived the
    /// body already, to size the brain from its sensors and joints: drawing a
    /// different seed here would score one body and breed another whenever
    /// the morphology has stochastic rules.
    ///
    /// The source is kept in the form the L-system writes it back in, which
    /// mutation and crossover keep, so two genotypes grow the same body
    /// exactly when their sources match (see [`Self::same_body`]).
    pub fn new(
        morphology: String,
        lsystem_seed: u64,
        encoding: BrainEncoding,
        controller: NeatGenome,
    ) -> Self {
        let morphology = System::from_source(&morphology)
            .map(|system| system.to_source())
            .unwrap_or(morphology);
        Self {
            morphology,
            controller,
            encoding,
            generation: 0,
            lsystem_seed,
            lsystem_iterations: INITIAL_LSYSTEM_ITERATIONS,
            rhythm_hz: DEFAULT_RHYTHM_HZ,
            body_age: 0,
            id: 0,
            steering_gain: 0.0,
        }
    }

    /// A target-task robot's first steering reflex: uniform in `[-1, 1]`.
    pub fn random_steering_gain(rng: &mut impl Rng) -> f32 {
        rng.random_range(-1.0..=1.0)
    }

    /// Whether `other` grows the same body: the same L-system source, seed
    /// and derivation steps.
    pub fn same_body(&self, other: &Self) -> bool {
        self.morphology == other.morphology
            && self.lsystem_seed == other.lsystem_seed
            && self.lsystem_iterations == other.lsystem_iterations
    }

    /// A child that varies only the body: the bodies of `self` and `other`
    /// crossed and then mutated, with `self`'s brain and rhythm unchanged, so
    /// the new body is first tried with a brain that already walks.
    pub fn body_child<R: Rng>(&self, other: &Self, rate: f32, rng: &mut R) -> Self {
        let mut child = Self {
            morphology: self.cross_morphology(other, rng),
            lsystem_seed: pick(self.lsystem_seed, other.lsystem_seed, rng),
            lsystem_iterations: pick(self.lsystem_iterations, other.lsystem_iterations, rng),
            generation: self.generation.max(other.generation) + 1,
            id: 0,
            ..self.clone()
        };
        child.mutate_body(rng, rate);
        child
    }

    /// A child that varies only the brain: `self`'s body unchanged, the
    /// brains of `self` and `other` crossed and then mutated, so the new
    /// brain is judged on a body it was bred for.
    pub fn brain_child<R: Rng>(&self, other: &Self, rate: f32, rng: &mut R) -> Self {
        let mut child = Self {
            controller: self.cross_controller(other, rng),
            rhythm_hz: pick(self.rhythm_hz, other.rhythm_hz, rng),
            steering_gain: pick_gain(self.steering_gain, other.steering_gain, rng),
            generation: self.generation.max(other.generation) + 1,
            id: 0,
            ..self.clone()
        };
        child.mutate_brain(rng, rate);
        child
    }

    /// Sets a newly bred child's body age from its parents: a parent's age
    /// plus one if the child kept that parent's body (the parent ages a
    /// generation while the child is evaluated), 0 for a new body.
    pub fn inherit_body_age(&mut self, parents: [&Self; 2]) {
        self.body_age = parents
            .iter()
            .filter(|parent| self.same_body(parent))
            .map(|parent| parent.body_age.saturating_add(1))
            .max()
            .unwrap_or(0);
    }

    /// The bodies' L-systems crossed (rules chosen, constants blended); this
    /// parent's source if either fails to parse or the crossover fails.
    fn cross_morphology<R: Rng>(&self, other: &Self, rng: &mut R) -> String {
        match (
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
        }
    }

    /// The brains' genomes crossed. A direct genome and a CPPN cannot be
    /// aligned gene by gene, so a child of mixed parents (the brain setting
    /// changed mid-run) keeps this parent's brain: the selector hands the
    /// fitter parent first.
    fn cross_controller<R: Rng>(&self, other: &Self, rng: &mut R) -> NeatGenome {
        if self.encoding == other.encoding {
            self.controller.crossover(&other.controller, rng)
        } else {
            self.controller.clone()
        }
    }

    /// A fresh robot's clock frequency: log-uniform between half and twice
    /// [`DEFAULT_RHYTHM_HZ`], so a new run starts with a spread of rhythms.
    pub fn random_rhythm(rng: &mut impl Rng) -> f32 {
        let octave = std::f32::consts::LN_2;
        DEFAULT_RHYTHM_HZ * rng.random_range(-octave..=octave).exp()
    }

    /// Varies the body: the L-system's constants, rules and operators, the
    /// seed of its stochastic rules, and its number of derivation steps.
    pub fn mutate_body<R: Rng>(&mut self, rng: &mut R, rate: f32) {
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

        // 3. Mutate L-System iteration count (±1, clamped to [1, MAX])
        if rng.random_bool((rate * 0.1) as f64) {
            let delta: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
            self.lsystem_iterations = (self.lsystem_iterations as i32 + delta)
                .clamp(1, MAX_LSYSTEM_ITERATIONS as i32)
                as usize;
        }
    }

    /// Varies the brain: the controller genome, the clock's frequency, and
    /// the steering reflex of a robot that has one.
    pub fn mutate_brain<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.controller.mutate(rng, rate);
        if rng.random_bool((rate * 0.5).clamp(0.0, 1.0) as f64) {
            let step = rng.random_range(-RHYTHM_STEP..=RHYTHM_STEP).exp();
            self.rhythm_hz = (self.rhythm_hz * step).clamp(MIN_RHYTHM_HZ, MAX_RHYTHM_HZ);
        }
        if self.steering_gain != 0.0 && rng.random_bool((rate * 0.5).clamp(0.0, 1.0) as f64) {
            let step = rng.random_range(-STEERING_STEP..=STEERING_STEP);
            self.steering_gain =
                (self.steering_gain + step).clamp(-MAX_STEERING_GAIN, MAX_STEERING_GAIN);
        }
    }
}

impl Genotype for AgentGenotype {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.mutate_body(rng, rate);
        self.mutate_brain(rng, rate);
    }

    /// Crosses body and brain together. The child's body age is 0 until
    /// [`AgentGenotype::inherit_body_age`] compares it with its parents.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let morphology = self.cross_morphology(other, rng);
        let controller = self.cross_controller(other, rng);
        Self {
            morphology,
            controller,
            encoding: self.encoding,
            generation: self.generation.max(other.generation) + 1,
            lsystem_seed: pick(self.lsystem_seed, other.lsystem_seed, rng),
            lsystem_iterations: pick(self.lsystem_iterations, other.lsystem_iterations, rng),
            rhythm_hz: pick(self.rhythm_hz, other.rhythm_hz, rng),
            body_age: 0,
            id: 0,
            steering_gain: pick_gain(self.steering_gain, other.steering_gain, rng),
        }
    }
}

/// One parent's steering gain, evenly; no draw when neither parent has a
/// reflex (robots bred to walk forward breed exactly as they did before the
/// reflex existed).
fn pick_gain(a: f32, b: f32, rng: &mut impl Rng) -> f32 {
    if a == 0.0 && b == 0.0 {
        0.0
    } else {
        pick(a, b, rng)
    }
}

/// `a` or `b`, evenly.
fn pick<T>(a: T, b: T, rng: &mut impl Rng) -> T {
    if rng.random_bool(0.5) { a } else { b }
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
        AgentGenotype::new(
            test_morphology(),
            7,
            BrainEncoding::Direct,
            NeatGenome::fully_connected(test_neat_config(), rng),
        )
    }

    fn make_cppn_genotype(rng: &mut Pcg64) -> AgentGenotype {
        AgentGenotype::new(
            test_morphology(),
            7,
            BrainEncoding::HyperNeat,
            NeatGenome::fully_connected(NeatConfig::cppn(7, 8), rng),
        )
    }

    #[test]
    fn new_genotype_keeps_the_seed_its_body_was_derived_with() {
        let mut rng = Pcg64::seed_from_u64(1);
        let g = AgentGenotype::new(
            test_morphology(),
            1234,
            BrainEncoding::Direct,
            NeatGenome::fully_connected(test_neat_config(), &mut rng),
        );
        assert_eq!(g.lsystem_seed, 1234);
        assert_eq!(g.lsystem_iterations, INITIAL_LSYSTEM_ITERATIONS);
    }

    #[test]
    fn the_rhythm_stays_in_range_and_children_inherit_a_parents() {
        let mut rng = Pcg64::seed_from_u64(8);
        for _ in 0..200 {
            let r = AgentGenotype::random_rhythm(&mut rng);
            assert!((0.5..=2.0).contains(&r), "fresh rhythm {r}");
        }
        let mut g = make_genotype(&mut rng);
        assert_eq!(g.rhythm_hz, DEFAULT_RHYTHM_HZ);
        let mut changed = false;
        for _ in 0..300 {
            let before = g.rhythm_hz;
            g.mutate_brain(&mut rng, 1.0);
            changed |= g.rhythm_hz != before;
            assert!((MIN_RHYTHM_HZ..=MAX_RHYTHM_HZ).contains(&g.rhythm_hz));
        }
        assert!(changed, "brain mutation never moved the rhythm");
        // Body mutation leaves the brain's rhythm alone.
        let before = g.rhythm_hz;
        g.mutate_body(&mut rng, 1.0);
        assert_eq!(g.rhythm_hz, before);
        let a = AgentGenotype {
            rhythm_hz: 0.5,
            ..make_genotype(&mut rng)
        };
        let b = AgentGenotype {
            rhythm_hz: 3.0,
            ..make_genotype(&mut rng)
        };
        let child = a.crossover(&b, &mut rng);
        assert!(child.rhythm_hz == 0.5 || child.rhythm_hz == 3.0);
    }

    #[test]
    fn a_genotype_keeps_its_body_as_the_l_system_writes_it() {
        let mut rng = Pcg64::seed_from_u64(9);
        let fixtures = crate::core::fixtures::ALL;
        for body in crate::core::presets::MORPHOLOGY_POOL.iter().chain(fixtures) {
            let g = AgentGenotype::new(
                body.to_string(),
                1,
                BrainEncoding::HyperNeat,
                NeatGenome::fully_connected(NeatConfig::cppn(7, 8), &mut rng),
            );
            let rewritten = System::from_source(&g.morphology)
                .expect("parses")
                .to_source();
            assert_eq!(rewritten, g.morphology);
            // So a body crossed with itself is the same body.
            assert!(g.crossover(&g, &mut rng).same_body(&g));
        }
    }

    #[test]
    fn a_child_takes_the_age_of_the_body_it_keeps() {
        let mut rng = Pcg64::seed_from_u64(10);
        let parent = AgentGenotype {
            body_age: 4,
            ..make_cppn_genotype(&mut rng)
        };
        let other = AgentGenotype {
            body_age: 9,
            lsystem_seed: 99,
            ..make_cppn_genotype(&mut rng)
        };
        let mut brain_child = parent.brain_child(&other, 1.0, &mut rng);
        assert!(brain_child.same_body(&parent));
        brain_child.inherit_body_age([&parent, &other]);
        assert_eq!(brain_child.body_age, 5);

        let mut new_body = (0..50)
            .map(|_| parent.body_child(&other, 1.0, &mut rng))
            .find(|c| !c.same_body(&parent) && !c.same_body(&other))
            .expect("body mutation changes a body");
        new_body.inherit_body_age([&parent, &other]);
        assert_eq!(new_body.body_age, 0);
        // A body child's brain is its first parent's.
        let brain = |g: &AgentGenotype| serde_json::to_string(&g.controller).expect("serializes");
        assert_eq!(brain(&new_body), brain(&parent));
        assert_eq!(new_body.rhythm_hz, parent.rhythm_hz);

        let mut adopted = other.clone();
        adopted.inherit_body_age([&parent, &other]);
        assert_eq!(adopted.body_age, 10);
    }

    #[test]
    fn an_archive_written_before_rhythms_reads_as_one_hertz() {
        let mut rng = Pcg64::seed_from_u64(6);
        let mut json = serde_json::to_value(make_genotype(&mut rng)).expect("serialize");
        json.as_object_mut()
            .expect("an object")
            .remove("rhythm_hz")
            .expect("has a rhythm");
        let g: AgentGenotype = serde_json::from_value(json).expect("deserialize");
        assert_eq!(g.rhythm_hz, DEFAULT_RHYTHM_HZ);
    }

    #[test]
    fn a_child_of_mixed_encodings_keeps_the_first_parents_brain() {
        let mut rng = Pcg64::seed_from_u64(3);
        let direct = make_genotype(&mut rng);
        let cppn = make_cppn_genotype(&mut rng);
        let child = cppn.crossover(&direct, &mut rng);
        assert_eq!(child.encoding, BrainEncoding::HyperNeat);
        assert_eq!(child.controller.input_ids.len(), 7);
        let child = direct.crossover(&cppn, &mut rng);
        assert_eq!(child.encoding, BrainEncoding::Direct);
        assert_eq!(child.controller.input_ids.len(), 4);
    }

    #[test]
    fn an_archive_written_before_hyperneat_reads_as_direct() {
        let mut rng = Pcg64::seed_from_u64(5);
        let mut json = serde_json::to_value(make_genotype(&mut rng)).expect("serialize");
        json.as_object_mut()
            .expect("an object")
            .remove("encoding")
            .expect("has an encoding");
        let g: AgentGenotype = serde_json::from_value(json).expect("deserialize");
        assert_eq!(g.encoding, BrainEncoding::Direct);
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
        let mut g = AgentGenotype::new(
            test_morphology(),
            99,
            BrainEncoding::Direct,
            NeatGenome::fully_connected(test_neat_config(), &mut rng),
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
        assert_eq!(g.encoding, g2.encoding);
        assert_eq!(g.controller.input_ids.len(), g2.controller.input_ids.len());
    }
}
