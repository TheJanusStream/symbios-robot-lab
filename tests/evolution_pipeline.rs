//! Integration tests for the evolutionary pipeline.
//!
//! Tests the full cycle: create population → assign fitness →
//! select + breed → verify next generation properties.

use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_genetics::Genotype;
use symbios_neat::NeatConfig;

// Re-use the crate's types directly (integration tests see the crate as external)
// Since this is a binary crate, we test the underlying libraries instead.

fn neat_config() -> NeatConfig {
    NeatConfig {
        num_inputs: 4,
        num_outputs: 8,
        output_activation: symbios_neat::Activation::Tanh,
        ..NeatConfig::default()
    }
}

fn make_genome(rng: &mut Pcg64) -> symbios_neat::NeatGenome {
    symbios_neat::NeatGenome::fully_connected(neat_config(), rng)
}

#[test]
fn neat_genome_evaluates_without_panic() {
    let mut rng = Pcg64::seed_from_u64(42);
    let genome = make_genome(&mut rng);

    let evaluator = symbios_neat::CppnEvaluator::new(&genome).expect("acyclic genome");
    let outputs = evaluator.evaluate(&[0.5, -0.5, 0.1, 0.0]);

    assert_eq!(outputs.len(), 8, "Should have 8 outputs (one per joint)");
    for (i, &val) in outputs.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "Output {} = {} should be in [-1, 1] (tanh activation)",
            i,
            val
        );
    }
}

#[test]
fn mutation_preserves_network_topology() {
    let mut rng = Pcg64::seed_from_u64(42);
    let mut genome = make_genome(&mut rng);

    for _ in 0..20 {
        genome.mutate(&mut rng, 0.3);
    }

    // After mutation, network should still have correct I/O counts
    assert_eq!(genome.input_ids.len(), 4);
    assert_eq!(genome.output_ids.len(), 8);

    // Should still be evaluable after breaking cycles
    genome.break_cycles();
    let evaluator = symbios_neat::CppnEvaluator::new(&genome).expect("acyclic after break_cycles");
    let outputs = evaluator.evaluate(&[1.0, 0.0, -1.0, 0.5]);
    assert_eq!(outputs.len(), 8);
}

#[test]
fn crossover_preserves_io_counts() {
    let mut rng = Pcg64::seed_from_u64(42);
    let a = make_genome(&mut rng);
    let b = make_genome(&mut rng);

    let child = a.crossover(&b, &mut rng);

    assert_eq!(child.input_ids.len(), 4);
    assert_eq!(child.output_ids.len(), 8);
}

#[test]
fn cvt_centroids_partition_descriptor_space() {
    // Verify CvtMapElites::with_lloyd produces enough centroids and that the
    // assign_to_centroid mapping is a function (deterministic, in-bounds).
    use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
    let cvt = CvtMapElites::<symbios_neat::NeatGenome>::with_lloyd(
        16,    // num_centroids
        2,     // descriptor_dim
        2_000, // num_samples
        10,    // lloyd_iters
        0.0, 1, 0, // unused fields
    );
    assert_eq!(cvt.num_centroids(), 16);

    let probe = [0.5_f32, 0.5];
    let idx = cvt.assign_to_centroid(&probe);
    assert!(idx < cvt.num_centroids());

    // Determinism: same descriptor maps to same centroid.
    assert_eq!(idx, cvt.assign_to_centroid(&probe));
}

#[test]
fn novelty_distance_is_symmetric_and_nonneg() {
    use symbios_genetics::algorithms::novelty_search::{BehaviourDistance, EuclideanDistance};
    let d = EuclideanDistance;
    let a = [0.0_f32, 0.0];
    let b = [3.0_f32, 4.0];
    let dist_ab = d.distance(&a, &b);
    let dist_ba = d.distance(&b, &a);
    assert!((dist_ab - 5.0).abs() < 1e-5, "Pythagorean distance");
    assert!((dist_ab - dist_ba).abs() < 1e-6, "symmetric");
    assert!(dist_ab >= 0.0, "non-negative");
    assert_eq!(d.distance(&a, &a), 0.0, "zero on identity");
}

#[test]
fn selection_pressure_increases_fitness() {
    // Simulate a simple selection loop: assign random fitness, select top half,
    // breed, repeat. Average fitness should not decrease over generations
    // when elitism is used.
    let mut rng = Pcg64::seed_from_u64(42);
    let pop_size = 20;
    let elite_count = 4;

    // Create initial population with assigned fitness
    let mut population: Vec<(symbios_neat::NeatGenome, f32)> = (0..pop_size)
        .map(|_| {
            let g = make_genome(&mut rng);
            // Fitness based on sum of absolute weights (arbitrary but deterministic)
            let fitness: f32 = g.connections.values().map(|c| c.weight.abs()).sum();
            (g, fitness)
        })
        .collect();

    let initial_best = population.iter().map(|(_, f)| *f).fold(0.0_f32, f32::max);

    // Run 5 generations of selection
    for _ in 0..5 {
        // Sort by fitness descending
        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut next_gen = Vec::with_capacity(pop_size);

        // Elites
        for (genome, _) in population.iter().take(elite_count) {
            next_gen.push(genome.clone());
        }

        // Breed from top half
        let parent_pool = pop_size / 2;
        while next_gen.len() < pop_size {
            let p1 = rand::random_range(0..parent_pool);
            let p2 = rand::random_range(0..parent_pool);
            let mut child = population[p1].0.crossover(&population[p2].0, &mut rng);
            child.mutate(&mut rng, 0.3);
            next_gen.push(child);
        }

        // Re-evaluate
        population = next_gen
            .into_iter()
            .map(|g| {
                let fitness: f32 = g.connections.values().map(|c| c.weight.abs()).sum();
                (g, fitness)
            })
            .collect();
    }

    let final_best = population.iter().map(|(_, f)| *f).fold(0.0_f32, f32::max);

    // With elitism, best fitness should never decrease
    assert!(
        final_best >= initial_best,
        "Best fitness should not decrease with elitism: initial={}, final={}",
        initial_best,
        final_best
    );
}
