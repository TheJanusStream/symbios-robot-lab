use crate::core::config::LabConfig;
use crate::core::population::{Agent, Population};
use crate::ui::lab_panel::Telemetry;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use symbios_genetics::Genotype;

/// Tracks time remaining in the current generation.
#[derive(Resource)]
pub struct GenerationTimer {
    pub timer: Timer,
    pub active: bool,
}

impl Default for GenerationTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(15.0, TimerMode::Once),
            active: false,
        }
    }
}

/// Activates the generation timer when the population is first spawned.
pub fn activate_timer(
    population: Res<Population>,
    config: Res<LabConfig>,
    mut gen_timer: ResMut<GenerationTimer>,
) {
    if population.is_changed() && !population.agents.is_empty() && !gen_timer.active {
        gen_timer.timer = Timer::from_seconds(config.generation_duration, TimerMode::Once);
        gen_timer.timer.reset();
        gen_timer.active = true;
        info!(
            "Generation {} started ({:.0}s timer)",
            population.generation, config.generation_duration
        );
    }
}

/// Handles end-of-generation: selection, breeding, despawning, and re-spawning.
pub fn tick_generation(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<LabConfig>,
    mut population: ResMut<Population>,
    mut gen_timer: ResMut<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
) {
    if !gen_timer.active {
        return;
    }

    gen_timer.timer.tick(time.delta());
    if !gen_timer.timer.is_finished() || !config.auto_restart {
        return;
    }

    // --- End of Generation ---
    let generation = population.generation;
    let best_fitness = population
        .agents
        .iter()
        .map(|a| a.fitness)
        .fold(0.0_f32, f32::max);

    info!(
        "Generation {} complete. Best fitness: {:.2}",
        generation, best_fitness
    );

    telemetry.best_per_generation.push(best_fitness);

    // Sort by fitness (descending); total_cmp handles NaN safely
    population.agents.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));

    // Selection + Breeding
    let mut rng = Pcg64::from_rng(&mut rand::rng());
    let elite_count = (config.population_size / 5).max(1); // Top 20%
    let mutation_rate = 0.3;

    let mut next_gen: Vec<Agent> = Vec::with_capacity(config.population_size);

    // Carry elites forward unchanged
    for agent in population.agents.iter().take(elite_count) {
        let mut elite = agent.clone();
        elite.fitness = 0.0;
        elite.root_entity = None;
        elite.genotype.generation = generation + 1;
        next_gen.push(elite);
    }

    // Fill remaining slots via crossover + mutation from top 50%
    let parent_pool_size = (config.population_size / 2).max(2);
    while next_gen.len() < config.population_size {
        let p1 = rand::random_range(0..parent_pool_size);
        let p2 = rand::random_range(0..parent_pool_size);
        let mut child_genotype = population.agents[p1]
            .genotype
            .crossover(&population.agents[p2].genotype, &mut rng);
        child_genotype.mutate(&mut rng, mutation_rate);
        child_genotype.generation = generation + 1;

        next_gen.push(Agent {
            genotype: child_genotype,
            fitness: 0.0,
            root_entity: None,
        });
    }

    // Despawn all old robot hierarchies via their root entities.
    // Bevy 0.18 despawn is recursive — children (modules, joints) are cleaned up automatically.
    for agent in &population.agents {
        if let Some(root) = agent.root_entity {
            commands.entity(root).try_despawn();
        }
    }

    // Install next generation
    let next_generation = generation + 1;
    *population = Population {
        agents: next_gen,
        generation: next_generation,
    };

    // Reset timer
    gen_timer.timer = Timer::from_seconds(config.generation_duration, TimerMode::Once);
    gen_timer.timer.reset();
    gen_timer.active = false; // Will be re-activated by activate_timer when population changes
}
