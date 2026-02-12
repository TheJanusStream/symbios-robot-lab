use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::simulation::lifecycle::derive_brain_topology;
use crate::ui::lab_panel::Telemetry;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;
use symbios_genetics::{Genotype, Phenotype};
use symbios_neat::NeatConfig;

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

/// MAP-Elites archive: a grid of best-in-niche individuals.
///
/// 2D behavioral descriptor:
///   - Axis 0: morphological complexity (num_hinge_joints / 20, clamped to [0,1])
///   - Axis 1: body size (num_modules / 25, clamped to [0,1])
///
/// Each cell stores the highest-fitness individual that maps to that niche.
#[derive(Resource)]
pub struct EliteArchive {
    archive: BTreeMap<[usize; 2], Phenotype<AgentGenotype>>,
    /// Cached list of occupied keys for O(1) random parent sampling.
    keys: Vec<[usize; 2]>,
    /// Number of bins per descriptor dimension.
    pub resolution: usize,
}

impl Default for EliteArchive {
    fn default() -> Self {
        Self {
            archive: BTreeMap::new(),
            keys: Vec::new(),
            resolution: 10,
        }
    }
}

impl EliteArchive {
    /// Maps a behavioral descriptor (values in [0,1]) to grid cell indices.
    fn map_to_index(&self, descriptor: &[f32; 2]) -> [usize; 2] {
        let map_dim = |v: f32| -> usize {
            let scaled = v.clamp(0.0, 1.0) * self.resolution as f32;
            (scaled.floor() as usize).min(self.resolution - 1)
        };
        [map_dim(descriptor[0]), map_dim(descriptor[1])]
    }

    /// Inserts an individual into the archive if its cell is empty or it has
    /// strictly higher fitness than the current occupant.
    fn insert(&mut self, genotype: AgentGenotype, fitness: f32, descriptor: [f32; 2]) {
        if !fitness.is_finite() {
            return;
        }
        let idx = self.map_to_index(&descriptor);

        let dominated = self
            .archive
            .get(&idx)
            .is_some_and(|e| fitness <= e.fitness && !e.fitness.is_nan());

        if !dominated {
            let is_new = !self.archive.contains_key(&idx);
            self.archive.insert(
                idx,
                Phenotype {
                    genotype,
                    fitness,
                    objectives: vec![fitness],
                    descriptor: descriptor.to_vec(),
                },
            );
            if is_new {
                self.keys.push(idx);
            }
        }
    }

    /// Returns the number of occupied cells.
    pub fn len(&self) -> usize {
        self.archive.len()
    }

    /// Samples a random parent genotype from the archive.
    fn sample_parent(&self, rng: &mut impl rand::Rng) -> Option<&AgentGenotype> {
        if self.keys.is_empty() {
            return None;
        }
        let key = &self.keys[rng.random_range(0..self.keys.len())];
        self.archive.get(key).map(|p| &p.genotype)
    }

    /// Returns the best fitness across all archive cells.
    pub fn best_fitness(&self) -> f32 {
        self.archive
            .values()
            .map(|p| p.fitness)
            .fold(0.0_f32, f32::max)
    }

    /// Clears the archive (used when reinitializing the population).
    pub fn clear(&mut self) {
        self.archive.clear();
        self.keys.clear();
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

/// MAP-Elites end-of-generation: insert evaluated agents into the archive,
/// then sample parents and mutate to create the next generation.
pub fn tick_generation(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<LabConfig>,
    mut population: ResMut<Population>,
    mut gen_timer: ResMut<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
    mut archive: ResMut<EliteArchive>,
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

    // 1. Insert all evaluated agents into the MAP-Elites archive
    for agent in &population.agents {
        archive.insert(agent.genotype.clone(), agent.fitness, agent.descriptor);
    }

    let best_fitness = archive.best_fitness();
    info!(
        "Generation {} complete. Archive: {}/{} cells, best fitness: {:.2}",
        generation,
        archive.len(),
        archive.resolution * archive.resolution,
        best_fitness,
    );

    telemetry.best_per_generation.push(best_fitness);
    telemetry.archive_coverage = archive.len();

    // 2. Sample parents from archive and create offspring via mutation
    let mut rng = Pcg64::from_rng(&mut rand::rng());
    let mutation_rate = 0.3;

    let mut next_gen: Vec<Agent> = Vec::with_capacity(config.population_size);

    for _ in 0..config.population_size {
        let parent = match archive.sample_parent(&mut rng) {
            Some(p) => p,
            None => continue,
        };

        let mut child = parent.clone();
        child.mutate(&mut rng, mutation_rate);
        child.generation = generation + 1;

        // Derive NEAT brain topology for the (potentially mutated) morphology.
        // Cache the result so spawn_agents can skip re-derivation.
        // If parsing fails, keep the parent's genome dimensions unchanged.
        let topology = derive_brain_topology(&child.morphology);
        child.cached_topology = topology;

        if let Some((ni, no)) = topology {
            // If morphology changed topology, rebuild the NEAT genome to match.
            // Otherwise keep the inherited (and potentially mutated) genome.
            if ni != child.controller.input_ids.len()
                || no != child.controller.output_ids.len()
            {
                let cfg = NeatConfig {
                    num_inputs: ni,
                    num_outputs: no,
                    output_activation: symbios_neat::Activation::Tanh,
                    ..NeatConfig::default()
                };
                child.controller = symbios_neat::NeatGenome::fully_connected(cfg, &mut rng);
            }
        }

        next_gen.push(Agent {
            genotype: child,
            fitness: 0.0,
            descriptor: [0.0, 0.0],
            root_entity: None,
        });
    }

    // 3. Despawn all old robot hierarchies
    for agent in &population.agents {
        if let Some(root) = agent.root_entity {
            commands.entity(root).try_despawn();
        }
    }

    // 4. Install next generation
    *population = Population {
        agents: next_gen,
        generation: generation + 1,
    };

    // 5. Reset timer
    gen_timer.timer = Timer::from_seconds(config.generation_duration, TimerMode::Once);
    gen_timer.timer.reset();
    gen_timer.active = false;
}
