use crate::core::config::{GeneticAlgorithm, LabConfig};
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::core::presets::MORPHOLOGY_POOL;
use crate::simulation::lifecycle::derive_blueprint;
use crate::ui::lab_panel::Telemetry;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use rand::Rng;
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;
use symbios_genetics::algorithms::nsga2::{Nsga2, SortWrapper};
use symbios_genetics::{Genotype, Phenotype};
use symbios_neat::NeatConfig;
use symbios_robot::RobotBlueprint;

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

/// Parent pool supporting multiple evolutionary algorithm strategies.
///
/// Stores evaluated `Phenotype`s with two objectives:
///   - Objective 0: displacement * avg_up_alignment (walker fitness — maximize)
///   - Objective 1: -(energy / displacement) (energy efficiency — maximize, i.e. minimize energy per meter)
///
/// Three selection modes:
///   - **NSGA-II**: non-dominated sort + crowding distance, binary tournament
///   - **SimpleGA**: sort by scalar fitness, tournament(3) with elitism
///   - **MAP-Elites**: behavioural-descriptor archive grid, random parent selection
#[derive(Resource, Default)]
pub struct ParentPool {
    /// Ranked phenotypes from previous generation(s).
    pool: Vec<Phenotype<AgentGenotype>>,
    /// Pareto rank per individual (0 = non-dominated front).
    ranks: Vec<usize>,
    /// Crowding distance per individual.
    crowding: Vec<f32>,
    /// MAP-Elites archive: cell key → (phenotype index in `pool`).
    archive: BTreeMap<Vec<usize>, usize>,
    /// MAP-Elites resolution (bins per descriptor dimension).
    archive_resolution: usize,
}

impl ParentPool {
    /// Number of individuals in the pool.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Best displacement (objective 0) across the pool.
    pub fn best_fitness(&self) -> f32 {
        self.pool.iter().map(|p| p.fitness).fold(0.0_f32, f32::max)
    }

    /// Number of individuals on the Pareto front (rank 0).
    pub fn front_size(&self) -> usize {
        self.ranks.iter().filter(|&&r| r == 0).count()
    }

    /// Number of occupied cells in the MAP-Elites archive.
    pub fn archive_size(&self) -> usize {
        self.archive.len()
    }

    /// Grid resolution (bins per dimension) for MAP-Elites.
    pub fn archive_resolution(&self) -> usize {
        self.archive_resolution
    }

    /// Returns `(cell_x, cell_y, fitness)` for each occupied archive cell.
    pub fn archive_data(&self) -> Vec<(usize, usize, f32)> {
        self.archive
            .iter()
            .filter_map(|(cell, &idx)| {
                let x = *cell.first()?;
                let y = *cell.get(1)?;
                Some((x, y, self.pool[idx].fitness))
            })
            .collect()
    }

    /// Returns `(objective_0, objective_1, rank)` for each individual in the pool.
    /// Individuals with non-finite objectives are excluded.
    pub fn objective_data(&self) -> Vec<(f32, f32, usize)> {
        self.pool
            .iter()
            .zip(self.ranks.iter())
            .filter_map(|(pheno, &rank)| {
                let o0 = *pheno.objectives.first()?;
                let o1 = *pheno.objectives.get(1)?;
                if o0.is_finite() && o1.is_finite() {
                    Some((o0, o1, rank))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clears the pool (used when reinitializing the population).
    pub fn clear(&mut self) {
        self.pool.clear();
        self.ranks.clear();
        self.crowding.clear();
        self.archive.clear();
    }

    // ── NSGA-II ──────────────────────────────────────────────────────

    /// Binary tournament (NSGA-II): pick two random individuals, return the
    /// better one. Prefers lower rank; ties broken by higher crowding distance.
    fn binary_tournament_nsga2(&self, rng: &mut impl Rng) -> usize {
        let n = self.pool.len();
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        match self.ranks[i].cmp(&self.ranks[j]) {
            std::cmp::Ordering::Less => i,
            std::cmp::Ordering::Greater => j,
            std::cmp::Ordering::Equal => {
                if self.crowding[i] >= self.crowding[j] {
                    i
                } else {
                    j
                }
            }
        }
    }

    /// Inserts evaluated agents, combines with existing pool, runs
    /// non-dominated sort + crowding distance, and truncates to `cap`.
    fn update_nsga2(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, cap: usize) {
        self.archive.clear();
        let mut combined = std::mem::take(&mut self.pool);
        combined.extend(evaluated);

        if combined.is_empty() {
            return;
        }

        let fronts = Nsga2::<AgentGenotype>::fast_non_dominated_sort(&combined);
        let mut selected: Vec<SortWrapper> = Vec::with_capacity(cap);

        for (rank, indices) in fronts.iter().enumerate() {
            let mut front: Vec<SortWrapper> = indices
                .iter()
                .map(|&i| SortWrapper {
                    index: i,
                    rank,
                    distance: 0.0,
                })
                .collect();

            Nsga2::<AgentGenotype>::calculate_crowding_distance(&mut front, &combined);

            if selected.len() + front.len() <= cap {
                selected.extend(front);
            } else {
                front.sort_by(|a, b| b.distance.total_cmp(&a.distance));
                selected.extend(front.into_iter().take(cap - selected.len()));
                break;
            }
        }

        self.ranks = selected.iter().map(|w| w.rank).collect();
        self.crowding = selected.iter().map(|w| w.distance).collect();
        self.pool = selected
            .into_iter()
            .map(|w| combined[w.index].clone())
            .collect();
    }

    // ── SimpleGA ─────────────────────────────────────────────────────

    /// Tournament selection (SimpleGA): pick 3 random individuals, return the
    /// one with the highest scalar fitness.
    fn tournament_simple(&self, rng: &mut impl Rng) -> usize {
        let n = self.pool.len();
        let tournament_size = 3.min(n);
        let mut best = rng.random_range(0..n);
        for _ in 1..tournament_size {
            let candidate = rng.random_range(0..n);
            if self.pool[candidate].fitness > self.pool[best].fitness
                || self.pool[best].fitness.is_nan()
            {
                best = candidate;
            }
        }
        best
    }

    /// Inserts evaluated agents, combines with existing pool, sorts by scalar
    /// fitness (descending), and truncates to `cap` (elitism = entire retained pool).
    fn update_simple(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, cap: usize) {
        self.archive.clear();
        let mut combined = std::mem::take(&mut self.pool);
        combined.extend(evaluated);

        if combined.is_empty() {
            return;
        }

        // Sort descending by fitness, NaN last
        combined.sort_by(|a, b| {
            match (a.fitness.is_nan(), b.fitness.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // NaN sorts after valid
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => b.fitness.total_cmp(&a.fitness),
            }
        });
        combined.truncate(cap);

        // All individuals get rank 0 and infinite crowding (not used for selection)
        self.ranks = vec![0; combined.len()];
        self.crowding = vec![f32::INFINITY; combined.len()];
        self.pool = combined;
    }

    // ── MAP-Elites ───────────────────────────────────────────────────

    /// Maps a two-element descriptor to archive cell indices.
    fn descriptor_to_cell(&self, descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| {
                let scaled = v.clamp(0.0, 1.0) * self.archive_resolution as f32;
                (scaled.floor() as usize).min(self.archive_resolution - 1)
            })
            .collect()
    }

    /// Select a random parent from the MAP-Elites archive.
    fn select_from_archive(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&Vec<usize>> = self.archive.keys().collect();
        let key = keys[rng.random_range(0..keys.len())];
        self.archive[key]
    }

    /// Inserts evaluated agents into the MAP-Elites archive. Each agent's
    /// descriptor is mapped to a grid cell; the cell keeps the fittest elite.
    /// The pool is rebuilt from archive contents after insertion.
    fn update_map_elites(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, _cap: usize) {
        if self.archive_resolution == 0 {
            self.archive_resolution = 20;
        }

        for pheno in evaluated {
            if pheno.fitness.is_nan()
                || pheno.fitness.is_infinite()
                || pheno.descriptor.iter().any(|v| v.is_nan())
            {
                continue;
            }
            let cell = self.descriptor_to_cell(&pheno.descriptor);
            let dominated = self
                .archive
                .get(&cell)
                .is_some_and(|&idx| pheno.fitness <= self.pool[idx].fitness);
            if !dominated {
                // Insert into pool and update archive index
                let new_idx = self.pool.len();
                self.pool.push(pheno);
                self.archive.insert(cell, new_idx);
            }
        }

        // Rebuild pool from archive to remove stale entries
        let mut new_pool = Vec::with_capacity(self.archive.len());
        let mut new_archive = BTreeMap::new();
        for (cell, &old_idx) in &self.archive {
            let new_idx = new_pool.len();
            new_pool.push(self.pool[old_idx].clone());
            new_archive.insert(cell.clone(), new_idx);
        }
        self.pool = new_pool;
        self.archive = new_archive;
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
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

/// End-of-generation: evaluate agents, select parents via the configured
/// algorithm, crossover + mutate to create offspring.
pub fn tick_generation(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<LabConfig>,
    mut population: ResMut<Population>,
    mut gen_timer: ResMut<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
    mut parent_pool: ResMut<ParentPool>,
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
    let algorithm = config.algorithm;

    // 1. Convert evaluated agents to Phenotypes with walker fitness objectives:
    //    obj[0] = displacement * avg_up_alignment (walker fitness — maximize)
    //    obj[1] = -(energy / displacement) (energy efficiency — maximize, lower cost is better)
    //    Agents with 0 displacement or 0 motors get -inf so any viable agent dominates them.
    let evaluated: Vec<Phenotype<AgentGenotype>> = population
        .agents
        .iter()
        .map(|agent| {
            let (objectives, descriptor) =
                if agent.num_motors > 4 && agent.displacement > f32::EPSILON {
                    let walker_fitness =
                        agent.displacement * agent.avg_up_alignment * (agent.avg_height + 0.5);
                    let energy_efficiency = -(agent.energy_consumed / agent.displacement);

                    // MAP-Elites descriptor: normalised (up_alignment, avg_height)
                    let desc = vec![
                        agent.avg_up_alignment.clamp(0.0, 1.0),
                        agent.avg_height.clamp(0.0, 1.0),
                    ];
                    (vec![walker_fitness, energy_efficiency], desc)
                } else {
                    (vec![f32::NEG_INFINITY, f32::NEG_INFINITY], vec![0.0, 0.0])
                };
            Phenotype {
                genotype: agent.genotype.clone(),
                fitness: agent.fitness,
                objectives,
                descriptor,
            }
        })
        .collect();

    // 2. Merge with parent pool using the selected algorithm
    match algorithm {
        GeneticAlgorithm::Nsga2 => parent_pool.update_nsga2(evaluated, config.population_size),
        GeneticAlgorithm::SimpleGA => parent_pool.update_simple(evaluated, config.population_size),
        GeneticAlgorithm::MapElites => {
            parent_pool.update_map_elites(evaluated, config.population_size)
        }
    }

    let best_fitness = parent_pool.best_fitness();
    info!(
        "Generation {} complete ({:?}). Pool: {} individuals, Pareto front: {}, best: {:.2}",
        generation,
        algorithm,
        parent_pool.len(),
        parent_pool.front_size(),
        best_fitness,
    );

    telemetry.best_per_generation.push(best_fitness);
    telemetry.pareto_front_size = parent_pool.front_size();

    let avg_fitness = if population.agents.is_empty() {
        0.0
    } else {
        let sum: f32 = population.agents.iter().map(|a| a.fitness).sum();
        sum / population.agents.len() as f32
    };
    telemetry.avg_per_generation.push(avg_fitness);

    // 3. Selection + crossover + mutation,
    //    with 10% fresh preset injection to maintain genetic diversity.
    let mut rng = Pcg64::from_rng(&mut rand::rng());
    let mutation_rate = 0.3;
    let num_fresh = config.population_size / 10;
    let num_evolved = config.population_size - num_fresh;

    let mut children: Vec<AgentGenotype> = Vec::with_capacity(config.population_size);

    // 3a. Evolved offspring from parent pool
    for _ in 0..num_evolved {
        if parent_pool.len() < 2 {
            break;
        }
        let (idx1, idx2) = match algorithm {
            GeneticAlgorithm::Nsga2 => (
                parent_pool.binary_tournament_nsga2(&mut rng),
                parent_pool.binary_tournament_nsga2(&mut rng),
            ),
            GeneticAlgorithm::SimpleGA => (
                parent_pool.tournament_simple(&mut rng),
                parent_pool.tournament_simple(&mut rng),
            ),
            GeneticAlgorithm::MapElites => (
                parent_pool.select_from_archive(&mut rng),
                parent_pool.select_from_archive(&mut rng),
            ),
        };
        // NEAT crossover assumes self (first arg) is the fitter parent.
        // Swap so the fitter parent calls crossover.
        let (fitter, weaker) = if parent_pool.pool[idx1].fitness >= parent_pool.pool[idx2].fitness {
            (
                &parent_pool.pool[idx1].genotype,
                &parent_pool.pool[idx2].genotype,
            )
        } else {
            (
                &parent_pool.pool[idx2].genotype,
                &parent_pool.pool[idx1].genotype,
            )
        };

        let mut child = fitter.crossover(weaker, &mut rng);
        child.mutate(&mut rng, mutation_rate);
        child.generation = generation + 1;
        children.push(child);
    }

    // 3b. Fresh preset injection (10%) to keep the gene pool diverse
    for _ in 0..num_fresh {
        let morphology = MORPHOLOGY_POOL[rng.random_range(0..MORPHOLOGY_POOL.len())];
        if let Some((_, ni, no)) = derive_blueprint(morphology) {
            let cfg = NeatConfig {
                num_inputs: ni,
                num_outputs: no,
                output_activation: symbios_neat::Activation::Tanh,
                ..NeatConfig::default()
            };
            let mut geno = AgentGenotype::new(morphology.to_string(), &cfg, &mut rng);
            geno.generation = generation + 1;
            children.push(geno);
        }
    }

    // 4. Parallel blueprint derivation
    let pool = ComputeTaskPool::get();
    let blueprints: Vec<Option<(RobotBlueprint, usize, usize)>> = pool.scope(|s| {
        for child in &children {
            let morphology = &child.morphology;
            s.spawn(async move { derive_blueprint(morphology) });
        }
    });

    // 5. Assemble agents, rebuild NEAT genomes where topology changed
    let mut next_gen: Vec<Agent> = Vec::with_capacity(children.len());
    for (mut child, derived) in children.into_iter().zip(blueprints) {
        let cached_blueprint;
        let num_motors;

        if let Some((bp, ni, no)) = derived {
            child.cached_topology = Some((ni, no));
            num_motors = no; // num_outputs == hinge joint count
            cached_blueprint = Some(bp);

            if ni != child.controller.input_ids.len() || no != child.controller.output_ids.len() {
                let cfg = NeatConfig {
                    num_inputs: ni,
                    num_outputs: no,
                    output_activation: symbios_neat::Activation::Tanh,
                    ..NeatConfig::default()
                };
                child.controller = symbios_neat::NeatGenome::fully_connected(cfg, &mut rng);
            }
        } else {
            child.cached_topology = None;
            num_motors = 0;
            cached_blueprint = None;
        }

        next_gen.push(Agent {
            genotype: child,
            fitness: 0.0,
            num_motors,
            root_entity: None,
            cached_blueprint,
            energy_consumed: 0.0,
            displacement: 0.0,
            avg_up_alignment: 0.0,
            avg_height: 0.0,
        });
    }

    // 6. Despawn all old robot hierarchies
    for agent in &population.agents {
        if let Some(root) = agent.root_entity {
            commands.entity(root).try_despawn();
        }
    }

    // 7. Install next generation
    *population = Population {
        agents: next_gen,
        generation: generation + 1,
    };

    // 8. Reset timer
    gen_timer.timer = Timer::from_seconds(config.generation_duration, TimerMode::Once);
    gen_timer.timer.reset();
    gen_timer.active = false;
}
