use crate::core::config::LabConfig;
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

/// NSGA-II parent pool: ranked individuals from previous generations.
///
/// Stores evaluated `Phenotype`s with two objectives:
///   - Objective 0: displacement (higher is better — maximize locomotion)
///   - Objective 1: negative module count (higher is better — minimize complexity)
///
/// At each generation boundary the current population is combined with the
/// parent pool, non-dominated sorted, and truncated to `population_size`.
/// Binary tournament selection from this pool produces the next generation.
#[derive(Resource, Default)]
pub struct ParentPool {
    /// Ranked phenotypes from previous generation(s).
    pool: Vec<Phenotype<AgentGenotype>>,
    /// Pareto rank per individual (0 = non-dominated front).
    ranks: Vec<usize>,
    /// Crowding distance per individual.
    crowding: Vec<f32>,
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

    /// Clears the pool (used when reinitializing the population).
    pub fn clear(&mut self) {
        self.pool.clear();
        self.ranks.clear();
        self.crowding.clear();
    }

    /// Binary tournament: pick two random individuals, return the better one.
    /// Prefers lower rank; ties broken by higher crowding distance.
    fn binary_tournament(&self, rng: &mut impl Rng) -> usize {
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
    fn update(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, cap: usize) {
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

/// NSGA-II end-of-generation: rank evaluated agents by [displacement, -modules],
/// select parents via binary tournament, crossover + mutate to create offspring.
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

    // 1. Convert evaluated agents to Phenotypes with NSGA-II objectives:
    //    obj[0] = displacement (maximize)
    //    obj[1] = -num_motors (maximize → minimizes motor count)
    //    Agents with 0 motors (parse failures / degenerate morphologies) get -inf
    //    on both objectives so any viable agent dominates them.
    let evaluated: Vec<Phenotype<AgentGenotype>> = population
        .agents
        .iter()
        .map(|agent| {
            let objectives = if agent.num_motors > 0 {
                vec![
                    agent.fitness * agent.num_motors as f32,
                    -(agent.num_motors as f32),
                ]
            } else {
                vec![f32::NEG_INFINITY, f32::NEG_INFINITY]
            };
            Phenotype {
                genotype: agent.genotype.clone(),
                fitness: agent.fitness,
                objectives,
                descriptor: vec![],
            }
        })
        .collect();

    // 2. Merge with parent pool and select top individuals via non-dominated sort
    parent_pool.update(evaluated, config.population_size);

    let best_fitness = parent_pool.best_fitness();
    info!(
        "Generation {} complete. Pool: {} individuals, Pareto front: {}, best displacement: {:.2}",
        generation,
        parent_pool.len(),
        parent_pool.front_size(),
        best_fitness,
    );

    telemetry.best_per_generation.push(best_fitness);
    telemetry.pareto_front_size = parent_pool.front_size();

    // 3. Binary tournament selection + crossover + mutation,
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
        let idx1 = parent_pool.binary_tournament(&mut rng);
        let idx2 = parent_pool.binary_tournament(&mut rng);
        let p1 = &parent_pool.pool[idx1].genotype;
        let p2 = &parent_pool.pool[idx2].genotype;

        let mut child = p1.crossover(p2, &mut rng);
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
