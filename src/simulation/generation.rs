use crate::core::config::{GeneticAlgorithm, LabConfig};
use crate::core::genome::AgentGenotype;
use crate::core::hotload::RuntimeMorphologyPool;
use crate::core::population::{Agent, Population};
use crate::core::presets::MORPHOLOGY_POOL;
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::scoring::evaluate_trajectory;
use crate::ui::lab_panel::Telemetry;
use bevy::prelude::*;
use bevy::tasks::ComputeTaskPool;
use rand::Rng;
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::BTreeMap;
use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
use symbios_genetics::algorithms::novelty_search::{BehaviourDistance, EuclideanDistance};
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

/// Number of CVT centroids (and therefore the maximum CVT archive size).
const CVT_NUM_CENTROIDS: usize = 64;
/// Descriptor dimensionality for CVT centroids — matches `evaluate_trajectory`'s
/// `(avg_up_alignment, avg_height_norm)` descriptor.
const CVT_DESCRIPTOR_DIM: usize = 2;
/// Lloyd sample count for CVT centroid generation.
const CVT_LLOYD_SAMPLES: usize = 10_000;
/// Lloyd iteration count.
const CVT_LLOYD_ITERS: usize = 30;
/// CVT centroid generation seed (deterministic across runs).
const CVT_SEED: u64 = 0xC0_FFEE;
/// Novelty search kNN window size (Lehman & Stanley default range).
const NOVELTY_K: usize = 15;
/// Novelty/fitness blend factor: 1.0 = pure novelty, 0.0 = fitness-only.
const NOVELTY_ALPHA: f32 = 0.7;

/// Parent pool supporting multiple evolutionary algorithm strategies.
///
/// Stores evaluated `Phenotype`s with two objectives:
///   - Objective 0: walker fitness (composite from upstream `Scorer`s)
///   - Objective 1: energy efficiency (negative cost-per-meter)
///
/// Five selection modes:
///   - **NSGA-II**: non-dominated sort + crowding distance, binary tournament
///   - **SimpleGA**: sort by scalar fitness, tournament(3) with elitism
///   - **MAP-Elites (grid)**: regular grid archive, random parent selection
///   - **CVT-MAP-Elites**: Voronoi archive over Lloyd-generated centroids
///   - **Novelty Search**: kNN behavioural-distance scoring blended with fitness
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
    /// Lloyd centroids for CVT-MAP-Elites, generated lazily on first use.
    cvt_centroids: Option<Vec<Vec<f32>>>,
    /// CVT archive: centroid index → phenotype index in `pool`.
    cvt_archive: BTreeMap<usize, usize>,
    /// Behavioural archive for novelty search (descriptor history).
    novelty_archive: Vec<Vec<f32>>,
    /// Per-individual novelty score from the most recent generation
    /// (parallel to `pool`). Empty when not running NoveltySearch.
    novelty_scores: Vec<f32>,
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

    /// Number of occupied cells in the grid MAP-Elites archive.
    pub fn archive_size(&self) -> usize {
        self.archive.len()
    }

    /// Grid resolution (bins per dimension) for MAP-Elites.
    pub fn archive_resolution(&self) -> usize {
        self.archive_resolution
    }

    /// Number of occupied cells in the CVT archive.
    pub fn cvt_archive_size(&self) -> usize {
        self.cvt_archive.len()
    }

    /// CVT coverage: occupied centroids / total centroids.
    pub fn cvt_coverage(&self) -> f32 {
        match &self.cvt_centroids {
            Some(c) if !c.is_empty() => self.cvt_archive.len() as f32 / c.len() as f32,
            _ => 0.0,
        }
    }

    /// Number of descriptors in the novelty archive.
    pub fn novelty_archive_size(&self) -> usize {
        self.novelty_archive.len()
    }

    /// CVT centroids (each is a descriptor-space point) for visualisation.
    pub fn cvt_centroids(&self) -> Option<&[Vec<f32>]> {
        self.cvt_centroids.as_deref()
    }

    /// `(centroid_index, fitness)` for occupied CVT cells.
    pub fn cvt_archive_data(&self) -> Vec<(usize, f32)> {
        self.cvt_archive
            .iter()
            .map(|(&cidx, &pool_idx)| (cidx, self.pool[pool_idx].fitness))
            .collect()
    }

    /// Returns `(cell_x, cell_y, fitness)` for each occupied grid archive cell.
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
        self.cvt_archive.clear();
        self.novelty_archive.clear();
        self.novelty_scores.clear();
    }

    /// Exports the current quality-diversity archive (grid MAP-Elites or CVT)
    /// as CSV. Returns `Ok(rows_written)`. Format:
    /// `cell_key, fitness, descriptor_0, descriptor_1, ...`.
    pub fn export_archive_csv<W: std::io::Write>(
        &self,
        algorithm: GeneticAlgorithm,
        mut writer: W,
    ) -> std::io::Result<usize> {
        let mut rows = 0;
        match algorithm {
            GeneticAlgorithm::MapElites => {
                writeln!(writer, "cell_x,cell_y,fitness,desc_0,desc_1")?;
                for (cell, &idx) in &self.archive {
                    let pheno = &self.pool[idx];
                    let x = cell.first().copied().unwrap_or(0);
                    let y = cell.get(1).copied().unwrap_or(0);
                    let d0 = pheno.descriptor.first().copied().unwrap_or(0.0);
                    let d1 = pheno.descriptor.get(1).copied().unwrap_or(0.0);
                    writeln!(writer, "{x},{y},{:.6},{:.6},{:.6}", pheno.fitness, d0, d1)?;
                    rows += 1;
                }
            }
            GeneticAlgorithm::CvtMapElites => {
                writeln!(writer, "centroid_idx,fitness,desc_0,desc_1")?;
                for (&cidx, &pool_idx) in &self.cvt_archive {
                    let pheno = &self.pool[pool_idx];
                    let d0 = pheno.descriptor.first().copied().unwrap_or(0.0);
                    let d1 = pheno.descriptor.get(1).copied().unwrap_or(0.0);
                    writeln!(writer, "{cidx},{:.6},{:.6},{:.6}", pheno.fitness, d0, d1)?;
                    rows += 1;
                }
            }
            _ => {}
        }
        Ok(rows)
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

    // ── MAP-Elites (grid) ────────────────────────────────────────────

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

    /// Select a random parent from the grid MAP-Elites archive.
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

    // ── CVT-MAP-Elites ───────────────────────────────────────────────

    /// Lazily generate centroids via upstream `CvtMapElites::with_lloyd`.
    fn ensure_cvt_centroids(&mut self) {
        if self.cvt_centroids.is_some() {
            return;
        }
        let cvt = CvtMapElites::<AgentGenotype>::with_lloyd(
            CVT_NUM_CENTROIDS,
            CVT_DESCRIPTOR_DIM,
            CVT_LLOYD_SAMPLES,
            CVT_LLOYD_ITERS,
            0.0, // mutation_rate unused (we don't call .step())
            1,   // batch_size unused (we don't call .step())
            CVT_SEED,
        );
        self.cvt_centroids = Some(cvt.centroids().to_vec());
    }

    /// Squared Euclidean distance to nearest centroid.
    fn nearest_centroid(&self, descriptor: &[f32]) -> Option<usize> {
        let centroids = self.cvt_centroids.as_ref()?;
        let mut best = (0, f32::INFINITY);
        for (i, c) in centroids.iter().enumerate() {
            let d: f32 = c
                .iter()
                .zip(descriptor.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if d < best.1 {
                best = (i, d);
            }
        }
        Some(best.0)
    }

    /// Random parent from the CVT archive.
    fn select_from_cvt(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&usize> = self.cvt_archive.keys().collect();
        let key = keys[rng.random_range(0..keys.len())];
        self.cvt_archive[key]
    }

    /// Inserts evaluated agents into the CVT archive (one elite per centroid).
    /// Rebuilds `pool` from the surviving archive entries.
    fn update_cvt_map_elites(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, _cap: usize) {
        self.ensure_cvt_centroids();

        for pheno in evaluated {
            if pheno.fitness.is_nan()
                || pheno.fitness.is_infinite()
                || pheno.descriptor.iter().any(|v| v.is_nan())
                || pheno.descriptor.len() != CVT_DESCRIPTOR_DIM
            {
                continue;
            }
            let Some(centroid_idx) = self.nearest_centroid(&pheno.descriptor) else {
                continue;
            };
            let dominated = self
                .cvt_archive
                .get(&centroid_idx)
                .is_some_and(|&idx| pheno.fitness <= self.pool[idx].fitness);
            if !dominated {
                let new_idx = self.pool.len();
                self.pool.push(pheno);
                self.cvt_archive.insert(centroid_idx, new_idx);
            }
        }

        let mut new_pool = Vec::with_capacity(self.cvt_archive.len());
        let mut new_archive = BTreeMap::new();
        for (&centroid_idx, &old_idx) in &self.cvt_archive {
            let new_idx = new_pool.len();
            new_pool.push(self.pool[old_idx].clone());
            new_archive.insert(centroid_idx, new_idx);
        }
        self.pool = new_pool;
        self.cvt_archive = new_archive;
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
    }

    // ── Novelty Search ───────────────────────────────────────────────

    /// kNN tournament: pick 3 random pool indices; return the one with the
    /// highest blended novelty+fitness score.
    fn tournament_novelty(&self, rng: &mut impl Rng) -> usize {
        let n = self.pool.len();
        let tournament_size = 3.min(n);
        let mut best = rng.random_range(0..n);
        let mut best_score = self.novelty_scores.get(best).copied().unwrap_or(0.0);
        for _ in 1..tournament_size {
            let candidate = rng.random_range(0..n);
            let score = self.novelty_scores.get(candidate).copied().unwrap_or(0.0);
            if score > best_score {
                best = candidate;
                best_score = score;
            }
        }
        best
    }

    /// Mean kNN distance from `descriptor` to `others` using upstream Euclidean.
    fn mean_knn_distance(&self, descriptor: &[f32], others: &[Vec<f32>]) -> f32 {
        let dist = EuclideanDistance;
        let mut distances: Vec<f32> = others
            .iter()
            .filter(|o| o.len() == descriptor.len())
            .map(|o| dist.distance(descriptor, o))
            .filter(|d| !d.is_nan())
            .collect();
        if distances.is_empty() {
            return 0.0;
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let take = NOVELTY_K.min(distances.len());
        distances.iter().take(take).sum::<f32>() / take as f32
    }

    /// Inserts evaluated agents, computes novelty (kNN to population ∪ archive),
    /// blends with fitness, and replaces the pool with the evaluated batch
    /// scored by blended novelty. Adds 5% of descriptors to the behaviour archive.
    fn update_novelty(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, _cap: usize) {
        self.archive.clear();
        self.cvt_archive.clear();

        if evaluated.is_empty() {
            self.pool.clear();
            self.novelty_scores.clear();
            self.ranks.clear();
            self.crowding.clear();
            return;
        }

        let descriptors: Vec<Vec<f32>> = evaluated.iter().map(|p| p.descriptor.clone()).collect();

        // Novelty[i] = mean kNN distance from descriptors[i] to (descriptors \ {i}) ∪ archive.
        let mut novelty = Vec::with_capacity(evaluated.len());
        for (i, desc) in descriptors.iter().enumerate() {
            if desc.is_empty() || desc.iter().any(|v| v.is_nan()) {
                novelty.push(0.0);
                continue;
            }
            let mut others: Vec<Vec<f32>> = descriptors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, d)| d.clone())
                .collect();
            others.extend(self.novelty_archive.iter().cloned());
            novelty.push(self.mean_knn_distance(desc, &others));
        }

        // Blended score for selection in the next generation.
        let scores: Vec<f32> = evaluated
            .iter()
            .zip(novelty.iter())
            .map(|(p, &nov)| {
                let fit = if p.fitness.is_nan() { 0.0 } else { p.fitness };
                NOVELTY_ALPHA * nov + (1.0 - NOVELTY_ALPHA) * fit
            })
            .collect();

        // Probabilistic archive policy (5%) — matches upstream defaults.
        for desc in &descriptors {
            if desc.is_empty() || desc.iter().any(|v| v.is_nan()) {
                continue;
            }
            if rand::random::<f32>() < 0.05 {
                self.novelty_archive.push(desc.clone());
            }
        }

        self.pool = evaluated;
        self.novelty_scores = scores;
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
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn tick_generation(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<LabConfig>,
    mut population: ResMut<Population>,
    mut gen_timer: ResMut<GenerationTimer>,
    mut telemetry: ResMut<Telemetry>,
    mut parent_pool: ResMut<ParentPool>,
    runtime_pool: Res<RuntimeMorphologyPool>,
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

    // 1. Convert evaluated agents to Phenotypes via the upstream-Scorer-driven
    //    `evaluate_trajectory`. Agents with too few motors or zero displacement
    //    get -inf objectives so any viable agent dominates them. We rebuild
    //    `AgentTrajectory` from the per-agent metric fields populated in
    //    `update_fitness_scores`.
    let evaluated: Vec<Phenotype<AgentGenotype>> = population
        .agents
        .iter()
        .map(|agent| {
            let (fitness, objectives, descriptor) =
                if agent.num_motors > 4 && agent.displacement > f32::EPSILON {
                    let traj = crate::simulation::scoring::AgentTrajectory {
                        forward_displacement: agent.displacement,
                        avg_velocity: agent.avg_velocity,
                        avg_up_alignment: agent.avg_up_alignment,
                        avg_height: agent.avg_height,
                        up_alignment_stddev: agent.up_alignment_stddev,
                        com_bounce: agent.com_bounce,
                        mechanical_work: agent.mechanical_work,
                        // Mass is unknown at this point (only the brain has
                        // it), but `cost_of_transport` was already computed
                        // during scoring. Back-derive the mass that gives
                        // the same CoT so `InverseCoT` lines up here too.
                        mass: if agent.cost_of_transport.is_finite()
                            && agent.cost_of_transport > 0.0
                            && agent.displacement > 0.0
                        {
                            agent.mechanical_work
                                / (agent.cost_of_transport * 9.81 * agent.displacement)
                        } else {
                            1.0
                        },
                        action_smoothness: agent.action_smoothness,
                        control_jerk: agent.control_jerk,
                        body_jerk: agent.body_jerk,
                        support_duty_factor: agent.support_duty_factor,
                        foot_transitions_per_sensor: agent.foot_transitions_per_sensor,
                        action_intensity: agent.action_intensity,
                    };
                    evaluate_trajectory(&traj)
                } else {
                    (f32::MIN, vec![f32::MIN, f32::MIN], vec![0.0, 0.0])
                };
            Phenotype {
                genotype: agent.genotype.clone(),
                fitness,
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
        GeneticAlgorithm::CvtMapElites => {
            parent_pool.update_cvt_map_elites(evaluated, config.population_size)
        }
        GeneticAlgorithm::NoveltySearch => {
            parent_pool.update_novelty(evaluated, config.population_size)
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
    let mutation_rate = config.mutation_rate;
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
            GeneticAlgorithm::CvtMapElites => (
                parent_pool.select_from_cvt(&mut rng),
                parent_pool.select_from_cvt(&mut rng),
            ),
            GeneticAlgorithm::NoveltySearch => (
                parent_pool.tournament_novelty(&mut rng),
                parent_pool.tournament_novelty(&mut rng),
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

    // 3b. Fresh preset injection (10%) to keep the gene pool diverse.
    //     Sources: hardcoded MORPHOLOGY_POOL ∪ hot-reloaded runtime pool from
    //     assets/morphologies/. Treats both as a single uniform distribution.
    let runtime_sources = runtime_pool.sources();
    let total_sources = MORPHOLOGY_POOL.len() + runtime_sources.len();
    for _ in 0..num_fresh {
        let pick = rng.random_range(0..total_sources);
        let morphology: String = if pick < MORPHOLOGY_POOL.len() {
            MORPHOLOGY_POOL[pick].to_string()
        } else {
            runtime_sources[pick - MORPHOLOGY_POOL.len()].1.clone()
        };
        let morphology = morphology.as_str();
        let fresh_seed: u64 = rng.random();
        if let Some((_, ni, no)) = derive_blueprint(morphology, fresh_seed, 3) {
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
            let seed = child.lsystem_seed;
            let iterations = child.lsystem_iterations;
            s.spawn(async move { derive_blueprint(morphology, seed, iterations) });
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
                child.controller.resize_io(ni, no, &mut rng);
            }
        } else {
            child.cached_topology = None;
            num_motors = 0;
            cached_blueprint = None;
        }

        next_gen.push(Agent::fresh(child, num_motors, cached_blueprint));
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
