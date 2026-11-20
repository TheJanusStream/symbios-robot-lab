//! The parent pool: evaluated individuals kept between generations, and the
//! six ways the lab selects parents from it.

use crate::core::config::{GeneticAlgorithm, LabConfig};
use crate::core::descriptor::QdDescriptor;
use crate::core::genome::AgentGenotype;
use crate::simulation::fitness::NUM_OBJECTIVES;
use bevy::prelude::*;
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};
use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
use symbios_genetics::algorithms::novelty_search::{BehaviourDistance, EuclideanDistance};
use symbios_genetics::algorithms::nsga2::{Nsga2, SortWrapper};
use symbios_genetics::{Genotype, Phenotype};

/// Number of CVT centroids (and therefore the maximum CVT archive size).
const CVT_NUM_CENTROIDS: usize = 64;
/// Dimensions of the CVT centroids: a descriptor space's two axes.
const CVT_DESCRIPTOR_DIM: usize = 2;
/// Lloyd sample count for CVT centroid generation.
const CVT_LLOYD_SAMPLES: usize = 10_000;
/// Lloyd iteration count.
const CVT_LLOYD_ITERS: usize = 30;
/// CVT centroid generation seed (the same centroids every run).
const CVT_SEED: u64 = 0xC0_FFEE;
/// Grid MAP-Elites bins per descriptor dimension.
const MAP_ELITES_RESOLUTION: usize = 20;
/// Novelty search kNN window size (Lehman & Stanley default range).
const NOVELTY_K: usize = 15;
/// Novelty/fitness blend factor: 1.0 = pure novelty, 0.0 = fitness-only.
const NOVELTY_ALPHA: f32 = 0.7;
/// Share of each generation's descriptors added to the novelty archive.
const NOVELTY_ARCHIVE_RATE: f64 = 0.05;
/// Fitter neighbours a robot's dominated novelty is measured to, the
/// authors' setting.
const DNS_K: usize = 3;
/// Tournament size for SimpleGA and novelty selection.
const TOURNAMENT_SIZE: usize = 3;

/// Evaluated `Phenotype`s kept between generations, with two objectives
/// (walker fitness, then `1 / (1 + Cost of Transport)`), and the state of
/// whichever algorithm is selecting:
///   - **NSGA-II**: non-dominated sort + crowding distance, binary tournament;
///     protecting new bodies adds a third objective, youth (minus the body's
///     age), as in Cheney et al. 2018
///   - **SimpleGA**: sort by scalar fitness, tournament(3) with elitism
///   - **MAP-Elites (grid)**: regular grid archive, random parent selection
///   - **CVT-MAP-Elites**: Voronoi archive over Lloyd-generated centroids
///   - **Novelty Search**: kNN behavioural-distance scoring blended with fitness
///   - **Dominated Novelty Search**: survival by distance to fitter
///     neighbours, random parent selection
///
/// The quality-diversity algorithms measure robots in the configured
/// descriptor space ([`QdDescriptor`]) and leave failed robots out.
#[derive(Resource, Default)]
pub struct ParentPool {
    pool: Vec<Phenotype<AgentGenotype>>,
    /// Pareto rank per individual (0 = non-dominated front).
    ranks: Vec<usize>,
    /// Crowding distance per individual.
    crowding: Vec<f32>,
    /// Grid MAP-Elites archive: cell key → index in `pool`.
    archive: BTreeMap<Vec<usize>, usize>,
    /// Lloyd centroids for CVT-MAP-Elites, generated on first use.
    cvt_centroids: Option<Vec<Vec<f32>>>,
    /// CVT archive: centroid index → index in `pool`.
    cvt_archive: BTreeMap<usize, usize>,
    /// Behavioural archive for novelty search: whole descriptors, measured in
    /// whichever space is configured.
    novelty_archive: Vec<Vec<f32>>,
    /// Blended novelty score per individual (parallel to `pool`).
    novelty_scores: Vec<f32>,
    /// The descriptor space of the last merge, which the archives are laid
    /// out in.
    space: QdDescriptor,
}

impl ParentPool {
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Best walker fitness held in the pool.
    pub fn best_fitness(&self) -> f32 {
        self.pool.iter().map(|p| p.fitness).fold(0.0_f32, f32::max)
    }

    /// Different bodies in the pool: brain-only children share their
    /// parent's body.
    pub fn distinct_bodies(&self) -> usize {
        self.pool
            .iter()
            .map(|p| {
                let g = &p.genotype;
                (g.morphology.as_str(), g.lsystem_seed, g.lsystem_iterations)
            })
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Generations the oldest body in the pool has been in it.
    pub fn oldest_body(&self) -> u32 {
        self.pool
            .iter()
            .map(|p| p.genotype.body_age)
            .max()
            .unwrap_or(0)
    }

    /// Individuals on the Pareto front (rank 0).
    pub fn front_size(&self) -> usize {
        self.ranks.iter().filter(|&&r| r == 0).count()
    }

    /// The descriptor space the archives are laid out in.
    pub fn descriptor_space(&self) -> QdDescriptor {
        self.space
    }

    /// Each walking individual's point in the descriptor space, with its
    /// fitness.
    pub fn points(&self) -> Vec<([f32; 2], f32)> {
        self.pool
            .iter()
            .filter(|p| is_viable(p))
            .map(|p| (self.space.project(&p.descriptor), p.fitness))
            .collect()
    }

    /// Occupied cells in the grid MAP-Elites archive.
    pub fn archive_size(&self) -> usize {
        self.archive.len()
    }

    /// Grid resolution (bins per dimension) for MAP-Elites.
    pub fn archive_resolution(&self) -> usize {
        MAP_ELITES_RESOLUTION
    }

    /// Occupied cells in the CVT archive.
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

    /// Descriptors in the novelty archive.
    pub fn novelty_archive_size(&self) -> usize {
        self.novelty_archive.len()
    }

    /// CVT centroids (descriptor-space points) for visualisation.
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

    /// `(cell_x, cell_y, fitness)` for each occupied grid archive cell.
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

    /// `(objective_0, objective_1, rank)` per walking individual. Failed
    /// robots' `f32::MIN` objectives are finite but would stretch a plot's
    /// axes to the limits of `f32`.
    pub fn objective_data(&self) -> Vec<(f32, f32, usize)> {
        self.pool
            .iter()
            .zip(self.ranks.iter())
            .filter(|(pheno, _)| is_viable(pheno))
            .filter_map(|(pheno, &rank)| {
                let o0 = *pheno.objectives.first()?;
                let o1 = *pheno.objectives.get(1)?;
                (o0.is_finite() && o1.is_finite()).then_some((o0, o1, rank))
            })
            .collect()
    }

    /// Clears the pool for a new run.
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
    /// as CSV and returns the rows written: each elite's cell, its fitness
    /// and its point on the descriptor space's two axes.
    pub fn export_archive_csv<W: std::io::Write>(
        &self,
        algorithm: GeneticAlgorithm,
        mut writer: W,
    ) -> std::io::Result<usize> {
        let [x_axis, y_axis] = self.space.axis_names().map(|name| name.replace(' ', "_"));
        let mut rows = 0;
        match algorithm {
            GeneticAlgorithm::MapElites => {
                writeln!(writer, "cell_x,cell_y,fitness,{x_axis},{y_axis}")?;
                for (cell, &idx) in &self.archive {
                    let pheno = &self.pool[idx];
                    let x = cell.first().copied().unwrap_or(0);
                    let y = cell.get(1).copied().unwrap_or(0);
                    let [d0, d1] = self.space.project(&pheno.descriptor);
                    writeln!(writer, "{x},{y},{:.6},{d0:.6},{d1:.6}", pheno.fitness)?;
                    rows += 1;
                }
            }
            GeneticAlgorithm::CvtMapElites => {
                writeln!(writer, "centroid_idx,fitness,{x_axis},{y_axis}")?;
                for (&cidx, &pool_idx) in &self.cvt_archive {
                    let pheno = &self.pool[pool_idx];
                    let [d0, d1] = self.space.project(&pheno.descriptor);
                    writeln!(writer, "{cidx},{:.6},{d0:.6},{d1:.6}", pheno.fitness)?;
                    rows += 1;
                }
            }
            _ => {}
        }
        Ok(rows)
    }

    /// Merges a generation's evaluated individuals into the pool with the
    /// configured algorithm, keeping at most the population size where the
    /// algorithm caps. The bodies already in the pool grow a generation older
    /// first.
    ///
    /// Only the selecting algorithm keeps its archive. A grid or CVT archive
    /// whose algorithm takes over, or whose descriptor space changed, is laid
    /// out afresh from the pool, so switching mid-run carries the parents
    /// over.
    pub fn merge(
        &mut self,
        config: &LabConfig,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        rng: &mut impl Rng,
    ) {
        for pheno in &mut self.pool {
            pheno.genotype.body_age = pheno.genotype.body_age.saturating_add(1);
        }
        let (algorithm, cap, space) = (config.algorithm, config.population_size, config.descriptor);
        if algorithm != GeneticAlgorithm::MapElites {
            self.archive.clear();
        }
        if algorithm != GeneticAlgorithm::CvtMapElites {
            self.cvt_archive.clear();
        }
        if algorithm != GeneticAlgorithm::NoveltySearch {
            self.novelty_scores.clear();
        }
        match algorithm {
            GeneticAlgorithm::Nsga2 => {
                self.update_nsga2(evaluated, cap, config.protects_new_bodies())
            }
            GeneticAlgorithm::SimpleGA => self.update_simple(evaluated, cap),
            GeneticAlgorithm::MapElites => self.update_map_elites(evaluated, space),
            GeneticAlgorithm::CvtMapElites => self.update_cvt_map_elites(evaluated, space),
            GeneticAlgorithm::NoveltySearch => self.update_novelty(evaluated, space, rng),
            GeneticAlgorithm::DominatedNovelty => self.update_dns(evaluated, cap, space),
        }
        self.space = space;
    }

    /// One offspring of two parents selected by the configured algorithm
    /// (the fitter parent first, as NEAT crossover expects): half the
    /// children vary only the body and half only the brain (see
    /// [`LabConfig::varies_body_or_brain`]), or body and brain are crossed
    /// and mutated together. `None` when the pool holds fewer than two
    /// individuals.
    pub fn breed(&self, config: &LabConfig, rng: &mut impl Rng) -> Option<AgentGenotype> {
        if self.pool.len() < 2 {
            return None;
        }
        let algorithm = config.algorithm;
        let (a, b) = (self.select(algorithm, rng), self.select(algorithm, rng));
        let (fitter, weaker) = if self.pool[a].fitness >= self.pool[b].fitness {
            (a, b)
        } else {
            (b, a)
        };
        let (fitter, weaker) = (&self.pool[fitter].genotype, &self.pool[weaker].genotype);
        let rate = config.mutation_rate;
        let mut child = if !config.varies_body_or_brain() {
            let mut child = fitter.crossover(weaker, rng);
            child.mutate(rng, rate);
            child
        } else if rng.random_bool(0.5) {
            fitter.body_child(weaker, rate, rng)
        } else {
            fitter.brain_child(weaker, rate, rng)
        };
        child.inherit_body_age([fitter, weaker]);
        Some(child)
    }

    fn select(&self, algorithm: GeneticAlgorithm, rng: &mut impl Rng) -> usize {
        match algorithm {
            GeneticAlgorithm::Nsga2 => self.binary_tournament_nsga2(rng),
            GeneticAlgorithm::SimpleGA => self.tournament_simple(rng),
            GeneticAlgorithm::MapElites => self.select_from_archive(rng),
            GeneticAlgorithm::CvtMapElites => self.select_from_cvt(rng),
            GeneticAlgorithm::NoveltySearch => self.tournament_novelty(rng),
            GeneticAlgorithm::DominatedNovelty => rng.random_range(0..self.pool.len()),
        }
    }

    // ── NSGA-II ──────────────────────────────────────────────────────

    /// Pick two random individuals and return the better: lower rank, ties
    /// broken by larger crowding distance.
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

    /// Combines the evaluated individuals with the pool, sorts into fronts
    /// with crowding distance, and truncates to `cap`. With
    /// `protect_new_bodies`, youth joins the objectives.
    fn update_nsga2(
        &mut self,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        cap: usize,
        protect_new_bodies: bool,
    ) {
        let mut combined = std::mem::take(&mut self.pool);
        combined.extend(evaluated);
        if combined.is_empty() {
            return;
        }
        for pheno in &mut combined {
            score_youth(pheno, protect_new_bodies);
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

    /// Tournament of three: the highest scalar fitness wins.
    fn tournament_simple(&self, rng: &mut impl Rng) -> usize {
        let n = self.pool.len();
        let mut best = rng.random_range(0..n);
        for _ in 1..TOURNAMENT_SIZE.min(n) {
            let candidate = rng.random_range(0..n);
            if self.pool[candidate].fitness > self.pool[best].fitness
                || self.pool[best].fitness.is_nan()
            {
                best = candidate;
            }
        }
        best
    }

    /// Combines the evaluated individuals with the pool, sorts by fitness
    /// (descending, NaN last) and truncates to `cap`: the whole retained pool
    /// is the elite.
    fn update_simple(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, cap: usize) {
        let mut combined = std::mem::take(&mut self.pool);
        combined.extend(evaluated);
        if combined.is_empty() {
            return;
        }
        combined.sort_by(|a, b| match (a.fitness.is_nan(), b.fitness.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.fitness.total_cmp(&a.fitness),
        });
        combined.truncate(cap);
        self.ranks = vec![0; combined.len()];
        self.crowding = vec![f32::INFINITY; combined.len()];
        self.pool = combined;
    }

    // ── Archives (grid MAP-Elites and CVT) ───────────────────────────

    /// The individuals an archive is to file this merge: the evaluated ones
    /// and, when the archive does not lay out the pool in `space` (its
    /// algorithm or descriptor space just took over), the pool's members too,
    /// emptying the pool and the archive to lay them out afresh.
    fn filing_candidates<K>(
        &mut self,
        archive: impl FnOnce(&mut Self) -> &mut BTreeMap<K, usize>,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        space: QdDescriptor,
    ) -> Vec<Phenotype<AgentGenotype>> {
        let pool_len = self.pool.len();
        let stale = space != self.space;
        let archive = archive(self);
        if !stale && archive.len() == pool_len {
            return evaluated;
        }
        archive.clear();
        let mut candidates = std::mem::take(&mut self.pool);
        candidates.extend(evaluated);
        candidates
    }

    /// Files `pheno` under `key` unless the archive's occupant there walks at
    /// least as well.
    fn file<K: Ord>(
        pool: &mut Vec<Phenotype<AgentGenotype>>,
        archive: &mut BTreeMap<K, usize>,
        key: K,
        pheno: Phenotype<AgentGenotype>,
    ) {
        let beaten = archive
            .get(&key)
            .is_some_and(|&idx| pheno.fitness <= pool[idx].fitness);
        if !beaten {
            archive.insert(key, pool.len());
            pool.push(pheno);
        }
    }

    /// Rebuilds the pool from an archive's occupants, in key order.
    fn compact<K: Ord + Clone>(&mut self, archive: &BTreeMap<K, usize>) -> BTreeMap<K, usize> {
        let mut pool = Vec::with_capacity(archive.len());
        let mut compacted = BTreeMap::new();
        for (key, &old) in archive {
            compacted.insert(key.clone(), pool.len());
            pool.push(self.pool[old].clone());
        }
        self.pool = pool;
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
        compacted
    }

    // ── MAP-Elites (grid) ────────────────────────────────────────────

    fn point_to_cell(point: [f32; 2]) -> Vec<usize> {
        point
            .iter()
            .map(|&v| {
                let scaled = v.clamp(0.0, 1.0) * MAP_ELITES_RESOLUTION as f32;
                (scaled.floor() as usize).min(MAP_ELITES_RESOLUTION - 1)
            })
            .collect()
    }

    fn select_from_archive(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&Vec<usize>> = self.archive.keys().collect();
        if keys.is_empty() {
            return rng.random_range(0..self.pool.len());
        }
        self.archive[keys[rng.random_range(0..keys.len())]]
    }

    /// Each walking individual's point in the descriptor space picks a grid
    /// cell, which keeps its fittest occupant; the pool is rebuilt from the
    /// archive.
    fn update_map_elites(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, space: QdDescriptor) {
        let candidates = self.filing_candidates(|pool| &mut pool.archive, evaluated, space);
        let mut archive = std::mem::take(&mut self.archive);
        for pheno in candidates.into_iter().filter(is_viable) {
            let cell = Self::point_to_cell(space.project(&pheno.descriptor));
            Self::file(&mut self.pool, &mut archive, cell, pheno);
        }
        self.archive = self.compact(&archive);
    }

    // ── CVT-MAP-Elites ───────────────────────────────────────────────

    fn ensure_cvt_centroids(&mut self) {
        if self.cvt_centroids.is_some() {
            return;
        }
        let cvt = CvtMapElites::<AgentGenotype>::with_lloyd(
            CVT_NUM_CENTROIDS,
            CVT_DESCRIPTOR_DIM,
            CVT_LLOYD_SAMPLES,
            CVT_LLOYD_ITERS,
            0.0, // mutation_rate: unused, the lab never calls `.step()`
            1,   // batch_size: unused, likewise
            CVT_SEED,
        );
        self.cvt_centroids = Some(cvt.centroids().to_vec());
    }

    fn nearest_centroid(&self, point: [f32; 2]) -> Option<usize> {
        let centroids = self.cvt_centroids.as_ref()?;
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: f32 = c.iter().zip(point).map(|(a, b)| (a - b).powi(2)).sum();
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn select_from_cvt(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&usize> = self.cvt_archive.keys().collect();
        if keys.is_empty() {
            return rng.random_range(0..self.pool.len());
        }
        self.cvt_archive[keys[rng.random_range(0..keys.len())]]
    }

    /// One elite per centroid nearest a walking individual's point; the pool
    /// is rebuilt from the archive.
    fn update_cvt_map_elites(
        &mut self,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        space: QdDescriptor,
    ) {
        self.ensure_cvt_centroids();
        let candidates = self.filing_candidates(|pool| &mut pool.cvt_archive, evaluated, space);
        let mut archive = std::mem::take(&mut self.cvt_archive);
        for pheno in candidates.into_iter().filter(is_viable) {
            if let Some(centroid) = self.nearest_centroid(space.project(&pheno.descriptor)) {
                Self::file(&mut self.pool, &mut archive, centroid, pheno);
            }
        }
        self.cvt_archive = self.compact(&archive);
    }

    // ── Novelty Search ───────────────────────────────────────────────

    /// Tournament of three on the blended novelty+fitness score.
    fn tournament_novelty(&self, rng: &mut impl Rng) -> usize {
        let n = self.pool.len();
        let score = |i: usize| self.novelty_scores.get(i).copied().unwrap_or(0.0);
        let mut best = rng.random_range(0..n);
        for _ in 1..TOURNAMENT_SIZE.min(n) {
            let candidate = rng.random_range(0..n);
            if score(candidate) > score(best) {
                best = candidate;
            }
        }
        best
    }

    /// Mean distance from `point` to its `k` nearest `others` (0 with no
    /// others).
    fn mean_knn_distance(point: [f32; 2], others: &[[f32; 2]], k: usize) -> f32 {
        let mut distances: Vec<f32> = others
            .iter()
            .map(|other| EuclideanDistance.distance(&point, other))
            .filter(|d| !d.is_nan())
            .collect();
        if distances.is_empty() {
            return 0.0;
        }
        distances.sort_by(f32::total_cmp);
        let take = k.min(distances.len());
        distances.iter().take(take).sum::<f32>() / take as f32
    }

    /// Scores each evaluated individual by novelty (kNN distance in the
    /// descriptor space to the rest of its generation and the archive)
    /// blended with fitness, replaces the pool with the batch, and archives a
    /// random share of the descriptors. Failed robots score no novelty and
    /// are neither archived nor measured against.
    fn update_novelty(
        &mut self,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        space: QdDescriptor,
        rng: &mut impl Rng,
    ) {
        let points: Vec<Option<[f32; 2]>> = evaluated
            .iter()
            .map(|p| is_viable(p).then(|| space.project(&p.descriptor)))
            .collect();
        let archived: Vec<[f32; 2]> = self
            .novelty_archive
            .iter()
            .map(|d| space.project(d))
            .collect();
        let scores: Vec<f32> = evaluated
            .iter()
            .zip(&points)
            .enumerate()
            .map(|(i, (pheno, point))| {
                let novelty = point.map_or(0.0, |point| {
                    let others: Vec<[f32; 2]> = points
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .filter_map(|(_, other)| *other)
                        .chain(archived.iter().copied())
                        .collect();
                    Self::mean_knn_distance(point, &others, NOVELTY_K)
                });
                // Failed robots carry f32::MIN fitness, which sinks their
                // blended score below any walking robot's.
                let fitness = if pheno.fitness.is_nan() {
                    0.0
                } else {
                    pheno.fitness
                };
                NOVELTY_ALPHA * novelty + (1.0 - NOVELTY_ALPHA) * fitness
            })
            .collect();

        for (pheno, point) in evaluated.iter().zip(&points) {
            if point.is_some() && rng.random_bool(NOVELTY_ARCHIVE_RATE) {
                self.novelty_archive.push(pheno.descriptor.clone());
            }
        }

        self.ranks = vec![0; evaluated.len()];
        self.crowding = vec![f32::INFINITY; evaluated.len()];
        self.pool = evaluated;
        self.novelty_scores = scores;
    }

    // ── Dominated Novelty Search ─────────────────────────────────────

    /// Dominated Novelty Search (Bahlous-Boldi et al., GECCO 2025): each
    /// walking robot, old or new, scores its mean distance in the descriptor
    /// space to its [`DNS_K`] nearest fitter robots, and the `cap` highest
    /// scores survive. A robot survives by being the best of its region or
    /// far from anyone better: local competition without a grid or bounds.
    /// The fittest robot has no fitter neighbour and always survives (only
    /// strictly fitter robots count, where the authors also count equals, so
    /// a tie never costs the best its place).
    fn update_dns(
        &mut self,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        cap: usize,
        space: QdDescriptor,
    ) {
        let mut combined = std::mem::take(&mut self.pool);
        combined.extend(evaluated);
        combined.retain(is_viable);
        let points: Vec<[f32; 2]> = combined
            .iter()
            .map(|p| space.project(&p.descriptor))
            .collect();
        let scores: Vec<f32> = combined
            .iter()
            .zip(&points)
            .map(|(pheno, &point)| {
                let fitter: Vec<[f32; 2]> = combined
                    .iter()
                    .zip(&points)
                    .filter(|(other, _)| other.fitness > pheno.fitness)
                    .map(|(_, &other)| other)
                    .collect();
                if fitter.is_empty() {
                    f32::INFINITY
                } else {
                    Self::mean_knn_distance(point, &fitter, DNS_K)
                }
            })
            .collect();
        let mut order: Vec<usize> = (0..combined.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        order.truncate(cap);
        self.pool = order.iter().map(|&i| combined[i].clone()).collect();
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
    }
}

/// Whether a phenotype is a robot that walked: failed robots score `f32::MIN`
/// (see [`crate::simulation::generation::phenotype_of`]), and a NaN would
/// poison every distance and comparison it entered.
fn is_viable(pheno: &Phenotype<AgentGenotype>) -> bool {
    pheno.fitness.is_finite()
        && pheno.fitness > f32::MIN
        && pheno.descriptor.iter().all(|v| v.is_finite())
}

/// Sets a phenotype's NSGA-II objectives for the coming selection: the
/// scored ones, then youth (minus the body's age) when protecting new bodies.
/// A robot that failed gets `f32::MIN` youth too, so no failure reaches the
/// front just by being new.
fn score_youth(pheno: &mut Phenotype<AgentGenotype>, protect_new_bodies: bool) {
    pheno.objectives.truncate(NUM_OBJECTIVES);
    if protect_new_bodies {
        let youth = if is_viable(pheno) {
            -(pheno.genotype.body_age as f32)
        } else {
            f32::MIN
        };
        pheno.objectives.push(youth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::BrainEncoding;
    use crate::core::descriptor::NUM_TRAITS;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios_neat::NeatConfig;

    /// A phenotype whose gait traits are `gait` (no body traits).
    fn phenotype(fitness: f32, gait: [f32; 2], rng: &mut Pcg64) -> Phenotype<AgentGenotype> {
        let config = NeatConfig {
            num_inputs: 3,
            num_outputs: 2,
            ..NeatConfig::default()
        };
        let mut descriptor = vec![0.0; NUM_TRAITS];
        descriptor[..2].copy_from_slice(&gait);
        Phenotype {
            genotype: AgentGenotype::new(
                "omega: B(1) J B(1)".to_string(),
                0,
                BrainEncoding::Direct,
                symbios_neat::NeatGenome::fully_connected(config, rng),
            ),
            fitness,
            objectives: vec![fitness, 0.5],
            descriptor,
        }
    }

    fn failed(rng: &mut Pcg64) -> Phenotype<AgentGenotype> {
        let mut pheno = phenotype(f32::MIN, [0.0, 0.0], rng);
        pheno.objectives = vec![f32::MIN; NUM_OBJECTIVES];
        pheno
    }

    fn config(algorithm: GeneticAlgorithm, cap: usize) -> LabConfig {
        LabConfig {
            algorithm,
            population_size: cap,
            ..LabConfig::default()
        }
    }

    fn protected(cap: usize) -> LabConfig {
        LabConfig {
            protect_new_bodies: true,
            ..config(GeneticAlgorithm::Nsga2, cap)
        }
    }

    #[test]
    fn a_pool_of_one_breeds_nothing() {
        let mut rng = Pcg64::seed_from_u64(3);
        let mut pool = ParentPool::default();
        let config = config(GeneticAlgorithm::MapElites, 8);
        pool.merge(
            &config,
            vec![phenotype(1.0, [0.5, 0.5], &mut rng)],
            &mut rng,
        );
        assert_eq!(pool.len(), 1);
        assert!(pool.breed(&config, &mut rng).is_none());
    }

    #[test]
    fn map_elites_keeps_the_fittest_per_cell_and_no_failures() {
        let mut rng = Pcg64::seed_from_u64(4);
        let mut pool = ParentPool::default();
        let batch = vec![
            phenotype(0.2, [0.51, 0.51], &mut rng),
            phenotype(0.9, [0.52, 0.52], &mut rng),
            phenotype(0.4, [0.05, 0.95], &mut rng),
            failed(&mut rng),
        ];
        pool.merge(&config(GeneticAlgorithm::MapElites, 8), batch, &mut rng);
        assert_eq!(pool.archive_size(), 2);
        assert_eq!(pool.best_fitness(), 0.9);
        assert!(pool.pool.iter().all(is_viable));
    }

    #[test]
    fn the_pareto_plot_leaves_failed_robots_out() {
        let mut rng = Pcg64::seed_from_u64(13);
        let mut pool = ParentPool::default();
        let batch = vec![phenotype(0.6, [0.5, 0.5], &mut rng), failed(&mut rng)];
        pool.merge(&config(GeneticAlgorithm::Nsga2, 8), batch, &mut rng);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.objective_data(), vec![(0.6, 0.5, 0)]);
    }

    #[test]
    fn nsga2_caps_the_pool() {
        let mut rng = Pcg64::seed_from_u64(5);
        let mut pool = ParentPool::default();
        let batch = (0..10)
            .map(|i| phenotype(i as f32 / 10.0, [0.5, 0.5], &mut rng))
            .collect();
        let config = config(GeneticAlgorithm::Nsga2, 6);
        pool.merge(&config, batch, &mut rng);
        assert_eq!(pool.len(), 6);
        assert!(pool.front_size() >= 1);
        assert!(pool.breed(&config, &mut rng).is_some());
    }

    #[test]
    fn novelty_archives_are_reproducible_from_the_seed() {
        let run = |seed: u64| {
            let mut rng = Pcg64::seed_from_u64(seed);
            let mut pool = ParentPool::default();
            let batch = (0..40)
                .map(|i| phenotype(0.1, [i as f32 / 40.0, 0.5], &mut rng))
                .collect();
            let config = config(GeneticAlgorithm::NoveltySearch, 40);
            pool.merge(&config, batch, &mut rng);
            pool.novelty_archive_size()
        };
        assert_eq!(run(9), run(9));
    }

    /// The fittest robot always survives, and among the rest a robot far
    /// from anyone fitter outlasts one crowded by fitter neighbours.
    #[test]
    fn dominated_novelty_keeps_the_best_and_the_isolated() {
        let mut rng = Pcg64::seed_from_u64(10);
        let mut pool = ParentPool::default();
        let batch = vec![
            phenotype(1.0, [0.5, 0.5], &mut rng),
            phenotype(0.9, [0.51, 0.5], &mut rng),
            phenotype(0.8, [0.5, 0.51], &mut rng),
            phenotype(0.7, [0.52, 0.52], &mut rng),
            // Weak, but alone in its corner of the gait space.
            phenotype(0.1, [0.05, 0.95], &mut rng),
            failed(&mut rng),
        ];
        let config = config(GeneticAlgorithm::DominatedNovelty, 3);
        pool.merge(&config, batch, &mut rng);
        let kept: Vec<f32> = pool.pool.iter().map(|p| p.fitness).collect();
        assert_eq!(kept.len(), 3);
        assert!(kept.contains(&1.0), "the best was lost: {kept:?}");
        assert!(kept.contains(&0.1), "the isolated robot was lost: {kept:?}");
        assert!(pool.breed(&config, &mut rng).is_some());
        // Its points are what the panel draws.
        assert_eq!(pool.points().len(), 3);
    }

    /// Body traits lay the grid out differently from gait traits; switching
    /// the descriptor space mid-run re-files the archive instead of mixing
    /// the two.
    #[test]
    fn a_new_descriptor_space_lays_the_archive_out_afresh() {
        let mut rng = Pcg64::seed_from_u64(11);
        let mut pool = ParentPool::default();
        let batch: Vec<_> = (0..4)
            .map(|i| {
                // Four gaits, but two bodies: feet 0.25 or 0.5, one length.
                let mut p = phenotype(0.1 * (i + 1) as f32, [0.2 * i as f32, 0.5], &mut rng);
                p.descriptor[2] = if i < 2 { 0.25 } else { 0.5 };
                p.descriptor[3] = 0.4;
                p
            })
            .collect();
        let gait = config(GeneticAlgorithm::MapElites, 8);
        pool.merge(&gait, batch, &mut rng);
        assert_eq!(pool.archive_size(), 4);
        let body = LabConfig {
            descriptor: QdDescriptor::Body,
            ..gait.clone()
        };
        pool.merge(&body, Vec::new(), &mut rng);
        assert_eq!(pool.archive_size(), 2);
        assert_eq!(pool.descriptor_space(), QdDescriptor::Body);
        // The fitter robot of each body kept its cell.
        let mut kept: Vec<f32> = pool.pool.iter().map(|p| p.fitness).collect();
        kept.sort_by(f32::total_cmp);
        assert_eq!(kept, vec![0.2, 0.4]);
    }

    /// An archive taking over from another algorithm files the pool it
    /// inherits, and one that lost its turn is dropped rather than left
    /// pointing into a pool it no longer indexes.
    #[test]
    fn switching_algorithms_carries_the_parents_over() {
        let mut rng = Pcg64::seed_from_u64(12);
        let mut pool = ParentPool::default();
        let spread = |rng: &mut Pcg64| -> Vec<_> {
            (0..10)
                .map(|i| phenotype(0.1 + 0.05 * i as f32, [0.1 * i as f32, 0.3], rng))
                .collect()
        };
        pool.merge(
            &config(GeneticAlgorithm::CvtMapElites, 16),
            spread(&mut rng),
            &mut rng,
        );
        assert!(pool.cvt_archive_size() > 1);
        // A small NSGA-II pool, then CVT again.
        pool.merge(&config(GeneticAlgorithm::Nsga2, 2), Vec::new(), &mut rng);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.cvt_archive_size(), 0);
        let cvt = config(GeneticAlgorithm::CvtMapElites, 16);
        pool.merge(&cvt, Vec::new(), &mut rng);
        assert_eq!(pool.cvt_archive_size(), pool.len());
        assert!(!pool.is_empty());
        assert_eq!(pool.breed(&cvt, &mut rng).is_some(), pool.len() >= 2);
        // MAP-Elites inherits the CVT elites.
        pool.merge(
            &config(GeneticAlgorithm::MapElites, 16),
            Vec::new(),
            &mut rng,
        );
        assert_eq!(pool.archive_size(), pool.len());
        assert!(!pool.is_empty());
    }

    #[test]
    fn protection_scores_youth_and_survivors_grow_older() {
        let mut rng = Pcg64::seed_from_u64(6);
        let mut pool = ParentPool::default();
        let batch = vec![
            phenotype(0.8, [0.5, 0.5], &mut rng),
            phenotype(0.4, [0.5, 0.5], &mut rng),
            failed(&mut rng),
        ];
        pool.merge(&protected(8), batch, &mut rng);
        let youth = |pool: &ParentPool| -> Vec<f32> {
            pool.pool
                .iter()
                .filter(|p| p.fitness > f32::MIN)
                .map(|p| p.objectives[NUM_OBJECTIVES])
                .collect()
        };
        assert!(pool.pool.iter().all(|p| p.objectives.len() == 3));
        assert_eq!(youth(&pool), vec![0.0, 0.0]);
        let failed = pool.pool.iter().find(|p| p.fitness == f32::MIN);
        assert_eq!(failed.expect("kept").objectives[NUM_OBJECTIVES], f32::MIN);

        // Another generation passes: the survivors are a generation older.
        pool.merge(&protected(8), Vec::new(), &mut rng);
        assert_eq!(youth(&pool), vec![-1.0, -1.0]);
        assert_eq!(pool.oldest_body(), 1);
        // Without protection the objectives are the scored ones again.
        pool.merge(&config(GeneticAlgorithm::Nsga2, 8), Vec::new(), &mut rng);
        assert!(
            pool.pool
                .iter()
                .all(|p| p.objectives.len() == NUM_OBJECTIVES)
        );
        assert_eq!(pool.oldest_body(), 2);
    }

    #[test]
    fn a_new_body_dominates_an_old_one_that_walks_no_better() {
        let mut rng = Pcg64::seed_from_u64(7);
        let mut old = phenotype(0.5, [0.5, 0.5], &mut rng);
        old.genotype.body_age = 5;
        let young = phenotype(0.5, [0.5, 0.5], &mut rng);
        let mut pool = ParentPool::default();
        pool.merge(&protected(8), vec![old, young], &mut rng);
        let rank_of_age = |age: u32| {
            let i = pool.pool.iter().position(|p| p.genotype.body_age == age);
            pool.ranks[i.expect("both kept")]
        };
        assert_eq!(rank_of_age(0), 0);
        assert_eq!(rank_of_age(5), 1);
    }

    /// Six parents, each with its own body, in a pool merged with `config`.
    fn six_bodies(config: &LabConfig, rng: &mut Pcg64) -> ParentPool {
        let mut pool = ParentPool::default();
        let batch = (0..6)
            .map(|i| {
                let mut p = phenotype(0.2 + 0.1 * i as f32, [0.5, 0.5], rng);
                p.genotype.morphology = format!("#define len {}\nomega: B(len) J B(len)", i + 1);
                p.genotype.lsystem_seed = i;
                p
            })
            .collect();
        pool.merge(config, batch, rng);
        pool
    }

    /// Breeds 200 children; returns how many changed the body only, the
    /// brain only, and both, checking each one's body age.
    fn breed_200(pool: &ParentPool, config: &LabConfig, rng: &mut Pcg64) -> [usize; 3] {
        let brain = |g: &AgentGenotype| {
            serde_json::to_string(&(&g.controller, g.rhythm_hz)).expect("serializes")
        };
        let parents: Vec<AgentGenotype> = pool.pool.iter().map(|p| p.genotype.clone()).collect();
        let mut counts = [0; 3];
        for _ in 0..200 {
            let child = pool.breed(config, rng).expect("breeds");
            let kept_body = parents.iter().find(|p| child.same_body(p));
            let kept_brain = parents.iter().any(|p| brain(p) == brain(&child));
            match kept_body {
                Some(parent) => assert_eq!(child.body_age, parent.body_age + 1),
                None => assert_eq!(child.body_age, 0),
            }
            match (kept_body.is_some(), kept_brain) {
                (false, true) => counts[0] += 1,
                (true, false) => counts[1] += 1,
                (false, false) => counts[2] += 1,
                (true, true) => {}
            }
        }
        counts
    }

    /// Varying body or brain (the default, and implied by protection): every
    /// child keeps a parent's body or a parent's brain, and both kinds of
    /// children are bred. Turned off, children change both.
    #[test]
    fn children_vary_the_body_or_the_brain_unless_told_otherwise() {
        let mut rng = Pcg64::seed_from_u64(8);
        let default = LabConfig {
            mutation_rate: 1.0,
            ..config(GeneticAlgorithm::Nsga2, 8)
        };
        let protected_only = LabConfig {
            vary_body_or_brain: false,
            ..LabConfig {
                mutation_rate: 1.0,
                ..protected(8)
            }
        };
        for config in [&default, &protected_only] {
            assert!(config.varies_body_or_brain());
            let pool = six_bodies(config, &mut rng);
            assert_eq!(pool.distinct_bodies(), 6);
            let [new_bodies, new_brains, both] = breed_200(&pool, config, &mut rng);
            assert_eq!(both, 0, "a child changed both");
            assert!(
                new_bodies > 20 && new_brains > 20,
                "{new_bodies} / {new_brains}"
            );
        }
        let classic = LabConfig {
            vary_body_or_brain: false,
            ..default
        };
        let pool = six_bodies(&classic, &mut rng);
        let [_, _, both] = breed_200(&pool, &classic, &mut rng);
        assert!(both > 100, "only {both} classic children changed both");
    }
}
