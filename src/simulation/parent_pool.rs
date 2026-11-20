//! The parent pool: evaluated individuals kept between generations, and the
//! five ways the lab selects parents from it.

use crate::core::config::GeneticAlgorithm;
use crate::core::genome::AgentGenotype;
use bevy::prelude::*;
use rand::Rng;
use std::collections::BTreeMap;
use symbios_genetics::algorithms::cvt_map_elites::CvtMapElites;
use symbios_genetics::algorithms::novelty_search::{BehaviourDistance, EuclideanDistance};
use symbios_genetics::algorithms::nsga2::{Nsga2, SortWrapper};
use symbios_genetics::{Genotype, Phenotype};

/// Number of CVT centroids (and therefore the maximum CVT archive size).
const CVT_NUM_CENTROIDS: usize = 64;
/// Descriptor dimensionality: `evaluate_trajectory`'s
/// `(support_duty_factor, action_intensity)`.
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
/// Tournament size for SimpleGA and novelty selection.
const TOURNAMENT_SIZE: usize = 3;

/// Evaluated `Phenotype`s kept between generations, with two objectives
/// (walker fitness, then `1 / (1 + Cost of Transport)`), and the state of
/// whichever algorithm is selecting:
///   - **NSGA-II**: non-dominated sort + crowding distance, binary tournament
///   - **SimpleGA**: sort by scalar fitness, tournament(3) with elitism
///   - **MAP-Elites (grid)**: regular grid archive, random parent selection
///   - **CVT-MAP-Elites**: Voronoi archive over Lloyd-generated centroids
///   - **Novelty Search**: kNN behavioural-distance scoring blended with fitness
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
    /// Behavioural archive for novelty search.
    novelty_archive: Vec<Vec<f32>>,
    /// Blended novelty score per individual (parallel to `pool`).
    novelty_scores: Vec<f32>,
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

    /// Individuals on the Pareto front (rank 0).
    pub fn front_size(&self) -> usize {
        self.ranks.iter().filter(|&&r| r == 0).count()
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

    /// `(objective_0, objective_1, rank)` per individual with finite objectives.
    pub fn objective_data(&self) -> Vec<(f32, f32, usize)> {
        self.pool
            .iter()
            .zip(self.ranks.iter())
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
    /// as CSV and returns the rows written. Format:
    /// `cell_key, fitness, descriptor_0, descriptor_1`.
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
                    writeln!(writer, "{x},{y},{:.6},{d0:.6},{d1:.6}", pheno.fitness)?;
                    rows += 1;
                }
            }
            GeneticAlgorithm::CvtMapElites => {
                writeln!(writer, "centroid_idx,fitness,desc_0,desc_1")?;
                for (&cidx, &pool_idx) in &self.cvt_archive {
                    let pheno = &self.pool[pool_idx];
                    let d0 = pheno.descriptor.first().copied().unwrap_or(0.0);
                    let d1 = pheno.descriptor.get(1).copied().unwrap_or(0.0);
                    writeln!(writer, "{cidx},{:.6},{d0:.6},{d1:.6}", pheno.fitness)?;
                    rows += 1;
                }
            }
            _ => {}
        }
        Ok(rows)
    }

    /// Merges a generation's evaluated individuals into the pool with the
    /// given algorithm, keeping at most `cap` where the algorithm caps.
    pub fn merge(
        &mut self,
        algorithm: GeneticAlgorithm,
        evaluated: Vec<Phenotype<AgentGenotype>>,
        cap: usize,
        rng: &mut impl Rng,
    ) {
        match algorithm {
            GeneticAlgorithm::Nsga2 => self.update_nsga2(evaluated, cap),
            GeneticAlgorithm::SimpleGA => self.update_simple(evaluated, cap),
            GeneticAlgorithm::MapElites => self.update_map_elites(evaluated),
            GeneticAlgorithm::CvtMapElites => self.update_cvt_map_elites(evaluated),
            GeneticAlgorithm::NoveltySearch => self.update_novelty(evaluated, rng),
        }
    }

    /// One offspring: two parents selected by `algorithm`, crossed (the
    /// fitter parent first, as NEAT crossover expects) and mutated. `None`
    /// when the pool holds fewer than two individuals.
    pub fn breed(
        &self,
        algorithm: GeneticAlgorithm,
        mutation_rate: f32,
        rng: &mut impl Rng,
    ) -> Option<AgentGenotype> {
        if self.pool.len() < 2 {
            return None;
        }
        let (a, b) = (self.select(algorithm, rng), self.select(algorithm, rng));
        let (fitter, weaker) = if self.pool[a].fitness >= self.pool[b].fitness {
            (a, b)
        } else {
            (b, a)
        };
        let mut child = self.pool[fitter]
            .genotype
            .crossover(&self.pool[weaker].genotype, rng);
        child.mutate(rng, mutation_rate);
        Some(child)
    }

    fn select(&self, algorithm: GeneticAlgorithm, rng: &mut impl Rng) -> usize {
        match algorithm {
            GeneticAlgorithm::Nsga2 => self.binary_tournament_nsga2(rng),
            GeneticAlgorithm::SimpleGA => self.tournament_simple(rng),
            GeneticAlgorithm::MapElites => self.select_from_archive(rng),
            GeneticAlgorithm::CvtMapElites => self.select_from_cvt(rng),
            GeneticAlgorithm::NoveltySearch => self.tournament_novelty(rng),
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
    /// with crowding distance, and truncates to `cap`.
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
        self.archive.clear();
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

    // ── MAP-Elites (grid) ────────────────────────────────────────────

    fn descriptor_to_cell(descriptor: &[f32]) -> Vec<usize> {
        descriptor
            .iter()
            .map(|&v| {
                let scaled = v.clamp(0.0, 1.0) * MAP_ELITES_RESOLUTION as f32;
                (scaled.floor() as usize).min(MAP_ELITES_RESOLUTION - 1)
            })
            .collect()
    }

    fn select_from_archive(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&Vec<usize>> = self.archive.keys().collect();
        self.archive[keys[rng.random_range(0..keys.len())]]
    }

    /// Each individual's descriptor picks a grid cell, which keeps its
    /// fittest occupant; the pool is rebuilt from the archive.
    fn update_map_elites(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>) {
        for pheno in evaluated {
            if !pheno.fitness.is_finite() || pheno.descriptor.iter().any(|v| v.is_nan()) {
                continue;
            }
            let cell = Self::descriptor_to_cell(&pheno.descriptor);
            let beaten = self
                .archive
                .get(&cell)
                .is_some_and(|&idx| pheno.fitness <= self.pool[idx].fitness);
            if !beaten {
                self.archive.insert(cell, self.pool.len());
                self.pool.push(pheno);
            }
        }
        let mut pool = Vec::with_capacity(self.archive.len());
        let mut archive = BTreeMap::new();
        for (cell, &old) in &self.archive {
            archive.insert(cell.clone(), pool.len());
            pool.push(self.pool[old].clone());
        }
        self.pool = pool;
        self.archive = archive;
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
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

    fn nearest_centroid(&self, descriptor: &[f32]) -> Option<usize> {
        let centroids = self.cvt_centroids.as_ref()?;
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: f32 = c.iter().zip(descriptor).map(|(a, b)| (a - b).powi(2)).sum();
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn select_from_cvt(&self, rng: &mut impl Rng) -> usize {
        let keys: Vec<&usize> = self.cvt_archive.keys().collect();
        self.cvt_archive[keys[rng.random_range(0..keys.len())]]
    }

    /// One elite per centroid; the pool is rebuilt from the archive.
    fn update_cvt_map_elites(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>) {
        self.ensure_cvt_centroids();
        for pheno in evaluated {
            if !pheno.fitness.is_finite()
                || pheno.descriptor.iter().any(|v| v.is_nan())
                || pheno.descriptor.len() != CVT_DESCRIPTOR_DIM
            {
                continue;
            }
            let Some(centroid) = self.nearest_centroid(&pheno.descriptor) else {
                continue;
            };
            let beaten = self
                .cvt_archive
                .get(&centroid)
                .is_some_and(|&idx| pheno.fitness <= self.pool[idx].fitness);
            if !beaten {
                self.cvt_archive.insert(centroid, self.pool.len());
                self.pool.push(pheno);
            }
        }
        let mut pool = Vec::with_capacity(self.cvt_archive.len());
        let mut archive = BTreeMap::new();
        for (&centroid, &old) in &self.cvt_archive {
            archive.insert(centroid, pool.len());
            pool.push(self.pool[old].clone());
        }
        self.pool = pool;
        self.cvt_archive = archive;
        self.ranks = vec![0; self.pool.len()];
        self.crowding = vec![f32::INFINITY; self.pool.len()];
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

    /// Mean distance from `descriptor` to its `NOVELTY_K` nearest `others`.
    fn mean_knn_distance(descriptor: &[f32], others: &[Vec<f32>]) -> f32 {
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
        distances.sort_by(f32::total_cmp);
        let take = NOVELTY_K.min(distances.len());
        distances.iter().take(take).sum::<f32>() / take as f32
    }

    /// Scores each evaluated individual by novelty (kNN distance to the rest
    /// of its generation and the archive) blended with fitness, replaces the
    /// pool with the batch, and archives a random share of the descriptors.
    fn update_novelty(&mut self, evaluated: Vec<Phenotype<AgentGenotype>>, rng: &mut impl Rng) {
        self.archive.clear();
        self.cvt_archive.clear();
        let valid = |d: &[f32]| !d.is_empty() && !d.iter().any(|v| v.is_nan());

        let descriptors: Vec<Vec<f32>> = evaluated.iter().map(|p| p.descriptor.clone()).collect();
        let scores: Vec<f32> = evaluated
            .iter()
            .enumerate()
            .map(|(i, pheno)| {
                let novelty = if valid(&pheno.descriptor) {
                    let others: Vec<Vec<f32>> = descriptors
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, d)| d.clone())
                        .chain(self.novelty_archive.iter().cloned())
                        .collect();
                    Self::mean_knn_distance(&pheno.descriptor, &others)
                } else {
                    0.0
                };
                // Invalid robots carry f32::MIN fitness, which sinks their
                // blended score below any valid robot's however novel they are.
                let fitness = if pheno.fitness.is_nan() {
                    0.0
                } else {
                    pheno.fitness
                };
                NOVELTY_ALPHA * novelty + (1.0 - NOVELTY_ALPHA) * fitness
            })
            .collect();

        for descriptor in descriptors {
            if valid(&descriptor) && rng.random_bool(NOVELTY_ARCHIVE_RATE) {
                self.novelty_archive.push(descriptor);
            }
        }

        self.ranks = vec![0; evaluated.len()];
        self.crowding = vec![f32::INFINITY; evaluated.len()];
        self.pool = evaluated;
        self.novelty_scores = scores;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::BrainEncoding;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios_neat::NeatConfig;

    fn phenotype(fitness: f32, descriptor: [f32; 2], rng: &mut Pcg64) -> Phenotype<AgentGenotype> {
        let config = NeatConfig {
            num_inputs: 3,
            num_outputs: 2,
            ..NeatConfig::default()
        };
        Phenotype {
            genotype: AgentGenotype::new(
                "omega: B(1) J B(1)".to_string(),
                0,
                BrainEncoding::Direct,
                symbios_neat::NeatGenome::fully_connected(config, rng),
            ),
            fitness,
            objectives: vec![fitness, 0.5],
            descriptor: descriptor.to_vec(),
        }
    }

    #[test]
    fn a_pool_of_one_breeds_nothing() {
        let mut rng = Pcg64::seed_from_u64(3);
        let mut pool = ParentPool::default();
        pool.merge(
            GeneticAlgorithm::MapElites,
            vec![phenotype(1.0, [0.5, 0.5], &mut rng)],
            8,
            &mut rng,
        );
        assert_eq!(pool.len(), 1);
        assert!(
            pool.breed(GeneticAlgorithm::MapElites, 0.3, &mut rng)
                .is_none()
        );
    }

    #[test]
    fn map_elites_keeps_the_fittest_per_cell() {
        let mut rng = Pcg64::seed_from_u64(4);
        let mut pool = ParentPool::default();
        let batch = vec![
            phenotype(0.2, [0.51, 0.51], &mut rng),
            phenotype(0.9, [0.52, 0.52], &mut rng),
            phenotype(0.4, [0.05, 0.95], &mut rng),
        ];
        pool.merge(GeneticAlgorithm::MapElites, batch, 8, &mut rng);
        assert_eq!(pool.archive_size(), 2);
        assert_eq!(pool.best_fitness(), 0.9);
    }

    #[test]
    fn nsga2_caps_the_pool() {
        let mut rng = Pcg64::seed_from_u64(5);
        let mut pool = ParentPool::default();
        let batch = (0..10)
            .map(|i| phenotype(i as f32 / 10.0, [0.5, 0.5], &mut rng))
            .collect();
        pool.merge(GeneticAlgorithm::Nsga2, batch, 6, &mut rng);
        assert_eq!(pool.len(), 6);
        assert!(pool.front_size() >= 1);
        assert!(pool.breed(GeneticAlgorithm::Nsga2, 0.3, &mut rng).is_some());
    }

    #[test]
    fn novelty_archives_are_reproducible_from_the_seed() {
        let run = |seed: u64| {
            let mut rng = Pcg64::seed_from_u64(seed);
            let mut pool = ParentPool::default();
            let batch = (0..40)
                .map(|i| phenotype(0.1, [i as f32 / 40.0, 0.5], &mut rng))
                .collect();
            pool.merge(GeneticAlgorithm::NoveltySearch, batch, 40, &mut rng);
            pool.novelty_archive_size()
        };
        assert_eq!(run(9), run(9));
    }
}
