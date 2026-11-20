//! The run's lineage: its best robot as of generation 0 and of every so many
//! generations since, for a race of generations, in which champions from
//! across the run's history walk side by side.

use crate::simulation::hall_of_fame::Champion;
use bevy::prelude::*;

/// Generations between milestones at the start of a run.
pub const MILESTONE_STRIDE: usize = 10;
/// Milestones kept: past this many, the stride doubles and every other
/// milestone leaves, so a long run keeps milestones spread over its whole
/// history in bounded memory.
pub const MAX_MILESTONES: usize = 16;
/// Robots in a race of generations.
pub const RACE_LANES: usize = 6;

/// The run's best robot as of a generation: the fittest walking member of
/// the parent pool once that generation's robots had joined it (its
/// average, if it had walked again).
#[derive(Clone)]
pub struct Milestone {
    pub generation: usize,
    pub champion: Champion,
}

/// The run's milestones, at generations on the stride, and its latest best.
#[derive(Resource)]
pub struct Lineage {
    milestones: Vec<Milestone>,
    stride: usize,
    latest: Option<Milestone>,
}

impl Default for Lineage {
    fn default() -> Self {
        Self {
            milestones: Vec::new(),
            stride: MILESTONE_STRIDE,
            latest: None,
        }
    }
}

impl Lineage {
    /// Records the run's best as of `generation`: as the latest best, and
    /// as a milestone if the generation is on the stride.
    pub fn record(&mut self, generation: usize, champion: Champion) {
        let milestone = Milestone {
            generation,
            champion,
        };
        if generation.is_multiple_of(self.stride) {
            self.milestones.push(milestone.clone());
            if self.milestones.len() > MAX_MILESTONES {
                self.stride *= 2;
                let stride = self.stride;
                self.milestones
                    .retain(|m| m.generation.is_multiple_of(stride));
            }
        }
        self.latest = Some(milestone);
    }

    pub fn milestones(&self) -> &[Milestone] {
        &self.milestones
    }

    /// Up to `lanes` robots to race, oldest first: the first milestone, the
    /// latest best, and milestones evenly spread between (all of them while
    /// the run is young).
    pub fn race(&self, lanes: usize) -> Vec<&Milestone> {
        let mut field: Vec<&Milestone> = self.milestones.iter().collect();
        if let Some(latest) = &self.latest
            && field
                .last()
                .is_none_or(|m| m.generation < latest.generation)
        {
            field.push(latest);
        }
        if field.len() > lanes {
            let last = field.len() - 1;
            let spread = (lanes.max(2) - 1) as f32;
            let mut picks: Vec<usize> = (0..lanes)
                .map(|i| (i as f32 * last as f32 / spread).round() as usize)
                .collect();
            if lanes == 1 {
                picks = vec![last];
            }
            picks.dedup();
            field = picks.into_iter().map(|i| field[i]).collect();
        }
        field
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{BrainEncoding, Senses};
    use crate::core::presets::MORPHOLOGY_POOL;
    use crate::simulation::lifecycle::fresh_agent;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn champion(fitness: f32) -> Champion {
        let mut rng = Pcg64::seed_from_u64(3);
        let agent = fresh_agent(
            MORPHOLOGY_POOL[0],
            BrainEncoding::HyperNeat,
            Senses::NONE,
            &mut rng,
        )
        .expect("derives");
        Champion {
            genotype: agent.genotype,
            fitness: Some(fitness),
            generation: None,
            walks: 1,
        }
    }

    fn generations<'a>(race: impl IntoIterator<Item = &'a Milestone>) -> Vec<usize> {
        race.into_iter().map(|m| m.generation).collect()
    }

    #[test]
    fn a_young_run_races_all_it_has() {
        let mut lineage = Lineage::default();
        assert!(lineage.race(RACE_LANES).is_empty());
        lineage.record(0, champion(0.1));
        assert_eq!(generations(lineage.race(RACE_LANES)), [0]);
        for generation in 1..=13 {
            lineage.record(generation, champion(0.1 * generation as f32));
        }
        // Milestones 0 and 10, and the latest best.
        assert_eq!(generations(lineage.milestones()), [0, 10]);
        assert_eq!(generations(lineage.race(RACE_LANES)), [0, 10, 13]);
        assert!((lineage.race(RACE_LANES)[2].champion.fitness.unwrap() - 1.3).abs() < 1e-6);
        assert_eq!(generations(lineage.race(1)), [13]);
        lineage.clear();
        assert!(lineage.race(RACE_LANES).is_empty());
    }

    /// Milestones thin out as the run grows, staying spread over its whole
    /// history, and a race spreads its lanes over them.
    #[test]
    fn a_long_run_keeps_milestones_spread_and_bounded() {
        let mut lineage = Lineage::default();
        for generation in 0..=1000 {
            lineage.record(generation, champion(1.0));
            assert!(lineage.milestones().len() <= MAX_MILESTONES);
        }
        let kept = generations(lineage.milestones());
        assert_eq!(kept[0], 0);
        let stride = kept[1];
        assert!(kept.windows(2).all(|w| w[1] - w[0] == stride), "{kept:?}");
        assert!(1000 - kept.last().unwrap() < stride);
        let race = generations(lineage.race(RACE_LANES));
        assert_eq!(race.len(), RACE_LANES);
        assert_eq!(race[0], 0);
        assert_eq!(*race.last().unwrap(), 1000);
        assert!(race.windows(2).all(|w| w[0] < w[1]), "{race:?}");

        let mut lineage = Lineage::default();
        for generation in 0..=200 {
            lineage.record(generation, champion(1.0));
        }
        assert_eq!(
            generations(lineage.race(RACE_LANES)),
            [0, 40, 80, 120, 160, 200]
        );
    }
}
