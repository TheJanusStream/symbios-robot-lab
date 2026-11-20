//! Per-generation statistics, kept across generations for the UI's graphs and
//! the headless runner's report.

use bevy::prelude::*;

/// What one finished generation scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationSummary {
    pub generation: usize,
    /// Best walker fitness among the robots evaluated this generation.
    pub best_fitness: f32,
    /// Mean walker fitness over the robots evaluated this generation
    /// (exploded robots count as 0).
    pub mean_fitness: f32,
    /// Best fitness held anywhere in the parent pool after this generation
    /// was merged into it (elitist algorithms keep earlier champions).
    pub pool_best_fitness: f32,
    /// Furthest forward any intact robot moved during the measured window (m),
    /// a raw motion signal that does not depend on the fitness factors.
    pub max_displacement: f32,
    /// Robots the physics blew apart this generation.
    pub exploded: usize,
    /// Robots evaluated this generation.
    pub evaluated: usize,
    /// Individuals in the parent pool after merging.
    pub pool_size: usize,
    /// Different bodies among them (brain-only children share a body).
    pub distinct_bodies: usize,
    /// Generations the oldest body among them has been in the pool.
    pub oldest_body: u32,
}

/// Statistics accumulated over the whole run.
#[derive(Resource, Default)]
pub struct Telemetry {
    /// One entry per finished generation, oldest first.
    pub history: Vec<GenerationSummary>,
    /// Best walker fitness seen at any moment of the run.
    pub best_all_time: f32,
    /// Individuals on the NSGA-II Pareto front (rank 0) after the last merge.
    pub pareto_front_size: usize,
}

impl Telemetry {
    /// Clears everything for a new run.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn record(&mut self, summary: GenerationSummary) {
        self.best_all_time = self.best_all_time.max(summary.best_fitness);
        self.history.push(summary);
    }

    /// Best fitness per generation, oldest first.
    pub fn best_series(&self) -> Vec<f32> {
        self.history.iter().map(|g| g.best_fitness).collect()
    }

    /// Mean fitness per generation, oldest first.
    pub fn mean_series(&self) -> Vec<f32> {
        self.history.iter().map(|g| g.mean_fitness).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(generation: usize, best: f32) -> GenerationSummary {
        GenerationSummary {
            generation,
            best_fitness: best,
            mean_fitness: best / 2.0,
            pool_best_fitness: best,
            max_displacement: 1.0,
            exploded: 0,
            evaluated: 4,
            pool_size: 4,
            distinct_bodies: 4,
            oldest_body: 0,
        }
    }

    #[test]
    fn best_all_time_keeps_the_maximum() {
        let mut t = Telemetry::default();
        t.record(summary(0, 0.5));
        t.record(summary(1, 0.2));
        assert_eq!(t.best_all_time, 0.5);
        assert_eq!(t.best_series(), vec![0.5, 0.2]);
        assert_eq!(t.mean_series(), vec![0.25, 0.1]);
    }
}
