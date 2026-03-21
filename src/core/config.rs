use bevy::prelude::*;

/// Which evolutionary algorithm drives selection and breeding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneticAlgorithm {
    /// Single-objective: tournament selection (size 3) with elitism.
    SimpleGA,
    /// Multi-objective: Pareto non-dominated sort + crowding distance.
    #[default]
    Nsga2,
    /// Quality-diversity: archive grid indexed by behavioural descriptor.
    MapElites,
}

impl GeneticAlgorithm {
    pub const ALL: [GeneticAlgorithm; 3] = [
        GeneticAlgorithm::SimpleGA,
        GeneticAlgorithm::Nsga2,
        GeneticAlgorithm::MapElites,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GeneticAlgorithm::SimpleGA => "SimpleGA",
            GeneticAlgorithm::Nsga2 => "NSGA-II",
            GeneticAlgorithm::MapElites => "MAP-Elites",
        }
    }
}

impl std::fmt::Display for GeneticAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// User-facing simulation parameters exposed via the UI panel.
#[derive(Resource)]
pub struct LabConfig {
    pub population_size: usize,
    pub generation_duration: f32, // Seconds per round
    pub time_scale: f32,
    pub auto_restart: bool,
    pub algorithm: GeneticAlgorithm,
    pub mutation_rate: f32,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            generation_duration: 15.0,
            time_scale: 5.0,
            auto_restart: true,
            algorithm: GeneticAlgorithm::default(),
            mutation_rate: 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = LabConfig::default();
        assert_eq!(config.population_size, 64);
        assert_eq!(config.generation_duration, 15.0);
        assert_eq!(config.time_scale, 5.0);
        assert!(config.auto_restart);
    }
}
