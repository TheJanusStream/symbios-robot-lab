use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Which evolutionary algorithm drives selection and breeding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneticAlgorithm {
    /// Single-objective: tournament selection (size 3) with elitism.
    SimpleGA,
    /// Multi-objective: Pareto non-dominated sort + crowding distance.
    #[default]
    Nsga2,
    /// Quality-diversity: regular grid archive indexed by behavioural descriptor.
    MapElites,
    /// Quality-diversity: Voronoi (CVT) archive over centroids generated via Lloyd's
    /// algorithm. Better coverage when the feasible region is irregular.
    CvtMapElites,
    /// Open-ended search: selection driven by behavioural novelty (kNN distance to
    /// archive + population), blended with fitness via `alpha`.
    NoveltySearch,
}

impl GeneticAlgorithm {
    pub const ALL: [GeneticAlgorithm; 5] = [
        GeneticAlgorithm::SimpleGA,
        GeneticAlgorithm::Nsga2,
        GeneticAlgorithm::MapElites,
        GeneticAlgorithm::CvtMapElites,
        GeneticAlgorithm::NoveltySearch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GeneticAlgorithm::SimpleGA => "SimpleGA",
            GeneticAlgorithm::Nsga2 => "NSGA-II",
            GeneticAlgorithm::MapElites => "MAP-Elites (grid)",
            GeneticAlgorithm::CvtMapElites => "CVT-MAP-Elites",
            GeneticAlgorithm::NoveltySearch => "Novelty Search",
        }
    }

    /// Parses the short names the headless CLI accepts (`simple`, `nsga2`,
    /// `map-elites`, `cvt`, `novelty`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "simple" => Some(GeneticAlgorithm::SimpleGA),
            "nsga2" => Some(GeneticAlgorithm::Nsga2),
            "map-elites" => Some(GeneticAlgorithm::MapElites),
            "cvt" => Some(GeneticAlgorithm::CvtMapElites),
            "novelty" => Some(GeneticAlgorithm::NoveltySearch),
            _ => None,
        }
    }
}

impl std::fmt::Display for GeneticAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// How a genotype's controller genome becomes a robot's brain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainEncoding {
    /// NEAT evolves the network itself: one input per sensor value, one
    /// output per driven joint, resized whenever the body changes.
    Direct,
    /// NEAT evolves a CPPN that paints the network onto the body: neurons sit
    /// at the sensors and joints, and the CPPN sets each link's weight from
    /// where its two neurons sit. One genome fits any body.
    HyperNeat,
}

impl BrainEncoding {
    pub const ALL: [BrainEncoding; 2] = [BrainEncoding::Direct, BrainEncoding::HyperNeat];

    /// The encoding of genotypes saved before the lab had a choice: every
    /// brain was direct.
    pub fn legacy() -> Self {
        BrainEncoding::Direct
    }

    pub fn label(self) -> &'static str {
        match self {
            BrainEncoding::Direct => "Direct NEAT",
            BrainEncoding::HyperNeat => "HyperNEAT",
        }
    }

    /// Parses the short names the headless CLI accepts (`direct`,
    /// `hyperneat`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "direct" => Some(BrainEncoding::Direct),
            "hyperneat" => Some(BrainEncoding::HyperNeat),
            _ => None,
        }
    }
}

impl std::fmt::Display for BrainEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// User-facing simulation parameters exposed via the UI panel and the
/// headless runner.
#[derive(Resource, Debug, Clone)]
pub struct LabConfig {
    pub population_size: usize,
    /// Seconds of simulated time per generation.
    pub generation_duration: f32,
    /// Speed-up of simulated time over wall-clock time in the windowed app.
    /// The headless runner ignores it and steps as fast as it can.
    pub time_scale: f32,
    pub auto_restart: bool,
    pub algorithm: GeneticAlgorithm,
    /// The brain new robots get (HyperNEAT by default: over 32 seeds of 30
    /// generations its best fitness came out about a fifth above direct
    /// NEAT's). Bred robots keep their parents' encoding, so changing it
    /// mid-run reaches the whole population only after a restart.
    pub brain: BrainEncoding,
    pub mutation_rate: f32,
    /// Magnitude of a single sideways impulse (N·s) applied to each robot's
    /// body roughly halfway through its evaluation window. Default 0.0 = off.
    /// Non-zero values pressure controllers toward perturbation-robust gaits.
    pub perturbation_strength: f32,
    /// Seed for every random choice the evolution makes. `None` draws one
    /// from the operating system when the run starts (and logs it, so the run
    /// can be repeated); headless runs and tests set it.
    pub seed: Option<u64>,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            generation_duration: 15.0,
            time_scale: 5.0,
            auto_restart: true,
            algorithm: GeneticAlgorithm::default(),
            brain: BrainEncoding::HyperNeat,
            mutation_rate: 0.3,
            perturbation_strength: 0.0,
            seed: None,
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
        assert_eq!(config.brain, BrainEncoding::HyperNeat);
        assert_eq!(config.seed, None);
    }

    #[test]
    fn every_algorithm_has_a_cli_name() {
        let names = ["simple", "nsga2", "map-elites", "cvt", "novelty"];
        for (name, algorithm) in names.iter().zip(GeneticAlgorithm::ALL) {
            assert_eq!(GeneticAlgorithm::from_cli_name(name), Some(algorithm));
        }
        assert_eq!(GeneticAlgorithm::from_cli_name("unknown"), None);
    }

    #[test]
    fn every_brain_encoding_has_a_cli_name() {
        for (name, encoding) in ["direct", "hyperneat"].iter().zip(BrainEncoding::ALL) {
            assert_eq!(BrainEncoding::from_cli_name(name), Some(encoding));
        }
        assert_eq!(BrainEncoding::from_cli_name("cppn"), None);
    }
}
