use crate::core::descriptor::QdDescriptor;
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
    /// Quality-diversity without a grid (Bahlous-Boldi et al. 2025): robots
    /// survive by their distance to the nearest fitter robots, so each is
    /// the best of its region or far from anyone better.
    DominatedNovelty,
}

impl GeneticAlgorithm {
    pub const ALL: [GeneticAlgorithm; 6] = [
        GeneticAlgorithm::SimpleGA,
        GeneticAlgorithm::Nsga2,
        GeneticAlgorithm::MapElites,
        GeneticAlgorithm::CvtMapElites,
        GeneticAlgorithm::NoveltySearch,
        GeneticAlgorithm::DominatedNovelty,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GeneticAlgorithm::SimpleGA => "SimpleGA",
            GeneticAlgorithm::Nsga2 => "NSGA-II",
            GeneticAlgorithm::MapElites => "MAP-Elites (grid)",
            GeneticAlgorithm::CvtMapElites => "CVT-MAP-Elites",
            GeneticAlgorithm::NoveltySearch => "Novelty Search",
            GeneticAlgorithm::DominatedNovelty => "Dominated Novelty Search",
        }
    }

    /// Parses the short names the headless CLI accepts (`simple`, `nsga2`,
    /// `map-elites`, `cvt`, `novelty`, `dns`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "simple" => Some(GeneticAlgorithm::SimpleGA),
            "nsga2" => Some(GeneticAlgorithm::Nsga2),
            "map-elites" => Some(GeneticAlgorithm::MapElites),
            "cvt" => Some(GeneticAlgorithm::CvtMapElites),
            "novelty" => Some(GeneticAlgorithm::NoveltySearch),
            "dns" => Some(GeneticAlgorithm::DominatedNovelty),
            _ => None,
        }
    }

    /// Whether the algorithm spreads robots over a descriptor space
    /// ([`LabConfig::descriptor`]).
    pub fn uses_descriptor(self) -> bool {
        !matches!(self, GeneticAlgorithm::SimpleGA | GeneticAlgorithm::Nsga2)
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

/// What robots are scored on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Walk as far as possible the way the robot faces when the measured
    /// window opens.
    #[default]
    Forward,
    /// Walk to a target: the robot's brain reads the target's bearing (two
    /// more inputs) and its walk scores the progress it makes toward it.
    Target,
}

impl Task {
    pub const ALL: [Task; 2] = [Task::Forward, Task::Target];

    pub fn label(self) -> &'static str {
        match self {
            Task::Forward => "Walk forward",
            Task::Target => "Walk to a target",
        }
    }

    /// Parses the short names the headless CLI accepts (`forward`,
    /// `target`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "forward" => Some(Task::Forward),
            "target" => Some(Task::Target),
            _ => None,
        }
    }

    /// Whether brains read a target's bearing.
    pub fn steers(self) -> bool {
        self == Task::Target
    }
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// The ground robots walk on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Terrain {
    /// The flat arena floor.
    #[default]
    Flat,
    /// Bumps a few metres across and up to 20 cm high.
    Rough,
    /// Hills tens of metres across and up to 3 m high: slopes to climb and
    /// descend.
    Hilly,
}

impl Terrain {
    pub const ALL: [Terrain; 3] = [Terrain::Flat, Terrain::Rough, Terrain::Hilly];

    pub fn label(self) -> &'static str {
        match self {
            Terrain::Flat => "Flat",
            Terrain::Rough => "Rough",
            Terrain::Hilly => "Hilly",
        }
    }

    /// Parses the short names the headless CLI accepts (`flat`, `rough`,
    /// `hilly`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "flat" => Some(Terrain::Flat),
            "rough" => Some(Terrain::Rough),
            "hilly" => Some(Terrain::Hilly),
            _ => None,
        }
    }
}

impl std::fmt::Display for Terrain {
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
    /// Each child varies either its body (crossed and mutated, keeping the
    /// fitter parent's brain) or its brain (keeping that parent's body), half
    /// and half, rather than both at once. A new body is then first tried
    /// with a brain that walks, and a new brain on a body it was bred for.
    /// On by default: over 32 seeds with NSGA-II it raised the pool's best
    /// fitness by about a quarter at 30 generations (+0.50, p = 0.05) and a
    /// seventh at 100, and halved the robots the physics blew apart.
    pub vary_body_or_brain: bool,
    /// Morphological innovation protection (Cheney et al. 2018), with
    /// NSGA-II: youth (generations since the body last changed) joins the
    /// objectives, so a new body has time to tune its brain before it must
    /// outwalk bodies whose brains have been tuned for many generations.
    /// Implies [`Self::vary_body_or_brain`]. Off by default: over 100
    /// generations it kept more and younger bodies (48 distinct against 36)
    /// and gained on varying alone from a slower start (-0.42 at 30
    /// generations, -0.16 at 100), the pattern Cheney et al. saw pay off over
    /// far longer runs.
    pub protect_new_bodies: bool,
    /// The two traits the quality-diversity algorithms spread robots over:
    /// gait, body, or feet and duty factor.
    pub descriptor: QdDescriptor,
    /// The ground: flat, or rough or hilly ground generated afresh each
    /// generation (from the run's seed and the generation), so no gait wins
    /// by fitting one patch of ground. Takes effect from the next generation.
    pub terrain: Terrain,
    /// What robots are scored on: walking forward, or walking to a target
    /// whose bearing their brains read. New robots get the task's brain
    /// inputs; bred ones are rewired for it (a direct brain gains or loses
    /// the two bearing inputs, a HyperNEAT brain is painted with or without
    /// them).
    pub task: Task,
    /// Every robot walks with one of its legs broken (its joints limp), a
    /// different leg each generation (drawn from the run's seed, the
    /// generation and the robot), so evolution favours gaits that survive
    /// losing any leg (Cully et al. 2015 showed robots recovering from such
    /// damage).
    pub leg_damage: bool,
    /// Magnitude of a single sideways impulse (N·s) applied to each robot's
    /// body roughly halfway through its evaluation window. Default 0.0 = off.
    /// Non-zero values pressure controllers toward perturbation-robust gaits.
    pub perturbation_strength: f32,
    /// Seed for every random choice the evolution makes. `None` draws one
    /// from the operating system when the run starts (and logs it, so the run
    /// can be repeated); headless runs and tests set it.
    pub seed: Option<u64>,
}

impl LabConfig {
    /// Whether new bodies are protected: asked for, and the algorithm is
    /// NSGA-II, the one whose Pareto selection can weigh youth against
    /// fitness.
    pub fn protects_new_bodies(&self) -> bool {
        self.protect_new_bodies && self.algorithm == GeneticAlgorithm::Nsga2
    }

    /// Whether each child varies its body or its brain, not both: asked for,
    /// or implied by protecting new bodies.
    pub fn varies_body_or_brain(&self) -> bool {
        self.vary_body_or_brain || self.protects_new_bodies()
    }
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
            vary_body_or_brain: true,
            protect_new_bodies: false,
            descriptor: QdDescriptor::default(),
            terrain: Terrain::default(),
            task: Task::default(),
            leg_damage: false,
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
        assert!(config.varies_body_or_brain() && !config.protects_new_bodies());
        assert_eq!(config.seed, None);
    }

    #[test]
    fn every_algorithm_has_a_cli_name() {
        let names = ["simple", "nsga2", "map-elites", "cvt", "novelty", "dns"];
        for (name, algorithm) in names.iter().zip(GeneticAlgorithm::ALL) {
            assert_eq!(GeneticAlgorithm::from_cli_name(name), Some(algorithm));
        }
        assert_eq!(GeneticAlgorithm::from_cli_name("unknown"), None);
    }

    #[test]
    fn every_task_has_a_cli_name() {
        for (name, task) in ["forward", "target"].iter().zip(Task::ALL) {
            assert_eq!(Task::from_cli_name(name), Some(task));
        }
        assert_eq!(Task::from_cli_name("run"), None);
        assert!(Task::Target.steers() && !Task::Forward.steers());
    }

    #[test]
    fn every_terrain_has_a_cli_name() {
        for (name, terrain) in ["flat", "rough", "hilly"].iter().zip(Terrain::ALL) {
            assert_eq!(Terrain::from_cli_name(name), Some(terrain));
        }
        assert_eq!(Terrain::from_cli_name("steep"), None);
    }

    #[test]
    fn every_brain_encoding_has_a_cli_name() {
        for (name, encoding) in ["direct", "hyperneat"].iter().zip(BrainEncoding::ALL) {
            assert_eq!(BrainEncoding::from_cli_name(name), Some(encoding));
        }
        assert_eq!(BrainEncoding::from_cli_name("cppn"), None);
    }
}
