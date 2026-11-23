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
    /// No network: NEAT evolves a CPPN that paints a pattern over time for
    /// each driven joint, restarted whenever the joint's foot touches down
    /// (single-unit pattern generators). One genome fits any body.
    Supg,
}

impl BrainEncoding {
    pub const ALL: [BrainEncoding; 3] = [
        BrainEncoding::Direct,
        BrainEncoding::HyperNeat,
        BrainEncoding::Supg,
    ];

    /// The encoding of genotypes saved before the lab had a choice: every
    /// brain was direct.
    pub fn legacy() -> Self {
        BrainEncoding::Direct
    }

    pub fn label(self) -> &'static str {
        match self {
            BrainEncoding::Direct => "Direct NEAT",
            BrainEncoding::HyperNeat => "HyperNEAT",
            BrainEncoding::Supg => "Pattern generators",
        }
    }

    /// Parses the short names the headless CLI accepts (`direct`,
    /// `hyperneat`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "direct" => Some(BrainEncoding::Direct),
            "hyperneat" => Some(BrainEncoding::HyperNeat),
            "supg" => Some(BrainEncoding::Supg),
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

/// What a robot's brain senses beyond its body's own sensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Senses {
    /// The target's bearing (the target task).
    pub bearing: bool,
    /// The body's tilt (see [`LabConfig::tilt_sense`]).
    pub tilt: bool,
}

impl Senses {
    /// Only the body's own sensors.
    pub const NONE: Senses = Senses {
        bearing: false,
        tilt: false,
    };
    /// The target task's bearing, without tilt.
    pub const BEARING: Senses = Senses {
        bearing: true,
        tilt: false,
    };
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
    /// The share of the population kept as parents by the algorithms with a
    /// parent pool of fixed size (NSGA-II, SimpleGA, Dominated Novelty
    /// Search): 1.0 keeps as many parents as robots per generation. Fewer
    /// parents breed from the best more often; more keep more kinds of body.
    /// Half (the default) measured best with NSGA-II over 32 seeds, judged by
    /// the champions walking again under fresh conditions: +0.43 at 30
    /// generations (p = 0.08) and +0.70 at 100 (p = 0.045) against keeping
    /// them all. SimpleGA and Dominated Novelty Search gained too over 16
    /// seeds (+0.52 and +0.22, not significant), keeping half as many kinds
    /// of body.
    pub parent_share: f32,
    /// Parents that walk again each generation: the best ones by their
    /// average score are cloned into the next generation, in children's
    /// places, and their new score joins their average. Conditions change
    /// every generation (each robot walks from its own spot, on new ground,
    /// toward new targets), so one walk can flatter a robot; walking again
    /// keeps the parents honest. At most a quarter of the population; none
    /// under novelty search, whose parents last one generation. The default
    /// is 8: over 16 seeds of 100 generations on the flat floor, champions
    /// kept 89% of their recorded score under fresh conditions against 70%
    /// with none walking again (p = 0.01), broke in 3% of their fresh walks
    /// against 9%, and walked as well for it (20.4 against 20.6, failed
    /// walks counted as 0). Under the lab's earlier fitness, whose gaits ran
    /// on the clock alone and repeated themselves, it paid only where
    /// conditions change (on hills +0.24 at 30 generations) and the default
    /// was none.
    pub retests: usize,
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
    /// Brains also sense the body's tilt: how far its forward and sideways
    /// axes lean from level (two more inputs). The IMU senses jolts, not
    /// lean, which might matter on slopes; measured over 32 seeds it did not
    /// help (-0.07 on hills, -0.18 on the flat floor, neither significant),
    /// so it is off by default.
    pub tilt_sense: bool,
    /// Every robot walks with one of its legs broken (its joints limp), a
    /// different leg each generation (drawn from the run's seed, the
    /// generation and the robot), in the hope that evolution favours gaits
    /// that survive losing any leg (Cully et al. 2015 showed robots
    /// recovering from such damage). Experimental: measured over 32 seeds,
    /// robots bred this way walked worse with a broken leg than robots bred
    /// intact (0.66 against 0.87 with HyperNEAT).
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

    /// What brains sense beyond their bodies' sensors.
    pub fn senses(&self) -> Senses {
        Senses {
            bearing: self.task.steers(),
            tilt: self.tilt_sense,
        }
    }

    /// How many parents a pool of fixed size keeps: the parent share of the
    /// population, at least 2 and at most the population.
    pub fn parents(&self) -> usize {
        let parents = (self.population_size as f32 * self.parent_share).round() as usize;
        parents.clamp(2, self.population_size.max(2))
    }

    /// How many parents walk again each generation: as asked, at most a
    /// quarter of the population, and none under novelty search.
    pub fn retests_per_generation(&self) -> usize {
        if self.algorithm == GeneticAlgorithm::NoveltySearch {
            return 0;
        }
        self.retests.min(self.population_size / 4)
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
            parent_share: 0.5,
            retests: 8,
            descriptor: QdDescriptor::default(),
            terrain: Terrain::default(),
            task: Task::default(),
            tilt_sense: false,
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
    fn parents_and_retests_stay_within_the_population() {
        let config = LabConfig {
            population_size: 64,
            parent_share: 0.5,
            retests: 8,
            ..LabConfig::default()
        };
        assert_eq!(config.parents(), 32);
        assert_eq!(config.retests_per_generation(), 8);
        let small = LabConfig {
            population_size: 12,
            parent_share: 0.01,
            retests: 8,
            ..config.clone()
        };
        assert_eq!(small.parents(), 2);
        assert_eq!(small.retests_per_generation(), 3);
        let novelty = LabConfig {
            algorithm: GeneticAlgorithm::NoveltySearch,
            ..config.clone()
        };
        assert_eq!(novelty.retests_per_generation(), 0);
        assert_eq!(LabConfig::default().parents(), 32);
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
        for (name, encoding) in ["direct", "hyperneat", "supg"]
            .iter()
            .zip(BrainEncoding::ALL)
        {
            assert_eq!(BrainEncoding::from_cli_name(name), Some(encoding));
        }
        assert_eq!(BrainEncoding::from_cli_name("cppn"), None);
    }
}
