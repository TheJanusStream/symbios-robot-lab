//! Evolution without a window: no renderer and no frame pacing, fixed steps
//! as fast as the machine allows. Used by the tests and `examples/evolve.rs`.
//!
//! ```no_run
//! use symbios_robot_lab::core::config::LabConfig;
//! use symbios_robot_lab::headless::HeadlessLab;
//!
//! let config = LabConfig { seed: Some(7), ..LabConfig::default() };
//! let mut lab = HeadlessLab::new(config);
//! let telemetry = lab.run(10, |g| println!("{}: {:.3}", g.generation, g.best_fitness)).unwrap();
//! println!("best: {}", telemetry.best_all_time);
//! ```

use crate::core::config::LabConfig;
use crate::core::descriptor::BodyTraits;
use crate::core::genome::AgentGenotype;
use crate::core::population::Population;
use crate::simulation::SimulationPlugin;
use crate::simulation::clock::GenerationClock;
use crate::simulation::fitness::{AgentTrajectory, trajectory_of};
use crate::simulation::generation::phenotype_of;
use crate::simulation::hall_of_fame::{HallOfFame, Immigrants};
use crate::simulation::lifecycle::SpawnRequest;
use crate::simulation::physics::FIXED_HZ;
use crate::simulation::rng::LabRng;
use crate::simulation::telemetry::{GenerationSummary, Telemetry};
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use std::time::Duration;

/// Fixed steps per app update. Bevy caps one frame's virtual time at 250 ms
/// (16 steps at 64 Hz), so this stays under it.
const STEPS_PER_UPDATE: u32 = 10;

/// Updates allowed per generation beyond its own steps before a run is
/// declared stalled (spawning and breeding take an update or two).
const STALL_SLACK_UPDATES: usize = 16;
/// Seeds of the conditions [`fresh_scores`] walks robots under, far from the
/// seeds runs use.
const FRESH_SEED_BASE: u64 = 0xF2E5_0000_0000;
/// Clones of a robot walking each fresh condition, from different spots.
pub const FRESH_CLONES: usize = 2;

/// Why a headless run stopped early.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadlessError {
    /// No generation finished within the step budget.
    Stalled { finished: usize, wanted: usize },
}

impl std::fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeadlessError::Stalled { finished, wanted } => write!(
                f,
                "the run stalled after {finished} of {wanted} generations"
            ),
        }
    }
}

impl std::error::Error for HeadlessError {}

/// A windowless lab: the simulation plugin on a minimal Bevy app whose clock
/// is advanced by hand.
pub struct HeadlessLab {
    app: App,
}

impl HeadlessLab {
    /// A lab that will run with `config`. `auto_restart` is forced on, since
    /// a headless run has nobody to press "next generation".
    pub fn new(config: LabConfig) -> Self {
        let config = LabConfig {
            auto_restart: true,
            ..config
        };
        let mut app = App::new();
        // One thread: avian's parallel solver orders contacts by whichever
        // thread finishes first, and a chaotic simulation turns that into a
        // different run. Single-threaded, a seed repeats a whole run exactly
        // (breeding too, since symbios 1.5.3); run several seeds as separate
        // processes to use more cores.
        let single_threaded = TaskPoolPlugin {
            task_pool_options: TaskPoolOptions::with_num_threads(1),
        };
        app.add_plugins((MinimalPlugins.set(single_threaded), TransformPlugin))
            .insert_resource(config)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                f64::from(STEPS_PER_UPDATE) / FIXED_HZ,
            )))
            .add_plugins(SimulationPlugin);
        app.finish();
        app.cleanup();
        Self { app }
    }

    /// Starts a new run and steps until `generations` generations have
    /// finished, calling `on_generation` after each one. Returns the run's
    /// telemetry.
    pub fn run(
        &mut self,
        generations: usize,
        mut on_generation: impl FnMut(&GenerationSummary),
    ) -> Result<&Telemetry, HeadlessError> {
        self.app.world_mut().write_message(SpawnRequest);
        let config = self.app.world().resource::<LabConfig>();
        let steps_per_generation =
            (f64::from(config.generation_duration) * FIXED_HZ).ceil() as usize;
        let budget = generations
            * (steps_per_generation / STEPS_PER_UPDATE as usize + STALL_SLACK_UPDATES)
            + STALL_SLACK_UPDATES;

        let mut reported = 0;
        for _ in 0..budget {
            self.app.update();
            let history = &self.app.world().resource::<Telemetry>().history;
            while reported < history.len().min(generations) {
                on_generation(&history[reported]);
                reported += 1;
            }
            if reported == generations {
                return Ok(self.telemetry());
            }
        }
        Err(HeadlessError::Stalled {
            finished: reported,
            wanted: generations,
        })
    }

    pub fn telemetry(&self) -> &Telemetry {
        self.app.world().resource::<Telemetry>()
    }

    pub fn population(&self) -> &Population {
        self.app.world().resource::<Population>()
    }

    /// The session's champions so far.
    pub fn hall_of_fame(&self) -> &HallOfFame {
        self.app.world().resource::<HallOfFame>()
    }

    /// Sends a robot into evolution: it joins the next run's first
    /// generation (or the next generation bred, mid-run).
    pub fn send_into_evolution(&mut self, genotype: AgentGenotype) {
        self.app
            .world_mut()
            .resource_mut::<Immigrants>()
            .0
            .push(genotype);
    }

    /// The seed the run drew from (set or drawn at startup).
    pub fn seed(&self) -> Option<u64> {
        self.app.world().get_resource::<LabRng>().map(LabRng::seed)
    }

    /// The underlying app, for tests that inspect the world.
    pub fn app_mut(&mut self) -> &mut App {
        &mut self.app
    }
}

/// How `genotype` walks under `conditions` fresh conditions: for each, a new
/// run with a seed of its own (so its own spots, ground, targets and broken
/// legs, whatever `config` asks for) in which clones of the robot walk one
/// generation. Returns each condition's mean walker fitness, a failed walk
/// counting as 0.
///
/// The honest measure of a robot evolved under changing conditions: the
/// score it was selected for came from one walk, and one walk can flatter.
pub fn fresh_scores(config: &LabConfig, genotype: &AgentGenotype, conditions: usize) -> Vec<f32> {
    condition_means(&fresh_walks(config, genotype, conditions))
}

/// Each condition's mean fitness over its clones, in order of condition.
pub fn condition_means(walks: &[FreshWalk]) -> Vec<f32> {
    let conditions = walks.iter().map(|w| w.condition + 1).max().unwrap_or(0);
    (0..conditions)
        .map(|k| {
            let (sum, clones) = walks
                .iter()
                .filter(|w| w.condition == k)
                .fold((0.0, 0), |(sum, clones), w| (sum + w.fitness, clones + 1));
            sum / clones.max(1) as f32
        })
        .collect()
}

/// One clone's walk under one fresh condition: its score and everything the
/// score was made from, so that a walk can be judged by other measures than
/// the fitness (how far, how fast for its size, at what cost).
#[derive(Debug, Clone)]
pub struct FreshWalk {
    /// Which of the fresh conditions, from 0.
    pub condition: usize,
    /// The walk's fitness, a failed walk counting as 0.
    pub fitness: f32,
    /// The physics blew the robot apart.
    pub exploded: bool,
    /// The robot's driven joints.
    pub motors: usize,
    /// Its feet and its size at rest, if its morphology builds a body.
    pub body: Option<BodyTraits>,
    /// What the walk measured.
    pub trajectory: AgentTrajectory,
}

/// Every clone's walk under `conditions` fresh conditions, in order of
/// condition ([`FRESH_CLONES`] walks each): what [`fresh_scores`] averages.
pub fn fresh_walks(
    config: &LabConfig,
    genotype: &AgentGenotype,
    conditions: usize,
) -> Vec<FreshWalk> {
    (0..conditions as u64)
        .flat_map(|k| {
            let mut lab = HeadlessLab::new(LabConfig {
                population_size: FRESH_CLONES,
                seed: Some(FRESH_SEED_BASE + k),
                ..config.clone()
            });
            // One generation, kept for scoring instead of bred from.
            lab.app.world_mut().resource_mut::<LabConfig>().auto_restart = false;
            for _ in 0..FRESH_CLONES {
                lab.send_into_evolution(genotype.clone());
            }
            lab.app.world_mut().write_message(SpawnRequest);
            let steps = (f64::from(config.generation_duration) * FIXED_HZ).ceil() as usize;
            for _ in 0..steps / STEPS_PER_UPDATE as usize + STALL_SLACK_UPDATES {
                lab.app.update();
                if lab.app.world().resource::<GenerationClock>().is_finished() {
                    break;
                }
            }
            let agents = &lab.population().agents;
            agents
                .iter()
                .map(|agent| FreshWalk {
                    condition: k as usize,
                    fitness: phenotype_of(agent).fitness.max(0.0),
                    exploded: agent.exploded,
                    motors: agent.num_motors,
                    body: agent.body_traits,
                    trajectory: trajectory_of(agent),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
