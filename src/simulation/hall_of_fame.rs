//! The hall of fame: the session's best robots, kept across runs, to watch
//! again, save as JSON, or send back into evolution, along with robots
//! imported from JSON.

use crate::core::config::{BrainEncoding, LabConfig, Task};
use crate::core::genome::{AgentGenotype, MAX_LSYSTEM_ITERATIONS, MAX_RHYTHM_HZ, MIN_RHYTHM_HZ};
use crate::core::population::Population;
use crate::simulation::brain::compile_brain;
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::rng::LabRng;
use crate::simulation::spawn::{RobotSpawn, SHOWCASE_INDEX, spawn_robot};
use crate::simulation::target::{UserTarget, target_for};
use crate::simulation::terrain::Ground;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use symbios_genetics::Phenotype;

/// Evaluated champions the hall keeps (imported robots come on top).
pub const HALL_SIZE: usize = 10;
/// The most parts an imported robot may have: the presets have at most 22,
/// and a JSON file could ask for a body the physics would choke on.
pub const MAX_IMPORTED_PARTS: usize = 256;
/// Where a showcased robot walks (world x and z): in front of the spawn
/// grid, clear of the population.
pub const SHOWCASE_SPOT: Vec2 = Vec2::new(0.0, -64.0);

/// A robot in the hall: its genotype and, unless it was imported, how it
/// scored. Saved and loaded as JSON.
#[derive(Clone, Serialize, Deserialize)]
pub struct Champion {
    pub genotype: AgentGenotype,
    /// Its walker fitness; `None` for an imported robot.
    pub fitness: Option<f32>,
    /// The generation it walked in; `None` for an imported robot.
    pub generation: Option<usize>,
}

impl Champion {
    /// The champion as pretty JSON, to save or share.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a champion serializes")
    }

    /// A robot from JSON: a saved champion, or a bare genotype. JSON can come
    /// from anywhere, so the robot must be one the lab could have bred: 1 to
    /// [`MAX_LSYSTEM_ITERATIONS`] derivation steps, a clock in the evolvable
    /// range, a brain genome that is whole, finite and compiles, and a body
    /// of at most [`MAX_IMPORTED_PARTS`] parts that drives joints.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let champion = match serde_json::from_str::<Champion>(text) {
            Ok(champion) => champion,
            Err(champion_err) => match serde_json::from_str::<AgentGenotype>(text) {
                Ok(genotype) => Champion {
                    genotype,
                    fitness: None,
                    generation: None,
                },
                Err(_) => return Err(format!("not a robot: {champion_err}")),
            },
        };
        let g = &champion.genotype;
        if !(1..=MAX_LSYSTEM_ITERATIONS).contains(&g.lsystem_iterations) {
            return Err(format!(
                "it grows its body in {} steps (the lab uses 1 to {MAX_LSYSTEM_ITERATIONS})",
                g.lsystem_iterations
            ));
        }
        if !(MIN_RHYTHM_HZ..=MAX_RHYTHM_HZ).contains(&g.rhythm_hz) {
            return Err(format!("its clock runs at {} Hz", g.rhythm_hz));
        }
        let c = &g.controller;
        let known = |id| c.nodes.contains_key(id);
        let whole = c
            .connections
            .values()
            .all(|x| known(x.input) && known(x.output))
            && c.input_ids.iter().chain(&c.output_ids).all(|&id| known(id))
            && c.bias_id.is_none_or(known);
        if !whole {
            return Err("its brain's genome links to neurons it lacks".to_string());
        }
        let finite = c.connections.values().all(|x| x.weight.is_finite())
            && c.nodes.values().all(|n| n.bias.is_finite());
        if !finite {
            return Err("its brain holds a weight that is not a number".to_string());
        }
        let Some(body) = derive_blueprint(&g.morphology, g.lsystem_seed, g.lsystem_iterations)
        else {
            return Err("its morphology builds no body".to_string());
        };
        if body.blueprint.modules.len() > MAX_IMPORTED_PARTS {
            return Err(format!(
                "its body has {} parts (at most {MAX_IMPORTED_PARTS})",
                body.blueprint.modules.len()
            ));
        }
        if body.num_outputs == 0 {
            return Err("its body drives no joint".to_string());
        }
        compile_brain(g, &body.blueprint, Task::Forward)
            .map_err(|err| format!("its brain does not compile: {err}"))?;
        Ok(champion)
    }

    /// A short name for lists and file names.
    pub fn label(&self) -> String {
        match (self.fitness, self.generation) {
            (Some(fitness), Some(generation)) => {
                format!("{fitness:.4} (generation {generation})")
            }
            _ => "imported".to_string(),
        }
    }
}

/// The hall: evaluated champions best first, then imported robots.
#[derive(Resource, Default)]
pub struct HallOfFame {
    champions: Vec<Champion>,
}

impl HallOfFame {
    pub fn champions(&self) -> &[Champion] {
        &self.champions
    }

    /// Offers a generation's best walking robot. It enters if the hall has
    /// room or it beats the weakest evaluated champion, who then leaves.
    pub fn offer(&mut self, generation: usize, evaluated: &[Phenotype<AgentGenotype>]) {
        let best = evaluated
            .iter()
            .filter(|p| p.fitness.is_finite() && p.fitness > 0.0)
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness));
        let Some(best) = best else {
            return;
        };
        let scored: Vec<f32> = self.champions.iter().filter_map(|c| c.fitness).collect();
        let weakest = scored.iter().copied().fold(f32::INFINITY, f32::min);
        if scored.len() >= HALL_SIZE && best.fitness <= weakest {
            return;
        }
        self.champions.push(Champion {
            genotype: best.genotype.clone(),
            fitness: Some(best.fitness),
            generation: Some(generation),
        });
        self.sort_and_trim();
    }

    /// Adds an imported robot.
    pub fn admit(&mut self, champion: Champion) {
        self.champions.push(champion);
        self.sort_and_trim();
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.champions.len() {
            self.champions.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.champions.clear();
    }

    /// Evaluated champions best first, imported robots after; the weakest
    /// evaluated champions leave beyond [`HALL_SIZE`].
    fn sort_and_trim(&mut self) {
        self.champions.sort_by(|a, b| match (a.fitness, b.fitness) {
            (Some(x), Some(y)) => y.total_cmp(&x),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        });
        let scored = self
            .champions
            .iter()
            .filter(|c| c.fitness.is_some())
            .count();
        if scored > HALL_SIZE {
            self.champions.drain(HALL_SIZE..scored);
        }
    }
}

/// Genotypes to send into evolution: they join the next generation bred (or
/// a restarted run's first), in place of offspring.
#[derive(Resource, Default)]
pub struct Immigrants(pub Vec<AgentGenotype>);

/// Marker for a showcased robot's root: a champion walking on its own, in
/// front of the population, unscored, until replaced or dismissed.
#[derive(Component)]
pub struct Showcase;

/// Asks for a robot to walk on its own (`Some`), or for the one walking to
/// leave (`None`).
#[derive(Message)]
pub struct ShowcaseRequest(pub Option<AgentGenotype>);

/// Answers the latest [`ShowcaseRequest`]: the previous showcase leaves and
/// the asked-for robot spawns at [`SHOWCASE_SPOT`] with a brain for the
/// current task (a direct brain resized to its body and task if it does not
/// fit them), and in the target task its own target.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn run_showcase(
    mut commands: Commands,
    mut requests: MessageReader<ShowcaseRequest>,
    config: Res<LabConfig>,
    ground: Res<Ground>,
    user_target: Res<UserTarget>,
    rng: Res<LabRng>,
    population: Res<Population>,
    showcases: Query<Entity, With<Showcase>>,
) {
    let Some(request) = requests.read().last() else {
        return;
    };
    for showcase in &showcases {
        commands.entity(showcase).despawn();
    }
    let Some(genotype) = &request.0 else {
        return;
    };
    let Some(body) = derive_blueprint(
        &genotype.morphology,
        genotype.lsystem_seed,
        genotype.lsystem_iterations,
    ) else {
        warn!("The showcased robot's morphology builds no body");
        return;
    };
    let mut genotype = genotype.clone();
    let inputs = body.inputs.for_task(config.task).total();
    if genotype.encoding == BrainEncoding::Direct
        && (genotype.controller.input_ids.len() != inputs
            || genotype.controller.output_ids.len() != body.num_outputs)
    {
        // Its own fixed seed: watching a robot leaves the run's draws alone.
        let mut resize_rng = Pcg64::seed_from_u64(0);
        genotype
            .controller
            .resize_io(inputs, body.num_outputs, &mut resize_rng);
    }
    let robot = RobotSpawn {
        genotype: &genotype,
        blueprint: &body.blueprint,
        index: SHOWCASE_INDEX,
        grid: SHOWCASE_SPOT,
        task: config.task,
        target: config.task.steers().then(|| {
            target_for(
                SHOWCASE_SPOT,
                rng.seed(),
                population.generation,
                *user_target,
            )
        }),
        damage: None,
    };
    match spawn_robot(&mut commands, robot, &ground, |_| ()) {
        Ok(robot) => {
            commands.entity(robot.root).insert(Showcase);
            info!("Showcasing a robot at {SHOWCASE_SPOT}");
        }
        Err(err) => warn!("The showcased robot could not spawn: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Task;
    use crate::core::presets::MORPHOLOGY_POOL;
    use crate::simulation::lifecycle::fresh_agent;

    fn hexapod(seed: u64) -> AgentGenotype {
        let mut rng = Pcg64::seed_from_u64(seed);
        fresh_agent(
            MORPHOLOGY_POOL[2],
            BrainEncoding::HyperNeat,
            Task::Forward,
            &mut rng,
        )
        .expect("hexapod derives")
        .genotype
    }

    fn phenotype(fitness: f32, seed: u64) -> Phenotype<AgentGenotype> {
        Phenotype {
            genotype: hexapod(seed),
            fitness,
            objectives: vec![fitness, 0.5],
            descriptor: Vec::new(),
        }
    }

    #[test]
    fn the_hall_keeps_the_best_of_each_generation_up_to_its_size() {
        let mut hall = HallOfFame::default();
        hall.offer(0, &[phenotype(0.0, 1), phenotype(f32::MIN, 2)]);
        assert!(
            hall.champions().is_empty(),
            "a robot that did not walk entered"
        );
        for generation in 0..15 {
            let batch = [
                phenotype(0.1 * generation as f32 + 0.05, 1),
                phenotype(0.1 * generation as f32 + 0.1, 2),
            ];
            hall.offer(generation, &batch);
        }
        let fitnesses: Vec<f32> = hall.champions().iter().filter_map(|c| c.fitness).collect();
        assert_eq!(fitnesses.len(), HALL_SIZE);
        assert!(fitnesses.windows(2).all(|w| w[0] >= w[1]));
        assert!((fitnesses[0] - 1.5).abs() < 1e-5);
        assert_eq!(hall.champions()[0].generation, Some(14));
        // Generations 5 to 14 hold the hall (0.6 to 1.5); a weaker one does
        // not displace anyone.
        hall.offer(15, &[phenotype(0.2, 3)]);
        assert!((hall.champions()[HALL_SIZE - 1].fitness.unwrap() - 0.6).abs() < 1e-5);
    }

    #[test]
    fn imported_robots_stay_after_the_evaluated_ones() {
        let mut hall = HallOfFame::default();
        hall.admit(Champion {
            genotype: hexapod(4),
            fitness: None,
            generation: None,
        });
        hall.offer(0, &[phenotype(0.5, 5)]);
        assert_eq!(hall.champions()[0].fitness, Some(0.5));
        assert_eq!(hall.champions()[1].label(), "imported");
        hall.remove(0);
        assert_eq!(hall.champions().len(), 1);
        hall.clear();
        assert!(hall.champions().is_empty());
    }

    #[test]
    fn a_champion_survives_json_and_a_bare_genotype_imports() {
        let champion = Champion {
            genotype: hexapod(6),
            fitness: Some(1.25),
            generation: Some(7),
        };
        let back = Champion::from_json(&champion.to_json()).expect("reads back");
        assert_eq!(back.fitness, Some(1.25));
        assert_eq!(back.generation, Some(7));
        assert!(back.genotype.same_body(&champion.genotype));
        assert_eq!(
            serde_json::to_string(&back.genotype.controller).unwrap(),
            serde_json::to_string(&champion.genotype.controller).unwrap()
        );
        let bare = serde_json::to_string(&champion.genotype).unwrap();
        let imported = Champion::from_json(&bare).expect("a bare genotype imports");
        assert_eq!(imported.fitness, None);
        assert!(Champion::from_json("{\"not\": \"a robot\"}").is_err());
        let rejected = |change: &dyn Fn(&mut AgentGenotype)| {
            let mut genotype = champion.genotype.clone();
            change(&mut genotype);
            Champion::from_json(&serde_json::to_string(&genotype).unwrap()).is_err()
        };
        assert!(rejected(
            &|g| g.morphology = "this is no L-system".to_string()
        ));
        assert!(rejected(&|g| g.lsystem_iterations = 40));
        assert!(rejected(&|g| g.lsystem_iterations = 0));
        assert!(rejected(&|g| g.rhythm_hz = 100.0));
        assert!(rejected(&|g| {
            let weight = &mut g.controller.connections.values_mut().next().unwrap().weight;
            *weight = f32::INFINITY;
        }));
    }
}
