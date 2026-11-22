//! The hall of fame: the best robots, kept across runs (and, in the window,
//! across sessions), to watch again, save as JSON, share as a link, or send
//! back into evolution, along with robots imported from JSON or a link; and
//! the showcase, where robots walk on their own, side by side in lanes.

use crate::core::config::{BrainEncoding, LabConfig, Senses};
use crate::core::genome::{
    AgentGenotype, MAX_LSYSTEM_ITERATIONS, MAX_RHYTHM_HZ, MAX_STEERING_GAIN, MIN_RHYTHM_HZ,
};
use crate::core::population::Population;
use crate::core::share;
use crate::simulation::brain::compile_brain;
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::rng::LabRng;
use crate::simulation::spawn::{RobotSpawn, SHOWCASE_LANES, showcase_index, spawn_robot};
use crate::simulation::target::{UserTarget, target_for};
use crate::simulation::terrain::Ground;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use symbios_genetics::Phenotype;

/// Evaluated champions the hall keeps (imported robots come on top).
pub const HALL_SIZE: usize = 10;
/// The most parts an imported robot may have: the body plans start with 7
/// or 8, and a JSON file could ask for a body the physics would choke on.
pub const MAX_IMPORTED_PARTS: usize = 256;
/// Where a showcased robot walks (world x and z): in front of the spawn
/// grid, clear of the population. Several walk in lanes centred on it.
pub const SHOWCASE_SPOT: Vec2 = Vec2::new(0.0, -64.0);
/// Distance (m) between showcase lanes, across the way robots face (world
/// +x, see [`crate::simulation::spawn::spawn_rotation`]): robots pass
/// through one another, so lanes need only keep them apart to the eye.
pub const LANE_SPACING: f32 = 6.0;
/// What comes before the share code in a share link: `...#robot=<code>`.
pub const SHARE_FRAGMENT: &str = "robot=";

/// A robot in the hall: its genotype and, unless it was imported, how it
/// scored. Saved and loaded as JSON.
#[derive(Clone, Serialize, Deserialize)]
pub struct Champion {
    pub genotype: AgentGenotype,
    /// Its walker fitness, averaged over its walks; `None` for an imported
    /// robot.
    pub fitness: Option<f32>,
    /// The generation it last walked in; `None` for an imported robot.
    pub generation: Option<usize>,
    /// How many walks its fitness averages: one score can flatter a robot,
    /// so a champion that walked again is a surer one.
    #[serde(default = "one_walk")]
    pub walks: u32,
}

fn one_walk() -> u32 {
    1
}

impl Champion {
    /// The champion as pretty JSON, to save or share.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a champion serializes")
    }

    /// A robot from JSON: a saved champion, or a bare genotype. JSON can come
    /// from anywhere, so the robot must be one the lab could have bred: 1 to
    /// [`MAX_LSYSTEM_ITERATIONS`] derivation steps, a clock and a steering
    /// gain in their evolvable ranges, a brain genome that is whole, finite
    /// and compiles, and a body
    /// of at most [`MAX_IMPORTED_PARTS`] parts that drives joints. It loses
    /// its identity: identities belong to the session that issued them, and
    /// one robot's could be another's here (it gets a new one if it joins
    /// evolution).
    pub fn from_json(text: &str) -> Result<Self, String> {
        let mut champion = match serde_json::from_str::<Champion>(text) {
            Ok(champion) => champion,
            Err(champion_err) => match serde_json::from_str::<AgentGenotype>(text) {
                Ok(genotype) => Champion {
                    genotype,
                    fitness: None,
                    generation: None,
                    walks: 1,
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
        if !(-MAX_STEERING_GAIN..=MAX_STEERING_GAIN).contains(&g.steering_gain) {
            return Err(format!(
                "its steering gain is {} (the lab's are -{MAX_STEERING_GAIN} to {MAX_STEERING_GAIN})",
                g.steering_gain
            ));
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
        compile_brain(g, &body.blueprint, Senses::NONE)
            .map_err(|err| format!("its brain does not compile: {err}"))?;
        champion.genotype.id = 0;
        Ok(champion)
    }

    /// The robot as a share code ([`share::encode`]): its genotype as
    /// compact JSON, without its identity or score, so that it arrives as
    /// an imported robot (its score belongs to the run it walked in).
    pub fn share_code(&self) -> String {
        let mut genotype = self.genotype.clone();
        genotype.id = 0;
        let json = serde_json::to_string(&genotype).expect("a genotype serializes");
        share::encode(json.as_bytes())
    }

    /// A link that opens the lab at `page` with the robot in it.
    pub fn share_link(&self, page: &str) -> String {
        let page = page.split('#').next().unwrap_or(page);
        format!("{page}#{SHARE_FRAGMENT}{}", self.share_code())
    }

    /// A robot from a share link, or the share code alone, checked as
    /// [`Self::from_json`] checks JSON.
    pub fn from_share(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let code = text
            .rsplit_once(SHARE_FRAGMENT)
            .map_or(text, |(_, code)| code);
        let code = code.split('&').next().unwrap_or(code);
        let bytes = share::decode(code)?;
        let json = String::from_utf8(bytes).map_err(|_| "the code holds no text".to_string())?;
        Self::from_json(&json)
    }

    /// Robots from what a user pasted or dropped: JSON as
    /// [`Self::all_from_json`] reads it, or a share link or code.
    pub fn import(text: &str) -> Result<Vec<Self>, String> {
        let text = text.trim();
        if text.starts_with('{') || text.starts_with('[') {
            Self::all_from_json(text)
        } else {
            Self::from_share(text).map(|champion| vec![champion])
        }
    }

    /// Robots from JSON: a whole hall (an array of champions, as
    /// [`HallOfFame::to_json`] writes it) or one robot as
    /// [`Self::from_json`] reads it. Each must pass that function's checks.
    pub fn all_from_json(text: &str) -> Result<Vec<Self>, String> {
        match serde_json::from_str::<Vec<serde_json::Value>>(text) {
            Ok(values) => values
                .into_iter()
                .enumerate()
                .map(|(i, value)| {
                    Self::from_json(&value.to_string()).map_err(|err| format!("robot {i}: {err}"))
                })
                .collect(),
            Err(_) => Self::from_json(text).map(|champion| vec![champion]),
        }
    }

    /// A short name for lists and file names.
    pub fn label(&self) -> String {
        match (self.fitness, self.generation) {
            (Some(fitness), Some(generation)) => {
                let walks = match self.walks {
                    1 => "1 walk".to_string(),
                    n => format!("{n} walks"),
                };
                format!("{fitness:.4} ({walks}, generation {generation})")
            }
            _ => "imported".to_string(),
        }
    }
}

/// The hall: evaluated champions best first, then imported robots.
#[derive(Resource, Default)]
pub struct HallOfFame {
    champions: Vec<Champion>,
    /// Counts the changes, so a keeper can tell when the hall needs saving.
    revision: u64,
}

impl HallOfFame {
    pub fn champions(&self) -> &[Champion] {
        &self.champions
    }

    /// A number that changes whenever the champions do.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The whole hall as pretty JSON: an array of champions.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.champions).expect("champions serialize")
    }

    /// [`Self::to_json`] without the whitespace: under half the length.
    pub fn to_compact_json(&self) -> String {
        serde_json::to_string(&self.champions).expect("champions serialize")
    }

    /// Offers a generation's best walking newcomer, on its one walk.
    pub fn offer_best(&mut self, generation: usize, evaluated: &[Phenotype<AgentGenotype>]) {
        let best = evaluated
            .iter()
            .filter(|p| p.fitness.is_finite() && p.fitness > 0.0)
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness));
        if let Some(best) = best {
            self.offer(Champion {
                genotype: best.genotype.clone(),
                fitness: Some(best.fitness),
                generation: Some(generation),
                walks: 1,
            });
        }
    }

    /// Offers a scored robot. If the hall holds the same robot (by identity)
    /// it takes the new average, walk count and generation, and may fall
    /// below the others; otherwise it enters if the hall has room or it
    /// beats the weakest evaluated champion, who then leaves.
    pub fn offer(&mut self, champion: Champion) {
        let Some(fitness) = champion.fitness else {
            self.admit(champion);
            return;
        };
        let id = champion.genotype.id;
        if id != 0
            && let Some(entry) = self.champions.iter_mut().find(|c| c.genotype.id == id)
        {
            entry.fitness = champion.fitness;
            entry.walks = champion.walks;
            entry.generation = champion.generation;
            self.sort_and_trim();
            return;
        }
        let scored: Vec<f32> = self.champions.iter().filter_map(|c| c.fitness).collect();
        let weakest = scored.iter().copied().fold(f32::INFINITY, f32::min);
        if scored.len() >= HALL_SIZE && fitness <= weakest {
            return;
        }
        self.champions.push(champion);
        self.sort_and_trim();
    }

    /// Adds an imported (or restored) robot, unless the hall already holds
    /// it; says whether it entered.
    pub fn admit(&mut self, champion: Champion) -> bool {
        if self
            .champions
            .iter()
            .any(|held| same_robot(&held.genotype, &champion.genotype))
        {
            return false;
        }
        self.champions.push(champion);
        self.sort_and_trim();
        true
    }

    /// Restores a hall [`Self::to_json`] wrote, robot by robot: one that
    /// fails the import checks (a genome an older lab wrote, say) is left
    /// out, and the problems are returned.
    pub fn restore(&mut self, text: &str) -> Vec<String> {
        let values = match serde_json::from_str::<Vec<serde_json::Value>>(text) {
            Ok(values) => values,
            Err(err) => return vec![format!("not a hall: {err}")],
        };
        let mut problems = Vec::new();
        for (i, value) in values.into_iter().enumerate() {
            match Champion::from_json(&value.to_string()) {
                Ok(champion) => {
                    self.admit(champion);
                }
                Err(err) => problems.push(format!("robot {i}: {err}")),
            }
        }
        problems
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.champions.len() {
            self.champions.remove(index);
            self.revision += 1;
        }
    }

    pub fn clear(&mut self) {
        if !self.champions.is_empty() {
            self.champions.clear();
            self.revision += 1;
        }
    }

    /// Evaluated champions best first, imported robots after; the weakest
    /// evaluated champions leave beyond [`HALL_SIZE`]. Every change to the
    /// champions ends here or in [`Self::remove`] and [`Self::clear`].
    fn sort_and_trim(&mut self) {
        self.revision += 1;
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

/// Whether two genotypes are the same robot: the same body and brain,
/// whatever their identities and the age of their bodies in a gene pool.
fn same_robot(a: &AgentGenotype, b: &AgentGenotype) -> bool {
    let plain = |genotype: &AgentGenotype| {
        let mut genotype = genotype.clone();
        genotype.id = 0;
        genotype.body_age = 0;
        serde_json::to_string(&genotype).ok()
    };
    plain(a) == plain(b)
}

/// Genotypes to send into evolution: they join the next generation bred (or
/// a restarted run's first), in place of offspring.
#[derive(Resource, Default)]
pub struct Immigrants(pub Vec<AgentGenotype>);

/// Marker for a showcased robot's root: a robot walking on its own, in front
/// of the population, unscored, until replaced or dismissed.
#[derive(Component)]
pub struct Showcase;

/// A showcased robot's lane, on its body (beside its brain): what to call
/// it and where it started (world x and z), for the race's standings.
#[derive(Component, Debug, Clone)]
pub struct Lane {
    pub lane: usize,
    pub name: String,
    pub start: Vec2,
}

impl Lane {
    /// How far (m, over the ground plane) the robot at `position` is from
    /// its start.
    pub fn distance(&self, position: Vec3) -> f32 {
        Vec2::new(position.x, position.z).distance(self.start)
    }
}

/// A robot to showcase, and what to call it.
#[derive(Clone)]
pub struct Entrant {
    pub name: String,
    pub genotype: AgentGenotype,
}

/// Asks for robots to walk on their own, side by side, one per lane (at
/// most [`SHOWCASE_LANES`]), or with none for the ones walking to leave.
#[derive(Message, Clone, Default)]
pub struct ShowcaseRequest(pub Vec<Entrant>);

impl ShowcaseRequest {
    /// Asks for one robot to walk on its own.
    pub fn one(name: impl Into<String>, genotype: AgentGenotype) -> Self {
        Self(vec![Entrant {
            name: name.into(),
            genotype,
        }])
    }

    /// Asks for the showcased robots to leave.
    pub fn dismiss() -> Self {
        Self::default()
    }
}

/// Where lane `lane` of `lanes` starts (world x and z): side by side across
/// [`SHOWCASE_SPOT`].
pub fn lane_start(lane: usize, lanes: usize) -> Vec2 {
    let across = lane as f32 - (lanes.max(1) - 1) as f32 / 2.0;
    SHOWCASE_SPOT + Vec2::new(0.0, across * LANE_SPACING)
}

/// Answers the latest [`ShowcaseRequest`]: the robots walking leave, and
/// the asked-for ones spawn in their lanes ([`lane_start`]) as brain
/// indices [`showcase_index`] of their lanes, each with a brain for the
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
    let lanes = request.0.len().min(SHOWCASE_LANES);
    for (lane, entrant) in request.0.iter().take(lanes).enumerate() {
        let start = lane_start(lane, lanes);
        let target = config
            .task
            .steers()
            .then(|| target_for(start, rng.seed(), population.generation, *user_target));
        let spawned = showcase(
            &mut commands,
            &config,
            &ground,
            entrant,
            lane,
            start,
            target,
        );
        match spawned {
            Ok(()) => info!("Showcasing {} at {start}", entrant.name),
            Err(err) => warn!("{} could not walk: {err}", entrant.name),
        }
    }
}

/// Spawns `entrant` to walk on its own in lane `lane`, from `start`.
fn showcase(
    commands: &mut Commands,
    config: &LabConfig,
    ground: &Ground,
    entrant: &Entrant,
    lane: usize,
    start: Vec2,
    target: Option<Vec2>,
) -> Result<(), String> {
    let genotype = &entrant.genotype;
    let body = derive_blueprint(
        &genotype.morphology,
        genotype.lsystem_seed,
        genotype.lsystem_iterations,
    )
    .ok_or("its morphology builds no body")?;
    let mut genotype = genotype.clone();
    let inputs = body.inputs.with_senses(config.senses()).total();
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
        index: showcase_index(lane),
        grid: start,
        senses: config.senses(),
        target,
        damage: None,
    };
    let lane = Lane {
        lane,
        name: entrant.name.clone(),
        start,
    };
    let spawned = spawn_robot(commands, robot, ground, |_| lane)?;
    commands.entity(spawned.root).insert(Showcase);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::HEXAPOD;
    use crate::simulation::lifecycle::fresh_agent;

    fn hexapod(seed: u64) -> AgentGenotype {
        let mut rng = Pcg64::seed_from_u64(seed);
        fresh_agent(HEXAPOD, BrainEncoding::HyperNeat, Senses::NONE, &mut rng)
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
        hall.offer_best(0, &[phenotype(0.0, 1), phenotype(f32::MIN, 2)]);
        assert!(
            hall.champions().is_empty(),
            "a robot that did not walk entered"
        );
        for generation in 0..15 {
            let batch = [
                phenotype(0.1 * generation as f32 + 0.05, 1),
                phenotype(0.1 * generation as f32 + 0.1, 2),
            ];
            hall.offer_best(generation, &batch);
        }
        let fitnesses: Vec<f32> = hall.champions().iter().filter_map(|c| c.fitness).collect();
        assert_eq!(fitnesses.len(), HALL_SIZE);
        assert!(fitnesses.windows(2).all(|w| w[0] >= w[1]));
        assert!((fitnesses[0] - 1.5).abs() < 1e-5);
        assert_eq!(hall.champions()[0].generation, Some(14));
        // Generations 5 to 14 hold the hall (0.6 to 1.5); a weaker one does
        // not displace anyone.
        hall.offer_best(15, &[phenotype(0.2, 3)]);
        assert!((hall.champions()[HALL_SIZE - 1].fitness.unwrap() - 0.6).abs() < 1e-5);
    }

    /// A champion that walks again takes its new average in place, and can
    /// lose its place at the top; robots without an identity never merge.
    #[test]
    fn a_champion_walking_again_takes_its_new_average() {
        let mut hall = HallOfFame::default();
        let mut lucky = hexapod(8);
        lucky.id = 5;
        let mut steady = hexapod(9);
        steady.id = 6;
        let entry = |genotype: &AgentGenotype, fitness, walks| Champion {
            genotype: genotype.clone(),
            fitness: Some(fitness),
            generation: Some(1),
            walks,
        };
        hall.offer(entry(&lucky, 2.0, 1));
        hall.offer(entry(&steady, 1.5, 1));
        assert_eq!(hall.champions()[0].genotype.id, 5);
        hall.offer(entry(&lucky, 1.0, 2));
        assert_eq!(hall.champions().len(), 2);
        assert_eq!(hall.champions()[0].genotype.id, 6);
        assert_eq!(hall.champions()[1].walks, 2);
        assert_eq!(
            hall.champions()[1].label(),
            "1.0000 (2 walks, generation 1)"
        );
        let mut nobody = hexapod(10);
        nobody.id = 0;
        hall.offer(entry(&nobody, 0.5, 1));
        hall.offer(entry(&nobody, 0.4, 1));
        assert_eq!(hall.champions().len(), 4);
    }

    /// A saved hall reads back whole; one robot reads as a hall of one.
    #[test]
    fn a_whole_hall_survives_json() {
        let mut hall = HallOfFame::default();
        hall.offer_best(0, &[phenotype(0.7, 11)]);
        hall.offer_best(1, &[phenotype(0.9, 12)]);
        let back = Champion::all_from_json(&hall.to_json()).expect("reads back");
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].fitness, Some(0.9));
        let one = Champion::all_from_json(&hall.champions()[1].to_json()).expect("one robot");
        assert_eq!(one.len(), 1);
        assert!(Champion::all_from_json("[1, 2]").is_err());
    }

    /// An imported robot loses its identity, so it never takes the place
    /// of the session's own robot with that number; the hall holds a robot
    /// once.
    #[test]
    fn imports_lose_their_identity_and_enter_once() {
        let mut genotype = hexapod(13);
        genotype.id = 42;
        let saved = Champion {
            genotype: genotype.clone(),
            fitness: None,
            generation: None,
            walks: 1,
        };
        let imported = Champion::from_json(&saved.to_json()).expect("imports");
        assert_eq!(imported.genotype.id, 0);
        let mut hall = HallOfFame::default();
        let before = hall.revision();
        assert!(hall.admit(imported.clone()));
        assert!(hall.revision() > before);
        let again = hall.revision();
        assert!(!hall.admit(imported), "the same robot entered twice");
        assert_eq!(hall.revision(), again);
        // A robot of the session with the same number walks: it enters on
        // its own, leaving the import alone.
        hall.offer(Champion {
            genotype,
            fitness: Some(0.8),
            generation: Some(3),
            walks: 1,
        });
        assert_eq!(hall.champions().len(), 2);
        assert_eq!(hall.champions()[1].label(), "imported");
    }

    /// A kept hall comes back robot by robot, leaving out what no longer
    /// imports.
    #[test]
    fn a_kept_hall_is_restored_without_its_broken_robots() {
        let mut hall = HallOfFame::default();
        hall.offer_best(0, &[phenotype(0.7, 14)]);
        hall.offer_best(1, &[phenotype(0.9, 15)]);
        let mut values: Vec<serde_json::Value> = serde_json::from_str(&hall.to_json()).unwrap();
        values[1]["genotype"]["lsystem_iterations"] = serde_json::json!(99);
        let text = serde_json::to_string(&values).unwrap();
        let mut restored = HallOfFame::default();
        let problems = restored.restore(&text);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].starts_with("robot 1"), "{problems:?}");
        assert_eq!(restored.champions().len(), 1);
        assert_eq!(restored.champions()[0].fitness, Some(0.9));
        assert_eq!(restored.champions()[0].genotype.id, 0);
        assert_eq!(restored.restore("not json").len(), 1);
    }

    /// A share link carries the robot, without its score, and survives being
    /// pasted with or without the page in front of the code.
    #[test]
    fn a_robot_travels_in_a_share_link() {
        let champion = Champion {
            genotype: hexapod(16),
            fitness: Some(1.5),
            generation: Some(9),
            walks: 4,
        };
        let page = "https://example.org/lab/";
        let link = champion.share_link(page);
        assert!(
            link.starts_with("https://example.org/lab/#robot="),
            "{link}"
        );
        let json_len = serde_json::to_string(&champion.genotype).unwrap().len();
        assert!(link.len() * 4 < json_len, "{} of {json_len}", link.len());
        let back = Champion::from_share(&link).expect("the link imports");
        assert_eq!(back.fitness, None);
        assert_eq!(back.label(), "imported");
        assert!(same_robot(&back.genotype, &champion.genotype));
        let code = champion.share_code();
        assert!(same_robot(
            &Champion::from_share(&code).unwrap().genotype,
            &champion.genotype
        ));
        // A link made on a page that already held a robot replaces it.
        assert_eq!(champion.share_link(&link), link);
        // Pasted text: JSON or a link.
        assert_eq!(Champion::import(&format!("  {link}\n")).unwrap().len(), 1);
        assert_eq!(Champion::import(&champion.to_json()).unwrap().len(), 1);
        assert!(Champion::import(&link[..link.len() - 40]).is_err());
        assert!(Champion::import("robot=").is_err());
    }

    #[test]
    fn imported_robots_stay_after_the_evaluated_ones() {
        let mut hall = HallOfFame::default();
        hall.admit(Champion {
            genotype: hexapod(4),
            fitness: None,
            generation: None,
            walks: 1,
        });
        hall.offer_best(0, &[phenotype(0.5, 5)]);
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
            walks: 3,
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
        assert!(rejected(&|g| g.steering_gain = 3.0));
        assert!(!rejected(&|g| g.steering_gain = -1.5));
        // JSON numbers past f32's range read as infinity.
        let mut value = serde_json::to_value(&champion.genotype).unwrap();
        value["steering_gain"] = serde_json::json!(1e39);
        assert!(Champion::from_json(&value.to_string()).is_err());
        assert!(rejected(&|g| {
            let weight = &mut g.controller.connections.values_mut().next().unwrap().weight;
            *weight = f32::INFINITY;
        }));
    }
}
