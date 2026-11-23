//! The end of a generation: score every agent, merge them into the parent
//! pool, breed the next generation and replace the robots.

use crate::core::config::{BrainEncoding, LabConfig, Senses};
use crate::core::descriptor::{ACTION_INTENSITY, DUTY_FACTOR, FEET, LENGTH, NUM_TRAITS};
use crate::core::genome::AgentGenotype;
use crate::core::morphology_pool::RuntimeMorphologyPool;
use crate::core::population::{Agent, Population, RobotIds};
use crate::simulation::clock::GenerationClock;
use crate::simulation::fitness::{NUM_OBJECTIVES, SCORING, evaluate_trajectory, trajectory_of};
use crate::simulation::hall_of_fame::{Champion, HallOfFame, Immigrants};
use crate::simulation::lifecycle::{DerivedBody, derive_blueprint, fresh_agent_from_pool};
use crate::simulation::lineage::Lineage;
use crate::simulation::parent_pool::ParentPool;
use crate::simulation::rng::LabRng;
use crate::simulation::telemetry::{GenerationSummary, Telemetry};
use bevy::prelude::*;
use bevy::tasks::{ComputeTaskPool, TaskPool};
use rand::Rng;
use symbios_genetics::Phenotype;

/// A robot needs more driven joints than this to be scored: bodies with
/// fewer were reaching the Pareto front by sliding with no motors at all.
pub const MIN_MOTORS: usize = 4;
/// Share of each new generation drawn fresh from the morphology pools, to
/// keep the gene pool diverse.
const FRESH_SHARE: usize = 10;

/// The phenotype an agent's evaluation produced, its descriptor holding all
/// four traits (see [`crate::core::descriptor`]). Exploded, under-motored and
/// non-progressing agents fail: they get `f32::MIN` fitness and objectives,
/// so any viable agent dominates them, and the quality-diversity archives
/// leave them out.
pub fn phenotype_of(agent: &Agent) -> Phenotype<AgentGenotype> {
    let viable =
        !agent.exploded && agent.num_motors > MIN_MOTORS && agent.displacement > f32::EPSILON;
    let mut descriptor = vec![0.0; NUM_TRAITS];
    let (fitness, objectives) = if viable {
        let (fitness, objectives, gait) = evaluate_trajectory(&trajectory_of(agent));
        descriptor[DUTY_FACTOR] = gait[0];
        descriptor[ACTION_INTENSITY] = gait[1];
        if let Some(body) = agent.body_traits {
            [descriptor[FEET], descriptor[LENGTH]] = body.scaled();
        }
        (fitness, objectives)
    } else {
        (f32::MIN, vec![f32::MIN; NUM_OBJECTIVES])
    };
    Phenotype {
        genotype: agent.genotype.clone(),
        fitness,
        objectives,
        descriptor,
    }
}

/// When the generation clock reaches its last step: evaluate, merge into the
/// parent pool, record the pool's best in the lineage, breed, despawn the
/// old robots and install the new agents (spawned on the next fixed step). Waits instead while
/// `LabConfig::auto_restart` is off.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn end_generation(
    mut commands: Commands,
    config: Res<LabConfig>,
    mut population: ResMut<Population>,
    mut clock: ResMut<GenerationClock>,
    mut telemetry: ResMut<Telemetry>,
    mut parent_pool: ResMut<ParentPool>,
    runtime_pool: Res<RuntimeMorphologyPool>,
    mut rng: ResMut<LabRng>,
    mut hall: ResMut<HallOfFame>,
    mut immigrants: ResMut<Immigrants>,
    mut ids: ResMut<RobotIds>,
    mut lineage: ResMut<Lineage>,
) {
    if !clock.is_finished() || !config.auto_restart {
        return;
    }
    let generation = population.generation;
    let evaluated: Vec<Phenotype<AgentGenotype>> =
        population.agents.iter().map(phenotype_of).collect();
    let summary = summarize(generation, &population.agents, &evaluated);

    // Parents that walked again add the walk to their average (and update
    // the hall of fame with it); the rest are newcomers to the pool.
    let mut newcomers = Vec::with_capacity(evaluated.len());
    for (agent, walk) in population.agents.iter().zip(evaluated) {
        if !agent.retest {
            newcomers.push(walk);
        } else if let Some((fitness, walks)) = parent_pool.retest(&walk) {
            hall.offer(Champion {
                genotype: walk.genotype,
                fitness: Some(fitness),
                generation: Some(generation),
                walks,
                scoring: SCORING,
            });
        }
    }
    hall.offer_best(generation, &newcomers);
    parent_pool.merge(&config, newcomers, rng.rng());
    if let Some((best, walks)) = parent_pool.best() {
        lineage.record(
            generation,
            Champion {
                genotype: best.genotype.clone(),
                fitness: Some(best.fitness),
                generation: Some(generation),
                walks,
                scoring: SCORING,
            },
        );
    }
    let summary = GenerationSummary {
        pool_best_fitness: parent_pool.best_fitness(),
        pool_size: parent_pool.len(),
        distinct_bodies: parent_pool.distinct_bodies(),
        oldest_body: parent_pool.oldest_body(),
        ..summary
    };
    info!(
        "Generation {generation} complete ({}): best {:.4}, mean {:.4}, pool best {:.4} of {}, {} exploded",
        config.algorithm,
        summary.best_fitness,
        summary.mean_fitness,
        summary.pool_best_fitness,
        summary.pool_size,
        summary.exploded,
    );
    telemetry.record(summary);

    let arrivals = std::mem::take(&mut immigrants.0);
    let next = breed_generation(
        &parent_pool,
        &config,
        &runtime_pool,
        arrivals,
        &mut ids,
        rng.rng(),
    );

    for agent in &population.agents {
        if let Some(root) = agent.root_entity {
            commands.entity(root).try_despawn();
        }
    }
    *population = Population {
        agents: next,
        generation: generation + 1,
    };
    clock.stop();
}

/// The generation's statistics before the merge.
fn summarize(
    generation: usize,
    agents: &[Agent],
    evaluated: &[Phenotype<AgentGenotype>],
) -> GenerationSummary {
    let scores: Vec<f32> = evaluated
        .iter()
        .map(|p| {
            if p.fitness.is_finite() {
                p.fitness.max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let best_fitness = scores.iter().copied().fold(0.0, f32::max);
    let mean_fitness = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f32>() / scores.len() as f32
    };
    let max_displacement = agents
        .iter()
        .filter(|a| !a.exploded && a.displacement.is_finite())
        .map(|a| a.displacement)
        .fold(0.0, f32::max);
    GenerationSummary {
        generation,
        best_fitness,
        mean_fitness,
        pool_best_fitness: 0.0,
        max_displacement,
        exploded: agents.iter().filter(|a| a.exploded).count(),
        evaluated: agents.len(),
        pool_size: 0,
        distinct_bodies: 0,
        oldest_body: 0,
    }
}

/// The next generation: `immigrants` (genotypes sent into evolution, as new
/// bodies to this run), offspring bred from the parent pool in their place,
/// the best parents walking again (see [`LabConfig::retests`]), and a share
/// of fresh bodies. A pool too small to breed from (all agents failed, say)
/// is topped up with fresh bodies rather than shrinking the population.
/// Every robot but the parents walking again gets a new identity.
fn breed_generation(
    parent_pool: &ParentPool,
    config: &LabConfig,
    runtime_pool: &RuntimeMorphologyPool,
    immigrants: Vec<AgentGenotype>,
    ids: &mut RobotIds,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    let retests = parent_pool.retest_candidates(config.retests_per_generation());
    let num_fresh = config.population_size / FRESH_SHARE;
    let num_bred = config.population_size - num_fresh - retests.len();

    let mut offspring = arrivals(immigrants, config.population_size - retests.len());
    let num_offspring = num_bred.saturating_sub(offspring.len());
    offspring.extend((0..num_offspring).map_while(|_| parent_pool.breed(config, rng)));
    let num_children = offspring.len();
    offspring.extend(retests);
    let mut agents = derive_offspring(offspring, config.senses(), rng);
    for agent in &mut agents[num_children..] {
        agent.retest = true;
    }
    while agents.len() < config.population_size {
        match fresh_agent_from_pool(runtime_pool, config.brain, config.senses(), rng) {
            Some(agent) => agents.push(agent),
            None => break,
        }
    }
    for agent in agents.iter_mut().filter(|a| !a.retest) {
        agent.genotype.id = ids.issue();
    }
    agents
}

/// Up to `room` immigrants, their bodies new to this gene pool (age 0).
pub(crate) fn arrivals(mut immigrants: Vec<AgentGenotype>, room: usize) -> Vec<AgentGenotype> {
    immigrants.truncate(room);
    for genotype in &mut immigrants {
        genotype.body_age = 0;
    }
    immigrants
}

/// Derives the offspring's bodies in parallel and resizes each direct brain
/// to its body's sensors and joints, and `senses`' inputs, where crossover,
/// mutation or a change of senses changed them. A HyperNEAT CPPN needs no
/// resizing: it paints whatever body it gets. A child that senses a bearing
/// (the target task) has a steering reflex.
pub(crate) fn derive_offspring(
    offspring: Vec<AgentGenotype>,
    senses: Senses,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    if offspring.is_empty() {
        return Vec::new();
    }
    // An app's task pool plugin has made the pool; a bare unit test has not.
    let pool = ComputeTaskPool::get_or_init(TaskPool::default);
    let derived: Vec<Option<DerivedBody>> = pool.scope(|scope| {
        for child in &offspring {
            scope.spawn(async move {
                derive_blueprint(
                    &child.morphology,
                    child.lsystem_seed,
                    child.lsystem_iterations,
                )
            });
        }
    });
    offspring
        .into_iter()
        .zip(derived)
        .map(|(mut child, derived)| match derived {
            Some(body) => {
                // A child of robots bred to walk forward gets a steering
                // reflex when the task turns to targets.
                if senses.bearing && child.steering_gain == 0.0 {
                    child.steering_gain = AgentGenotype::random_steering_gain(rng);
                }
                let inputs = body.inputs.with_senses(senses).total();
                if child.encoding == BrainEncoding::Direct
                    && (inputs != child.controller.input_ids.len()
                        || body.num_outputs != child.controller.output_ids.len())
                {
                    child.controller.resize_io(inputs, body.num_outputs, rng);
                }
                Agent::fresh(child, body.num_outputs, Some(body.blueprint))
            }
            None => Agent::fresh(child, 0, None),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::{CRAWLER, HEXAPOD};
    use crate::simulation::lifecycle::fresh_agent;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn walker(rng: &mut Pcg64) -> Agent {
        let mut agent = fresh_agent(HEXAPOD, BrainEncoding::Direct, Senses::NONE, rng)
            .expect("hexapod derives");
        agent.displacement = 1.0;
        agent.avg_velocity = 0.5;
        agent.com_bounce = 0.05;
        agent.foot_transitions_per_sensor = 3.0;
        agent.mass = 10.0;
        agent
    }

    #[test]
    fn an_exploded_agent_is_dominated_whatever_it_measured() {
        let mut rng = Pcg64::seed_from_u64(1);
        let intact = walker(&mut rng);
        let mut exploded = walker(&mut rng);
        exploded.exploded = true;
        assert!(phenotype_of(&intact).fitness > 0.0);
        assert_eq!(phenotype_of(&exploded).fitness, f32::MIN);
    }

    /// A walker's descriptor holds its gait and its body: the hexapod stands
    /// on six feet and spans about 2.4 m.
    #[test]
    fn a_phenotype_describes_its_gait_and_its_body() {
        let mut rng = Pcg64::seed_from_u64(4);
        let mut hexapod = walker(&mut rng);
        hexapod.support_duty_factor = 0.6;
        let descriptor = phenotype_of(&hexapod).descriptor;
        assert_eq!(descriptor.len(), NUM_TRAITS);
        assert_eq!(descriptor[DUTY_FACTOR], 0.6);
        assert_eq!(descriptor[FEET], 6.0 / crate::core::descriptor::FEET_SCALE);
        let length = descriptor[LENGTH] * crate::core::descriptor::LENGTH_SCALE;
        assert!((2.0..3.0).contains(&length), "hexapod length {length} m");
        hexapod.exploded = true;
        assert_eq!(phenotype_of(&hexapod).descriptor, vec![0.0; NUM_TRAITS]);
    }

    /// The best parents walk again in children's places, keeping their
    /// identities; every other robot gets a new one.
    #[test]
    fn the_best_parents_walk_again_among_new_robots() {
        let mut rng = Pcg64::seed_from_u64(7);
        let config = LabConfig {
            population_size: 12,
            retests: 3,
            ..LabConfig::default()
        };
        let mut ids = RobotIds::default();
        let parents: Vec<Phenotype<AgentGenotype>> = (0..6)
            .map(|i| {
                let mut agent = walker(&mut rng);
                agent.genotype.id = ids.issue();
                agent.displacement = 0.5 + 0.1 * i as f32;
                phenotype_of(&agent)
            })
            .collect();
        let mut pool = ParentPool::default();
        pool.merge(&config, parents, &mut rng);
        let best: Vec<u64> = pool.retest_candidates(3).iter().map(|g| g.id).collect();

        let next = breed_generation(
            &pool,
            &config,
            &RuntimeMorphologyPool::default(),
            Vec::new(),
            &mut ids,
            &mut rng,
        );
        assert_eq!(next.len(), 12);
        let retests: Vec<u64> = next
            .iter()
            .filter(|a| a.retest)
            .map(|a| a.genotype.id)
            .collect();
        assert_eq!(retests, best);
        let mut all: Vec<u64> = next.iter().map(|a| a.genotype.id).collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 12, "identities repeat");
        assert!(next.iter().filter(|a| !a.retest).all(|a| a.genotype.id > 6));
    }

    #[test]
    fn a_pool_too_small_to_breed_is_topped_up_with_fresh_bodies() {
        let mut rng = Pcg64::seed_from_u64(2);
        let config = LabConfig {
            population_size: 12,
            ..LabConfig::default()
        };
        let next = breed_generation(
            &ParentPool::default(),
            &config,
            &RuntimeMorphologyPool::default(),
            Vec::new(),
            &mut RobotIds::default(),
            &mut rng,
        );
        assert_eq!(next.len(), 12);
        assert!(next.iter().all(|a| a.cached_blueprint.is_some()));
    }

    #[test]
    fn offspring_keep_a_cppn_whatever_body_they_grow_and_resize_a_direct_brain() {
        let mut rng = Pcg64::seed_from_u64(3);
        let hexapod = |encoding, rng: &mut Pcg64| {
            fresh_agent(HEXAPOD, encoding, Senses::NONE, rng)
                .expect("hexapod derives")
                .genotype
        };
        let crawler = |encoding, rng: &mut Pcg64| {
            fresh_agent(CRAWLER, encoding, Senses::NONE, rng)
                .expect("crawler derives")
                .genotype
        };
        // Each child keeps its genome but grows the other body.
        let mut cppn_child = hexapod(BrainEncoding::HyperNeat, &mut rng);
        cppn_child.morphology = crawler(BrainEncoding::HyperNeat, &mut rng).morphology;
        let mut direct_child = hexapod(BrainEncoding::Direct, &mut rng);
        let crawler_body = crawler(BrainEncoding::Direct, &mut rng);
        direct_child.morphology = crawler_body.morphology.clone();
        let crawler_inputs = crawler_body.controller.input_ids.len();

        let agents = derive_offspring(vec![cppn_child, direct_child], Senses::NONE, &mut rng);
        assert_eq!(agents[0].genotype.controller.input_ids.len(), 7);
        assert_eq!(
            agents[1].genotype.controller.input_ids.len(),
            crawler_inputs
        );
    }

    /// Switching to the target task rewires bred direct brains for the
    /// bearing: two more inputs on the same body.
    #[test]
    fn offspring_bred_for_the_target_task_read_the_bearing() {
        let mut rng = Pcg64::seed_from_u64(6);
        let parent = fresh_agent(HEXAPOD, BrainEncoding::Direct, Senses::NONE, &mut rng)
            .expect("hexapod derives")
            .genotype;
        let forward_inputs = parent.controller.input_ids.len();
        let agents = derive_offspring(vec![parent], Senses::BEARING, &mut rng);
        assert_eq!(
            agents[0].genotype.controller.input_ids.len(),
            forward_inputs + crate::simulation::sensors::NUM_BEARING_INPUTS
        );
    }
}
