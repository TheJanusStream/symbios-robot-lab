//! The end of a generation: score every agent, merge them into the parent
//! pool, breed the next generation and replace the robots.

use crate::core::config::{BrainEncoding, LabConfig};
use crate::core::genome::AgentGenotype;
use crate::core::morphology_pool::RuntimeMorphologyPool;
use crate::core::population::{Agent, Population};
use crate::simulation::clock::GenerationClock;
use crate::simulation::fitness::{evaluate_trajectory, trajectory_of};
use crate::simulation::lifecycle::{DerivedBody, derive_blueprint, fresh_agent_from_pool};
use crate::simulation::parent_pool::ParentPool;
use crate::simulation::rng::LabRng;
use crate::simulation::telemetry::{GenerationSummary, Telemetry};
use bevy::prelude::*;
use bevy::tasks::{ComputeTaskPool, TaskPool};
use rand::Rng;
use symbios_genetics::Phenotype;

/// A robot needs more driven joints than this to be scored: bodies with
/// fewer were reaching the Pareto front by sliding with no motors at all.
const MIN_MOTORS: usize = 4;
/// Share of each new generation drawn fresh from the morphology pools, to
/// keep the gene pool diverse.
const FRESH_SHARE: usize = 10;

/// The phenotype an agent's evaluation produced. Exploded, under-motored and
/// non-progressing agents get `f32::MIN` objectives, so any viable agent
/// dominates them.
pub fn phenotype_of(agent: &Agent) -> Phenotype<AgentGenotype> {
    let viable =
        !agent.exploded && agent.num_motors > MIN_MOTORS && agent.displacement > f32::EPSILON;
    let (fitness, objectives, descriptor) = if viable {
        evaluate_trajectory(&trajectory_of(agent))
    } else {
        (f32::MIN, vec![f32::MIN, f32::MIN], vec![0.0, 0.0])
    };
    Phenotype {
        genotype: agent.genotype.clone(),
        fitness,
        objectives,
        descriptor,
    }
}

/// When the generation clock reaches its last step: evaluate, merge into the
/// parent pool, breed, despawn the old robots and install the new agents
/// (spawned on the next fixed step). Waits instead while
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
) {
    if !clock.is_finished() || !config.auto_restart {
        return;
    }
    let generation = population.generation;
    let evaluated: Vec<Phenotype<AgentGenotype>> =
        population.agents.iter().map(phenotype_of).collect();
    let summary = summarize(generation, &population.agents, &evaluated);

    parent_pool.merge(
        config.algorithm,
        evaluated,
        config.population_size,
        rng.rng(),
    );
    let summary = GenerationSummary {
        pool_best_fitness: parent_pool.best_fitness(),
        pool_size: parent_pool.len(),
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
    telemetry.pareto_front_size = parent_pool.front_size();

    let next = breed_generation(&parent_pool, &config, &runtime_pool, rng.rng());

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
    }
}

/// The next generation: offspring bred from the parent pool plus a share of
/// fresh bodies. A pool too small to breed from (all agents failed, say) is
/// topped up with fresh bodies rather than shrinking the population.
fn breed_generation(
    parent_pool: &ParentPool,
    config: &LabConfig,
    runtime_pool: &RuntimeMorphologyPool,
    rng: &mut impl Rng,
) -> Vec<Agent> {
    let num_fresh = config.population_size / FRESH_SHARE;
    let num_bred = config.population_size - num_fresh;

    let offspring: Vec<AgentGenotype> = (0..num_bred)
        .map_while(|_| parent_pool.breed(config.algorithm, config.mutation_rate, rng))
        .collect();
    let mut agents = derive_offspring(offspring, rng);
    while agents.len() < config.population_size {
        match fresh_agent_from_pool(runtime_pool, config.brain, rng) {
            Some(agent) => agents.push(agent),
            None => break,
        }
    }
    agents
}

/// Derives the offspring's bodies in parallel and resizes each direct brain
/// to its body's sensors and joints where crossover or mutation changed them.
/// A HyperNEAT CPPN needs no resizing: it paints whatever body it gets.
fn derive_offspring(offspring: Vec<AgentGenotype>, rng: &mut impl Rng) -> Vec<Agent> {
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
                let inputs = body.inputs.total();
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
    use crate::core::presets::MORPHOLOGY_POOL;
    use crate::simulation::lifecycle::fresh_agent;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn walker(rng: &mut Pcg64) -> Agent {
        let mut agent =
            fresh_agent(MORPHOLOGY_POOL[2], BrainEncoding::Direct, rng).expect("hexapod derives");
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
            &mut rng,
        );
        assert_eq!(next.len(), 12);
        assert!(next.iter().all(|a| a.cached_blueprint.is_some()));
    }

    #[test]
    fn offspring_keep_a_cppn_whatever_body_they_grow_and_resize_a_direct_brain() {
        let mut rng = Pcg64::seed_from_u64(3);
        let hexapod = |encoding, rng: &mut Pcg64| {
            fresh_agent(MORPHOLOGY_POOL[2], encoding, rng)
                .expect("hexapod derives")
                .genotype
        };
        let crawler = |encoding, rng: &mut Pcg64| {
            fresh_agent(MORPHOLOGY_POOL[4], encoding, rng)
                .expect("crawler derives")
                .genotype
        };
        // Each child keeps its genome but grows the other preset's body.
        let mut cppn_child = hexapod(BrainEncoding::HyperNeat, &mut rng);
        cppn_child.morphology = crawler(BrainEncoding::HyperNeat, &mut rng).morphology;
        let mut direct_child = hexapod(BrainEncoding::Direct, &mut rng);
        let crawler_body = crawler(BrainEncoding::Direct, &mut rng);
        direct_child.morphology = crawler_body.morphology.clone();
        let crawler_inputs = crawler_body.controller.input_ids.len();

        let agents = derive_offspring(vec![cppn_child, direct_child], &mut rng);
        assert_eq!(agents[0].genotype.controller.input_ids.len(), 7);
        assert_eq!(
            agents[1].genotype.controller.input_ids.len(),
            crawler_inputs
        );
    }
}
