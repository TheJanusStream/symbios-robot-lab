//! From morphology source to agents: deriving a body's blueprint, creating
//! fresh agents from the preset pool, and starting a new run.

use crate::core::config::{BrainEncoding, LabConfig, Senses};
use crate::core::genome::{AgentGenotype, INITIAL_LSYSTEM_ITERATIONS};
use crate::core::morphology_pool::RuntimeMorphologyPool;
use crate::core::population::{Agent, Population, RobotIds};
use crate::core::presets::MORPHOLOGY_POOL;
use crate::simulation::clock::GenerationClock;
use crate::simulation::generation::{arrivals, derive_offspring};
use crate::simulation::hall_of_fame::Immigrants;
use crate::simulation::hyperneat::fresh_cppn;
use crate::simulation::lineage::Lineage;
use crate::simulation::parent_pool::ParentPool;
use crate::simulation::rng::LabRng;
use crate::simulation::sensors::InputLayout;
use crate::simulation::telemetry::Telemetry;
use avian3d::prelude::MassPropertiesBundle;
use bevy::prelude::*;
use rand::Rng;
use symbios::System;
use symbios_neat::{Activation, NeatConfig, NeatGenome};
use symbios_robot::{JointType, RobotBlueprint, RobotConfig, RobotInterpreter, ShapePrimitive};

/// Smallest radius or half-extent a part may have (m). Mutation can shrink a
/// shape parameter to zero, and a zero-size part has no mass or inertia,
/// which avian warns can produce NaN values (#81).
const MIN_PART_EXTENT: f32 = 0.04;

/// Attempts at drawing a morphology that derives, before giving up on a slot.
const FRESH_ATTEMPTS: usize = 8;

/// A blueprint and the brain shape it needs.
#[derive(Debug, Clone)]
pub struct DerivedBody {
    pub blueprint: RobotBlueprint,
    pub inputs: InputLayout,
    /// One per driven joint: hinges, then prismatic joints.
    pub num_outputs: usize,
}

/// Derives `morphology` with `seed` for `iterations` steps and builds its
/// blueprint, with every part at least `MIN_PART_EXTENT` in size. `None`
/// if the source does not parse or produces no parts.
pub fn derive_blueprint(morphology: &str, seed: u64, iterations: usize) -> Option<DerivedBody> {
    let mut sys = System::from_source(morphology).ok()?;
    sys.set_seed(seed);
    sys.derive(iterations).ok()?;

    let mut interpreter = RobotInterpreter::new(RobotConfig::default());
    interpreter.populate_standard_symbols(&sys.interner);
    let mut blueprint = interpreter.build_blueprint(&sys.state);
    if blueprint.modules.is_empty() {
        return None;
    }
    sanitize_parts(&mut blueprint);

    let inputs = InputLayout::of_blueprint(&blueprint);
    let num_outputs = blueprint
        .joints
        .iter()
        .filter(|j| {
            matches!(
                j.joint_type,
                JointType::Hinge { .. } | JointType::Prismatic { .. }
            )
        })
        .count();
    Some(DerivedBody {
        blueprint,
        inputs,
        num_outputs,
    })
}

/// Grows every part to at least [`MIN_PART_EXTENT`] and recomputes its mass.
fn sanitize_parts(blueprint: &mut RobotBlueprint) {
    for module in blueprint.modules.values_mut() {
        module.density = module.density.max(f32::EPSILON);
        module.shape = match module.shape {
            ShapePrimitive::Box(half) => {
                ShapePrimitive::Box(half.max(Vec3::splat(MIN_PART_EXTENT)))
            }
            ShapePrimitive::Cylinder { radius, height } => ShapePrimitive::Cylinder {
                radius: radius.max(MIN_PART_EXTENT),
                height: height.max(2.0 * MIN_PART_EXTENT),
            },
            ShapePrimitive::Sphere(radius) => ShapePrimitive::Sphere(radius.max(MIN_PART_EXTENT)),
            ShapePrimitive::Capsule { radius, height } => ShapePrimitive::Capsule {
                radius: radius.max(MIN_PART_EXTENT),
                height: height.max(0.0),
            },
        };
        module.mass =
            MassPropertiesBundle::from_shape(&module.shape.to_bevy_primitive(), module.density)
                .mass
                .0;
    }
}

/// The NEAT configuration for a brain with these inputs and outputs.
pub fn brain_config(num_inputs: usize, num_outputs: usize) -> NeatConfig {
    NeatConfig {
        num_inputs,
        num_outputs,
        output_activation: Activation::Tanh,
        ..NeatConfig::default()
    }
}

/// A morphology drawn uniformly from the presets and the runtime pool.
pub fn pick_morphology(runtime_pool: &RuntimeMorphologyPool, rng: &mut impl Rng) -> String {
    let runtime = runtime_pool.sources();
    let pick = rng.random_range(0..MORPHOLOGY_POOL.len() + runtime.len());
    match MORPHOLOGY_POOL.get(pick) {
        Some(source) => (*source).to_string(),
        None => runtime[pick - MORPHOLOGY_POOL.len()].1.clone(),
    }
}

/// A fresh agent: a body derived from `morphology` with a new seed, and a
/// fully connected controller: a brain sized for the body and `senses` (the
/// target task's bearing, the tilt), or for HyperNEAT or pattern generators
/// a CPPN, whose shape no body or sense changes. A robot that senses a
/// bearing is born with a steering reflex. The genotype keeps the seed the body was
/// derived with, so the body it is scored on is the body it breeds.
pub fn fresh_agent(
    morphology: &str,
    encoding: BrainEncoding,
    senses: Senses,
    rng: &mut impl Rng,
) -> Option<Agent> {
    let seed: u64 = rng.random();
    let derived = derive_blueprint(morphology, seed, INITIAL_LSYSTEM_ITERATIONS)?;
    if derived.num_outputs == 0 {
        return None;
    }
    let controller = match encoding {
        BrainEncoding::Direct => NeatGenome::fully_connected(
            brain_config(
                derived.inputs.with_senses(senses).total(),
                derived.num_outputs,
            ),
            rng,
        ),
        BrainEncoding::HyperNeat => fresh_cppn(rng),
        BrainEncoding::Supg => crate::simulation::supg::fresh_supg_cppn(rng),
    };
    let mut genotype = AgentGenotype::new(morphology.to_string(), seed, encoding, controller);
    genotype.rhythm_hz = AgentGenotype::random_rhythm(rng);
    if senses.bearing {
        genotype.steering_gain = AgentGenotype::random_steering_gain(rng);
    }
    Some(Agent::fresh(
        genotype,
        derived.num_outputs,
        Some(derived.blueprint),
    ))
}

/// A fresh agent from the presets and the runtime pool, retrying a few
/// draws if a morphology fails to derive.
pub fn fresh_agent_from_pool(
    runtime_pool: &RuntimeMorphologyPool,
    encoding: BrainEncoding,
    senses: Senses,
    rng: &mut impl Rng,
) -> Option<Agent> {
    (0..FRESH_ATTEMPTS).find_map(|_| {
        let morphology = pick_morphology(runtime_pool, rng);
        fresh_agent(&morphology, encoding, senses, rng)
    })
}

/// Starts a new run: the next fixed step spawns a fresh Generation 0.
#[derive(Message)]
pub struct SpawnRequest;

/// Handles [`SpawnRequest`]: despawns the robots of any previous run, clears
/// the parent pool, the statistics and the lineage, and creates Generation
/// 0: any robots
/// sent into evolution, then fresh bodies from the preset and runtime pools.
#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn start_new_run(
    mut commands: Commands,
    mut requests: MessageReader<SpawnRequest>,
    mut population: ResMut<Population>,
    mut parent_pool: ResMut<ParentPool>,
    mut telemetry: ResMut<Telemetry>,
    mut clock: ResMut<GenerationClock>,
    mut rng: ResMut<LabRng>,
    config: Res<LabConfig>,
    runtime_pool: Res<RuntimeMorphologyPool>,
    mut immigrants: ResMut<Immigrants>,
    mut ids: ResMut<RobotIds>,
    mut lineage: ResMut<Lineage>,
) {
    if requests.read().count() == 0 {
        return;
    }
    info!("Initializing Generation 0...");
    for agent in &population.agents {
        if let Some(root) = agent.root_entity {
            commands.entity(root).try_despawn();
        }
    }
    parent_pool.clear();
    telemetry.reset();
    lineage.clear();
    clock.stop();

    let arrivals = arrivals(std::mem::take(&mut immigrants.0), config.population_size);
    let mut agents = derive_offspring(arrivals, config.senses(), rng.rng());
    let num_fresh = config.population_size - agents.len();
    agents.extend((0..num_fresh).filter_map(|_| {
        fresh_agent_from_pool(&runtime_pool, config.brain, config.senses(), rng.rng())
    }));
    for agent in &mut agents {
        agent.genotype.id = ids.issue();
    }
    if agents.len() < config.population_size {
        warn!(
            "Only {} of {} Generation 0 bodies derived",
            agents.len(),
            config.population_size
        );
    }
    *population = Population::new(agents);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::{self, HEXAPOD, QUADRUPED};
    use crate::simulation::hyperneat::{CPPN_INPUTS, CPPN_OUTPUTS};
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    /// A fresh direct brain reads the target's bearing in the target task:
    /// two more inputs, the same outputs.
    #[test]
    fn a_fresh_direct_brain_is_sized_for_its_task() {
        let mut rng = Pcg64::seed_from_u64(12);
        let size = |task, rng: &mut Pcg64| {
            let agent =
                fresh_agent(HEXAPOD, BrainEncoding::Direct, task, rng).expect("hexapod derives");
            let c = &agent.genotype.controller;
            (c.input_ids.len(), c.output_ids.len())
        };
        let (forward_in, forward_out) = size(Senses::NONE, &mut rng);
        let (target_in, target_out) = size(Senses::BEARING, &mut rng);
        assert_eq!(
            target_in,
            forward_in + crate::simulation::sensors::NUM_BEARING_INPUTS
        );
        assert_eq!(target_out, forward_out);
    }

    #[test]
    fn every_body_derives_driven() {
        for (i, source) in MORPHOLOGY_POOL.iter().chain(fixtures::ALL).enumerate() {
            let derived = derive_blueprint(source, 5, INITIAL_LSYSTEM_ITERATIONS)
                .unwrap_or_else(|| panic!("body {i} did not derive"));
            assert!(derived.num_outputs > 0, "body {i} drives no joint");
            assert!(derived.inputs.total() > derived.inputs.hinges);
        }
    }

    #[test]
    fn a_fresh_agent_is_scored_on_the_body_it_breeds() {
        let mut rng = Pcg64::seed_from_u64(11);
        // The quadruped rolls 1, 2 or 3 leg segments per leg, so a different
        // seed would usually build a different body.
        let agent = fresh_agent(QUADRUPED, BrainEncoding::Direct, Senses::NONE, &mut rng)
            .expect("quadruped derives");
        let rederived = derive_blueprint(
            &agent.genotype.morphology,
            agent.genotype.lsystem_seed,
            agent.genotype.lsystem_iterations,
        )
        .expect("rederives");
        let cached = agent.cached_blueprint.as_ref().expect("blueprint cached");
        assert_eq!(rederived.blueprint.modules.len(), cached.modules.len());
        assert_eq!(rederived.blueprint.joints.len(), cached.joints.len());
        assert_eq!(
            agent.genotype.controller.input_ids.len(),
            rederived.inputs.total()
        );
    }

    #[test]
    fn a_fresh_hyperneat_agent_gets_a_cppn_whatever_its_body() {
        let mut rng = Pcg64::seed_from_u64(12);
        for source in MORPHOLOGY_POOL.iter().chain(fixtures::ALL) {
            let agent = fresh_agent(source, BrainEncoding::HyperNeat, Senses::NONE, &mut rng)
                .expect("the body derives");
            assert_eq!(agent.genotype.encoding, BrainEncoding::HyperNeat);
            assert_eq!(agent.genotype.controller.input_ids.len(), CPPN_INPUTS);
            assert_eq!(agent.genotype.controller.output_ids.len(), CPPN_OUTPUTS);
        }
    }

    #[test]
    fn degenerate_parts_are_grown_to_a_massive_size() {
        // A zero-radius capsule and a zero-length box.
        let derived =
            derive_blueprint("omega: K(0.0, 0.0) J B(0.0, 0.0, 0.0)", 0, 1).expect("derives");
        for module in derived.blueprint.modules.values() {
            assert!(module.mass > 0.0, "part has no mass: {:?}", module.shape);
        }
    }
}
