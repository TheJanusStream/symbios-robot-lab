use crate::core::config::LabConfig;
use crate::core::genome::AgentGenotype;
use crate::core::population::{Agent, Population};
use crate::core::presets::MORPHOLOGY_POOL;
use crate::simulation::brain::compute_num_inputs;
use bevy::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use symbios::System;
use symbios_neat::NeatConfig;
use symbios_robot::{JointType, RobotBlueprint, RobotConfig, RobotInterpreter, SensorType};

/// Parses an L-System morphology string and builds the robot blueprint.
///
/// Returns the blueprint along with brain topology `(num_inputs, num_outputs)`,
/// or `None` if the morphology fails to parse.
pub fn derive_blueprint(
    morphology: &str,
    seed: u64,
    iterations: usize,
) -> Option<(RobotBlueprint, usize, usize)> {
    let mut sys = System::from_source(morphology).ok()?;
    sys.set_seed(seed);
    sys.derive(iterations).ok()?;

    let mut interpreter = RobotInterpreter::new(RobotConfig::default());
    interpreter.populate_standard_symbols(&sys.interner);
    let blueprint = interpreter.build_blueprint(&sys.state);

    let mut num_imus = 0usize;
    let mut num_touches = 0usize;
    for module in blueprint.modules.values() {
        for sensor in &module.sensors {
            match sensor.sensor_type {
                SensorType::IMU => num_imus += 1,
                SensorType::Touch => num_touches += 1,
                _ => {}
            }
        }
    }

    let num_outputs = blueprint
        .joints
        .iter()
        .filter(|j| j.joint_type == JointType::Hinge)
        .count();

    let num_inputs = compute_num_inputs(num_imus, num_touches, num_outputs);
    Some((blueprint, num_inputs, num_outputs))
}

#[derive(Message)]
pub struct SpawnRequest;

pub fn manage_generations(
    mut commands: Commands,
    mut population: ResMut<Population>,
    config: Res<LabConfig>,
    mut spawn_events: MessageReader<SpawnRequest>,
    mut gen_timer: ResMut<crate::simulation::generation::GenerationTimer>,
) {
    for _ in spawn_events.read() {
        info!("Initializing Generation 0...");

        // Despawn any existing robot hierarchies from a previous run
        for agent in &population.agents {
            if let Some(root) = agent.root_entity {
                commands.entity(root).try_despawn();
            }
        }

        // Reset generation timer
        gen_timer.active = false;

        let mut rng = Pcg64::from_rng(&mut rand::rng());

        let new_agents: Vec<Agent> = (0..config.population_size)
            .map(|_| {
                let morphology = MORPHOLOGY_POOL[rand::random_range(0..MORPHOLOGY_POOL.len())];

                let seed: u64 = rng.random();
                let derived = derive_blueprint(morphology, seed, 3);
                let (blueprint, num_inputs, num_outputs) = match derived {
                    Some((bp, ni, no)) => (Some(bp), ni, no),
                    None => {
                        warn!("Failed to derive brain topology, using fallback");
                        (None, 8, 4)
                    }
                };

                let neat_config = NeatConfig {
                    num_inputs,
                    num_outputs,
                    output_activation: symbios_neat::Activation::Tanh,
                    ..NeatConfig::default()
                };

                Agent {
                    genotype: AgentGenotype::new(morphology.to_string(), &neat_config, &mut rng),
                    fitness: 0.0,
                    num_motors: num_outputs,
                    root_entity: None,
                    body_entity: None,
                    cached_blueprint: blueprint,
                    energy_consumed: 0.0,
                    displacement: 0.0,
                    avg_up_alignment: 0.0,
                    avg_height: 0.0,
                }
            })
            .collect();

        *population = Population::new(new_agents);
    }
}
