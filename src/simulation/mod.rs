//! The evolution loop, independent of any window: physics bodies, sensors,
//! brains, scoring and the generational lifecycle. [`SimulationPlugin`] runs
//! it in a windowed app or headless (see `crate::headless`).

pub mod arena;
pub mod body;
pub mod brain;
pub mod clock;
pub mod collision;
pub mod fitness;
pub mod generation;
pub mod hall_of_fame;
pub mod hyperneat;
pub mod lifecycle;
pub mod lineage;
pub mod metrics;
pub mod parent_pool;
pub mod physics;
pub mod rng;
pub mod sensors;
pub mod spawn;
pub mod supg;
pub mod target;
pub mod telemetry;
pub mod terrain;

use crate::core::config::LabConfig;
use crate::core::morphology_pool::RuntimeMorphologyPool;
use crate::core::population::Population;
use avian3d::prelude::*;
use bevy::prelude::*;

/// The fixed-step systems of the evolution loop, in the order they run.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvolutionSystems;

/// Physics, resources and the fixed-step evolution loop. Insert a
/// [`LabConfig`] before adding it to configure the run; the default is used
/// otherwise.
pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(PhysicsPlugins::default())
            .insert_resource(SubstepCount(physics::SUBSTEPS))
            .insert_resource(Gravity(Vec3::NEG_Y * physics::GRAVITY))
            .init_resource::<LabConfig>()
            .init_resource::<Population>()
            .init_resource::<crate::core::population::RobotIds>()
            .init_resource::<parent_pool::ParentPool>()
            .init_resource::<telemetry::Telemetry>()
            .init_resource::<clock::GenerationClock>()
            .init_resource::<RuntimeMorphologyPool>()
            .init_resource::<terrain::Ground>()
            .init_resource::<spawn::SpawnShift>()
            .init_resource::<target::UserTarget>()
            .init_resource::<hall_of_fame::HallOfFame>()
            .init_resource::<hall_of_fame::Immigrants>()
            .init_resource::<lineage::Lineage>()
            .add_message::<hall_of_fame::ShowcaseRequest>()
            .add_message::<lifecycle::SpawnRequest>()
            .add_systems(Startup, (arena::setup_arena, rng::init_rng))
            .add_systems(
                FixedUpdate,
                (
                    lifecycle::start_new_run,
                    terrain::lay_ground,
                    spawn::spawn_agents,
                    hall_of_fame::run_showcase,
                    clock::advance_clock,
                    target::advance_targets,
                    target::follow_user_target,
                    metrics::check_explosion,
                    sensors::gather_sensor_inputs,
                    sensors::sense_tilt,
                    brain::drive_motors,
                    metrics::apply_perturbations,
                    metrics::track_metrics,
                    metrics::update_fitness_scores,
                    generation::end_generation,
                )
                    .chain()
                    .in_set(EvolutionSystems),
            );
    }
}
