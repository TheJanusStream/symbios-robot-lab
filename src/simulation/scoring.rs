use crate::core::population::Population;
use crate::simulation::brain::RobotBrain;
use bevy::prelude::*;

/// Records each agent's spawn position so we can measure displacement.
#[derive(Component)]
pub struct SpawnPosition(pub Vec3);

/// Calculates fitness as horizontal displacement from spawn position.
/// Called each frame to keep fitness scores up to date for telemetry;
/// generation rollover reads the final values.
pub fn update_fitness_scores(
    brains: Query<(&RobotBrain, &Transform, &SpawnPosition)>,
    mut population: ResMut<Population>,
) {
    for (brain, transform, spawn_pos) in &brains {
        let displacement = transform.translation - spawn_pos.0;
        // Use horizontal distance only (XZ plane) to avoid rewarding falling
        let fitness = Vec2::new(displacement.x, displacement.z).length();

        if let Some(agent) = population.agents.get_mut(brain.agent_index) {
            agent.fitness = fitness;
        }
    }
}
