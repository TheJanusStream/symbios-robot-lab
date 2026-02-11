use avian3d::prelude::*;
use bevy::prelude::*;

/// Physics layers to prevent agents from colliding with each other
/// while still colliding with the arena floor.
#[derive(PhysicsLayer, Clone, Copy, Debug, Default)]
pub enum GameLayer {
    #[default]
    Arena,
    Robot,
}

/// Assigns collision layers to all dynamic rigid bodies (robots)
/// so they collide with the arena but ghost through each other.
pub fn assign_collision_layers(
    mut commands: Commands,
    new_bodies: Query<Entity, Added<RigidBody>>,
    bodies: Query<&RigidBody>,
) {
    for entity in &new_bodies {
        if let Ok(body) = bodies.get(entity) {
            let layers = match body {
                RigidBody::Static => {
                    // Arena floor: belongs to Arena layer, collides with Robot
                    CollisionLayers::new(GameLayer::Arena, GameLayer::Robot)
                }
                _ => {
                    // Robot parts: belong to Robot layer, collide with Arena only
                    CollisionLayers::new(GameLayer::Robot, GameLayer::Arena)
                }
            };
            commands.entity(entity).insert(layers);
        }
    }
}
