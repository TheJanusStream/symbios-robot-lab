use avian3d::prelude::*;

/// Physics layers: robots collide with the arena but ghost through each
/// other, so each robot is evaluated alone however crowded the arena is.
#[derive(PhysicsLayer, Clone, Copy, Debug, Default)]
pub enum GameLayer {
    #[default]
    Arena,
    Robot,
}

/// Layers for the arena's static geometry: collides with robots.
pub fn arena_layers() -> CollisionLayers {
    CollisionLayers::new(GameLayer::Arena, GameLayer::Robot)
}

/// Layers for a robot part: collides with the arena only.
pub fn robot_layers() -> CollisionLayers {
    CollisionLayers::new(GameLayer::Robot, GameLayer::Arena)
}
