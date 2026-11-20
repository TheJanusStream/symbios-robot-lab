//! The arena's physics: a static floor. Lights, camera and the floor's look
//! belong to the windowed app.

use crate::simulation::collision::arena_layers;
use avian3d::prelude::*;
use bevy::prelude::*;

/// Marker for the arena floor entity.
#[derive(Component)]
pub struct ArenaFloor;

/// Side length of the square floor (m).
pub const FLOOR_SIZE: f32 = 500.0;

/// Thickness of the floor's collider (m).
const FLOOR_THICKNESS: f32 = 0.1;

/// Spawns the floor collider with its top face at `y = 0`, where robots are
/// placed: a collider centred on `y = 0` would start every foot inside it.
pub fn setup_arena(mut commands: Commands) {
    commands.spawn((
        ArenaFloor,
        Name::new("Arena floor"),
        Transform::from_xyz(0.0, -FLOOR_THICKNESS / 2.0, 0.0),
        RigidBody::Static,
        Collider::cuboid(FLOOR_SIZE, FLOOR_THICKNESS, FLOOR_SIZE),
        arena_layers(),
    ));
}
