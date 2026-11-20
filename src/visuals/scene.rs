//! The arena's look: light, camera and the floor's mesh.

use bevy::{post_process::bloom::Bloom, prelude::*, render::view::Tonemapping};
use bevy_panorbit_camera::PanOrbitCamera;
use std::f32::consts::TAU;
use symbios_robot_lab::simulation::arena::{ArenaFloor, FLOOR_SIZE};

pub fn setup_scene(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(20.0, 50.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        PanOrbitCamera {
            focus: Vec3::new(-25.0, 0.0, -15.0),
            yaw: Some(-TAU / 5.0),
            pitch: Some(TAU / 8.0),
            radius: Some(80.0),
            button_orbit: MouseButton::Right,
            button_pan: MouseButton::Middle,
            ..default()
        },
        Camera3d::default(),
        // Bevy's default tonemapper (TonyMcMapface) needs lookup tables the
        // lab leaves out of its bundle; this one computes its curve.
        Tonemapping::SomewhatBoringDisplayTransform,
        Bloom::NATURAL,
    ));
}

/// Draws a plane where the simulation's floor collider has its top face
/// (`y = 0`), once the floor exists.
pub fn dress_arena_floor(
    mut commands: Commands,
    floors: Query<(), Added<ArenaFloor>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for () in &floors {
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(FLOOR_SIZE, FLOOR_SIZE))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.1, 0.1, 0.15),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::IDENTITY,
            Name::new("Arena floor plane"),
        ));
    }
}
