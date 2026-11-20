use std::f32::consts::TAU;

use avian3d::prelude::*;
use bevy::{post_process::bloom::Bloom, prelude::*};
use bevy_panorbit_camera::PanOrbitCamera;

pub fn setup_arena(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // 1. The Arena Floor
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(500.0, 500.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.15),
            perceptual_roughness: 0.8,
            ..default()
        })),
        Transform::IDENTITY,
        RigidBody::Static,
        Collider::cuboid(500.0, 0.1, 500.0),
    ));

    // 2. Lighting
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(20.0, 50.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // 3. Camera
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
        Bloom::NATURAL, // Enable Bloom
    ));
}
