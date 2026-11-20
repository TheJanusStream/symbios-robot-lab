//! Draws robots: a mesh and material for every part the simulation spawns,
//! and a marker on every end effector.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use symbios_robot::{MaterialId, ShapePrimitive};
use symbios_robot_lab::core::population::Population;
use symbios_robot_lab::simulation::body::{EndEffectors, RobotPart};

/// Part colours by blueprint material id (the `'` symbol sets it; most
/// bodies use 0). Ids past the end wrap around.
const PART_COLORS: [Color; 6] = [
    Color::srgb(0.25, 0.72, 0.32),
    Color::srgb(0.95, 0.55, 0.15),
    Color::srgb(0.25, 0.5, 0.9),
    Color::srgb(0.92, 0.82, 0.25),
    Color::srgb(0.65, 0.35, 0.85),
    Color::srgb(0.2, 0.75, 0.75),
];

/// Radius of an end-effector marker (m).
const MARKER_RADIUS: f32 = 0.08;

/// Shape parameters are rounded to this (m) when looking up a shared mesh.
const MESH_KEY_QUANTUM: f32 = 0.001;

/// Materials robot parts and markers are drawn with.
#[derive(Resource)]
pub struct RobotMaterials {
    parts: Vec<Handle<StandardMaterial>>,
    marker: Handle<StandardMaterial>,
}

impl RobotMaterials {
    fn part(&self, id: MaterialId) -> Handle<StandardMaterial> {
        self.parts[usize::from(id) % self.parts.len()].clone()
    }
}

pub fn setup_robot_materials(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let parts = PART_COLORS
        .iter()
        .map(|&base_color| {
            materials.add(StandardMaterial {
                base_color,
                perceptual_roughness: 0.4,
                metallic: 0.1,
                ..default()
            })
        })
        .collect();
    let marker = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.2, 1.0),
        emissive: LinearRgba::new(2.0, 0.5, 2.0, 1.0),
        unlit: true,
        ..default()
    });
    commands.insert_resource(RobotMaterials { parts, marker });
}

/// A shape rounded to [`MESH_KEY_QUANTUM`], identifying a shared mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ShapeKey {
    Box([i32; 3]),
    Cylinder(i32, i32),
    Sphere(i32),
    Capsule(i32, i32),
}

fn quantize(v: f32) -> i32 {
    (v / MESH_KEY_QUANTUM).round() as i32
}

impl ShapeKey {
    fn of(shape: &ShapePrimitive) -> Self {
        match *shape {
            ShapePrimitive::Box(h) => Self::Box([quantize(h.x), quantize(h.y), quantize(h.z)]),
            ShapePrimitive::Cylinder { radius, height } => {
                Self::Cylinder(quantize(radius), quantize(height))
            }
            ShapePrimitive::Sphere(radius) => Self::Sphere(quantize(radius)),
            ShapePrimitive::Capsule { radius, height } => {
                Self::Capsule(quantize(radius), quantize(height))
            }
        }
    }
}

fn mesh_of(shape: &ShapePrimitive) -> Mesh {
    match *shape {
        ShapePrimitive::Box(half) => Cuboid::from_size(half * 2.0).into(),
        ShapePrimitive::Cylinder { radius, height } => Cylinder::new(radius, height).into(),
        ShapePrimitive::Sphere(radius) => Sphere::new(radius).into(),
        ShapePrimitive::Capsule { radius, height } => Capsule3d::new(radius, height).into(),
    }
}

/// Meshes shared by every part of the same shape, so a generation's robots
/// upload a few dozen meshes instead of one per part. The cache is emptied
/// when a new generation starts, which bounds it to one generation's shapes
/// (evolution keeps inventing new ones).
#[derive(Resource, Default)]
pub struct PartMeshCache {
    meshes: HashMap<ShapeKey, Handle<Mesh>>,
    marker: Option<Handle<Mesh>>,
    generation: Option<usize>,
}

impl PartMeshCache {
    fn mesh(&mut self, shape: &ShapePrimitive, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
        self.meshes
            .entry(ShapeKey::of(shape))
            .or_insert_with(|| meshes.add(mesh_of(shape)))
            .clone()
    }

    fn marker(&mut self, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
        self.marker
            .get_or_insert_with(|| meshes.add(Sphere::new(MARKER_RADIUS)))
            .clone()
    }

    fn start_generation(&mut self, generation: usize) {
        if self.generation != Some(generation) {
            self.meshes.clear();
            self.generation = Some(generation);
        }
    }
}

/// Gives every newly spawned part its mesh and material.
pub fn attach_part_meshes(
    mut commands: Commands,
    parts: Query<(Entity, &RobotPart), Added<RobotPart>>,
    population: Res<Population>,
    mut cache: ResMut<PartMeshCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<RobotMaterials>,
) {
    if parts.is_empty() {
        return;
    }
    cache.start_generation(population.generation);
    for (entity, part) in &parts {
        commands.entity(entity).insert((
            Mesh3d(cache.mesh(&part.shape, &mut meshes)),
            MeshMaterial3d(materials.part(part.material)),
        ));
    }
}

/// Puts a small glowing sphere on every end effector of a newly spawned
/// robot, riding the part it is pinned to.
pub fn attach_end_effector_markers(
    mut commands: Commands,
    roots: Query<(&EndEffectors, &Children), Added<EndEffectors>>,
    parts: Query<&RobotPart>,
    mut cache: ResMut<PartMeshCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<RobotMaterials>,
) {
    for (effectors, children) in &roots {
        for effector in &effectors.0 {
            let Some(part) = children.iter().find(|&child| {
                parts
                    .get(child)
                    .is_ok_and(|p| p.module_id == effector.module_id)
            }) else {
                continue;
            };
            commands.spawn((
                Mesh3d(cache.marker(&mut meshes)),
                MeshMaterial3d(materials.marker.clone()),
                Transform::from_translation(effector.local_position)
                    .with_rotation(effector.local_rotation),
                ChildOf(part),
                Name::new(format!("End effector {}", effector.id)),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_within_a_millimetre_share_a_mesh() {
        let a = ShapePrimitive::Capsule {
            radius: 0.2,
            height: 1.0,
        };
        let b = ShapePrimitive::Capsule {
            radius: 0.2002,
            height: 1.0001,
        };
        let c = ShapePrimitive::Capsule {
            radius: 0.25,
            height: 1.0,
        };
        assert_eq!(ShapeKey::of(&a), ShapeKey::of(&b));
        assert_ne!(ShapeKey::of(&a), ShapeKey::of(&c));
    }
}
