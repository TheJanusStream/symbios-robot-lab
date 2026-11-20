//! The target task in the window: a glowing post at every target, and the
//! user's target placed with a click on the ground.

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;
use symbios_robot_lab::simulation::collision::GameLayer;
use symbios_robot_lab::simulation::target::{UserTarget, WalkTarget};
use symbios_robot_lab::simulation::terrain::Ground;

/// Height (m) of a target's post.
const POST_HEIGHT: f32 = 2.0;
/// Targets closer than this (m) share a post.
const SAME_POST: f32 = 0.1;
/// How far (m) a click's ray looks for the ground.
const CLICK_RANGE: f32 = 2000.0;

/// Whether a click on the ground places the user's target (and so does not
/// select a robot).
#[derive(Resource, Default)]
pub struct TargetPlacement {
    pub active: bool,
}

/// Marker for a target's post.
#[derive(Component)]
pub struct TargetPost;

/// The posts' shared mesh and glowing material.
#[derive(Resource)]
pub struct TargetPostAssets {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

pub fn setup_target_posts(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(TargetPostAssets {
        mesh: meshes.add(Cylinder::new(0.08, POST_HEIGHT)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.75, 0.2),
            emissive: LinearRgba::rgb(4.0, 2.4, 0.4),
            ..default()
        }),
    });
}

/// Puts a post on the ground at every target whenever the targets change
/// (robots spawn with them, the user moves them, robots despawn): one post
/// for the user's shared target, one per robot otherwise.
pub fn draw_target_posts(
    mut commands: Commands,
    assets: Res<TargetPostAssets>,
    ground: Res<Ground>,
    targets: Query<&WalkTarget>,
    changed: Query<(), Changed<WalkTarget>>,
    mut removed: RemovedComponents<WalkTarget>,
    posts: Query<Entity, With<TargetPost>>,
) {
    let removed_any = removed.read().count() > 0;
    if changed.is_empty() && !removed_any && !ground.is_changed() {
        return;
    }
    for post in &posts {
        commands.entity(post).despawn();
    }
    let mut points: Vec<Vec2> = Vec::new();
    for target in &targets {
        if !points.iter().any(|p| p.distance(target.point) < SAME_POST) {
            points.push(target.point);
        }
    }
    for point in points {
        let base = ground.height_at(point.x, point.y);
        commands.spawn((
            TargetPost,
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(assets.material.clone()),
            Transform::from_xyz(point.x, base + POST_HEIGHT / 2.0, point.y),
            Name::new("Target post"),
        ));
    }
}

/// While placing, a left click on the ground (not on the panel) sets the
/// target every robot walks to.
pub fn place_user_target(
    placement: Res<TargetPlacement>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    spatial_query: SpatialQuery,
    mut contexts: EguiContexts,
    mut user_target: ResMut<UserTarget>,
) {
    if !placement.active || !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if contexts
        .ctx_mut()
        .is_ok_and(|ctx| ctx.is_pointer_over_egui())
    {
        return;
    }
    let (Ok(window), Ok((camera, camera_transform))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    let ground_only = SpatialQueryFilter::from_mask(GameLayer::Arena);
    if let Some(hit) =
        spatial_query.cast_ray(ray.origin, ray.direction, CLICK_RANGE, true, &ground_only)
    {
        let point = ray.get_point(hit.distance);
        user_target.0 = Some(Vec2::new(point.x, point.z));
        info!("Target placed at ({:.1}, {:.1})", point.x, point.z);
    }
}
