//! What the windowed lab draws: the arena, the robots' parts and markers,
//! and the camera that follows a selected robot.

pub mod robot_render;
pub mod scene;
pub mod targets;
pub mod ui;

use bevy::prelude::*;

pub struct VisualsPlugin;

impl Plugin for VisualsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ui::CameraTarget>()
            .init_resource::<targets::TargetPlacement>()
            .init_resource::<robot_render::PartMeshCache>()
            .add_systems(
                Startup,
                (
                    scene::setup_scene,
                    robot_render::setup_robot_materials,
                    targets::setup_target_posts,
                ),
            )
            .add_systems(
                Update,
                (
                    scene::dress_arena_floor,
                    scene::draw_ground,
                    targets::draw_target_posts,
                    targets::place_user_target,
                    robot_render::attach_part_meshes,
                    robot_render::attach_end_effector_markers,
                    ui::click_to_follow,
                    ui::follow_target,
                ),
            );
    }
}
