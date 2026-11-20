use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_panorbit_camera::PanOrbitCamera;
use symbios_robot_lab::simulation::brain::RobotBrain;

/// Tracks which agent the camera is following, if any.
#[derive(Resource, Default)]
pub struct CameraTarget {
    pub agent_index: Option<usize>,
}

/// Handles click-to-follow: raycasts from mouse position to find the clicked robot.
pub fn click_to_follow(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    brains: Query<(&RobotBrain, &Transform)>,
    mut target: ResMut<CameraTarget>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else {
        return;
    };

    // Find the closest robot root to the ray
    let mut best: Option<(usize, f32)> = None;
    for (brain, transform) in &brains {
        let to_agent = transform.translation - ray.origin;
        let t = to_agent.dot(*ray.direction);
        if t < 0.0 {
            continue;
        }
        let closest_point = ray.origin + *ray.direction * t;
        let dist_sq = (closest_point - transform.translation).length_squared();
        // Click radius threshold (generous for easy selection)
        if dist_sq < 9.0 && (best.is_none() || t < best.unwrap().1) {
            best = Some((brain.agent_index, t));
        }
    }

    if let Some((index, _)) = best {
        target.agent_index = Some(index);
        info!("Camera now tracking Agent {}", index);
    } else {
        target.agent_index = None;
    }
}

/// Smoothly moves the camera focus to track the selected agent.
pub fn follow_target(
    target: Res<CameraTarget>,
    brains: Query<(&RobotBrain, &Transform)>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    let Some(target_index) = target.agent_index else {
        return;
    };

    let Some((_, transform)) = brains.iter().find(|(b, _)| b.agent_index == target_index) else {
        return;
    };

    for mut cam in &mut cameras {
        cam.target_focus = transform.translation;
    }
}
