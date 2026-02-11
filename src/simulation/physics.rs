use crate::core::config::LabConfig;
use avian3d::prelude::*;
use bevy::prelude::*;

pub fn setup_physics(mut commands: Commands) {
    // Set standard earth gravity
    commands.insert_resource(Gravity(Vec3::NEG_Y * 9.81));
}

/// Syncs the UI time-scale slider to Bevy's virtual time, which drives physics.
pub fn sync_time_scale(config: Res<LabConfig>, mut time: ResMut<Time<Virtual>>) {
    if config.is_changed() {
        time.set_relative_speed(config.time_scale);
    }
}
