use avian3d::prelude::*;
use bevy::prelude::*;

pub fn setup_physics(mut commands: Commands) {
    // Set standard earth gravity
    commands.insert_resource(Gravity(Vec3::NEG_Y * 9.81));

    // Optional: Configure physics time step for stability if needed later
    // commands.insert_resource(Time::new_with_rate(60.0));
}
