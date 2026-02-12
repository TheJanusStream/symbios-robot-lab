use crate::core::config::LabConfig;
use avian3d::prelude::*;
use bevy::prelude::*;
use std::time::Duration;

/// Maximum number of FixedUpdate steps allowed per render frame.
/// Prevents the "spiral of death" where FixedUpdate catch-up causes
/// longer frames, which cause more catch-up, causing freezes.
const MAX_FIXED_STEPS_PER_FRAME: f64 = 10.0;

/// Default fixed timestep frequency in Bevy (Hz).
const FIXED_HZ: f64 = 64.0;

pub fn setup_physics(mut commands: Commands) {
    commands.insert_resource(Gravity(Vec3::NEG_Y * 9.81));
}

/// Syncs the UI time-scale slider to Bevy's virtual time, which drives physics.
/// Also caps `max_delta` to prevent the FixedUpdate spiral-of-death: at high
/// time scales (e.g. 10x), the accumulated virtual time per frame can trigger
/// dozens of FixedUpdate steps, each running NEAT evaluation for all agents.
/// If those steps exceed the frame budget, the next frame accumulates even MORE
/// time, creating a self-reinforcing freeze. Capping max_delta ensures the
/// simulation gracefully slows down instead of locking up.
pub fn sync_time_scale(config: Res<LabConfig>, mut time: ResMut<Time<Virtual>>) {
    if config.is_changed() {
        time.set_relative_speed(config.time_scale);

        // max_delta caps the RAW (pre-scaling) delta accepted per frame.
        // virtual_delta = min(real_delta, max_delta) * speed
        // fixed_steps = virtual_delta / fixed_timestep
        // To cap fixed_steps at N: max_delta = N / FIXED_HZ / speed
        let speed = (config.time_scale as f64).max(0.01);
        let max_delta_secs = (MAX_FIXED_STEPS_PER_FRAME / FIXED_HZ / speed).clamp(0.002, 0.25);
        time.set_max_delta(Duration::from_secs_f64(max_delta_secs));
    }
}
