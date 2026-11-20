//! Physics constants and the windowed app's time scaling.

use crate::core::config::LabConfig;
use bevy::prelude::*;
use std::time::Duration;

/// Maximum number of FixedUpdate steps allowed per render frame.
/// Prevents the "spiral of death" where FixedUpdate catch-up causes
/// longer frames, which cause more catch-up, causing freezes.
pub const MAX_FIXED_STEPS_PER_FRAME: f64 = 10.0;

/// Bevy's fixed timestep frequency (Hz), which the physics and the
/// generation clock run at.
pub const FIXED_HZ: f64 = 64.0;

/// Solver substeps per fixed step.
pub const SUBSTEPS: u32 = 12;

/// Gravitational acceleration (m/s²).
pub const GRAVITY: f32 = 9.81;

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

        // Since Bevy 0.20, max_delta caps the SCALED delta accepted per frame:
        //   virtual_delta = min(real_delta * speed, max_delta)
        //   fixed_steps   = virtual_delta * FIXED_HZ
        // so capping fixed_steps at N is max_delta = N / FIXED_HZ, whatever
        // the speed. Bevy 0.18 clamped the raw delta before scaling, and the
        // cap was divided by the speed; under 0.20 that division held a 5x
        // run at about 1x.
        time.set_max_delta(Duration::from_secs_f64(
            MAX_FIXED_STEPS_PER_FRAME / FIXED_HZ,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;

    /// Runs one frame of `raw` real time through `sync_time_scale` at the
    /// lab's default config and returns the virtual time that frame produced.
    fn virtual_delta_of_one_frame(raw: Duration) -> Duration {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<LabConfig>()
            .insert_resource(TimeUpdateStrategy::ManualDuration(raw))
            .add_systems(Update, sync_time_scale);
        // The first update starts the clock (no delta) and applies the config;
        // the second advances real time by `raw` at the configured speed.
        app.update();
        app.update();
        app.world().resource::<Time<Virtual>>().delta()
    }

    #[test]
    fn a_normal_frame_runs_at_the_configured_time_scale() {
        let raw = Duration::from_millis(16);
        let speed = f64::from(LabConfig::default().time_scale);
        assert!(
            speed > 1.0,
            "the default time scale should speed the lab up"
        );
        assert_eq!(virtual_delta_of_one_frame(raw), raw.mul_f64(speed));
    }

    #[test]
    fn a_slow_frame_runs_at_most_the_capped_number_of_fixed_steps() {
        let virtual_delta = virtual_delta_of_one_frame(Duration::from_millis(500));
        let fixed_steps = virtual_delta.as_secs_f64() * FIXED_HZ;
        assert_eq!(fixed_steps, MAX_FIXED_STEPS_PER_FRAME);
    }
}
