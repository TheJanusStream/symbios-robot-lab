//! The generation clock counts fixed physics steps, so every robot in every
//! generation is evaluated for the same number of steps whatever the frame
//! rate, and a headless run steps exactly as a windowed one does.

use bevy::prelude::*;

/// Fraction of a generation spent settling before fitness is measured, so the
/// robot's first fall onto the floor does not count as motion.
pub const SETTLING_FRACTION: f32 = 1.0 / 3.0;

/// Where the current generation is, in fixed steps.
#[derive(Resource, Debug, Default, Clone)]
pub struct GenerationClock {
    step: u32,
    total_steps: u32,
    settle_steps: u32,
    step_secs: f32,
    running: bool,
}

impl GenerationClock {
    /// Starts a generation lasting `duration_secs` of simulated time, at
    /// fixed steps of `step_secs`.
    pub fn start(&mut self, duration_secs: f32, step_secs: f32) {
        let total = (duration_secs / step_secs).round().max(1.0) as u32;
        self.step = 0;
        self.total_steps = total;
        self.settle_steps = (total as f32 * SETTLING_FRACTION).round() as u32;
        self.step_secs = step_secs;
        self.running = true;
    }

    /// Stops the clock at the end of a generation (or a reset).
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Advances one fixed step, stopping at the generation's last step.
    pub fn advance(&mut self) {
        if self.running && self.step < self.total_steps {
            self.step += 1;
        }
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    /// True once the settling period is over and fitness is being measured.
    pub fn is_settled(&self) -> bool {
        self.running && self.step >= self.settle_steps
    }

    /// True on the generation's last step.
    pub fn is_finished(&self) -> bool {
        self.running && self.step >= self.total_steps
    }

    /// Fixed steps since this generation's robots were spawned.
    pub fn step(&self) -> u32 {
        self.step
    }

    /// The step at which settling ends.
    pub fn settle_step(&self) -> u32 {
        self.settle_steps
    }

    /// The generation's last step.
    pub fn total_steps(&self) -> u32 {
        self.total_steps
    }

    /// Simulated seconds since this generation's robots were spawned.
    pub fn elapsed_secs(&self) -> f32 {
        self.step as f32 * self.step_secs
    }

    /// Simulated seconds of the measured window so far (0 while settling).
    pub fn window_elapsed_secs(&self) -> f32 {
        self.step.saturating_sub(self.settle_steps) as f32 * self.step_secs
    }

    /// Simulated seconds the generation lasts.
    pub fn duration_secs(&self) -> f32 {
        self.total_steps as f32 * self.step_secs
    }

    /// Simulated seconds until the generation ends.
    pub fn remaining_secs(&self) -> f32 {
        self.total_steps.saturating_sub(self.step) as f32 * self.step_secs
    }

    /// Fraction of the generation done, in `[0, 1]`.
    pub fn progress(&self) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.step as f32 / self.total_steps as f32
        }
    }
}

/// Advances the clock by one fixed step.
pub fn advance_clock(mut clock: ResMut<GenerationClock>) {
    clock.advance();
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: f32 = 1.0 / 64.0;

    #[test]
    fn a_generation_lasts_its_duration_in_fixed_steps() {
        let mut clock = GenerationClock::default();
        clock.start(15.0, STEP);
        assert_eq!(clock.total_steps(), 960);
        assert_eq!(clock.settle_step(), 320);
        assert!(!clock.is_settled());
        for _ in 0..320 {
            clock.advance();
        }
        assert!(clock.is_settled());
        assert!(!clock.is_finished());
        for _ in 320..960 {
            clock.advance();
        }
        assert!(clock.is_finished());
        assert_eq!(clock.elapsed_secs(), 15.0);
        assert_eq!(clock.window_elapsed_secs(), 10.0);
    }

    #[test]
    fn the_clock_stops_at_the_last_step() {
        let mut clock = GenerationClock::default();
        clock.start(1.0, STEP);
        for _ in 0..1000 {
            clock.advance();
        }
        assert_eq!(clock.step(), clock.total_steps());
        assert_eq!(clock.remaining_secs(), 0.0);
    }

    #[test]
    fn a_stopped_clock_does_not_advance() {
        let mut clock = GenerationClock::default();
        clock.start(1.0, STEP);
        clock.stop();
        clock.advance();
        assert_eq!(clock.step(), 0);
        assert!(!clock.is_settled());
        assert!(!clock.is_finished());
    }
}
