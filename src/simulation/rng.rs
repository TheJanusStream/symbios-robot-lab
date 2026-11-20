//! The run's single source of randomness: every random choice the evolution
//! makes draws from [`LabRng`], so a seeded run repeats exactly.

use crate::core::config::LabConfig;
use bevy::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// Seeded random number generator for one evolution run.
#[derive(Resource)]
pub struct LabRng {
    rng: Pcg64,
    seed: u64,
}

impl LabRng {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: Pcg64::seed_from_u64(seed),
            seed,
        }
    }

    /// A generator seeded from the operating system's entropy, for
    /// interactive runs that did not ask for a seed.
    pub fn from_entropy() -> Self {
        Self::from_seed(rand::rng().random())
    }

    /// The seed this run started from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn rng(&mut self) -> &mut Pcg64 {
        &mut self.rng
    }
}

/// Inserts the run's [`LabRng`] at startup from [`LabConfig::seed`], unless
/// one is already there, and logs the seed so the run can be repeated.
pub fn init_rng(mut commands: Commands, config: Res<LabConfig>, existing: Option<Res<LabRng>>) {
    if existing.is_some() {
        return;
    }
    let rng = match config.seed {
        Some(seed) => LabRng::from_seed(seed),
        None => LabRng::from_entropy(),
    };
    info!("Evolution seed: {}", rng.seed());
    commands.insert_resource(rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_draws_the_same_numbers() {
        let mut a = LabRng::from_seed(42);
        let mut b = LabRng::from_seed(42);
        let xs: Vec<u64> = (0..8).map(|_| a.rng().random()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.rng().random()).collect();
        assert_eq!(xs, ys);
        assert_eq!(a.seed(), 42);
    }
}
