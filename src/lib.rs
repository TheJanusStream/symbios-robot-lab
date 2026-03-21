//! Symbios Robot Lab — an evolutionary robotics sandbox.
//!
//! Evolves soft-bodied robots whose morphology (L-System) and brain (NEAT
//! neural network) are co-optimised via configurable evolutionary algorithms
//! (NSGA-II, SimpleGA, MAP-Elites). Built on [Bevy](https://bevyengine.org/)
//! with Avian3D physics.

pub mod core;
pub use crate::core::presets::MORPHOLOGY_POOL;
