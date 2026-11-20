//! Symbios Robot Lab: an evolutionary robotics sandbox.
//!
//! Co-evolves robot bodies (L-System morphologies) and brains (NEAT networks)
//! with configurable evolutionary algorithms (NSGA-II, SimpleGA, MAP-Elites,
//! CVT-MAP-Elites, novelty search), on [Bevy](https://bevyengine.org/) with
//! Avian3D physics.
//!
//! The library holds everything that does not draw: [`core`] data types, the
//! [`simulation`] loop as a Bevy plugin, and a [`headless`] runner. The
//! binary adds the window, the panels and the meshes.

pub mod core;
pub mod headless;
pub mod simulation;

pub use crate::core::presets::MORPHOLOGY_POOL;
