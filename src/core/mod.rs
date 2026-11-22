//! Core data types: configuration, genotypes, population state, and the body plans.

pub mod config;
pub mod descriptor;
#[cfg(test)]
pub(crate) mod fixtures;
pub mod genome;
#[cfg(not(target_arch = "wasm32"))]
pub mod hotload;
pub mod morphology_pool;
pub mod population;
pub mod presets;
pub mod share;
