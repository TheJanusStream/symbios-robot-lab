use crate::core::genome::AgentGenotype;
use bevy::prelude::*;

#[derive(Clone)]
pub struct Agent {
    pub genotype: AgentGenotype,
    pub fitness: f32,
    /// The root entity of the spawned robot in the Bevy world
    pub root_entity: Option<Entity>,
}

#[derive(Resource, Default)]
pub struct Population {
    pub agents: Vec<Agent>,
    pub generation: usize,
}

impl Population {
    pub fn new(agents: Vec<Agent>) -> Self {
        Self {
            agents,
            generation: 0,
        }
    }
}
