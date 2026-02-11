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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::genome::AgentGenotype;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios_neat::NeatConfig;

    fn make_agent(rng: &mut Pcg64) -> Agent {
        let config = NeatConfig {
            num_inputs: 4,
            num_outputs: 2,
            ..NeatConfig::default()
        };
        Agent {
            genotype: AgentGenotype::new("omega: B(1)".to_string(), &config, rng),
            fitness: 0.0,
            root_entity: None,
        }
    }

    #[test]
    fn new_population_starts_at_generation_zero() {
        let mut rng = Pcg64::seed_from_u64(42);
        let agents = vec![make_agent(&mut rng), make_agent(&mut rng)];
        let pop = Population::new(agents);

        assert_eq!(pop.generation, 0);
        assert_eq!(pop.agents.len(), 2);
    }

    #[test]
    fn default_population_is_empty() {
        let pop = Population::default();
        assert!(pop.agents.is_empty());
        assert_eq!(pop.generation, 0);
    }
}
