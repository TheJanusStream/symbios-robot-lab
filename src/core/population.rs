use crate::core::descriptor::BodyTraits;
use crate::core::genome::AgentGenotype;
use bevy::prelude::*;
use symbios_robot::RobotBlueprint;

/// A single individual in the population: genotype, cached phenotype data, and
/// per-generation fitness metrics populated by the scoring systems.
#[derive(Clone)]
pub struct Agent {
    pub genotype: AgentGenotype,
    pub fitness: f32,
    /// The robot's driven joints (hinges and sliders).
    pub num_motors: usize,
    /// The root entity of the spawned robot in the Bevy world
    pub root_entity: Option<Entity>,
    /// The body entity holding the `RobotBrain` component (first module).
    /// Used for O(1) brain lookup in systems like `check_explosion`.
    pub body_entity: Option<Entity>,
    /// Cached blueprint from L-System derivation, avoids re-parsing at spawn.
    pub cached_blueprint: Option<RobotBlueprint>,
    /// The body's feet and length, the body traits of its descriptor. Kept
    /// after spawning takes the blueprint.
    pub body_traits: Option<BodyTraits>,
    /// Set when the physics blew the robot apart this generation. An exploded
    /// agent is despawned at once and scores as dominated by every intact
    /// one, whatever its metrics read before the explosion.
    pub exploded: bool,
    /// Total mass of the robot's modules (kg), the denominator of the Cost
    /// of Transport. Set at spawn.
    pub mass: f32,
    /// Cumulative commanded-velocity energy proxy (kept for inspector display).
    pub energy_consumed: f32,
    /// Horizontal displacement from baseline (set by scoring).
    pub displacement: f32,
    /// Average root up-alignment during evaluation (0..1).
    pub avg_up_alignment: f32,
    /// Average height during evaluation (metres, unbounded above 0).
    pub avg_height: f32,

    // ── SOTA metrics (populated by `update_fitness_scores`) ────────────────
    /// Work the motors did (J), the numerator of CoT.
    pub mechanical_work: f32,
    /// Stddev of `up_alignment` over the eval window. Lower = more stable
    /// orientation.
    pub up_alignment_stddev: f32,
    /// Stddev of body Y, the "CoM bounce" amplitude. Drives
    /// `ComBounceReward`; near-zero is the vibrobot signature.
    pub com_bounce: f32,
    /// Mean forward velocity (m/s) over the post-settling window.
    /// Drives `VelocityGate`.
    pub avg_velocity: f32,
    /// How often the command of the joint that turns most often turned
    /// round and back (Hz). Drives `SteadyCommands`, the anti-vibration
    /// gate.
    pub command_turn_hz: f32,
    /// Mean per-tick `|Δv_body|² / Δt`. Lower = smoother body motion.
    pub body_jerk: f32,
    /// How far the hinges turned (rad) and the sliders slid (m), summed
    /// over the driven joints. With the body's size and the distance, the
    /// economy NSGA-II's second objective scores.
    pub hinge_travel: f32,
    pub slider_travel: f32,
    /// Mean fraction of touch sensors in contact each tick. QD axis.
    pub support_duty_factor: f32,
    /// Mean contact↔air transitions per touch sensor over the window.
    /// Drives `FootAirTimeFactor`.
    pub foot_transitions_per_sensor: f32,
    /// Mean per-joint commanded-velocity magnitude. QD axis.
    pub action_intensity: f32,
    /// Cost of Transport (dimensionless `E / (m·g·d)`), from the work the
    /// motors did. Shown in the inspector; no objective.
    pub cost_of_transport: f32,
    /// A parent walking again, whose score joins the average of its earlier
    /// walks rather than entering the parent pool as a newcomer.
    pub retest: bool,
}

impl Agent {
    /// Fresh-individual constructor: zeroed metrics, no spawned entities,
    /// optional cached blueprint (whose body traits it measures). Avoids 14
    /// boilerplate `: 0.0` lines at every call site that creates a new agent.
    pub fn fresh(
        genotype: AgentGenotype,
        num_motors: usize,
        cached_blueprint: Option<RobotBlueprint>,
    ) -> Self {
        Self {
            genotype,
            fitness: 0.0,
            num_motors,
            root_entity: None,
            body_entity: None,
            body_traits: cached_blueprint.as_ref().map(BodyTraits::of),
            cached_blueprint,
            exploded: false,
            mass: 0.0,
            energy_consumed: 0.0,
            displacement: 0.0,
            avg_up_alignment: 0.0,
            avg_height: 0.0,
            mechanical_work: 0.0,
            up_alignment_stddev: 0.0,
            com_bounce: 0.0,
            avg_velocity: 0.0,
            command_turn_hz: 0.0,
            body_jerk: 0.0,
            hinge_travel: 0.0,
            slider_travel: 0.0,
            support_duty_factor: 0.0,
            foot_transitions_per_sensor: 0.0,
            action_intensity: 0.0,
            cost_of_transport: 0.0,
            retest: false,
        }
    }
}

/// Hands out robot identities ([`AgentGenotype::id`]): every robot bred,
/// drawn fresh or imported gets the next one when it joins a generation.
/// Never reset, so an identity stays unique across restarts within a session
/// (the hall of fame outlives runs).
#[derive(Resource, Default)]
pub struct RobotIds {
    last: u64,
}

impl RobotIds {
    /// The next identity; never 0, which marks a robot without one.
    pub fn issue(&mut self) -> u64 {
        self.last += 1;
        self.last
    }
}

/// The current generation of agents being evaluated in the simulation.
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
    use crate::core::config::BrainEncoding;
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
        Agent::fresh(
            AgentGenotype::new(
                "omega: B(1)".to_string(),
                0,
                BrainEncoding::Direct,
                symbios_neat::NeatGenome::fully_connected(config, rng),
            ),
            0,
            None,
        )
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
