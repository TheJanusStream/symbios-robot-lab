//! Single-unit pattern generators (Morse, Risi, Snyder and Stanley 2013): a
//! brain without a network. Each driven joint plays a pattern over time,
//! painted by an evolved CPPN from where the joint sits, and the pattern of
//! a joint on a leg restarts whenever that leg's foot touches down. The
//! touch makes the rhythm listen to the ground: a leg that lands early or
//! late starts its next step from its landing, so legs fall into step with
//! the ground and with each other. Of five encodings Tarapore and Mouret
//! compared (2015), this one evolved best and was always the first to
//! recover a gait after damage.
//!
//! The CPPN reads the joint's place on the body (in the HyperNEAT body
//! frame) and the phase of its pattern (as its sine and cosine, so the
//! pattern joins up end to end), and gives the joint's target position (-1
//! to 1) and, at phase 0, its phase offset, which spreads the joints'
//! patterns apart: a knee can lag its hip, legs placed apart can step
//! apart. A pattern lasts one period of the robot's evolved rhythm. Joints
//! off any leg, and legs without a touch sensor, play on without restarts.

use crate::simulation::body::{legs, modules_in_order};
use crate::simulation::hyperneat::Substrate;
use crate::simulation::sensors::InputLayout;
use bevy::prelude::*;
use rand::Rng;
use symbios_neat::{CppnEvaluator, EvalScratchpad, NeatConfig, NeatGenome};
use symbios_robot::{JointType, RobotBlueprint, SensorType};

/// CPPN inputs: the joint's place (x, y, z), then the sine and cosine of its
/// pattern's phase.
pub const SUPG_INPUTS: usize = 5;
/// CPPN outputs: the joint's target, then its phase offset.
pub const SUPG_OUTPUTS: usize = 2;
const TARGET: usize = 0;
const OFFSET: usize = 1;
/// A touch input at or above this reads as the foot on the ground.
const CONTACT: f32 = 0.5;

/// The NEAT configuration of a pattern-generator CPPN.
pub fn supg_config() -> NeatConfig {
    NeatConfig::cppn(SUPG_INPUTS, SUPG_OUTPUTS)
}

/// A fresh pattern-generator genome: a fully connected CPPN with random
/// weights.
pub fn fresh_supg_cppn(rng: &mut impl Rng) -> NeatGenome {
    NeatGenome::fully_connected(supg_config(), rng)
}

/// One driven joint's pattern generator.
#[derive(Debug, Clone)]
struct PatternJoint {
    /// Where the joint sits, in the body frame.
    position: Vec3,
    /// Its phase offset, 0 to 1.
    offset: f32,
    /// The leg whose touchdowns restart its pattern, if any.
    leg: Option<usize>,
}

/// A leg's clock: the time since its foot last touched down.
#[derive(Debug, Clone, Default)]
struct LegClock {
    /// The leg's touch inputs, where the senses hold them.
    touches: Vec<usize>,
    since_touchdown: f32,
    down: bool,
}

/// A robot's pattern generators, one per driven joint, in output order.
#[derive(Debug, Clone)]
pub struct SupgBrain {
    cppn: CppnEvaluator,
    scratch: EvalScratchpad,
    joints: Vec<PatternJoint>,
    legs: Vec<LegClock>,
    /// Time the brain has run, for joints that play freely.
    running: f32,
    /// The clock's reading at the last evaluation.
    last: Option<f32>,
}

impl SupgBrain {
    /// The pattern generators `cppn` paints on `blueprint`'s driven joints.
    pub fn new(cppn: CppnEvaluator, blueprint: &RobotBlueprint) -> Self {
        let positions = Substrate::of_blueprint(blueprint).joints;
        let leg_of = legs(blueprint);
        let mut clocks = vec![LegClock::default(); leg_of.values().max().map_or(0, |&n| n + 1)];
        // Touch inputs come in module order, each module's mounts in turn
        // (see `InputLayout`).
        let layout = InputLayout::of_blueprint(blueprint);
        let mut touch = 0;
        for (id, module) in modules_in_order(blueprint) {
            for mount in &module.sensors {
                if mount.sensor_type == SensorType::Touch {
                    if let Some(&leg) = leg_of.get(&id) {
                        clocks[leg].touches.push(layout.touch_offset(touch));
                    }
                    touch += 1;
                }
            }
        }
        // Driven joints in output order: hinges, then sliders.
        let hinges = blueprint
            .joints
            .iter()
            .filter(|j| matches!(j.joint_type, JointType::Hinge { .. }));
        let sliders = blueprint
            .joints
            .iter()
            .filter(|j| matches!(j.joint_type, JointType::Prismatic { .. }));
        let mut scratch = cppn.create_scratchpad();
        let mut out = [0.0; SUPG_OUTPUTS];
        let joints = hinges
            .chain(sliders)
            .zip(positions)
            .map(|(joint, position)| {
                let leg = leg_of
                    .get(&joint.child_id)
                    .copied()
                    .filter(|&leg| !clocks[leg].touches.is_empty());
                let inputs = [position.x, position.y, position.z, 0.0, 1.0];
                cppn.evaluate_into(&inputs, &mut out, &mut scratch);
                PatternJoint {
                    position,
                    offset: (out[OFFSET] + 1.0) / 2.0,
                    leg,
                }
            })
            .collect();
        Self {
            cppn,
            scratch,
            joints,
            legs: clocks,
            running: 0.0,
            last: None,
        }
    }

    /// Driven joints with a pattern.
    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }

    /// Legs whose touchdowns restart their joints' patterns.
    pub fn num_listening_legs(&self) -> usize {
        self.legs
            .iter()
            .filter(|leg| !leg.touches.is_empty())
            .count()
    }

    /// The joints' targets now: each leg's clock restarts when its foot
    /// touches down (read from `senses`), and each joint plays its pattern
    /// at its leg's time since touchdown (or the brain's running time), one
    /// pattern per period of `rhythm_hz`, shifted by its offset.
    pub fn evaluate(&mut self, senses: &[f32], elapsed: f32, rhythm_hz: f32, outputs: &mut [f32]) {
        let dt = self.last.map_or(0.0, |last| (elapsed - last).max(0.0));
        self.last = Some(elapsed);
        self.running += dt;
        for leg in &mut self.legs {
            let down = leg
                .touches
                .iter()
                .any(|&i| senses.get(i).is_some_and(|&v| v >= CONTACT));
            if down && !leg.down {
                leg.since_touchdown = 0.0;
            } else {
                leg.since_touchdown += dt;
            }
            leg.down = down;
        }
        let mut out = [0.0; SUPG_OUTPUTS];
        for (joint, output) in self.joints.iter().zip(outputs.iter_mut()) {
            let time = joint
                .leg
                .map_or(self.running, |leg| self.legs[leg].since_touchdown);
            let phase = (time * rhythm_hz + joint.offset).fract() * std::f32::consts::TAU;
            let p = joint.position;
            let inputs = [p.x, p.y, p.z, phase.sin(), phase.cos()];
            self.cppn
                .evaluate_into(&inputs, &mut out, &mut self.scratch);
            *output = out[TARGET];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::HEXAPOD;
    use crate::simulation::lifecycle::derive_blueprint;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    fn hexapod_brain(seed: u64) -> (SupgBrain, InputLayout) {
        let mut rng = Pcg64::seed_from_u64(seed);
        let body = derive_blueprint(HEXAPOD, 1, 3).expect("hexapod derives");
        let cppn = CppnEvaluator::new(&fresh_supg_cppn(&mut rng)).expect("acyclic");
        (SupgBrain::new(cppn, &body.blueprint), body.inputs)
    }

    /// The hexapod's 18 joints each get a pattern, and its six feet each
    /// restart a leg's.
    #[test]
    fn every_joint_plays_a_pattern_and_every_foot_listens() {
        let (brain, _) = hexapod_brain(1);
        assert_eq!(brain.num_joints(), 18);
        assert_eq!(brain.num_listening_legs(), 6);
        assert!(brain.joints.iter().all(|j| j.leg.is_some()));
        assert!(brain.joints.iter().all(|j| (0.0..=1.0).contains(&j.offset)));
    }

    /// A pattern repeats each period while no foot lands, and a touchdown
    /// restarts its leg's joints from their offsets.
    #[test]
    fn a_touchdown_restarts_the_legs_pattern() {
        let (mut brain, layout) = hexapod_brain(2);
        let mut senses = vec![0.0; layout.total()];
        let play = |brain: &mut SupgBrain, senses: &[f32], t: f32| {
            let mut out = vec![0.0; 18];
            brain.evaluate(senses, t, 1.0, &mut out);
            out
        };
        // Feet in the air: one period on, the same targets.
        let start = play(&mut brain, &senses, 0.0);
        let later = play(&mut brain, &senses, 0.3);
        let period = play(&mut brain, &senses, 1.0);
        assert!(start.iter().zip(&period).all(|(a, b)| (a - b).abs() < 1e-4));
        assert!(start.iter().zip(&later).any(|(a, b)| (a - b).abs() > 1e-4));
        // Every foot lands at t = 1.3: every leg starts over, as at t = 0.
        let first_leg_touch = brain.legs[0].touches[0];
        for leg in &brain.legs {
            for &i in &leg.touches {
                senses[i] = 1.0;
            }
        }
        let landed = play(&mut brain, &senses, 1.3);
        assert!(start.iter().zip(&landed).all(|(a, b)| (a - b).abs() < 1e-4));
        // Staying down does not restart again.
        let stance = play(&mut brain, &senses, 1.6);
        assert!(
            landed
                .iter()
                .zip(&stance)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
        assert!(senses[first_leg_touch] >= CONTACT);
    }
}
