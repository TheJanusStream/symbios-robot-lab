//! The brain: one controller per robot (a feed-forward network evaluated
//! from its sensor inputs, or pattern generators restarted by its feet),
//! whose outputs drive the robot's joint motors.

use crate::core::config::{BrainEncoding, Senses};
use crate::core::genome::AgentGenotype;
use crate::simulation::clock::GenerationClock;
use crate::simulation::hyperneat::{CPPN_INPUTS, CPPN_OUTPUTS, Substrate, paint_brain};
use crate::simulation::supg::{SUPG_INPUTS, SUPG_OUTPUTS, SupgBrain};
use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_neat::{CppnEvaluator, EvaluatorError, FeedforwardNetwork, Scratchpad};
use symbios_robot::RobotBlueprint;

/// The speed (rad/s) the metrics read a hinge output of ±1 as, when they
/// judge the brain's commands for smoothness and intensity: the scale their
/// thresholds were set at when outputs drove motor speeds. The motors now
/// take outputs as positions (see [`drive_motors`]).
pub const MAX_MOTOR_VELOCITY: f32 = 3.0;
/// The most torque (N·m) any hinge motor may exert; a joint whose blueprint
/// limit gives a smaller effort is held to that. Also the torque scale of
/// the metrics' energy and work estimates.
pub const MAX_MOTOR_TORQUE: f32 = 5000.0;
/// The speed (m/s) the metrics read a slider output of ±1 as.
pub const MAX_PRISMATIC_VELOCITY: f32 = 1.5;
/// The most force (N) any slider motor may exert; a joint whose blueprint
/// limit gives a smaller effort is held to that.
pub const MAX_PRISMATIC_FORCE: f32 = 5000.0;
/// Evaluate brains every Nth fixed step; motors hold their last command in
/// between. Sensors still read every step, so the IMU's finite differences
/// stay one step long.
pub const BRAIN_EVAL_INTERVAL: u32 = 2;

/// Range given to a driven hinge that has no limit (rad either side of rest).
pub const DEFAULT_HINGE_RANGE: f32 = std::f32::consts::FRAC_PI_4;
/// Range given to a driven slider that has no limit (m either side of rest).
pub const DEFAULT_SLIDER_RANGE: f32 = 0.1;

/// A joint the brain drives, and the range its output maps onto.
#[derive(Debug, Clone, Copy)]
pub struct DrivenJoint {
    pub entity: Entity,
    /// Lower end of the joint's travel (rad for a hinge, m for a slider).
    pub min: f32,
    /// Upper end of the joint's travel.
    pub max: f32,
    /// The most its motor may exert (N·m for a hinge, N for a slider).
    pub max_effort: f32,
    /// The leg it is on (see [`crate::simulation::body::legs`]), if any.
    pub leg: Option<usize>,
    /// Broken: its motor lets go, and the joint swings or slides freely
    /// within its limits.
    pub broken: bool,
}

impl DrivenJoint {
    /// The joint position an output in `[-1, 1]` asks for: -1 is the bottom of
    /// the range, +1 the top.
    pub fn target(&self, output: f32) -> f32 {
        let t = (output.clamp(-1.0, 1.0) + 1.0) / 2.0;
        self.min + t * (self.max - self.min)
    }
}

/// Why a genotype yielded no brain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrainError {
    /// The controller genome (a direct brain or a CPPN) has a cycle.
    Cyclic,
    /// A CPPN controller (HyperNEAT or pattern generators) is not shaped like
    /// the lab's: it `needs` some (inputs, outputs) and `has` others.
    CppnShape {
        needs: (usize, usize),
        has: (usize, usize),
    },
}

impl std::fmt::Display for BrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainError::Cyclic => f.write_str("the brain's genome has a cycle"),
            BrainError::CppnShape { needs, has } => write!(
                f,
                "the brain's CPPN needs {} inputs and {} outputs, this one has {} and {}",
                needs.0, needs.1, has.0, has.1
            ),
        }
    }
}

impl std::error::Error for BrainError {}

impl From<EvaluatorError> for BrainError {
    fn from(_: EvaluatorError) -> Self {
        BrainError::Cyclic
    }
}

/// What turns a robot's senses into its joints' targets.
pub enum Controller {
    /// A feed-forward network reading the senses: the genome itself (direct
    /// NEAT) or painted from it onto the body (HyperNEAT).
    Network {
        network: FeedforwardNetwork,
        scratchpad: Scratchpad,
    },
    /// Pattern generators, one per driven joint, restarted by the feet (see
    /// [`crate::simulation::supg`]).
    Patterns(SupgBrain),
}

impl Controller {
    pub fn network(network: FeedforwardNetwork) -> Self {
        let scratchpad = network.create_scratchpad();
        Self::Network {
            network,
            scratchpad,
        }
    }

    /// Outputs per evaluation: one per driven joint.
    pub fn num_outputs(&self) -> usize {
        match self {
            Self::Network { network, .. } => network.num_outputs(),
            Self::Patterns(patterns) => patterns.num_joints(),
        }
    }

    /// The senses a network reads; `None` for pattern generators, which
    /// read only the touch sensors, wherever the input layout puts them.
    pub fn num_inputs(&self) -> Option<usize> {
        match self {
            Self::Network { network, .. } => Some(network.num_inputs()),
            Self::Patterns(_) => None,
        }
    }
}

/// The controller that drives `blueprint`'s body with `senses`: the genome
/// itself for a direct encoding (sized for them when it was bred), the
/// network its CPPN paints onto the body, with their inputs, for HyperNEAT,
/// and the pattern generators its CPPN paints on the body's joints for
/// pattern generators.
pub fn compile_brain(
    genotype: &AgentGenotype,
    blueprint: &RobotBlueprint,
    senses: Senses,
) -> Result<Controller, BrainError> {
    let mut genome = genotype.controller.clone();
    genome.break_cycles();
    let compiled = CppnEvaluator::new(&genome)?;
    let shaped = |needs: (usize, usize)| {
        let has = (compiled.num_inputs(), compiled.num_outputs());
        if has == needs {
            Ok(())
        } else {
            Err(BrainError::CppnShape { needs, has })
        }
    };
    match genotype.encoding {
        BrainEncoding::Direct => Ok(Controller::network(compiled.network().clone())),
        BrainEncoding::HyperNeat => {
            shaped((CPPN_INPUTS, CPPN_OUTPUTS))?;
            let network = paint_brain(&compiled, &Substrate::of_body(blueprint, senses));
            Ok(Controller::network(network))
        }
        BrainEncoding::Supg => {
            shaped((SUPG_INPUTS, SUPG_OUTPUTS))?;
            Ok(Controller::Patterns(SupgBrain::new(compiled, blueprint)))
        }
    }
}

/// A target-task robot's steering reflex (Lan et al. 2020): each driven
/// joint's output is scaled by `1 + gain x side x turn`, within `[0, 2]`, so
/// the joints on one side of the body swing shorter than the other's and
/// the robot turns. The brain still reads the bearing and can steer on its
/// own; the reflex gives every body, whatever its shape, a way to turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Steering {
    /// The evolved gain ([`AgentGenotype::steering_gain`]).
    pub gain: f32,
    /// Each driven joint's side of the body, -1 to 1, in output order (see
    /// [`crate::simulation::hyperneat::joint_sides`]).
    pub sides: Vec<f32>,
    /// Where the target lies: its bearing over pi, -1 to 1, 0 straight
    /// ahead. Updated with the senses.
    pub turn: f32,
}

impl Steering {
    /// The scale on driven joint `index`'s output.
    pub fn scale(&self, index: usize) -> f32 {
        let side = self.sides.get(index).copied().unwrap_or(0.0);
        (1.0 + self.gain * side * self.turn).clamp(0.0, 2.0)
    }
}

/// A robot's controller and the joints it drives. Lives on the body part.
#[derive(Component)]
pub struct RobotBrain {
    pub controller: Controller,
    pub agent_index: usize,
    /// Inputs, filled every fixed step by `gather_sensor_inputs` in the
    /// layout of [`crate::simulation::sensors::InputLayout`].
    pub sensor_inputs: Vec<f32>,
    /// The network's last outputs: hinge commands first, then prismatic.
    pub outputs: Vec<f32>,
    /// Hinge joints, matching the first outputs in order.
    pub hinge_joints: Vec<DrivenJoint>,
    /// Prismatic joints, matching the outputs after the hinges.
    pub prismatic_joints: Vec<DrivenJoint>,
    /// The robot's parts other than the body, for the explosion check.
    pub parts: Vec<Entity>,
    /// Frequency (Hz) of the clock this brain reads, from the genotype.
    pub rhythm_hz: f32,
    /// The steering reflex of a robot walking to targets.
    pub steering: Option<Steering>,
    /// Where the brain reads its body's tilt, with the tilt sense.
    pub tilt_input: Option<usize>,
}

impl RobotBrain {
    fn driven_joints(&self) -> impl Iterator<Item = &DrivenJoint> {
        self.hinge_joints.iter().chain(&self.prismatic_joints)
    }

    /// The legs with a driven joint, in order.
    pub fn legs(&self) -> Vec<usize> {
        let mut legs: Vec<usize> = self.driven_joints().filter_map(|j| j.leg).collect();
        legs.sort_unstable();
        legs.dedup();
        legs
    }

    /// The legs with a broken joint, in order.
    pub fn broken_legs(&self) -> Vec<usize> {
        let mut legs: Vec<usize> = self
            .driven_joints()
            .filter(|j| j.broken)
            .filter_map(|j| j.leg)
            .collect();
        legs.sort_unstable();
        legs.dedup();
        legs
    }

    /// Breaks every driven joint on `leg`: the leg hangs limp.
    pub fn break_leg(&mut self, leg: usize) {
        for joint in self
            .hinge_joints
            .iter_mut()
            .chain(&mut self.prismatic_joints)
        {
            if joint.leg == Some(leg) {
                joint.broken = true;
            }
        }
    }

    /// Mends every broken joint.
    pub fn mend(&mut self) {
        for joint in self
            .hinge_joints
            .iter_mut()
            .chain(&mut self.prismatic_joints)
        {
            joint.broken = false;
        }
    }

    /// Joints the brain drives (hinges, then sliders).
    pub fn driven_joint_count(&self) -> usize {
        self.hinge_joints.len() + self.prismatic_joints.len()
    }

    /// The brain's last outputs (hinges, then sliders) on the metrics' speed
    /// scale ([`MAX_MOTOR_VELOCITY`], [`MAX_PRISMATIC_VELOCITY`]): what the
    /// metrics judge for smoothness and intensity, however the motors carry
    /// them out.
    pub fn commands(&self) -> impl Iterator<Item = f32> + '_ {
        let hinges = self.hinge_joints.len();
        self.outputs
            .iter()
            .take(self.driven_joint_count())
            .enumerate()
            .map(move |(i, &output)| {
                if i < hinges {
                    output * MAX_MOTOR_VELOCITY
                } else {
                    output * MAX_PRISMATIC_VELOCITY
                }
            })
    }
}

/// Evaluates every brain from its gathered inputs and sets its joints'
/// motor targets. Runs after `gather_sensor_inputs`, every
/// [`BRAIN_EVAL_INTERVAL`] steps of the generation clock.
///
/// Position control: each output picks a target within its joint's range
/// and avian's motor (a critically damped 5 Hz spring by default) servoes
/// the joint there. Every driven joint's motor runs unless the joint is
/// broken: avian makes joints with their motors off, and a hinge without a
/// blueprint limit used to stay that way, a "motor" that never moved. The motor's velocity target stays 0. Driving the
/// velocity target instead, as the lab once did, leaves that spring pulling
/// every joint back to rest, so a full-speed command held a hinge only about
/// 0.19 rad from it.
pub fn drive_motors(
    mut brains: Query<&mut RobotBrain>,
    mut hinges: Query<&mut RevoluteJoint>,
    mut sliders: Query<&mut PrismaticJoint>,
    clock: Res<GenerationClock>,
) {
    if !clock.step().is_multiple_of(BRAIN_EVAL_INTERVAL) {
        return;
    }
    for mut brain in &mut brains {
        let RobotBrain {
            controller,
            sensor_inputs,
            outputs,
            hinge_joints,
            prismatic_joints,
            steering,
            rhythm_hz,
            ..
        } = &mut *brain;
        outputs.resize(controller.num_outputs(), 0.0);
        match controller {
            Controller::Network {
                network,
                scratchpad,
            } => network.evaluate_into(sensor_inputs, outputs, scratchpad),
            Controller::Patterns(patterns) => {
                patterns.evaluate(sensor_inputs, clock.elapsed_secs(), *rhythm_hz, outputs);
            }
        }
        if let Some(steering) = steering {
            for (i, output) in outputs.iter_mut().enumerate() {
                *output *= steering.scale(i);
            }
        }
        // What the motors carry out (and so what the metrics judge): the
        // steering reflex can push an output past the range's ends.
        for output in outputs.iter_mut() {
            *output = output.clamp(-1.0, 1.0);
        }

        for (driven, &output) in hinge_joints.iter().zip(outputs.iter()) {
            if let Ok(mut joint) = hinges.get_mut(driven.entity) {
                joint.motor.target_position = driven.target(output);
                joint.motor.target_velocity = 0.0;
                joint.motor.max_torque = driven.max_effort;
                joint.motor.enabled = !driven.broken;
            }
        }
        let slider_outputs = outputs.iter().skip(hinge_joints.len());
        for (driven, &output) in prismatic_joints.iter().zip(slider_outputs) {
            if let Ok(mut joint) = sliders.get_mut(driven.entity) {
                joint.motor.target_position = driven.target(output);
                joint.motor.target_velocity = 0.0;
                joint.motor.max_force = driven.max_effort;
                joint.motor.enabled = !driven.broken;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_steering_reflex_shortens_one_side() {
        let steering = Steering {
            gain: -1.0,
            sides: vec![1.0, -1.0, 0.0],
            turn: 0.5,
        };
        // Target to the positive side: that side's joints swing shorter.
        assert_eq!(steering.scale(0), 0.5);
        assert_eq!(steering.scale(1), 1.5);
        assert_eq!(steering.scale(2), 1.0);
        assert_eq!(steering.scale(9), 1.0);
        let strong = Steering {
            gain: 2.0,
            turn: 1.0,
            ..steering
        };
        assert_eq!((strong.scale(0), strong.scale(1)), (2.0, 0.0));
    }
}
