//! The brain: one feed-forward network per robot, evaluated from its sensor
//! inputs, whose outputs drive the robot's joint motors.

use crate::core::config::{BrainEncoding, Task};
use crate::core::genome::AgentGenotype;
use crate::simulation::clock::GenerationClock;
use crate::simulation::hyperneat::{CPPN_INPUTS, CPPN_OUTPUTS, Substrate, paint_brain};
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
    /// A HyperNEAT controller is not shaped like the lab's CPPN.
    CppnShape { inputs: usize, outputs: usize },
}

impl std::fmt::Display for BrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainError::Cyclic => f.write_str("the brain's genome has a cycle"),
            BrainError::CppnShape { inputs, outputs } => write!(
                f,
                "a HyperNEAT CPPN needs {CPPN_INPUTS} inputs and {CPPN_OUTPUTS} outputs, \
                 this one has {inputs} and {outputs}"
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

/// The network that drives `blueprint`'s body at `task`: the controller
/// itself for a direct encoding (sized for the task when it was bred), the
/// network its CPPN paints onto the body, with the task's inputs, for
/// HyperNEAT.
pub fn compile_brain(
    genotype: &AgentGenotype,
    blueprint: &RobotBlueprint,
    task: Task,
) -> Result<FeedforwardNetwork, BrainError> {
    let mut genome = genotype.controller.clone();
    genome.break_cycles();
    let compiled = CppnEvaluator::new(&genome)?;
    match genotype.encoding {
        BrainEncoding::Direct => Ok(compiled.network().clone()),
        BrainEncoding::HyperNeat => {
            let (inputs, outputs) = (compiled.num_inputs(), compiled.num_outputs());
            if (inputs, outputs) != (CPPN_INPUTS, CPPN_OUTPUTS) {
                return Err(BrainError::CppnShape { inputs, outputs });
            }
            Ok(paint_brain(&compiled, &Substrate::of_body(blueprint, task)))
        }
    }
}

/// A robot's controller and the joints it drives. Lives on the body part.
#[derive(Component)]
pub struct RobotBrain {
    pub network: FeedforwardNetwork,
    pub scratchpad: Scratchpad,
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
            network,
            scratchpad,
            sensor_inputs,
            outputs,
            hinge_joints,
            prismatic_joints,
            ..
        } = &mut *brain;
        outputs.resize(network.num_outputs(), 0.0);
        network.evaluate_into(sensor_inputs, outputs, scratchpad);

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
