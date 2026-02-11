use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_neat::CppnEvaluator;

/// Marker linking a joint entity back to its owning agent.
#[derive(Component)]
pub struct AgentJoint {
    pub agent_index: usize,
    pub joint_index: usize,
}

/// Holds the compiled NEAT network for a single agent.
#[derive(Component)]
pub struct RobotBrain {
    pub evaluator: CppnEvaluator,
    pub agent_index: usize,
}

/// Standard inputs fed to every NEAT controller each tick:
///   0: sin(time)    — oscillation clock
///   1: cos(time)    — phase-shifted clock
///   2: body pitch   — forward/backward tilt (from gravity direction)
///   3: body roll    — left/right tilt
///
/// Standard outputs (one per joint):
///   Each output is tanh-activated → [-1, 1], mapped to target_velocity.
pub const NUM_SENSOR_INPUTS: usize = 4;

/// Maximum motor speed (rad/s) that an output of ±1 maps to.
const MAX_MOTOR_VELOCITY: f32 = 15.0;

/// Maximum torque (N·m) each motor can apply.
const MAX_MOTOR_TORQUE: f32 = 5000.0;

/// System that evaluates each agent's NEAT brain and drives its joint motors.
///
/// Two-pass approach to avoid O(N×M) nested iteration:
///   1. Evaluate all brains and index outputs by agent_index.
///   2. Iterate joints once, looking up pre-computed outputs.
pub fn drive_motors(
    brains: Query<(&RobotBrain, &Transform)>,
    mut joints: Query<(&AgentJoint, &mut RevoluteJoint)>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();

    // Pass 1: evaluate each brain and store outputs by agent index
    let mut motor_outputs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(brains.iter().len());
    for (brain, root_tf) in &brains {
        let forward = root_tf.forward();
        let up = root_tf.up();
        let pitch = forward.y.asin();
        let roll = up.x.atan2(up.y);

        let inputs = [t.sin(), t.cos(), pitch, roll];
        let outputs = brain.evaluator.evaluate(&inputs);
        motor_outputs.push((brain.agent_index, outputs));
    }

    // Build a lookup: agent_index → &[f32] outputs
    // Agent indices are dense 0..N, so a Vec lookup is faster than HashMap.
    let max_index = motor_outputs.iter().map(|(i, _)| *i).max().unwrap_or(0);
    let mut output_table: Vec<Option<&[f32]>> = vec![None; max_index + 1];
    for (idx, outputs) in &motor_outputs {
        output_table[*idx] = Some(outputs.as_slice());
    }

    // Pass 2: iterate joints once, apply pre-computed motor commands
    for (agent_joint, mut joint) in &mut joints {
        if let Some(Some(&output)) = output_table
            .get(agent_joint.agent_index)
            .and_then(|outputs| outputs.map(|o| o.get(agent_joint.joint_index)))
        {
            joint.motor.target_velocity = output * MAX_MOTOR_VELOCITY;
            joint.motor.max_torque = MAX_MOTOR_TORQUE;
        }
    }
}
