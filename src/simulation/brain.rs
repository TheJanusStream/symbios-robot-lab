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
pub fn drive_motors(
    brains: Query<(&RobotBrain, &Transform)>,
    mut joints: Query<(&AgentJoint, &mut RevoluteJoint)>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();

    for (brain, root_tf) in &brains {
        // Build sensor inputs
        let forward = root_tf.forward();
        let up = root_tf.up();
        // Pitch: angle between forward vector and horizontal plane
        let pitch = forward.y.asin();
        // Roll: angle between up vector and world up
        let roll = up.x.atan2(up.y);

        let inputs = [t.sin(), t.cos(), pitch, roll];

        let outputs = brain.evaluator.evaluate(&inputs);

        // Apply outputs to matching joints
        for (agent_joint, mut joint) in &mut joints {
            if agent_joint.agent_index != brain.agent_index {
                continue;
            }
            if let Some(&output) = outputs.get(agent_joint.joint_index) {
                // output is tanh-activated: [-1, 1] → motor velocity
                joint.motor.target_velocity = output * MAX_MOTOR_VELOCITY;
                joint.motor.max_torque = MAX_MOTOR_TORQUE;
            }
        }
    }
}
