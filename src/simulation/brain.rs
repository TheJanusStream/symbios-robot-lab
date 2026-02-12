use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_neat::CppnEvaluator;

/// Marker linking a module (rigid body) entity to its owning agent.
/// Used to detect physics explosions (parts drifting apart).
#[derive(Component)]
pub struct AgentModule {
    pub agent_index: usize,
}

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
    /// Number of inputs the NEAT genome expects (clocks + IMU + touch).
    pub num_inputs: usize,
}

/// Marker linking an IMU-bearing module entity to its owning agent.
/// The IMU contributes 3 inputs: pitch, roll, and yaw of the module.
#[derive(Component)]
pub struct AgentImu {
    pub agent_index: usize,
    pub sensor_index: usize,
}

/// Marker linking a Touch-bearing module entity to its owning agent.
/// Touch contributes 1 input: 1.0 if in contact, 0.0 otherwise.
#[derive(Component)]
pub struct AgentTouch {
    pub agent_index: usize,
    pub sensor_index: usize,
}

/// Number of clock inputs (sin(t), cos(t)).
pub const NUM_CLOCK_INPUTS: usize = 2;
/// Each IMU contributes 3 values (pitch, roll, yaw).
pub const INPUTS_PER_IMU: usize = 3;
/// Each touch sensor contributes 1 value (contact boolean).
pub const INPUTS_PER_TOUCH: usize = 1;

/// Computes the total number of NEAT inputs for a given sensor topology.
///
/// Layout:
///   0..2:          sin(t), cos(t)  — oscillation clocks
///   2..2+3N_imu:   pitch, roll, yaw per IMU sensor
///   ..+N_touch:    contact flag per touch sensor
///
/// Outputs (one per hinge joint):
///   Each output is tanh-activated → [-1, 1], mapped to target_velocity.
pub fn compute_num_inputs(num_imus: usize, num_touches: usize) -> usize {
    NUM_CLOCK_INPUTS + num_imus * INPUTS_PER_IMU + num_touches * INPUTS_PER_TOUCH
}

/// Maximum motor speed (rad/s) that an output of ±1 maps to.
const MAX_MOTOR_VELOCITY: f32 = 5.0;

/// Maximum torque (N·m) each motor can apply.
const MAX_MOTOR_TORQUE: f32 = 1000.0;

/// System that evaluates each agent's NEAT brain and drives its joint motors.
///
/// Gathers sensor data (IMU orientation, touch contacts), feeds the NEAT
/// network, and applies motor outputs to joints.
pub fn drive_motors(
    brains: Query<&RobotBrain>,
    imus: Query<(&AgentImu, &Transform)>,
    touches: Query<(&AgentTouch, Option<&CollidingEntities>)>,
    mut joints: Query<(&AgentJoint, &mut RevoluteJoint)>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();

    let max_index = match brains.iter().map(|b| b.agent_index).max() {
        Some(m) => m,
        None => return,
    };

    // Collect IMU readings per agent. Key: (agent_index, sensor_index).
    let mut imu_data: Vec<Vec<(usize, f32, f32, f32)>> = vec![vec![]; max_index + 1];
    for (imu, tf) in &imus {
        let forward = tf.forward();
        let up = tf.up();
        let pitch = forward.y.asin();
        let roll = up.x.atan2(up.y);
        let yaw = forward.x.atan2(forward.z);
        if let Some(bucket) = imu_data.get_mut(imu.agent_index) {
            bucket.push((imu.sensor_index, pitch, roll, yaw));
        }
    }
    // Sort each bucket so sensor_index order is deterministic.
    for bucket in &mut imu_data {
        bucket.sort_by_key(|(idx, _, _, _)| *idx);
    }

    // Collect Touch readings per agent.
    let mut touch_data: Vec<Vec<(usize, f32)>> = vec![vec![]; max_index + 1];
    for (touch, colliding) in &touches {
        let contact = match colliding {
            Some(c) if !c.is_empty() => 1.0,
            _ => 0.0,
        };
        if let Some(bucket) = touch_data.get_mut(touch.agent_index) {
            bucket.push((touch.sensor_index, contact));
        }
    }
    for bucket in &mut touch_data {
        bucket.sort_by_key(|(idx, _)| *idx);
    }

    // Evaluate each brain with assembled inputs and store motor outputs.
    let mut motor_outputs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(brains.iter().len());
    for brain in &brains {
        let idx = brain.agent_index;
        let mut inputs = Vec::with_capacity(brain.num_inputs);

        // Clock
        inputs.push(t.sin());
        inputs.push(t.cos());

        // IMU data (pitch, roll, yaw per sensor)
        if let Some(bucket) = imu_data.get(idx) {
            for &(_, pitch, roll, yaw) in bucket {
                inputs.push(pitch);
                inputs.push(roll);
                inputs.push(yaw);
            }
        }

        // Touch data (contact per sensor)
        if let Some(bucket) = touch_data.get(idx) {
            for &(_, contact) in bucket {
                inputs.push(contact);
            }
        }

        // Pad/truncate to the NEAT genome's expected input count.
        inputs.resize(brain.num_inputs, 0.0);

        let outputs = brain.evaluator.evaluate(&inputs);
        motor_outputs.push((idx, outputs));
    }

    // Build lookup: agent_index → &[f32] outputs
    let mut output_table: Vec<Option<&[f32]>> = vec![None; max_index + 1];
    for (idx, outputs) in &motor_outputs {
        output_table[*idx] = Some(outputs.as_slice());
    }

    // Apply motor commands to joints.
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
