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

/// IMU sensor attached to a module rigid body.
///
/// Provides 6 inputs per tick:
///   - Linear acceleration  (x, y, z) — world-space, derived from velocity delta
///   - Angular acceleration (x, y, z) — world-space, derived from velocity delta
///
/// Previous-frame velocities are stored to compute finite-difference acceleration.
#[derive(Component)]
pub struct AgentImu {
    pub agent_index: usize,
    pub sensor_index: usize,
    pub prev_linear_vel: Vec3,
    pub prev_angular_vel: Vec3,
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
/// Each IMU contributes 6 values (linear accel xyz, angular accel xyz).
pub const INPUTS_PER_IMU: usize = 6;
/// Each touch sensor contributes 1 value (contact boolean).
pub const INPUTS_PER_TOUCH: usize = 1;

/// Computes the total number of NEAT inputs for a given sensor topology.
///
/// Layout:
///   0..2:          sin(t), cos(t)  — oscillation clocks
///   2..2+6N_imu:   linear accel (x,y,z) + angular accel (x,y,z) per IMU
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

/// Persistent buffers for `drive_motors`, reused across frames to avoid
/// per-frame heap allocation.
#[derive(Default)]
pub struct DriveMotorsBuffers {
    /// Flat list of (agent_index, sensor_index, [6 floats]) for IMU data.
    imu_flat: Vec<(usize, usize, [f32; 6])>,
    /// Flat list of (agent_index, sensor_index, contact) for touch data.
    touch_flat: Vec<(usize, usize, f32)>,
    /// Per-agent start/end indices into `imu_flat` (sorted).
    imu_ranges: Vec<(usize, usize)>,
    /// Per-agent start/end indices into `touch_flat` (sorted).
    touch_ranges: Vec<(usize, usize)>,
    /// Reusable input buffer fed to the NEAT evaluator.
    inputs: Vec<f32>,
    /// Flat storage for all motor outputs, laid out agent-by-agent.
    output_storage: Vec<f32>,
    /// Per-agent (start, len) into `output_storage`.
    output_ranges: Vec<(usize, usize)>,
}

/// System that evaluates each agent's NEAT brain and drives its joint motors.
///
/// Gathers sensor data (IMU acceleration, touch contacts), feeds the NEAT
/// network, and applies motor outputs to joints.
/// Uses `Local` buffers to avoid per-frame heap allocations.
#[allow(clippy::too_many_arguments)]
pub fn drive_motors(
    brains: Query<&RobotBrain>,
    mut imus: Query<(&mut AgentImu, &LinearVelocity, &AngularVelocity)>,
    touches: Query<(&AgentTouch, Option<&CollidingEntities>)>,
    mut joints: Query<(&AgentJoint, &mut RevoluteJoint)>,
    time: Res<Time>,
    mut bufs: Local<DriveMotorsBuffers>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs();

    let max_index = match brains.iter().map(|b| b.agent_index).max() {
        Some(m) => m,
        None => return,
    };
    let bucket_count = max_index + 1;

    // --- Collect IMU readings into a flat buffer ---
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };
    bufs.imu_flat.clear();
    for (mut imu, lin_vel, ang_vel) in &mut imus {
        let lin_accel = (lin_vel.0 - imu.prev_linear_vel) * inv_dt;
        let ang_accel = (ang_vel.0 - imu.prev_angular_vel) * inv_dt;

        imu.prev_linear_vel = lin_vel.0;
        imu.prev_angular_vel = ang_vel.0;

        bufs.imu_flat.push((
            imu.agent_index,
            imu.sensor_index,
            [
                lin_accel.x,
                lin_accel.y,
                lin_accel.z,
                ang_accel.x,
                ang_accel.y,
                ang_accel.z,
            ],
        ));
    }
    // Sort by (agent_index, sensor_index) so we can build contiguous ranges.
    bufs.imu_flat
        .sort_unstable_by_key(|(agent, sensor, _)| (*agent, *sensor));

    // Build per-agent ranges into the sorted flat buffer.
    bufs.imu_ranges.clear();
    bufs.imu_ranges.resize(bucket_count, (0, 0));
    {
        let mut i = 0;
        while i < bufs.imu_flat.len() {
            let agent = bufs.imu_flat[i].0;
            let start = i;
            while i < bufs.imu_flat.len() && bufs.imu_flat[i].0 == agent {
                i += 1;
            }
            if agent < bucket_count {
                bufs.imu_ranges[agent] = (start, i);
            }
        }
    }

    // --- Collect Touch readings into a flat buffer ---
    bufs.touch_flat.clear();
    for (touch, colliding) in &touches {
        let contact = match colliding {
            Some(c) if !c.is_empty() => 1.0,
            _ => 0.0,
        };
        bufs.touch_flat
            .push((touch.agent_index, touch.sensor_index, contact));
    }
    bufs.touch_flat
        .sort_unstable_by_key(|(agent, sensor, _)| (*agent, *sensor));

    bufs.touch_ranges.clear();
    bufs.touch_ranges.resize(bucket_count, (0, 0));
    {
        let mut i = 0;
        while i < bufs.touch_flat.len() {
            let agent = bufs.touch_flat[i].0;
            let start = i;
            while i < bufs.touch_flat.len() && bufs.touch_flat[i].0 == agent {
                i += 1;
            }
            if agent < bucket_count {
                bufs.touch_ranges[agent] = (start, i);
            }
        }
    }

    // --- Evaluate each brain, reusing the input buffer ---
    // Destructure to get independent borrows on each field.
    let DriveMotorsBuffers {
        imu_flat,
        touch_flat,
        imu_ranges,
        touch_ranges,
        inputs,
        output_storage,
        output_ranges,
    } = &mut *bufs;

    output_storage.clear();
    output_ranges.clear();
    output_ranges.resize(bucket_count, (0, 0));

    for brain in &brains {
        let idx = brain.agent_index;
        inputs.clear();

        // Clock
        inputs.push(t.sin());
        inputs.push(t.cos());

        // IMU data (linear accel xyz, angular accel xyz per sensor)
        if let Some(&(start, end)) = imu_ranges.get(idx) {
            for entry in &imu_flat[start..end] {
                inputs.extend_from_slice(&entry.2);
            }
        }

        // Touch data (contact per sensor)
        if let Some(&(start, end)) = touch_ranges.get(idx) {
            for entry in &touch_flat[start..end] {
                inputs.push(entry.2);
            }
        }

        // Pad/truncate to the NEAT genome's expected input count.
        inputs.resize(brain.num_inputs, 0.0);

        let outputs = brain.evaluator.evaluate(inputs);
        let out_start = output_storage.len();
        output_storage.extend_from_slice(&outputs);
        if idx < bucket_count {
            output_ranges[idx] = (out_start, outputs.len());
        }
    }

    // --- Apply motor commands to joints ---
    for (agent_joint, mut joint) in &mut joints {
        let idx = agent_joint.agent_index;
        if let Some(&(start, len)) = bufs.output_ranges.get(idx)
            && let Some(&output) = bufs
                .output_storage
                .get(start + agent_joint.joint_index)
                .filter(|_| agent_joint.joint_index < len)
        {
            joint.motor.target_velocity = output * MAX_MOTOR_VELOCITY;
            joint.motor.max_torque = MAX_MOTOR_TORQUE;
        }
    }
}
