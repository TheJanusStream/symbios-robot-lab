use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_neat::{CppnEvaluator, EvalScratchpad};

/// Marker linking a module (rigid body) entity to its owning agent.
#[derive(Component)]
pub struct AgentModule;

/// Marker linking a joint entity back to its owning agent.
#[derive(Component)]
pub struct AgentJoint;

/// Holds the compiled NEAT network for a single agent, along with direct
/// entity references to its sensors and joints for O(1) access.
#[derive(Component)]
pub struct RobotBrain {
    pub evaluator: CppnEvaluator,
    /// Reusable scratchpad for allocation-free NEAT evaluation.
    pub scratchpad: EvalScratchpad,
    pub agent_index: usize,
    /// Number of inputs the NEAT genome expects (clocks + IMU + touch).
    pub num_inputs: usize,
    /// IMU sensor entities, ordered by sensor_index.
    pub imu_entities: Vec<Entity>,
    /// Touch sensor entities, ordered by sensor_index.
    pub touch_entities: Vec<Entity>,
    /// Hinge joint entities, ordered by joint_index (matches NEAT output order).
    pub joint_entities: Vec<Entity>,
    /// Module entities excluding the brain/body entity, for explosion checks.
    pub module_entities: Vec<Entity>,
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
    pub prev_linear_vel: Vec3,
    pub prev_angular_vel: Vec3,
}

/// Marker linking a Touch-bearing module entity to its owning agent.
/// Touch contributes 1 input: 1.0 if in contact, 0.0 otherwise.
#[derive(Component)]
pub struct AgentTouch;

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
    /// Reusable input buffer fed to the NEAT evaluator.
    inputs: Vec<f32>,
    /// Reusable output buffer written by `evaluate_into`.
    outputs: Vec<f32>,
    /// Collected motor commands: (joint_entity, output_value).
    motor_commands: Vec<(Entity, f32)>,
}

/// System that evaluates each agent's NEAT brain and drives its joint motors.
///
/// Uses direct entity references stored on `RobotBrain` for O(1) sensor/joint
/// access, eliminating the previous per-frame global sort. The NEAT evaluator
/// uses `evaluate_into` with a per-brain scratchpad to avoid output allocation.
pub fn drive_motors(
    mut brains: Query<&mut RobotBrain>,
    mut imus: Query<(&mut AgentImu, &LinearVelocity, &AngularVelocity)>,
    touches: Query<(Option<&CollidingEntities>,), With<AgentTouch>>,
    mut joints: Query<&mut RevoluteJoint>,
    time: Res<Time>,
    mut bufs: Local<DriveMotorsBuffers>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs();
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };

    bufs.motor_commands.clear();

    for brain in &mut brains {
        bufs.inputs.clear();

        // Clock
        bufs.inputs.push(t.sin());
        bufs.inputs.push(t.cos());

        // IMU data via direct entity refs (sorted by sensor_index at spawn)
        for &imu_entity in &brain.imu_entities {
            if let Ok((mut imu, lin_vel, ang_vel)) = imus.get_mut(imu_entity) {
                let lin_accel = (lin_vel.0 - imu.prev_linear_vel) * inv_dt;
                let ang_accel = (ang_vel.0 - imu.prev_angular_vel) * inv_dt;
                imu.prev_linear_vel = lin_vel.0;
                imu.prev_angular_vel = ang_vel.0;
                bufs.inputs.extend_from_slice(&[
                    lin_accel.x,
                    lin_accel.y,
                    lin_accel.z,
                    ang_accel.x,
                    ang_accel.y,
                    ang_accel.z,
                ]);
            }
        }

        // Touch data via direct entity refs (sorted by sensor_index at spawn)
        for &touch_entity in &brain.touch_entities {
            if let Ok((colliding,)) = touches.get(touch_entity) {
                let contact = match colliding {
                    Some(c) if !c.is_empty() => 1.0,
                    _ => 0.0,
                };
                bufs.inputs.push(contact);
            }
        }

        // Pad/truncate to the NEAT genome's expected input count.
        let DriveMotorsBuffers {
            inputs,
            outputs,
            motor_commands,
        } = &mut *bufs;
        inputs.resize(brain.num_inputs, 0.0);

        // Evaluate into reusable output buffer with per-brain scratchpad.
        let num_out = brain.evaluator.num_outputs();
        outputs.resize(num_out, 0.0);
        let brain = brain.into_inner();
        brain
            .evaluator
            .evaluate_into(inputs, outputs, &mut brain.scratchpad);

        // Collect motor commands via direct joint entity refs
        for (ji, &joint_entity) in brain.joint_entities.iter().enumerate() {
            if let Some(&output) = outputs.get(ji) {
                motor_commands.push((joint_entity, output));
            }
        }
    }

    // Apply motor commands to joints.
    for &(joint_entity, output) in &bufs.motor_commands {
        if let Ok(mut joint) = joints.get_mut(joint_entity) {
            joint.motor.target_velocity = output * MAX_MOTOR_VELOCITY;
            joint.motor.max_torque = MAX_MOTOR_TORQUE;
        }
    }
}
