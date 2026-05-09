use avian3d::prelude::*;
use bevy::prelude::*;
use symbios_neat::{CppnEvaluator, EvalScratchpad};

/// Marker linking a module (rigid body) entity to its owning agent.
#[derive(Component)]
pub struct AgentModule;

/// Joint proprioception sensor attached to a hinge joint entity.
/// Provides 1 input per tick: the current joint angle normalized to [-1, 1].
///
/// `brain_entity` and `input_offset` enable cache-friendly linear iteration,
/// matching the pattern used by `AgentImu` and `AgentTouch`.
#[derive(Component)]
pub struct AgentJoint {
    /// Entity that holds this joint's `RobotBrain`.
    pub brain_entity: Entity,
    /// Index in `RobotBrain.sensor_inputs` for this joint's angle value.
    pub input_offset: usize,
}

/// Tag for a prismatic (slider) joint driven by the brain. No proprioception
/// input wired yet — the brain only writes its `target_velocity`. The output
/// index is the joint's position in `RobotBrain.prismatic_entities`.
#[derive(Component)]
pub struct AgentPrismaticJoint;

/// Ultrasonic distance sensor mounted on a module. On each tick we raycast
/// forward (along the module's local +X) up to `ULTRASONIC_MAX_RANGE` and
/// write `1.0 - hit_distance / max_range` (so closer = larger reading).
///
/// `brain_entity` and `input_offset` enable cache-friendly linear iteration.
#[derive(Component)]
pub struct AgentUltrasonic {
    /// Entity that holds this sensor's `RobotBrain`.
    pub brain_entity: Entity,
    /// Index in `RobotBrain.sensor_inputs` for this sensor's reading.
    pub input_offset: usize,
}

/// Visual marker for a `RobotBlueprint::end_effectors` entry. Pure cosmetic;
/// not yet used by scoring or the brain. Kept so future scoring can find them.
#[derive(Component)]
pub struct AgentEndEffector;

/// Maximum range of the ultrasonic sensor (metres). Hits beyond this clamp to 0.
pub const ULTRASONIC_MAX_RANGE: f32 = 8.0;

/// Holds the compiled NEAT network for a single agent, along with direct
/// entity references to its sensors and joints for O(1) access.
///
/// Beyond the controller and sensor wiring, the brain owns the per-fixed-step
/// metric accumulators consumed by [`crate::simulation::scoring`]. These cover
/// energy / mechanical work, action smoothness, body jerk, orientation
/// variance, periodic forward-progress sampling for R², and support-duty
/// counting. All accumulators are filled by `track_metrics` and reset per
/// generation when the agent respawns.
#[derive(Component)]
pub struct RobotBrain {
    pub evaluator: CppnEvaluator,
    /// Reusable scratchpad for allocation-free NEAT evaluation.
    pub scratchpad: EvalScratchpad,
    pub agent_index: usize,
    /// Pre-allocated input buffer filled by `gather_sensor_inputs`.
    /// Layout: [sin(t), cos(t), imu0..., touch0..., ultrasonic0..., joint0_angle, ...].
    pub sensor_inputs: Vec<f32>,
    /// Hinge joint entities, ordered by joint_index (matches the FIRST N
    /// motor outputs of the NEAT network).
    pub joint_entities: Vec<Entity>,
    /// Prismatic (slider) joint entities, ordered by index. Outputs are
    /// laid out AFTER all hinge outputs, so `output[hinge_count + i]`
    /// drives `prismatic_entities[i]`.
    pub prismatic_entities: Vec<Entity>,
    /// Module entities excluding the brain/body entity, for explosion checks.
    pub module_entities: Vec<Entity>,
    /// Total mass of all modules (computed once at spawn from the blueprint).
    /// Denominator of the Cost-of-Transport metric (`E / (m·g·d)`).
    pub total_mass: f32,
    /// Total touch sensors on this agent (denominator for support duty factor).
    pub touch_sensor_count: u16,

    // ── Energy / mechanical work ───────────────────────────────────────────
    /// Coarse motor energy proxy: sum of `|τ_max · v_target| · dt` per joint.
    /// Kept as the lab's historical "energy" channel; not the CoT numerator.
    pub energy_consumed: f32,
    /// Better mechanical-work approximation: sum of `|τ_max · Δθ|` from actual
    /// joint-angle finite differences. Used for Cost-of-Transport.
    pub mechanical_work: f32,

    // ── Posture (mean + variance via Welford-style accumulators) ───────────
    /// Sum of `dot(local_up, world_up)` per tick (mean = sum/ticks).
    pub up_alignment_sum: f32,
    /// Sum of `up_alignment²` per tick — pairs with `up_alignment_sum` to
    /// derive the population-variance estimator `E[X²] - E[X]²`.
    pub up_alignment_sq_sum: f32,
    /// Sum of body Y per tick (mean height = sum/ticks).
    pub height_sum: f32,

    // ── Smoothness (per-joint Δv, ΔΔv, body Δv) ────────────────────────────
    /// Per-driven-joint commanded velocity at the previous tracked tick.
    /// Length = `joint_entities.len() + prismatic_entities.len()`.
    /// First N entries are revolute (rad/s), then prismatic (m/s).
    pub prev_target_velocities: Vec<f32>,
    /// Per-driven-joint commanded velocity TWO tracked ticks ago. Used with
    /// `prev_target_velocities` for the second-derivative anti-vibration term:
    /// `Σ |a_t − 2·a_{t−1} + a_{t−2}|²`. Same length as `prev_target_velocities`.
    pub prev_prev_target_velocities: Vec<f32>,
    /// Per-revolute-joint angle at the previous tracked tick, used both for
    /// mechanical-work estimation and to detect tracked-tick boundaries.
    /// Length = `joint_entities.len()`.
    pub prev_joint_angles: Vec<f32>,
    /// Sum of squared per-tick Δ(target_velocity) across all joints.
    /// Larger = more chattery control. `SmoothnessReward = 1 / (1 + mean)`.
    pub action_smoothness_sum: f32,
    /// Sum of squared per-tick Δ²(target_velocity) across all joints.
    /// Catches sign-flipping vibration that a first-derivative penalty can
    /// miss (a ±ε oscillation has small |Δv|=2ε but |ΔΔv|=4ε per tick).
    /// `ControlJerkReward = 1 / (1 + mean / scale)`.
    pub control_jerk_sum: f32,
    /// Body LinearVelocity at the previous tracked tick.
    pub prev_body_lin_vel: Vec3,
    /// Sum of squared per-tick Δ(body_velocity). Approximates body jerk.
    pub body_jerk_sum: f32,

    // ── Body-Y oscillation (CoM bounce signature) ──────────────────────────
    /// Sum of `body_y²` per tick. Pairs with `height_sum` to derive
    /// `stddev(body_y)` via the population-variance estimator. Walking has
    /// characteristic vertical bounce; vibrobots are flat.
    pub height_sq_sum: f32,

    // ── Support / step-cycle counters ──────────────────────────────────────
    /// Sum of (touch sensors in contact) across all tracked ticks. Mean
    /// support duty per tick = `touch_contact_ticks / (touch_sensor_count·ticks)`.
    pub touch_contact_ticks: u32,
    /// Total number of contact↔air transitions across all touch sensors,
    /// summed over the eval window. Real walking cycles feet through air;
    /// vibrobots have ~0 transitions.
    pub foot_transition_count: u32,

    // ── Robustness (perturbation injection) ────────────────────────────────
    /// Once-per-agent flag flipped by `apply_perturbations` after the impulse
    /// has been applied, so we don't re-perturb on every tick.
    pub perturbed: bool,

    // ── Bookkeeping ────────────────────────────────────────────────────────
    /// Number of ticks metrics have been tracked (denominator for means).
    pub metric_ticks: u32,
}

/// IMU sensor attached to a module rigid body.
///
/// Provides 6 inputs per tick:
///   - Linear acceleration  (x, y, z) — world-space, derived from velocity delta
///   - Angular acceleration (x, y, z) — world-space, derived from velocity delta
///
/// Previous-frame velocities are stored to compute finite-difference acceleration.
/// `brain_entity` and `input_offset` enable cache-friendly linear iteration in
/// `gather_sensor_inputs` — sensors write directly to their brain's input buffer
/// without the brain needing to look them up.
#[derive(Component)]
pub struct AgentImu {
    pub prev_linear_vel: Vec3,
    pub prev_angular_vel: Vec3,
    /// Entity that holds this sensor's `RobotBrain`.
    pub brain_entity: Entity,
    /// Start index in `RobotBrain.sensor_inputs` for this IMU's 6 values.
    pub input_offset: usize,
}

/// Touch sensor attached to a module rigid body.
/// Touch contributes 1 input: 1.0 if in contact, 0.0 otherwise.
///
/// `brain_entity` and `input_offset` enable cache-friendly linear iteration.
/// `prev_in_contact` lets `track_metrics` count contact↔air transitions per
/// foot — the signature of a real step cycle vs. a vibrating slide.
#[derive(Component)]
pub struct AgentTouch {
    /// Entity that holds this sensor's `RobotBrain`.
    pub brain_entity: Entity,
    /// Index in `RobotBrain.sensor_inputs` for this touch's value.
    pub input_offset: usize,
    /// Contact state at the previous tracked tick. Initialised `false` at
    /// spawn; the first tracked tick may register a spurious transition,
    /// which is acceptable noise vs. the per-window total.
    pub prev_in_contact: bool,
}

/// Number of clock inputs (sin(t), cos(t)).
pub const NUM_CLOCK_INPUTS: usize = 2;
/// Each IMU contributes 6 values (linear accel xyz, angular accel xyz).
pub const INPUTS_PER_IMU: usize = 6;
/// Each touch sensor contributes 1 value (contact boolean).
pub const INPUTS_PER_TOUCH: usize = 1;
/// Each ultrasonic sensor contributes 1 value (proximity in [0, 1]).
pub const INPUTS_PER_ULTRASONIC: usize = 1;
/// Each hinge joint contributes 1 value (current angle / π, normalized to [-1, 1]).
pub const INPUTS_PER_JOINT: usize = 1;

/// Computes the total number of NEAT inputs for a given sensor topology.
///
/// Layout:
///   0..2:                sin(t), cos(t)  — oscillation clocks
///   2..2+6N_imu:         linear accel (x,y,z) + angular accel (x,y,z) per IMU
///   ..+N_touch:          contact flag per touch sensor
///   ..+N_ultra:          proximity reading per ultrasonic sensor
///   ..+N_hinge:          current angle per hinge joint (normalized to [-1, 1])
///
/// Outputs (one per hinge then one per prismatic joint):
///   Each output is tanh-activated → [-1, 1], mapped to target_velocity.
pub fn compute_num_inputs(
    num_imus: usize,
    num_touches: usize,
    num_ultrasonics: usize,
    num_hinges: usize,
) -> usize {
    NUM_CLOCK_INPUTS
        + num_imus * INPUTS_PER_IMU
        + num_touches * INPUTS_PER_TOUCH
        + num_ultrasonics * INPUTS_PER_ULTRASONIC
        + num_hinges * INPUTS_PER_JOINT
}

/// Maximum motor speed (rad/s) that an output of ±1 maps to.
const MAX_MOTOR_VELOCITY: f32 = 3.0;

/// Maximum torque (N·m) each motor can apply.
pub const MAX_MOTOR_TORQUE: f32 = 5000.0;

/// Evaluate brains every Nth fixed step. Motors hold their last commanded
/// velocity between evaluations. This halves NEAT CPU cost while
/// `gather_sensor_inputs` still runs every step to keep IMU data fresh.
const BRAIN_EVAL_INTERVAL: u32 = 2;

/// Persistent buffers for `drive_motors`, reused across frames to avoid
/// per-frame heap allocation.
#[derive(Default)]
pub struct DriveMotorsBuffers {
    /// Reusable output buffer written by `evaluate_into`.
    outputs: Vec<f32>,
    /// Collected motor commands: (joint_entity, output_value).
    motor_commands: Vec<(Entity, f32)>,
}

/// Gathers sensor data from all IMU, touch, and ultrasonic sensors into their
/// owning brain's `sensor_inputs` buffer. Iterates sensor archetypes linearly
/// (cache-friendly) rather than doing per-brain random entity lookups.
///
/// Runs in `FixedUpdate` before `drive_motors`.
pub fn gather_sensor_inputs(
    mut brains: Query<&mut RobotBrain>,
    mut imus: Query<(&mut AgentImu, &LinearVelocity, &AngularVelocity)>,
    touches: Query<(&AgentTouch, Option<&CollidingEntities>)>,
    ultrasonics: Query<(&AgentUltrasonic, &GlobalTransform)>,
    agent_joints: Query<(&AgentJoint, &RevoluteJoint)>,
    rotations: Query<&Rotation>,
    spatial_query: SpatialQuery,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();
    let dt = time.delta_secs();
    let inv_dt = if dt > 0.0 { 1.0 / dt } else { 0.0 };

    // Phase 1: Initialize all brain input buffers with clock signals.
    for mut brain in &mut brains {
        brain.sensor_inputs.fill(0.0);
        if brain.sensor_inputs.len() >= NUM_CLOCK_INPUTS {
            brain.sensor_inputs[0] = t.sin();
            brain.sensor_inputs[1] = t.cos();
        }
    }

    // Phase 2: Linear pass over all IMU sensors — contiguous archetype iteration.
    for (mut imu, lin_vel, ang_vel) in &mut imus {
        let lin_accel = (lin_vel.0 - imu.prev_linear_vel) * inv_dt;
        let ang_accel = (ang_vel.0 - imu.prev_angular_vel) * inv_dt;
        imu.prev_linear_vel = lin_vel.0;
        imu.prev_angular_vel = ang_vel.0;

        if let Ok(mut brain) = brains.get_mut(imu.brain_entity) {
            let o = imu.input_offset;
            if o + 5 < brain.sensor_inputs.len() {
                brain.sensor_inputs[o] = lin_accel.x;
                brain.sensor_inputs[o + 1] = lin_accel.y;
                brain.sensor_inputs[o + 2] = lin_accel.z;
                brain.sensor_inputs[o + 3] = ang_accel.x;
                brain.sensor_inputs[o + 4] = ang_accel.y;
                brain.sensor_inputs[o + 5] = ang_accel.z;
            }
        }
    }

    // Phase 3: Linear pass over all touch sensors.
    for (touch, colliding) in &touches {
        let contact = match colliding {
            Some(c) if !c.is_empty() => 1.0,
            _ => 0.0,
        };
        if let Ok(mut brain) = brains.get_mut(touch.brain_entity)
            && touch.input_offset < brain.sensor_inputs.len()
        {
            brain.sensor_inputs[touch.input_offset] = contact;
        }
    }

    // Phase 3b: Ultrasonic raycasts. Cast forward (sensor's world +X) up to
    // ULTRASONIC_MAX_RANGE; write 1.0 - hit/max so closer = larger reading.
    // Excluding self-body avoids the sensor reading its own mount.
    for (sensor, transform) in &ultrasonics {
        let origin = transform.translation();
        let direction = transform.rotation() * Vec3::X;
        let dir = match Dir3::new(direction) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let filter = SpatialQueryFilter::default();
        let proximity =
            match spatial_query.cast_ray(origin, dir, ULTRASONIC_MAX_RANGE, true, &filter) {
                Some(hit) => (1.0 - (hit.distance / ULTRASONIC_MAX_RANGE)).clamp(0.0, 1.0),
                None => 0.0,
            };
        if let Ok(mut brain) = brains.get_mut(sensor.brain_entity)
            && sensor.input_offset < brain.sensor_inputs.len()
        {
            brain.sensor_inputs[sensor.input_offset] = proximity;
        }
    }

    // Phase 4: Linear pass over all hinge joint angle sensors.
    // Computes the signed angle around the hinge axis from the two body
    // rotations and their local joint frames, normalized to [-1, 1].
    for (agent_joint, joint) in &agent_joints {
        let (Ok(rot1), Ok(rot2)) = (rotations.get(joint.body1), rotations.get(joint.body2)) else {
            continue;
        };
        let basis1 = joint.local_basis1().unwrap_or(Quat::IDENTITY);
        let basis2 = joint.local_basis2().unwrap_or(Quat::IDENTITY);
        let axis = joint.hinge_axis;

        // Perpendicular reference vector to the hinge axis.
        let perp = if axis.x.abs() < 0.9 {
            axis.cross(Vec3::X).normalize()
        } else {
            axis.cross(Vec3::Y).normalize()
        };

        let a1 = (rot1.0 * basis1) * axis;
        let b1 = (rot1.0 * basis1) * perp;
        let b2 = (rot2.0 * basis2) * perp;

        let sin_angle = b1.cross(b2).dot(a1);
        let cos_angle = b1.dot(b2);
        let angle = sin_angle.atan2(cos_angle);

        if let Ok(mut brain) = brains.get_mut(agent_joint.brain_entity)
            && agent_joint.input_offset < brain.sensor_inputs.len()
        {
            brain.sensor_inputs[agent_joint.input_offset] = angle * std::f32::consts::FRAC_1_PI;
        }
    }
}

/// Maximum prismatic motor velocity (m/s) at output ±1.
const MAX_PRISMATIC_VELOCITY: f32 = 1.5;
/// Maximum prismatic motor force (N).
const MAX_PRISMATIC_FORCE: f32 = 5000.0;

/// Evaluates each agent's NEAT brain using pre-gathered sensor inputs
/// and drives its joint motors (revolute first, then prismatic).
///
/// Reads from `RobotBrain.sensor_inputs` (filled by `gather_sensor_inputs`)
/// and writes motor commands to `RevoluteJoint` and `PrismaticJoint`
/// components.
///
/// Runs in `FixedUpdate` after `gather_sensor_inputs`, but only evaluates
/// every `BRAIN_EVAL_INTERVAL` steps. Motors hold their last commanded
/// velocity between evaluations. This halves the NEAT CPU cost — the
/// most expensive per-frame work — without noticeably degrading locomotion.
pub fn drive_motors(
    mut brains: Query<&mut RobotBrain>,
    mut joints: Query<&mut RevoluteJoint>,
    mut prismatics: Query<&mut PrismaticJoint>,
    mut bufs: Local<DriveMotorsBuffers>,
    mut step: Local<u32>,
) {
    *step = step.wrapping_add(1);
    if !step.is_multiple_of(BRAIN_EVAL_INTERVAL) {
        return;
    }

    bufs.motor_commands.clear();
    let mut prismatic_commands: Vec<(Entity, f32)> = Vec::new();

    for brain in &mut brains {
        let DriveMotorsBuffers {
            outputs,
            motor_commands,
        } = &mut *bufs;

        let num_out = brain.evaluator.num_outputs();
        outputs.resize(num_out, 0.0);
        let brain = brain.into_inner();
        brain
            .evaluator
            .evaluate_into(&brain.sensor_inputs, outputs, &mut brain.scratchpad);

        // First N_hinge outputs drive revolute joints.
        for (ji, &joint_entity) in brain.joint_entities.iter().enumerate() {
            if let Some(&output) = outputs.get(ji) {
                motor_commands.push((joint_entity, output));
            }
        }
        // Next N_prismatic outputs drive prismatic joints.
        let prismatic_offset = brain.joint_entities.len();
        for (pi, &joint_entity) in brain.prismatic_entities.iter().enumerate() {
            if let Some(&output) = outputs.get(prismatic_offset + pi) {
                prismatic_commands.push((joint_entity, output));
            }
        }
    }

    for &(joint_entity, output) in &bufs.motor_commands {
        if let Ok(mut joint) = joints.get_mut(joint_entity) {
            joint.motor.target_velocity = output * MAX_MOTOR_VELOCITY;
            joint.motor.max_torque = MAX_MOTOR_TORQUE;
        }
    }
    for &(joint_entity, output) in &prismatic_commands {
        if let Ok(mut joint) = prismatics.get_mut(joint_entity) {
            joint.motor.target_velocity = output * MAX_PRISMATIC_VELOCITY;
            joint.motor.max_force = MAX_PRISMATIC_FORCE;
            joint.motor.enabled = true;
        }
    }
}
