use crate::core::population::Population;
use crate::simulation::brain::{
    AgentEndEffector, AgentImu, AgentJoint, AgentModule, AgentPrismaticJoint, AgentTouch,
    AgentUltrasonic, RobotBrain,
};
use crate::simulation::brain::{
    INPUTS_PER_IMU, INPUTS_PER_JOINT, INPUTS_PER_TOUCH, INPUTS_PER_ULTRASONIC, NUM_CLOCK_INPUTS,
};
use crate::simulation::lifecycle::derive_blueprint;
use crate::simulation::scoring::FitnessBaseline;
use avian3d::prelude::CollidingEntities;
use bevy::prelude::*;
use bevy_symbios::materials::{MaterialPalette, setup_material_assets};
use bevy_symbios::robot::spawn_robot;
use symbios_neat::CppnEvaluator;
use symbios_robot::{JointType, RobotBlueprint, SensorType};

pub fn setup_materials(
    commands: Commands,
    materials: ResMut<Assets<StandardMaterial>>,
    images: ResMut<Assets<Image>>,
) {
    setup_material_assets(commands, materials, images);
}

/// Reactively spawns robots when the Population resource changes.
/// Attaches RobotBrain (with entity refs) and sensor/joint components.
/// Uses cached blueprints when available to avoid redundant L-System parsing.
pub fn spawn_agents(
    mut commands: Commands,
    mut population: ResMut<Population>,
    palette: Res<MaterialPalette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !population.is_changed() || population.agents.is_empty() {
        return;
    }

    let pop_size = population.agents.len();
    let cols = (pop_size as f32).sqrt().ceil() as usize;
    let spacing = 10.0;
    let offset = (cols as f32 * spacing) / 2.0;

    for (i, agent) in population.agents.iter_mut().enumerate() {
        if agent.root_entity.is_some() {
            continue;
        }

        // Use cached blueprint or derive fresh from L-System morphology.
        let blueprint = match agent.cached_blueprint.take() {
            Some(bp) => bp,
            None => match derive_blueprint(
                &agent.genotype.morphology,
                agent.genotype.lsystem_seed,
                agent.genotype.lsystem_iterations,
            ) {
                Some((bp, _, _)) => bp,
                None => {
                    warn!("Agent {}: failed to parse morphology", i);
                    continue;
                }
            },
        };

        let stats = count_blueprint_stats(&blueprint);

        agent.num_motors = stats.num_hinges + stats.num_prismatics;

        info!(
            "Spawning Agent {}: {} modules, {}H/{}P joints, {} IMU, {} touch, {} ultrasonic, {} EE",
            i,
            stats.num_modules,
            stats.num_hinges,
            stats.num_prismatics,
            stats.num_imus,
            stats.num_touches,
            stats.num_ultrasonics,
            blueprint.end_effectors.len(),
        );

        // Calculate spawn position with height derived from bounding box
        let row = i / cols;
        let col = i % cols;
        let x = col as f32 * spacing - offset;
        let z = row as f32 * spacing - offset;
        let spawn_rot = Quat::from_rotation_x(-std::f32::consts::PI);
        let aabb = blueprint.aabb(spawn_rot);
        let spawn_y = -aabb.min.y;
        let spawn_pos = Vec3::new(x, spawn_y, z);
        let mut genome = agent.genotype.controller.clone();
        genome.break_cycles();
        let evaluator = match CppnEvaluator::new(&genome) {
            Ok(e) => e,
            Err(err) => {
                warn!("Agent {}: failed to compile NEAT genome: {:?}", i, err);
                continue;
            }
        };
        let scratchpad = evaluator.create_scratchpad();

        let root = commands
            .spawn((
                Transform::from_translation(spawn_pos),
                Visibility::default(),
                Name::new(format!("Agent_{}", i)),
            ))
            .id();

        // Spawn physics hierarchy with rotation only — translation is handled
        // by the root entity. Using the full spawn_transform here would double
        // the translation since Bevy child transforms are relative to the parent.
        let spawned = spawn_robot(
            &mut commands,
            &blueprint,
            &palette,
            &mut meshes,
            Transform::from_rotation(spawn_rot),
        );

        // Tag all modules with AgentModule for explosion detection
        for &module_entity in &spawned.modules {
            commands.entity(module_entity).insert(AgentModule);
        }

        // Parent all spawned entities under root for proper cleanup on despawn
        {
            let mut root_cmds = commands.entity(root);
            for &module_entity in &spawned.modules {
                root_cmds.add_child(module_entity);
            }
            for &(joint_entity, _) in &spawned.joints {
                root_cmds.add_child(joint_entity);
            }
        }

        let num_inputs = genome.input_ids.len();

        // The body entity (first module) will hold the RobotBrain.
        let body_entity = match spawned.modules.first() {
            Some(&e) => e,
            None => continue,
        };

        // Attach sensor components with brain_entity and input_offset so
        // gather_sensor_inputs can iterate sensors linearly and write
        // directly to the brain's input buffer.
        let mut imu_index = 0usize;
        let mut touch_index = 0usize;
        let mut ultrasonic_index = 0usize;
        let touch_offset_base = NUM_CLOCK_INPUTS + stats.num_imus * INPUTS_PER_IMU;
        let ultrasonic_offset_base = touch_offset_base + stats.num_touches * INPUTS_PER_TOUCH;
        for &(sensor_entity, sensor_type) in &spawned.sensors {
            match sensor_type {
                SensorType::IMU => {
                    let input_offset = NUM_CLOCK_INPUTS + imu_index * INPUTS_PER_IMU;
                    commands.entity(sensor_entity).insert(AgentImu {
                        prev_linear_vel: Vec3::ZERO,
                        prev_angular_vel: Vec3::ZERO,
                        brain_entity: body_entity,
                        input_offset,
                    });
                    imu_index += 1;
                }
                SensorType::Touch => {
                    let input_offset = touch_offset_base + touch_index;
                    commands.entity(sensor_entity).insert((
                        AgentTouch {
                            brain_entity: body_entity,
                            input_offset,
                            prev_in_contact: false,
                        },
                        CollidingEntities::default(),
                    ));
                    touch_index += 1;
                }
                SensorType::Ultrasonic => {
                    let input_offset =
                        ultrasonic_offset_base + ultrasonic_index * INPUTS_PER_ULTRASONIC;
                    commands.entity(sensor_entity).insert(AgentUltrasonic {
                        brain_entity: body_entity,
                        input_offset,
                    });
                    ultrasonic_index += 1;
                }
                _ => {}
            }
        }

        let joint_input_base =
            ultrasonic_offset_base + stats.num_ultrasonics * INPUTS_PER_ULTRASONIC;
        let mut hinge_index = 0usize;
        let mut joint_entities = Vec::with_capacity(stats.num_hinges);
        let mut prismatic_entities = Vec::with_capacity(stats.num_prismatics);
        for &(joint_entity, joint_type) in &spawned.joints {
            match joint_type {
                JointType::Hinge { .. } => {
                    let input_offset = joint_input_base + hinge_index * INPUTS_PER_JOINT;
                    commands.entity(joint_entity).insert(AgentJoint {
                        brain_entity: body_entity,
                        input_offset,
                    });
                    joint_entities.push(joint_entity);
                    hinge_index += 1;
                }
                JointType::Prismatic { .. } => {
                    commands.entity(joint_entity).insert(AgentPrismaticJoint);
                    prismatic_entities.push(joint_entity);
                }
                _ => {}
            }
        }

        // End-effector visual markers — small magenta unlit spheres parented to
        // the EE's host module so they track its motion. Pure cosmetic; the EE
        // payload (id, local_position) lives in `blueprint.end_effectors`.
        spawn_end_effector_markers(
            &mut commands,
            &mut meshes,
            &mut materials,
            &blueprint,
            &spawned.modules,
        );

        // Module entities excluding the body (first module) for explosion checks.
        let module_entities: Vec<Entity> = spawned.modules.iter().skip(1).copied().collect();

        // Total mass of all modules — denominator of the Cost-of-Transport metric.
        let total_mass: f32 = blueprint.modules.values().map(|m| m.mass).sum();
        let driven_joint_count = joint_entities.len() + prismatic_entities.len();
        let revolute_joint_count = joint_entities.len();

        // Attach brain to the body entity with pre-allocated sensor input buffer.
        commands.entity(body_entity).insert((
            RobotBrain {
                evaluator,
                scratchpad,
                agent_index: i,
                sensor_inputs: vec![0.0; num_inputs],
                joint_entities,
                prismatic_entities,
                module_entities,
                total_mass,
                touch_sensor_count: stats.num_touches as u16,
                energy_consumed: 0.0,
                mechanical_work: 0.0,
                up_alignment_sum: 0.0,
                up_alignment_sq_sum: 0.0,
                height_sum: 0.0,
                height_sq_sum: 0.0,
                prev_target_velocities: vec![0.0; driven_joint_count],
                prev_prev_target_velocities: vec![0.0; driven_joint_count],
                prev_joint_angles: vec![0.0; revolute_joint_count],
                action_smoothness_sum: 0.0,
                control_jerk_sum: 0.0,
                prev_body_lin_vel: Vec3::ZERO,
                body_jerk_sum: 0.0,
                touch_contact_ticks: 0,
                foot_transition_count: 0,
                perturbed: false,
                metric_ticks: 0,
            },
            FitnessBaseline(None),
        ));

        agent.root_entity = Some(root);
        agent.body_entity = Some(body_entity);
    }
}

/// Spawns one small unlit-magenta sphere per `EndEffector`, parented to the EE's
/// host module so the marker follows the limb in real time.
fn spawn_end_effector_markers(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    blueprint: &RobotBlueprint,
    module_entities: &[Entity],
) {
    if blueprint.end_effectors.is_empty() {
        return;
    }
    let module_id_to_entity: std::collections::HashMap<_, _> = blueprint
        .modules
        .keys()
        .copied()
        .zip(module_entities.iter().copied())
        .collect();
    let mesh = meshes.add(Sphere::new(0.08).mesh().ico(2).unwrap());
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.2, 1.0),
        emissive: LinearRgba::new(2.0, 0.5, 2.0, 1.0),
        unlit: true,
        ..default()
    });
    for ee in &blueprint.end_effectors {
        let Some(&module_entity) = module_id_to_entity.get(&ee.module_id) else {
            continue;
        };
        let marker = commands
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(ee.local_position).with_rotation(ee.local_rotation),
                AgentEndEffector,
                Name::new(format!("EE_{}", ee.id)),
            ))
            .id();
        commands.entity(module_entity).add_child(marker);
    }
}

/// Topology counts derived from a blueprint, used both for log output and for
/// computing brain input/output offsets at spawn time.
struct BlueprintStats {
    num_modules: usize,
    num_hinges: usize,
    num_prismatics: usize,
    num_imus: usize,
    num_touches: usize,
    num_ultrasonics: usize,
}

fn count_blueprint_stats(blueprint: &RobotBlueprint) -> BlueprintStats {
    let num_modules = blueprint.modules.len();
    let num_hinges = blueprint
        .joints
        .iter()
        .filter(|j| matches!(j.joint_type, JointType::Hinge { .. }))
        .count();
    let num_prismatics = blueprint
        .joints
        .iter()
        .filter(|j| matches!(j.joint_type, JointType::Prismatic { .. }))
        .count();
    let mut num_imus = 0usize;
    let mut num_touches = 0usize;
    let mut num_ultrasonics = 0usize;
    for module in blueprint.modules.values() {
        for sensor in &module.sensors {
            match sensor.sensor_type {
                SensorType::IMU => num_imus += 1,
                SensorType::Touch => num_touches += 1,
                SensorType::Ultrasonic => num_ultrasonics += 1,
                _ => {}
            }
        }
    }
    BlueprintStats {
        num_modules,
        num_hinges,
        num_prismatics,
        num_imus,
        num_touches,
        num_ultrasonics,
    }
}
