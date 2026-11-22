//! HyperNEAT brains: the genome is a CPPN (compositional pattern-producing
//! network) that paints a body's brain onto the body's geometry
//! (Stanley, D'Ambrosio and Gauci 2009).
//!
//! Every brain input except the clock sits at a point on the body: an IMU's
//! six channels just around the IMU, a touch or distance sensor at its
//! mount, a hinge's angle at the hinge, a slider's extension at the slider,
//! and in the target task the target's bearing by the body's root part. Each driven joint carries two neurons
//! where it sits: an interneuron and the motor neuron whose output drives the
//! joint. The CPPN is asked, for every pair of neurons, whether and how
//! strongly they connect given where they sit, and for every joint, its
//! rhythm (the weights from the clock's sine and cosine, so its phase and
//! strength) and both neurons' biases.
//!
//! Because the brain is a function of the body's geometry rather than a list
//! of numbered inputs, one genome drives any body: an offspring that grows a
//! leg gets a controller for it, and legs placed alike get alike
//! controllers. The direct encoding numbers its inputs and outputs instead,
//! so their meaning shifts whenever crossover or mutation changes the body.
//!
//! Positions are in the robot's spawn frame (y up), centred on its bounding
//! box and scaled so that its longest half-extent is 1.

use crate::core::config::Senses;
use crate::simulation::body::modules_in_order;
use crate::simulation::sensors::NUM_CLOCK_INPUTS;
use crate::simulation::spawn::spawn_rotation;
use bevy::prelude::*;
use rand::Rng;
use symbios_neat::substrate::{
    Substrate as SubstrateGraph, SubstrateNode, SubstratePainter, substrate_to_network_with,
    write_endpoints,
};
use symbios_neat::{Activation, CppnEvaluator, FeedforwardNetwork, NeatConfig, NeatGenome};
use symbios_robot::{JointDefinition, JointType, RobotBlueprint, SensorType};

/// CPPN inputs: the source neuron's position, the target neuron's position
/// and the distance between them.
pub const CPPN_INPUTS: usize = 7;
/// The distance input.
const DISTANCE: usize = 6;

/// CPPN outputs, one per kind of link or value it paints. Giving each kind
/// its own output (a "multi-spatial substrate", Pugh and Stanley 2013) lets
/// even a CPPN with no hidden nodes paint unrelated patterns for each; in
/// particular every joint's sine and cosine weights, and so its phase, vary
/// independently over the body.
pub const CPPN_OUTPUTS: usize = 8;
/// Sensor neuron to interneuron, asked at (sensor, joint).
const SENSOR_TO_INTERNEURON: usize = 0;
/// Sensor neuron to motor neuron, asked at (sensor, joint).
const SENSOR_TO_MOTOR: usize = 1;
/// Interneuron to motor neuron, asked at (joint, joint).
const INTERNEURON_TO_MOTOR: usize = 2;
/// Clock sine to motor neuron, asked at the joint's own position.
const CLOCK_SIN_TO_MOTOR: usize = 3;
/// Clock cosine to motor neuron, asked at the joint's own position.
const CLOCK_COS_TO_MOTOR: usize = 4;
/// Interneuron bias, asked at the joint's own position.
const INTERNEURON_BIAS: usize = 5;
/// Motor neuron bias, asked at the joint's own position.
const MOTOR_BIAS: usize = 6;
/// Link expression (Verbancsics and Stanley 2011): a link between two placed
/// neurons exists only where this output is positive, so the CPPN paints
/// where connections are as well as how strong.
const LINK_EXPRESSION: usize = 7;

/// The weight a CPPN output of ±1 paints. Small, so that motor neurons work
/// in `tanh`'s linear range and follow the clock as smooth waves rather than
/// slamming between their limits. Over 30 generations HyperNEAT reached a
/// final pool-best fitness of 0.63 at 3.0 (most fresh motors sat pinned),
/// 1.36 at 1.0 and 1.5, and 1.79 to 1.88 anywhere from 0.25 to 0.6, against
/// 1.48 for direct NEAT (16 to 32 seeds each); 0.5 sits in that plateau.
/// Direct NEAT gained nothing from smaller weights.
pub const MAX_WEIGHT: f32 = 0.5;
/// The bias a CPPN output of ±1 paints.
pub const MAX_BIAS: f32 = 1.0;
/// A fresh CPPN expresses links shorter than `LOCALITY_SEED_BIAS /
/// LOCALITY_SEED_SLOPE` (0.4 of the body's longest half-extent): its
/// link-expression output starts as `tanh(bias - slope * distance)`.
/// Weights a CPPN paints vary smoothly over the body, so neighbouring
/// inputs pull a neuron the same way; if every link were expressed, a
/// body's many feet and channels would add up and pin its motors at their
/// limits. Evolution can widen, narrow or reshape the rule.
const LOCALITY_SEED_BIAS: f32 = 1.0;
const LOCALITY_SEED_SLOPE: f32 = 2.5;
/// How far an IMU's input neurons sit from it: its linear-acceleration
/// channels on the positive side of each axis of the part it is mounted on,
/// its angular ones on the negative side. Without the offsets all six would
/// sit at one point and get the same weights.
const IMU_CHANNEL_SPREAD: f32 = 0.25;
/// How far from the body's root part the bearing's input neurons sit (in the
/// target task): the cosine's (how far ahead the target lies) toward the
/// robot's heading, the sine's (how far to the side) toward the side where
/// the sine is positive. Apart, they get different weights.
const BEARING_SPREAD: f32 = 0.25;
/// Smallest half-extent the frame divides by, for a degenerate body.
const MIN_FRAME_SCALE: f32 = 1e-3;

/// The NEAT configuration of a HyperNEAT genome's CPPN: periodic and
/// symmetric activations for the patterns, `tanh` outputs.
pub fn cppn_config() -> NeatConfig {
    NeatConfig::cppn(CPPN_INPUTS, CPPN_OUTPUTS)
}

/// A fresh HyperNEAT genome: a fully connected CPPN with random weights,
/// except that its link expression starts as the locality rule above (from
/// the bias and the distance only).
pub fn fresh_cppn(rng: &mut impl Rng) -> NeatGenome {
    let mut genome = NeatGenome::fully_connected(cppn_config(), rng);
    let expression = genome.output_ids[LINK_EXPRESSION];
    let bias = genome.bias_id;
    let distance = genome.input_ids[DISTANCE];
    for connection in genome.connections.values_mut() {
        if connection.output == expression {
            connection.weight = if Some(connection.input) == bias {
                LOCALITY_SEED_BIAS
            } else if connection.input == distance {
                -LOCALITY_SEED_SLOPE
            } else {
                0.0
            };
        }
    }
    genome
}

/// Where a body's neurons sit.
#[derive(Debug, Clone, PartialEq)]
pub struct Substrate {
    /// One position per brain input after the clock, in
    /// [`InputLayout`](crate::simulation::sensors::InputLayout) order.
    pub sensors: Vec<Vec3>,
    /// One position per driven joint (hinges, then sliders, each in
    /// blueprint order), where its interneuron and motor neuron sit.
    pub joints: Vec<Vec3>,
}

impl Substrate {
    /// The substrate of a blueprint's body in its rest pose, walking
    /// forward.
    pub fn of_blueprint(blueprint: &RobotBlueprint) -> Self {
        Self::of_body(blueprint, Senses::NONE)
    }

    /// The substrate of a blueprint's body in its rest pose for `task`: in
    /// the target task the bearing's two input neurons follow the sensors'.
    pub fn of_body(blueprint: &RobotBlueprint, senses: Senses) -> Self {
        let frame = BodyFrame::of(blueprint);
        let mut imus = Vec::new();
        let mut touches = Vec::new();
        let mut ultrasonics = Vec::new();
        for (_, module) in modules_in_order(blueprint) {
            let (position, rotation) = module.transform;
            for mount in &module.sensors {
                let at = frame.point(position + rotation * mount.local_position);
                match mount.sensor_type {
                    SensorType::IMU => {
                        let axes = [Vec3::X, Vec3::Y, Vec3::Z]
                            .map(|axis| frame.direction(rotation * axis) * IMU_CHANNEL_SPREAD);
                        // Linear x, y, z, then angular x, y, z, as the IMU reads.
                        imus.extend(axes.iter().map(|&offset| at + offset));
                        imus.extend(axes.iter().map(|&offset| at - offset));
                    }
                    SensorType::Touch => touches.push(at),
                    SensorType::Ultrasonic => ultrasonics.push(at),
                    SensorType::Camera | SensorType::Lidar => {}
                }
            }
        }

        let is_hinge = |j: &&JointDefinition| matches!(j.joint_type, JointType::Hinge { .. });
        let is_slider = |j: &&JointDefinition| matches!(j.joint_type, JointType::Prismatic { .. });
        let at_joint = |j: &JointDefinition| frame.point(joint_position(blueprint, j));
        let hinges: Vec<Vec3> = blueprint
            .joints
            .iter()
            .filter(is_hinge)
            .map(at_joint)
            .collect();
        let sliders: Vec<Vec3> = blueprint
            .joints
            .iter()
            .filter(is_slider)
            .map(at_joint)
            .collect();

        let mut sensors = imus;
        sensors.extend(touches);
        sensors.extend(ultrasonics);
        sensors.extend(&hinges);
        sensors.extend(&sliders);
        if senses.bearing {
            sensors.extend(bearing_neurons(blueprint, &frame));
        }
        if senses.tilt {
            sensors.extend(tilt_neurons(blueprint, &frame));
        }
        let mut joints = hinges;
        joints.extend(sliders);
        Self { sensors, joints }
    }

    /// Number of brain inputs: the clock, then one per sensor neuron.
    pub fn num_inputs(&self) -> usize {
        NUM_CLOCK_INPUTS + self.sensors.len()
    }
}

/// Where the bearing's sine and cosine neurons sit: beside and ahead of the
/// part that carries the brain (the first module), by its heading (its x
/// axis) over the ground at rest.
fn bearing_neurons(blueprint: &RobotBlueprint, frame: &BodyFrame) -> [Vec3; 2] {
    let (at, heading, side) = root_axes(blueprint, frame);
    [at + side * BEARING_SPREAD, at + heading * BEARING_SPREAD]
}

/// Where the tilt's neurons sit: above the bearing's (forward lean, then
/// sideways lean), so the four get different weights.
fn tilt_neurons(blueprint: &RobotBlueprint, frame: &BodyFrame) -> [Vec3; 2] {
    let (at, heading, side) = root_axes(blueprint, frame);
    let up = Vec3::Y * BEARING_SPREAD;
    [
        at + heading * BEARING_SPREAD + up,
        at + side * BEARING_SPREAD + up,
    ]
}

/// The part that carries the brain (the first module), in the frame: where
/// it sits, its heading (its x axis) over the ground at rest, and the side a
/// quarter turn from the heading toward where the bearing's sine is positive
/// (see `crate::simulation::target::bearing`).
fn root_axes(blueprint: &RobotBlueprint, frame: &BodyFrame) -> (Vec3, Vec3, Vec3) {
    let modules = modules_in_order(blueprint);
    let Some(&(_, root)) = modules.first() else {
        return (Vec3::ZERO, Vec3::X, Vec3::Z);
    };
    let (position, rotation) = root.transform;
    let heading = frame.direction(rotation * Vec3::X);
    let heading = Vec3::new(heading.x, 0.0, heading.z)
        .try_normalize()
        .unwrap_or(Vec3::X);
    let side = Vec3::new(-heading.z, 0.0, heading.x);
    (frame.point(position), heading, side)
}

/// Each driven joint's place across the body, in output order (hinges, then
/// sliders): where the part it moves sits along the root part's sideways
/// axis, from -1 to 1, 1 for the part farthest out that way. A hip counts by its
/// thigh, so a leg's joints share its side. The steering reflex scales each
/// joint's swing by it. All 0 for a body with every part on its midline.
pub fn joint_sides(blueprint: &RobotBlueprint) -> Vec<f32> {
    let frame = BodyFrame::of(blueprint);
    let (root, _, side) = root_axes(blueprint, &frame);
    let is_hinge = |j: &&JointDefinition| matches!(j.joint_type, JointType::Hinge { .. });
    let is_slider = |j: &&JointDefinition| matches!(j.joint_type, JointType::Prismatic { .. });
    let moved = |j: &JointDefinition| {
        blueprint
            .modules
            .get(&j.child_id)
            .map_or(root, |m| frame.point(m.transform.0))
    };
    let hinges = blueprint.joints.iter().filter(is_hinge);
    let sliders = blueprint.joints.iter().filter(is_slider);
    let offsets: Vec<f32> = hinges
        .chain(sliders)
        .map(|joint| (moved(joint) - root).dot(side))
        .collect();
    let reach = offsets.iter().fold(0.0_f32, |reach, o| reach.max(o.abs()));
    if reach < 1e-6 {
        return vec![0.0; offsets.len()];
    }
    offsets.iter().map(|o| o / reach).collect()
}

/// A joint's position in the blueprint's frame: its anchor on the parent
/// part (on the child if the parent is missing).
fn joint_position(blueprint: &RobotBlueprint, joint: &JointDefinition) -> Vec3 {
    let on = |id, anchor: Vec3| {
        blueprint.modules.get(&id).map(|m| {
            let (position, rotation) = m.transform;
            position + rotation * anchor
        })
    };
    on(joint.parent_id, joint.anchor_parent)
        .or_else(|| on(joint.child_id, joint.anchor_child))
        .unwrap_or(Vec3::ZERO)
}

/// Maps blueprint points into the substrate's frame.
struct BodyFrame {
    rotation: Quat,
    centre: Vec3,
    scale: f32,
}

impl BodyFrame {
    fn of(blueprint: &RobotBlueprint) -> Self {
        let rotation = spawn_rotation();
        let aabb = blueprint.aabb(rotation);
        let (min, max) = (Vec3::from(aabb.min), Vec3::from(aabb.max));
        Self {
            rotation,
            centre: (min + max) / 2.0,
            scale: ((max - min) / 2.0).max_element().max(MIN_FRAME_SCALE),
        }
    }

    fn point(&self, blueprint_point: Vec3) -> Vec3 {
        (self.rotation * blueprint_point - self.centre) / self.scale
    }

    fn direction(&self, blueprint_direction: Vec3) -> Vec3 {
        self.rotation * blueprint_direction
    }
}

/// The kind of a candidate link, which decides the CPPN output its weight is
/// read from (see [`CPPN_OUTPUTS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkKind {
    ClockSin,
    ClockCos,
    SensorToInterneuron,
    SensorToMotor,
    InterneuronToMotor,
}

/// A body's brain as symbios-neat paints it: the inputs (the two clock
/// neurons, whose positions no query reads, then the sensor neurons), an
/// interneuron and a motor neuron at each driven joint, and every candidate
/// link with its kind.
///
/// Links are listed per joint for the clock, then per sensor and joint, then
/// per interneuron and joint. The order sets the order in which each neuron
/// sums its inputs, so keeping it keeps a seed's brains bit for bit.
struct NeuronGraph {
    nodes: Vec<SubstrateNode>,
    links: Vec<(usize, usize)>,
    kinds: Vec<LinkKind>,
    inputs: Vec<usize>,
    motors: Vec<usize>,
    first_motor: usize,
}

impl NeuronGraph {
    fn new(substrate: &Substrate) -> Self {
        let node = |at: Vec3, activation| SubstrateNode {
            coords: at.to_array().to_vec(),
            activation,
            bias: 0.0,
        };
        let mut nodes: Vec<SubstrateNode> = (0..NUM_CLOCK_INPUTS)
            .map(|_| node(Vec3::ZERO, Activation::Identity))
            .chain(
                substrate
                    .sensors
                    .iter()
                    .map(|&at| node(at, Activation::Identity)),
            )
            .collect();
        let inputs = (0..nodes.len()).collect();
        let first_interneuron = nodes.len();
        nodes.extend(
            substrate
                .joints
                .iter()
                .map(|&at| node(at, Activation::Tanh)),
        );
        let first_motor = nodes.len();
        nodes.extend(
            substrate
                .joints
                .iter()
                .map(|&at| node(at, Activation::Tanh)),
        );
        let joints = substrate.joints.len();

        let mut links = Vec::new();
        let mut kinds = Vec::new();
        let mut add = |from, to, kind| {
            links.push((from, to));
            kinds.push(kind);
        };
        for joint in 0..joints {
            add(0, first_motor + joint, LinkKind::ClockSin);
            add(1, first_motor + joint, LinkKind::ClockCos);
        }
        for sensor in 0..substrate.sensors.len() {
            let input = NUM_CLOCK_INPUTS + sensor;
            for joint in 0..joints {
                add(
                    input,
                    first_interneuron + joint,
                    LinkKind::SensorToInterneuron,
                );
                add(input, first_motor + joint, LinkKind::SensorToMotor);
            }
        }
        for source in 0..joints {
            for joint in 0..joints {
                let from = first_interneuron + source;
                add(from, first_motor + joint, LinkKind::InterneuronToMotor);
            }
        }
        Self {
            nodes,
            links,
            kinds,
            inputs,
            motors: (first_motor..first_motor + joints).collect(),
            first_motor,
        }
    }

    fn position(&self, node: usize) -> Vec3 {
        Vec3::from_slice(&self.nodes[node].coords)
    }

    /// A query about one neuron itself: its position as both ends, distance
    /// 0.
    fn ask_at(&self, node: usize, query: &mut [f32]) {
        let here = self.position(node).to_array();
        query[..3].copy_from_slice(&here);
        query[3..6].copy_from_slice(&here);
    }
}

impl SubstrateGraph for NeuronGraph {
    fn nodes(&self) -> &[SubstrateNode] {
        &self.nodes
    }
    fn links(&self) -> &[(usize, usize)] {
        &self.links
    }
    fn input_indices(&self) -> &[usize] {
        &self.inputs
    }
    fn output_indices(&self) -> &[usize] {
        &self.motors
    }
}

/// How the CPPN's outputs become the brain: links between placed neurons
/// are asked about with both positions and their distance, and exist where
/// the link-expression output is positive; a joint's rhythm and both its
/// neurons' biases are asked about at the joint itself.
struct BrainPainter;

impl SubstratePainter<NeuronGraph> for BrainPainter {
    fn link_query(&self, graph: &NeuronGraph, link: usize, query: &mut [f32]) {
        let (from, to) = graph.links[link];
        match graph.kinds[link] {
            LinkKind::ClockSin | LinkKind::ClockCos => graph.ask_at(to, query),
            _ => {
                let written = write_endpoints(graph, link, query);
                query[written] = graph.position(from).distance(graph.position(to));
            }
        }
    }

    fn link_weight(&self, graph: &NeuronGraph, link: usize, outputs: &[f32]) -> Option<f32> {
        let output = match graph.kinds[link] {
            // Every joint has a rhythm; a CPPN output of 0 makes it a still one.
            LinkKind::ClockSin => CLOCK_SIN_TO_MOTOR,
            LinkKind::ClockCos => CLOCK_COS_TO_MOTOR,
            _ if outputs[LINK_EXPRESSION] <= 0.0 => return None,
            LinkKind::SensorToInterneuron => SENSOR_TO_INTERNEURON,
            LinkKind::SensorToMotor => SENSOR_TO_MOTOR,
            LinkKind::InterneuronToMotor => INTERNEURON_TO_MOTOR,
        };
        let weight = outputs[output] * MAX_WEIGHT;
        (weight != 0.0).then_some(weight)
    }

    fn bias_query(&self, graph: &NeuronGraph, node: usize, query: &mut [f32]) -> bool {
        graph.ask_at(node, query);
        true
    }

    fn bias(&self, graph: &NeuronGraph, node: usize, outputs: &[f32]) -> f32 {
        let output = if node < graph.first_motor {
            INTERNEURON_BIAS
        } else {
            MOTOR_BIAS
        };
        outputs[output] * MAX_BIAS
    }
}

/// Paints the brain `cppn` describes onto `substrate`: inputs in
/// [`InputLayout`](crate::simulation::sensors::InputLayout) order, one
/// output per driven joint. `cppn` must have [`CPPN_INPUTS`] inputs and
/// [`CPPN_OUTPUTS`] outputs.
pub fn paint_brain(cppn: &CppnEvaluator, substrate: &Substrate) -> FeedforwardNetwork {
    substrate_to_network_with(cppn, &NeuronGraph::new(substrate), &BrainPainter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fixtures::{self, HEXAPOD, INSECT, QUADRUPED};
    use crate::core::presets::MORPHOLOGY_POOL;
    use crate::simulation::lifecycle::derive_blueprint;
    use crate::simulation::sensors::InputLayout;
    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    use symbios_neat::ConnectionGene;

    fn body(source: &str) -> RobotBlueprint {
        derive_blueprint(source, 3, 3)
            .expect("the body derives")
            .blueprint
    }

    /// The body plans and the fixtures: every kind of body the lab has met.
    fn any_body() -> impl Iterator<Item = (usize, &'static str)> {
        MORPHOLOGY_POOL
            .iter()
            .chain(fixtures::ALL)
            .copied()
            .enumerate()
    }

    /// A CPPN with no hidden nodes in which each listed output is
    /// `tanh(bias + distance_weight * distance)` and every other output 0.
    fn cppn_painting(outputs: &[(usize, f32, f32)]) -> CppnEvaluator {
        let mut genome = NeatGenome::minimal(cppn_config());
        let bias_node = genome.bias_id.expect("a CPPN has a bias node");
        let distance = genome.input_ids[DISTANCE];
        for (i, &(output, bias, distance_weight)) in outputs.iter().enumerate() {
            let target = genome.output_ids[output];
            let innovation = 1000 + 2 * i as u64;
            genome
                .connections
                .insert(ConnectionGene::new(innovation, bias_node, target, bias));
            genome.connections.insert(ConnectionGene::new(
                innovation + 1,
                distance,
                target,
                distance_weight,
            ));
        }
        CppnEvaluator::new(&genome).expect("acyclic")
    }

    #[test]
    fn every_body_gets_a_neuron_per_input_and_joint() {
        for (i, source) in any_body() {
            let derived = derive_blueprint(source, 3, 3).expect("the body derives");
            let substrate = Substrate::of_blueprint(&derived.blueprint);
            assert_eq!(
                substrate.num_inputs(),
                InputLayout::of_blueprint(&derived.blueprint).total(),
                "body {i}"
            );
            assert_eq!(substrate.joints.len(), derived.num_outputs, "body {i}");
        }
    }

    /// The target task adds the bearing's two neurons after the sensors',
    /// apart from each other and by the body's root part, so a fresh CPPN
    /// paints them a brain of the steering layout's size.
    #[test]
    fn steering_adds_the_bearing_neurons_apart() {
        let mut rng = Pcg64::seed_from_u64(9);
        let cppn = CppnEvaluator::new(&fresh_cppn(&mut rng)).expect("acyclic");
        for (i, source) in any_body() {
            let blueprint = body(source);
            let forward = Substrate::of_blueprint(&blueprint);
            let steering = Substrate::of_body(&blueprint, Senses::BEARING);
            assert_eq!(
                steering.sensors[..forward.sensors.len()],
                forward.sensors[..]
            );
            let [sin, cos] = [
                steering.sensors[forward.sensors.len()],
                steering.sensors[forward.sensors.len() + 1],
            ];
            assert!((sin - cos).length() > 0.1, "body {i}: {sin} {cos}");
            let layout = InputLayout::of_blueprint(&blueprint).with_senses(Senses::BEARING);
            assert_eq!(steering.num_inputs(), layout.total(), "body {i}");
            assert_eq!(paint_brain(&cppn, &steering).num_inputs(), layout.total());
        }
    }

    /// Legs on either side of a body read opposite sides, in [-1, 1]. The
    /// quadruped stands in a "+" (front and back legs on the midline) and
    /// grows legs of random length, so only its side legs read sides.
    #[test]
    fn joints_know_which_side_of_the_body_they_are_on() {
        for (i, source) in [QUADRUPED, HEXAPOD].into_iter().enumerate() {
            let blueprint = body(source);
            let sides = joint_sides(&blueprint);
            assert_eq!(
                sides.len(),
                Substrate::of_blueprint(&blueprint).joints.len()
            );
            assert!(sides.iter().all(|s| (-1.0..=1.0).contains(s)), "body {i}");
            let left = sides.iter().filter(|&&s| s > 0.2).count();
            let right = sides.iter().filter(|&&s| s < -0.2).count();
            assert!(left > 0 && right > 0, "body {i}: {left} vs {right}");
            assert!(sides.iter().any(|&s| (s.abs() - 1.0).abs() < 1e-6));
        }
        // The hexapod's three legs a side split evenly, every joint (hips
        // too) taking its leg's side.
        let hexapod = joint_sides(&body(HEXAPOD));
        let left = hexapod.iter().filter(|&&s| s > 0.0).count();
        let right = hexapod.iter().filter(|&&s| s < 0.0).count();
        assert_eq!((left, right), (9, 9), "{hexapod:?}");
    }

    /// The tilt adds two neurons after the bearing's, apart from all of
    /// them, and a CPPN paints a brain of the layout's size.
    #[test]
    fn the_tilt_adds_two_neurons_of_its_own() {
        let mut rng = Pcg64::seed_from_u64(10);
        let cppn = CppnEvaluator::new(&fresh_cppn(&mut rng)).expect("acyclic");
        let blueprint = body(HEXAPOD);
        let senses = Senses {
            bearing: true,
            tilt: true,
        };
        let bearing = Substrate::of_body(&blueprint, Senses::BEARING);
        let both = Substrate::of_body(&blueprint, senses);
        assert_eq!(both.sensors.len(), bearing.sensors.len() + 2);
        let added = &both.sensors[bearing.sensors.len() - 2..];
        for (i, a) in added.iter().enumerate() {
            for b in &added[i + 1..] {
                assert!((*a - *b).length() > 0.1, "{a} {b}");
            }
        }
        let layout = InputLayout::of_blueprint(&blueprint).with_senses(senses);
        assert_eq!(paint_brain(&cppn, &both).num_inputs(), layout.total());
    }

    #[test]
    fn neurons_sit_within_the_normalized_body() {
        let reach = 1.0 + IMU_CHANNEL_SPREAD + 1e-4;
        for (i, source) in any_body() {
            let substrate = Substrate::of_blueprint(&body(source));
            for p in substrate.sensors.iter().chain(&substrate.joints) {
                assert!(p.abs().max_element() <= reach, "body {i}: {p}");
            }
        }
    }

    #[test]
    fn a_fresh_cppn_paints_a_brain_shaped_for_any_body() {
        let mut rng = Pcg64::seed_from_u64(4);
        let cppn = CppnEvaluator::new(&fresh_cppn(&mut rng)).expect("acyclic");
        for (_, source) in any_body() {
            let substrate = Substrate::of_blueprint(&body(source));
            let network = paint_brain(&cppn, &substrate);
            assert_eq!(network.num_inputs(), substrate.num_inputs());
            assert_eq!(network.num_outputs(), substrate.joints.len());
            let outputs = network.evaluate(&vec![0.5; substrate.num_inputs()]);
            assert!(outputs.iter().all(|o| o.is_finite() && o.abs() <= 1.0));
        }
    }

    #[test]
    fn a_silent_cppn_paints_a_resting_brain() {
        let cppn = CppnEvaluator::new(&NeatGenome::minimal(cppn_config())).expect("acyclic");
        let substrate = Substrate::of_blueprint(&body(HEXAPOD));
        let network = paint_brain(&cppn, &substrate);
        let outputs = network.evaluate(&vec![1.0; substrate.num_inputs()]);
        assert!(outputs.iter().all(|&o| o == 0.0), "{outputs:?}");
    }

    #[test]
    fn the_clock_drives_every_joint_the_cppn_gives_a_rhythm() {
        // Every joint's sine weight MAX_WEIGHT * tanh(5), nothing else, so
        // every motor neuron reads tanh(weight * sine).
        let cppn = cppn_painting(&[(CLOCK_SIN_TO_MOTOR, 5.0, 0.0)]);
        let substrate = Substrate::of_blueprint(&body(QUADRUPED));
        let network = paint_brain(&cppn, &substrate);
        let weight = MAX_WEIGHT * 5.0f32.tanh();
        let mut inputs = vec![0.0; substrate.num_inputs()];
        for sine in [1.0, 0.5, -1.0] {
            inputs[0] = sine;
            let expected = (weight * sine).tanh();
            let outputs = network.evaluate(&inputs);
            assert!(
                outputs.iter().all(|&o| (o - expected).abs() < 1e-5),
                "sine {sine}: {outputs:?}, expected {expected}"
            );
        }
    }

    #[test]
    fn a_cppn_can_wire_each_hinge_angle_to_its_own_motor() {
        // Every link expressed, and sensor-to-motor weights falling from
        // strongly positive at distance 0 to negative beyond 0.25, so each
        // hinge's angle neuron (at the hinge) excites the motor neuron at the
        // same hinge most.
        let cppn = cppn_painting(&[(LINK_EXPRESSION, 5.0, 0.0), (SENSOR_TO_MOTOR, 1.5, -6.0)]);
        let blueprint = body(HEXAPOD);
        let substrate = Substrate::of_blueprint(&blueprint);
        let layout = InputLayout::of_blueprint(&blueprint);
        let network = paint_brain(&cppn, &substrate);
        // At distance 0 the weight is MAX_WEIGHT * tanh(1.5).
        let own_expected = (MAX_WEIGHT * 1.5f32.tanh()).tanh();
        for hinge in 0..layout.hinges {
            let mut inputs = vec![0.0; substrate.num_inputs()];
            inputs[layout.joint_offset(hinge)] = 1.0;
            let outputs = network.evaluate(&inputs);
            let own = outputs[hinge];
            assert!(
                (own - own_expected).abs() < 1e-5,
                "hinge {hinge}: own motor {own}, expected {own_expected}"
            );
            for (other, &o) in outputs.iter().enumerate() {
                let same_place = substrate.joints[other].distance(substrate.joints[hinge]) < 1e-4;
                assert!(
                    same_place || o < own,
                    "hinge {hinge}: motor {other} {o} vs {own}"
                );
            }
        }
    }

    #[test]
    fn unexpressed_links_carry_nothing_however_strong() {
        let cppn = cppn_painting(&[(LINK_EXPRESSION, -5.0, 0.0), (SENSOR_TO_MOTOR, 5.0, 0.0)]);
        let substrate = Substrate::of_blueprint(&body(QUADRUPED));
        let network = paint_brain(&cppn, &substrate);
        let outputs = network.evaluate(&vec![1.0; substrate.num_inputs()]);
        assert!(outputs.iter().all(|&o| o == 0.0), "{outputs:?}");
    }

    #[test]
    fn a_fresh_cppn_links_only_neurons_near_each_other() {
        // A sensor reaches a motor directly when they lie within the seed's
        // radius, or through an interneuron within it of both; beyond twice
        // the radius it cannot reach it at all.
        let reach = 2.0 * LOCALITY_SEED_BIAS / LOCALITY_SEED_SLOPE;
        let mut rng = Pcg64::seed_from_u64(9);
        let cppn = CppnEvaluator::new(&fresh_cppn(&mut rng)).expect("acyclic");
        let substrate = Substrate::of_blueprint(&body(INSECT));
        let network = paint_brain(&cppn, &substrate);
        let resting = network.evaluate(&vec![0.0; substrate.num_inputs()]);
        let mut far_pairs = 0;
        for (sensor, &at) in substrate.sensors.iter().enumerate() {
            let mut inputs = vec![0.0; substrate.num_inputs()];
            inputs[NUM_CLOCK_INPUTS + sensor] = 1.0;
            let outputs = network.evaluate(&inputs);
            for (joint, &joint_at) in substrate.joints.iter().enumerate() {
                if at.distance(joint_at) > reach + 1e-4 {
                    far_pairs += 1;
                    assert_eq!(outputs[joint], resting[joint], "sensor {sensor} -> {joint}");
                }
            }
        }
        assert!(
            far_pairs > 0,
            "the insect should have distant sensor-joint pairs"
        );
    }
}
