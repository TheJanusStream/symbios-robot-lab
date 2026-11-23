//! What quality-diversity search spreads robots over: four traits of a robot,
//! two of how it walked and two of its body, and the pairs of them an archive
//! can be laid out by ([`QdDescriptor`]).
//!
//! A phenotype's descriptor holds all four traits, so an archive can be laid
//! out again by another pair without evaluating anyone twice.

use bevy::math::Isometry3d;
use bevy::prelude::*;
use bevy::shape::Bounded3d;
use symbios_robot::{RobotBlueprint, SensorType};

/// Traits a phenotype's descriptor holds, each scaled to `[0, 1]`, in the
/// order of the indices below.
pub const NUM_TRAITS: usize = 4;
/// The share of the time the robot's feet were on the ground.
pub const DUTY_FACTOR: usize = 0;
/// How fast the brain moved the joints: the mean commanded joint speed,
/// scaled.
pub const ACTION_INTENSITY: usize = 1;
/// How many feet (touch sensors) the body has, over [`FEET_SCALE`].
pub const FEET: usize = 2;
/// The body's longest horizontal extent at rest, over [`LENGTH_SCALE`].
pub const LENGTH: usize = 3;

/// A body with this many feet or more sits at the top of the feet axis. Each
/// count below it falls in its own column of the MAP-Elites grid.
pub const FEET_SCALE: f32 = 12.0;
/// A body this long (m) or longer sits at the top of the length axis. The
/// body plans start at 3.3 to 3.6 m and evolution lengthens them.
pub const LENGTH_SCALE: f32 = 6.0;

/// Which two traits the quality-diversity algorithms spread robots over: the
/// axes of the MAP-Elites grid and the CVT centroids, and the space novelty
/// and dominated novelty are measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QdDescriptor {
    /// How the robot walks: duty factor and action intensity.
    #[default]
    Gait,
    /// What the robot is: feet and length.
    Body,
    /// How many feet, and how much of the time they stand: feet and duty
    /// factor.
    BodyAndGait,
}

impl QdDescriptor {
    pub const ALL: [QdDescriptor; 3] = [
        QdDescriptor::Gait,
        QdDescriptor::Body,
        QdDescriptor::BodyAndGait,
    ];

    pub fn label(self) -> &'static str {
        match self {
            QdDescriptor::Gait => "Gait",
            QdDescriptor::Body => "Body",
            QdDescriptor::BodyAndGait => "Body and gait",
        }
    }

    /// Parses the short names the headless CLI accepts (`gait`, `body`,
    /// `body-gait`).
    pub fn from_cli_name(name: &str) -> Option<Self> {
        match name {
            "gait" => Some(QdDescriptor::Gait),
            "body" => Some(QdDescriptor::Body),
            "body-gait" => Some(QdDescriptor::BodyAndGait),
            _ => None,
        }
    }

    /// The traits on the x and y axes.
    pub fn axes(self) -> [usize; 2] {
        match self {
            QdDescriptor::Gait => [DUTY_FACTOR, ACTION_INTENSITY],
            QdDescriptor::Body => [FEET, LENGTH],
            QdDescriptor::BodyAndGait => [FEET, DUTY_FACTOR],
        }
    }

    /// The names of the x and y axes.
    pub fn axis_names(self) -> [&'static str; 2] {
        self.axes().map(|axis| match axis {
            DUTY_FACTOR => "duty factor",
            ACTION_INTENSITY => "action intensity",
            FEET => "feet",
            _ => "length",
        })
    }

    /// A descriptor's point in this space. A trait the descriptor lacks
    /// reads as 0.
    pub fn project(self, descriptor: &[f32]) -> [f32; 2] {
        self.axes()
            .map(|axis| descriptor.get(axis).copied().unwrap_or(0.0))
    }
}

impl std::fmt::Display for QdDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A body's traits, unscaled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTraits {
    /// Touch sensors: the lab puts one on every foot.
    pub feet: usize,
    /// The longest horizontal extent of the parts at rest (m).
    pub length: f32,
    /// The edge (m) of the cube that holds as much as the box around the
    /// parts at rest: how big the body is, whatever its shape.
    pub size: f32,
}

impl BodyTraits {
    pub fn of(blueprint: &RobotBlueprint) -> Self {
        let feet = blueprint
            .modules
            .values()
            .flat_map(|module| &module.sensors)
            .filter(|sensor| sensor.sensor_type == SensorType::Touch)
            .count();
        let mut bounds: Option<(Vec3, Vec3)> = None;
        for module in blueprint.modules.values() {
            let (position, rotation) = module.transform;
            let aabb = module
                .shape
                .to_bevy_primitive()
                .aabb_3d(Isometry3d::new(position, rotation));
            let (min, max) = (Vec3::from(aabb.min), Vec3::from(aabb.max));
            bounds = Some(match bounds {
                Some((lo, hi)) => (lo.min(min), hi.max(max)),
                None => (min, max),
            });
        }
        let length = bounds.map_or(0.0, |(lo, hi)| (hi.x - lo.x).max(hi.z - lo.z));
        let size = bounds.map_or(0.0, |(lo, hi)| {
            let extent = hi - lo;
            (extent.x * extent.y * extent.z).max(0.0).cbrt()
        });
        Self { feet, length, size }
    }

    /// The feet and length traits, scaled to `[0, 1]`.
    pub fn scaled(self) -> [f32; 2] {
        [
            (self.feet as f32 / FEET_SCALE).min(1.0),
            (self.length / LENGTH_SCALE).clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_space_projects_its_two_traits() {
        let descriptor = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(QdDescriptor::Gait.project(&descriptor), [0.1, 0.2]);
        assert_eq!(QdDescriptor::Body.project(&descriptor), [0.3, 0.4]);
        assert_eq!(QdDescriptor::BodyAndGait.project(&descriptor), [0.3, 0.1]);
        // A descriptor without body traits reads them as 0.
        assert_eq!(QdDescriptor::Body.project(&[0.5, 0.5]), [0.0, 0.0]);
    }

    #[test]
    fn every_space_has_a_cli_name_and_named_axes() {
        for (name, space) in ["gait", "body", "body-gait"].iter().zip(QdDescriptor::ALL) {
            assert_eq!(QdDescriptor::from_cli_name(name), Some(space));
            assert!(space.axis_names().iter().all(|axis| !axis.is_empty()));
        }
        assert_eq!(QdDescriptor::from_cli_name("legs"), None);
        assert_eq!(QdDescriptor::Body.axis_names(), ["feet", "length"]);
    }

    #[test]
    fn each_foot_count_below_the_scale_has_its_own_grid_column() {
        let column = |feet: usize| {
            let x = BodyTraits {
                feet,
                length: 1.0,
                size: 1.0,
            }
            .scaled()[0];
            (x * 20.0).floor().min(19.0) as usize
        };
        let columns: Vec<usize> = (0..=12).map(column).collect();
        assert!(columns.windows(2).all(|w| w[0] < w[1]), "{columns:?}");
        assert_eq!(column(30), 19);
    }
}
