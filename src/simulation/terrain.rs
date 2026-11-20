//! The ground robots walk on: the flat arena floor, or a patch of rough or
//! hilly ground laid over it, generated afresh each generation from the run's
//! seed and the generation. Every generation walks new ground, so no gait
//! wins by fitting one patch, and runs with the same seed walk the same
//! grounds. The windowed app draws the patch from the same height map.

use crate::core::config::{LabConfig, Terrain};
use crate::core::population::Population;
use crate::simulation::clock::GenerationClock;
use crate::simulation::collision::arena_layers;
use crate::simulation::rng::{LabRng, generation_seed};
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_symbios_ground::{FbmNoise, HeightMap, TerrainGenerator, build_heightfield_collider};

/// Samples along each side of the patch, one a metre: a 192 m square centred
/// on the origin, wider than the spawn grid of the largest population (100
/// robots, 100 m) plus the furthest walks.
const PATCH_SAMPLES: usize = 193;
/// Metres between samples.
const PATCH_SCALE: f32 = 1.0;
/// Width (m) of the band along the patch's edge over which it eases down to
/// the floor.
const EDGE_FADE: f32 = 16.0;

/// The patch's side (m), from its first sample to its last.
fn patch_extent() -> f32 {
    (PATCH_SAMPLES - 1) as f32 * PATCH_SCALE
}

/// How a terrain's patch is generated.
struct Relief {
    /// Noise cells across the patch: its features are about
    /// `patch_extent / cells` metres across.
    cells: f32,
    /// Octaves of noise, each twice as fine and `persistence` as tall.
    octaves: u32,
    persistence: f32,
    /// The highest the patch rises (m).
    height: f32,
}

fn relief(terrain: Terrain) -> Option<Relief> {
    match terrain {
        Terrain::Flat => None,
        // Bumps about 2 m across, with finer unevenness on them.
        Terrain::Rough => Some(Relief {
            cells: 96.0,
            octaves: 3,
            persistence: 0.5,
            height: 0.2,
        }),
        // Hills about 40 m across, with gentler rises on their flanks.
        Terrain::Hilly => Some(Relief {
            cells: 5.0,
            octaves: 4,
            persistence: 0.4,
            height: 3.0,
        }),
    }
}

/// The seed of a generation's ground.
pub fn ground_seed(run_seed: u64, generation: usize) -> u64 {
    generation_seed(run_seed, generation, 0)
}

/// The ground under this generation's robots.
#[derive(Resource, Default)]
pub struct Ground {
    /// The patch's heights (m above the floor); `None` on flat ground.
    map: Option<HeightMap>,
    /// The terrain and seed the patch was generated from.
    built: Option<(Terrain, u64)>,
}

/// Marker for the patch's collider.
#[derive(Component)]
pub struct GroundPatch;

impl Ground {
    /// A patch of `terrain` generated from `seed`; flat ground for
    /// [`Terrain::Flat`].
    pub fn generate(terrain: Terrain, seed: u64) -> Self {
        let Some(relief) = relief(terrain) else {
            return Self::default();
        };
        let mut map = HeightMap::new(PATCH_SAMPLES, PATCH_SAMPLES, PATCH_SCALE);
        FbmNoise {
            seed,
            octaves: relief.octaves,
            persistence: relief.persistence,
            lacunarity: 2.0,
            base_frequency: relief.cells,
        }
        .generate(&mut map);
        map.normalize();
        let extent = patch_extent();
        for z in 0..PATCH_SAMPLES {
            for x in 0..PATCH_SAMPLES {
                let (u, v) = (x as f32 * PATCH_SCALE, z as f32 * PATCH_SCALE);
                let edge = u.min(v).min(extent - u).min(extent - v);
                let t = (edge / EDGE_FADE).clamp(0.0, 1.0);
                let fade = t * t * (3.0 - 2.0 * t);
                let height = map.get(x, z) * relief.height * fade;
                map.set(x, z, height);
            }
        }
        Self {
            map: Some(map),
            built: Some((terrain, seed)),
        }
    }

    /// The ground's height (m) at world `(x, z)`: the patch's where it lies,
    /// the floor's (0) elsewhere.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let Some(map) = &self.map else {
            return 0.0;
        };
        let half = patch_extent() / 2.0;
        let (u, v) = (x + half, z + half);
        let on_patch = |w: f32| (0.0..=patch_extent()).contains(&w);
        if on_patch(u) && on_patch(v) {
            map.get_height_at(u, v)
        } else {
            0.0
        }
    }

    /// The highest ground (m) under the rectangle from `min` to `max` (world
    /// x and z), sampled at half the patch's spacing.
    pub fn highest_under(&self, min: Vec2, max: Vec2) -> f32 {
        if self.map.is_none() {
            return 0.0;
        }
        let step = PATCH_SCALE / 2.0;
        let steps = |span: f32| (span / step).ceil().max(1.0) as usize;
        let (nx, nz) = (steps(max.x - min.x), steps(max.y - min.y));
        let mut highest = 0.0_f32;
        for i in 0..=nx {
            for j in 0..=nz {
                let x = min.x + (max.x - min.x) * i as f32 / nx as f32;
                let z = min.y + (max.y - min.y) * j as f32 / nz as f32;
                highest = highest.max(self.height_at(x, z));
            }
        }
        highest
    }

    /// The patch's height map and the world position of its first sample,
    /// where its mesh starts; `None` on flat ground.
    pub fn patch(&self) -> Option<(&HeightMap, Vec3)> {
        let half = patch_extent() / 2.0;
        self.map
            .as_ref()
            .map(|map| (map, Vec3::new(-half, 0.0, -half)))
    }

    /// The terrain and seed the patch was generated from; `None` on flat
    /// ground.
    pub fn built(&self) -> Option<(Terrain, u64)> {
        self.built
    }
}

/// Lays the ground for the generation about to spawn (the clock stops
/// between generations): a new patch each generation on rough or hilly
/// terrain, none on flat. A terrain chosen mid-generation waits for the next.
pub fn lay_ground(
    mut commands: Commands,
    config: Res<LabConfig>,
    population: Res<Population>,
    clock: Res<GenerationClock>,
    rng: Res<LabRng>,
    mut ground: ResMut<Ground>,
    patches: Query<Entity, With<GroundPatch>>,
) {
    if clock.is_running() || population.agents.is_empty() {
        return;
    }
    let wanted = relief(config.terrain).is_some().then(|| {
        (
            config.terrain,
            ground_seed(rng.seed(), population.generation),
        )
    });
    if ground.built == wanted {
        return;
    }
    for patch in &patches {
        commands.entity(patch).despawn();
    }
    *ground = match wanted {
        Some((terrain, seed)) => Ground::generate(terrain, seed),
        None => Ground::default(),
    };
    if let Some((map, _)) = ground.patch() {
        // The collider is centred on its entity, the patch on the origin.
        commands.spawn((
            GroundPatch,
            Name::new("Ground patch"),
            Transform::IDENTITY,
            RigidBody::Static,
            build_heightfield_collider(map),
            arena_layers(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_ground_is_the_floor() {
        let ground = Ground::generate(Terrain::Flat, 1);
        assert!(ground.patch().is_none());
        assert_eq!(ground.height_at(3.0, -7.0), 0.0);
        assert_eq!(
            ground.highest_under(Vec2::splat(-5.0), Vec2::splat(5.0)),
            0.0
        );
    }

    /// The patch keeps within its terrain's height, rises somewhere in the
    /// middle and eases down to the floor at its edge.
    #[test]
    fn a_patch_stays_within_its_height_and_meets_the_floor() {
        for (terrain, height) in [(Terrain::Rough, 0.2), (Terrain::Hilly, 3.0)] {
            let ground = Ground::generate(terrain, 7);
            let (map, origin) = ground.patch().expect("a patch");
            assert_eq!(origin, Vec3::new(-96.0, 0.0, -96.0));
            let highest = map.data().iter().copied().fold(f32::MIN, f32::max);
            assert!(
                highest > 0.3 * height && highest <= height + 1e-5,
                "{terrain}: highest {highest}"
            );
            assert!(map.data().iter().all(|&h| h >= 0.0));
            for (x, z) in [(-96.0, 0.0), (96.0, 10.0), (0.0, -96.0), (200.0, 0.0)] {
                assert!(
                    ground.height_at(x, z).abs() < 1e-5,
                    "{terrain} at ({x}, {z})"
                );
            }
            let middle = ground.highest_under(Vec2::splat(-10.0), Vec2::splat(10.0));
            assert!(middle > 0.0, "{terrain}: flat in the middle");
            assert!(middle >= ground.height_at(0.0, 0.0));
        }
    }

    #[test]
    fn each_generation_walks_new_ground_and_a_seed_repeats_it() {
        let seeds: Vec<u64> = (0..50).map(|g| ground_seed(42, g)).collect();
        let mut distinct = seeds.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), seeds.len());
        assert_ne!(ground_seed(42, 0), ground_seed(43, 0));
        let a = Ground::generate(Terrain::Rough, ground_seed(42, 3));
        let b = Ground::generate(Terrain::Rough, ground_seed(42, 3));
        assert_eq!(
            a.patch().map(|(m, _)| m.data()),
            b.patch().map(|(m, _)| m.data())
        );
    }
}
