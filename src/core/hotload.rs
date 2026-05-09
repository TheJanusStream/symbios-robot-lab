//! Hot-reload of `.lsys` morphology files via `bevy_symbios::loader`.
//!
//! Drop a `.lsys` file into `assets/morphologies/` and it shows up in the
//! evolution pool on the next generation. Edit it in your editor and Bevy's
//! `file_watcher` triggers an `AssetEvent::Modified`, which refreshes the
//! cached source string in [`RuntimeMorphologyPool`].
//!
//! The runtime pool augments the static [`crate::core::presets::MORPHOLOGY_POOL`]
//! at parent-pool fresh-injection time (see `tick_generation`).

use bevy::asset::LoadedFolder;
use bevy::prelude::*;
use bevy_symbios::loader::LSystemSource;

/// Currently-loaded morphology source strings, keyed by their asset path stem.
/// Lookups in `tick_generation` go through [`Self::sources`], which returns a
/// snapshot suitable for random sampling.
#[derive(Resource, Default)]
pub struct RuntimeMorphologyPool {
    /// (asset_path, raw_source) pairs. Source is the original `.lsys` text,
    /// re-derived from the parsed `LSystemSource` via `to_source()`.
    entries: Vec<(String, String)>,
}

impl RuntimeMorphologyPool {
    pub fn sources(&self) -> &[(String, String)] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)] // Pairs with `len` for the `len_without_is_empty` clippy lint.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Tracks the load state of `assets/morphologies/`. The handle is held only to
/// keep the folder asset alive — accessing it would force-load synchronously.
#[derive(Resource)]
pub struct MorphologyFolderHandle(#[allow(dead_code)] pub Handle<LoadedFolder>);

/// Loads the `morphologies/` folder at startup. Bevy's asset server walks the
/// folder, kicks off `LSystemAssetLoader` for every `.lsys` it finds, and
/// emits `AssetEvent::Added` for each one as they finish loading.
pub fn load_morphologies_folder(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load_folder("morphologies");
    commands.insert_resource(MorphologyFolderHandle(handle));
}

/// Refreshes the runtime pool whenever a `LSystemSource` asset is added or
/// modified. Removed assets drop out of the pool. Source paths are best-effort
/// (Bevy's asset events carry an `AssetId`, not a path), so we resolve via the
/// asset server.
pub fn refresh_runtime_pool(
    mut events: MessageReader<AssetEvent<LSystemSource>>,
    sources: Res<Assets<LSystemSource>>,
    asset_server: Res<AssetServer>,
    mut pool: ResMut<RuntimeMorphologyPool>,
) {
    let mut dirty = false;
    for event in events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                let Some(asset) = sources.get(*id) else {
                    continue;
                };
                let source = asset.0.to_source();
                let path = asset_server
                    .get_path(*id)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| format!("asset_{:?}", id));
                if let Some(slot) = pool.entries.iter_mut().find(|(p, _)| p == &path) {
                    slot.1 = source;
                } else {
                    pool.entries.push((path, source));
                }
                dirty = true;
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                let Some(path) = asset_server.get_path(*id) else {
                    continue;
                };
                let path = path.to_string();
                pool.entries.retain(|(p, _)| p != &path);
                dirty = true;
            }
        }
    }
    if dirty {
        info!("Runtime morphology pool now has {} entries", pool.len());
    }
}
