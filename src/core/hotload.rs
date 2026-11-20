//! Hot-reload of `.lsys` morphology files via `bevy_symbios::loader`
//! (native builds only: a browser cannot list an asset folder).
//!
//! Drop a `.lsys` file into `assets/morphologies/` and it shows up in the
//! evolution pool on the next generation. Edit it in your editor and Bevy's
//! `file_watcher` triggers an `AssetEvent::Modified`, which refreshes the
//! cached source string in [`RuntimeMorphologyPool`].

use crate::core::morphology_pool::RuntimeMorphologyPool;
use bevy::asset::LoadedFolder;
use bevy::prelude::*;
use bevy_symbios::loader::{LSystemAssetPlugin, LSystemSource};

/// Loads `assets/morphologies/` and keeps [`RuntimeMorphologyPool`] in step
/// with the files in it.
pub struct MorphologyHotloadPlugin;

impl Plugin for MorphologyHotloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(LSystemAssetPlugin)
            .init_resource::<RuntimeMorphologyPool>()
            .add_systems(Startup, load_morphologies_folder)
            .add_systems(Update, refresh_runtime_pool);
    }
}

/// Keeps the folder asset alive; reading the handle would force a load.
#[derive(Resource)]
pub struct MorphologyFolderHandle(#[allow(dead_code)] pub Handle<LoadedFolder>);

/// Bevy's asset server walks the folder, runs the `.lsys` loader for every
/// file it finds, and emits an `AssetEvent` for each one as it loads.
pub fn load_morphologies_folder(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load_folder("morphologies");
    commands.insert_resource(MorphologyFolderHandle(handle));
}

/// Refreshes the runtime pool whenever an `LSystemSource` asset is added,
/// modified or removed. Asset events carry an id, not a path, so the path is
/// resolved through the asset server.
pub fn refresh_runtime_pool(
    mut events: MessageReader<AssetEvent<LSystemSource>>,
    sources: Res<Assets<LSystemSource>>,
    asset_server: Res<AssetServer>,
    mut pool: ResMut<RuntimeMorphologyPool>,
) {
    let mut changed = false;
    for event in events.read() {
        match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                let Some(asset) = sources.get(*id) else {
                    continue;
                };
                let path = asset_server
                    .get_path(*id)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| format!("asset_{id:?}"));
                pool.upsert(path, asset.0.to_source());
                changed = true;
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                if let Some(path) = asset_server.get_path(*id) {
                    pool.remove(&path.to_string());
                    changed = true;
                }
            }
            AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
    if changed {
        info!("Runtime morphology pool now has {} entries", pool.len());
    }
}
