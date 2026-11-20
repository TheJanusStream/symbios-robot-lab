//! Morphologies loaded at runtime, which join the static
//! [`crate::core::presets::MORPHOLOGY_POOL`] when fresh bodies are injected.
//!
//! The pool is plain data so the simulation can use it on every target; only
//! the native app fills it, from `.lsys` files (see `core::hotload`).

use bevy::prelude::*;

/// Currently loaded morphology sources, keyed by their asset path.
#[derive(Resource, Default)]
pub struct RuntimeMorphologyPool {
    /// `(asset_path, source)` pairs.
    entries: Vec<(String, String)>,
}

impl RuntimeMorphologyPool {
    pub fn sources(&self) -> &[(String, String)] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds the source at `path`, or replaces it if that path is loaded.
    pub fn upsert(&mut self, path: String, source: String) {
        match self.entries.iter_mut().find(|(p, _)| *p == path) {
            Some(slot) => slot.1 = source,
            None => self.entries.push((path, source)),
        }
    }

    /// Drops the source loaded from `path`, if any.
    pub fn remove(&mut self, path: &str) {
        self.entries.retain(|(p, _)| p != path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_replaces_a_path_already_loaded() {
        let mut pool = RuntimeMorphologyPool::default();
        pool.upsert("a.lsys".into(), "omega: B(1)".into());
        pool.upsert("a.lsys".into(), "omega: B(2)".into());
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.sources()[0].1, "omega: B(2)");
    }

    #[test]
    fn remove_drops_only_that_path() {
        let mut pool = RuntimeMorphologyPool::default();
        pool.upsert("a.lsys".into(), "omega: B(1)".into());
        pool.upsert("b.lsys".into(), "omega: B(2)".into());
        pool.remove("a.lsys");
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.sources()[0].0, "b.lsys");
    }
}
