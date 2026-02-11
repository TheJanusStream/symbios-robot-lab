use bevy::prelude::*;

#[derive(Resource)]
pub struct LabConfig {
    pub population_size: usize,
    pub generation_duration: f32, // Seconds per round
    pub time_scale: f32,
    pub auto_restart: bool,
}

impl Default for LabConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            generation_duration: 15.0,
            time_scale: 1.0,
            auto_restart: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = LabConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.generation_duration, 15.0);
        assert_eq!(config.time_scale, 1.0);
        assert!(!config.auto_restart);
    }
}
