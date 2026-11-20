//! The hall of fame window: the session's champions, to watch walking on
//! their own, copy or save as JSON, or send back into evolution, and robots
//! to import from JSON (pasted, or on the desktop dropped onto the window).

use crate::visuals::ui::CameraTarget;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use symbios_robot_lab::simulation::hall_of_fame::{
    Champion, HallOfFame, Immigrants, Showcase, ShowcaseRequest,
};
use symbios_robot_lab::simulation::spawn::SHOWCASE_INDEX;

/// What the user typed into the import box, and what the last import said.
#[derive(Default)]
pub struct ImportBox {
    text: String,
    status: Option<String>,
}

enum Action {
    Watch(usize),
    Breed(usize),
    Remove(usize),
}

pub fn render_hall_of_fame(
    mut contexts: EguiContexts,
    mut hall: ResMut<HallOfFame>,
    mut immigrants: ResMut<Immigrants>,
    mut showcase: MessageWriter<ShowcaseRequest>,
    mut camera: ResMut<CameraTarget>,
    showcases: Query<(), With<Showcase>>,
    mut import: Local<ImportBox>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Hall of fame")
        .default_open(false)
        .default_pos([1240.0, 20.0])
        .show(ctx, |ui| {
            if hall.champions().is_empty() {
                ui.label("The best robot of each generation lands here.");
            }
            let mut action = None;
            egui::Grid::new("hall_grid")
                .num_columns(5)
                .spacing([6.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    for (i, champion) in hall.champions().iter().enumerate() {
                        ui.label(champion.label());
                        if ui
                            .button("Watch")
                            .on_hover_text("It walks on its own in front of the population")
                            .clicked()
                        {
                            action = Some(Action::Watch(i));
                        }
                        if ui
                            .button("Copy")
                            .on_hover_text("Copy its JSON to paste elsewhere")
                            .clicked()
                        {
                            ui.ctx().copy_text(champion.to_json());
                        }
                        if ui.button("Save").on_hover_text("Save its JSON").clicked() {
                            save(champion);
                        }
                        if ui
                            .button("Breed")
                            .on_hover_text("Send it into the next generation")
                            .clicked()
                        {
                            action = Some(Action::Breed(i));
                        }
                        if ui.small_button("x").on_hover_text("Remove").clicked() {
                            action = Some(Action::Remove(i));
                        }
                        ui.end_row();
                    }
                });
            match action {
                Some(Action::Watch(i)) => {
                    showcase.write(ShowcaseRequest(Some(hall.champions()[i].genotype.clone())));
                    camera.agent_index = Some(SHOWCASE_INDEX);
                }
                Some(Action::Breed(i)) => {
                    immigrants.0.push(hall.champions()[i].genotype.clone());
                    import.status = Some("It joins the next generation".to_string());
                }
                Some(Action::Remove(i)) => hall.remove(i),
                None => {}
            }
            ui.horizontal(|ui| {
                if !showcases.is_empty() && ui.button("Stop watching").clicked() {
                    showcase.write(ShowcaseRequest(None));
                    camera.agent_index = None;
                }
                if !hall.champions().is_empty() && ui.button("Clear the hall").clicked() {
                    hall.clear();
                }
            });

            ui.separator();
            ui.label("Import a robot: paste its JSON (or drop a .json file on the window)");
            ui.add(
                egui::TextEdit::multiline(&mut import.text)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text("{ \"genotype\": ... }"),
            );
            if ui.button("Import").clicked() {
                import.status = Some(admit(&mut hall, &import.text));
                if import.status.as_deref() == Some(IMPORTED) {
                    import.text.clear();
                }
            }
            if let Some(status) = &import.status {
                ui.label(status);
            }
        });
}

const IMPORTED: &str = "Imported: watch it, or breed from it";

/// Imports a robot's JSON into the hall, and says how it went.
fn admit(hall: &mut HallOfFame, text: &str) -> String {
    match Champion::from_json(text) {
        Ok(champion) => {
            hall.admit(champion);
            IMPORTED.to_string()
        }
        Err(err) => format!("Not imported: {err}"),
    }
}

/// Saves a champion's JSON (to the working directory on the desktop, as a
/// download in the browser).
fn save(champion: &Champion) {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let file_name = match champion.fitness {
        Some(fitness) => format!("robot_{fitness:.3}_{timestamp}.json"),
        None => format!("robot_{timestamp}.json"),
    };
    match super::download::save_file(
        &file_name,
        "application/json",
        champion.to_json().as_bytes(),
    ) {
        Ok(()) => info!("Saved {file_name}"),
        Err(err) => warn!("Could not save {file_name}: {err}"),
    }
}

/// Imports `.json` files dropped onto the window (the desktop only: a
/// browser hands the page no file paths).
#[cfg(not(target_arch = "wasm32"))]
pub fn import_dropped_files(
    mut drops: MessageReader<bevy::window::FileDragAndDrop>,
    mut hall: ResMut<HallOfFame>,
) {
    for drop in drops.read() {
        if let bevy::window::FileDragAndDrop::DroppedFile { path_buf, .. } = drop {
            match std::fs::read_to_string(path_buf) {
                Ok(text) => info!("{}: {}", path_buf.display(), admit(&mut hall, &text)),
                Err(err) => warn!("Could not read {}: {err}", path_buf.display()),
            }
        }
    }
}
