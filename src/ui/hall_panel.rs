//! The hall of fame window: the best robots (kept between sessions), to
//! watch walking on their own, share as a link, copy or save as JSON, or
//! send back into evolution; robots to import from JSON or a share link
//! (pasted, or on the desktop a .json file dropped onto the window); and a
//! race of the run's generations, with the showcase's standings.

use crate::visuals::ui::CameraTarget;
use avian3d::prelude::Position;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use symbios_robot_lab::simulation::hall_of_fame::{
    Champion, Entrant, HallOfFame, Immigrants, Lane, Showcase, ShowcaseRequest,
};
use symbios_robot_lab::simulation::lineage::{Lineage, RACE_LANES};
use symbios_robot_lab::simulation::spawn::{SHOWCASE_INDEX, showcase_index};

/// What the user typed into the import box, and what the last action said.
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

#[expect(
    clippy::too_many_arguments,
    reason = "a Bevy system: each parameter is a query or resource it reads"
)]
pub fn render_hall_of_fame(
    mut contexts: EguiContexts,
    mut hall: ResMut<HallOfFame>,
    lineage: Res<Lineage>,
    mut immigrants: ResMut<Immigrants>,
    mut showcase: MessageWriter<ShowcaseRequest>,
    mut camera: ResMut<CameraTarget>,
    showcases: Query<(), With<Showcase>>,
    lanes: Query<(&Lane, &Position)>,
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
                ui.label("The best robots of the generations land here.");
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
                            .button("Breed")
                            .on_hover_text("Send it into the next generation")
                            .clicked()
                        {
                            action = Some(Action::Breed(i));
                        }
                        ui.menu_button("Share", |ui| {
                            if ui
                                .button("Copy a link")
                                .on_hover_text(
                                    "A link that opens the lab with this robot in it \
                                     (its body and brain, not its score)",
                                )
                                .clicked()
                            {
                                ui.ctx()
                                    .copy_text(champion.share_link(&super::keep::share_page()));
                                import.status = Some("Link copied".to_string());
                            }
                            if ui
                                .button("Copy its JSON")
                                .on_hover_text("To paste elsewhere")
                                .clicked()
                            {
                                ui.ctx().copy_text(champion.to_json());
                                import.status = Some("JSON copied".to_string());
                            }
                            if ui.button("Save its JSON").clicked() {
                                save(champion);
                            }
                        });
                        if ui.small_button("x").on_hover_text("Remove").clicked() {
                            action = Some(Action::Remove(i));
                        }
                        ui.end_row();
                    }
                });
            match action {
                Some(Action::Watch(i)) => {
                    let champion = &hall.champions()[i];
                    showcase.write(ShowcaseRequest::one(
                        champion.label(),
                        champion.genotype.clone(),
                    ));
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
                    showcase.write(ShowcaseRequest::dismiss());
                    camera.agent_index = None;
                }
                if !hall.champions().is_empty() && ui.button("Clear the hall").clicked() {
                    hall.clear();
                }
            });
            ui.weak(super::keep::kept_where());

            ui.separator();
            race(ui, &lineage, &lanes, &mut showcase, &mut camera);

            ui.separator();
            ui.label(
                "Import a robot: paste its JSON or share link \
                 (or drop a .json file on the window)",
            );
            ui.add(
                egui::TextEdit::multiline(&mut import.text)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .hint_text("{ \"genotype\": ... } or https://...#robot=..."),
            );
            if ui.button("Import").clicked() {
                let (status, imported) = admit(&mut hall, &import.text);
                if imported {
                    import.text.clear();
                }
                import.status = Some(status);
            }
            if let Some(status) = &import.status {
                ui.label(status);
            }
        });
}

/// The race of generations: a button that sends the run's champions from
/// across its history into the showcase, and the standings of whatever
/// walks there.
fn race(
    ui: &mut egui::Ui,
    lineage: &Lineage,
    lanes: &Query<(&Lane, &Position)>,
    showcase: &mut MessageWriter<ShowcaseRequest>,
    camera: &mut CameraTarget,
) {
    let field = lineage.race(RACE_LANES);
    ui.horizontal(|ui| {
        ui.strong("Race of generations");
        if field.len() < 2 {
            ui.weak("(once the run has bred a generation)");
            return;
        }
        let generations: Vec<String> = field.iter().map(|m| m.generation.to_string()).collect();
        let hover = format!(
            "The run's best robots as of generations {} walk side by side",
            generations.join(", ")
        );
        if ui.button("Race").on_hover_text(hover).clicked() {
            let entrants = field
                .iter()
                .map(|milestone| Entrant {
                    name: format!("Generation {}", milestone.generation),
                    genotype: milestone.champion.genotype.clone(),
                })
                .collect();
            showcase.write(ShowcaseRequest(entrants));
            camera.agent_index = Some(showcase_index(field.len() - 1));
        }
    });
    let mut standings: Vec<(&Lane, f32)> = lanes
        .iter()
        .map(|(lane, position)| (lane, lane.distance(position.0)))
        .collect();
    if standings.is_empty() {
        return;
    }
    standings.sort_by_key(|(lane, _)| lane.lane);
    let lead = standings
        .iter()
        .map(|&(_, distance)| distance)
        .fold(0.0, f32::max);
    egui::Grid::new("race_grid")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (lane, distance) in standings {
                let text = format!("{distance:.1} m");
                if standings_lead(distance, lead) {
                    ui.strong(&lane.name);
                    ui.strong(text);
                } else {
                    ui.label(&lane.name);
                    ui.label(text);
                }
                ui.end_row();
            }
        })
        .response
        .on_hover_text("How far each robot is from where it started");
}

/// Whether a robot `distance` from its start leads (or ties) a field whose
/// best is `lead`, once anyone has moved.
fn standings_lead(distance: f32, lead: f32) -> bool {
    lead > 0.05 && distance >= lead - 1e-3
}

/// Imports robots from pasted JSON or a share link into the hall, and says
/// how it went (and whether anything was read).
fn admit(hall: &mut HallOfFame, text: &str) -> (String, bool) {
    match Champion::import(text) {
        Ok(champions) => {
            let read = champions.len();
            let entered = champions
                .into_iter()
                .filter(|champion| hall.admit(champion.clone()))
                .count();
            let status = match (read, entered) {
                (_, 0) => "Already in the hall".to_string(),
                (1, 1) => "Imported: watch it, or breed from it".to_string(),
                (read, entered) if read == entered => format!("Imported {entered} robots"),
                (read, entered) => {
                    format!(
                        "Imported {entered} robots ({} already in the hall)",
                        read - entered
                    )
                }
            };
            (status, true)
        }
        Err(err) => (format!("Not imported: {err}"), false),
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
                Ok(text) => info!("{}: {}", path_buf.display(), admit(&mut hall, &text).0),
                Err(err) => warn!("Could not read {}: {err}", path_buf.display()),
            }
        }
    }
}
