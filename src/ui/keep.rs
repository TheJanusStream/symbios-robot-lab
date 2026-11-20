//! Keeping the hall of fame between sessions (in the browser's local
//! storage, or on the desktop in a file in the user's data folder), and
//! robots arriving in share links: in the browser the page's own link, on
//! the desktop a link given as an argument.

use crate::visuals::ui::CameraTarget;
use bevy::prelude::*;
use symbios_robot_lab::simulation::hall_of_fame::{
    Champion, HallOfFame, SHARE_FRAGMENT, ShowcaseRequest,
};
use symbios_robot_lab::simulation::spawn::SHOWCASE_INDEX;

/// The page share links open from the desktop: the lab as deployed.
const DEPLOYED_PAGE: &str = "https://thejanusstream.github.io/symbios-robot-lab/";

/// The hall's revision when it was last kept (or restored).
#[derive(Resource, Default)]
pub struct Keeper {
    kept: u64,
}

/// A robot that arrived in a share link, to walk once the lab runs.
#[derive(Resource, Default)]
pub struct Arrival(Option<Champion>);

/// Restores the kept hall, and takes in a robot arriving in a share link.
/// A kept hall that does not come back whole (unreadable, or with robots
/// the import checks refuse) is first copied aside, since the next save
/// replaces it.
pub fn restore_hall(
    mut hall: ResMut<HallOfFame>,
    mut keeper: ResMut<Keeper>,
    mut arrival: ResMut<Arrival>,
) {
    let whole = match store::load() {
        Ok(Some(text)) => {
            let problems = hall.restore(&text);
            for problem in &problems {
                warn!("The kept hall of fame lost a robot: {problem}");
            }
            let held = hall.champions().len();
            let robots = if held == 1 { "robot" } else { "robots" };
            info!("The hall of fame holds {held} {robots}");
            problems.is_empty()
        }
        Ok(None) => true,
        Err(err) => {
            warn!("Could not read the kept hall of fame: {err}");
            false
        }
    };
    if !whole {
        match store::copy_aside() {
            Ok(place) => warn!("The kept hall of fame, as it was, is copied to {place}"),
            Err(err) => warn!("Could not copy the kept hall of fame aside: {err}"),
        }
    }
    keeper.kept = hall.revision();
    if let Some(link) = store::arriving_link() {
        match Champion::from_share(&link) {
            Ok(champion) => {
                hall.admit(champion.clone());
                arrival.0 = Some(champion);
            }
            Err(err) => warn!("The shared robot could not come in: {err}"),
        }
    }
}

/// Keeps the hall whenever it changes.
pub fn keep_hall(hall: Res<HallOfFame>, mut keeper: ResMut<Keeper>) {
    if keeper.kept == hall.revision() {
        return;
    }
    keeper.kept = hall.revision();
    if let Err(err) = store::save(&hall) {
        warn!("Could not keep the hall of fame: {err}");
    }
}

/// Showcases a robot that arrived in a share link, on the lab's first fixed
/// step (a request sent sooner could expire before the showcase reads it).
pub fn showcase_arrival(
    mut arrival: ResMut<Arrival>,
    mut showcase: MessageWriter<ShowcaseRequest>,
    mut camera: ResMut<CameraTarget>,
) {
    if let Some(champion) = arrival.0.take() {
        showcase.write(ShowcaseRequest::one("Shared robot", champion.genotype));
        camera.agent_index = Some(SHOWCASE_INDEX);
    }
}

/// Where the hall is kept, for the user.
pub fn kept_where() -> String {
    store::kept_where()
}

/// The page a share link opens.
pub fn share_page() -> String {
    store::page()
}

#[cfg(not(target_arch = "wasm32"))]
mod store {
    use super::{DEPLOYED_PAGE, HallOfFame, SHARE_FRAGMENT};
    use std::path::PathBuf;

    /// The file the hall is kept in, in the user's data folder:
    /// `$XDG_DATA_HOME` or `~/.local/share` on Linux, `~/Library/Application
    /// Support` on macOS, `%APPDATA%` on Windows.
    fn path() -> Option<PathBuf> {
        let home = || std::env::var_os("HOME").map(PathBuf::from);
        let folder = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            home().map(|home| home.join("Library/Application Support"))
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .filter(|folder| folder.is_absolute())
                .or_else(|| home().map(|home| home.join(".local/share")))
        };
        folder.map(|folder| folder.join("symbios-robot-lab").join("hall_of_fame.json"))
    }

    pub fn load() -> Result<Option<String>, String> {
        let Some(path) = path() else {
            return Ok(None);
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(format!("{}: {err}", path.display())),
        }
    }

    /// Writes the hall (as pretty JSON, which imports elsewhere like any
    /// saved hall) beside its file and then moves it into place, so a crash
    /// midway leaves the last hall whole.
    pub fn save(hall: &HallOfFame) -> Result<(), String> {
        let path = path().ok_or("there is no data folder to keep it in")?;
        let fail = |err: std::io::Error| format!("{}: {err}", path.display());
        if let Some(folder) = path.parent() {
            std::fs::create_dir_all(folder).map_err(fail)?;
        }
        let part = path.with_extension("json.part");
        std::fs::write(&part, hall.to_json()).map_err(fail)?;
        std::fs::rename(&part, &path).map_err(fail)
    }

    /// Copies the kept hall beside it, as `hall_of_fame.unread.json`; says
    /// where.
    pub fn copy_aside() -> Result<String, String> {
        let path = path().ok_or("there is no data folder")?;
        let copy = path.with_file_name("hall_of_fame.unread.json");
        std::fs::copy(&path, &copy).map_err(|err| format!("{}: {err}", copy.display()))?;
        Ok(copy.display().to_string())
    }

    pub fn kept_where() -> String {
        match path() {
            Some(path) => format!("Kept between sessions in {}", path.display()),
            None => "Not kept: there is no data folder".to_string(),
        }
    }

    /// A share link given on the command line.
    pub fn arriving_link() -> Option<String> {
        std::env::args()
            .skip(1)
            .find(|arg| arg.contains(SHARE_FRAGMENT))
    }

    pub fn page() -> String {
        DEPLOYED_PAGE.to_string()
    }
}

#[cfg(target_arch = "wasm32")]
mod store {
    use super::{DEPLOYED_PAGE, HallOfFame, SHARE_FRAGMENT};
    use symbios_robot_lab::core::share;
    use wasm_bindgen::JsValue;

    /// The local storage entry the hall is kept in: its compact JSON as a
    /// share code, about 9 KB for ten champions. The browser allows a site
    /// a few megabytes, shared by every page on it.
    const KEY: &str = "symbios-robot-lab/hall-of-fame";
    /// Where a kept hall that did not come back whole is copied.
    const UNREAD_KEY: &str = "symbios-robot-lab/hall-of-fame-unread";
    /// The most the kept hall may inflate to. It is the lab's own, so the
    /// limit is far above a share code's ([`share::MAX_INFLATED`]): a hall of
    /// big direct brains passes a megabyte of JSON.
    const MAX_KEPT: usize = 256 << 20;

    fn storage() -> Result<web_sys::Storage, String> {
        web_sys::window()
            .ok_or("there is no window")?
            .local_storage()
            .map_err(|err| format!("{err:?}"))?
            .ok_or_else(|| "the browser keeps no local storage".to_string())
    }

    pub fn load() -> Result<Option<String>, String> {
        let Some(code) = storage()?.get_item(KEY).map_err(|err| format!("{err:?}"))? else {
            return Ok(None);
        };
        let bytes = share::decode_within(&code, MAX_KEPT)?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| "it holds no text".to_string())
    }

    /// Copies the kept hall to its own entry; says where.
    pub fn copy_aside() -> Result<String, String> {
        let storage = storage()?;
        let kept = storage
            .get_item(KEY)
            .map_err(|err| format!("{err:?}"))?
            .ok_or("nothing is kept")?;
        storage
            .set_item(UNREAD_KEY, &kept)
            .map_err(|err| format!("{err:?}"))?;
        Ok(format!("local storage as {UNREAD_KEY}"))
    }

    pub fn save(hall: &HallOfFame) -> Result<(), String> {
        let code = share::encode(hall.to_compact_json().as_bytes());
        storage()?
            .set_item(KEY, &code)
            .map_err(|err| format!("{err:?}"))
    }

    pub fn kept_where() -> String {
        "Kept in this browser between visits".to_string()
    }

    /// The page's own link, if it carries a robot. The robot then leaves the
    /// address, so reloading the page does not bring it in again.
    pub fn arriving_link() -> Option<String> {
        let window = web_sys::window()?;
        let location = window.location();
        let hash = location.hash().ok()?;
        if !hash.contains(SHARE_FRAGMENT) {
            return None;
        }
        let bare = format!(
            "{}{}",
            location.pathname().unwrap_or_default(),
            location.search().unwrap_or_default()
        );
        if let Ok(history) = window.history() {
            let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&bare));
        }
        Some(hash)
    }

    pub fn page() -> String {
        let location = web_sys::window().map(|window| window.location());
        match location.and_then(|l| Some((l.origin().ok()?, l.pathname().ok()?))) {
            Some((origin, path)) => format!("{origin}{path}"),
            None => DEPLOYED_PAGE.to_string(),
        }
    }
}
