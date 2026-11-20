//! Saving a file the user asked for: written to the working directory on the
//! desktop, offered as a download in the browser (which has no file system).

/// Saves `bytes` as `file_name` (of MIME type `mime`, used in the browser).
#[cfg(not(target_arch = "wasm32"))]
pub fn save_file(file_name: &str, _mime: &str, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(file_name, bytes)
}

/// Saves `bytes` as `file_name` (of MIME type `mime`): the bytes become a
/// Blob, and a temporary link to it is clicked so the browser downloads it.
#[cfg(target_arch = "wasm32")]
pub fn save_file(file_name: &str, mime: &str, bytes: &[u8]) -> std::io::Result<()> {
    use wasm_bindgen::{JsCast, JsValue};

    fn js_error(what: &str) -> impl Fn(JsValue) -> std::io::Error + '_ {
        move |err| std::io::Error::other(format!("{what}: {err:?}"))
    }

    let window = web_sys::window().ok_or_else(|| std::io::Error::other("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| std::io::Error::other("no document"))?;
    let body = document
        .body()
        .ok_or_else(|| std::io::Error::other("no document body"))?;

    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(bytes));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(js_error("creating the blob"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(js_error("creating the object URL"))?;

    let anchor: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(js_error("creating the link"))?
        .dyn_into()
        .map_err(|_| std::io::Error::other("the link is not an anchor"))?;
    anchor.set_href(&url);
    anchor.set_download(file_name);
    // Some browsers only follow a link that is in the document.
    body.append_child(&anchor)
        .map_err(js_error("attaching the link"))?;
    anchor.click();
    anchor.remove();
    web_sys::Url::revoke_object_url(&url).map_err(js_error("revoking the object URL"))
}
