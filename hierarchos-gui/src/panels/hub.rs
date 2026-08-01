// panels/hub.rs — HuggingFace Model Browser
//
// Allows searching, browsing, and downloading Hierarchos models directly
// from the HuggingFace Hub.

use egui::{self, CornerRadius as Rounding, ProgressBar, RichText, ScrollArea, Stroke, Vec2};
use serde::Deserialize;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use crate::bridge::PythonBridge;
use crate::theme::{get_accent, HierarchosColors};

/// Basic model info from the HF API.
#[derive(Debug, Clone, Deserialize)]
pub struct HFModel {
    #[serde(default, alias = "modelId")]
    pub id: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub downloads: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct HFModelDetails {
    siblings: Option<Vec<HFSibling>>,
}

#[derive(Debug, Clone, Deserialize)]
struct HFSibling {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Clone)]
pub enum DownloadState {
    NotStarted,
    Downloading { progress: f32, status: String },
    Complete { path: String },
    Error(String),
}

pub struct HubState {
    pub search_query: String,
    pub is_searching: bool,
    pub search_error: Option<String>,
    pub search_results: Vec<HFModel>,

    // Store results of async searches
    search_rx: Option<std::sync::mpsc::Receiver<Result<Vec<HFModel>, String>>>,

    // Download tracking: model_id -> State
    pub downloads: std::collections::HashMap<String, Arc<Mutex<DownloadState>>>,
}

impl Default for HubState {
    fn default() -> Self {
        Self {
            search_query: "hierarchos".to_string(), // Default search query
            is_searching: false,
            search_error: None,
            search_results: Vec::new(),
            search_rx: None,
            downloads: std::collections::HashMap::new(),
        }
    }
}

pub fn draw_hub_panel(
    ui: &mut egui::Ui,
    state: &mut HubState,
    settings: &mut crate::panels::settings::AppSettings,
    bridge: &PythonBridge,
) {
    ui.vertical(|ui| {
        // Header
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("🤗 HuggingFace Hub")
                    .size(24.0)
                    .strong()
                    .color(HierarchosColors::TEXT_PRIMARY),
            );
        });
        ui.add_space(8.0);
        ui.label(
            RichText::new("Discover and download Hierarchos models.")
                .color(HierarchosColors::TEXT_SECONDARY),
        );
        ui.add_space(16.0);

        draw_local_model_loader(ui, settings, bridge);
        ui.add_space(20.0);

        ui.separator();
        ui.add_space(14.0);
        ui.label(
            RichText::new("Hugging Face Search")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(16.0)
                .strong(),
        );
        ui.add_space(10.0);

        // Search Bar
        ui.horizontal(|ui| {
            let search_box = egui::TextEdit::singleline(&mut state.search_query)
                .hint_text("Search models...")
                .desired_width(400.0)
                .margin(Vec2::new(10.0, 10.0));

            let response = ui.add(search_box);

            let search_btn = egui::Button::new(RichText::new("Search").size(14.0))
                .fill(get_accent().primary)
                .corner_radius(Rounding::same(6));

            if (ui.add(search_btn).clicked()
                || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))))
                && !state.is_searching
            {
                start_search(state);
            }
        });

        ui.add_space(16.0);

        // Check for search results
        if let Some(rx) = &state.search_rx {
            if let Ok(result) = rx.try_recv() {
                state.is_searching = false;
                state.search_rx = None;
                match result {
                    Ok(models) => {
                        state.search_results = models;
                        state.search_error = None;
                    }
                    Err(e) => {
                        state.search_error = Some(e);
                    }
                }
            }
        }

        if state.is_searching {
            ui.spinner();
            ui.label("Searching...");
            return;
        }

        if let Some(err) = &state.search_error {
            ui.label(RichText::new(format!("Error: {}", err)).color(HierarchosColors::ERROR));
            return;
        }

        if state.search_results.is_empty() {
            ui.label(RichText::new("No models found.").color(HierarchosColors::TEXT_MUTED));
            return;
        }

        // Results list
        let models = state.search_results.clone();
        ScrollArea::vertical().show(ui, |ui| {
            for model in &models {
                draw_model_card(ui, model, &mut state.downloads, settings, bridge);
                ui.add_space(12.0);
            }
        });
    });
}

fn draw_local_model_loader(
    ui: &mut egui::Ui,
    settings: &mut crate::panels::settings::AppSettings,
    bridge: &PythonBridge,
) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("Local Model Source")
                .size(15.0)
                .strong()
                .color(HierarchosColors::TEXT_PRIMARY),
        );
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            let source_width = (ui.available_width() - 310.0).max(160.0);
            ui.add(
                egui::TextEdit::singleline(&mut settings.model_path)
                    .desired_width(source_width)
                    .hint_text("Model folder, direct .pt file, or HF repo id"),
            );

            if ui
                .button("Folder")
                .on_hover_text("Choose a local model directory")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    settings.model_path = path.display().to_string();
                }
            }

            if ui
                .button(".pt")
                .on_hover_text("Choose a PyTorch checkpoint")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PyTorch checkpoint", &["pt"])
                    .pick_file()
                {
                    settings.model_path = path.display().to_string();
                }
            }

            let can_load = !settings.model_path.trim().is_empty();
            let label = if bridge.is_connected() {
                "Load"
            } else {
                "Connect & Load"
            };
            let load_button = egui::Button::new(
                RichText::new(label)
                    .color(HierarchosColors::TEXT_ON_PRIMARY)
                    .size(13.0),
            )
            .fill(get_accent().primary)
            .corner_radius(Rounding::same(6));

            if ui.add_enabled(can_load, load_button).clicked() {
                request_model_load(settings, bridge);
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Device")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            for device in settings.available_devices.clone() {
                let selected = settings.device == device;
                let fill = if selected {
                    get_accent().primary_dim
                } else {
                    HierarchosColors::BG_SURFACE
                };
                let text = if selected {
                    HierarchosColors::TEXT_ON_PRIMARY
                } else {
                    HierarchosColors::TEXT_SECONDARY
                };
                let hover = match device.as_str() {
                    "Auto" => "Uses CUDA on NVIDIA GPUs when available; otherwise CPU.",
                    "CUDA" => "NVIDIA CUDA acceleration. Requires a CUDA-capable GPU and driver.",
                    "CPU" => "Portable fallback for non-NVIDIA PCs and handhelds.",
                    "DirectML" => "Optional DirectML backend if torch-directml is installed.",
                    _ => "",
                };

                if ui
                    .add(
                        egui::Button::new(RichText::new(&device).color(text).size(12.0))
                            .fill(fill)
                            .stroke(Stroke::new(
                                1.0,
                                if selected {
                                    get_accent().primary
                                } else {
                                    HierarchosColors::BORDER_SUBTLE
                                },
                            ))
                            .corner_radius(Rounding::same(6)),
                    )
                    .on_hover_text(hover)
                    .clicked()
                {
                    settings.device = device;
                }
            }
        });

        if settings.pending_load {
            ui.add_space(8.0);
            ui.label(
                RichText::new("Backend is starting; the selected model will load automatically.")
                    .color(HierarchosColors::WARNING)
                    .size(12.0),
            );
        }
    });
}

fn draw_model_card(
    ui: &mut egui::Ui,
    model: &HFModel,
    downloads: &mut std::collections::HashMap<String, Arc<Mutex<DownloadState>>>,
    settings: &mut crate::panels::settings::AppSettings,
    bridge: &PythonBridge,
) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(&model.id)
                        .size(16.0)
                        .strong()
                        .color(get_accent().primary),
                );
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "Author: {}",
                            model.author.as_deref().unwrap_or("unknown")
                        ))
                        .size(12.0)
                        .color(HierarchosColors::TEXT_SECONDARY),
                    );
                    ui.label(
                        RichText::new("•")
                            .size(12.0)
                            .color(HierarchosColors::TEXT_MUTED),
                    );
                    ui.label(
                        RichText::new(format!("Downloads: {}", model.downloads.unwrap_or(0)))
                            .size(12.0)
                            .color(HierarchosColors::TEXT_SECONDARY),
                    );
                    ui.label(
                        RichText::new("•")
                            .size(12.0)
                            .color(HierarchosColors::TEXT_MUTED),
                    );
                    ui.label(
                        RichText::new(format!(
                            "Updated: {}",
                            model
                                .last_modified
                                .as_deref()
                                .unwrap_or("unknown")
                                .split('T')
                                .next()
                                .unwrap_or("unknown")
                        ))
                        .size(12.0)
                        .color(HierarchosColors::TEXT_SECONDARY),
                    );
                });
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let dl_state_arc = downloads
                    .entry(model.id.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(DownloadState::NotStarted)))
                    .clone();
                let dl_state = dl_state_arc.lock().unwrap().clone();

                match dl_state {
                    DownloadState::NotStarted => {
                        ui.horizontal(|ui| {
                            if ui.button("Load Repo").clicked() {
                                settings.model_path = model.id.clone();
                                request_model_load(settings, bridge);
                            }
                            if ui.button("⬇ Download").clicked() {
                                start_download(model.id.clone(), dl_state_arc);
                            }
                        });
                    }
                    DownloadState::Downloading { progress, status } => {
                        ui.vertical(|ui| {
                            ui.add(ProgressBar::new(progress).desired_width(150.0).text(status));
                        });
                    }
                    DownloadState::Complete { path } => {
                        if ui.button("Load Downloaded").clicked() {
                            settings.model_path = path;
                            request_model_load(settings, bridge);
                        }
                    }
                    DownloadState::Error(err) => {
                        ui.label(
                            RichText::new(format!("Error: {}", err)).color(HierarchosColors::ERROR),
                        );
                        if ui.button("Retry").clicked() {
                            start_download(model.id.clone(), dl_state_arc);
                        }
                    }
                }
            });
        });
    });
}

fn request_model_load(settings: &mut crate::panels::settings::AppSettings, bridge: &PythonBridge) {
    if settings.model_path.trim().is_empty() {
        return;
    }

    if bridge.is_connected() {
        bridge.load_model(settings.model_path.clone(), settings.device.clone());
        settings.pending_load = false;
    } else {
        settings.pending_load = true;
        bridge.connect(&settings.python_path);
    }
}

fn start_search(state: &mut HubState) {
    state.is_searching = true;
    state.search_error = None;
    let (tx, rx) = std::sync::mpsc::channel();
    state.search_rx = Some(rx);

    let query = {
        let trimmed = state.search_query.trim();
        if trimmed.is_empty() {
            "hierarchos".to_string()
        } else {
            trimmed.to_string()
        }
    };

    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .user_agent("Hierarchos-GUI/0.1")
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                let _ = tx.send(Err(format!("Could not create HTTP client: {}", e)));
                return;
            }
        };

        let res = match client
            .get("https://huggingface.co/api/models")
            .header("Accept", "application/json")
            .query(&[
                ("search", query.as_str()),
                ("sort", "downloads"),
                ("direction", "-1"),
                ("limit", "50"),
            ])
            .send()
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
                return;
            }
        };

        let status = res.status();
        let body = match res.text() {
            Ok(body) => body,
            Err(e) => {
                let _ = tx.send(Err(format!("Could not read Hugging Face response: {}", e)));
                return;
            }
        };

        if !status.is_success() {
            let preview: String = body.chars().take(180).collect();
            let _ = tx.send(Err(format!("HTTP Error: {} {}", status, preview)));
            return;
        }

        match serde_json::from_str::<Vec<HFModel>>(&body) {
            Ok(mut models) => {
                models.retain(|m| !m.id.is_empty());
                // Sort by downloads descending
                models.sort_by(|a, b| b.downloads.unwrap_or(0).cmp(&a.downloads.unwrap_or(0)));
                let _ = tx.send(Ok(models));
            }
            Err(e) => {
                let preview: String = body.chars().take(180).collect();
                let _ = tx.send(Err(format!(
                    "Could not parse Hugging Face response: {}. Response starts with: {}",
                    e, preview
                )));
            }
        }
    });
}

fn start_download(model_id: String, state: Arc<Mutex<DownloadState>>) {
    let api_url = format!("https://huggingface.co/api/models/{}", model_id);
    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);
    let target_dir = crate::embedded::get_models_dir().join(model_id.replace("/", "_"));

    if let Err(e) = std::fs::create_dir_all(&target_dir) {
        *state.lock().unwrap() = DownloadState::Error(format!("Could not create directory: {}", e));
        return;
    }

    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .user_agent("Hierarchos-GUI/0.1")
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                *state.lock().unwrap() =
                    DownloadState::Error(format!("Could not create HTTP client: {}", e));
                return;
            }
        };

        let details = match client
            .get(&api_url)
            .send()
            .and_then(|r| r.error_for_status())
        {
            Ok(r) => match r.text() {
                Ok(body) => match serde_json::from_str::<HFModelDetails>(&body) {
                    Ok(details) => details,
                    Err(e) => {
                        *state.lock().unwrap() =
                            DownloadState::Error(format!("Could not parse model file list: {}", e));
                        return;
                    }
                },
                Err(e) => {
                    *state.lock().unwrap() =
                        DownloadState::Error(format!("Could not read model file list: {}", e));
                    return;
                }
            },
            Err(e) => {
                *state.lock().unwrap() =
                    DownloadState::Error(format!("Could not query model files: {}", e));
                return;
            }
        };

        let mut files: Vec<HFSibling> = details
            .siblings
            .unwrap_or_default()
            .into_iter()
            .filter(|s| should_download_file(&s.rfilename))
            .collect();

        files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));

        if !files
            .iter()
            .any(|s| s.rfilename.to_ascii_lowercase().ends_with(".pt"))
        {
            *state.lock().unwrap() = DownloadState::Error(
                "No .pt Hierarchos checkpoint found in this repo.".to_string(),
            );
            return;
        }

        let total_known_bytes: u64 = files.iter().filter_map(|f| f.size).sum();
        let mut completed_known_bytes: u64 = 0;
        let total_files = files.len().max(1);

        for (i, file) in files.iter().enumerate() {
            let encoded_path = file
                .rfilename
                .split('/')
                .map(|part| urlencoding::encode(part).to_string())
                .collect::<Vec<_>>()
                .join("/");
            let file_url = format!("{}/{}", base_url, encoded_path);
            let target_path = target_dir.join(&file.rfilename);

            if let Some(parent) = target_path.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    *state.lock().unwrap() =
                        DownloadState::Error(format!("Could not create directory: {}", e));
                    return;
                }
            }

            {
                let mut s = state.lock().unwrap();
                *s = DownloadState::Downloading {
                    progress: i as f32 / total_files as f32,
                    status: format!("Downloading {}", file.rfilename),
                };
            }

            let mut res = match client
                .get(&file_url)
                .send()
                .and_then(|r| r.error_for_status())
            {
                Ok(r) => r,
                Err(e) => {
                    *state.lock().unwrap() = DownloadState::Error(format!(
                        "Download failed for {}: {}",
                        file.rfilename, e
                    ));
                    return;
                }
            };

            let mut dest = match std::fs::File::create(&target_path) {
                Ok(f) => f,
                Err(e) => {
                    *state.lock().unwrap() =
                        DownloadState::Error(format!("Could not write {}: {}", file.rfilename, e));
                    return;
                }
            };

            let mut buffer = [0_u8; 256 * 1024];
            let mut downloaded_this_file: u64 = 0;
            loop {
                let read = match res.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => {
                        *state.lock().unwrap() =
                            DownloadState::Error(format!("Network read failed: {}", e));
                        return;
                    }
                };

                if let Err(e) = dest.write_all(&buffer[..read]) {
                    *state.lock().unwrap() = DownloadState::Error(format!("Write failed: {}", e));
                    return;
                }

                downloaded_this_file += read as u64;
                let progress = if total_known_bytes > 0 {
                    (completed_known_bytes + downloaded_this_file) as f32 / total_known_bytes as f32
                } else {
                    (i as f32 + 0.5) / total_files as f32
                };

                let mut s = state.lock().unwrap();
                *s = DownloadState::Downloading {
                    progress: progress.clamp(0.0, 0.99),
                    status: format!("Downloading {}", file.rfilename),
                };
            }

            completed_known_bytes += file.size.unwrap_or(downloaded_this_file);
        }

        *state.lock().unwrap() = DownloadState::Complete {
            path: target_dir.to_string_lossy().to_string(),
        };
    });
}

fn should_download_file(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.ends_with(".pt")
        || lower.ends_with(".json")
        || lower.ends_with(".model")
        || lower == "merges.txt"
        || lower == "vocab.json"
        || lower == "tokenizer.json"
        || lower == "tokenizer_config.json"
        || lower == "special_tokens_map.json"
        || lower == "added_tokens.json"
}
