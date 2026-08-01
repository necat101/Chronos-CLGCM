// panels/chat.rs — Chat Interface Panel
//
// The primary user-facing view for interacting with Hierarchos.
// Features message bubbles, streaming tokens, slash commands,
// sampling parameter sidebar, and feedback buttons.

use crate::bridge::{PythonBridge, SamplingParams};
use crate::theme::{get_accent, HierarchosColors};
use crate::widgets::message_bubble::{typing_indicator, MessageBubble, MessageRole};
use egui::{self, Color32, CornerRadius as Rounding, RichText, ScrollArea, Stroke, Vec2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// A single chat message.
#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole_,
    pub content: String,
    pub timestamp: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum MessageRole_ {
    User,
    Assistant,
    System,
}

#[derive(Clone, Serialize, Deserialize)]
struct ChatSessionData {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub sampling: SamplingParams,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub runtime_state_path: String,
}

#[derive(Clone)]
pub struct ChatSessionSummary {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: usize,
}

/// State for the chat panel.
pub struct ChatState {
    pub active_session_id: String,
    pub sessions: Vec<ChatSessionSummary>,
    pub messages: Vec<ChatMessage>,
    pub input_text: String,
    pub sampling: SamplingParams,
    pub is_generating: bool,
    pub current_stream: String,
    pub show_sidebar: bool,
    pub show_history: bool,
    pub scroll_to_bottom: bool,
    pub total_tokens: u64,
    storage_dir: PathBuf,
}

impl Default for ChatState {
    fn default() -> Self {
        Self::load_or_create()
    }
}

impl ChatState {
    fn load_or_create() -> Self {
        let storage_dir = crate::embedded::get_app_data_dir().join("chats");
        let _ = fs::create_dir_all(storage_dir.join("runtime"));

        let mut sessions = Self::load_session_summaries(&storage_dir);
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        if let Some(first) = sessions.first().cloned() {
            if let Ok(data) = Self::read_session_file(&storage_dir, &first.id) {
                return Self {
                    active_session_id: data.id,
                    sessions,
                    messages: data.messages,
                    input_text: String::new(),
                    sampling: data.sampling,
                    is_generating: false,
                    current_stream: String::new(),
                    show_sidebar: false,
                    show_history: false,
                    scroll_to_bottom: true,
                    total_tokens: data.total_tokens,
                    storage_dir,
                };
            }
        }

        let id = Self::new_session_id();
        let now = Self::now_datetime();
        let mut state = Self {
            active_session_id: id,
            sessions: Vec::new(),
            messages: vec![Self::welcome_message()],
            input_text: String::new(),
            sampling: SamplingParams::default(),
            is_generating: false,
            current_stream: String::new(),
            show_sidebar: false,
            show_history: false,
            scroll_to_bottom: true,
            total_tokens: 0,
            storage_dir,
        };
        state.sessions.push(ChatSessionSummary {
            id: state.active_session_id.clone(),
            title: "New chat".to_string(),
            created_at: now.clone(),
            updated_at: now,
            message_count: state.messages.len(),
        });
        let _ = state.save_active_chat();
        state
    }

    fn now_timestamp() -> String {
        chrono::Local::now().format("%H:%M").to_string()
    }

    fn now_datetime() -> String {
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    }

    fn new_session_id() -> String {
        format!(
            "chat-{}-{}",
            chrono::Local::now().format("%Y%m%d%H%M%S%3f"),
            std::process::id()
        )
    }

    fn welcome_message() -> ChatMessage {
        ChatMessage {
            role: MessageRole_::System,
            content: "Welcome to Hierarchos. Load a model via Settings, then start chatting.\nCommands: /new, /clear, /reset, /reset_ltm, /status, /temp <f>, /topk <n>, /topp <f>, /threads <n>".to_string(),
            timestamp: Self::now_timestamp(),
        }
    }

    fn chat_file_path(storage_dir: &Path, id: &str) -> PathBuf {
        storage_dir.join(format!("{}.json", id))
    }

    fn runtime_file_path(storage_dir: &Path, id: &str) -> PathBuf {
        storage_dir.join("runtime").join(format!("{}.pt", id))
    }

    fn read_session_file(storage_dir: &Path, id: &str) -> Result<ChatSessionData, String> {
        let path = Self::chat_file_path(storage_dir, id);
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("Could not read {}: {}", path.display(), e))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("Could not parse {}: {}", path.display(), e))
    }

    fn load_session_summaries(storage_dir: &Path) -> Vec<ChatSessionSummary> {
        let mut summaries = Vec::new();
        let entries = match fs::read_dir(storage_dir) {
            Ok(entries) => entries,
            Err(_) => return summaries,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(data) = serde_json::from_str::<ChatSessionData>(&text) else {
                continue;
            };
            summaries.push(ChatSessionSummary {
                id: data.id,
                title: data.title,
                created_at: data.created_at,
                updated_at: data.updated_at,
                message_count: data.messages.len(),
            });
        }

        summaries
    }

    fn title_from_messages(&self) -> String {
        self.messages
            .iter()
            .find(|msg| msg.role == MessageRole_::User && !msg.content.trim().is_empty())
            .map(|msg| {
                let mut title = msg.content.trim().replace('\n', " ");
                if title.chars().count() > 42 {
                    title = title.chars().take(39).collect::<String>() + "...";
                }
                title
            })
            .unwrap_or_else(|| "New chat".to_string())
    }

    fn upsert_active_summary(&mut self, title: String, updated_at: String) {
        let summary = ChatSessionSummary {
            id: self.active_session_id.clone(),
            title,
            created_at: self
                .sessions
                .iter()
                .find(|session| session.id == self.active_session_id)
                .map(|session| session.created_at.clone())
                .unwrap_or_else(|| updated_at.clone()),
            updated_at,
            message_count: self.messages.len(),
        };

        if let Some(existing) = self
            .sessions
            .iter_mut()
            .find(|session| session.id == self.active_session_id)
        {
            *existing = summary;
        } else {
            self.sessions.push(summary);
        }
        self.sessions
            .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    }

    pub fn runtime_state_path(&self) -> PathBuf {
        Self::runtime_file_path(&self.storage_dir, &self.active_session_id)
    }

    pub fn runtime_state_path_string(&self) -> String {
        self.runtime_state_path().to_string_lossy().to_string()
    }

    pub fn runtime_state_exists(&self) -> bool {
        self.runtime_state_path().exists()
    }

    pub fn save_active_chat(&mut self) -> Result<(), String> {
        fs::create_dir_all(self.storage_dir.join("runtime")).map_err(|e| {
            format!(
                "Could not create chat storage at {}: {}",
                self.storage_dir.display(),
                e
            )
        })?;

        let title = self.title_from_messages();
        let updated_at = Self::now_datetime();
        let data = ChatSessionData {
            id: self.active_session_id.clone(),
            title: title.clone(),
            created_at: self
                .sessions
                .iter()
                .find(|session| session.id == self.active_session_id)
                .map(|session| session.created_at.clone())
                .unwrap_or_else(|| updated_at.clone()),
            updated_at: updated_at.clone(),
            messages: self.messages.clone(),
            sampling: self.sampling.clone(),
            total_tokens: self.total_tokens,
            runtime_state_path: self.runtime_state_path_string(),
        };

        let path = Self::chat_file_path(&self.storage_dir, &self.active_session_id);
        let temp_path = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(&data)
            .map_err(|e| format!("Could not serialize chat: {}", e))?;
        fs::write(&temp_path, text)
            .map_err(|e| format!("Could not write {}: {}", temp_path.display(), e))?;
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| format!("Could not replace {}: {}", path.display(), e))?;
        }
        fs::rename(&temp_path, &path)
            .map_err(|e| format!("Could not replace {}: {}", path.display(), e))?;

        self.upsert_active_summary(title, updated_at);
        Ok(())
    }

    pub fn switch_to_session(&mut self, id: &str) -> Result<(), String> {
        if id == self.active_session_id {
            return Ok(());
        }
        let _ = self.save_active_chat();
        let data = Self::read_session_file(&self.storage_dir, id)?;
        self.active_session_id = data.id;
        self.messages = data.messages;
        self.input_text.clear();
        self.sampling = data.sampling;
        self.is_generating = false;
        self.current_stream.clear();
        self.total_tokens = data.total_tokens;
        self.scroll_to_bottom = true;
        Ok(())
    }

    /// Clear visible chat history without resetting backend runtime state.
    pub fn clear_visible_chat(&mut self) {
        self.messages.clear();
        self.input_text.clear();
        self.current_stream.clear();
        self.is_generating = false;
        self.total_tokens = 0;
        self.scroll_to_bottom = true;
        self.on_status("Chat cleared.".to_string());
    }

    /// Start a fresh conversation and clear local streaming counters.
    pub fn start_new_chat(&mut self) {
        let _ = self.save_active_chat();
        self.active_session_id = Self::new_session_id();
        self.messages.clear();
        self.input_text.clear();
        self.current_stream.clear();
        self.is_generating = false;
        self.total_tokens = 0;
        self.scroll_to_bottom = true;
        self.messages.push(Self::welcome_message());
        let _ = self.save_active_chat();
    }

    /// Handle a streamed token.
    pub fn on_token(&mut self, token: String) {
        self.current_stream.push_str(&token);
        self.total_tokens += 1;
        self.scroll_to_bottom = true;
    }

    /// Handle generation complete.
    pub fn on_generation_complete(&mut self) {
        if !self.current_stream.is_empty() {
            self.messages.push(ChatMessage {
                role: MessageRole_::Assistant,
                content: self.current_stream.clone(),
                timestamp: Self::now_timestamp(),
            });
            self.current_stream.clear();
        }
        self.is_generating = false;
        let _ = self.save_active_chat();
    }

    /// Handle system status message.
    pub fn on_status(&mut self, msg: String) {
        self.messages.push(ChatMessage {
            role: MessageRole_::System,
            content: msg,
            timestamp: Self::now_timestamp(),
        });
        self.scroll_to_bottom = true;
        let _ = self.save_active_chat();
    }
}

/// Draw the chat panel.
pub fn draw_chat_panel(ui: &mut egui::Ui, state: &mut ChatState, bridge: &PythonBridge) {
    let total_width = ui.available_width();

    // Chat header — always at the top
    draw_chat_header(ui, state, bridge);
    ui.add_space(4.0);

    // Calculate sidebar width
    let chat_content_width = total_width;

    // Remaining height for messages + input
    let remaining_height = ui.available_height();
    let input_area_height = 56.0;
    let messages_height = (remaining_height - input_area_height - 12.0).max(120.0);

    // Main horizontal split: chat area | sidebar
    ui.horizontal_top(|ui| {
        // Chat messages + input column
        ui.vertical(|ui| {
            ui.set_width(chat_content_width);

            // Messages scroll area
            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_DARKEST)
                .corner_radius(Rounding::same(8))
                .inner_margin(egui::Margin::same(12));

            frame.show(ui, |ui| {
                ScrollArea::vertical()
                    .max_height(messages_height)
                    .auto_shrink([false, false])
                    .stick_to_bottom(state.scroll_to_bottom)
                    .show(ui, |ui| {
                        ui.set_min_width(chat_content_width - 50.0);

                        for msg in &state.messages {
                            let role = match msg.role {
                                MessageRole_::User => MessageRole::User,
                                MessageRole_::Assistant => MessageRole::Assistant,
                                MessageRole_::System => MessageRole::System,
                            };

                            let bubble = MessageBubble {
                                role,
                                content: &msg.content,
                                timestamp: &msg.timestamp,
                                is_streaming: false,
                            };
                            bubble.show(ui);
                            ui.add_space(8.0);
                        }

                        // Streaming message
                        if state.is_generating && !state.current_stream.is_empty() {
                            let bubble = MessageBubble {
                                role: MessageRole::Assistant,
                                content: &state.current_stream,
                                timestamp: &chrono::Local::now().format("%H:%M").to_string(),
                                is_streaming: true,
                            };
                            bubble.show(ui);
                            ui.add_space(8.0);
                            ui.ctx().request_repaint();
                        } else if state.is_generating {
                            typing_indicator(ui);
                            ui.ctx().request_repaint();
                        }

                        // Feedback buttons for last assistant message
                        if !state.is_generating && state.messages.len() >= 2 {
                            if let Some(last) = state.messages.last() {
                                if last.role == MessageRole_::Assistant {
                                    ui.horizontal(|ui| {
                                        ui.add_space(8.0);
                                        if ui
                                            .add(
                                                egui::Button::new(RichText::new("👍").size(16.0))
                                                    .fill(Color32::TRANSPARENT)
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        HierarchosColors::BORDER_SUBTLE,
                                                    ))
                                                    .corner_radius(Rounding::same(6)),
                                            )
                                            .on_hover_text("Good response — reinforce memory")
                                            .clicked()
                                        {
                                            bridge.send_feedback(true);
                                            state.on_status("Positive feedback sent.".to_string());
                                        }
                                        if ui
                                            .add(
                                                egui::Button::new(RichText::new("👎").size(16.0))
                                                    .fill(Color32::TRANSPARENT)
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        HierarchosColors::BORDER_SUBTLE,
                                                    ))
                                                    .corner_radius(Rounding::same(6)),
                                            )
                                            .on_hover_text("Bad response — penalize memory")
                                            .clicked()
                                        {
                                            bridge.send_feedback(false);
                                            state.on_status("Negative feedback sent.".to_string());
                                        }
                                    });
                                }
                            }
                        }

                        state.scroll_to_bottom = false;
                    });
            });

            ui.add_space(4.0);

            // Input area
            draw_input_area(ui, state, bridge);
        });
    });

    if state.show_sidebar {
        draw_sampling_window(ui.ctx(), state, bridge);
    }

    if state.show_history {
        draw_history_window(ui.ctx(), state, bridge);
    }
}

fn draw_chat_header(ui: &mut egui::Ui, state: &mut ChatState, bridge: &PythonBridge) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new("💬 Chat")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(18.0)
                .strong(),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Toggle sidebar
            let sidebar_text = if state.show_sidebar {
                "⚙ Hide"
            } else {
                "⚙ Sampling"
            };
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(sidebar_text)
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    )
                    .fill(HierarchosColors::BG_CARD)
                    .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                    .corner_radius(Rounding::same(6)),
                )
                .clicked()
            {
                state.show_sidebar = !state.show_sidebar;
            }

            if ui
                .add(
                    egui::Button::new(
                        RichText::new("History")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    )
                    .fill(HierarchosColors::BG_CARD)
                    .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                    .corner_radius(Rounding::same(6)),
                )
                .on_hover_text("Reopen a saved chat session")
                .clicked()
            {
                state.show_history = !state.show_history;
            }

            if ui
                .add(
                    egui::Button::new(
                        RichText::new("Clear")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    )
                    .fill(HierarchosColors::BG_CARD)
                    .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                    .corner_radius(Rounding::same(6)),
                )
                .on_hover_text("Clear visible chat history")
                .clicked()
            {
                if state.is_generating {
                    bridge.stop_generation();
                }
                state.clear_visible_chat();
            }

            if ui
                .add(
                    egui::Button::new(
                        RichText::new("New")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    )
                    .fill(HierarchosColors::BG_CARD)
                    .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                    .corner_radius(Rounding::same(6)),
                )
                .on_hover_text("Start a new chat and reset runtime state")
                .clicked()
            {
                start_new_chat_session(state, bridge);
            }

            // Token count
            if state.total_tokens > 0 {
                ui.label(
                    RichText::new(format!("{} tokens", state.total_tokens))
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0),
                );
            }
        });
    });
}

fn draw_input_area(ui: &mut egui::Ui, state: &mut ChatState, bridge: &PythonBridge) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_SURFACE)
        .stroke(Stroke::new(1.0, get_accent().border))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::symmetric(12, 8));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let text_edit = egui::TextEdit::singleline(&mut state.input_text)
                .desired_width(ui.available_width() - 120.0)
                .hint_text(
                    RichText::new("Type a message or /command...")
                        .color(HierarchosColors::TEXT_MUTED),
                )
                .text_color(HierarchosColors::TEXT_PRIMARY)
                .frame(false);

            let response = ui.add(text_edit);

            // Send on Enter
            let enter_pressed =
                response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            if state.is_generating {
                // Stop button
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("⏹ Stop")
                                .color(HierarchosColors::ERROR)
                                .size(13.0),
                        )
                        .fill(HierarchosColors::BG_CARD)
                        .stroke(Stroke::new(
                            1.0,
                            HierarchosColors::ERROR.linear_multiply(0.4),
                        ))
                        .corner_radius(Rounding::same(8)),
                    )
                    .clicked()
                {
                    bridge.stop_generation();
                    state.is_generating = false;
                }
            } else {
                // Send button
                let can_submit =
                    !bridge.is_generating() && !bridge.is_training() && !bridge.is_loading();
                let send_clicked = ui
                    .add_enabled(
                        can_submit,
                        egui::Button::new(
                            RichText::new("Send →")
                                .color(HierarchosColors::TEXT_ON_PRIMARY)
                                .size(13.0),
                        )
                        .fill(get_accent().primary)
                        .corner_radius(Rounding::same(8))
                        .min_size(Vec2::new(70.0, 30.0)),
                    )
                    .clicked();

                if can_submit
                    && (send_clicked || enter_pressed)
                    && !state.input_text.trim().is_empty()
                {
                    let text = state.input_text.trim().to_string();
                    state.input_text.clear();

                    // Check for slash commands
                    if text.starts_with('/') {
                        if !handle_local_command(state, bridge, &text) {
                            state.messages.push(ChatMessage {
                                role: MessageRole_::System,
                                content: format!("Command: {}", text),
                                timestamp: ChatState::now_timestamp(),
                            });
                            bridge.execute_command(text.clone());
                            let _ = state.save_active_chat();
                        }
                    } else {
                        // Regular message
                        state.messages.push(ChatMessage {
                            role: MessageRole_::User,
                            content: text.clone(),
                            timestamp: ChatState::now_timestamp(),
                        });
                        state.is_generating = true;
                        state.current_stream.clear();
                        let _ = state.save_active_chat();
                        bridge.send_message(text, state.sampling.clone());
                    }
                    state.scroll_to_bottom = true;

                    // Refocus input
                    response.request_focus();
                }
            }
        });
    });
}

fn start_new_chat_session(state: &mut ChatState, bridge: &PythonBridge) {
    if state.is_generating {
        bridge.stop_generation();
    }
    state.start_new_chat();
    if bridge.is_connected() {
        bridge.reset_chat_runtime_state(state.runtime_state_path_string());
    } else {
        state.on_status("New chat started. Backend state will reset after connecting.".to_string());
    }
}

fn draw_history_window(ctx: &egui::Context, state: &mut ChatState, bridge: &PythonBridge) {
    let mut open = state.show_history;
    let sessions = state.sessions.clone();
    let active_id = state.active_session_id.clone();
    let mut selected: Option<String> = None;

    egui::Window::new("Chats")
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(360.0)
        .default_height(420.0)
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-18.0, 86.0))
        .show(ctx, |ui| {
            ui.label(
                RichText::new("Saved Chats")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(14.0)
                    .strong(),
            );
            ui.add_space(8.0);

            ScrollArea::vertical()
                .max_height(350.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for session in &sessions {
                        let is_active = session.id == active_id;
                        let label = format!(
                            "{}\n{} | {} messages",
                            session.title, session.updated_at, session.message_count
                        );
                        let response = ui.add_enabled(
                            !state.is_generating,
                            egui::Button::new(
                                RichText::new(label)
                                    .color(if is_active {
                                        HierarchosColors::TEXT_ON_PRIMARY
                                    } else {
                                        HierarchosColors::TEXT_PRIMARY
                                    })
                                    .size(12.0),
                            )
                            .fill(if is_active {
                                get_accent().primary_dim
                            } else {
                                HierarchosColors::BG_CARD
                            })
                            .stroke(Stroke::new(
                                1.0,
                                if is_active {
                                    get_accent().primary
                                } else {
                                    HierarchosColors::BORDER_SUBTLE
                                },
                            ))
                            .corner_radius(Rounding::same(6))
                            .min_size(Vec2::new(ui.available_width(), 48.0)),
                        );

                        if response.clicked() && !is_active {
                            selected = Some(session.id.clone());
                        }
                        ui.add_space(6.0);
                    }
                });
        });

    if let Some(id) = selected {
        match state.switch_to_session(&id) {
            Ok(()) => {
                if bridge.is_connected() {
                    if state.runtime_state_exists() {
                        bridge.load_chat_runtime_state(state.runtime_state_path_string());
                    } else {
                        bridge.reset_chat_runtime_state(state.runtime_state_path_string());
                    }
                } else {
                    state.on_status(
                        "Chat reopened. Its runtime state will restore after connecting."
                            .to_string(),
                    );
                }
                open = false;
            }
            Err(err) => state.on_status(format!("Could not reopen chat: {}", err)),
        }
    }

    state.show_history = open;
}

fn handle_local_command(state: &mut ChatState, bridge: &PythonBridge, text: &str) -> bool {
    let mut parts = text.split_whitespace();
    let command = parts.next().unwrap_or("").to_ascii_lowercase();
    let value = parts.next();

    match command.as_str() {
        "/new" | "/new_chat" | "/new-chat" | "/newchat" => {
            start_new_chat_session(state, bridge);
            true
        }
        "/clear" | "/clear_chat" | "/clear-chat" => {
            if state.is_generating {
                bridge.stop_generation();
            }
            state.clear_visible_chat();
            true
        }
        "/reset" | "/reset_chat" | "/reset-chat" => {
            start_new_chat_session(state, bridge);
            true
        }
        "/temp" | "/temperature" => {
            if let Some(raw) = value {
                match raw.parse::<f32>() {
                    Ok(v) if (0.0..=2.0).contains(&v) => {
                        state.sampling.temperature = v;
                        state.on_status(format!("Temperature set to {:.2}.", v));
                    }
                    _ => state.on_status("Temperature must be between 0.0 and 2.0.".to_string()),
                }
            } else {
                state.on_status(format!(
                    "Temperature is {:.2}. Use /temp <0.0-2.0>.",
                    state.sampling.temperature
                ));
            }
            true
        }
        "/topk" | "/top-k" => {
            if let Some(raw) = value {
                match raw.parse::<u32>() {
                    Ok(v) if v <= 200 => {
                        state.sampling.top_k = v;
                        state.on_status(format!("Top-K set to {}.", v));
                    }
                    _ => state.on_status("Top-K must be an integer from 0 to 200.".to_string()),
                }
            } else {
                state.on_status(format!(
                    "Top-K is {}. Use /topk <0-200>.",
                    state.sampling.top_k
                ));
            }
            true
        }
        "/topp" | "/top-p" => {
            if let Some(raw) = value {
                match raw.parse::<f32>() {
                    Ok(v) if (0.0..=1.0).contains(&v) => {
                        state.sampling.top_p = v;
                        state.on_status(format!("Top-P set to {:.2}.", v));
                    }
                    _ => state.on_status("Top-P must be between 0.0 and 1.0.".to_string()),
                }
            } else {
                state.on_status(format!(
                    "Top-P is {:.2}. Use /topp <0.0-1.0>.",
                    state.sampling.top_p
                ));
            }
            true
        }
        "/tokens" | "/max_new_tokens" | "/max-tokens" => {
            if let Some(raw) = value {
                match raw.parse::<u32>() {
                    Ok(v) if (1..=4096).contains(&v) => {
                        state.sampling.max_new_tokens = v;
                        state.on_status(format!("Max new tokens set to {}.", v));
                    }
                    _ => state.on_status("Max new tokens must be from 1 to 4096.".to_string()),
                }
            } else {
                state.on_status(format!(
                    "Max new tokens is {}. Use /tokens <1-4096>.",
                    state.sampling.max_new_tokens
                ));
            }
            true
        }
        "/threads" | "/cpu_threads" | "/cpu-threads" => {
            let max_threads = max_cpu_threads();
            if let Some(raw) = value {
                match raw.parse::<u32>() {
                    Ok(v) if (1..=max_threads).contains(&v) => {
                        state.sampling.cpu_threads = v;
                        bridge.set_cpu_threads(v);
                        state.on_status(format!("CPU chat threads set to {}.", v));
                    }
                    _ => state.on_status(format!(
                        "CPU chat threads must be from 1 to {}.",
                        max_threads
                    )),
                }
            } else {
                state.on_status(format!(
                    "CPU chat threads is {}. Use /threads <1-{}>.",
                    state.sampling.cpu_threads, max_threads
                ));
            }
            true
        }
        "/sampling" => {
            state.show_sidebar = true;
            state.on_status("Sampling controls opened.".to_string());
            true
        }
        _ => false,
    }
}

fn draw_sampling_window(ctx: &egui::Context, state: &mut ChatState, bridge: &PythonBridge) {
    let mut open = state.show_sidebar;
    egui::Window::new("Sampling")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-18.0, 86.0))
        .show(ctx, |ui| {
            draw_sampling_sidebar(ui, state, bridge, 310.0);
        });
    state.show_sidebar = open;
}

fn draw_sampling_sidebar(
    ui: &mut egui::Ui,
    state: &mut ChatState,
    bridge: &PythonBridge,
    width: f32,
) {
    ui.vertical(|ui| {
        ui.set_width(width);

        let frame = egui::Frame::new()
            .fill(HierarchosColors::BG_SURFACE)
            .corner_radius(Rounding::same(10))
            .inner_margin(egui::Margin::same(14));

        frame.show(ui, |ui| {
            ui.label(
                RichText::new("Sampling Parameters")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(14.0)
                    .strong(),
            );
            ui.add_space(12.0);

            // Temperature
            ui.label(
                RichText::new("Temperature")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::Slider::new(&mut state.sampling.temperature, 0.0..=2.0)
                    .step_by(0.05)
                    .text(RichText::new("").size(11.0)),
            );
            ui.add_space(8.0);

            // Top-K
            ui.label(
                RichText::new("Top-K")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::Slider::new(&mut state.sampling.top_k, 0..=200)
                    .text(RichText::new("").size(11.0)),
            );
            ui.add_space(8.0);

            // Top-P
            ui.label(
                RichText::new("Top-P")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::Slider::new(&mut state.sampling.top_p, 0.0..=1.0)
                    .step_by(0.01)
                    .text(RichText::new("").size(11.0)),
            );
            ui.add_space(8.0);

            // Repetition Penalty
            ui.label(
                RichText::new("Repetition Penalty")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::Slider::new(&mut state.sampling.repetition_penalty, 1.0..=2.0)
                    .step_by(0.05)
                    .text(RichText::new("").size(11.0)),
            );
            ui.add_space(8.0);

            // Max New Tokens
            ui.label(
                RichText::new("Max New Tokens")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::Slider::new(&mut state.sampling.max_new_tokens, 16..=2048)
                    .text(RichText::new("").size(11.0)),
            );
            ui.add_space(8.0);

            // CPU threads
            ui.label(
                RichText::new("CPU Threads")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            let max_threads = max_cpu_threads();
            ui.add(
                egui::Slider::new(&mut state.sampling.cpu_threads, 1..=max_threads)
                    .text(RichText::new("").size(11.0)),
            )
            .on_hover_text(
                "Controls PyTorch CPU worker threads for chat. GPU users can leave this low; CPU users can raise it.",
            );
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        bridge.is_connected(),
                        egui::Button::new(
                            RichText::new("Apply Threads")
                                .color(HierarchosColors::TEXT_SECONDARY)
                                .size(12.0),
                        )
                        .fill(HierarchosColors::BG_CARD)
                        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                        .corner_radius(Rounding::same(6)),
                    )
                    .on_hover_text("Apply this CPU thread count to the connected backend now")
                    .clicked()
                {
                    bridge.set_cpu_threads(state.sampling.cpu_threads);
                    state.on_status(format!(
                        "CPU chat threads set to {}.",
                        state.sampling.cpu_threads
                    ));
                }
                ui.label(
                    RichText::new(format!("1-{} logical threads", max_threads))
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0),
                );
            });

            ui.add_space(16.0);

            // Reset button
            if ui
                .add(
                    egui::Button::new(
                        RichText::new("Reset to Defaults")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    )
                    .fill(HierarchosColors::BG_CARD)
                    .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                    .corner_radius(Rounding::same(6)),
                )
                .clicked()
            {
                state.sampling = SamplingParams::default();
            }
        });
    });
}

fn max_cpu_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(8)
        .max(1)
}
