use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidFormat(String),
    InvalidConfig(String),
    MissingTensor(String),
    Shape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NonFinite(String),
    InvalidToken(usize),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Json(err) => write!(f, "JSON error: {err}"),
            Self::InvalidFormat(msg) => write!(f, "invalid Hierarchos model file: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid Hierarchos config: {msg}"),
            Self::MissingTensor(name) => write!(f, "missing tensor '{name}'"),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{name}' has shape {actual:?}; expected {expected:?}"
            ),
            Self::NonFinite(where_) => write!(f, "non-finite value produced in {where_}"),
            Self::InvalidToken(id) => write!(f, "token id {id} is outside the model vocabulary"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
