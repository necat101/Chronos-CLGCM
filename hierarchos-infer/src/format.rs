use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::{Error, ModelConfig, Result};

const MAGIC: &[u8; 8] = b"HRF32\0\x01\0";
const VERSION: u32 = 1;
const MAX_CONFIG_BYTES: usize = 16 * 1024 * 1024;
const MAX_TENSORS: usize = 100_000;
const MAX_NAME_BYTES: usize = 4096;
const MAX_DIMS: usize = 8;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn scalar(&self, name: &str) -> Result<f32> {
        if self.data.len() != 1 {
            return Err(Error::Shape {
                name: name.to_owned(),
                expected: vec![],
                actual: self.dims.clone(),
            });
        }
        Ok(self.data[0])
    }

    pub fn expect_shape(self, name: &str, expected: &[usize]) -> Result<Self> {
        if self.dims != expected {
            return Err(Error::Shape {
                name: name.to_owned(),
                expected: expected.to_vec(),
                actual: self.dims,
            });
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub struct ModelFile {
    pub config: ModelConfig,
    pub tensors: HashMap<String, Tensor>,
}

impl ModelFile {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::InvalidFormat("bad magic".into()));
        }
        let version = read_u32(&mut reader)?;
        if version != VERSION {
            return Err(Error::InvalidFormat(format!(
                "unsupported format version {version}; expected {VERSION}"
            )));
        }
        let config_len = read_u64(&mut reader)? as usize;
        if config_len > MAX_CONFIG_BYTES {
            return Err(Error::InvalidFormat(format!(
                "config JSON is too large ({config_len} bytes)"
            )));
        }
        let tensor_count = read_u32(&mut reader)? as usize;
        if tensor_count > MAX_TENSORS {
            return Err(Error::InvalidFormat(format!(
                "tensor count {tensor_count} exceeds limit {MAX_TENSORS}"
            )));
        }

        let mut config_bytes = vec![0u8; config_len];
        reader.read_exact(&mut config_bytes)?;
        let config: ModelConfig = serde_json::from_slice(&config_bytes)?;
        config.validate()?;

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name_len = read_u16(&mut reader)? as usize;
            if name_len == 0 || name_len > MAX_NAME_BYTES {
                return Err(Error::InvalidFormat(format!(
                    "invalid tensor name length {name_len}"
                )));
            }
            let mut name_bytes = vec![0u8; name_len];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|_| Error::InvalidFormat("tensor name is not UTF-8".into()))?;

            let ndim = read_u8(&mut reader)? as usize;
            if ndim > MAX_DIMS {
                return Err(Error::InvalidFormat(format!(
                    "tensor '{name}' has {ndim} dimensions; maximum is {MAX_DIMS}"
                )));
            }
            let mut dims = Vec::with_capacity(ndim);
            let mut element_count = 1usize;
            for _ in 0..ndim {
                let dim = read_u64(&mut reader)? as usize;
                element_count = element_count.checked_mul(dim).ok_or_else(|| {
                    Error::InvalidFormat(format!("tensor '{name}' shape overflows usize"))
                })?;
                dims.push(dim);
            }
            let stored_count = read_u64(&mut reader)? as usize;
            if stored_count != element_count {
                return Err(Error::InvalidFormat(format!(
                    "tensor '{name}' stores {stored_count} values for shape {dims:?} ({element_count} expected)"
                )));
            }
            let byte_count = element_count.checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat(format!("tensor '{name}' byte size overflows usize"))
            })?;
            let mut bytes = vec![0u8; byte_count];
            reader.read_exact(&mut bytes)?;
            let mut data = Vec::with_capacity(element_count);
            for chunk in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !value.is_finite() {
                    return Err(Error::NonFinite(format!("weight tensor '{name}'")));
                }
                data.push(value);
            }
            if tensors
                .insert(name.clone(), Tensor { dims, data })
                .is_some()
            {
                return Err(Error::InvalidFormat(format!(
                    "duplicate tensor name '{name}'"
                )));
            }
        }

        Ok(Self { config, tensors })
    }
}

pub(crate) struct TensorMap(pub(crate) HashMap<String, Tensor>);

impl TensorMap {
    pub fn take(&mut self, name: &str) -> Result<Tensor> {
        self.0
            .remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }

    pub fn take_shape(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        self.take(name)?.expect_shape(name, shape)
    }

    pub fn take_scalar(&mut self, name: &str) -> Result<f32> {
        self.take(name)?.scalar(name)
    }

    pub fn take_optional(&mut self, name: &str) -> Option<Tensor> {
        self.0.remove(name)
    }
}

fn read_u8(reader: &mut impl Read) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_u16(reader: &mut impl Read) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_fixed_width() {
        assert_eq!(MAGIC.len(), 8);
    }
}
