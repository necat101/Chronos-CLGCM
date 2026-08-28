use crate::format::TensorMap;
use crate::math::{dot, Matrix};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub(crate) struct MemoryHit {
    pub index: Option<usize>,
    pub value: Vec<f32>,
    pub timestamp: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct Ltm {
    slots: usize,
    key_dim: usize,
    val_dim: usize,
    keys: Matrix,
    vals: Matrix,
    timestamps: Vec<f32>,
}

impl Ltm {
    pub fn load(map: &mut TensorMap, slots: usize, key_dim: usize, val_dim: usize) -> Result<Self> {
        let keys = Matrix::from_tensor("ltm.keys", map.take("ltm.keys")?, slots, key_dim)?;
        let vals = Matrix::from_tensor("ltm.vals", map.take("ltm.vals")?, slots, val_dim)?;
        let timestamps = match map.take_optional("ltm.timestamps") {
            Some(tensor) => {
                if tensor.dims != [slots] {
                    return Err(Error::Shape {
                        name: "ltm.timestamps".into(),
                        expected: vec![slots],
                        actual: tensor.dims,
                    });
                }
                tensor.data
            }
            None => vec![0.0; slots],
        };
        Ok(Self {
            slots,
            key_dim,
            val_dim,
            keys,
            vals,
            timestamps,
        })
    }

    /// Exact read-only LTM retrieval. Hierarchos concatenates raw selected
    /// values; similarities only address slots and are not softmax weights.
    pub fn retrieve_topk(
        &self,
        query: &[f32],
        topk: usize,
        fast_vals: Option<&[f32]>,
    ) -> Result<Vec<MemoryHit>> {
        if query.len() != self.key_dim {
            return Err(Error::InvalidConfig(format!(
                "LTM query width {} does not match {}",
                query.len(),
                self.key_dim
            )));
        }
        if let Some(fast) = fast_vals {
            if fast.len() != self.slots * self.val_dim {
                return Err(Error::InvalidConfig(
                    "LTM fast-state geometry mismatch".into(),
                ));
            }
        }

        let k = topk.min(self.slots);
        let scale = 1.0 / (self.key_dim as f32).sqrt();
        // Keep only K scores while scanning the key table. This avoids a full
        // slots-sized score allocation for every generated token.
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for slot in 0..self.slots {
            let score = dot(query, self.keys.row(slot)) * scale;
            if !score.is_finite() {
                continue;
            }
            let insert_at = best
                .iter()
                .position(|&(old, _)| score > old)
                .unwrap_or(best.len());
            if insert_at < k {
                best.insert(insert_at, (score, slot));
                if best.len() > k {
                    best.pop();
                }
            } else if best.len() < k {
                best.push((score, slot));
            }
        }

        let mut hits = Vec::with_capacity(topk);
        for (_, slot) in best {
            let mut value = self.vals.row(slot).to_vec();
            if let Some(fast) = fast_vals {
                let base = slot * self.val_dim;
                for i in 0..self.val_dim {
                    value[i] += fast[base + i];
                }
            }
            hits.push(MemoryHit {
                index: Some(slot),
                value,
                timestamp: self.timestamps[slot],
            });
        }
        while hits.len() < topk {
            hits.push(MemoryHit {
                index: None,
                value: vec![0.0; self.val_dim],
                timestamp: 0.0,
            });
        }
        Ok(hits)
    }
}
