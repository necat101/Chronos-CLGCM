use std::collections::HashMap;

/// Incremental Rapid Online Suffix Automaton used by Hierarchos exact memory.
///
/// This is a direct batch-1 translation of `hierarchos.utils.rosa.ROSAState` and
/// `_rosa_incremental`. The most-recent end-position tie break is preserved.
#[derive(Debug, Clone)]
pub(crate) struct RosaState {
    transitions: Vec<HashMap<usize, usize>>,
    suffix_links: Vec<isize>,
    lengths: Vec<usize>,
    endpos: Vec<isize>,
    last_state: usize,
    tokens: Vec<usize>,
    max_context: usize,
}

impl RosaState {
    pub fn new(max_context: usize) -> Self {
        Self {
            transitions: vec![HashMap::new()],
            suffix_links: vec![-1],
            lengths: vec![0],
            endpos: vec![-1],
            last_state: 0,
            tokens: Vec::new(),
            max_context,
        }
    }

    fn reset(&mut self) {
        self.transitions.clear();
        self.transitions.push(HashMap::new());
        self.suffix_links.clear();
        self.suffix_links.push(-1);
        self.lengths.clear();
        self.lengths.push(0);
        self.endpos.clear();
        self.endpos.push(-1);
        self.last_state = 0;
        self.tokens.clear();
    }

    /// Extend the automaton and return the ROSA prediction for this token.
    pub fn push(&mut self, token: usize) -> Option<usize> {
        if self.max_context > 0 && self.tokens.len() >= self.max_context {
            self.reset();
        }

        let i = self.tokens.len();
        self.tokens.push(token);

        let r = self.transitions.len();
        self.transitions.push(HashMap::new());
        self.suffix_links.push(-1);
        self.lengths.push(self.lengths[self.last_state] + 1);
        self.endpos.push(-1);

        let mut p = self.last_state as isize;
        while p != -1 && !self.transitions[p as usize].contains_key(&token) {
            self.transitions[p as usize].insert(token, r);
            p = self.suffix_links[p as usize];
        }

        if p == -1 {
            self.suffix_links[r] = 0;
        } else {
            let p_index = p as usize;
            let q = self.transitions[p_index][&token];
            if self.lengths[p_index] + 1 == self.lengths[q] {
                self.suffix_links[r] = q as isize;
            } else {
                let u = self.transitions.len();
                self.transitions.push(self.transitions[q].clone());
                self.suffix_links.push(self.suffix_links[q]);
                self.lengths.push(self.lengths[p_index] + 1);
                self.endpos.push(self.endpos[q]);

                while p != -1 {
                    let index = p as usize;
                    if self.transitions[index].get(&token).copied() != Some(q) {
                        break;
                    }
                    self.transitions[index].insert(token, u);
                    p = self.suffix_links[index];
                }
                self.suffix_links[q] = u as isize;
                self.suffix_links[r] = u as isize;
            }
        }

        self.last_state = r;

        // Predict by walking the suffix chain for the longest previous match.
        let mut v = self.last_state as isize;
        let mut prediction = None;
        while v != -1 {
            let index = v as usize;
            if self.lengths[index] > 0 && self.endpos[index] >= 0 {
                let next_pos = self.endpos[index] as usize + 1;
                if next_pos < self.tokens.len() {
                    prediction = Some(self.tokens[next_pos]);
                }
                break;
            }
            v = self.suffix_links[index];
        }

        // Preserve the Python implementation's rightmost-end-position update.
        v = self.last_state as isize;
        while v != -1 {
            let index = v as usize;
            if self.endpos[index] >= i as isize {
                break;
            }
            self.endpos[index] = i as isize;
            v = self.suffix_links[index];
        }
        prediction
    }
}

#[cfg(test)]
mod tests {
    use super::RosaState;

    #[test]
    fn repeated_pattern_predicts_next_token() {
        let mut rosa = RosaState::new(0);
        let input = [1, 2, 1, 2, 1];
        let predictions: Vec<_> = input.into_iter().map(|t| rosa.push(t)).collect();
        assert_eq!(predictions[0], None);
        assert_eq!(predictions[1], None);
        assert_eq!(predictions[2], Some(2));
        assert_eq!(predictions[3], Some(1));
        assert_eq!(predictions[4], Some(2));
    }

    #[test]
    fn bounded_context_resets_at_segment_boundary() {
        let mut rosa = RosaState::new(2);
        assert_eq!(rosa.push(7), None);
        assert_eq!(rosa.push(8), None);
        // A new deterministic segment starts before the third token.
        assert_eq!(rosa.push(7), None);
    }
}
