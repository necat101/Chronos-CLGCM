import random
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch

from hierarchos.utils.rosa import (
    ROSA,
    precompute_rosa_ids_for_chunks,
    rosa_async_pipeline,
    rosa_batch_parallel,
    rosa_single,
)


def _decode_sentinel(values, sentinel):
    return [(-1 if int(v) == sentinel else int(v)) for v in values]


def _wiki_rosa_reference(x):
    """Literal expanded form of the RWKV wiki ROSA reference."""
    n = len(x)
    y = [-1] * n
    s = 2 * n + 1
    b = [None] * s
    c = [-1] * s
    d = [0] * s
    e = [-1] * s
    b[0] = {}
    g = 0
    z = 1
    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u
        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]
                break
            v = c[v]
        y[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]
    return y


def _bruteforce_rosa_reference(x):
    """Slow semantic reference: longest previous suffix, newest end-position tie."""
    y = []
    for i in range(len(x)):
        best_len = 0
        best_end = -1
        for match_len in range(1, i + 2):
            suffix_start = i - match_len + 1
            suffix = x[suffix_start:i + 1]
            for end in range(match_len - 1, i):
                start = end - match_len + 1
                if x[start:end + 1] == suffix and (match_len > best_len or end > best_end):
                    best_len = match_len
                    best_end = end
        y.append(x[best_end + 1] if best_end >= 0 else -1)
    return y


class ROSACorrectnessTests(unittest.TestCase):
    def test_reference_matches_rwkv_wiki_and_semantics(self):
        cases = [
            [],
            [1],
            [1, 2, 1, 2, 3, 1, 2],
            [4, 4, 4, 5, 4, 4, 6],
            [0, 1, 0, 1, 0, 2, 0, 1, 0],
        ]
        rng = random.Random(20260522)
        cases.extend([[rng.randrange(5) for _ in range(length)] for length in range(1, 24)])

        for seq in cases:
            self.assertEqual(ROSA(seq), _wiki_rosa_reference(seq))
            self.assertEqual(ROSA(seq), _bruteforce_rosa_reference(seq))

    def test_incremental_matches_reference(self):
        rng = random.Random(1337)
        for length in range(1, 64):
            for _ in range(25):
                seq = [rng.randrange(9) for _ in range(length)]
                expected = _wiki_rosa_reference(seq)

                state = None
                actual = []
                pos = 0
                while pos < length:
                    step = rng.randrange(1, 8)
                    preds, state = rosa_single(seq[pos:pos + step], state)
                    actual.extend(preds)
                    pos += step

                self.assertEqual(actual, expected)

    def test_batch_matches_reference(self):
        batch = [
            [1, 2, 1, 2, 3, 1, 2],
            [4, 4, 4, 5, 4, 4, 6],
            [7, 8, 9],
        ]
        preds, states = rosa_batch_parallel(batch)
        self.assertEqual(preds, [ROSA(seq) for seq in batch])
        self.assertEqual([state.tokens for state in states], batch)

    def test_async_chunking_matches_reference_without_sliding(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4, 1, 2, 3]
        chunks = [3, 1, 4, 3]
        sentinel = 99
        device = torch.device("cpu")
        past = None
        states = None
        actual = []
        pos = 0

        for step in chunks:
            cur = seq[pos:pos + step]
            finalize = rosa_async_pipeline(
                torch.tensor([cur], dtype=torch.long),
                past,
                states,
                sentinel,
                device,
                rosa_max_ctx=64,
            )
            out, past, states = finalize()
            actual.extend(_decode_sentinel(out[0].tolist(), sentinel))
            pos += step

        self.assertEqual(actual, ROSA(seq))

    def test_async_resume_from_saved_past_tokens_without_automaton_state(self):
        seq = [8, 1, 8, 1, 2, 8, 1, 2, 3, 8, 1, 2]
        sentinel = 99
        device = torch.device("cpu")

        first = torch.tensor([seq[:7]], dtype=torch.long)
        finalize = rosa_async_pipeline(
            first,
            past_tokens=None,
            rosa_states=None,
            vocab_size=sentinel,
            device=device,
            rosa_max_ctx=64,
        )
        _, saved_past, first_states = finalize()
        self.assertIsNone(saved_past)
        saved_past = torch.tensor(
            [first_states[0].tokens],
            dtype=torch.long,
        )

        # Chat persistence stores the capped token window, not the Python
        # automaton object. Rebuilding from saved tokens must preserve ROSA.
        second = torch.tensor([seq[7:]], dtype=torch.long)
        finalize = rosa_async_pipeline(
            second,
            past_tokens=saved_past,
            rosa_states=None,
            vocab_size=sentinel,
            device=device,
            rosa_max_ctx=64,
        )
        out, past, states = finalize()

        expected = ROSA(seq)[7:]
        self.assertEqual(_decode_sentinel(out[0].tolist(), sentinel), expected)
        self.assertIsNone(past)
        self.assertEqual(states[0].tokens, seq)

    def test_async_full_history_alignment_across_chunks(self):
        seq = [6, 3, 6, 3, 7, 6, 3, 8, 6]
        chunks = [2, 3, 1, 3]
        cap = 4
        sentinel = 99
        device = torch.device("cpu")
        past = None
        states = None
        pos = 0

        for step in chunks:
            cur = seq[pos:pos + step]
            end = pos + len(cur)
            expected = ROSA(seq[:end])[-len(cur):]

            finalize = rosa_async_pipeline(
                torch.tensor([cur], dtype=torch.long),
                past,
                states,
                sentinel,
                device,
                rosa_max_ctx=cap,
            )
            out, past, states = finalize()
            actual = _decode_sentinel(out[0].tolist(), sentinel)

            self.assertEqual(len(actual), len(cur))
            self.assertEqual(actual, expected)
            self.assertIsNone(past)
            self.assertEqual(states[0].tokens, seq[:end])
            pos = end

    def test_async_chunk_longer_than_window_preserves_full_history_and_model_length(self):
        sentinel = 99
        finalize = rosa_async_pipeline(
            torch.tensor([[6, 3]], dtype=torch.long),
            past_tokens=None,
            rosa_states=None,
            vocab_size=sentinel,
            device=torch.device("cpu"),
            rosa_max_ctx=1,
        )
        out, past, states = finalize()

        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(_decode_sentinel(out[0].tolist(), sentinel), [-1, -1])
        self.assertIsNone(past)
        self.assertEqual(states[0].tokens, [6, 3])

    def test_versioned_bounded_mode_is_chunk_invariant_and_strictly_bounded(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1]
        sentinel = 99
        cap = 4
        expected = precompute_rosa_ids_for_chunks(
            seq,
            vocab_size=sentinel,
            chunk_size=5,
            rosa_max_ctx=cap,
            enforce_max_context=True,
        )

        for chunks in ([1] * len(seq), [3, 2, 5, 3], [7, 6]):
            past = None
            states = None
            actual = []
            pos = 0
            for width in chunks:
                cur = seq[pos:pos + width]
                finalize = rosa_async_pipeline(
                    torch.tensor([cur], dtype=torch.long),
                    past_tokens=past,
                    rosa_states=states,
                    vocab_size=sentinel,
                    device=torch.device("cpu"),
                    rosa_max_ctx=cap,
                    enforce_max_context=True,
                )
                out, past, states = finalize()
                actual.extend(out[0].tolist())
                self.assertIsNone(past)
                self.assertLessEqual(len(states[0].tokens), cap)
                pos += width
            self.assertEqual(actual, expected)

    def test_concurrent_async_batches_have_independent_result_storage(self):
        batches = [
            [[1, 2, 1, 3], [4, 4, 5, 4]],
            [[7, 8, 7, 9], [2, 3, 2, 4]],
            [[5, 6, 5, 7], [8, 8, 9, 8]],
            [[9, 1, 9, 2], [3, 4, 3, 5]],
        ]
        sentinel = 99

        def run(batch):
            finalize = rosa_async_pipeline(
                torch.tensor(batch, dtype=torch.long),
                past_tokens=None,
                rosa_states=None,
                vocab_size=sentinel,
                device=torch.device("cpu"),
            )
            out, _past, _states = finalize()
            return out.tolist()

        with ThreadPoolExecutor(max_workers=len(batches)) as pool:
            actual = list(pool.map(run, batches))

        expected = [
            [
                [sentinel if pred == -1 else pred for pred in ROSA(row)]
                for row in batch
            ]
            for batch in batches
        ]
        self.assertEqual(actual, expected)

    def test_precomputed_chunk_ids_match_async_pipeline(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4, 1, 2, 3, 5, 1]
        sentinel = 99
        chunk_size = 4
        cap = 7
        expected = precompute_rosa_ids_for_chunks(seq, sentinel, chunk_size, cap)

        device = torch.device("cpu")
        past = None
        states = None
        actual = []
        for pos in range(0, len(seq), chunk_size):
            cur = seq[pos:pos + chunk_size]
            finalize = rosa_async_pipeline(
                torch.tensor([cur], dtype=torch.long),
                past,
                states,
                sentinel,
                device,
                rosa_max_ctx=cap,
            )
            out, past, states = finalize()
            actual.extend(out[0].tolist())

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
