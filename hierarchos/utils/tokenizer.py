"""Tokenizer identities shared by training, evaluation, and inference.

A matching vocabulary size is not enough for model compatibility: two
tokenizers can assign different IDs to the same text while exposing the same
number of tokens.  Persist and compare a content fingerprint so inference
cannot silently feed a checkpoint a different token language.
"""

import hashlib
import json


_BEHAVIORAL_TOKENIZER_ATTRIBUTES = (
    "do_lower_case",
    "strip_accents",
    "tokenize_chinese_chars",
    "split_special_tokens",
    "clean_up_tokenization_spaces",
    "add_prefix_space",
)


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8", errors="surrogatepass")


def _tokenizer_behavior_payload(tokenizer):
    """Collect tokenizer rules that are not recoverable from ``get_vocab``.

    A vocabulary maps already-produced tokens to IDs. It does not describe how
    text becomes those tokens: normalizers, pre-tokenizers, BPE merge ranks,
    SentencePiece models, and AddedToken flags can all change the resulting ID
    sequence while leaving ``get_vocab()`` byte-for-byte identical.
    """
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    }

    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        try:
            serialized = backend_to_str()
            try:
                serialized = json.loads(serialized)
            except (TypeError, ValueError, json.JSONDecodeError):
                serialized = str(serialized)
            payload["backend_tokenizer"] = serialized
        except Exception as exc:
            raise ValueError(
                "Could not serialize fast-tokenizer behavior for identity validation."
            ) from exc

    sentencepiece = getattr(tokenizer, "sp_model", None)
    serialize_sentencepiece = getattr(sentencepiece, "serialized_model_proto", None)
    if callable(serialize_sentencepiece):
        try:
            model_bytes = bytes(serialize_sentencepiece())
            payload["sentencepiece_sha256"] = hashlib.sha256(model_bytes).hexdigest()
        except Exception as exc:
            raise ValueError(
                "Could not serialize SentencePiece behavior for identity validation."
            ) from exc

    bpe_ranks = getattr(tokenizer, "bpe_ranks", None)
    if isinstance(bpe_ranks, dict):
        try:
            normalized_ranks = [
                ([str(part) for part in pair], int(rank))
                for pair, rank in bpe_ranks.items()
            ]
            normalized_ranks.sort(key=lambda item: (item[1], item[0]))
            payload["bpe_ranks"] = normalized_ranks
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Could not serialize slow-tokenizer BPE merge ranks for identity validation."
            ) from exc

    added_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added_decoder, dict):
        normalized_added_tokens = []
        for token_id, token in added_decoder.items():
            try:
                normalized_token_id = int(token_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Could not serialize AddedToken identifiers for identity validation."
                ) from exc
            entry = {
                "id": normalized_token_id,
                "content": str(getattr(token, "content", token)),
            }
            for name in ("single_word", "lstrip", "rstrip", "normalized", "special"):
                value = getattr(token, name, None)
                if isinstance(value, bool):
                    entry[name] = value
            normalized_added_tokens.append(entry)
        normalized_added_tokens.sort(key=lambda entry: entry["id"])
        payload["added_tokens_decoder"] = normalized_added_tokens

    attributes = {}
    for name in _BEHAVIORAL_TOKENIZER_ATTRIBUTES:
        value = getattr(tokenizer, name, None)
        if isinstance(value, (bool, int, float, str)) or value is None:
            attributes[name] = value
    if attributes:
        payload["attributes"] = attributes
    return payload


def tokenizer_vocab_size(tokenizer) -> int:
    try:
        return int(len(tokenizer))
    except Exception:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        return 50257


def tokenizer_identity(tokenizer):
    """Return the tokenizer content identity used by training checkpoints."""
    hasher = hashlib.sha256()
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": tokenizer_vocab_size(tokenizer),
    }
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path:
        identity["name_or_path"] = str(name_or_path)
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None
    if isinstance(vocab, dict):
        for token, token_id in sorted(
            vocab.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        ):
            encoded = str(token).encode("utf-8", errors="surrogatepass")
            hasher.update(len(encoded).to_bytes(4, "little", signed=False))
            hasher.update(encoded)
            hasher.update(int(token_id).to_bytes(8, "little", signed=True))
    else:
        # This is deliberately weaker than the normal vocabulary fingerprint,
        # but remains deterministic for tokenizer implementations that do not
        # expose get_vocab().
        hasher.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
    special_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_map, dict):
        normalized_specials = {
            str(key): str(value)
            for key, value in sorted(special_map.items())
        }
        hasher.update(
            json.dumps(
                normalized_specials,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8", errors="surrogatepass")
        )
        identity["special_tokens_map"] = normalized_specials
    identity["sha256"] = hasher.hexdigest()
    # Keep ``sha256`` byte-for-byte compatible with existing checkpoints. The
    # v2 digest adds the text->token behavior that a vocabulary-only identity
    # cannot prove. Exact-resume comparison explicitly treats absence of this
    # field as legacy metadata, while v2 artifacts require a v2 match.
    behavior_hasher = hashlib.sha256()
    behavior_hasher.update(b"hierarchos-tokenizer-behavior-v2\0")
    behavior_hasher.update(identity["sha256"].encode("ascii"))
    behavior_hasher.update(_canonical_json_bytes(_tokenizer_behavior_payload(tokenizer)))
    identity["behavior_sha256_v2"] = behavior_hasher.hexdigest()
    return identity


def checkpoint_tokenizer_identity(checkpoint_metadata):
    """Extract the training tokenizer identity from retained checkpoint metadata."""
    if not isinstance(checkpoint_metadata, dict):
        return None
    direct_identity = checkpoint_metadata.get("tokenizer_identity")
    if not isinstance(direct_identity, dict):
        direct_identity = None
    run_identity = checkpoint_metadata.get("run_identity")
    run_tokenizer_identity = (
        run_identity.get("tokenizer")
        if isinstance(run_identity, dict)
        else None
    )
    if not isinstance(run_tokenizer_identity, dict):
        run_tokenizer_identity = None

    # Merge/export checkpoints may add a directly authenticated tokenizer
    # fingerprint even when a legacy base had no exact-run identity.  If both
    # copies exist, require them to describe the same token language instead of
    # silently choosing whichever field was inspected first.
    if direct_identity is not None and run_tokenizer_identity is not None:
        direct_vocab = direct_identity.get("vocab_size")
        run_vocab = run_tokenizer_identity.get("vocab_size")
        if (
            direct_vocab is not None
            and run_vocab is not None
            and int(direct_vocab) != int(run_vocab)
        ):
            raise ValueError(
                "Conflicting tokenizer vocabulary sizes in checkpoint metadata."
            )
        direct_digest = direct_identity.get("sha256")
        run_digest = run_tokenizer_identity.get("sha256")
        if (
            isinstance(direct_digest, str)
            and direct_digest.strip()
            and isinstance(run_digest, str)
            and run_digest.strip()
            and direct_digest.strip().lower() != run_digest.strip().lower()
        ):
            raise ValueError(
                "Conflicting tokenizer content fingerprints in checkpoint metadata."
            )
        direct_behavior_digest = direct_identity.get("behavior_sha256_v2")
        run_behavior_digest = run_tokenizer_identity.get("behavior_sha256_v2")
        if (
            isinstance(direct_behavior_digest, str)
            and direct_behavior_digest.strip()
            and isinstance(run_behavior_digest, str)
            and run_behavior_digest.strip()
            and direct_behavior_digest.strip().lower()
            != run_behavior_digest.strip().lower()
        ):
            raise ValueError(
                "Conflicting tokenizer behavior fingerprints in checkpoint metadata."
            )
    if direct_identity is not None and run_tokenizer_identity is not None:
        # Never let a redundant legacy copy downgrade a behavior-authenticated
        # identity. Both legacy vocab digests were checked above; prefer the
        # copy that additionally binds normalization/merge/tokenization rules.
        if (
            not direct_identity.get("behavior_sha256_v2")
            and run_tokenizer_identity.get("behavior_sha256_v2")
        ):
            return run_tokenizer_identity
    return direct_identity or run_tokenizer_identity


def validate_inference_tokenizer_identity(tokenizer, checkpoint_metadata):
    """Fail closed when a checkpoint proves that token IDs do not match.

    Returns ``True`` when a strong saved fingerprint was checked and ``False``
    for legacy checkpoints that only permit a vocabulary-size check.
    """
    saved = checkpoint_tokenizer_identity(checkpoint_metadata)
    if not isinstance(saved, dict):
        return False

    saved_vocab_size = saved.get("vocab_size")
    current = tokenizer_identity(tokenizer)
    if (
        saved_vocab_size is not None
        and int(saved_vocab_size) != int(current["vocab_size"])
    ):
        raise ValueError(
            "Tokenizer vocabulary size disagrees with the tokenizer recorded "
            f"during training ({current['vocab_size']} != {int(saved_vocab_size)})."
        )

    saved_digest = saved.get("sha256")
    if not isinstance(saved_digest, str) or not saved_digest.strip():
        return False
    if saved_digest.strip().lower() != current["sha256"].lower():
        raise ValueError(
            "Tokenizer content fingerprint disagrees with the tokenizer used "
            "during training. A same-size vocabulary with different token IDs "
            "cannot preserve training/inference logit parity."
        )
    saved_behavior_digest = saved.get("behavior_sha256_v2")
    if saved_behavior_digest is None:
        return True
    if (
        not isinstance(saved_behavior_digest, str)
        or len(saved_behavior_digest.strip()) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in saved_behavior_digest.strip()
        )
    ):
        raise ValueError("Saved tokenizer behavior fingerprint is not a valid SHA-256 digest.")
    if saved_behavior_digest.strip().lower() != current["behavior_sha256_v2"].lower():
        raise ValueError(
            "Tokenizer behavior fingerprint disagrees with the tokenizer used "
            "during training. Normalization, pre-tokenization, merge rules, or "
            "AddedToken behavior changed despite a matching vocabulary."
        )
    return True
