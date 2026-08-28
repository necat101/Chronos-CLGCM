#!/usr/bin/env python3
"""
Test: Z-Loss Boolean Indexing Under BFloat16 AMP
=================================================
Targeted test for the masked_scatter_ crash caused by boolean indexing
in the z-loss regularization path under BFloat16 autocast.

This is the exact code path that crashed at step ~2156 with:
  RuntimeError: masked_scatter_: expected self and source to have same 
  dtypes but got BFloat16 and Float

The fix wraps the z-loss block in autocast(enabled=False).
"""
import sys
sys.path.insert(0, '.')
import torch
from torch.amp import autocast
from hierarchos import HierarchosCore, AttrDict


def make_config():
    return AttrDict(
        vocab_size=500,
        context_dim=32,
        h_hidden=32,
        l_hidden=32,
        ltm_slots=64,
        ltm_key_dim=16,
        ltm_val_dim=16,
        ltm_topk=2,
        persistent_dim=16,
        max_h_steps=4,
        max_l_steps=3,
        h_stride=4,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        commitment_loss_weight=0.5,
        ponder_loss_weight=0.01,
        z_loss_weight=1e-4,  # Explicitly enable z-loss
        use_deepembed=True,
        use_rosa=True,
        compile=False,
        detach_every_n_steps=32,
    )


def test_zloss_backward_bf16():
    """Z-loss backward under BFloat16 must not crash with masked_scatter_ dtype mismatch."""
    print("=== Test: Z-Loss Backward Under BFloat16 Autocast ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = x.clone()
    labels[:, 0] = -100  # Mask first token

    # Forward + backward under bf16 autocast
    with autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
        out = model(x, labels=labels)
        loss = out['loss']

    assert loss is not None, "FAIL: loss is None"
    assert loss.item() > 0, f"FAIL: loss is non-positive: {loss.item()}"
    print(f"  Loss (with z-loss): {loss.item():.4f}")

    # THIS IS THE CRASH POINT
    loss.backward()

    # Verify gradients flowed through z-loss path
    lm_head_grad = model.lm_head.weight.grad
    assert lm_head_grad is not None, "FAIL: lm_head has no gradient"
    assert not torch.isnan(lm_head_grad).any(), "FAIL: lm_head gradient has NaN"
    print(f"  lm_head grad norm: {lm_head_grad.norm().item():.6f}")
    print("[PASS] Z-loss backward under BFloat16 completed without crash")


def test_zloss_backward_multichunk():
    """Z-loss backward across TBPTT chunks under BFloat16."""
    print("\n=== Test: Z-Loss Multi-Chunk TBPTT Under BFloat16 ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, cfg.vocab_size, (2, 24))
    labels = x.clone()
    labels[:, 0] = -100

    chunk_size = 8
    h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state = None, None, None, None, None, None

    for chunk_idx in range(3):
        start_t = chunk_idx * chunk_size
        end_t = start_t + chunk_size
        chunk_x = x[:, start_t:end_t]
        chunk_labels = labels[:, start_t:end_t]

        with autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
            out = model(
                chunk_x, labels=chunk_labels,
                h_state=h_state, l_state=l_state,
                prev_context=prev_ctx, target_context=target_ctx,
                drift_state=drift_state, ltm_memory_state=ltm_state,
                global_pos_offset=start_t,
            )

        chunk_loss = out['loss'] / 3.0
        chunk_loss.backward()

        h_state = out['h_state'].detach()
        l_state = out['l_state'].detach()
        prev_ctx = out['prev_context'].detach()
        target_ctx = out['target_context'].detach()
        drift_state = out['drift_state'].detach()
        ltm_mem = out.get('ltm_memory_state')
        if ltm_mem is not None:
            ltm_state = tuple(
                s.detach() if hasattr(s, 'detach') else s for s in ltm_mem
            )

    optimizer.step()
    optimizer.zero_grad()

    # Verify no NaN/Inf in model params
    for name, p in model.named_parameters():
        assert not torch.isnan(p.data).any(), f"FAIL: {name} has NaN after multi-chunk z-loss"
        assert not torch.isinf(p.data).any(), f"FAIL: {name} has Inf after multi-chunk z-loss"

    print("  3 chunks processed + optimizer step completed")
    print("[PASS] Multi-chunk z-loss backward under BFloat16 succeeded")


def test_zloss_zero_valid_tokens():
    """Z-loss must handle edge case of all tokens masked (-100)."""
    print("\n=== Test: Z-Loss With No Valid Tokens ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = torch.full_like(x, -100)  # ALL tokens masked

    with autocast(device_type='cpu', dtype=torch.bfloat16, enabled=True):
        out = model(x, labels=labels)
        loss = out['loss']

    assert loss is not None, "FAIL: loss is None when all tokens masked"
    loss.backward()  # Must not crash
    print(f"  Loss with all masked: {loss.item():.4f}")
    print("[PASS] Z-loss handles no-valid-tokens edge case")


if __name__ == "__main__":
    print("=" * 60)
    print("Z-Loss Boolean Indexing Safety Test (BFloat16 AMP)")
    print("=" * 60)

    tests = [
        ("Z-Loss Backward BF16", test_zloss_backward_bf16),
        ("Z-Loss Multi-Chunk BF16", test_zloss_backward_multichunk),
        ("Z-Loss No Valid Tokens", test_zloss_zero_valid_tokens),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True))
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Z-LOSS SAFETY TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}]: {name}")

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\nCRITICAL: Z-loss backward is unsafe under BFloat16 AMP!")
        sys.exit(1)
    else:
        print("\nAll z-loss tests passed - safe for datacenter training!")
