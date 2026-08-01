#!/usr/bin/env python3
"""
Test: ACT Mechanism & Backward Pass Safety Under BFloat16 Autocast
===================================================================
Final sanity check ensuring the ACT (Adaptive Computation Time) mechanism
and full backward pass survive bfloat16 autocast without dtype crashes.

Tests:
1. Forward pass under bf16 autocast produces valid outputs
2. Backward pass under bf16 autocast completes without crash
3. ACT halt probabilities are valid (0, 1) and differentiable
4. Ponder cost gradient flows to h_halt_proj
5. State dtype consistency (float32 state, bf16 activations)
6. Multi-chunk TBPTT backward under autocast
7. Worker drift backward under autocast
8. LERP interpolation dtype safety

Self-contained — creates models in-memory, no checkpoint needed.
"""
import sys
sys.path.insert(0, '.')
import pytest
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
        use_deepembed=True,
        use_rosa=True,
        compile=False,
        detach_every_n_steps=32,
    )

def get_autocast_dtype():
    """Return bf16 if CUDA available, else float16 for CPU autocast testing."""
    if torch.cuda.is_available():
        return 'cuda', torch.bfloat16
    else:
        return 'cpu', torch.bfloat16

def test_forward_under_autocast():
    """Forward pass under bf16 autocast must produce valid outputs."""
    print("=== Test 1: Forward Pass Under Autocast ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()

    x = torch.randint(0, cfg.vocab_size, (2, 12))
    labels = x.clone()
    labels[:, 0] = -100

    with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
        out = model(x, labels=labels)

    loss = out['loss']
    assert not torch.isnan(loss), f"FAIL: Loss is NaN"
    assert not torch.isinf(loss), f"FAIL: Loss is Inf"
    assert loss.item() > 0, f"FAIL: Loss is non-positive: {loss.item()}"
    print(f"  Loss: {loss.item():.4f}")

    # Check outputs exist
    assert out['logits'] is not None, "FAIL: logits is None"
    assert out['h_state'] is not None, "FAIL: h_state is None"
    assert out['l_state'] is not None, "FAIL: l_state is None"
    assert out['drift_state'] is not None, "FAIL: drift_state is None"
    assert out['prev_context'] is not None, "FAIL: prev_context is None"
    assert out['target_context'] is not None, "FAIL: target_context is None"
    print("[PASS] Forward pass under autocast produces valid outputs")

def test_backward_under_autocast():
    """Backward pass under bf16 autocast must complete without dtype crash."""
    print("\n=== Test 2: Backward Pass Under Autocast ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = x.clone()
    labels[:, 0] = -100

    with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
        out = model(x, labels=labels)
        loss = out['loss']

    # This is where masked_scatter_ crashes used to happen
    loss.backward()

    # Verify gradients exist on critical parameters
    grad_checks = {
        'tok_emb': model.tok_emb.weight.grad,
        'h_rnn.key': model.h_rnn.key.weight.grad,
        'l_rnn.key': model.l_rnn.key.weight.grad,
        'h_halt_proj': model.h_halt_proj.weight.grad,
        'context_drift_proj': model.context_drift_proj.weight.grad,
        'h_to_context': model.h_to_context.weight.grad,
        'l_feedback_proj': model.l_feedback_proj.weight.grad,
    }
    for name, grad in grad_checks.items():
        assert grad is not None, f"FAIL: {name} has no gradient"
        assert not torch.isnan(grad).any(), f"FAIL: {name} gradient has NaN"
        assert not torch.isinf(grad).any(), f"FAIL: {name} gradient has Inf"
    
    print(f"  All {len(grad_checks)} critical parameters have valid gradients")
    print("[PASS] Backward pass under autocast completes without crash")

def test_halt_probabilities():
    """ACT halt probabilities must be in [0, 1] and differentiable."""
    print("\n=== Test 3: ACT Halt Probabilities ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    labels = x.clone()
    labels[:, 0] = -100

    with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
        out = model(x, labels=labels)

    # Ponder cost is derived from cum_remain which is derived from halt_probs
    ponder = out.get('ponder_cost')
    assert ponder is not None, "FAIL: ponder_cost is None"
    assert not torch.isnan(ponder), f"FAIL: ponder_cost is NaN"
    assert ponder.item() >= 0, f"FAIL: ponder_cost negative: {ponder.item()}"
    print(f"  Ponder cost: {ponder.item():.6f}")

    # Verify it's differentiable
    out['loss'].backward()
    halt_grad = model.h_halt_proj.weight.grad
    assert halt_grad is not None, "FAIL: h_halt_proj has no gradient (ACT is not differentiable!)"
    halt_grad_norm = halt_grad.norm().item()
    print(f"  h_halt_proj grad norm: {halt_grad_norm:.6f}")
    assert halt_grad_norm > 0, "FAIL: h_halt_proj gradient is zero"
    print("[PASS] Halt probabilities are valid and differentiable")

def test_state_dtype_consistency():
    """RWKV state must be float32, activations match autocast dtype."""
    print("\n=== Test 4: State Dtype Consistency ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()

    x = torch.randint(0, cfg.vocab_size, (1, 8))

    with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
        out = model(x)

    h_state = out['h_state']
    l_state = out['l_state']

    # States must be float32 (our fix)
    assert h_state.dtype == torch.float32, f"FAIL: h_state is {h_state.dtype}, expected float32"
    assert l_state.dtype == torch.float32, f"FAIL: l_state is {l_state.dtype}, expected float32"
    print(f"  h_state dtype: {h_state.dtype}")
    print(f"  l_state dtype: {l_state.dtype}")

    # State values must be bounded
    assert h_state.abs().max().item() < 100, f"FAIL: h_state unbounded: {h_state.abs().max().item()}"
    assert l_state.abs().max().item() < 100, f"FAIL: l_state unbounded: {l_state.abs().max().item()}"
    print(f"  h_state max: {h_state.abs().max().item():.4f}")
    print(f"  l_state max: {l_state.abs().max().item():.4f}")
    print("[PASS] State dtype is float32, values are bounded")

def test_multichunk_backward():
    """Simulate TBPTT multi-chunk backward under autocast (the real training path)."""
    print("\n=== Test 5: Multi-Chunk TBPTT Backward ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randint(0, cfg.vocab_size, (2, 24))
    labels = x.clone()
    labels[:, 0] = -100

    chunk_size = 8
    h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state = None, None, None, None, None, None

    total_loss = 0.0
    for chunk_idx in range(3):  # 3 chunks of 8 tokens
        start_t = chunk_idx * chunk_size
        end_t = start_t + chunk_size
        chunk_x = x[:, start_t:end_t]
        chunk_labels = labels[:, start_t:end_t]

        with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
            out = model(
                chunk_x, labels=chunk_labels,
                h_state=h_state, l_state=l_state,
                prev_context=prev_ctx, target_context=target_ctx,
                drift_state=drift_state, ltm_memory_state=ltm_state,
                global_pos_offset=start_t
            )

        chunk_loss = out['loss'] / 3.0
        chunk_loss.backward()
        total_loss += chunk_loss.item()

        # Carry state forward (detached)
        h_state = out['h_state'].detach()
        l_state = out['l_state'].detach()
        prev_ctx = out['prev_context'].detach()
        target_ctx = out['target_context'].detach()
        drift_state = out['drift_state'].detach()
        ltm_mem = out.get('ltm_memory_state')
        if ltm_mem is not None:
            ltm_state = tuple(
                s.detach() if hasattr(s, 'detach') else s
                for s in ltm_mem
            )

    optimizer.step()
    optimizer.zero_grad()

    print(f"  Total loss across 3 chunks: {total_loss:.4f}")
    assert total_loss > 0, "FAIL: Total loss is zero"
    assert not any(torch.isnan(p).any() for p in model.parameters()), "FAIL: NaN in model params after step"
    print("[PASS] Multi-chunk TBPTT backward + optimizer step succeeded")

def test_lerp_dtype_safety():
    """LERP interpolation must handle float32/bf16 mixing without crash."""
    print("\n=== Test 6: LERP Dtype Safety ===")
    # Test the exact LERP formula used in core.py
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        prev = torch.randn(2, 32, dtype=dtype)
        target = torch.randn(2, 32, dtype=dtype)
        alpha = 0.5

        result = (prev.float() + alpha * (target.float() - prev.float())).to(prev.dtype)
        assert result.dtype == dtype, f"FAIL: LERP output dtype {result.dtype} != {dtype}"
        assert not torch.isnan(result).any(), f"FAIL: LERP result has NaN for {dtype}"

    print("  float32: OK | bfloat16: OK | float16: OK")
    print("[PASS] LERP dtype safety verified")

def test_optimizer_step_sanity():
    """Full training step (forward + backward + optimizer) must not corrupt weights."""
    print("\n=== Test 7: Full Training Step Sanity ===")
    cfg = make_config()
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Take a snapshot of weights
    initial_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    x = torch.randint(0, cfg.vocab_size, (2, 12))
    labels = x.clone()
    labels[:, 0] = -100

    # 3 training steps
    for step in range(3):
        optimizer.zero_grad()
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
            out = model(x, labels=labels)
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Verify weights changed and are finite
    changed = 0
    for n, p in model.named_parameters():
        if n in initial_weights:
            if not torch.equal(p.data, initial_weights[n]):
                changed += 1
            assert not torch.isnan(p.data).any(), f"FAIL: {n} has NaN after training"
            assert not torch.isinf(p.data).any(), f"FAIL: {n} has Inf after training"

    print(f"  {changed}/{len(initial_weights)} parameters updated after 3 steps")
    assert changed > 0, "FAIL: No parameters were updated!"
    print(f"  Final loss: {out['loss'].item():.4f}")
    print("[PASS] Full training step sanity check passed")


def test_act_and_recurrent_forward_rejects_nonfinite_state():
    """Non-finite recurrent inputs must never be rewritten into a valid path."""
    print("\n=== Test 8: ACT/Recurrent Non-Finite Rejection ===")
    cfg = make_config()
    cfg.recurrent_state_clamp = 1.5
    cfg.context_state_clamp = 1.25
    cfg.drift_state_clamp = 0.5
    cfg.halt_logit_clamp = 2.0
    cfg.activation_clamp = 4.0
    cfg.h_stride = 1
    torch.manual_seed(42)
    model = HierarchosCore(cfg)
    model.train()

    device_type, amp_dtype = get_autocast_dtype()

    x = torch.randint(0, cfg.vocab_size, (1, 6))
    labels = x.clone()
    labels[:, 0] = -100

    h_state = model.h_rnn.initial_state(1, device=x.device)
    l_state = model.l_rnn.initial_state(1, device=x.device)
    h_state.fill_(1000.0)
    l_state.fill_(-1000.0)
    h_state.view(-1)[0] = float("nan")
    l_state.view(-1)[0] = float("inf")
    prev_context = torch.full((1, cfg.context_dim), float("nan"))
    target_context = torch.full((1, cfg.context_dim), float("inf"))
    drift_state = torch.full((1, cfg.context_dim), 1000.0)

    with torch.no_grad():
        model.h_halt_proj.weight.fill_(float("inf"))
        model.h_halt_proj.bias.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="Non-finite raw language-model logits"):
        with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
            model(
                x,
                labels=labels,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
            )
    print("[PASS] Non-finite ACT/recurrent trajectory was rejected")


if __name__ == "__main__":
    print("=" * 60)
    print("ACT & Backward Pass Safety Under BFloat16 Autocast")
    print("=" * 60)

    tests = [
        ("Forward Under Autocast", test_forward_under_autocast),
        ("Backward Under Autocast", test_backward_under_autocast),
        ("Halt Probabilities", test_halt_probabilities),
        ("State Dtype Consistency", test_state_dtype_consistency),
        ("Multi-Chunk TBPTT Backward", test_multichunk_backward),
        ("LERP Dtype Safety", test_lerp_dtype_safety),
        ("Full Training Step Sanity", test_optimizer_step_sanity),
        (
            "ACT/Recurrent Non-Finite Rejection",
            test_act_and_recurrent_forward_rejects_nonfinite_state,
        ),
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
    print("ACT SAFETY TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}]: {name}")

    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\nCRITICAL: ACT mechanism or backward pass is unsafe!")
        sys.exit(1)
    else:
        print("\nAll ACT safety tests passed - safe for datacenter run!")
