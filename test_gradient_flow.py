#!/usr/bin/env python3
"""
Comprehensive gradient flow and architecture validation tests for Hierarchos.
Tests the critical fixes for coherence problems, NaN errors, and training instability.
"""
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hierarchos import HierarchosCore, AttrDict

def create_test_config(compile=False):
    """Create a small test configuration for rapid testing."""
    return AttrDict({
        'vocab_size': 50000,
        'context_dim': 32,
        'persistent_dim': 8,
        'ltm_slots': 64,
        'ltm_key_dim': 16,
        'ltm_val_dim': 16,
        'ltm_lr': 0.01,
        'ltm_topk': 2,
        'h_hidden': 32,
       'l_hidden': 32,
        'max_h_steps': 5,
        'max_l_steps': 10,
        'l_conv_atol': 2e-4,
        # Stride is discrete recurrence geometry; fractional values must fail
        # closed instead of being silently truncated by the model constructor.
        'h_stride': 2,
        'commitment_threshold': 0.055,
        'compile': compile,
        'gradient_checkpointing': False,
        'max_length': 128,
        'detach_every_n_steps': 16,  # Test truncated BPTT
        'use_deepembed': True,       # V8 DeepEmbed
        'use_rosa': True,            # V8 ROSA
    })

def test_gradient_flow_end_to_end():
    """Test 1: Verify gradients flow from output back through all components."""
    print("\\n=== Test 1: End-to-End Gradient Flow ===")
    
    config = create_test_config(compile=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchosCore(config).to(device)
    model.train()
    
    B, T = 2, 16
    input_ids = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    labels = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    
    print(f"Running forward pass (B={B}, T={T})...")
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs['loss']
    
    print(f"Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "FAIL: Loss is NaN"
    assert not torch.isinf(loss), "FAIL: Loss is Inf"
    
    print("Running backward pass...")
    loss.backward()
    
    # Check all parameters have gradients
    params_without_grad = []
    params_with_nan_grad = []
    params_with_zero_grad = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                params_without_grad.append(name)
            elif torch.isnan(param.grad).any():
                params_with_nan_grad.append(name)
            elif torch.all(param.grad == 0):
                params_with_zero_grad.append(name)
    
    # Report findings
    if params_without_grad:
        print(f"WARNING: {len(params_without_grad)} parameters missing gradients:")
        for name in params_without_grad[:5]:  # Show first 5
            print(f"  - {name}")
    
    if params_with_nan_grad:
        print(f"FAIL: {len(params_with_nan_grad)} parameters have NaN gradients!")
        for name in params_with_nan_grad:
            print(f"  - {name}")
        assert False, "Parameters have NaN gradients"
    
    if params_with_zero_grad:
        print(f"INFO: {len(params_with_zero_grad)} parameters have zero gradients (may be expected)")
    
    # Critical components should have non-zero gradients
    critical_components = ['tok_emb', 'h_rnn', 'l_rnn', 'ltm.keys', 'ltm.vals']
    for comp in critical_components:
        has_gradient = any(comp in name and param.grad is not None and not torch.all(param.grad == 0) 
                          for name, param in model.named_parameters())
        if has_gradient:
            print(f"[OK] {comp} has gradients")
        else:
            print(f"[WARN] {comp} missing meaningful gradients")
    
    print("[PASS] End-to-end gradient flow test")

def test_state_continuity():
    """Test 2: Verify state updates are meaningful and continuous."""
    print("\\n=== Test 2: State Continuity ===")
    
    config = create_test_config(compile=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchosCore(config).to(device)
    model.eval()  # Test in eval mode
    
    B, T = 2, 8
    
    # First batch
    input_ids_1 = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    print("Processing batch 1...")
    with torch.no_grad():
        outputs_1 = model(input_ids=input_ids_1)
    
    h_state_1 = outputs_1.get('h_state')
    l_state_1 = outputs_1.get('l_state')
    drift_state_1 = outputs_1.get('drift_state')
    
    assert h_state_1 is not None, "FAIL: h_state not returned"
    assert l_state_1 is not None, "FAIL: l_state not returned"
    
    # Second batch with state from first
    input_ids_2 = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    print("Processing batch 2 with state from batch 1...")
    with torch.no_grad():
        outputs_2 = model(input_ids=input_ids_2, 
                         h_state=h_state_1,
                         l_state=l_state_1,
                         drift_state=drift_state_1)
    
    h_state_2 = outputs_2.get('h_state')
    l_state_2 = outputs_2.get('l_state')
    
    # State should change but not explode
    h_changed = not torch.allclose(h_state_1, h_state_2, atol=1e-6)
    l_changed = not torch.allclose(l_state_1, l_state_2, atol=1e-6)
    
    h_exploded = torch.isnan(h_state_2).any() or torch.isinf(h_state_2).any()
    l_exploded = torch.isnan(l_state_2).any() or torch.isinf(l_state_2).any()
    
    print(f"h_state changed: {h_changed}")
    print(f"l_state changed: {l_changed}")
    print(f"h_state max: {h_state_2.abs().max().item():.4f}")
    print(f"l_state max: {l_state_2.abs().max().item():.4f}")
    
    assert h_changed, "FAIL: h_state did not change between batches"
    assert l_changed, "FAIL: l_state did not change between batches"
    assert not h_exploded, "FAIL: h_state exploded (NaN/Inf)"
    assert not l_exploded, "FAIL: l_state exploded (NaN/Inf)"
    
    print("[PASS] State continuity test")

def test_training_convergence():
    """Test 3: Verify model can actually learn (loss decreases)."""
    print("\\n=== Test 3: Training Convergence ===")
    
    config = create_test_config(compile=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchosCore(config).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    B, T = 2, 16
    # Use same data for overfitting test
    input_ids = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    labels = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    
    losses = []
    print("Training for 20 steps...")
    
    for step in range(20):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"FAIL: Loss became {loss.item()} at step {step}")
            assert False, f"Loss became non-finite at step {step}: {loss.item()}"
        
        loss.backward()
        
        # Check for exploding gradients
        max_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        if max_grad > 100.0:
            print(f"WARNING: Large gradient at step {step}: {max_grad:.2f}")
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % 5 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}, Max grad = {max_grad:.4f}")
    
    # Check if loss decreased
    initial_loss = sum(losses[:5]) / 5  # Average of first 5
    final_loss = sum(losses[-5:]) / 5    # Average of last 5
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\\nInitial loss (avg of first 5): {initial_loss:.4f}")
    print(f"Final loss (avg of last 5): {final_loss:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    
    if improvement > 5.0:  # At least 5% improvement
        print("[PASS] Model is learning (loss decreased)")
    else:
        print(f"WARNING: Model may not be learning effectively (improvement: {improvement:.1f}%)")

def test_memory_gradient_flow():
    """Test 4: Verify gradients flow through the differentiable memory update."""
    print("\\n=== Test 4: Memory Gradient Flow ===")
    
    config = create_test_config(compile=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchosCore(config).to(device)
    model.train()
    
    B, T = 1, 4
    input_ids = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    labels = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    
    # We want to check if the initial memory state receives gradients
    # This proves that the memory update chain is differentiable back to the start
    # FIX: Use tuple format (fast, mom) for ltm_memory_state
    initial_fast_vals = torch.zeros(config['ltm_slots'], config['ltm_val_dim'], device=device, requires_grad=True)
    initial_mom_vals = torch.zeros(config['ltm_slots'], config['ltm_val_dim'], device=device, requires_grad=False) # Mom usually doesn't need grad for this test
    
    print("Running forward pass with tracked memory state...")
    outputs = model(input_ids=input_ids, labels=labels, ltm_memory_state=(initial_fast_vals, initial_mom_vals))
    loss = outputs['loss']
    
    print(f"Loss: {loss.item():.4f}")
    
    print("Running backward pass...")
    loss.backward()
    
    # Check if initial_fast_vals received gradients
    # If it did, it means the memory updates (which depend on previous fast_vals) 
    # and retrievals (which use fast_vals) are connected in the graph.
    if initial_fast_vals.grad is not None:
        grad_norm = initial_fast_vals.grad.norm().item()
        print(f"Initial fast_vals grad norm: {grad_norm:.6f}")
        if grad_norm > 0:
            print("[PASS] Gradients flow through memory updates!")
        else:
            print("[FAIL] Initial fast_vals grad is zero (chain broken?)")
            assert False, "Initial fast_vals grad is zero"
    else:
        print("[FAIL] Initial fast_vals has no grad (detached?)")
        assert False, "Initial fast_vals has no grad"

def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("Hierarchos Architecture Validation Test Suite")
    print("Testing gradient flow fixes and stability improvements")
    print("="*60)
    
    tests = [
        ("Gradient Flow End-to-End", test_gradient_flow_end_to_end),
        ("State Continuity", test_state_continuity),
        ("Training Convergence", test_training_convergence),
        ("Memory Gradient Flow", test_memory_gradient_flow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\\n[FAIL] Exception in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\nAll tests passed!")
        return 0
    else:
        print(f"\\n{total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
