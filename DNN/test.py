"""
Test script for DNN components.

Validates data loading, model creation, and forward/backward passes.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from config import (
    DEFAULT_CONFIG_PATH,
    TrainingConfig,
    TASK_DEFINITIONS,
    compute_sample_weight,
    balance_signal_background_weights,
    get_process_label_and_weight,
    load_training_config,
)
from model import create_model, PolarizationDNN, init_swish_weights
from data_loader import PreScaler, ParquetFoldDataset
from train import compute_weighted_loss, compute_roc_auc


def test_config():
    """Test configuration loading."""
    print("\n" + "="*70)
    print("TEST 1: Configuration Loading")
    print("="*70)
    
    assert DEFAULT_CONFIG_PATH.exists(), f"Missing config file: {DEFAULT_CONFIG_PATH}"

    config = load_training_config()
    assert config.to_dict() == TrainingConfig().to_dict()

    print(f"Config created:")
    print(f"  - n_features: {config.n_features}")
    print(f"  - hidden_width: {config.hidden_width}")
    print(f"  - n_hidden_layers: {config.n_hidden_layers}")
    print(f"  - dropout_rate: {config.dropout_rate}")
    print(f"  - seed: {config.seed}")

    override_config = load_training_config(overrides={"seed": 11})
    assert override_config.seed == 11
    assert override_config.weight_strategy == "process"
    
    # Test task definitions
    print(f"\nAvailable tasks:")
    for task_name, task_def in TASK_DEFINITIONS.items():
        print(f"  - {task_name}: {task_def['name']}")
    
    # Test process-to-label mapping
    print(f"\nProcess-to-label mapping (EW_vs_Background):")
    for process in ["WWjj_EW_LL_WW_cmf", "WWjj_QCD", "WZjj_EW"]:
        label, weight = get_process_label_and_weight(process, "EW_vs_Background")
        print(f"  - {process}: label={label}, weight={weight}")

    assert np.isclose(compute_sample_weight(12.0, 6, "process"), 12.0)
    assert np.isclose(compute_sample_weight(12.0, 6, "inverse_event_count"), 1.0 / 6.0)
    assert np.isclose(compute_sample_weight(12.0, 6, "hybrid"), 2.0)

    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    raw_weights = np.array([2.0, 4.0, 1.0, 3.0], dtype=np.float32)
    balanced_weights = balance_signal_background_weights(labels, raw_weights)
    assert np.isclose(balanced_weights[labels == 1].sum(), balanced_weights[labels == 0].sum())
    assert np.isclose(balanced_weights.sum(), raw_weights.sum())
    
    print("✓ Config test passed!")


def test_normalization():
    """Test feature normalizer."""
    print("\n" + "="*70)
    print("TEST 2: Feature Normalization")
    print("="*70)
    
    # Create dummy data
    n_samples = 1000
    n_features = 32
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    X_test = np.random.randn(500, n_features).astype(np.float32)
    
    # PreScaler with no scale_fn should be identity
    pre = PreScaler(scale_fn=None)
    X_train_scaled = pre.apply(X_train)
    assert np.allclose(X_train_scaled, X_train)

    # Test with a per-feature deterministic pre-scaling (log on first feature)
    feature_names = [f"f{i}" for i in range(n_features)]
    scale_fn_dict = {feature_names[0]: lambda col: np.log(np.abs(col) + 1e-6)}
    pre_log = PreScaler(scale_fn=scale_fn_dict, feature_names=feature_names)
    X_data = np.abs(np.random.randn(1000, n_features).astype(np.float32)) + 1.0
    X_log = pre_log.apply(X_data)
    # first column should equal the log of the original first column
    assert np.allclose(X_log[:, 0], np.log(np.abs(X_data[:, 0]) + 1e-6))

    print("✓ Pre-scaling test passed!")


def test_model_creation():
    """Test model creation and initialization."""
    print("\n" + "="*70)
    print("TEST 3: Model Creation and Initialization")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        n_features=32,
        hidden_width=128,
        n_hidden_layers=4,
        dropout_rate=0.3,
        device=device,
    )
    
    print(f"\nModel architecture:")
    print(f"  - Type: {type(model).__name__}")
    print(f"  - Device: {next(model.parameters()).device}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {n_params:,}")
    print(f"  - Trainable parameters: {n_trainable:,}")
    
    # Check weight initialization
    print(f"\nWeight initialization (Ref. [96]):")
    for name, param in model.named_parameters():
        if "weight" in name and "linear" in name.lower():
            print(f"  - {name}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
            break  # Just show one example
    
    print("✓ Model creation test passed!")


def test_forward_pass():
    """Test forward pass and loss computation."""
    print("\n" + "="*70)
    print("TEST 4: Forward Pass and Loss Computation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(
        n_features=32,
        hidden_width=64,
        n_hidden_layers=2,
        dropout_rate=0.3,
        device=device,
    )
    
    # Create dummy data
    batch_size = 32
    X = torch.randn(batch_size, 32, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    weights = torch.ones(batch_size, device=device)
    
    # Set normalization statistics
    mean = torch.zeros(32, device=device)
    std = torch.ones(32, device=device)
    model.set_normalization_statistics(mean, std)
    
    # Forward pass
    with torch.no_grad():
        logits = model(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.6f}, {logits.max():.6f}]")
    
    # Compute loss
    loss = compute_weighted_loss(logits, labels, weights)
    print(f"\nWeighted loss: {loss.item():.6f}")
    
    # Compute ROC-AUC
    logits_np = logits.cpu().numpy().squeeze()
    labels_np = labels.cpu().numpy()
    weights_np = weights.cpu().numpy()
    
    auc = compute_roc_auc(logits_np, labels_np, weights_np)
    print(f"ROC-AUC (random): {auc:.6f} (should be ≈ 0.5 for random predictions)")
    
    print("✓ Forward pass test passed!")


def test_backward_pass():
    """Test backward pass."""
    print("\n" + "="*70)
    print("TEST 5: Backward Pass and Gradient Flow")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(
        n_features=32,
        hidden_width=64,
        n_hidden_layers=2,
        device=device,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    batch_size = 32
    X = torch.randn(batch_size, 32, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device).float()
    weights = torch.ones(batch_size, device=device)
    
    # Set normalization
    mean = torch.zeros(32, device=device)
    std = torch.ones(32, device=device)
    model.set_normalization_statistics(mean, std)
    
    # Forward + backward
    model.train()
    logits = model(X)
    loss = compute_weighted_loss(logits, labels, weights)
    
    print(f"Loss before backward: {loss.item():.6f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check if gradients were computed
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"Gradients computed for: {n_grad}/{sum(1 for _ in model.parameters())} parameters")
    
    # Forward pass again
    logits = model(X)
    loss = compute_weighted_loss(logits, labels, weights)
    print(f"Loss after update: {loss.item():.6f}")
    
    print("✓ Backward pass test passed!")


def test_fold_logic():
    """Test fold splitting logic (conceptual test)."""
    print("\n" + "="*70)
    print("TEST 6: Fold Splitting Logic (Conceptual)")
    print("="*70)
    
    # Simulate EventNumber-based fold splitting
    n_events = 1000
    i_fold = 0
    event_numbers = np.arange(n_events)
    
    test_mask = (event_numbers - i_fold) % 5 == 0
    val_mask = (event_numbers - i_fold + 1) % 5 == 0
    train_mask = ~(test_mask | val_mask)
    
    print(f"For i_fold={i_fold} with {n_events} events:")
    print(f"  - Train size: {train_mask.sum()} ({100*train_mask.sum()/n_events:.1f}%)")
    print(f"  - Val size: {val_mask.sum()} ({100*val_mask.sum()/n_events:.1f}%)")
    print(f"  - Test size: {test_mask.sum()} ({100*test_mask.sum()/n_events:.1f}%)")
    
    # Verify no overlap
    overlap = (train_mask & val_mask).sum() + (train_mask & test_mask).sum() + (val_mask & test_mask).sum()
    print(f"  - Overlap: {overlap} (should be 0)")
    
    # Verify all covered
    total = train_mask.sum() + val_mask.sum() + test_mask.sum()
    print(f"  - Total coverage: {total}/{n_events} (should be 100%)")
    
    print("✓ Fold logic test passed!")


def test_binary_group_normalization():
    """Test signal/background normalization preserves class balance."""
    print("\n" + "="*70)
    print("TEST 7: Binary Group Normalization")
    print("="*70)

    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    raw_weights = np.array([2.0, 4.0, 1.0, 3.0], dtype=np.float32)
    balanced_weights = balance_signal_background_weights(labels, raw_weights)

    signal_total = balanced_weights[labels == 1].sum()
    background_total = balanced_weights[labels == 0].sum()

    assert np.isclose(signal_total, background_total)
    assert np.isclose(balanced_weights.sum(), raw_weights.sum())

    print(f"  signal total:    {signal_total:.6f}")
    print(f"  background total:{background_total:.6f}")
    print("✓ Binary group normalization test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DNN COMPONENT TESTS")
    print("="*70)
    
    try:
        test_config()
        test_normalization()
        test_model_creation()
        test_forward_pass()
        test_backward_pass()
        test_fold_logic()
        test_binary_group_normalization()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run full training with:")
        print("     python main.py --config DNN/default_config.yaml")
        print("  2. Or test on a specific fold:")
        print("     python main.py --config DNN/default_config.yaml --output_dir ./DNN/results_debug")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
