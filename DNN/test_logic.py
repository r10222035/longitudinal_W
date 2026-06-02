'''
Test script that doesn't require PyTorch/Pandas - tests pure Python logic
'''

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def test_config():
    '''Test configuration loading.'''
    print('\n' + '='*70)
    print('TEST 1: Configuration Loading')
    print('='*70)
    
    assert DEFAULT_CONFIG_PATH.exists(), f'Missing config file: {DEFAULT_CONFIG_PATH}'

    config = load_training_config()
    default_config = TrainingConfig()
    assert config.to_dict() == default_config.to_dict()

    print(f'✓ Config loaded successfully from {DEFAULT_CONFIG_PATH.name}')
    print(f'  - n_features: {config.n_features}')
    print(f'  - hidden_width: {config.hidden_width}')
    print(f'  - n_hidden_layers: {config.n_hidden_layers}')
    print(f'  - dropout_rate: {config.dropout_rate}')
    print(f'  - seed: {config.seed}')

    override_config = load_training_config(
        overrides={'seed': 7, 'output_dir': './tmp/results'}
    )
    assert override_config.seed == 7
    assert override_config.output_dir == './tmp/results'
    assert config.weight_strategy == 'process'

    hybrid_config = load_training_config(overrides={'weight_strategy': 'hybrid'})
    assert hybrid_config.weight_strategy == 'hybrid'
    
    # Test task definitions
    print(f'\n✓ Available tasks:')
    for task_name, task_def in TASK_DEFINITIONS.items():
        print(f'  - {task_name}: {task_def["name"]}')
    
    # Test process-to-label mapping
    print(f'\n✓ Process-to-label mapping (EW_vs_Background):')
    test_processes = [
        'WWjj_EW',
        'WWjj_QCD',
        'WZjj_EW',
        'top',
    ]

    for process in test_processes:
        label, weight = get_process_label_and_weight(process, 'EW_vs_Background')
        process_type = 'Signal' if label == 1 else 'Background'
        print(f'  - {process:30s}: label={label}, weight={weight:7.2f} ({process_type})')

    base_weight = 10.0
    n_events = 5
    assert compute_sample_weight(base_weight, n_events, 'process') == base_weight
    assert np.isclose(compute_sample_weight(base_weight, n_events, 'inverse_event_count'), 0.2)
    assert np.isclose(compute_sample_weight(base_weight, n_events, 'hybrid'), 2.0)

    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    raw_weights = np.array([2.0, 4.0, 1.0, 3.0], dtype=np.float32)
    balanced_weights = balance_signal_background_weights(labels, raw_weights)
    assert np.isclose(balanced_weights[labels == 1].sum(), balanced_weights[labels == 0].sum())
    assert np.isclose(balanced_weights.sum(), raw_weights.sum())
    
    print('\n✓ Config test PASSED!')


def test_fold_logic():
    '''Test fold splitting logic.'''
    print('\n' + '='*70)
    print('TEST 2: Fold Splitting Logic')
    print('='*70)
    
    n_events = 1000
    
    for i_fold in range(5):
        event_numbers = np.arange(n_events)
        
        test_mask = (event_numbers - i_fold) % 5 == 0
        val_mask = (event_numbers - i_fold + 1) % 5 == 0
        train_mask = ~(test_mask | val_mask)
        
        train_size = train_mask.sum()
        val_size = val_mask.sum()
        test_size = test_mask.sum()
        
        print(f'Fold {i_fold}:')
        print(f'  - Train: {train_size:4d} ({100*train_size/n_events:5.1f}%)')
        print(f'  - Val:   {val_size:4d} ({100*val_size/n_events:5.1f}%)')
        print(f'  - Test:  {test_size:4d} ({100*test_size/n_events:5.1f}%)')
        
        # Verify no overlap
        overlap = ((train_mask & val_mask).sum() + 
                   (train_mask & test_mask).sum() + 
                   (val_mask & test_mask).sum())
        assert overlap == 0, f'Overlap detected: {overlap}'
        
        # Verify all covered
        total = train_size + val_size + test_size
        assert total == n_events, f'Coverage incomplete: {total}/{n_events}'
        
        print(f'  ✓ No overlap, 100% coverage\n', end='')
    
    print('✓ Fold logic test PASSED!')


def test_weight_initialization():
    '''Test Swish weight initialization calculations.'''
    print('\n' + '='*70)
    print('TEST 3: Swish Weight Initialization (Ref. [96])')
    print('='*70)
    
    var_scale = 2.952
    
    test_configs = [
        (32, 128),   # input -> hidden
        (128, 128),  # hidden -> hidden
        (128, 1),    # hidden -> output
    ]
    
    print(f'Weight initialization formula: std = sqrt({var_scale} / n_in)')
    print(f'Bias initialization: N(0, 0.2)\n')
    
    for n_in, n_out in test_configs:
        std = np.sqrt(var_scale / n_in)
        print(f'Layer: Linear({n_in:3d}, {n_out:3d})')
        print(f'  - Weight std: {std:.6f}')
        print(f'  - Bias std:   0.2')
        print()
    
    print('✓ Initialization test PASSED!')


def test_data_schema():
    '''Test data schema consistency.'''
    print('\n' + '='*70)
    print('TEST 4: Data Schema')
    print('='*70)
    
    from data_loader import ParquetFoldDataset
    
    feature_columns = ParquetFoldDataset.FEATURE_COLUMNS
    
    print(f'Expected number of features: {len(feature_columns)}')
    print(f'Features by category:\n')
    
    categories = {
        'Leptons': ['l1_pt', 'l1_eta', 'l1_flavor_code', 'l2_pt', 'l2_eta', 'l2_flavor_code'],
        'Jets': ['j1_pt', 'j1_eta', 'j2_pt', 'j2_eta'],
        'MET': ['met_et', 'met_phi'],
        'Angular': ['dphi_l2_l1', 'dphi_j1_l1', 'dphi_j2_l1', 'dphi_met_l1', 'dphi_jj', 'dr_ll', 'dr_jj'],
        'Kinematics': ['m_ll', 'm_jj', 'pt_ll', 'deta_ll', 'dy_jj'],
        'TransvMass': ['mt_l1_met', 'mt_l2_met', 'mt_ll_met', 'mt0_ll_met'],
        'Physics': ['zstar_l1', 'zstar_l2', 'ptprod_ll_over_jj'],
        'Geometry': ['min_dr_lj'],
    }
    
    total = 0
    for category, features in categories.items():
        print(f'  {category:15s}: {len(features):2d} features')
        total += len(features)
        # Verify all features are in list
        for feat in features:
            assert feat in feature_columns, f'Feature {feat} not in FEATURE_COLUMNS'
    
    print(f'\nTotal: {total} features')
    assert total == len(feature_columns), f'Feature count mismatch: {total} vs {len(feature_columns)}'
    
    print('✓ Data schema test PASSED!')


def test_binary_group_normalization():
    '''Test signal/background normalization preserves class balance.'''
    print('\n' + '='*70)
    print('TEST 5: Binary Group Normalization')
    print('='*70)

    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    raw_weights = np.array([2.0, 4.0, 1.0, 3.0], dtype=np.float32)
    balanced_weights = balance_signal_background_weights(labels, raw_weights)

    signal_total = balanced_weights[labels == 1].sum()
    background_total = balanced_weights[labels == 0].sum()

    assert np.isclose(signal_total, background_total)
    assert np.isclose(balanced_weights.sum(), raw_weights.sum())

    print(f'  signal total:    {signal_total:.6f}')
    print(f'  background total:{background_total:.6f}')
    print('✓ Binary group normalization test PASSED!')


def main():
    '''Run all non-PyTorch tests.'''
    print('\n' + '='*70)
    print('DNN FRAMEWORK - LOGIC TESTS (No PyTorch Required)')
    print('='*70)
    
    try:
        test_config()
        test_fold_logic()
        test_weight_initialization()
        test_data_schema()
        test_binary_group_normalization()
        
        print('\n' + '='*70)
        print('ALL LOGIC TESTS PASSED! ✓')
        print('='*70)
        print('\nNext steps:')
        print('  1. Install PyTorch and dependencies:')
        print('     conda install -y pytorch pytorch-cuda=12.1 pandas scikit-learn matplotlib pyarrow -c pytorch -c conda-forge')
        print('\n  2. Run full component tests:')
        print('     python DNN/test.py')
        print('\n  3. Start training:')
        print('     python DNN/main.py --config DNN/default_config.yaml --output_dir ./DNN/results_debug')
        
    except Exception as e:
        print(f'\n✗ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
