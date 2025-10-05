"""
Basic tests for ASAN components
"""

import torch
import pytest
import numpy as np
from typing import Dict, List

# Test imports
try:
    from models.asan_predictor import ASANPredictor, ASANConfig
    from models.trajectory_encoder import TrajectoryEncoder
    from models.wavelets import TemporalWaveletTransform
    from data.synthetic_llm_simulator import SyntheticLLMSimulator, SimulatorConfig
    from utils.config import ASANConfig as Config
except ImportError:
    pytest.skip("ASAN modules not available", allow_module_level=True)


def test_asan_config():
    """Test ASAN configuration"""
    config = ASANConfig()
    assert config.encoding_dim > 0
    assert config.decomposition_levels >= 0
    assert config.num_harm_categories > 0


def test_synthetic_simulator():
    """Test synthetic LLM simulator"""
    config = SimulatorConfig()
    simulator = SyntheticLLMSimulator(config)
    
    # Generate safe trajectory
    safe_traj = simulator.simulate_safe_trajectory(length=10)
    assert safe_traj['label'] == 'safe'
    assert len(safe_traj['generated_tokens']) == 10
    
    # Generate harmful trajectory
    harmful_traj = simulator.simulate_harmful_trajectory('jailbreak', length=10)
    assert harmful_traj['label'] == 'harmful'
    assert len(harmful_traj['generated_tokens']) == 10


def test_trajectory_encoder():
    """Test trajectory encoder"""
    input_dims = {
        'attention': 72,  # 6 features * 12 layers
        'hidden_state': 96,  # 8 features * 12 layers
        'token_prob': 7
    }
    
    encoder = TrajectoryEncoder(input_dims, encoding_dim=128)
    
    # Create dummy trajectory data
    attention_patterns = {
        0: [torch.randn(10, 10) for _ in range(5)],
        6: [torch.randn(10, 10) for _ in range(5)],
        11: [torch.randn(10, 10) for _ in range(5)]
    }
    
    hidden_states = {
        0: [torch.randn(10, 768) for _ in range(5)],
        6: [torch.randn(10, 768) for _ in range(5)],
        11: [torch.randn(10, 768) for _ in range(5)]
    }
    
    token_probs = [torch.randn(50257) for _ in range(5)]
    
    # Test encoding
    encoded = encoder(attention_patterns, hidden_states, token_probs)
    assert encoded.shape == (1, 5, 128)  # [batch, timesteps, encoding_dim]


def test_wavelet_transform():
    """Test wavelet transform"""
    wavelet = TemporalWaveletTransform(wavelet='db4', levels=3)
    
    # Create dummy trajectory
    trajectory = torch.randn(1, 10, 64)  # [batch, timesteps, features]
    
    # Test decomposition
    coeffs = wavelet(trajectory)
    assert len(coeffs) == 4  # 3 detail levels + 1 approximation
    
    # Test reconstruction
    reconstructed = wavelet.reconstruct_signal(coeffs)
    assert reconstructed.shape == trajectory.shape


def test_asan_predictor():
    """Test ASAN predictor"""
    config = ASANConfig(
        attention_dim=72,
        hidden_state_dim=96,
        token_prob_dim=7,
        encoding_dim=128,
        decomposition_levels=3
    )
    
    predictor = ASANPredictor(config)
    
    # Create dummy trajectory data
    attention_patterns = {
        0: [torch.randn(10, 10) for _ in range(5)],
        6: [torch.randn(10, 10) for _ in range(5)],
        11: [torch.randn(10, 10) for _ in range(5)]
    }
    
    hidden_states = {
        0: [torch.randn(10, 768) for _ in range(5)],
        6: [torch.randn(10, 768) for _ in range(5)],
        11: [torch.randn(10, 768) for _ in range(5)]
    }
    
    token_probs = [torch.randn(50257) for _ in range(5)]
    
    # Test prediction
    with torch.no_grad():
        prediction = predictor(attention_patterns, hidden_states, token_probs)
    
    assert 'harm_probability' in prediction
    assert 'harm_category' in prediction
    assert 'confidence' in prediction
    assert 'frequency_contributions' in prediction
    
    assert prediction['harm_probability'].shape == (1,)
    assert prediction['harm_category'].shape == (1, config.num_harm_categories)
    assert prediction['confidence'].shape == (1,)


def test_end_to_end_simulation():
    """Test end-to-end simulation"""
    # Create simulator
    simulator_config = SimulatorConfig()
    simulator = SyntheticLLMSimulator(simulator_config)
    
    # Generate dataset
    safe_trajectories, harmful_trajectories = simulator.create_balanced_dataset(
        n_samples_per_class=10
    )
    
    assert len(safe_trajectories) == 10
    assert len(harmful_trajectories) == 10
    
    # Create ASAN predictor
    asan_config = ASANConfig(
        attention_dim=72,
        hidden_state_dim=96,
        token_prob_dim=7,
        encoding_dim=128,
        decomposition_levels=3
    )
    
    predictor = ASANPredictor(asan_config)
    
    # Test on sample trajectories
    test_trajectories = safe_trajectories[:3] + harmful_trajectories[:3]
    
    predictions = []
    with torch.no_grad():
        for trajectory in test_trajectories:
            prediction = predictor(
                trajectory['attention_patterns'],
                trajectory['hidden_states'],
                trajectory['token_probs']
            )
            predictions.append(prediction)
    
    assert len(predictions) == 6
    
    # Check that predictions are reasonable
    for i, prediction in enumerate(predictions):
        harm_prob = prediction['harm_probability'].item()
        confidence = prediction['confidence'].item()
        
        assert 0 <= harm_prob <= 1
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    # Run basic tests
    print("Running ASAN basic tests...")
    
    try:
        test_asan_config()
        print("✓ ASAN config test passed")
        
        test_synthetic_simulator()
        print("✓ Synthetic simulator test passed")
        
        test_trajectory_encoder()
        print("✓ Trajectory encoder test passed")
        
        test_wavelet_transform()
        print("✓ Wavelet transform test passed")
        
        test_asan_predictor()
        print("✓ ASAN predictor test passed")
        
        test_end_to_end_simulation()
        print("✓ End-to-end simulation test passed")
        
        print("\n🎉 All tests passed! ASAN is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
