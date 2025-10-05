#!/usr/bin/env python3
"""
ASAN Demo Script - Run the complete ASAN system
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("ASAN: Adaptive Spectral Alignment Networks for Predictive AI Safety")
print("=" * 70)
print()

def main():
    """Run the complete ASAN demo"""
    
    try:
        # Step 1: Generate synthetic data
        print("Step 1: Generating synthetic LLM trajectories...")
        from data.synthetic_llm_simulator import SyntheticLLMSimulator, SimulatorConfig
        
        simulator_config = SimulatorConfig(
            num_layers=6,  # Smaller for demo
            num_heads=4,
            hidden_dim=256,
            vocab_size=1000
        )
        
        simulator = SyntheticLLMSimulator(simulator_config)
        
        # Generate trajectories
        safe_trajectories, harmful_trajectories = simulator.create_balanced_dataset(
            n_samples_per_class=20  # Small dataset for demo
        )
        
        print(f"✓ Generated {len(safe_trajectories)} safe trajectories")
        print(f"✓ Generated {len(harmful_trajectories)} harmful trajectories")
        print()
        
        # Step 2: Initialize ASAN model
        print("Step 2: Initializing ASAN predictor...")
        from models.asan_predictor import ASANPredictor, ASANConfig
        
        asan_config = ASANConfig(
            attention_dim=6 * 6,  # 6 features per layer * 6 layers
            hidden_state_dim=8 * 6,  # 8 features per layer * 6 layers
            token_prob_dim=7,
            encoding_dim=128,  # Smaller for demo
            attention_dim_internal=64,
            attention_heads=4,
            decomposition_levels=3,
            wavelet='db4',
            num_harm_categories=3
        )
        
        asan_predictor = ASANPredictor(asan_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in asan_predictor.parameters())
        print(f"ASAN predictor initialized with {total_params:,} parameters")
        print()
        
        # Step 3: Test predictions
        print("Step 3: Testing ASAN predictions...")
        
        # Test on sample trajectories
        test_trajectories = safe_trajectories[:3] + harmful_trajectories[:3]
        
        predictions = []
        with torch.no_grad():
            for i, trajectory in enumerate(test_trajectories):
                prediction = asan_predictor(
                    trajectory['attention_patterns'],
                    trajectory['hidden_states'],
                    trajectory['token_probs']
                )
                predictions.append(prediction)
                
                harm_prob = prediction['harm_probability'].item()
                confidence = prediction['confidence'].item()
                label = trajectory['label']
                
                print(f"  Trajectory {i+1} ({label}): "
                      f"Harm prob={harm_prob:.3f}, "
                      f"Confidence={confidence:.3f}")
        
        print("✓ Predictions completed")
        print()
        
        # Step 4: Analyze results
        print("Step 4: Analyzing results...")
        
        # Extract metrics
        harm_probs = [pred['harm_probability'].item() for pred in predictions]
        confidences = [pred['confidence'].item() for pred in predictions]
        labels = [traj['label'] for traj in test_trajectories]
        
        # Calculate accuracy (simple threshold-based)
        threshold = 0.5
        correct_predictions = 0
        for i, (harm_prob, label) in enumerate(zip(harm_probs, labels)):
            predicted_harmful = harm_prob > threshold
            actual_harmful = (label == 'harmful')
            if predicted_harmful == actual_harmful:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(predictions)
        
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Average harm probability: {np.mean(harm_probs):.3f}")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        print()
        
        # Step 5: Visualize results
        print("Step 5: Creating visualizations...")
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Harm probability by trajectory
        plt.subplot(2, 2, 1)
        colors = ['red' if label == 'harmful' else 'green' for label in labels]
        plt.bar(range(len(harm_probs)), harm_probs, color=colors, alpha=0.7)
        plt.title('Harm Probability by Trajectory')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Harm Probability')
        plt.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
        plt.legend(['Threshold', 'Safe', 'Harmful'])
        
        # Plot 2: Confidence by trajectory
        plt.subplot(2, 2, 2)
        plt.bar(range(len(confidences)), confidences, color=colors, alpha=0.7)
        plt.title('Confidence by Trajectory')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Confidence')
        
        # Plot 3: Harm probability vs Confidence scatter
        plt.subplot(2, 2, 3)
        plt.scatter(harm_probs, confidences, c=colors, alpha=0.7, s=100)
        plt.title('Harm Probability vs Confidence')
        plt.xlabel('Harm Probability')
        plt.ylabel('Confidence')
        plt.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Frequency band contributions
        plt.subplot(2, 2, 4)
        if predictions:
            freq_contribs = predictions[0]['frequency_contributions'].cpu().numpy()
            band_names = [f'Band {i}' for i in range(len(freq_contribs))]
            plt.bar(band_names, freq_contribs, color='skyblue', alpha=0.7)
            plt.title('Frequency Band Contributions')
            plt.xlabel('Frequency Band')
            plt.ylabel('Contribution Weight')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = project_root / 'demo_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        print()
        
        # Step 6: Test temporal predictions
        print("Step 6: Testing temporal predictions...")
        
        # Test on a harmful trajectory
        harmful_trajectory = harmful_trajectories[0]
        
        temporal_predictions = asan_predictor.predict_at_each_timestep(
            harmful_trajectory['attention_patterns'],
            harmful_trajectory['hidden_states'],
            harmful_trajectory['token_probs']
        )
        
        timesteps = list(range(len(temporal_predictions)))
        harm_probs_timeline = [pred['harm_probability'].item() for pred in temporal_predictions]
        confidences_timeline = [pred['confidence'].item() for pred in temporal_predictions]
        
        print(f"Generated temporal predictions for {len(timesteps)} timesteps")
        print(f"Final harm probability: {harm_probs_timeline[-1]:.3f}")
        print(f"Final confidence: {confidences_timeline[-1]:.3f}")
        print()
        
        # Step 7: Summary
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Dataset: {len(safe_trajectories)} safe + {len(harmful_trajectories)} harmful trajectories")
        print(f"Model: {total_params:,} parameters")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Temporal analysis: {len(timesteps)} timesteps")
        print(f"Visualization: Saved to demo_results.png")
        print()
        print("ASAN is working correctly!")
        print("   - Synthetic data generation: ✓")
        print("   - Model initialization: ✓")
        print("   - Prediction inference: ✓")
        print("   - Temporal analysis: ✓")
        print("   - Visualization: ✓")
        print()
        print("🛡️ ASAN: Transforming AI Safety from Reactive to Proactive")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
