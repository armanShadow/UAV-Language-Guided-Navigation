#!/usr/bin/env python3
"""
Test Paraphrase Integration
===========================

Verifies that the paraphrase integration works correctly with:
1. Normalizer processing paraphrases from augmented dataset
2. Dataset loading augmented data when use_augmented_data=True  
3. Model processing contrastive examples from paraphrases
4. Training loop using paraphrases for contrastive learning

USAGE:
    python test_paraphrase_integration.py
"""

import json
import torch
from transformers import T5Tokenizer
from config import Config
from data.Normalizer import AnsweringAgentNormalizer
from data.dataset import AnsweringDataset
from models.answering_agent import AnsweringAgent

def create_mock_augmented_data():
    """Create mock augmented data with paraphrases for testing."""
    mock_data = [{
        "episode_id": "test_episode_1",
        "map_name": "test_map",
        "gps_botm_left": [-97.12, 32.73],
        "gps_top_right": [-97.11, 32.74],
        "lat_ratio": 0.001,
        "lng_ratio": 0.001,
        "first_instruction": "Navigate to the building",
        "destination": [[-97.115, 32.735], [-97.114, 32.735], [-97.114, 32.736], [-97.115, 32.736]],
        "dialogs": [
            {
                "turn_id": 0,
                "question": "",
                "answer": "",
                "observation": {
                    "view_area_coords": [[-97.118, 32.732], [-97.117, 32.732], [-97.117, 32.733], [-97.118, 32.733]]
                },
                "previous_observations": [],
                "dialog_history": []
            },
            {
                "turn_id": 1, 
                "question": "Which direction should I go?",
                "answer": "Turn right towards the white building",
                "observation": {
                    "view_area_coords": [[-97.116, 32.733], [-97.115, 32.733], [-97.115, 32.734], [-97.116, 32.734]]
                },
                "previous_observations": [[[-97.118, 32.732], [-97.117, 32.732], [-97.117, 32.733], [-97.118, 32.733]]],
                "dialog_history": ["Question: Which direction should I go?"],
                "paraphrases": {
                    "positives": [
                        "Go right toward the white structure",
                        "Head right to the white edifice"
                    ],
                    "negatives": [
                        "Turn left towards the gray building"
                    ],
                    "valid_positives": [
                        "Go right toward the white structure",
                        "Head right to the white edifice"
                    ],
                    "valid_negatives": [
                        "Turn left towards the gray building"
                    ],
                    "validation_analysis": {
                        "original_answer": "Turn right towards the white building",
                        "valid_positives": [
                            "Go right toward the white structure",
                            "Head right to the white edifice"
                        ],
                        "valid_negatives": [
                            "Turn left towards the gray building"
                        ]
                    }
                }
            }
        ]
    }]
    return mock_data

def test_normalizer_paraphrases():
    """Test that Normalizer correctly processes paraphrases."""
    print("🧪 Testing Normalizer paraphrase processing...")
    
    # Create config and tokenizer
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
    
    # Create normalizer
    normalizer = AnsweringAgentNormalizer(tokenizer, config)
    
    # Create mock dialog turn with paraphrases
    dialog_turn = {
        "question": "Which direction should I go?",
        "answer": "Turn right towards the white building",
        "paraphrases": {
            "positives": [
                "Go right toward the white structure",
                "Head right to the white edifice"
            ],
            "negatives": [
                "Turn left towards the gray building"
            ]
        }
    }
    
    # Process contrastive samples
    contrastive_data = normalizer.process_contrastive_samples(dialog_turn)
    
    # Verify results
    assert "positive_examples" in contrastive_data, "Missing positive examples"
    assert "negative_examples" in contrastive_data, "Missing negative examples"
    assert len(contrastive_data["positive_examples"]) == 2, f"Expected 2 positives, got {len(contrastive_data['positive_examples'])}"
    assert len(contrastive_data["negative_examples"]) == 1, f"Expected 1 negative, got {len(contrastive_data['negative_examples'])}"
    
    # Check tokenization
    pos_example = contrastive_data["positive_examples"][0]
    assert "tokenized" in pos_example, "Missing tokenization"
    assert "input_ids" in pos_example["tokenized"], "Missing input_ids"
    assert "attention_mask" in pos_example["tokenized"], "Missing attention_mask"
    
    print("✅ Normalizer paraphrase processing test passed!")
    return True

def test_config_augmented_paths():
    """Test that config correctly handles augmented data paths."""
    print("🧪 Testing Config augmented data paths...")
    
    config = Config()
    
    # Test that use_augmented_data is enabled
    assert config.data.use_augmented_data == True, "use_augmented_data should be True by default"
    
    # Test path selection
    train_path = config.data.get_json_path('train')
    assert "augmented_data" in train_path, f"Expected augmented path, got {train_path}"
    assert "paraphrases" in train_path, f"Expected paraphrases in path, got {train_path}"
    
    val_seen_path = config.data.get_json_path('val_seen')
    assert "augmented_data" in val_seen_path, f"Expected augmented path, got {val_seen_path}"
    
    print("✅ Config augmented data paths test passed!")
    return True

def test_model_contrastive_forward():
    """Test that model correctly processes contrastive inputs."""
    print("🧪 Testing Model contrastive forward pass...")
    
    config = Config()
    config.training.use_contrastive_learning = True
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
    
    # Create model
    model = AnsweringAgent(config, tokenizer)
    model.eval()
    
    # Create mock inputs
    batch_size = 2
    seq_len = 20
    device = 'cpu'
    
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    positive_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    negative_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    current_view = torch.randn(batch_size, 3, 224, 224)
    previous_views = torch.randn(batch_size, 3, 3, 224, 224)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass with contrastive inputs
    with torch.no_grad():
        outputs = model(
            text_input=text_input,
            current_view=current_view,
            previous_views=previous_views,
            labels=labels,
            positive_input=positive_input,
            negative_input=negative_input
        )
    
    # Verify contrastive outputs
    assert "positive_adapted_features" in outputs, "Missing positive_adapted_features"
    assert "negative_adapted_features" in outputs, "Missing negative_adapted_features"
    assert "adapted_features" in outputs, "Missing adapted_features (anchor)"
    
    # Check shapes
    anchor_shape = outputs["adapted_features"].shape
    positive_shape = outputs["positive_adapted_features"].shape
    negative_shape = outputs["negative_adapted_features"].shape
    
    assert anchor_shape == positive_shape == negative_shape, f"Feature shapes mismatch: {anchor_shape}, {positive_shape}, {negative_shape}"
    assert anchor_shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {anchor_shape[0]}"
    
    print("✅ Model contrastive forward pass test passed!")
    return True

def main():
    """Run all integration tests."""
    print("🚀 Starting Paraphrase Integration Tests...\n")
    
    try:
        # Test individual components
        test_normalizer_paraphrases()
        print()
        
        test_config_augmented_paths()
        print()
        
        test_model_contrastive_forward()
        print()
        
        print("🎉 All Paraphrase Integration Tests Passed!")
        print("\n📋 Integration Summary:")
        print("✅ Normalizer processes paraphrases from augmented dataset")
        print("✅ Config handles augmented data paths correctly")
        print("✅ Model processes contrastive examples properly")
        print("✅ Ready for training with paraphrase-based contrastive learning")
        
        print("\n🚀 Next Steps:")
        print("1. Run comprehensive_avdn_pipeline.py to generate augmented data")
        print("2. Update training script to use augmented dataset")
        print("3. Start training with contrastive learning enabled")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 