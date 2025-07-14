# Two-Pipeline Architecture for UAV Navigation Contrastive Learning

## 🏗️ Architecture Overview

The two-pipeline architecture separates paraphrase generation from validation, enabling modular optimization and iterative refinement for high-quality contrastive learning samples.

### 📋 Core Components

1. **Pipeline 1: Paraphrase Generation** (`paraphrase_generation_pipeline.py`)
   - Mixtral-8x7B-Instruct model for natural language paraphrasing
   - Focused prompts for positive and negative sample generation
   - Spatial term extraction and preservation guidance

2. **Pipeline 2: Validation Pipeline** (`validation_pipeline.py`)
   - Embedding-based semantic similarity assessment
   - Rule-based spatial feature preservation analysis
   - Separate validation logic for positive vs negative samples

3. **Iterative Refinement Loop** (`iterative_contrastive_pipeline.py`)
   - Combines both pipelines with iterative generation
   - Continues until sufficient valid samples are obtained
   - Performance tracking and batch processing capabilities

## 🚀 Quick Start

### Basic Usage

```python
from iterative_contrastive_pipeline import IterativeContrastivePipeline

# Initialize pipeline
pipeline = IterativeContrastivePipeline(
    generation_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    validation_model="sentence-transformers/all-MiniLM-L6-v2",
    max_iterations=3,
    target_positives=2,
    target_negatives=1
)

# Load models
pipeline.initialize()

# Generate contrastive samples
instruction = "Turn right and fly over the white building at 3 o'clock"
result = pipeline.generate_contrastive_samples(instruction)

print(f"Success: {result['success']}")
print(f"Positives: {result['positives']}")
print(f"Negatives: {result['negatives']}")
```

### Testing the Architecture

```bash
cd AnsweringAgent/src/data
python test_two_pipeline_architecture.py
```

## 📊 Pipeline Details

### Pipeline 1: Paraphrase Generation

**Purpose**: Generate high-quality paraphrases using Mixtral-8x7B-Instruct

**Key Features**:
- **Spatial Term Extraction**: Identifies landmarks, directions, clock positions
- **Strategic Prompting**: Separate prompts for positive and negative generation
- **Natural Language Focus**: Emphasizes natural diversity over template patterns

**Generation Strategies**:
- `combined`: Single prompt for both positives and negatives
- `separate`: Separate prompts for each type
- `positive_only`: Generate only positive paraphrases
- `negative_only`: Generate only negative paraphrases

**Example Output**:
```python
# Original: "Turn right and fly over the white building at 3 o'clock"
{
    'positives': [
        "Make a right turn and navigate above the white structure at 3 o'clock",
        "Go right and soar over the white edifice at 3 o'clock"
    ],
    'negatives': [
        "Turn left and fly over the gray building at 9 o'clock"
    ]
}
```

### Pipeline 2: Validation Pipeline

**Purpose**: Assess spatial accuracy and quality of generated paraphrases

**Validation Components**:
- **Embedding Similarity**: Semantic similarity using sentence transformers
- **Direction Preservation**: Clock directions, cardinal directions, spatial synonyms
- **Landmark Preservation**: Building types, spatial relationships
- **Feature Analysis**: Comprehensive spatial feature extraction

**Positive Validation Criteria**:
- Embedding similarity > 0.6
- Direction similarity > 0.7  
- Landmark similarity > 0.5
- Combined score > 0.65

**Negative Validation Criteria**:
- Embedding similarity > 0.7 (text should be similar)
- Direction OR landmark changed (spatial elements must differ)
- Maintains realistic navigation structure

**Example Validation**:
```python
{
    'is_valid': True,
    'embedding_similarity': 0.89,
    'direction_similarity': 1.0,
    'landmark_similarity': 0.85,
    'combined_score': 0.91
}
```

### Iterative Refinement Loop

**Purpose**: Combine generation and validation with quality assurance

**Process Flow**:
1. Generate paraphrases using Pipeline 1
2. Validate using Pipeline 2
3. Check if targets are met (2 positives + 1 negative)
4. If insufficient, regenerate (up to max_iterations)
5. Return best results found

**Performance Tracking**:
- Success rates per instruction
- Average iterations required
- Processing time metrics
- Validation score distributions

## 🔧 Configuration Options

### Generation Pipeline Configuration

```python
ParaphraseGenerationPipeline(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Mixtral model
    # Spatial terms automatically loaded from AVDN analysis
)
```

### Validation Pipeline Configuration

```python
ValidationPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Embedding model
    # Spatial features and thresholds configured automatically
)
```

### Iterative Pipeline Configuration

```python
IterativeContrastivePipeline(
    generation_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    validation_model="sentence-transformers/all-MiniLM-L6-v2",
    max_iterations=3,        # Maximum refinement attempts
    target_positives=2,      # Required positive samples
    target_negatives=1       # Required negative samples
)
```

## 📈 Performance Characteristics

### Expected Performance
- **Success Rate**: 85-95% for typical UAV navigation instructions
- **Processing Time**: 15-30 seconds per instruction (depending on hardware)
- **Iterations**: 1-2 iterations average for successful generations
- **Quality**: High spatial accuracy with natural language diversity

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (RTX 2080 Ti or better)
- **Memory**: 16GB+ RAM, 8GB+ VRAM for Mixtral-8x7B
- **Storage**: 50GB+ for model files

### Optimization Tips
- Use quantization (8-bit) to reduce memory usage
- Batch process multiple instructions for efficiency
- Adjust max_iterations based on quality requirements
- Monitor validation thresholds for your specific use case

## 🧪 Testing and Validation

### Individual Pipeline Tests

```bash
# Test generation pipeline only
python -c "from paraphrase_generation_pipeline import test_paraphrase_pipeline; test_paraphrase_pipeline()"

# Test validation pipeline only  
python -c "from validation_pipeline import test_validation_pipeline; test_validation_pipeline()"

# Test complete iterative pipeline
python -c "from iterative_contrastive_pipeline import test_iterative_pipeline; test_iterative_pipeline()"
```

### Comprehensive Testing

```bash
# Run all tests
python test_two_pipeline_architecture.py
```

Expected output for successful test:
```
🧪 Testing Two-Pipeline Architecture
================================================================================
🔧 Testing Paraphrase Generation Pipeline
✅ Generation model loaded successfully
🔍 Testing Validation Pipeline
✅ Validation model loaded successfully
🔄 Testing Iterative Contrastive Pipeline
✅ Both pipelines initialized successfully
🚀 Testing End-to-End Workflow
✅ Success rate: 100.0%
🎉 ALL TESTS PASSED!
```

## 📁 File Structure

```
AnsweringAgent/src/data/
├── paraphrase_generation_pipeline.py    # Pipeline 1: Generation
├── validation_pipeline.py               # Pipeline 2: Validation  
├── iterative_contrastive_pipeline.py    # Combined iterative system
├── test_two_pipeline_architecture.py    # Comprehensive testing
├── TWO_PIPELINE_ARCHITECTURE.md         # This documentation
└── enhanced_mixtral_paraphraser.py      # Legacy single-pipeline approach
```

## 🔄 Migration from Enhanced Mixtral Paraphraser

### Key Differences

| Aspect | Enhanced Paraphraser | Two-Pipeline Architecture |
|--------|---------------------|---------------------------|
| **Structure** | Monolithic class | Modular pipelines |
| **Validation** | Integrated validation | Separate validation pipeline |
| **Quality Control** | Single-pass generation | Iterative refinement |
| **Optimization** | Mixed concerns | Specialized optimization |
| **Testing** | Coupled testing | Independent component tests |
| **Maintenance** | Complex debugging | Clear separation of concerns |

### Migration Benefits

1. **Modularity**: Test and optimize each pipeline independently
2. **Quality**: Iterative refinement ensures higher success rates
3. **Flexibility**: Easy to swap validation models or strategies
4. **Reliability**: Clearer error handling and debugging
5. **Extensibility**: Simple to add new validation criteria or generation strategies

## 🚀 Dataset Processing

### Batch Processing

```python
# Process multiple instructions
instructions = [
    "Turn right and fly over the white building at 3 o'clock",
    "Go straight ahead towards the gray road",
    "Navigate to the brown house at 6 o'clock"
]

results = pipeline.process_instruction_batch(instructions)
```

### Dataset-Scale Processing

```python
# Process entire AVDN dataset
summary = pipeline.process_dataset(
    dataset_path="processed_data/train_data.json",
    output_path="augmented_data/train_contrastive.json",
    max_samples=100,  # Process subset for testing
    sample_randomly=True
)

print(f"Success rate: {summary['success_rate']:.1f}%")
print(f"Processing time: {summary['processing_time']:.1f}s")
```

## 🔍 Advanced Usage

### Custom Validation Thresholds

```python
# Modify validation pipeline thresholds
validation_pipeline = ValidationPipeline()
validation_pipeline.load_embedding_model()

# Custom positive validation
def custom_positive_validation(original, paraphrase):
    result = validation_pipeline.validate_positive_paraphrase(original, paraphrase)
    # Apply custom criteria
    result['is_valid'] = (
        result['embedding_similarity'] > 0.8 and  # Stricter similarity
        result['direction_similarity'] > 0.9      # Stricter direction preservation
    )
    return result
```

### Generation Strategy Comparison

```python
# Compare different generation strategies
generation_pipeline = ParaphraseGenerationPipeline()
generation_pipeline.load_model()

instruction = "Turn right and fly over the white building at 3 o'clock"

# Test different strategies
strategies = ["combined", "separate"]
for strategy in strategies:
    results = generation_pipeline.generate_paraphrases(instruction, strategy=strategy)
    print(f"{strategy}: {len(results['positives'])} pos, {len(results['negatives'])} neg")
```

## 📊 Monitoring and Metrics

### Real-time Statistics

```python
# Get pipeline performance metrics
stats = pipeline.get_statistics()

print(f"Total processed: {stats['total_instructions_processed']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Avg iterations: {stats['average_iterations_per_instruction']:.1f}")
print(f"Avg validation score: {stats['average_validation_score']:.3f}")
```

### Quality Analysis

```python
# Analyze validation results in detail
result = pipeline.generate_contrastive_samples(instruction)
if result['success']:
    validation_results = result['validation_results']
    
    print("Positive validation scores:")
    for pos_result in validation_results['positive_results']:
        if pos_result['is_valid']:
            print(f"  {pos_result['combined_score']:.3f}: {pos_result['text']}")
    
    print("Negative validation scores:")
    for neg_result in validation_results['negative_results']:
        if neg_result['is_valid']:
            print(f"  Spatial changed: {neg_result['spatial_changed']}: {neg_result['text']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check CUDA availability: `torch.cuda.is_available()`
   - Verify model names and accessibility
   - Ensure sufficient GPU memory

2. **Low Success Rates**
   - Reduce validation thresholds for initial testing
   - Increase max_iterations for more refinement attempts
   - Check input instruction quality and complexity

3. **Performance Issues**
   - Use 8-bit quantization for memory efficiency
   - Process smaller batches
   - Monitor GPU memory usage

### Debug Mode

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Detailed logging will show:
# - Model loading progress
# - Generation attempts
# - Validation scores
# - Iteration details
```

## 🎯 Integration with Training Pipeline

### Contrastive Learning Integration

```python
# Generate augmented dataset for training
dataset_summary = pipeline.process_dataset(
    dataset_path="processed_data/train_data.json",
    output_path="augmented_data/train_contrastive.json"
)

# Load augmented data for contrastive learning
with open("augmented_data/train_contrastive.json", 'r') as f:
    augmented_episodes = json.load(f)

for episode in augmented_episodes:
    original = episode['original_instruction']
    positives = episode['positive_paraphrases']
    negatives = episode['negative_paraphrases']
    
    # Use for contrastive learning training
    # ...
```

---

## 📚 References

- **AVDN Dataset**: Aerial Vision and Dialog Navigation dataset analysis
- **Mixtral-8x7B**: Mistral AI's mixture of experts language model
- **Sentence Transformers**: Semantic text embeddings for similarity assessment
- **Contrastive Learning**: Learning representations through positive and negative sample comparison

This architecture addresses the key challenges identified in the enhanced Mixtral paraphraser by separating concerns, enabling iterative refinement, and providing comprehensive validation for high-quality contrastive learning samples. 