# UAV Navigation Agent with Contrastive Learning

## Project Overview
The project focuses on enhancing a UAV (Unmanned Aerial Vehicle) navigation agent through contrastive learning to improve semantic understanding and generalization. The agent is trained on approximately 3,000 dialogue turns to interpret natural language instructions and navigate aerial environments.

## Core Components

### 1. Answering Agent Architecture
- **Base Model**: T5-based encoder-decoder architecture
- **Visual Processing**: Custom feature extractor for aerial imagery
- **Cross-Modal Fusion**: Bidirectional attention between text and visual features
- **Temporal Context**: Processes current and previous views with attention mechanism
- **Curriculum Learning**: Gradually reduces reliance on destination views during training

### 2. Contrastive Learning Implementation
- **ContrastiveSampleGenerator**: Generates positive (paraphrases) and negative (contradictory) examples
- **Multiple Loss Types**: Supports triplet loss, InfoNCE/NT-Xent loss, and supervised contrastive loss
- **Embedding Extraction**: Extracts embeddings from model outputs for contrastive comparison
- **Loss Integration**: Weighted combination with other losses (CE, reconstruction, distribution similarity)

### 3. Positive Example Generation
- **Language Model Paraphrasing**: Uses T5-based paraphrasing model
- **Template-Based Paraphrasing**: Navigation-specific templates with extracted terms
- **Simple Variations**: Fallback for short answers or when other methods fail
- **Semantic Similarity Filtering**: Ensures generated positives maintain meaning while being diverse

### 4. Negative Example Generation
- **Template-Based Negatives**: Contradictory navigation instructions
- **Alternative Answer Negatives**: Uses answers from other dialog turns
- **Rule-Based Negatives**: Systematic transformations (direction reversal, landmark substitution)
- **Semantic Frame Negatives**: Changes multiple elements of navigation instructions

### 5. Navigation Terminology Analysis
- **Direction Terms**: Cardinal directions, relative directions, clock positions
- **Landmark Terms**: Buildings, natural features, infrastructure elements
- **Visual Attributes**: Colors, shapes, sizes, spatial relationships
- **Complexity Analysis**: Measures spatial reasoning level, landmark references, directional complexity

### 6. Dataset Augmentation
- **Augmentation Process**: Adds positive and negative examples to each dialog turn
- **Complexity Metadata**: Adds information for curriculum learning
- **Tokenization**: Pre-processes all examples for efficient training

### 7. Training Integration
- **Loss Weighting**: Dynamic weight scheduling for different loss components
- **Mixed Precision Training**: Supports faster training with lower memory usage
- **Gradient Accumulation**: Enables effective training with larger batch sizes
- **EMA Model Averaging**: Maintains a moving average of model weights for stable evaluation

## Technical Details
- **Framework**: PyTorch with Transformers library
- **Visual Backbone**: Custom feature extractor for aerial imagery
- **Language Model**: T5-based encoder-decoder
- **Sentence Embeddings**: Uses sentence-transformers models for semantic similarity
- **Distributed Training**: Supports multi-GPU training with DDP
- **Memory Optimization**: Gradient bucketing, mixed precision, and efficient data loading

## Advanced Approaches
- **Back-translation**: Alternative paraphrasing technique
- **Language Model Paraphrasing**: T5/BART-based paraphrase generation
- **Semantic Frame Transformation**: Systematic alteration of navigation elements
- **Dataset-aware Negative Mining**: Selects challenging negatives based on dataset patterns
- **Clustering-based Template Selection**: Optimizes template usage based on instruction types 