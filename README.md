# FLAIR: Flexible Allocation of Informative Retrieval via Reinforcement Learning

FLAIR is a reinforcement learning framework for optimizing Retrieval-Augmented Generation (RAG) systems. It trains RL agents to dynamically select the optimal number of documents (top-k) for retrieval, improving both retrieval quality and computational efficiency.

## 🎯 Overview

Traditional RAG systems use a fixed top-k value for document retrieval, which may not be optimal for all queries. FLAIR addresses this limitation by:

- **Adaptive Top-k Selection**: Uses reinforcement learning to learn query-specific optimal k values
- **Multi-Agent Support**: Implements DQN, Double DQN, and PPO algorithms
- **Efficient Retrieval**: FAISS-based similarity search with embedding models
- **Comprehensive Evaluation**: Combines recall@k, MRR@k, and efficiency metrics

## 🏗️ Architecture

```
Query → Embedding → Agent → Top-k Selection → Retrieved Documents
  ↓         ↓         ↓           ↓              ↓
Dataset → Retriever → Environment → Action → Reward (Recall+MRR-Cost)
```

### Key Components

- **Environment** (`RAGTopKEnv`): Gym-based RL environment for top-k selection
- **Agents**: DQN, Double DQN, and PPO implementations
- **Retriever**: FAISS-based document retrieval with configurable embedding models
- **Trainer**: Unified training system with early stopping and performance monitoring

## 📋 Requirements

- Python 3.8+
- PyTorch (CUDA support recommended)
- Transformers (Hugging Face)
- FAISS
- OpenAI Gym

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/nqdhocai/FLAIR
cd FLAIR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Format

FLAIR expects datasets in the following structure:
```
data/
  dataset_name/
    ├── corpus.csv          # Document collection (id, text)
    ├── question_train.csv  # Training queries (qid, question)
    ├── question_valid.csv  # Validation queries (qid, question)
    ├── question_test.csv   # Test queries (qid, question)
    └── ground_truth.json   # Relevance labels {qid: [doc_ids]}
```

### Supported Datasets
- MS-MARCO (default)
- Custom datasets following the above format

## 🎮 Usage

### Basic Training

```bash
python src/main.py \
    --embedding_model_id "sentence-transformers/all-MiniLM-L6-v2" \
    --agent_type "PPO" \
    --epochs 50 \
    --dataset_id "ms-marco" \
    --data_dir "./data"
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding_model_id` | str | **Required** | Hugging Face model ID for embeddings |
| `--agent_type` | str | "PPO" | RL agent type: "DQN", "DoubleDQN", "PPO" |
| `--epochs` | int | 5 | Number of training epochs |
| `--dataset_id` | str | "ms-marco" | Dataset directory name |
| `--data_dir` | str | "../data" | Path to data directory |
| `--k_candidates` | list | [1,2,...,10] | Possible k values for top-k selection |

### Example Usage

```bash
# Train PPO agent with BERT embeddings
python src/main.py \
    --embedding_model_id "sentence-transformers/all-mpnet-base-v2" \
    --agent_type "PPO" \
    --epochs 100 \
    --k_candidates 1 2 3 4 5 6 7 8 9 10

# Train DQN agent for quick experiments
python src/main.py \
    --embedding_model_id "sentence-transformers/all-MiniLM-L6-v2" \
    --agent_type "DQN" \
    --epochs 25
```

## 🧠 RL Agents

### 1. DQN (Deep Q-Network)
- **Architecture**: 3-layer MLP with dropout
- **Features**: Experience replay, target network
- **Use Case**: Baseline discrete action selection

### 2. Double DQN
- **Architecture**: Same as DQN with double Q-learning
- **Features**: Reduces overestimation bias
- **Use Case**: More stable training than vanilla DQN

### 3. PPO (Proximal Policy Optimization)
- **Architecture**: Actor-Critic with shared layers
- **Features**: Policy gradient with clipping
- **Use Case**: Better sample efficiency and stability

## 🎯 Reward Function

The reward function balances retrieval quality and efficiency:

```
R = α·recall@k + (1-α)·MRR@k - γ·(k/K_max)·exp(−λ·max(0, k − k∗))
```

With additional components:
- **Soft penalty**: Exponential decay for k > k*
- **No-overlap penalty**: Penalty when no relevant documents are retrieved
- **Normalization**: Rewards clamped to [-1, +1]

### Parameters
- `α = 0.6`: Balance between recall and MRR
- `γ = 0.1`: Cost coefficient for efficiency
- `β₀ = 0.5`: Adaptive penalty strength
- `no_overlap_penalty = 0.2`: Penalty for zero hits

## 📈 Monitoring & Evaluation

### Training Metrics
- **Average Reward**: Per-epoch reward progression
- **Loss**: Agent-specific loss curves
- **Early Stopping**: Automatic stopping on plateau

### Performance Analysis
```python
from src.trainer import plot_training_stats

# Plot training results
plot_training_stats(rewards, losses, agent_name="PPO")
```

## 🔧 Customization

### Custom Embedding Models
```python
# Use any Hugging Face model
--embedding_model_id "microsoft/DialoGPT-medium"
--embedding_model_id "facebook/dpr-question_encoder-single-nq-base"
```

### Custom Reward Functions
Modify `compute_rewards()` in `src/enviroment.py`:
```python
def compute_rewards(self, retrieved_ids, gold_ids, **kwargs):
    # Implement custom reward logic
    return rewards
```

### Custom Agents
Extend base classes in `src/agents/`:
```python
class CustomAgent:
    def select_action(self, state):
        # Custom action selection
        pass
    
    def update(self):
        # Custom learning update
        pass
```

## 📁 Project Structure

```
FLAIR/
├── src/
│   ├── main.py              # Main training script
│   ├── enviroment.py        # RAG environment
│   ├── trainer.py           # Unified training system
│   ├── dataset.py           # Data loading utilities
│   ├── agents/
│   │   ├── dqn.py          # DQN implementation
│   │   ├── ddqn.py         # Double DQN implementation
│   │   └── ppo.py          # PPO implementation
│   └── retriever/
│       ├── retriever.py    # FAISS-based retrieval
│       └── embedder.py     # Embedding model wrapper
├── data/                    # Dataset directory
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in embedding model
   - Use smaller embedding models
   - Process datasets in chunks

2. **Slow Training**
   - Enable CUDA if available
   - Use `cache_precompute=True` in environment
   - Reduce dataset size for experimentation

3. **Poor Convergence**
   - Adjust learning rates
   - Tune reward function parameters
   - Increase training epochs

### Performance Tips

- **GPU Usage**: Ensure PyTorch uses GPU for embeddings
- **Memory Optimization**: Use gradient checkpointing for large models
- **Batch Processing**: Optimize batch sizes based on available memory

## 📚 Citation

If you use FLAIR in your research, please cite:

```bibtex
@article{flair2025,
  title={FLAIR: Flexible Allocation of Informative Retrieval via Reinforcement Learning},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- FAISS team for efficient similarity search
- OpenAI Gym for RL environment framework
- PyTorch team for deep learning infrastructure