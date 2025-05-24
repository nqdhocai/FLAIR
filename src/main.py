from agents import DQNAgent, DoubleDQNAgent, PPOAgent
from enviroment import RAGTopKEnv
from trainer import train_agent 
from retriever import FaissRetriever
from dataset import load_dataset

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text embedding model")
    
    parser.add_argument("--embedding_model_id", type=str, required=True, help="Hugging Face model ID for embedding")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--dataset_id", type=str, help="Dataset ID", default="ms-marco")
    parser.add_argument("--data_dir", type=str, help="Data directory", default="../data")
    parser.add_argument("--agent_type", type=str, choices=["DQN", "DoubleDQN", "PPO"], default="PPO", help="Type of agent to use")
    parser.add_argument("--k_candidates", type=int, nargs="*", default=[i for i in range(1, 11)], help="List of k candidates for the environment")

    args = parser.parse_args()

    # Load dataset
    train_dataset, valid_dataset, test_dataset, corpus = load_dataset(args.dataset_id, args.data_dir)
    print(f"Train dataset size: {len(train_dataset):,}")

    # Initialize retriever
    retriever = FaissRetriever(embedding_model_id=args.embedding_model_id)
    retriever.build_index(corpus["text"].tolist(), corpus["id"].tolist())

    # Initialize environment
    env = RAGTopKEnv(
        corpus=corpus,
        retriever=retriever,
        k_candidates=args.k_candidates,
    )

    # Initialize agent
    state_dim = env.state_dim
    action_dim = env.action_space.n

    if args.agent_type == "DQN":
        agent = DQNAgent(
            state_dim=state_dim,
            num_actions=action_dim
        )
    elif args.agent_type == "DoubleDQN":
        agent = DoubleDQNAgent(
            state_dim=state_dim,
            num_actions=action_dim
        )
    elif args.agent_type == "PPO":
        agent = PPOAgent(
            state_dim=state_dim,
            num_actions=action_dim
        )
    else:
        raise ValueError(f"Unsupported agent type: {args.agent_type}")
    
    # Train agent
    rewards, losses = train_agent(
        env=env,
        agent=agent,
        epochs=args.epochs
    )

    # Print training results
    print(f"Training completed with {args.agent_type} agent.")
    print(f"Rewards: {rewards}")
    print(f"Losses: {losses}")