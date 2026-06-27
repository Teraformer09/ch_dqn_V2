import matplotlib.pyplot as plt
import pandas as pd
from chdqn.config import ChDQNConfig
from chdqn.trainer_rl import CartPoleRLRunner
from pathlib import Path

def run_learning_proof():
    print("Starting Empirical Learning Proof (300 episodes, V2, Gaussian)...")
    
    config = ChDQNConfig(
        train_episodes=300,
        max_steps_per_episode=200,
        batch_size=32,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,  # Decay over ~25-50 episodes depending on length
        min_replay_sequences=32,
        reward_scale=1.0,  # Use raw rewards for clearer interpretation (1.0 per step)
        seed=42
    )

    runner = CartPoleRLRunner(config, model_type="V2", noise_type="gaussian")
    log_path = "results/learning_proof_v2.csv"
    
    # Run training
    summary = runner.train(log_path=log_path, seed=42)
    
    print(f"\nTraining Complete.")
    print(f"Final Evaluation Reward: {summary.evaluation.mean_reward:.2f}")
    
    # Generate Plots
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Reward Curve
    axes[0, 0].plot(df['episode'], df['reward'].rolling(window=10).mean(), label='MA-10 Reward', color='blue')
    axes[0, 0].set_title("Learning Curve (Moving Average)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)

    # 2. TD Variance
    axes[0, 1].plot(df['episode'], df['td_var'].rolling(window=10).mean(), label='TD Var', color='red')
    axes[0, 1].set_title("TD Variance")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].grid(True)

    # 3. Adaptive Signals (Alpha and C_t)
    axes[1, 0].plot(df['episode'], df['alpha_t'], label='Alpha_t', color='green')
    axes[1, 0].plot(df['episode'], df['c_t'], label='Reliability (c_t)', color='orange')
    axes[1, 0].set_title("Adaptive Signals")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Latent Gap
    axes[1, 1].plot(df['episode'], df['latent_gap'].rolling(window=10).mean(), label='Latent Gap', color='purple')
    axes[1, 1].set_title("Latent Gap (Causal vs Smoother)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Gap")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = "output/figures/learning_proof_results.png"
    plt.savefig(plot_path)
    print(f"Resulting plots saved to {plot_path}")

if __name__ == "__main__":
    run_learning_proof()
