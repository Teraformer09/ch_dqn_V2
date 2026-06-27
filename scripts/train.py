import os
import subprocess
from chdqn.config import ChDQNConfig

def run_model(model_name, seed):
    print(f"Starting {model_name} seed {seed}...")
    config = ChDQNConfig(
        train_episodes=300,
        max_steps_per_episode=200,
        batch_size=32,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon_decay=10000,
        min_replay_sequences=200,
        is_non_stationary=True,
        seed=seed
    )
    # Since I'm running in a script, I'll just use the trainer directly
    from chdqn.trainer_rl import CartPoleRLRunner
    runner = CartPoleRLRunner(config, model_type=model_name)
    log_path = f"results/{model_name}_run_seed{seed}.csv"
    runner.train(log_path=log_path)

if __name__ == "__main__":
    import sys
    model = sys.argv[1]
    seed = int(sys.argv[2])
    run_model(model, seed)
