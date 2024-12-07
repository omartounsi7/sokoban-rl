import sys
import os
import time
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN

from src.constants import ACTION_SPACE, SUPERBONUS, SUPERMALUS

TIME_STEPS = 10000
MAX_ACTIONS = 100

def sb3_train_model():
    if len(sys.argv) != 3:
        print("Usage: python sb3_solve_puzzle.py <puzzle_file> <algorithm>")
        print("Available algorithms: PPO, A2C, DQN")
        sys.exit(1)

    puzzle_path = sys.argv[1]
    algorithm = sys.argv[2]

    try:
        gym.register(
            id='SokobanEnv-v0',
            entry_point='src.SokobanEnv:SokobanEnv',
            kwargs={"level_file": puzzle_path}
        )
    except FileNotFoundError:
        print(f"Error: Puzzle file '{puzzle_path}' not found.")
        sys.exit(1)

    env = gym.make('SokobanEnv-v0')
    env.reset()

    models_dir = "models/" + algorithm
    log_dir = "logs"
    model_path = f"{models_dir}/latest.zip"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if algorithm == "PPO":
        model_cls = PPO
    elif algorithm == "A2C":
        model_cls = A2C
    elif algorithm == "DQN":
        model_cls = DQN
    else:
        print(f"Error: Unknown algorithm '{algorithm}'.")
        print("Available algorithms: PPO, A2C, DQN")
        sys.exit(1)

    print(f"Running training with {algorithm}...")
    print("IMPORTANT: This will run indefinitely. Press CTRL+C to stop training and switch to playback.")
    print("Run tensorboard --logdir=logs to monitor metrics.")

    try:
        if os.path.exists(model_path):
            print(f"Loaded existing model from {model_path}")
            model = model_cls.load(model_path, env=env)
        else:
            model = model_cls('MlpPolicy', env, tensorboard_log=log_dir)

        iters = 0
        while True:
            print("Timesteps: " + str(TIME_STEPS * iters))
            iters += 1
            model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name=algorithm)
            model.save(f"{models_dir}/latest")
    
    except KeyboardInterrupt:
        print("Training interrupted. Model saved.")
        model.save(f"{models_dir}/latest")
        play(env, model)

def play(env, model):
    print("Playing level according to the learned policy...")
    obs, info = env.reset()
    done = False
    steps = 0
    action_sequence = []
    while not done and steps < MAX_ACTIONS:
        action, _states = model.predict(obs)
        action_sequence.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        env.render()
        steps += 1
        time.sleep(0.5)
    env.close()

    if reward == SUPERBONUS:
        print("Game won!")
    elif reward == SUPERMALUS:
        print("Game lost.")
    
    literal_policy = [ACTION_SPACE[action] for action in action_sequence]
    print("Number of actions: " + str(steps))
    print("Learned policy: " + str(literal_policy))


if __name__ == "__main__":
    sb3_train_model()