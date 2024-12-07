import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO


def train_sb3_model():
    if len(sys.argv) != 2:
        print("Usage: python sb3_template.py <puzzle_file>")
        sys.exit(1)

    puzzle_path = sys.argv[1]

    gym.register(
        id='SokobanEnv-v0',
        entry_point='src.SokobanEnv:SokobanEnv',
        kwargs={"level_file": puzzle_path}
    )

    env = gym.make('SokobanEnv-v0')
    env.reset()

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    
    autoplay_sb3(env, model)
    env.close()

def autoplay_sb3(env, model):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        env.render()
        time.sleep(1)


if __name__ == "__main__":
    train_sb3_model()