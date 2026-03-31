# test_baseline.py
from config import Config
from envs import AdaptivePIDEnv
import numpy as np

cfg = Config()
env = AdaptivePIDEnv(config=cfg.env)

for ep in range(5):
    obs, info = env.reset()
    done = False
    total_reward = 0
    post_errors = []
    
    while not done:
        action = np.zeros(env.action_space.shape[0])  # always default gains
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if step_info["mass_perturbed"]:
            post_errors.append(step_info["error_norm"])
        if step_info["mass_perturbed"] and len(post_errors) == 1:
            print(f"  Mass dropped to: {step_info['mass']:.4f} kg (lost {0.027 - step_info['mass']:.4f})")
    
    post_err = np.mean(post_errors) if post_errors else 0
    print(f"Ep {ep} | Reward: {total_reward:.1f} | Post-err: {post_err:.4f}")

env.close()