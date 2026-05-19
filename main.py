from zero_g_env import ZeroGAnt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class FinalPosErrorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            error = info.get("final_pos_error")
            if error is not None:
                self.logger.record("eval/final_pos_error", error)
        return True
    
# 1. Instantiate your custom class instead of using gym.make
env = ZeroGAnt(render_mode="human")

# Create the PPO RL Agent
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

# Create the SAC RL Agent
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./logs")

pos_error_callback = FinalPosErrorCallback()

observation, info = env.reset()

model.learn(total_timesteps=100000, callback=pos_error_callback)

# model.save("ppo_zer_g_cube")
model.save("sac_zero_g_cube")

# Evaluation loop
num_episodes = 100
episodes_completed = 0
episodes_succeeded = 0

observation, info = env.reset()

"""
for _ in range(1000):
    # Using the PPO model
    action, _states = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

print(f"reward: {reward}")
"""

while episodes_completed < num_episodes:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        episodes_completed += 1

        if info.get("is_success", False):
            episodes_succeeded += 1

        observation, info = env.reset()
    
success_rate = (episodes_succeeded / num_episodes) * 100
print(f"Success Rate: {episodes_succeeded}/{num_episodes} = {success_rate:.1f}%")

env.close()
