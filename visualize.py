from zero_g_env import ZeroGAnt
from stable_baselines3 import PPO, SAC

env = ZeroGAnt(
    render_mode="human",
    use_noise=False,
    use_missing=False,
    use_delays=False
)

# Load trained model
model = PPO.load("ppo_zero_g_cube_seed_0")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Success:", info.get("is_success", False))
        obs, info = env.reset()