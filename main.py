from zero_g_env import ZeroGAnt

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

full_obs_config = dict( 
        use_noise=False, 
        use_missing=False, 
        use_delays=False
    )

class FinalPosErrorCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            error = info.get("final_pos_error")
            if error is not None:
                self.logger.record("train/final_pos_error", error)
        return True
    

def make_env(rank, seed):
    def _init():
        env = ZeroGAnt(render_mode=None, **full_obs_config)
        env.reset(seed=seed + rank)
        return env
    return _init

def evaluate_model(model, num_episodes=100):
    eval_env = ZeroGAnt(render_mode=None, **full_obs_config)

    episodes_completed = 0
    episodes_succeeded = 0

    obs, info = eval_env.reset()

    while episodes_completed < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        if terminated or truncated:
            episodes_completed += 1

            if info.get("is_success", False):
                episodes_succeeded += 1

            obs, info = eval_env.reset()
        
    eval_env.close()

    success_rate = episodes_succeeded / num_episodes * 100
    return success_rate

if __name__ == "__main__":
    
    total_timesteps = 100_000
    n_envs = 5
    seeds = [0, 1, 2, 3, 4]

    policy_kwargs = dict(net_arch=[256, 256])

    algorithms = { "PPO": PPO }
    #algorithms = { "SAC": SAC }

    results = {}

    for algo_name, AlgoClass in algorithms.items():
        results[algo_name] = []

        for seed in seeds:
            print(f"\nTraining {algo_name} with seed {seed}")

            train_env = SubprocVecEnv(
                [make_env(i, seed) for i in range(n_envs)],
                start_method="spawn"
            )

            train_env = VecMonitor(train_env)

            model = AlgoClass( "MlpPolicy", train_env, verbose=1, tensorboard_log="./logs", policy_kwargs=policy_kwargs, seed=seed)

            model.learn(total_timesteps=total_timesteps, callback=FinalPosErrorCallback(), tb_log_name=f"{algo_name}_seed_{seed}")

            model.save(f"{algo_name.lower()}_zero_g_cube_seed_{seed}")

            success_rate = evaluate_model(model, num_episodes=100)
            results[algo_name].append(success_rate)

            print(f"{algo_name} seed {seed} success rate: {success_rate:.1f}%")

            train_env.close()

    print("\nFinal Results")
    for algo_name, scores in results.items():
        avg = sum(scores) / len(scores)
        print(f"{algo_name}: {scores} | Average = {avg:.1f}%")