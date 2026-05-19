import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium import spaces
import collections

class ZeroGAnt(AntEnv):
    def __init__(self, **kwargs):
        # Random goal initialize
        self.random_goals = False

        # Default fixed goal
        self.target_pos = np.array([2.0, 2.0, 2.0])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Goal will appear in this bubble
        self.goal_range = 3.0
        
        # Partial Observability Flags
        self.use_noise = True
        self.use_missing = True
        self.use_delays = True

        self.noise_std = 0.01
        self.missing_prob = 0.05
        self.delay_steps = 3

        # Buffer to store the last few raw observation
        self.obs_buffer = collections.deque(maxlen=self.delay_steps + 1)

        # Targets for the agent to reach
        #self.target_pos = np.array([2.0, 2.0, 2.0])
        #self.target_quat = np.array([1.0, 0.0, 0.0, 0.0]) # No rotation
        self.max_steps = 1000
        self.current_step = 0

        super().__init__(
            xml_file="C:/Users/selly/2026/School/zero_g_rl_project/custom_ant.xml",
            terminate_when_unhealthy=False,
            **kwargs
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

    def _get_obs(self):

        # Position Error (Target - Current) 
        # qpos[0:3] is x, y, z
        true_pos_error = self.target_pos - self.data.qpos[0:3]
        # Orientation Error
        # qpos[3:7] is the quaternion. We use the absolute difference here
        # or you can use mujoco.mju_subQuat for a more precise delta.
        ori_error = self.target_quat - self.data.qpos[3:7]
        # Linear Velocity (x, y, z)
        lin_vel = self.data.qvel[0:3]
        # Angular Velocity (roll, pitch, yaw)
        ang_vel = self.data.qvel[3:6]
        # Time Remaining (normalized 0 to 1)
        time_rem = np.array([(self.max_steps - self.current_step) / self.max_steps])

        # Raw Observations
        raw_obs = np.concatenate([true_pos_error, ori_error, lin_vel, ang_vel, time_rem]).astype(np.float32)

        # Partial Observability
        # Handle Delays
        if self.use_delays:
            self.obs_buffer.append(raw_obs)
            obs = self.obs_buffer[0]
        else:
            obs = raw_obs

        # Handle Noise
        if self.use_noise:
            noise = np.random.normal(0, self.noise_std, size=obs.shape)
            noise[-1] = 0
            obs = obs + noise

        # Handle Missing Values
        if self.use_missing:
            mask = np.random.choice([0, 1], size=obs.shape[0]-1, p=[self.missing_prob, 1-self.missing_prob])
            obs[:-1] = obs[:-1] * mask

        return obs.astype(np.float32)

    def step(self, action):
        # --- 1. GROUND TRUTH (Before Move) ---
        # We use data.qpos directly to get the "perfect" physics state for the reward
        true_pos_prev = self.data.qpos[0:3].copy()
        true_ori_prev = self.data.qpos[3:7].copy()
        prev_dist = np.linalg.norm(self.target_pos - true_pos_prev)
        prev_ori_dist = np.linalg.norm(self.target_quat - true_ori_prev)
        prev_total_error = prev_dist + prev_ori_dist

        # --- 2. PHYSICS STEP ---
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1

        # --- 3. GROUND TRUTH (After Move) ---
        true_pos_curr = self.data.qpos[0:3].copy()
        true_ori_curr = self.data.qpos[3:7].copy()
        curr_dist = np.linalg.norm(self.target_pos - true_pos_curr)
        curr_ori_dist = np.linalg.norm(self.target_quat - true_ori_curr)
        curr_total_error = curr_dist + curr_ori_dist
        
        # Ground Truth velocities for penalties
        true_lin_vel = self.data.qvel[0:3]
        true_ang_vel = self.data.qvel[3:6]

        # --- 4. REWARD CALCULATION (Based on Truth) ---
        # Delta-shaping: Positive if the real cube moved closer to the real goal
        pose_shaping = prev_total_error - curr_total_error
        
        # Penalties
        vel_penalty = np.linalg.norm(true_lin_vel) + np.linalg.norm(true_ang_vel)
        
        # Final Reward
        reward = (15.0 * pose_shaping) - (0.1 * vel_penalty)

        # --- 5. OBSERVATION (The "Fuzzy" Reality) ---
        # This calls your new _get_obs() with noise/delays/missing values
        obs = self._get_obs()

        # --- 6. TERMINATION / TRUNCATION ---
        # We terminate if the ACTUAL cube gets within a threshold
        terminated = bool(curr_dist < 0.05 and curr_ori_dist < 0.05)
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"is_success": terminated, "final_pos_error": curr_dist if {terminated or truncated} else None}

    def reset_model(self):
        self.current_step = 0

        # Randomize target if flag is True
        if self.random_goals:

            low_bounds = np.array([-4.0, -1.2, -1.2])
            high_bound = np.array([4.0, 1.2, 1.2])
            
            self.target_pos = self.np_random.uniform(
                low=-self.goal_range,
                high=self.goal_range,
                size=3
            )

        self.model.site('target_marker').pos[:] = self.target_pos

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)

        return self._get_obs()