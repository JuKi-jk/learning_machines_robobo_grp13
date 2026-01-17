import cv2
import random
import os
import numpy as np
import csv
import time


from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import Env
from gymnasium.spaces import Box


def calculate_fitness(irs, action, max_speed, save_to_results=False, filename="Default_Fitness"):
    max_ir = max(irs)
    # Normalize speed from [-1,1] to [0,1], so that the reward for backing up is not worse than when collding
    left_nor = (action[0] + 1) / 2
    right_nor = (action[1] + 1) / 2
    forward = (left_nor + right_nor) * max_speed
    turning = 1 - np.sqrt(abs(left_nor - right_nor))
    collision = 1 - max_ir
    step_penalty = 0.3
    fitness = forward * turning * collision - step_penalty

    if save_to_results:
        results_dir = "/root/results/"
        file_exists = os.path.isfile(results_dir + filename)

        with open(results_dir + filename, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header once
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "left_action",
                    "right_action",
                    "forward",
                    "turning",
                    "collision",
                    "fitness"
                ])

            writer.writerow([
                time.time(),
                float(action[0]),
                float(action[1]),
                float(forward),
                float(turning),
                float(collision),
                float(fitness)
            ])

    return fitness


def normalize_irs(rob, irs):
    if isinstance(rob, SimulationRobobo):
        irs = irs / 10.0
    # return np.clip(irs / 100, 0, 1)

    irs = irs.astype(np.float32)
    irs = np.maximum(irs, 1.0)  # avoid log(0)

    norm = np.zeros_like(irs)

    # Threshold between "far" and "close", for extra detail in small changes for faraway proximity detection
    THRESH = 150.0

    # Far: amplify small differences
    mask_far = irs <= THRESH
    norm[mask_far] = np.sqrt(irs[mask_far] / THRESH) * 0.3  # maps 0..250 -> 0..0.3

    # Close: log compression
    mask_close = irs > THRESH
    log_vals = np.log(irs[mask_close])
    IR_MIN = np.log(THRESH + 1e-6)
    IR_MAX = np.log(4e7)
    norm[mask_close] = 0.3 + (log_vals - IR_MIN) / (IR_MAX - IR_MIN) * 0.7  # maps to 0.3..1.0

    return np.clip(norm, 0.0, 1.0)


class RoboboEnv(Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        if isinstance(self.rob, SimulationRobobo):
            self.start_position = self.rob.get_position()

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        self.max_steps = 400
        self.current_step = 0

        self.max_speed = 60


    def step(self, action):
        """Execute one step of the environment"""
        self.current_step += 1

        left_speed = int(action[0] * self.max_speed)
        right_speed = int(action[1] * self.max_speed)

        self.rob.move_blocking(left_speed, right_speed, 200)
        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        obs = normalize_irs(self.rob, irs)

        reward = calculate_fitness(obs, action, self.max_speed, True, "results.csv")

        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}
        return obs, reward, np.array(terminated, dtype=bool), np.array(truncated, dtype=bool), info


    def reset(self, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()

        self.rob.reset_wheels()
        self.current_step = 0
        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        obs = normalize_irs(self.rob, irs)
        info = {}
        return obs, info


    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()


def task_1_learning(rob: IRobobo, model_name):
    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO("MlpPolicy", env, verbose=1)

    print("Simulation started, training...")
    # Save every 3k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=3000,
        save_path="/root/models/",
        name_prefix=model_name
    )

    model.learn(total_timesteps=45000, callback=[checkpoint_callback])
    print("saved model")
    model.save("/root/models/" + model_name)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def task_1_predict(rob: IRobobo, model_name):
    print(f"Predicting trained model {model_name}...")
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.load("/root/models/" + model_name)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        max_abs = np.max(np.abs(action))  # 0.8
        # Scale action, because predicted action values are extremely low
        if max_abs > 0:  # Highest wheel action value is scaled to 0.8 for smooth movement
            scale = 0.8 / max_abs
        else:
            scale = 1.0
        obs, _, _, _ = env.step(action * scale)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def run_task_1(rob: IRobobo, mode="predict", model_name="Default_PPO"):
    if mode == "learn":
        task_1_learning(rob, model_name)
    else:
        task_1_predict(rob, model_name)


