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

MODE = None

MAX_STEPS = 400

# Constants for fitness calculation
MAX_SPEED = 60
TURN_SPEED = 25
FOOD_REWARD = 1000
IR_NORM_FACTOR = 1000.0 # For normalizing IR readings
PHONE_TILT = 101
PHONE_PAN = 193
ACTION_EPS = 0.03  # how similar is "same"
LOW_MOVEMENT_THRESH = 0.03 # Below this is considered no movement
STAGNATION_PENALTY = 2.0 # Per step of stagnation
NO_FOOD_PENALTY = 1.0 # Per step without food
SUCCESS_BONUS = 1000 # For completing the task
FAILURE_PENALTY = 500 # For failing the task
AFTER_FOOD_PENALTY = 100 # Penalty for moving fast right after eating
COLLISION_PENALTY = 100 # Penalty for being too close to wall

def calculate_fitness(obs, action, found_food, stagnation_steps, no_food_steps, save_to_results=False, filename="Default_Fitness"):
    x_center = obs[0]
    green_ratio = obs[1]
    avg_irs = obs[3]

    collision = 1.0 - avg_irs
    if green_ratio > 0.05:
        collision = 1.0

    align_reward = 1.0 - abs(x_center) # closer to 0 gives better value

    fitness = 0
    fitness += action[0] * MAX_SPEED
    fitness *= collision
    fitness += FOOD_REWARD * green_ratio * align_reward * action[0] # reward going towards food
    fitness -= STAGNATION_PENALTY * stagnation_steps # discourage not moving and stagnating
    fitness -= NO_FOOD_PENALTY * no_food_steps # discourage not finding food for long
    fitness -= COLLISION_PENALTY * avg_irs  # discourage getting too close to obstacles
    fitness -= abs(action[1]*TURN_SPEED) / 5 # Small turning penalty
    if no_food_steps <= 5:
        fitness -= AFTER_FOOD_PENALTY * action[0]  # discourage speeding right after eating
    if found_food:
        fitness += FOOD_REWARD

    return fitness

IR_BASELINES = 5.8
ALPHA = 1.0
def normalize_irs(rob, irs):
    irs = irs[[2, 3, 4, 5, 7]]  # only front sensors

    # SIMULATION CALIBRATION
    if isinstance(rob, SimulationRobobo):
        irs[0] -= 46.4
        irs[1] -= 46.4
    irs = irs - IR_BASELINES
    irs = np.clip(irs, 0, None)

    # Log normalization
    irs = np.clip(irs, 0, IR_NORM_FACTOR)
    log_ir = np.log1p(irs * ALPHA)
    log_ir_norm = log_ir / np.log1p(IR_NORM_FACTOR * ALPHA)

    ir_avg = np.mean(log_ir_norm )
    ir_left = np.mean(log_ir_norm [[0, 4]])
    ir_right = np.mean(log_ir_norm [[1, 3]])
    ir_front = log_ir_norm [2]

    return ir_avg, ir_left, ir_right, ir_front, log_ir_norm

def find_food(rob: IRobobo):
    # Camera readings
    img = rob.read_image_front()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask of green
    mask = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return 0.0, 0.0

    # Find largest blob, and extract center and area features
    blobs = stats[1:]
    centers = centroids[1:]
    largest_idx = np.argmax(blobs[:, cv2.CC_STAT_AREA])
    area = blobs[largest_idx, cv2.CC_STAT_AREA]
    cx, cy = centers[largest_idx]

    green_ratio = area / (img.shape[0] * img.shape[1])
    x_center = (cx / img.shape[1] - 0.5) * 2
    return x_center, green_ratio

def termination(rob, food, avg_irs, green_ratio, no_food_steps):
    truncated = False
    terminated = False
    extra_reward = 0
    if rob.get_sim_time() > 180 * 3: # Time limit reached
        print("Time limit reached")
        truncated = True
        extra_reward = 0

    if food >= 7: # All food found
        print("All food found")
        terminated = True
        extra_reward = SUCCESS_BONUS + (180*3 - rob.get_sim_time()) * 10  # faster is better

    if avg_irs > 0.85 and green_ratio == 0: # Collission termination
        print("Collision detected")
        terminated = True
        extra_reward -= FAILURE_PENALTY

    if no_food_steps > 120: # Too long no food found
        print("No food found in 120 steps")
        terminated = True
        extra_reward -= FAILURE_PENALTY

    if terminated or truncated:
        print("At", rob.get_sim_time())

    return terminated, truncated, extra_reward


class RoboboEnv(Env):
    def __init__(self, rob: IRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        if isinstance(self.rob, SimulationRobobo):
            self.start_position = self.rob.get_position()

        self.action_space = Box(low=np.array([0.0, -1.0], dtype=np.float32),
                                high=np.array([1.0, 1.0], dtype=np.float32),
                                dtype=np.float32)

        self.observation_space = Box(
            low=np.array([-1.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.prev_action = np.array([0,0], dtype=np.float32)
        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.current_food = 0
        self.action_stagnation_steps = 0
        self.steps_without_food = 0

        self.max_speed = MAX_SPEED
        self.max_turn_speed = TURN_SPEED

    def step(self, action):
        """Execute one step of the environment"""
        self.current_step += 1

        # Convert action to wheel speeds
        forward = float(action[0]) * self.max_speed
        turn = float(action[1]) * self.max_turn_speed
        left_speed = int(forward - turn)
        right_speed = int(forward + turn)

        self.rob.move_blocking(left_speed, right_speed, 200)

        # Compare current action to previous, to increase stagnation counter
        movement = (abs(left_speed) + abs(right_speed)) / (2 * self.max_speed)
        if np.linalg.norm(action - self.prev_action) < ACTION_EPS or movement < LOW_MOVEMENT_THRESH:
            self.action_stagnation_steps += 1
        else:
            self.action_stagnation_steps = max(0, self.action_stagnation_steps - 3)
        self.prev_action = action.copy()

        food_inc = False
        if isinstance(self.rob, SimulationRobobo):
            number = self.rob.get_nr_food_collected()
            if number > self.current_food:
                self.current_food = number
                food_inc = True
                self.steps_without_food = 0
            else:
                self.steps_without_food += 1

        # IR readings
        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        avg_irs, ir_left, ir_right, ir_front, norm_irs = normalize_irs(self.rob, irs)

        # Camera readings
        x_center, green_ratio = find_food(self.rob)

        # Features
        food_found = 1 if green_ratio > 0.1 else 0
        obs = np.array([
            x_center,
            green_ratio,
            food_found,
            avg_irs,
            ir_left,
            ir_right,
            ir_front
        ], dtype=np.float32)

        reward = calculate_fitness(obs, action, food_inc, self.action_stagnation_steps, self.steps_without_food, True, "results.csv")
        if MODE == "learn":
            terminated, truncated, extra_reward = termination(self.rob, self.current_food, avg_irs, green_ratio, self.steps_without_food)
        else:
            terminated = self.current_step >= self.max_steps
            truncated = False
            extra_reward = 0
            if self.current_food >= 7: # All food found
                print("\nAll food found \(00)/\n")
                terminated = True

        reward += extra_reward
        info = {}
        return obs, reward, np.array(terminated, dtype=bool), np.array(truncated, dtype=bool), info

    def reset(self, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()

            # Set phone almost straightup, so that it does not see floating food
            self.rob.set_phone_pan_blocking(PHONE_PAN, 50)
            self.rob.set_phone_tilt_blocking(PHONE_TILT, 50)

        self.rob.reset_wheels()
        self.current_step = 0
        self.prev_action = np.array([0,0], dtype=np.float32)
        self.current_food = 0
        self.action_stagnation_steps = 0
        self.steps_without_food = 0

        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        avg_irs, ir_left, ir_right, ir_front, _ = normalize_irs(self.rob, irs)
        x_center, green_ratio = find_food(self.rob)

        food_found = 1 if green_ratio > 0.1 else 0
        obs = np.array([
            x_center,
            green_ratio,
            food_found,
            avg_irs,
            ir_left,
            ir_right,
            ir_front
        ], dtype=np.float32)

        info = {}
        return obs, info

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()


def task_2_learning(rob: IRobobo, model_name):
    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO("MlpPolicy", env, verbose=1)

    print("Simulation started, training task 2...")
    # Save every 5k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="/root/models/",
        name_prefix=model_name
    )

    model.learn(total_timesteps=100000, callback=[checkpoint_callback])
    print("saved model")
    model.save("/root/models/" + model_name)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def task_2_predict(rob: IRobobo, model_name):
    print(f"Predicting trained model {model_name}...")

    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO.load("/root/models/" + model_name, env=env)

    obs = env.reset()
    for _ in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def run_task_2(rob: IRobobo, mode="predict", model_name="Default_PPO"):
    global MODE
    if mode == "learn":
        MODE = "learn"
        task_2_learning(rob, model_name)
    else:
        MODE = "predict"
        task_2_predict(rob, model_name)
