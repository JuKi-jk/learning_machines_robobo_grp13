import cv2
import numpy as np
import os
import csv
import time
from collections import deque

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
MAX_STEPS = 500

# Constants for fitness calculation
MAX_SPEED = 60
TURN_SPEED = 20
RED_REWARD = 1000
GREEN_REWARD = 5000
CONSTANT_GREEN_REWARD = 300
IR_NORM_FACTOR = 1000.0 # For normalizing IR readings
PHONE_TILT = 92.5
PHONE_PAN = 180
SUCCESS_BONUS = 7500 # For completing the task
FAILURE_PENALTY = 200 # For failing the task
NO_CHANGE_PENALTY = 0.1 # Per step without food
COLLISION_PENALTY = 50 # Penalty for being too close to wall
X_MARGIN = 0.25  # Margin for x alignment when starting to push
IR_WINDOW_SIZE = 3


def save_results(filename, components):
    results_dir = "/root/results/"

    with open(results_dir + filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            float(components[0]),
            float(components[1]),
            float(components[2]),
            float(components[3]),
            float(components[4]),
            float(components[5]),
            float(components[6]),
            float(components[7]),
            float(components[8])
        ])

def calculate_fitness(obs, action, time_step, ir_window, save_to_results=False, filename="Default_Fitness"):
    x_red, red_ratio, x_green, green_ratio, avg_irs, _, _, _, red_aligned = obs

    fitness = 0
    fitness += min(RED_REWARD * red_ratio * (1.0 - abs(x_red)) * action[0], 200) # reward going towards food
    if red_aligned:
        if green_ratio > 0:
            fitness += CONSTANT_GREEN_REWARD * action[0] * (1.0 - abs(x_green))  # bonus for pushing when aligned
        fitness += GREEN_REWARD * green_ratio * (1.0 - abs(x_green)) * action[0]

    if red_ratio == 0 and green_ratio == 0:
        fitness += 3.0 * action[0]
    if action[0] < 0.1:
        fitness -= 3.0


    fitness -= NO_CHANGE_PENALTY * time_step # Time penalty
    if np.all(np.array(ir_window) > 0.3) and len(ir_window) == IR_WINDOW_SIZE:
        fitness -= COLLISION_PENALTY * avg_irs  # Discourage getting too close to obstacles

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

    ir_avg = np.mean(log_ir_norm[[0, 1, 3, 4]]) # Average is taken without the front IR for this task
    ir_left = np.mean(log_ir_norm[[0, 4]])
    ir_right = np.mean(log_ir_norm[[1, 3]])
    ir_front = log_ir_norm[2]

    return ir_avg, ir_left, ir_right, ir_front, log_ir_norm

def extract_blob_features(mask, img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return 1.0, 0.0  # x_center, area_ratio, distance_proxy

    blobs = stats[1:]
    centers = centroids[1:]
    largest_idx = np.argmax(blobs[:, cv2.CC_STAT_AREA])
    area = blobs[largest_idx, cv2.CC_STAT_AREA]
    cx, cy = centers[largest_idx]

    area_ratio = area / (img.shape[0] * img.shape[1])
    x_center = (cx / img.shape[1] - 0.5) * 2

    return x_center, area_ratio

def process_image(rob: IRobobo):
    # Camera readings
    img = rob.read_image_front()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # masks
    green_mask = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))
    red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))

    x_green, green_area = extract_blob_features(green_mask, img)
    x_red, red_area = extract_blob_features(red_mask, img)

    return x_green, green_area, x_red, red_area

def termination(rob, ir_left, ir_right, avg_ir, steps_not_pushing, ir_window):
    truncated = False
    terminated = False
    extra_reward = 0
    if isinstance(rob, SimulationRobobo):
        if rob.get_sim_time() > 300: # Time limit reached
            print("Time limit reached")
            truncated = True

        if rob.base_detects_food(): # Food collected
            print("Food on base detected")
            terminated = True
            extra_reward += SUCCESS_BONUS - rob.get_sim_time() * 2  # faster is better

    if MODE == "learn":
        if steps_not_pushing > 200: # Too long no food found
            print("Not pusshing in 200 steps")
            terminated = True
            extra_reward -= FAILURE_PENALTY

        if ir_left > 0.86 or ir_right > 0.86 or avg_ir > 0.68: # collision detected
            print("Collision detected")
            terminated = True
            extra_reward -= FAILURE_PENALTY

    if isinstance(rob, SimulationRobobo):
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
            low=np.array([-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = MAX_STEPS
        self.current_step = 0
        self.steps_not_pushing = 0
        self.ir_window = deque(maxlen=IR_WINDOW_SIZE)

        # self.action_stagnation_steps = 0

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

        # IR readings
        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        avg_irs, ir_left, ir_right, ir_front, norm_irs = normalize_irs(self.rob, irs)
        self.ir_window.append(avg_irs)

        # Camera readings
        x_green, green_ratio, x_red, red_ratio = process_image(self.rob)

        red_aligned = False
        if red_ratio > 0 and abs(x_red) < X_MARGIN: # Determine if pushing
            red_aligned = True
        else:
            self.steps_not_pushing += 1

        # Features
        obs = np.array([
            x_red,
            red_ratio,
            x_green,
            green_ratio,
            avg_irs,
            ir_left,
            ir_right,
            ir_front,
            red_aligned
        ], dtype=np.float32)

        reward = calculate_fitness(obs, action, self.current_step, self.ir_window)
        terminated, truncated, extra_reward = termination(self.rob, ir_left, ir_right,avg_irs, self.steps_not_pushing, self.ir_window)
        reward += extra_reward
        info = {}

        # if MODE == "predict":
        #     save_results("results.csv", components=[action[0], action[1], avg_irs, green_ratio, 1.0 - abs(x_center), self.action_stagnation_steps, self.steps_without_food, self.current_food, reward])
        return obs, reward, np.array(terminated, dtype=bool), np.array(truncated, dtype=bool), info

    def reset(self, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()

            # Set phone almost straightup, so that it does not see floating food
            self.rob.set_phone_pan_blocking(PHONE_PAN, 50)
            self.rob.set_phone_tilt_blocking(PHONE_TILT, 50)

        self.rob.reset_wheels()
        self.ir_window.clear()
        self.current_step = 0
        self.steps_not_pushing = 0
        red_aligned = False

        irs = np.array(self.rob.read_irs(), dtype=np.float32)
        avg_irs, ir_left, ir_right, ir_front, _ = normalize_irs(self.rob, irs)
        # Camera readings
        x_green, green_ratio, x_red, red_ratio = process_image(self.rob)

        # Features
        obs = np.array([
            x_red,
            red_ratio,
            x_green,
            green_ratio,
            avg_irs,
            ir_left,
            ir_right,
            ir_front,
            red_aligned
        ], dtype=np.float32)

        info = {}
        return obs, info

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()


def task_3_learning(rob: IRobobo, model_name):
    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO("MlpPolicy", env, verbose=1)
    # model = PPO.load("/root/models/Default_PPO_85000_steps", env=env)

    print("Simulation started, training task 3...")
    # Save every 5k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=3000,
        save_path="/root/models/",
        name_prefix=model_name
    )

    model.learn(total_timesteps=100000, callback=[checkpoint_callback], reset_num_timesteps=False)
    print("saved model")
    model.save("/root/models/" + model_name)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

def task_3_predict(rob: IRobobo, model_name):
    print(f"Predicting trained model {model_name}...")

    results_dir = "/root/results/"
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "forward",
            "turning",
            "avg_irs",
            "green_ratio",
            "align_reward",
            "stagnation",
            "no_food_steps",
            "food_collected",
            "fitness"
        ])

    env = DummyVecEnv([lambda: RoboboEnv(rob)])
    model = PPO.load("/root/models/" + model_name, env=env)

    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, _, terminated, _ = env.step(action)
        if terminated:
            break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def run_task_3(rob: IRobobo, mode="predict", model_name="Default_PPO"):
    global MODE
    if mode == "learn":
        MODE = "learn"
        task_3_learning(rob, model_name)
    else:
        MODE = "predict"
        task_3_predict(rob, model_name)
