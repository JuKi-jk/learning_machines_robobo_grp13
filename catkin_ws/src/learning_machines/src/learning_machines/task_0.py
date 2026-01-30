import cv2
import numpy as np

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

DEVIATION_THRESHOLD = 4
FORWARD_SPEED = 35
TURN_SPEED = 30
STEP_TIME = 500
SLEEP_DT = 0.06
LR = 0.05
STEPS = 25

def measure_baseline(rob, samples=20):
    values = []
    for _ in range(samples):
        irs = rob.read_irs()
        if irs[4] is not None:
            values.append(irs[4])
        rob.sleep(0.02)
    return sum(values) / len(values)


def task_0(rob: IRobobo):
    # rob.move_blocking(100, 100, 1000)
    print("Started task 0")

    base = measure_baseline(rob, samples=30)

    ir_values = []
    behaviour = []
    for _ in range(STEPS):
        # check obstacle
        print("Checking IR sensors...")
        irs = rob.read_irs()
        print(irs[4])
        ir_values.append(irs[4])
        if irs[4] - base > DEVIATION_THRESHOLD:
            print("Obstacle detected → turning right")
            rob.move_blocking(TURN_SPEED, -TURN_SPEED, STEP_TIME)
            behaviour.append("T")
        else:
            rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, STEP_TIME)
            behaviour.append("F")

        # update baseline value with an LR incase the baseline did not give a good picture
        base = (base * (1 - LR)) + (irs[4] * LR)

        rob.sleep(SLEEP_DT)

    print("IR values:", [f"{v:.2f}" for v in ir_values])
    print("Behaviour:", behaviour)




def run_task_0(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    test(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()





IR_BASELINES = 5.8
ALPHA = 1.0
IR_NORM_FACTOR = 1000.0
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

    ir_avg = np.mean(log_ir_norm)
    ir_left = np.mean(log_ir_norm[[0, 4]])
    ir_right = np.mean(log_ir_norm[[1, 3]])
    ir_front = log_ir_norm[2]

    return ir_avg, ir_left, ir_right, ir_front, log_ir_norm


def extract_blob_features(mask, img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return 0.0, 0.0  # x_center, area_ratio, distance_proxy

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
    # red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))

    x_green, green_area = extract_blob_features(green_mask, img)
    x_red, red_area = extract_blob_features(red_mask, img)

    return x_green, green_area, x_red, red_area


def test(rob: IRobobo):
    print("Started task test")
    # rob.set_phone_pan_blocking(180, 50)
    rob.set_phone_tilt_blocking(109, 50)
    # rob.set_phone_tilt_blocking(109, 50)
    # print("hey",rob.read_phone_tilt())
    rob.sleep(2)
    # rob.set_phone_tilt_blocking(50, 50)
    # print("hey",rob.read_phone_tilt())

    speed = 40
    prev_red_ratio = 0.0
    # for i in range(150):
        # print("hey",rob.read_phone_tilt())
        # rob.sleep(1)

        # if i < 10:
        #     speed = -30
        # else:
        #     speed = 30
        #------------------------------------
        # irs = np.array(rob.read_irs(), dtype=np.float32)
        # ir_avg, ir_left, ir_right, ir_front, log_ir_norm = normalize_irs(rob, irs)
        # print("IR avg", ir_avg, ir_front)
        #------------------------------------

        #------------------------------------
        # x_green, green_ratio, x_red, red_ratio = process_image(rob)
        # if red_ratio == 0 and prev_red_ratio > 0:
            # print("Lost sight of red food")
        # prev_red_ratio = red_ratio
        # print(red_ratio)
        # print(rob.get_nr_food_collected())

        # print(
        #     f"Green: x={x_green:.2f}, area={green_ratio:.4f} |\nRed: x={x_red:.2f}, area={red_ratio:.4f}"
        # )
        # print("r:", 2000 * red_ratio * (1.0 - abs(x_red))) # reward going towards food)
        # print("g:", 2000 * green_ratio * (1.0 - abs(x_green))) # reward going towards food)

        #------------------------------------

        # rob_pos = rob.get_position()        # Position(x, y, z)
        # base_pos = rob.get_base_position()  # Position(x, y, z)
        # dx = rob_pos.x - base_pos.x
        # dy = rob_pos.y - base_pos.y
        # dz = rob_pos.z - base_pos.z
        # distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        # if rob.base_detects_food(): # Food collected
        #     print("Food on base detected")
        #     if isinstance(rob, SimulationRobobo):
        #         extra_reward = 1000 - rob.get_sim_time() * 2  # faster is better
        #         print("Extra reward:", extra_reward)

        # if i > 20:
        #     rob.move_blocking(speed, -speed, 200)
        # else:
        # rob.move_blocking(speed, speed, 200)


        # if i % 10 == 0:
        #     img = rob.read_image_front()
        #     save_path = f"/root/results/figures/step_{i}.png"
        #     cv2.imwrite(save_path, img)
        #     print(f"Saved image to {save_path}")

        # rob.sleep(1)
    print("---")




