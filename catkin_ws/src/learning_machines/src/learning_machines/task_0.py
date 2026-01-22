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
STEPS = 20

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





def test(rob: IRobobo):
    print("Started task test")

    ir_values = []
    speed = 10
    for _ in range(150):

        #------------------------------------
        irs = np.array(rob.read_irs(), dtype=np.float32)
        # irs = np.clip(irs, 0, 60000)
        weights = np.array([1, 1, 2, 2, 2, 2, 1, 1])  # front sensors heavier
        ir_avg = np.average(irs, weights=weights)
        ir_norm = np.clip(ir_avg / 60000, 0, 1)
        print("IR avg", ir_avg)
        #------------------------------------

        #------------------------------------
        if isinstance(rob, SimulationRobobo):
            number = rob.get_nr_food_collected()
            print("Food collected:", number)
        #------------------------------------

        #------------------------------------
        img = rob.read_image_front()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask of green
        mask = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))
        green_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        print("Green ratio:", green_ratio)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            # No food visible → safe defaults
            x_center = 0.0
            green_ratio = 0.0
        else:
            mean_x = np.mean(xs)
            x_center = mean_x / img.shape[1]  # normalize to [0, 1]
            x_center = (x_center - 0.5) * 2 # normalize to [-1, 1]

        print("Center:", x_center)
        #------------------------------------



        ir_values.append(max(irs))
        rob.move_blocking(speed, -speed, 200)




