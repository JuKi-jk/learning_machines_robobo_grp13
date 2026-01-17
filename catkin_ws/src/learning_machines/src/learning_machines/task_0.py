import cv2

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

    task_0(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()





# def task_1(rob: IRobobo):
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
#     print("Started task 1")

#     ir_values = []
#     speed = 40
#     for _ in range(150):
#         irs = rob.read_irs()
#         print(max(irs))
#         ir_values.append(max(irs))
#         rob.move_blocking(speed, speed, 100)

#     print("max:" , max(ir_values))
#     if isinstance(rob, SimulationRobobo):
#         rob.stop_simulation()



