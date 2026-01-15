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


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    test_emotions(rob)
    test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

    test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

################
# test_actions.py
import time

def task0(rob,
          steps=600, dt=0.05,
          forward=25, turn=35,
          front_band=(1,2,3,4),
          near_offset=8.0, clear_offset=4.0):

    def front_signal(ir):   # higher = closer
        return max(ir[i] for i in front_band)

    # make sure sim is running
    if hasattr(rob, "play_simulation"):
        rob.play_simulation()

    # interface is responsive (sim scripts / ROS ready)
    for _ in range(60):
        try:
            rob.read_irs()
            break
        except Exception:
            print("Waiting for robot/sim...")
            time.sleep(0.1)

    # base zero in hardware so update: wait until IR values are actually non-zero (hardware often starts with zeros)
    print("[Task0] Waiting for IR sensors to become active...")
    for _ in range(100):  # up to ~10 seconds
        ir = list(rob.read_irs())
        if any(v > 0 for v in ir):
            print("[Task0] IR active:", ir)
            break
        time.sleep(0.1)
    else:
        print("[Task0] WARNING: IR stayed zero; continuing anyway (check app/robot awake).")

    # baseline calibration (cuz we need to have some idea what "clear" is)
    samples = []
    for _ in range(30):
        ir = list(rob.read_irs())
        samples.append(front_signal(ir))
        time.sleep(0.02)
    base = sum(samples) / len(samples)

    near_th  = base + near_offset
    clear_th = base + clear_offset
    print(f"base={base:.2f} near={near_th:.2f} clear={clear_th:.2f} front={front_band}")

    mode = "forward"

    for _ in range(steps):
        ir = list(rob.read_irs())
        fs = front_signal(ir)

        if mode == "forward" and fs >= near_th:
            mode = "turn"
        elif mode == "turn" and fs <= clear_th:
            mode = "forward"

        if mode == "forward":
            left, right = -forward, -forward     # forward = negative in this setup
        else:
            left, right = -turn, turn            # turn right

        rob.move_blocking(left, right, int(dt * 1000))

    rob.move_blocking(0, 0, 250)