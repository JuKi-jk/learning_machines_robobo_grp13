#!/usr/bin/env python3
# import sys, time
# from robobo_interface import SimulationRobobo, HardwareRobobo

# def task0(rob,
#           steps=600, dt=0.05,
#           forward=25, turn=35,
#           front_band=(1,2,3,4),
#           near_offset=8.0, clear_offset=4.0):

#     def front_signal(ir):   # higher = closer
#         return max(ir[i] for i in front_band)

#     # make sure sim is running
#     if hasattr(rob, "play_simulation"):
#         rob.play_simulation()

#     # interface is responsive (sim scripts / ROS ready)
#     for _ in range(60):
#         try:
#             rob.read_irs()
#             break
#         except Exception:
#             print("Waiting for robot/sim...")
#             time.sleep(0.1)

#     # base zero in hardware so update: wait until IR values are actually non-zero (hardware often starts with zeros)
#     print("[Task0] Waiting for IR sensors to become active...")
#     for _ in range(100):  # up to ~10 seconds
#         ir = list(rob.read_irs())
#         if any(v > 0 for v in ir):
#             print("[Task0] IR active:", ir)
#             break
#         time.sleep(0.1)
#     else:
#         print("[Task0] WARNING: IR stayed zero; continuing anyway (check app/robot awake).")

#     # baseline calibration (cuz we need to have some idea what "clear" is)
#     samples = []
#     for _ in range(30):
#         ir = list(rob.read_irs())
#         samples.append(front_signal(ir))
#         time.sleep(0.02)
#     base = sum(samples) / len(samples)

#     near_th  = base + near_offset
#     clear_th = base + clear_offset
#     print(f"base={base:.2f} near={near_th:.2f} clear={clear_th:.2f} front={front_band}")

#     mode = "forward"

#     for _ in range(steps):
#         ir = list(rob.read_irs())
#         fs = front_signal(ir)

#         if mode == "forward" and fs >= near_th:
#             mode = "turn"
#         elif mode == "turn" and fs <= clear_th:
#             mode = "forward"

#         if mode == "forward":
#             left, right = -forward, -forward     # forward = negative in this setup
#         else:
#             left, right = -turn, turn            # turn right

#         rob.move_blocking(left, right, int(dt * 1000))

#     rob.move_blocking(0, 0, 250)


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         raise ValueError("Pass --hardware or --simulation")

#     if sys.argv[1] == "--hardware":
#         rob = HardwareRobobo(camera=False)
#     elif sys.argv[1] == "--simulation":
#         rob = SimulationRobobo()
#     else:
#         raise ValueError("Use --hardware or --simulation")

#     task0(rob)


import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
from learning_machines import run_task_0



if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # run_all_actions(rob)
    run_task_0(rob)
