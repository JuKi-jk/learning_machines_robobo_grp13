#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions
from learning_machines import run_task_0
from learning_machines import run_task_1
from learning_machines import run_task_2
from learning_machines import run_task_3

if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 3:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify.
            sudo bash ./scripts/run.sh --simulation task1 <learn/predict> <model_name> <port>
            <model_name>: model name which one to save to when learning or to load from when predicting
            """
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        if len(sys.argv) > 5:
            rob = SimulationRobobo(api_port=(sys.argv[5]))
        else:
            rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    if sys.argv[2] == "test":
        run_all_actions(rob)
    elif sys.argv[2] == "task0":
        run_task_0(rob)
    elif sys.argv[2] == "task1":
        if len(sys.argv) > 3:
            if len(sys.argv) > 4:
                run_task_1(rob, mode=sys.argv[3], model_name=sys.argv[4])
            else:
                run_task_1(rob, mode=sys.argv[3])
        else:
            run_task_1(rob)
    elif sys.argv[2] == "task2":
        if len(sys.argv) > 3:
            if len(sys.argv) > 4:
                run_task_2(rob, mode=sys.argv[3], model_name=sys.argv[4])
            else:
                run_task_2(rob, mode=sys.argv[3])
        else:
            run_task_2(rob)
    elif sys.argv[2] == "task3":
        if len(sys.argv) > 3:
            if len(sys.argv) > 4:
                run_task_3(rob, mode=sys.argv[3], model_name=sys.argv[4])
            else:
                run_task_3(rob, mode=sys.argv[3])
        else:
            run_task_3(rob)