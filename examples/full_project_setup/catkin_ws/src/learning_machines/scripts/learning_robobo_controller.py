#!/usr/bin/env python3
import argparse

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, run_ppo_maze, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--ppo-maze", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--steps-per-update", type=int, default=512)
    parser.add_argument("--action-ms", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--model-out", type=str, default="results/ppo_maze.pt")
    parser.add_argument("--model-in", type=str, default="results/ppo_maze.pt")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--no-explore-reward", action="store_true")
    parser.add_argument("--ir-clip", type=float, default=6000000.0)
    parser.add_argument("--near-ir", type=float, default=0.6)
    parser.add_argument("--collision-ir", type=float, default=0.9)
    parser.add_argument("--ir-scale-min", type=float, default=1.0)
    parser.add_argument("--ir-scale-max", type=float, default=1.0)
    parser.add_argument("--ir-noise-std", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.hardware and not args.simulation:
        raise ValueError("Pass `--hardware` or `--simulation` to specify the target.")
    if args.hardware and args.simulation:
        raise ValueError("Choose only one of `--hardware` or `--simulation`.")

    rob = HardwareRobobo(camera=True) if args.hardware else SimulationRobobo()

    if args.ppo_maze:
        config = PPOConfig(
            total_updates=args.updates,
            steps_per_update=args.steps_per_update,
            action_duration_ms=args.action_ms,
            max_episode_steps=args.max_episode_steps,
            model_out=args.model_out,
            allow_exploration_reward=not args.no_explore_reward,
            ir_clip=args.ir_clip,
            near_ir=args.near_ir,
            collision_ir=args.collision_ir,
            ir_scale_min=args.ir_scale_min,
            ir_scale_max=args.ir_scale_max,
            ir_noise_std=args.ir_noise_std,
        )
        if args.eval:
            config.allow_exploration_reward = False
            run_ppo_maze(
                rob,
                config,
                eval_only=True,
                model_in=args.model_in,
                eval_episodes=args.eval_episodes,
            )
        else:
            if args.hardware:
                config.allow_exploration_reward = False
            run_ppo_maze(rob, config)
    else:
        run_all_actions(rob)
