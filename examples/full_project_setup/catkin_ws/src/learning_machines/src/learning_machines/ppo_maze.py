from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from robobo_interface import IRobobo, SimulationRobobo


@dataclass
class PPOConfig:
    seed: int = 0
    action_duration_ms: int = 200
    max_episode_steps: int = 400
    steps_per_update: int = 512
    total_updates: int = 200
    epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    ir_clip: float = 6000000.0
    collision_ir: float = 0.9
    near_ir: float = 0.6
    ir_scale_min: float = 1.0
    ir_scale_max: float = 1.0
    ir_noise_std: float = 0.0
    collision_penalty: float = 2.0
    near_penalty: float = 0.5
    step_penalty: float = 0.01
    near_penalty: float = 0.5
    step_penalty: float = 0.01
    explore_reward: float = 1.0
    explore_cell_size: float = 0.25
    forward_reward: float = 0.01
    allow_exploration_reward: bool = True
    safety_stop_ms: int = 200
    model_out: str = "results/ppo_maze.pt"
    actions: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (50, 50),  # forward
            (30, 60),  # forward-left
            (60, 30),  # forward-right
            (-30, 30),  # turn left in place
            (30, -30),  # turn right in place
            (0, 0),  # stop
        ]
    )


class RoboboMazeEnv:
    def __init__(self, rob: IRobobo, config: PPOConfig) -> None:
        self.rob = rob
        self.config = config
        self.steps = 0
        self.visited = set()
        self._scale = 1.0
        self._initial_pos = None
        self._initial_orient = None
        if isinstance(rob, SimulationRobobo):
            self._initial_pos = rob.get_position()
            self._initial_orient = rob.get_orientation()

    def reset(self) -> np.ndarray:
        self._scale = float(
            np.random.uniform(self.config.ir_scale_min, self.config.ir_scale_max)
        )
        if isinstance(self.rob, SimulationRobobo) and self._initial_pos is not None:
            self.rob.set_position(self._initial_pos, self._initial_orient)
            self.rob.reset_wheels()
        self.steps = 0
        self.visited = set()
        return self._get_obs()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        left, right = self.config.actions[action_index]
        self.rob.move_blocking(left, right, self.config.action_duration_ms)
        obs = self._get_obs()
        reward, done = self._compute_reward(obs, left, right)
        if done and self.config.safety_stop_ms > 0:
            # Ensure the robot stops after a near-collision.
            self.rob.move_blocking(0, 0, self.config.safety_stop_ms)
        self.steps += 1
        if self.steps >= self.config.max_episode_steps:
            done = True
        return obs, reward, done

    def _get_obs(self) -> np.ndarray:
        irs = self.rob.read_irs()
        values = np.array([float(v) if v else 0.0 for v in irs], dtype=np.float32)
        if self.config.ir_noise_std > 0.0:
            values += np.random.normal(0.0, self.config.ir_noise_std, values.shape)
        values *= self._scale
        values = np.clip(values, 0.0, self.config.ir_clip) / self.config.ir_clip
        return values

    def _compute_reward(
        self, obs: np.ndarray, left_speed: int, right_speed: int
    ) -> Tuple[float, bool]:
        max_ir = float(np.max(obs))
        reward = -self.config.step_penalty
        done = False
        strans = float(left_speed + right_speed)
        if strans > 0:
            reward += self.config.forward_reward * strans

        if max_ir >= self.config.near_ir:
            reward -= self.config.near_penalty * (max_ir - self.config.near_ir)

        if max_ir >= self.config.collision_ir:
            reward -= self.config.collision_penalty
            done = True

        if (
            self.config.allow_exploration_reward
            and isinstance(self.rob, SimulationRobobo)
        ):
            pos = self.rob.get_position()
            cell = (
                int(pos.x / self.config.explore_cell_size),
                int(pos.y / self.config.explore_cell_size),
            )
            if cell not in self.visited:
                self.visited.add(cell)
                reward += self.config.explore_reward

        return reward, done


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def _compute_gae(
    rewards: List[float],
    dones: List[bool],
    values: List[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def run_ppo_maze(
    rob: IRobobo,
    config: PPOConfig,
    *,
    eval_only: bool = False,
    model_in: Optional[str] = None,
    eval_episodes: int = 3,
) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = RoboboMazeEnv(rob, config)
    obs = env.reset()
    obs_dim = obs.shape[0]
    action_dim = len(config.actions)

    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model_out = Path(config.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    if model_in is not None:
        model.load_state_dict(torch.load(model_in, map_location="cpu"))

    if eval_only:
        for ep in range(eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                with torch.no_grad():
                    logits, _value = model(obs_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, done = env.step(action)
                total_reward += reward
            print(f"eval_episode={ep + 1} total_reward={total_reward:.3f}")
        return

    for update in range(config.total_updates):
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []

        for _ in range(config.steps_per_update):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

            next_obs, reward, done = env.step(int(action.item()))

            batch_obs.append(obs)
            batch_actions.append(int(action.item()))
            batch_logprobs.append(float(logprob.item()))
            batch_rewards.append(float(reward))
            batch_dones.append(done)
            batch_values.append(float(value.item()))

            obs = env.reset() if done else next_obs

        with torch.no_grad():
            last_value = model(torch.from_numpy(obs).float().unsqueeze(0))[1].item()

        advantages, returns = _compute_gae(
            batch_rewards,
            batch_dones,
            batch_values,
            last_value,
            config.gamma,
            config.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.from_numpy(np.array(batch_obs)).float()
        actions_tensor = torch.from_numpy(np.array(batch_actions)).long()
        old_logprobs_tensor = torch.from_numpy(np.array(batch_logprobs)).float()
        returns_tensor = torch.from_numpy(returns).float()
        advantages_tensor = torch.from_numpy(advantages).float()

        for _ in range(config.epochs):
            indices = np.random.permutation(len(batch_obs))
            for start in range(0, len(batch_obs), config.minibatch_size):
                end = start + config.minibatch_size
                mb_idx = indices[start:end]

                logits, values = model(obs_tensor[mb_idx])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions_tensor[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs_tensor[mb_idx])
                clipped = torch.clamp(
                    ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                )
                policy_loss = -torch.min(
                    ratio * advantages_tensor[mb_idx],
                    clipped * advantages_tensor[mb_idx],
                ).mean()

                value_loss = (returns_tensor[mb_idx] - values).pow(2).mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (update + 1) % 10 == 0:
            avg_reward = float(np.mean(batch_rewards))
            print(
                f"update={update + 1} "
                f"avg_reward={avg_reward:.3f} "
                f"visited_cells={len(env.visited)}"
            )

    torch.save(model.state_dict(), model_out)
    print(f"Saved PPO model to {model_out}")
