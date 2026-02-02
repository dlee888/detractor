import os
import random
from typing import Any

import numpy as np
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from game.card import Card
from rl.env import TractorEnv
from rl.modules import ActionMaskingTorchRandomModule, ActionMaskingTorchRLModule


TEAMMATE_SELF_PROB = 0.8


def make_policy_mapping_fn(
    opponent_pool: list[str] | None = None,
    random_opp_rate: float = 0.5,
    past_opp_rate: float = 0.5,
):
    episode_data = {}

    def _map(agent_id, episode, **kwargs):
        if id(episode) not in episode_data:
            episode_data[id(episode)] = {}
        data = episode_data[id(episode)]
        if agent_id % 2 == 0:
            if (
                agent_id == 2
                and opponent_pool
                and random.random() >= TEAMMATE_SELF_PROB
            ):
                data[f"chosen_opponent{agent_id}"] = random.choice(opponent_pool)
            else:
                data[f"chosen_opponent{agent_id}"] = "shared_policy"
            return data[f"chosen_opponent{agent_id}"]
        if f"chosen_opponent{agent_id}" not in data:
            roll = random.random()
            if roll < random_opp_rate:
                data[f"chosen_opponent{agent_id}"] = "random"
                data[f"chosen_opponent{(agent_id + 2) % 4}"] = "random"
            elif opponent_pool is not None and roll < random_opp_rate + past_opp_rate:
                data[f"chosen_opponent{agent_id}"] = random.choice(opponent_pool)
                data[f"chosen_opponent{(agent_id + 2) % 4}"] = random.choice(
                    opponent_pool
                )
            else:
                data[f"chosen_opponent{agent_id}"] = "shared_policy"
                if not opponent_pool or random.random() < TEAMMATE_SELF_PROB:
                    data[f"chosen_opponent{(agent_id + 2) % 4}"] = "shared_policy"
                else:
                    data[f"chosen_opponent{(agent_id + 2) % 4}"] = random.choice(
                        opponent_pool
                    )
        # print(f"chosen_opponent{agent_id}", data[f"chosen_opponent{agent_id}"])
        return data[f"chosen_opponent{agent_id}"]

    return _map


class SelfPlayWinRateCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        main_reward = sum(episode.agent_episodes[0].rewards)

        metrics_logger.log_value(
            f"main_vs_{episode.agent_episodes[1].module_id}", main_reward, reduce="mean"
        )
        metrics_logger.log_value(
            f"main_vs_{episode.agent_episodes[1].module_id[:4]}",
            main_reward,
            reduce="mean",
        )


def build_algo(
    name: str,
    run_config: dict[str, Any],
    random_opp_rate: float = 0.5,
    past_opp_rate: float = 0.5,
):
    random_spec = RLModuleSpec(
        module_class=ActionMaskingTorchRandomModule,
    )
    base_spec = RLModuleSpec(
        module_class=ActionMaskingTorchRLModule,
        model_config=run_config["model"],
    )
    opponent_ids = [f"self_{i}" for i in range(run_config["training"]["num_opponents"])]
    config = (
        PPOConfig()
        .environment(env=TractorEnv, disable_env_checking=True)
        .callbacks(SelfPlayWinRateCallback)
        .multi_agent(
            policies={"shared_policy", "random", *opponent_ids},
            policy_mapping_fn=make_policy_mapping_fn(
                opponent_ids, random_opp_rate, past_opp_rate
            ),
            policy_states_are_swappable=True,
            policies_to_train=["shared_policy"],
        )
        .training(**run_config["hyperparameters"])
        .env_runners(
            # create_local_env_runner=False,
            create_env_on_local_worker=False,
            **run_config["resources"],
        )
        .learners(
            **run_config["learner_resources"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": base_spec,
                    "random": random_spec,
                    **{oid: base_spec for oid in opponent_ids},
                }
            )
        )
    )
    algo = config.build()
    if run_config["training"]["restore"]:
        checkpoint_path = (
            run_config["training"]["restore_from"] or f"checkpoints/{name}"
        )
        if os.path.exists(checkpoint_path):
            algo.restore(os.path.abspath(checkpoint_path))
        else:
            print(f"WARNING: No checkpoint found at path {checkpoint_path}")
    return algo


def run_inference(module, agent_obs) -> int:
    # device = torch.device("cpu")
    device = torch.device("cuda")
    module.to(device)
    obs_batch = {
        "observations": torch.tensor(
            np.expand_dims(agent_obs["observations"], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.tensor(
            np.expand_dims(agent_obs["action_mask"], axis=0),
            dtype=torch.float32,
            device=device,
        ),
    }

    # Forward pass through the module
    inference_out = module.forward_inference({"obs": obs_batch})

    logits = inference_out["action_dist_inputs"]

    # Build the dist using RLlib's distribution class
    dist_class = module.get_exploration_action_dist_cls()
    action_dist = dist_class(logits)
    action = action_dist.sample().item()
    return action


def get_value(module, agent_obs) -> int:
    device = torch.device("cuda")
    module.to(device)
    obs_batch = {
        "obs": torch.tensor(
            np.expand_dims(agent_obs["observations"], axis=0),
            dtype=torch.float32,
            device=device,
        ),
    }

    # Forward pass through the module
    return module.compute_values(obs_batch)


def get_random_action(env: TractorEnv):
    possible_actions: list[list[int]] = []

    def enumerate(curr: list[int]):
        mask = env.get_action_mask()
        if mask[54] == 1.0:
            possible_actions.append(curr)
        for i in range(54):
            if mask[i] == 1.0:
                new = curr.copy()
                new.append(i)
                env.partial_selection.append(Card.from_index(i))
                enumerate(new)
                _ = env.partial_selection.pop(-1)

    enumerate([])

    return random.choice(possible_actions)
