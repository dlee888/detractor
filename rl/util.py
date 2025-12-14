import random
import torch
import numpy as np

from game.card import Card
from rl.env import TractorEnv


def run_inference(module, agent_obs) -> int:
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
