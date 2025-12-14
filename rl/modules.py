import torch
from typing import Any

from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)


class ActionMaskingTorchRandomModule(ActionMaskingTorchRLModule):
    """Torch version of random action selection module."""

    def _compute_logits(self, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper method to compute logits from action mask.

        Args:
            action_mask: Action mask tensor.

        Returns:
            Logits tensor with -inf for masked actions, 0 for valid actions.
        """
        logits = torch.where(
            action_mask == 1.0,
            torch.zeros_like(action_mask),
            torch.full_like(action_mask, -1e10),
        )
        return logits

    def forward_inference(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        return {"action_dist_inputs": logits}

    def forward_exploration(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        return {"action_dist_inputs": logits}

    def forward_train(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        batch_size = action_mask.shape[0]
        vf_preds = torch.zeros(
            batch_size, dtype=torch.float32, device=action_mask.device
        )
        return {"action_dist_inputs": logits, "vf_preds": vf_preds}
