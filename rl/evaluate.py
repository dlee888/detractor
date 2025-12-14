import argparse
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from rl.env import TractorEnv
from rl.util import get_random_action, run_inference
from rl.train import make_policy_mapping_fn
from rl.modules import ActionMaskingTorchRandomModule


def evaluate_agents(checkpoint_path: str, num_episodes: int = 25):
    """
    Evaluate trained agents using the NEW RLModule API (RLlib 2.10+).
    """

    print(f"Restoring PPO checkpoint: {checkpoint_path}")
    config = (
        PPOConfig()
        .environment(env=TractorEnv, disable_env_checking=True)
        .multi_agent(
            policies={"shared_policy", "random"},
            policy_mapping_fn=make_policy_mapping_fn(),
            policy_states_are_swappable=True,
            policies_to_train=["shared_policy"],
        )
        .training(
            lr=2e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.001,
            train_batch_size=20000,
            minibatch_size=128,
            num_epochs=3,
        )
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=32,
            num_gpus_per_env_runner=0.125,
            rollout_fragment_length=64,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config={
                            "fcnet_hiddens": [512, 256],
                            "fcnet_activation": "relu",
                        },
                    ),
                    "random": RLModuleSpec(
                        module_class=ActionMaskingTorchRandomModule,
                    ),
                }
            )
        )
    )

    algo = config.build()
    algo.restore(checkpoint_path)
    # algo = PPO.from_checkpoint(checkpoint_path, load_state=False)

    policy_id = "shared_policy"
    module = algo.get_module(policy_id)

    print(f"Using policy '{policy_id}' for evaluation.")

    # -------------------------------------------------
    # 4. Run evaluation episodes
    # -------------------------------------------------
    env = TractorEnv()
    episode_rewards: list[float] = []
    wins = 0
    wins_def, num_def = 0, 0
    wins_att, num_att = 0, 0

    for ep in range(num_episodes):
        obs, infos = env.reset()
        rewards_sum = {agent: 0.0 for agent in env.possible_agents}
        terminated = {agent: False for agent in env.possible_agents}

        step = 0
        while not env.game.game_over():
            actions = {}

            # -------------------------------------------------
            # Compute actions for each agent
            # -------------------------------------------------
            if env.game.current_hand.next_player % 2 == 0:
                for agent_id, agent_obs in obs.items():
                    if not terminated[agent_id]:
                        action = run_inference(module, agent_obs)
                        actions[agent_id] = action
                obs, rewards, terms, truncs, infos = env.step(actions)

                for agent_id, r in rewards.items():
                    rewards_sum[agent_id] += r
                for agent_id in env.possible_agents:
                    terminated[agent_id] = terms.get(agent_id, False) or truncs.get(
                        agent_id, False
                )
            else:
                action = get_random_action(env)
                for c in action:
                    obs, _, terminated, _, _ = env.step({env.game.current_hand.next_player: c})
                obs, rewards, terminated, _, _ = env.step({env.game.current_hand.next_player: 54})
                for agent_id, r in rewards.items():
                    rewards_sum[agent_id] += r

            step += 1
            if step > 1000:
                print("WARNING: Episode exceeded 1000 steps force termination")
                break

        episode_rewards.append(rewards_sum[0] + rewards_sum[2])
        if env.game.host % 2 == 0:
            wins += env.game.attacker_points < 80
            wins_def += env.game.attacker_points < 80
            num_def += 1
        else:
            wins += env.game.attacker_points >= 80
            wins_att += env.game.attacker_points >= 80
            num_att += 1

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(
        f"Average reward: {5 * np.mean(episode_rewards) + 100} \u00b1 {np.std(episode_rewards) / np.sqrt(len(episode_rewards)) * 5}"
    )
    wr = wins / num_episodes
    wr_std = np.sqrt(wr * (1 - wr) / num_episodes)
    print(
        f"Win rate: {wr} \u00b1 {wr_std}. Win rate when defending: {wins_def / num_def}, win rate when attacking: {wins_att / num_att}"
    )
    # Cleanup
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    evaluate_agents(args.path, 1000)
