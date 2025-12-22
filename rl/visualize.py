import argparse
import json

import numpy as np

from rl.env import TractorEnv
from rl.util import build_algo, get_value, run_inference


def evaluate_agents(config_name: str, num_episodes: int = 1):
    """
    Evaluate trained agents using the NEW RLModule API (RLlib 2.10+).
    """

    with open(f"configs/{config_name}.json") as f:
        config = json.load(f)

    config["training"]["restore"] = True
    algo = build_algo(config_name, config)

    policy_id = "shared_policy"
    module = algo.get_module(policy_id)

    print(f"Using policy '{policy_id}' for evaluation.")

    env = TractorEnv()
    episode_rewards = []

    for ep in range(num_episodes):
        print("\n" + "=" * 60)
        print(f"Starting Episode {ep + 1}/{num_episodes}")
        print("=" * 60)

        # Reset environment
        obs, infos = env.reset()
        rewards_sum = {agent: 0.0 for agent in env.possible_agents}
        terminated = {agent: False for agent in env.possible_agents}

        step = 0
        while not all(terminated.values()):
            actions = {}
            print(f"\nStep {step}")

            for agent_id, agent_obs in obs.items():
                if not terminated[agent_id]:
                    action = run_inference(module, agent_obs)
                    print(f"Value function: {get_value(module, agent_obs)}")
                    actions[agent_id] = action
            print(f"Actions: {actions}")

            obs, rewards, terms, truncs, infos = env.step(actions)
            print(f"Rewards: {rewards}")

            for agent_id, r in rewards.items():
                rewards_sum[agent_id] += r
            for agent_id in env.possible_agents:
                terminated[agent_id] = terms.get(agent_id, False) or truncs.get(
                    agent_id, False
                )

            env.render()
            step += 1
            if step > 1000:
                print("WARNING: Episode exceeded 1000 steps force termination")
                break

        episode_rewards.append(rewards_sum)
        print(f"\nEpisode {ep + 1} Rewards: {rewards_sum}")

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for agent_id in env.possible_agents:
        avg_reward = np.mean(
            [episode_rewards[ep][agent_id] for ep in range(num_episodes)]
        )
        print(f"{agent_id}: Avg Reward = {avg_reward:.2f}")

    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    evaluate_agents(args.name, args.n)
