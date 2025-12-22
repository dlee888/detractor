import argparse
import json
import tqdm

import numpy as np

from rl.env import TractorEnv
from rl.util import build_algo, get_random_action, run_inference


def evaluate_agents(config_name: str, num_episodes: int = 25):
    with open(f"configs/{config_name}.json") as f:
        config = json.load(f)

    config["training"]["restore"] = True
    algo = build_algo(config_name, config)

    policy_id = "shared_policy"
    module = algo.get_module(policy_id)

    print(f"Using policy '{policy_id}' for evaluation.")

    env = TractorEnv()
    episode_rewards: list[float] = []
    wins = 0
    wins_def, num_def = 0, 0
    wins_att, num_att = 0, 0

    for ep in tqdm.trange(num_episodes):
        obs, infos = env.reset()
        rewards_sum = {agent: 0.0 for agent in env.possible_agents}
        terminated = {agent: False for agent in env.possible_agents}

        step = 0
        while not env.game.game_over():
            actions = {}
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
                    obs, _, terminated, _, _ = env.step(
                        {env.game.current_hand.next_player: c}
                    )
                obs, rewards, terminated, _, _ = env.step(
                    {env.game.current_hand.next_player: 54}
                )
                for agent_id, r in rewards.items():
                    rewards_sum[agent_id] += r

            step += 1
            if step > 1000:
                print("WARNING: Episode exceeded 1000 steps force termination")
                break

        assert rewards_sum[0] == rewards_sum[2]
        episode_rewards.append(rewards_sum[0])
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
        f"Average reward: {np.mean(episode_rewards)} \u00b1 {np.std(episode_rewards) / np.sqrt(len(episode_rewards))}"
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
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    evaluate_agents(args.name, args.n)
