import argparse
import json
import tqdm

import numpy as np

from rl.env import TractorEnv
from rl.modules import ActionMaskingHeuristicsModule
from rl.util import build_algo, get_random_action, run_inference


def evaluate_agents(config_name: str, num_episodes: int = 25, heuristic: bool = False):
    with open(f"configs/{config_name}.json") as f:
        config = json.load(f)

    config["training"]["restore"] = True
    algo = build_algo(config_name, config)

    policy_id = "shared_policy"
    module = algo.get_module(policy_id)

    print(f"Using policy '{policy_id}' for evaluation.")

    env = TractorEnv()
    heuristic_module = ActionMaskingHeuristicsModule(
        observation_space=env.get_observation_space(0),
        action_space=env.get_action_space(0),
        model_config={},
    )

    episode_rewards: list[float] = []
    wins = 0
    wins_def, num_def = 0, 0
    wins_att, num_att = 0, 0
    score_def, score_att = [], []

    for _ in tqdm.trange(num_episodes):
        obs, _ = env.reset()
        rewards_sum = {agent: 0.0 for agent in env.possible_agents}

        def step_and_track_reward(actions):
            obs, rewards, _, _, _ = env.step(actions)

            for agent_id, r in rewards.items():
                rewards_sum[agent_id] += r
            return obs

        step = 0
        while not env.game.game_over():
            actions = {}
            if env.game.current_hand.next_player % 2 == 0:
                for agent_id, agent_obs in obs.items():
                    action = run_inference(module, agent_obs)
                    actions[agent_id] = action
                obs = step_and_track_reward(actions)
            else:
                if heuristic:
                    for agent_id, agent_obs in obs.items():
                        action = run_inference(heuristic_module, agent_obs)
                        actions[agent_id] = action
                    obs = step_and_track_reward(actions)
                else:
                    action = get_random_action(env)
                    for c in action:
                        obs = step_and_track_reward({env.game.current_hand.next_player: c})
                    obs = step_and_track_reward({env.game.current_hand.next_player: 54})

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
            score_def.append(env.game.defender_points)
        else:
            wins += env.game.attacker_points >= 80
            wins_att += env.game.attacker_points >= 80
            num_att += 1
            score_att.append(env.game.attacker_points)

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
    print(
        f"Average score when defending: {np.mean(score_def)}, average_score when attacking: {np.mean(score_att)}"
    )
    # Cleanup
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--heuristic", action="store_true")
    args = parser.parse_args()
    evaluate_agents(args.name, args.n, args.heuristic)
