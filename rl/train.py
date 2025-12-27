import argparse
import json
import os
import time
from pprint import pprint
from typing import Any

from rl.util import build_algo


random_opp_rate = 0.2
past_opp_rate = 0.8


def main(name: str, run_config: dict[str, Any]) -> None:
    global random_opp_rate, past_opp_rate

    algo = build_algo(name, run_config, random_opp_rate, past_opp_rate)

    def update_module_on_runner(env_runner):
        for o, s in states.items():
            env_runner.module[o].set_state(s)
        return True

    print("Starting training...")
    num_iterations = run_config["training"]["iterations"]
    checkpoint_interval = run_config["training"]["checkpoint_interval"]
    num_opps = run_config["training"]["num_opponents"]

    states = {}
    init_weights = algo.get_module("shared_policy").get_state()
    for i in range(num_opps):
        states[f"self_{i}"] = init_weights
    algo.env_runner_group.foreach_env_runner(update_module_on_runner)

    try:
        log_file_path = f"episode_rewards_{name}.csv"
        with open(log_file_path, "w") as f:
            f.write("iteration,episode_return_mean,timestamp\n")

            for i in range(num_iterations):
                starttime = time.time()
                result = algo.train()

                print(f"\n{'=' * 60}")
                print(f"Iteration {i + 1}/{num_iterations}")
                print(f"{'=' * 60}")
                print(f"{time.time() - starttime:.3f}s taken")
                stats = result.get("env_runners", {})
                episode_returns_mean = stats.get("agent_episode_returns_mean", {})
                episode_return_mean = episode_returns_mean.get("0", "N")
                print(f"Episode Reward Mean: {episode_return_mean}")
                print(f"Episode Length Mean: {stats.get('episode_len_mean', 'N/A')}")
                print(f"Episodes This Iter: {stats.get('num_episodes', 'N/A')}")
                print(f"Env Steps Sampled: {stats.get('num_env_steps_sampled', 'N/A')}")

                # Log to file (only if it's a real number)
                if isinstance(episode_return_mean, (int, float)):
                    f.write(f"{i+1},{episode_return_mean},{time.time()}\n")
                    f.flush()
                else:
                    print(f"Failed to log reward: {type(episode_return_mean)}")

                if (i + 1) % checkpoint_interval == 0:
                    checkpoint = algo.save(os.path.abspath(f"checkpoints/{name}"))
                    print(f"\nCheckpoint saved at: {checkpoint.checkpoint}")
                    # random_opp_rate *= 0.95
                    # past_opp_rate = 1 - random_opp_rate
                    # print(f"New random opp rate: {random_opp_rate}")
                    curr_weights = algo.get_module("shared_policy").get_state()
                    opp_id = ((i + 1) // checkpoint_interval) % num_opps
                    opp = f"self_{opp_id}"
                    states[opp] = curr_weights
                    algo.get_module(opp).set_state(curr_weights)
                    algo.env_runner_group.foreach_env_runner(update_module_on_runner)
                    print(f"Loaded state into {opp}")
    except KeyboardInterrupt:
        pass
    finally:
        final_checkpoint = algo.save(os.path.abspath(f"checkpoints/{name}"))
        print(f"\nFinal checkpoint saved at: {final_checkpoint.checkpoint}")
        pprint(f"{final_checkpoint.metrics}")
        algo.stop()
        print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    with open(f"configs/{args.name}.json") as f:
        config = json.load(f)
    main(args.name, config)
