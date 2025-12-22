import argparse
import json
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Any

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from rl.env import TractorEnv
from rl.modules import ActionMaskingTorchRLModule
from rl.util import build_algo, make_policy_mapping_fn


def checkpoint_module_spec(path: str, observation_space, action_space) -> RLModuleSpec:
    ckpt = Path(path).expanduser()
    return RLModuleSpec(
        module_class=ActionMaskingTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            "fcnet_hiddens": [1024, 1024, 1024],
            "fcnet_activation": "relu",
        },
        load_state_path=str(ckpt / "learner_group" / "rl_module_state.pkl"),
    )


random_opp_rate = 0.5
past_opp_rate = 0.5


def main(name: str, run_config: dict[str, Any]) -> None:
    global random_opp_rate, past_opp_rate

    algo = build_algo(run_config)

    print("Starting training...")
    num_iterations = run_config["training"]["iterations"]

    # Open a log file once for the whole run
    opponent_pool = []
    MAX_POOL = 25
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
                # pprint(stats)
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

                if (i + 1) % 50 == 0:
                    checkpoint = algo.save(os.path.abspath(f"checkpoints/{name}"))
                    print(f"\nCheckpoint saved at: {checkpoint.checkpoint}")
                    random_opp_rate *= 0.95
                    past_opp_rate = 1 - random_opp_rate
                    print(f"New random opp rate: {random_opp_rate}")
                    new_policy = f"self_t{i}"
                    env = TractorEnv()
                    checkpoint_spec = checkpoint_module_spec(
                        checkpoint.checkpoint.path,
                        env.get_observation_space(0),
                        env.get_action_space(0),
                    )
                    opponent_pool.append(new_policy)
                    while len(opponent_pool) > MAX_POOL:
                        old_module = opponent_pool.pop(0)
                        algo.remove_module(module_id=old_module)
                    algo.add_module(
                        module_id=new_policy,
                        module_spec=checkpoint_spec,
                        new_agent_to_module_mapping_fn=make_policy_mapping_fn(
                            opponent_pool, random_opp_rate, past_opp_rate
                        ),
                    )
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
