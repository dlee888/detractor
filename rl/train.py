import random
import time
from pprint import pprint
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.utils.framework import get_device

from rl.env import TractorEnv
from rl.modules import ActionMaskingTorchRandomModule


def checkpoint_module_spec(path: str, observation_space, action_space) -> RLModuleSpec:
    ckpt = Path(path).expanduser()
    return RLModuleSpec(
        module_class=ActionMaskingTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "relu",
        },
        load_state_path=str(ckpt / "learner_group" / "rl_module_state.pkl"),
    )


RANDOM_OPP_RATE = 0.2
PAST_OPP_RATE = 0.8
TEAMMATE_SELF_PROB = 0.7


def make_policy_mapping_fn(opponent_pool: list[str] | None = None):
    episode_data = {}

    def _map(agent_id, episode, **kwargs):
        if id(episode) not in episode_data:
            episode_data[id(episode)] = {}
        data = episode_data[id(episode)]
        if agent_id % 2 == 0:
            if agent_id == 2 and opponent_pool and random.random() >= TEAMMATE_SELF_PROB:
                data[f"chosen_opponent{agent_id}"] = random.choice(opponent_pool)
            else:
                data[f"chosen_opponent{agent_id}"] = "shared_policy"
            return data[f"chosen_opponent{agent_id}"]
        if f"chosen_opponent{agent_id}" not in data:
            roll = random.random()
            if roll < RANDOM_OPP_RATE:
                data[f"chosen_opponent{agent_id}"] = "random"
            elif opponent_pool is not None and roll < RANDOM_OPP_RATE + PAST_OPP_RATE:
                data[f"chosen_opponent{agent_id}"] = random.choice(opponent_pool)
            else:
                data[f"chosen_opponent{agent_id}"] = "shared_policy"
        return data[f"chosen_opponent{agent_id}"]

    return _map


def main(params, name: str, checkpoint_path: str = "") -> None:
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
            num_epochs=2,
        )
        .env_runners(
            create_local_env_runner=False,
            create_env_on_local_worker=False,
            num_env_runners=8,
            num_envs_per_env_runner=8,
            num_gpus_per_env_runner=0.125,
            rollout_fragment_length=64,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config={
                            "fcnet_hiddens": [1024, 1024],
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
    if checkpoint_path:
        algo.restore(checkpoint_path)

    print("Starting training...")
    num_iterations = params["iter"]

    # Open a log file once for the whole run
    opponent_pool = []
    MAX_POOL = 20
    try:
        log_file_path = f"episode_rewards_{name}.csv"
        with open(log_file_path, "w") as f:
            f.write("iteration,episode_return_mean\n")

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
                episode_return_mean = episode_returns_mean.get(
                    "0", "N"
                ) + episode_returns_mean.get("2", "/A")
                print(f"Episode Reward Mean: {episode_return_mean}")
                print(f"Episode Length Mean: {stats.get('episode_len_mean', 'N/A')}")
                print(f"Episodes This Iter: {stats.get('num_episodes', 'N/A')}")
                print(f"Env Steps Sampled: {stats.get('num_env_steps_sampled', 'N/A')}")

                # Log to file (only if it's a real number)
                if isinstance(episode_return_mean, (int, float)):
                    f.write(f"{i+1},{episode_return_mean}\n")
                    f.flush()
                else:
                    print(f"Failed to log reward: {type(episode_return_mean)}")

                if (i + 1) % 20 == 0:
                    checkpoint = algo.save()
                    print(f"\nCheckpoint saved at: {checkpoint.checkpoint}")
                    new_policy = f"self_t{i}"
                    env = TractorEnv()
                    checkpoint_spec = checkpoint_module_spec(
                        checkpoint.checkpoint.path,
                        env.get_observation_space(0),
                        env.get_action_space(0),
                    )
                    opponent_pool.append(new_policy)
                    if len(opponent_pool) > MAX_POOL:
                        opponent_pool = opponent_pool[-MAX_POOL:]
                    algo.add_module(
                        module_id=new_policy,
                        module_spec=checkpoint_spec,
                        new_agent_to_module_mapping_fn=make_policy_mapping_fn(
                            opponent_pool
                        ),
                    )
    except KeyboardInterrupt:
        pass
    finally:
        final_checkpoint = algo.save(
            f"/home/dlee888/projects/human_tractor/finding-friends/checkpoints/{name}"
        )
        print(f"\nFinal checkpoint saved at: {final_checkpoint.checkpoint}")
        pprint(f"{final_checkpoint.metrics}")
        algo.stop()
        print("\nTraining complete!")


if __name__ == "__main__":
    main({"iter": 10000}, "OP_1024_1024")
