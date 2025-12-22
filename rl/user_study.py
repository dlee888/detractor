import argparse
import json
import os
import random
import traceback

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from game.action import Action
from game.card import NUM_CARDS, Card, Rank, Suit
from game.game import NUM_PLAYERS
from game.hand import Hand
from rl.env import TractorEnv
from rl.modules import ActionMaskingTorchRandomModule
from rl.train import make_policy_mapping_fn
from rl.util import run_inference

RANKS = "3456789TJQKA2jJ"
SUITS = "CDHS"


def parse_card(card_name: str) -> Card:
    rank = RANKS.index(card_name[0])
    if len(card_name) == 1:
        assert card_name.lower() == "j"
        suit = 0
    else:
        suit = SUITS.index(card_name[1])
    return Card(rank=Rank(rank), suit=Suit(suit))


def parse_cards(cards: str, expected_len: int | None = None) -> list[Card]:
    res: list[Card] = []
    for card in cards.split():
        res.append(parse_card(card))
    if expected_len is not None:
        assert len(res) == expected_len
    return res


def input_value(prompt: str, validator, submitter=None):
    if submitter is None:
        submitter = validator
    while True:
        try:
            print(prompt, end="")
            inp = input()
            validator(inp)
            return submitter(inp)
        except Exception as e:
            print(e)


def validate_lead(inp):
    assert 0 <= int(inp) < 4


def parse_yn(inp):
    if inp.lower() == "y" or inp.lower() == "yes":
        return True
    if inp.lower() == "n" or inp.lower() == "no":
        return False
    raise ValueError


def run_user_study(checkpoint_path: str, run_name: str):
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

    policy_id = "shared_policy"
    module = algo.get_module(policy_id)

    print(f"Using policy '{policy_id}' for evaluation.")

    os.makedirs(f"user_studies/{run_name}")

    with open(f"user_studies/{run_name}/events.log", "a") as f:
        f.write("========= BEGIN USER STUDY LOG ==========\n")
        env = TractorEnv()
        trump_suit = input_value(
            "Enter trump suit: ", lambda inp: Suit(SUITS.index(inp))
        )
        f.write(f"Trump suit: {trump_suit.name}\n")
        hand = input_value("Enter hand: ", lambda inp: parse_cards(inp, 25))
        f.write(f"Player hand: {hand}\n")
        for card in env.game.players[0].hand.copy():
            env.game.players[0].remove_card(card, env.game.trump_suit)
        for card in hand:
            env.game.players[0].add_card(card, trump_suit)
        env.game.trump_suit = trump_suit

        lead = input_value(
            "Who is first to act? Enter as places clockwise from bot: ",
            validate_lead,
            lambda inp: int(inp),
        )
        assert lead is not None

        collaborate = [0, 0]
        rewards = [0, 0]
        total = [0, 0]

        try:
            while True:
                if input_value("Is game over? [y/n] ", parse_yn):
                    break
                f.write(f"First to act: {lead}\n")
                env.game.current_hand = Hand(NUM_PLAYERS, lead)
                while env.game.current_hand.next_player != 0:
                    hand_played = input_value(
                        f"Enter player {env.game.current_hand.next_player}'s action: ",
                        parse_cards,
                    )
                    f.write(
                        f"Player {env.game.current_hand.next_player} played {hand_played}\n"
                    )
                    env.game.current_hand.actions[env.game.current_hand.next_player] = (
                        Action(hand_played)
                    )
                    env.game.current_hand.next_player = (
                        env.game.current_hand.next_player + 1
                    ) % NUM_PLAYERS
                bot_play: list[Card] = []
                while True:
                    obs = env.get_observation()
                    action = run_inference(module, obs[0])
                    if action == NUM_CARDS:
                        break
                    bot_play.append(Card.from_index(action))
                    env.partial_selection.append(Card.from_index(action))
                f.write(f"Bot played {bot_play}\n")
                human_play = input_value("Enter human action: ", parse_cards)
                f.write(f"Human played {human_play}\n")
                # choose_bot = random.randint(0, 1)
                choose_bot = 1
                if choose_bot == 1:
                    print(f"Resulting play: {bot_play}")
                    f.write("Chose bot play\n")
                    env.game.current_hand.actions[0] = Action(bot_play)
                else:
                    print(f"Resulting play: {human_play}")
                    f.write("Chose human play\n")
                    env.game.current_hand.actions[0] = Action(human_play)
                env.game.current_hand.next_player = (
                    env.game.current_hand.next_player + 1
                ) % NUM_PLAYERS
                env.partial_selection = []
                for card in env.game.current_hand.actions[0].cards:
                    env.game.players[0].remove_card(card, trump_suit)

                while not env.game.current_hand.is_complete():
                    hand_played = input_value(
                        f"Enter player {env.game.current_hand.next_player}'s action: ",
                        parse_cards,
                    )
                    f.write(
                        f"Player {env.game.current_hand.next_player} played {hand_played}\n"
                    )
                    env.game.current_hand.actions[env.game.current_hand.next_player] = (
                        Action(hand_played)
                    )
                    env.game.current_hand.next_player = (
                        env.game.current_hand.next_player + 1
                    ) % NUM_PLAYERS
                env.game.hand_history.append(env.game.current_hand)
                lead = input_value(
                    "Who won? Enter as places clockwise from bot: ",
                    validate_lead,
                    lambda inp: int(inp),
                )
                assert lead is not None
                f.write(f"Finished current hand with winner {lead}\n")
                is_collaborative = input_value(
                    "For player 2, did you understand the hand played? ", parse_yn
                )
                f.write(f"Player 2 understood the hand played: {is_collaborative}\n")
                collaborate[choose_bot] += is_collaborative
                if lead % 2 == 0:
                    rewards[choose_bot] += env.game.current_hand.points()
                total[choose_bot] += 1
            f.write("========= END USER STUDY LOG ==========\n")
        except:
            traceback.print_exc()
        finally:
            with open(f"user_studies/{run_name}/results.json", "w") as g:
                json.dump(
                    {"collaborate": collaborate, "rewards": rewards, "totals": total},
                    g,
                    indent=2,
                )

    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    run_user_study(args.path, args.run_name)
