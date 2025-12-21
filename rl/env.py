import traceback
from typing import Any, override

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID

from game.action import Action, ActionType
from game.card import Card, NUM_CARDS, NUM_SUITS
from game.game import TractorGame, NUM_PLAYERS
from game.player import Player
from game.util import get_effective_rank, get_tractors, get_pairs


OBS_SIZE = (
    NUM_CARDS
    + NUM_SUITS
    + NUM_CARDS * (NUM_PLAYERS - 1)
    + NUM_PLAYERS
    + NUM_CARDS
    + NUM_CARDS * NUM_PLAYERS
)

REWARD_SCALE = 20.0


class TractorEnv(MultiAgentEnv):
    possible_agents: list[AgentID]
    agents: list[AgentID]
    game: TractorGame
    partial_selection: list[Card]
    hands_encodings: list[np.ndarray]
    hand_history_encodings: list[np.ndarray]
    trump_encoding: np.ndarray
    partial_selection_encoding: np.ndarray

    def __init__(self, config: None = None) -> None:
        super().__init__()

        self.possible_agents = list(range(NUM_PLAYERS))
        self.agents = self.possible_agents
        self.game = TractorGame()
        self.partial_selection = []
        self.hands_encodings = []
        self.hand_history_encodings = []
        self.trump_encoding = np.zeros((NUM_SUITS,))
        self.trump_encoding[self.game.trump_suit.value] = 1.0
        for i in range(NUM_PLAYERS):
            self.hands_encodings.append(np.zeros((NUM_CARDS,)))
            for card in self.game.players[i].hand:
                self.hands_encodings[i][card.get_index()] += 1.0
            self.hand_history_encodings.append(np.zeros((NUM_CARDS,)))
        self.partial_selection_encoding = np.zeros((NUM_CARDS,))

    @override
    def get_observation_space(
        self, agent_id: AgentID
    ) -> gym.Space[dict[str, np.ndarray]]:
        return gym.spaces.Dict(
            {
                "observations": gym.spaces.Box(
                    0.0, 10.0, shape=(OBS_SIZE,), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(
                    0.0, 1.0, (NUM_CARDS + 1,), dtype=np.float32
                ),
            }
        )

    @override
    def get_action_space(self, agent_id: AgentID) -> gym.Space[np.int64]:
        return gym.spaces.Discrete(NUM_CARDS + 1)

    def encode_state(self) -> np.ndarray:
        next_player_ind = self.game.current_hand.next_player

        obs = [self.hands_encodings[next_player_ind], self.trump_encoding]

        # Cards in current trick
        cum_size = 0
        trick_encoding = np.zeros(((NUM_PLAYERS - 1) * NUM_CARDS,))
        for offset in range(1, NUM_PLAYERS):
            other = self.game.current_hand.actions[
                (next_player_ind + offset) % NUM_PLAYERS
            ]
            if other is not None:
                for card in other.cards:
                    trick_encoding[cum_size + card.get_index()] += 1
            cum_size += NUM_CARDS
        obs.append(trick_encoding)

        # Lead player encoding
        lead_player_encoding = np.zeros((NUM_PLAYERS,))
        lead_player_encoding[self.game.current_hand.lead_player - next_player_ind] = 1.0
        obs.append(lead_player_encoding)

        # Partial selection encoding
        obs.append(self.partial_selection_encoding)

        # Hand history encoding
        for offset in range(NUM_PLAYERS):
            obs.append(self.hand_history_encodings[(next_player_ind + offset) % NUM_PLAYERS])

        return np.concat(obs)

    def get_lead_player_mask(self, player: Player) -> np.ndarray:
        mask = np.zeros((NUM_CARDS + 1,))

        if len(self.partial_selection) == 0:
            for card in player.hand:
                mask[card.get_index()] = 1
            return mask
        if len(self.partial_selection) == 1:
            mask[NUM_CARDS] = 1
            card1 = self.partial_selection[-1]
            seen_flag = False
            for card in player.cards_by_suit[self.game.get_effective_suit(card1)]:
                if card == card1:
                    if seen_flag:
                        mask[card.get_index()] = 1
                        break
                    else:
                        seen_flag = True
            return mask
        if len(self.partial_selection) == 2:
            card1 = self.partial_selection[-1]
            cards_available = player.cards_by_suit[
                self.game.get_effective_suit(card1)
            ].copy()
            for card in self.partial_selection:
                cards_available.remove(card)
            cards_available = sorted(cards_available, key=lambda x: x.get_index())
            mask[NUM_CARDS] = 1
            for i in range(len(cards_available) - 1):
                is_tractor = cards_available[i] == cards_available[i + 1]
                is_tractor = (
                    is_tractor
                    and abs(
                        get_effective_rank(cards_available[i], self.game.trump_suit)
                        - get_effective_rank(
                            self.partial_selection[0], self.game.trump_suit
                        )
                    )
                    == 1
                )
                is_tractor = is_tractor and self.game.get_effective_suit(
                    cards_available[i]
                ) == self.game.get_effective_suit(self.partial_selection[0])
                if is_tractor:
                    mask[cards_available[i].get_index()] = 1
            return mask
        if len(self.partial_selection) == 3:
            mask[self.partial_selection[-1].get_index()] = 1
            return mask
        mask[NUM_CARDS] = 1
        return mask

    def get_follow_player_mask(self, player: Player, lead_action: Action) -> np.ndarray:
        mask = np.zeros((NUM_CARDS + 1,))

        if len(self.partial_selection) == len(lead_action.cards):
            mask[NUM_CARDS] = 1
            return mask

        lead_suit = lead_action.get_suit(self.game.trump_suit)
        assert lead_suit is not None
        lead_action_class = lead_action.classify(lead_suit, self.game.trump_suit)

        suit_cards = player.cards_by_suit[lead_suit].copy()
        for card in self.partial_selection:
            if self.game.get_effective_suit(card) == lead_suit:
                suit_cards.remove(card)

        if suit_cards:
            tractors = get_tractors(
                player.cards_by_suit[lead_suit], self.game.trump_suit
            )
            pairs = get_pairs(player.cards_by_suit[lead_suit])
            num_selected = len(self.partial_selection)
            if lead_action_class.action_type == ActionType.TRACTOR:
                if len(tractors) > 0:
                    if num_selected == 0:
                        for tractor in tractors:
                            for pair in pairs:
                                if (
                                    get_effective_rank(pair, self.game.trump_suit)
                                    == tractor - 1
                                ):
                                    mask[pair.get_index()] = 1
                    elif num_selected % 2 == 1:
                        mask[self.partial_selection[-1].get_index()] = 1
                    else:
                        assert num_selected == 2
                        prev_card = self.partial_selection[-1]
                        for pair in pairs:
                            if (
                                get_effective_rank(pair, self.game.trump_suit)
                                == get_effective_rank(prev_card, self.game.trump_suit)
                                + 1
                            ):
                                mask[pair.get_index()] = 1
                    return mask
                elif len(pairs) >= 2:
                    if num_selected % 2 == 1:
                        mask[self.partial_selection[-1].get_index()] = 1
                    else:
                        for pair in pairs:
                            if pair in suit_cards:
                                mask[pair.get_index()] = 1
                    return mask
                elif len(pairs) == 1:
                    if num_selected == 0:
                        for pair in pairs:
                            mask[pair.get_index()] = 1
                    elif num_selected == 1:
                        mask[self.partial_selection[-1].get_index()] = 1
                    else:
                        for card in suit_cards:
                            mask[card.get_index()] = 1
                    return mask
            elif lead_action_class.action_type == ActionType.PAIR:
                if len(pairs) > 0:
                    if num_selected == 1:
                        mask[self.partial_selection[-1].get_index()] = 1
                    else:
                        assert num_selected == 0
                        for pair in pairs:
                            mask[pair.get_index()] = 1
                    return mask
            # Otherwise we have no pairs or tractors we are forced to play
            for card in suit_cards:
                mask[card.get_index()] = 1
        else:
            cards_available = player.hand.copy()
            for card in self.partial_selection:
                cards_available.remove(card)
            for card in cards_available:
                mask[card.get_index()] = 1
        return mask

    def get_action_mask(self) -> np.ndarray:
        next_player = self.game.current_hand.next_player
        lead_player = self.game.current_hand.lead_player

        if next_player == lead_player:
            return self.get_lead_player_mask(self.game.players[next_player])

        lead_action = self.game.current_hand.actions[lead_player]
        assert lead_action is not None

        return self.get_follow_player_mask(self.game.players[next_player], lead_action)

    def get_observation(self) -> dict[AgentID, dict[str, np.ndarray]]:
        next_player_ind = self.game.current_hand.next_player
        obs = self.encode_state()
        mask = self.get_action_mask()
        # print(f"Player {next_player_ind} action mask is {mask}")
        return {next_player_ind: {"observations": obs, "action_mask": mask}}

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentID, dict[str, np.ndarray]], dict[AgentID, dict[str, Any]]]:
        self.agents = self.possible_agents
        self.game = TractorGame()
        self.partial_selection = []
        self.hands_encodings = []
        self.hand_history_encodings = []
        self.trump_encoding = np.zeros((NUM_SUITS,))
        self.trump_encoding[self.game.trump_suit.value] = 1.0
        for i in range(NUM_PLAYERS):
            self.hands_encodings.append(np.zeros((NUM_CARDS,)))
            for card in self.game.players[i].hand:
                self.hands_encodings[i][card.get_index()] += 1.0
            self.hand_history_encodings.append(np.zeros((NUM_CARDS,)))
        self.partial_selection_encoding = np.zeros((NUM_CARDS,))
        return self.get_observation(), {}

    @override
    def step(self, action_dict: dict[AgentID, int]) -> tuple[
        dict[AgentID, dict[str, np.ndarray]],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        """
        Returns observation dict, rewards dict, termination/truncation dicts, and infos dict
        """

        try:
            rewards = {agent: 0.0 for agent in self.possible_agents}

            next_player = self.game.current_hand.next_player
            action = action_dict[next_player]
            if action == NUM_CARDS:
                action = Action(self.partial_selection)
                result = self.game.play_action(action)
                if result is not None:
                    score, winner = result
                    rewards[winner] += score / REWARD_SCALE
                    rewards[(winner + 2) % NUM_PLAYERS] += score / REWARD_SCALE
                    rewards[(winner + 1) % NUM_PLAYERS] -= score / REWARD_SCALE
                    rewards[(winner + 3) % NUM_PLAYERS] -= score / REWARD_SCALE
                self.hands_encodings[next_player] -= self.partial_selection_encoding
                self.hand_history_encodings[next_player] += self.partial_selection_encoding
                self.partial_selection = []
                self.partial_selection_encoding = np.zeros((NUM_CARDS,))
            else:
                self.partial_selection.append(Card.from_index(action))
                self.partial_selection_encoding[action] += 1.0

            terminateds = {
                agent: self.game.game_over() for agent in self.possible_agents
            }
            truncateds = {agent: False for agent in self.possible_agents}
            terminateds["__all__"] = self.game.game_over()
            truncateds["__all__"] = False

            return self.get_observation(), rewards, terminateds, truncateds, {}
        except Exception as e:
            self.render()
            traceback.print_exc()
            raise e

    @override
    def render(self, mode: str = "human") -> None:
        """Render human-readable game state."""
        if mode != "human":
            return

        print("\n" + "=" * 60)
        print(
            f"Trump: 2 of {self.game.trump_suit.name} | Host: Player {self.game.host}"
        )
        print(
            f"Defending team points: {self.game.defender_points} | Attacking team points: {self.game.attacker_points}"
        )
        print(f"Current Player: Player {self.game.current_hand.next_player}")

        for i, player in enumerate(self.game.players):
            if i == self.game.current_hand.next_player:
                print(
                    (
                        f"Player {i}: {len(player.hand)} cards"
                        f"| Partial Selection: {' '.join([str(c) for c in self.partial_selection])}\n"
                        f"Hand: {player}"
                    )
                )
            elif (action := self.game.current_hand.actions[i]) is not None:
                print(
                    (
                        f"Player {i}: {len(player.hand)} cards"
                        f"| Current Trick: {action}\n"
                        f"Hand: {player}"
                    )
                )
            else:
                print(
                    (
                        f"Player {i}: {len(player.hand)} cards"
                        f"| Current Trick: no cards\n"
                        f"Hand: {player}"
                    )
                )
        print("=" * 60 + "\n")
