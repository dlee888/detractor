"""Game session management for DeTractor web interface."""

import json
import random
from enum import Enum
from typing import Any

import numpy as np

from game.action import Action
from game.card import NUM_CARDS, Card, Suit
from game.game import NUM_PLAYERS
from rl.env import TractorEnv


class PlayerType(Enum):
    HUMAN = "human"
    RANDOM = "random"
    HEURISTIC = "heuristic"
    PPO = "ppo"

    @classmethod
    def from_string(cls, s: str) -> "PlayerType":
        return cls(s.lower())


class GameSession:
    """Manages a game session with bots and history tracking.

    Wraps TractorEnv to reuse its action mask and game logic.
    """

    def __init__(self, player_types: list[PlayerType], seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        self.player_types = player_types
        self.env = TractorEnv()
        self.env.reset()
        self.history: list[dict[str, Any]] = []
        self.state_history: list[bytes] = []
        self.last_player: int = -1

        self._save_state()

    @property
    def game(self):
        return self.env.game

    @property
    def partial_selection(self):
        return self.env.partial_selection

    def _save_state(self):
        """Save current state for undo."""
        state = {
            "game": self._serialize_game(),
            "partial_selection": [c.get_index() for c in self.env.partial_selection],
            "hands_encodings": [e.tolist() for e in self.env.hands_encodings],
            "hand_history_encodings": [e.tolist() for e in self.env.hand_history_encodings],
            "partial_selection_encoding": self.env.partial_selection_encoding.tolist(),
            "out_encoding": self.env.out_encoding.tolist(),
        }
        self.state_history.append(json.dumps(state).encode())

    def _serialize_game(self) -> dict:
        """Serialize game state to dict."""
        return {
            "players": [[c.get_index() for c in p.hand] for p in self.game.players],
            "kitty": [c.get_index() for c in self.game.kitty],
            "trump_suit": self.game.trump_suit.value,
            "host": self.game.host,
            "attacker_points": self.game.attacker_points,
            "defender_points": self.game.defender_points,
            "current_hand": {
                "lead_player": self.game.current_hand.lead_player,
                "next_player": self.game.current_hand.next_player,
                "actions": [
                    [c.get_index() for c in a.cards] if a else None
                    for a in self.game.current_hand.actions
                ],
            },
            "hand_history": [
                {
                    "lead_player": h.lead_player,
                    "actions": [
                        [c.get_index() for c in a.cards] if a else None
                        for a in h.actions
                    ],
                }
                for h in self.game.hand_history
            ],
        }

    def _restore_state(self, state: dict):
        """Restore env state from serialized dict."""
        from game.hand import Hand
        from game.player import Player

        # Restore game
        game = self.game
        game.trump_suit = Suit(state["game"]["trump_suit"])
        game.host = state["game"]["host"]
        game.attacker_points = state["game"]["attacker_points"]
        game.defender_points = state["game"]["defender_points"]
        game.kitty = [Card.from_index(i) for i in state["game"]["kitty"]]

        # Restore players
        for i, hand_indices in enumerate(state["game"]["players"]):
            player = Player(i)
            for idx in hand_indices:
                player.add_card(Card.from_index(idx), game.trump_suit)
            game.players[i] = player

        # Restore current hand
        ch_data = state["game"]["current_hand"]
        game.current_hand = Hand(NUM_PLAYERS, ch_data["lead_player"])
        game.current_hand.next_player = ch_data["next_player"]
        for i, action_data in enumerate(ch_data["actions"]):
            if action_data is not None:
                game.current_hand.actions[i] = Action([Card.from_index(idx) for idx in action_data])

        # Restore hand history
        game.hand_history = []
        for h_data in state["game"]["hand_history"]:
            hand = Hand(NUM_PLAYERS, h_data["lead_player"])
            for i, action_data in enumerate(h_data["actions"]):
                if action_data is not None:
                    hand.actions[i] = Action([Card.from_index(idx) for idx in action_data])
            game.hand_history.append(hand)

        # Restore env state
        self.env.partial_selection = [Card.from_index(i) for i in state["partial_selection"]]
        self.env.hands_encodings = [np.array(e) for e in state["hands_encodings"]]
        self.env.hand_history_encodings = [np.array(e) for e in state["hand_history_encodings"]]
        self.env.partial_selection_encoding = np.array(state["partial_selection_encoding"])
        self.env.out_encoding = np.array(state["out_encoding"])

    def undo(self) -> bool:
        """Undo to previous state."""
        if len(self.state_history) <= 1:
            return False
        self.state_history.pop()
        state = json.loads(self.state_history[-1].decode())
        self._restore_state(state)
        return True

    def clear_selection(self):
        """Clear partial selection by restoring current saved state."""
        if self.state_history:
            state = json.loads(self.state_history[-1].decode())
            self._restore_state(state)

    def is_game_over(self) -> bool:
        return self.game.game_over()

    def is_bot_turn(self) -> bool:
        current = self.game.current_hand.next_player
        return self.player_types[current] != PlayerType.HUMAN

    def get_current_player(self) -> int:
        return self.game.current_hand.next_player

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask using TractorEnv's logic."""
        return self.env.get_action_mask()

    def play_human_action(self, card_indices: list[int]) -> dict[str, Any] | None:
        """Play cards selected by human."""
        self.last_player = self.game.current_hand.next_player

        for idx in card_indices:
            action_dict = {self.last_player: idx}
            self.env.step(action_dict)

        # Check if we just committed (partial_selection is now empty after a commit)
        if len(self.env.partial_selection) == 0:
            self._save_state()
            self.history.append({
                "player": self.last_player,
                "cards": card_indices,
                "type": "human",
            })

        return None

    def sandbox_play(self, player_idx: int, card_indices: list[int]) -> None:
        """Play cards for any player in sandbox mode.

        Ensures cards are in the player's hand, then plays through the engine
        so tricks resolve, scores update, and cards are removed automatically.
        """
        current = self.game.current_hand.next_player
        if current != player_idx:
            raise ValueError(f"It's P{current}'s turn, not P{player_idx}'s")

        # Ensure each card is in the player's hand
        for idx in card_indices:
            card = Card.from_index(idx)
            if card not in self.game.players[player_idx].hand:
                self.game.players[player_idx].add_card(card, self.game.trump_suit)
                self.env.hands_encodings[player_idx][idx] += 1.0

        self.last_player = player_idx

        # Play each card through the engine (adds to partial_selection)
        for idx in card_indices:
            self.env.step({player_idx: idx})

        # Commit (resolves trick if all 4 have played)
        self.env.step({player_idx: NUM_CARDS})

        self._save_state()
        self.history.append({
            "player": player_idx,
            "cards": card_indices,
            "type": "sandbox",
        })

    def play_bot_turn(self) -> tuple[dict[str, Any], list[Card]]:
        """Execute a bot's turn."""
        current_player = self.game.current_hand.next_player
        player_type = self.player_types[current_player]
        self.last_player = current_player

        all_cards = []

        while True:
            mask = self._get_action_mask()

            if player_type == PlayerType.RANDOM:
                action_idx = self._random_action(mask)
            elif player_type == PlayerType.HEURISTIC:
                action_idx = self._heuristic_action(mask)
            elif player_type == PlayerType.PPO:
                action_idx = self._ppo_action(mask)
            else:
                raise ValueError(f"Unknown player type: {player_type}")

            if action_idx == NUM_CARDS:
                # Commit
                self.env.step({current_player: NUM_CARDS})
                break
            else:
                card = Card.from_index(action_idx)
                all_cards.append(card)
                self.env.step({current_player: action_idx})

        self._save_state()

        thinking = self._generate_thinking(player_type, all_cards)
        self.history.append({
            "player": self.last_player,
            "cards": [c.get_index() for c in all_cards],
            "type": player_type.value,
            "thinking": thinking,
        })

        return thinking, all_cards

    def _random_action(self, mask: np.ndarray) -> int:
        valid = [i for i in range(len(mask)) if mask[i] == 1]
        card_actions = [i for i in valid if i < NUM_CARDS]
        if card_actions:
            return random.choice(card_actions)
        return valid[0] if valid else NUM_CARDS

    def _heuristic_action(self, mask: np.ndarray) -> int:
        """Use the heuristic module's logic."""
        from game.card import TRUMP_RANKS, Rank
        from game.util import get_effective_rank

        valid = [i for i in range(len(mask)) if mask[i] == 1]
        if valid == [NUM_CARDS]:
            return NUM_CARDS

        card_actions = [i for i in valid if i < NUM_CARDS]
        if not card_actions:
            return NUM_CARDS

        current_player = self.game.current_hand.next_player
        player = self.game.players[current_player]
        lead_player = self.game.current_hand.lead_player
        is_leading = current_player == lead_player
        trump_suit = self.game.trump_suit

        def is_trump(card: Card) -> bool:
            return card.rank in TRUMP_RANKS or card.suit == trump_suit

        def points(card: Card) -> int:
            return card.points()

        # Complete pair if one card selected
        if len(self.env.partial_selection) == 1:
            selected = self.env.partial_selection[0]
            same_rank = [idx for idx in card_actions if Card.from_index(idx).rank == selected.rank]
            if same_rank:
                return same_rank[0]

        # Extend multi-card if possible
        if len(self.env.partial_selection) >= 2:
            extend = [idx for idx in card_actions if idx not in [c.get_index() for c in self.env.partial_selection]]
            if extend:
                return max(extend, key=lambda i: Card.from_index(i).rank.value)

        scores = {}
        if is_leading:
            for idx in card_actions:
                card = Card.from_index(idx)
                if card.rank == Rank.ACE and not is_trump(card):
                    scores[idx] = 1000
                elif sum(1 for c in player.hand if c == card) >= 2:
                    scores[idx] = 800 + card.rank.value
                elif is_trump(card) and points(card) == 0:
                    scores[idx] = 600 - card.rank.value
                else:
                    base = 100
                    if points(card) > 0:
                        base -= 20
                    if is_trump(card):
                        base -= 10
                    scores[idx] = base - card.rank.value
        else:
            lead_action = self.game.current_hand.actions[lead_player]
            lead_suit = lead_action.get_suit(trump_suit) if lead_action else trump_suit

            # Determine if team winning
            best_action = None
            best_class = None
            best_idx = None
            for i, action in enumerate(self.game.current_hand.actions):
                if action is None:
                    continue
                cls = action.classify(lead_suit, trump_suit)
                if best_class is None or best_class < cls:
                    best_class = cls
                    best_action = action
                    best_idx = i

            team_winning = best_idx is not None and (best_idx - current_player) % 2 == 0

            for idx in card_actions:
                card = Card.from_index(idx)
                pts = points(card)

                if team_winning:
                    if pts > 0:
                        scores[idx] = 500 + pts
                    else:
                        base = 200
                        if is_trump(card):
                            base -= 30
                        scores[idx] = base - card.rank.value
                else:
                    can_beat = False
                    if best_action and len(best_action.cards) == 1:
                        best_card = best_action.cards[0]
                        if card.suit == lead_suit and not is_trump(card):
                            if get_effective_rank(card, trump_suit) > get_effective_rank(best_card, trump_suit):
                                can_beat = True
                        if is_trump(card) and not is_trump(best_card):
                            can_beat = True

                    if can_beat:
                        scores[idx] = 800 + get_effective_rank(card, trump_suit)
                    else:
                        base = 300
                        if pts > 0:
                            base -= 80
                        if is_trump(card):
                            base -= 40
                        scores[idx] = base - card.rank.value

        return max(scores.keys(), key=lambda k: scores[k])

    def _ppo_action(self, mask: np.ndarray, checkpoint_name: str | None = None) -> int:
        import torch

        module = self._get_ppo_module(checkpoint_name)
        device = torch.device("cuda")
        module.to(device)

        obs = self.env.get_observation()
        current = self.game.current_hand.next_player
        agent_obs = obs[current]

        obs_batch = {
            "observations": torch.tensor(
                np.expand_dims(agent_obs["observations"], axis=0),
                dtype=torch.float32, device=device,
            ),
            "action_mask": torch.tensor(
                np.expand_dims(agent_obs["action_mask"], axis=0),
                dtype=torch.float32, device=device,
            ),
        }
        inference_out = module.forward_inference({"obs": obs_batch})
        logits = inference_out["action_dist_inputs"]
        dist_class = module.get_exploration_action_dist_cls()
        action_dist = dist_class(logits)
        return action_dist.sample().item()

    @staticmethod
    def get_available_checkpoints() -> list[str]:
        """Return list of config names that have checkpoints available."""
        import os
        import glob
        names = []
        for path in glob.glob("configs/*.json"):
            name = os.path.splitext(os.path.basename(path))[0]
            cp_path = f"checkpoints/{name}"
            if os.path.exists(cp_path):
                names.append(name)
        return names

    def _get_ppo_module(self, checkpoint_name: str | None = None):
        """Load PPO module from checkpoint. Caches by checkpoint name."""
        cache_key = f"_ppo_module_{checkpoint_name or 'default'}"
        if hasattr(GameSession, cache_key):
            return getattr(GameSession, cache_key)

        import os
        import json as _json
        from rl.util import build_algo

        available = self.get_available_checkpoints()
        if not available:
            raise RuntimeError(
                "PPO not available: no checkpoint found. "
                "Need a configs/<name>.json with matching checkpoints/<name>/ directory."
            )

        name = checkpoint_name if checkpoint_name in available else available[0]
        config_path = f"configs/{name}.json"
        with open(config_path) as f:
            config = _json.load(f)

        config["training"]["restore"] = True
        config["training"]["restore_from"] = f"checkpoints/{name}"

        algo = build_algo(name, config)
        module = algo.get_module("shared_policy")
        setattr(GameSession, cache_key, module)
        setattr(GameSession, f"_ppo_algo_{name}", algo)  # keep reference so it's not GC'd
        print(f"PPO: Loaded checkpoint '{name}' successfully")
        return module

    def _generate_thinking(self, player_type: PlayerType, cards: list[Card]) -> dict[str, Any]:
        from game.card import Rank
        thinking = {
            "player_type": player_type.value,
            "cards_played": [str(c) for c in cards],
        }

        if player_type == PlayerType.HEURISTIC:
            if len(cards) == 1:
                card = cards[0]
                if card.rank == Rank.ACE:
                    thinking["reason"] = "Playing Ace to secure trick"
                elif card.points() > 0:
                    thinking["reason"] = "Dumping point cards to teammate"
                else:
                    thinking["reason"] = "Discarding low card"
            elif len(cards) == 2:
                thinking["reason"] = "Playing pair for strength"
            elif len(cards) == 4:
                thinking["reason"] = "Playing tractor for maximum strength"
        elif player_type == PlayerType.RANDOM:
            thinking["reason"] = "Random selection"

        return thinking

    def get_bot_suggestion(self, player_idx: int | None = None, bot_type: str = "heuristic", checkpoint: str | None = None) -> dict[str, Any]:
        """Get a suggestion for the given player using the specified bot type."""
        current = self.game.current_hand.next_player
        if player_idx is not None and player_idx != current:
            return {"action": "error", "reason": f"It's P{current}'s turn, not P{player_idx}'s"}

        if len(self.game.players[current].hand) == 0:
            return {"action": "error", "reason": f"P{current} has no cards set"}

        mask = self._get_action_mask()
        if bot_type == "ppo":
            available = self.get_available_checkpoints()
            if not available:
                return {"action": "error", "reason": "PPO not available: no checkpoint found (need configs/<name>.json + checkpoints/<name>/)"}
            if checkpoint and checkpoint not in available:
                return {"action": "error", "reason": f"Checkpoint '{checkpoint}' not found. Available: {available}"}
            action_idx = self._ppo_action(mask, checkpoint)
        elif bot_type == "random":
            action_idx = self._random_action(mask)
        else:
            action_idx = self._heuristic_action(mask)

        if action_idx == NUM_CARDS:
            return {"action": "commit", "reason": "Commit current selection"}

        card = Card.from_index(action_idx)
        return {
            "action": "play",
            "card": str(card),
            "card_index": action_idx,
            "reason": f"Suggested by {bot_type}",
        }

    def get_human_seats(self) -> list[int]:
        return [i for i, pt in enumerate(self.player_types) if pt == PlayerType.HUMAN]

    def _hand_sort_key(self, card: Card):
        """Sort key that groups trump cards together at the end."""
        from game.card import TRUMP_RANKS
        from game.util import get_effective_rank

        trump_suit = self.game.trump_suit
        is_trump = card.rank in TRUMP_RANKS or card.suit == trump_suit

        if is_trump:
            # Trump cards go last, sorted by effective rank
            return (1, 0, get_effective_rank(card, trump_suit), card.suit.value)
        else:
            # Non-trump: group by suit, then by rank
            return (0, card.suit.value, card.rank.value, 0)

    def get_state_for_client(self) -> dict[str, Any]:
        """Get game state formatted for web client."""
        current_player = self.game.current_hand.next_player

        hands = []
        for i, player in enumerate(self.game.players):
            hand = sorted(player.hand, key=self._hand_sort_key)
            hands.append([{
                "index": int(c.get_index()),
                "suit": c.suit.name,
                "rank": c.rank.name,
                "str": str(c),
            } for c in hand])

        trick = []
        for i, action in enumerate(self.game.current_hand.actions):
            if action:
                trick.append({
                    "player": int(i),
                    "cards": [{
                        "index": int(c.get_index()),
                        "suit": c.suit.name,
                        "rank": c.rank.name,
                        "str": str(c),
                    } for c in action.cards],
                })
            else:
                trick.append(None)

        legal_cards = []
        can_commit = False
        if self.player_types[current_player] == PlayerType.HUMAN:
            mask = self._get_action_mask()
            legal_cards = [int(i) for i in range(NUM_CARDS) if mask[i] == 1]
            can_commit = bool(mask[NUM_CARDS] == 1)

        return {
            "trump_suit": self.game.trump_suit.name,
            "host": int(self.game.host),
            "current_player": int(current_player),
            "lead_player": int(self.game.current_hand.lead_player),
            "hands": hands,
            "trick": trick,
            "defender_points": int(self.game.defender_points),
            "attacker_points": int(self.game.attacker_points),
            "legal_cards": legal_cards,
            "can_commit": can_commit,
            "partial_selection": [int(c.get_index()) for c in self.env.partial_selection],
            "history_length": int(len(self.history)),
            "game_over": bool(self.game.game_over()),
        }

    def export_history(self) -> list[dict[str, Any]]:
        return self.history.copy()

    @classmethod
    def from_history(cls, history: list[dict[str, Any]]) -> "GameSession":
        return cls([PlayerType.HUMAN] * 4)

    @classmethod
    def create_sandbox_mode(
        cls, trump_suit_name: str = "SPADES", lead_player: int = 0, my_seats: list[int] | None = None
    ) -> "GameSession":
        """Create a session for sandbox mode with empty hands."""
        from game.hand import Hand
        from game.player import Player

        if my_seats is None:
            my_seats = [0]

        session = cls([PlayerType.HUMAN] * 4)
        session.my_seats = my_seats

        # Clear the hands
        trump_suit = Suit[trump_suit_name]
        session.game.sandbox = True
        session.game.trump_suit = trump_suit
        session.game.host = 0
        session.game.attacker_points = 0
        session.game.defender_points = 0
        session.game.kitty = []
        session.game.current_hand = Hand(NUM_PLAYERS, lead_player)
        session.game.hand_history = []

        for i in range(NUM_PLAYERS):
            session.game.players[i] = Player(i)
            session.env.hands_encodings[i] = np.zeros((NUM_CARDS,))

        session.env.trump_encoding = np.zeros((4,))
        session.env.trump_encoding[trump_suit.value] = 1.0
        session.env.partial_selection = []
        session.env.partial_selection_encoding = np.zeros((NUM_CARDS,))

        session._save_state()
        return session

    def set_lead_player(self, player_idx: int) -> None:
        """Set lead player for current trick (sandbox mode)."""
        from game.hand import Hand

        self.game.current_hand = Hand(NUM_PLAYERS, player_idx)
        self._save_state()

    def set_player_hand(self, player_idx: int, card_indices: list[int]) -> None:
        """Set a player's hand manually (for sandbox mode)."""
        from game.player import Player

        player = Player(player_idx)
        for idx in card_indices:
            player.add_card(Card.from_index(idx), self.game.trump_suit)
        self.game.players[player_idx] = player

        # Update encoding
        self.env.hands_encodings[player_idx] = np.zeros((NUM_CARDS,))
        for idx in card_indices:
            self.env.hands_encodings[player_idx][idx] += 1.0

        self._save_state()

    def set_trick_action(self, player_idx: int, card_indices: list[int]) -> None:
        """Set a player's action in the current trick (for sandbox mode)."""
        if not card_indices:
            self.game.current_hand.actions[player_idx] = None
        else:
            cards = [Card.from_index(idx) for idx in card_indices]
            self.game.current_hand.actions[player_idx] = Action(cards)

        lead = self.game.current_hand.lead_player
        for offset in range(NUM_PLAYERS):
            p = (lead + offset) % NUM_PLAYERS
            if self.game.current_hand.actions[p] is None:
                self.game.current_hand.next_player = p
                break

        self._save_state()
