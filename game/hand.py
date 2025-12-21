from game.action import Action, ActionClass
from game.base import BaseModel
from game.card import Suit
from game.player import Player
from game.util import get_pairs, get_tractors


class Hand(BaseModel):
    actions: list[Action | None]
    num_players: int
    lead_player: int
    next_player: int

    def __init__(self, num_players: int, lead_player: int) -> None:
        self.actions = [None for _ in range(num_players)]
        self.num_players = num_players
        self.lead_player = lead_player
        self.next_player = lead_player

    def verify_next_action(
        self, action: Action, player: Player, trump_suit: Suit
    ) -> None:
        if self.next_player == self.lead_player:
            if action.get_suit(trump_suit) is None:
                raise Exception(
                    f"Lead action must have all cards of the same suit, got {action.cards}"
                )
            if len(action.cards) == 2:
                if len(get_pairs(action.cards)) != 1:
                    raise Exception(
                        f"Lead action with two cards must be pair, got {action.cards}"
                    )
            if len(action.cards) == 4:
                if len(get_tractors(action.cards, trump_suit)) != 1:
                    raise Exception(
                        f"Lead action with four cards must be tractor, got {action.cards}"
                    )
            return
        lead_action = self.actions[self.lead_player]
        assert lead_action is not None
        lead_suit = lead_action.get_suit(trump_suit)
        assert lead_suit is not None
        suit = action.get_suit(trump_suit)
        if len(suit_cards := player.cards_by_suit[lead_suit]) >= len(lead_action.cards):
            if suit != lead_suit:
                raise Exception(
                    f"Player has enough cards of suit {lead_suit.name} left, but played {action.cards}"
                )
            if len(get_tractors(lead_action.cards, trump_suit)) > 0:
                if (
                    len(get_tractors(suit_cards, trump_suit)) > 0
                    and len(get_tractors(action.cards, trump_suit)) == 0
                ):
                    raise Exception(f"Player has a tractor, but played {action.cards}")
                if min(len(get_pairs(suit_cards)), 2) > len(get_pairs(action.cards)):
                    raise Exception(
                        f"Player has {len(get_pairs(suit_cards))} pairs, but played {action.cards}"
                    )
            elif len(get_pairs(lead_action.cards)) > 0:
                if min(
                    len(get_pairs(suit_cards)), len(get_pairs(lead_action.cards))
                ) > len(get_pairs(action.cards)):
                    raise Exception(
                        f"Player has {len(get_pairs(suit_cards))} pairs, but played {action.cards}"
                    )

    def play_action(self, action: Action, player: Player, trump_suit: Suit) -> None:
        if self.actions[self.next_player] is not None:
            raise Exception("Cannot play action: player has already played")
        # self.verify_next_action(action, player, trump_suit)
        self.actions[self.next_player] = action
        self.next_player = (self.next_player + 1) % self.num_players

    def is_complete(self) -> bool:
        return (
            self.next_player == self.lead_player
            and self.actions[self.next_player] is not None
        )

    def points(self) -> int:
        total = 0
        for action in self.actions:
            if action is not None:
                total += sum(card.points() for card in action.cards)
        return total

    def winner(self, trump_suit: Suit) -> tuple[int, ActionClass]:
        keys: list[tuple[int, ActionClass]] = []
        assert (lead_action := self.actions[self.lead_player]) is not None
        lead_suit = lead_action.get_suit(trump_suit)
        assert lead_suit is not None
        for i in range(self.num_players - 1, -1, -1):
            # Use python's stable sorting to break ties correctly
            current_player = (self.lead_player + i) % self.num_players
            action = self.actions[current_player]
            if action is None:
                raise Exception("Cannot determine winner unless hand is finished")
            keys.append((current_player, action.classify(lead_suit, trump_suit)))
        return sorted(keys, key=lambda x: x[1])[-1]
