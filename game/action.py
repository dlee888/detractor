from enum import Enum
from typing import override

from game.base import BaseModel
from game.card import TRUMP_RANKS, Card, Suit
from game.util import get_tractors, get_pairs


class ActionType(Enum):
    SINGLE = 0
    PAIR = 1
    TRACTOR = 2


class ActionSuitType(Enum):
    OTHER = 0
    SAME = 1
    TRUMP = 2


class ActionClass(BaseModel):
    action_type: ActionType
    suit_type: ActionSuitType
    high_rank: int

    def __init__(
        self, action_type: ActionType, suit_type: ActionSuitType, high_rank: int
    ) -> None:
        self.action_type = action_type
        self.suit_type = suit_type
        self.high_rank = high_rank

    def __lt__(self, other: "ActionClass") -> bool:
        if self.suit_type != other.suit_type:
            return self.suit_type.value < other.suit_type.value
        if self.action_type != other.action_type:
            return self.action_type.value < other.action_type.value
        return self.high_rank < other.high_rank


class Action(BaseModel):
    cards: list[Card]

    def __init__(self, cards: list[Card]) -> None:
        assert len(cards) > 0
        self.cards = cards

    def get_suit(self, trump_suit: Suit) -> Suit | None:
        suit = None
        for card in self.cards:
            if card.rank in TRUMP_RANKS:
                if suit is None:
                    suit = trump_suit
                elif suit != trump_suit:
                    return None
            else:
                if suit is None:
                    suit = card.suit
                elif suit != card.suit:
                    return None
        return suit

    def classify(self, lead_suit: Suit, trump_suit: Suit) -> ActionClass:
        suit = self.get_suit(trump_suit)
        if suit == trump_suit:
            suit_type = ActionSuitType.TRUMP
        elif suit == lead_suit:
            suit_type = ActionSuitType.SAME
        else:
            suit_type = ActionSuitType.OTHER

        if suit is not None:
            tractors = get_tractors(self.cards, trump_suit)
            if len(tractors) > 0:
                high_rank = max(tractors)
                action_type = ActionType.TRACTOR
            else:
                pairs = get_pairs(self.cards)
                if len(pairs) > 0:
                    high_rank = max([p.rank.value for p in pairs])
                    action_type = ActionType.PAIR
                else:
                    high_rank = max(card.rank.value for card in self.cards)
                    action_type = ActionType.SINGLE
        else:
            high_rank = max(card.rank.value for card in self.cards)
            action_type = ActionType.SINGLE

        return ActionClass(action_type, suit_type, high_rank)

    @override
    def __str__(self) -> str:
        sorted_cards = sorted(self.cards, key=lambda x: x.rank)
        return " ".join([str(c) for c in sorted_cards])

    @override
    def __repr__(self) -> str:
        return str(self)
