from typing import override

from game.base import BaseModel
from game.card import Card, Suit, TRUMP_RANKS


class Player(BaseModel):
    player_id: int
    hand: list[Card]
    cards_by_suit: dict[Suit, list[Card]]

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.hand = []
        self.cards_by_suit = {suit: [] for suit in Suit}

    def add_card(self, card: Card, trump_suit: Suit) -> None:
        self.hand.append(card)
        if card.rank in TRUMP_RANKS:
            self.cards_by_suit[trump_suit].append(card)
        else:
            self.cards_by_suit[card.suit].append(card)

    def remove_card(self, card: Card, trump_suit: Suit) -> None:
        self.hand.remove(card)
        if card.rank in TRUMP_RANKS:
            self.cards_by_suit[trump_suit].remove(card)
        else:
            self.cards_by_suit[card.suit].remove(card)

    @override
    def __str__(self) -> str:
        cards = self.cards_by_suit.values()
        sorted_suits = [
            " ".join([str(c) for c in sorted(suit, key=lambda x: (x.rank, x.suit.value))])
            for suit in cards
        ]
        return " | ".join([suit.ljust(35) for suit in sorted_suits])

    @override
    def __repr__(self) -> str:
        return str(self)
