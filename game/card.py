from enum import Enum
from typing import override

from game.base import BaseModel


class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(Enum):
    THREE = 0
    FOUR = 1
    FIVE = 2
    SIX = 3
    SEVEN = 4
    EIGHT = 5
    NINE = 6
    TEN = 7
    JACK = 8
    QUEEN = 9
    KING = 10
    ACE = 11
    TWO = 12
    BJ = 13
    RJ = 14

    def __lt__(self, other: "Rank") -> bool:
        return self.value < other.value


TRUMP_RANKS = [Rank.TWO, Rank.BJ, Rank.RJ]

NUM_SUITS: int = 4
NUM_RANKS: int = 15
NUM_CARDS: int = 54


class Card(BaseModel):
    suit: Suit
    rank: Rank

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self.suit = suit
        self.rank = rank

    @staticmethod
    def from_index(index: int) -> "Card":
        if index == 52:
            return Card(Rank.BJ, Suit(0))
        if index == 53:
            return Card(Rank.RJ, Suit(0))
        return Card(Rank(index // NUM_SUITS), Suit(index % NUM_SUITS))

    @staticmethod
    def construct_deck():
        return [Card.from_index(i) for i in range(NUM_CARDS)]

    def points(self) -> int:
        if self.rank == Rank.FIVE:
            return 5
        if self.rank in [Rank.TEN, Rank.KING]:
            return 10
        return 0

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return super().__eq__(other)

    @override
    def __str__(self) -> str:
        ranks = "3456789TJQKA2jJ"
        suits = "CDHS"
        return f"{ranks[self.rank.value]}{'' if self.rank.value >= Rank.BJ.value else suits[self.suit.value]}"

    @override
    def __repr__(self):
        return self.__str__()

    def get_index(self) -> int:
        if self.rank == Rank.BJ:
            return 52
        if self.rank == Rank.RJ:
            return 53
        return self.rank.value * NUM_SUITS + self.suit.value

    @override
    def __hash__(self):
        return self.get_index()
