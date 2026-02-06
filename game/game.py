import itertools
import random

from game.action import Action, ActionType
from game.base import BaseModel
from game.card import TRUMP_RANKS, Card, Rank, Suit
from game.hand import Hand
from game.player import Player
from game.util import rank_kitty_cards_to_discard

KITTY_MULTIPLIERS = {ActionType.SINGLE: 2, ActionType.PAIR: 4, ActionType.TRACTOR: 8}

NUM_PLAYERS: int = 4
HAND_SIZE: int = 25
KITTY_SIZE: int = 8


class TractorGame(BaseModel):
    players: list[Player]
    kitty: list[Card]
    trump_suit: Suit
    host: int
    hand_history: list[Hand]
    current_hand: Hand
    attacker_points: int
    defender_points: int

    def __init__(self) -> None:
        hands, kitty, host, trump_suit = self.deal_new_deck()
        self.players = [Player(i) for i in range(NUM_PLAYERS)]
        self.kitty = kitty
        self.trump_suit = trump_suit
        self.host = host
        self.hand_history = []
        self.current_hand = Hand(NUM_PLAYERS, host)
        self.attacker_points = 0
        self.defender_points = 0
        self.sandbox = False
        for player in range(NUM_PLAYERS):
            for card in hands[player]:
                self.players[player].add_card(card, trump_suit)

    def deal_new_deck(self) -> tuple[list[list[Card]], list[Card], int, Suit]:
        """
        Creates a new deck of 108 cards and deals them.
        Returns:
            - hands: list of NUM_PLAYERS hands, each stored as a list of Card
            - kitty: a list of KITTY_SIZE cards representing the kitty
            - host: an integer representing the index of the first
                    player to get dealt a 2
            - trump_suit: a Suit representing the suit of the first
                          2 that was dealt
        """

        deck = Card.construct_deck() + Card.construct_deck()
        random.shuffle(deck)

        assert len(deck) == HAND_SIZE * NUM_PLAYERS + KITTY_SIZE
        hands = [deck[HAND_SIZE * i : HAND_SIZE * (i + 1)] for i in range(NUM_PLAYERS)]
        kitty = deck[-KITTY_SIZE:]
        host = 0
        trump_suit = Suit(0)
        for i, player in itertools.product(range(HAND_SIZE), range(NUM_PLAYERS)):
            if hands[player][i].rank == Rank.TWO:
                host = player
                trump_suit = hands[player][i].suit
                break

        # Heuristic kitty exchange
        cards_to_exchange = hands[host].copy() + kitty.copy()
        exchanged_cards = rank_kitty_cards_to_discard(
            cards_to_exchange, trump_suit, KITTY_SIZE
        )
        hands[host].extend(kitty)
        for card in exchanged_cards:
            hands[host].remove(card)

        return hands, exchanged_cards, host, trump_suit

    def get_effective_suit(self, card: Card) -> Suit:
        if card.rank in TRUMP_RANKS:
            return self.trump_suit
        return card.suit

    def play_action(self, action: Action) -> tuple[int, int] | None:
        """
        Plays an action for the next player to act.
        Returns (score, winner) if the trick is over, otherwise None
        """
        player = self.current_hand.next_player
        self.current_hand.play_action(action, self.players[player], self.trump_suit)
        for card in action.cards:
            self.players[player].remove_card(card, self.trump_suit)

        if self.current_hand.is_complete():
            score = self.current_hand.points()
            winner, win_type = self.current_hand.winner(self.trump_suit)

            self.hand_history.append(self.current_hand)
            self.current_hand = Hand(NUM_PLAYERS, winner)

            # Sanity check
            if not self.sandbox:
                for player in self.players:
                    assert len(player.hand) == len(self.players[0].hand)

            if self.game_over():
                score += (
                    sum(card.points() for card in self.kitty)
                    * KITTY_MULTIPLIERS[win_type.action_type]
                )

            if abs(winner - self.host) % 2 == 0:
                self.defender_points += score
            else:
                self.attacker_points += score

            return score, winner

    def game_over(self) -> bool:
        if self.sandbox:
            return False
        for player in self.players:
            if len(player.hand) > 0:
                return False
        return True
