from collections import defaultdict

from game.card import TRUMP_RANKS, Card, Rank, Suit


def rank_kitty_cards_to_discard(
    combined_hand: list[Card], trump_suit: Suit, num_to_discard: int
) -> list[Card]:
    """
    Takes a combined_hand of HAND_SIZE + KITTY_SIZE and heuristically decides
    which num_to_discard cards would be best to put into the kitty.

    Returns:
        - list[Card]: The cards to put back into the kitty
    """

    suit_counts: dict[Suit, int] = defaultdict(lambda: 0)
    for card in combined_hand:
        if card.rank in TRUMP_RANKS:
            suit_counts[trump_suit] += 1
        else:
            suit_counts[card.suit] += 1

    cards_to_discard: list[Card] = []
    for suit, _ in sorted(suit_counts.items(), key=lambda x: x[1]):
        if len(cards_to_discard) >= num_to_discard:
            break
        if suit == trump_suit:
            continue
        cards_of_suit = [
            c for c in combined_hand if c.suit == suit and c.rank not in TRUMP_RANKS
        ]
        for card in sorted(
            cards_of_suit,
            key=lambda x: x.rank.value,
        ):
            if len(cards_to_discard) >= num_to_discard:
                break
            if card.rank.value > Rank.TEN.value:
                continue
            if card.rank == Rank.FIVE and suit_counts[trump_suit] < 10:
                continue
            cards_to_discard.append(card)
    for card in sorted(combined_hand, key=lambda x: x.rank.value):
        if len(cards_to_discard) >= num_to_discard:
            break
        if card in cards_to_discard:
            continue
        cards_to_discard.append(card)
    return cards_to_discard


def get_effective_rank(card: Card, trump_suit: Suit) -> int:
    """
    Effective rank of card:
     - 3-A: Actual rank
     - 2 non trump: 2.rank
     - 2 trump and joker: actual rank + 1
    """

    if card.rank in [Rank.BJ, Rank.RJ]:
        return card.rank.value + 1
    if card.rank == Rank.TWO and card.suit == trump_suit:
        return card.rank.value + 1
    return card.rank.value


def get_tractors(cards: list[Card], trump_suit: Suit) -> list[int]:
    """
    Returns a list of ranks each representing the highest Rank of a tractor.
    Note: returned tractors may overlap!
    For example, if the cards are [6S, 6S, 7S, 7S, 8S, 8S], the return value
    should be [Rank.SEVEN, Rank.EIGHT]
    This method assumes that all cards in the input array have the same suit.
    """

    # for i in range(1, len(cards)):
    #     if cards[i].suit != cards[0].suit and cards[i].rank not in TRUMP_RANKS:
    #         raise ValueError(f"All cards must be the same suit, got {cards}")

    ordered_cards = sorted(
        cards, key=lambda x: (get_effective_rank(x, trump_suit), x.suit == trump_suit)
    )
    result: list[int] = []
    for i in range(len(cards) - 3):
        if (
            ordered_cards[i] == ordered_cards[i + 1]
            and ordered_cards[i + 2] == ordered_cards[i + 3]
            and get_effective_rank(ordered_cards[i], trump_suit) + 1
            == get_effective_rank(ordered_cards[i + 2], trump_suit)
        ):
            result.append(get_effective_rank(ordered_cards[i + 2], trump_suit))
    return result


def get_pairs(cards: list[Card]) -> list[Card]:
    """
    Returns a list of ranks of pairs in the given list of cards.
    Assumes that all cards in the input array have the same suit.
    """

    # for i in range(1, len(cards)):
    #     if cards[i].suit != cards[0].suit and cards[i].rank not in TRUMP_RANKS:
    #         raise ValueError(f"All cards must be the same suit, got {cards}")

    ordered_cards = sorted(cards, key=lambda x: (x.rank.value, x.suit.value))
    result: list[Card] = []
    for i in range(len(cards) - 1):
        if ordered_cards[i] == ordered_cards[i + 1]:
            result.append(ordered_cards[i])
    return result
