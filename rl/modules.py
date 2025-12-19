import torch
from typing import Any

from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from game.card import Card, NUM_CARDS, NUM_SUITS, Rank, TRUMP_RANKS
from game.action import Action, ActionType, ActionClass, ActionSuitType
from game.game import NUM_PLAYERS


class ActionMaskingTorchRandomModule(ActionMaskingTorchRLModule):
    """Torch version of random action selection module."""

    def _compute_logits(self, action_mask: torch.Tensor) -> torch.Tensor:
        """Helper method to compute logits from action mask."""
        logits = torch.where(
            action_mask == 1.0,
            torch.zeros_like(action_mask),
            torch.full_like(action_mask, -1e10),
        )
        return logits

    def forward_inference(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        return {"action_dist_inputs": logits}

    def forward_exploration(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        return {"action_dist_inputs": logits}

    def forward_train(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        logits = self._compute_logits(action_mask)
        batch_size = action_mask.shape[0]
        vf_preds = torch.zeros(
            batch_size, dtype=torch.float32, device=action_mask.device
        )
        return {"action_dist_inputs": logits, "vf_preds": vf_preds}


class ActionmMaskingHeuristicsModule(ActionMaskingTorchRLModule):
    """Heuristic, rule-based action selection module.

    High-level behavior (single-step view):
    - Uses only what is in the current observation: current trick cards, full per-player
      card history (aggregated over past tricks), trump suit, our current hand and
      partial_selection, and the action_mask. It does not track long-term void suits or
      perform hidden-card inference.
    - When we are leading a trick (i.e., before any cards are in the current trick):
        1. If we have a non-trump Ace, we start the trick with that Ace (single or as
           the first card of a later pair/tractor, depending on what the mask allows).
        2. Otherwise, we explicitly try to lead multi-card strength:
           - When we have multiple copies of a rank and the action mask permits
             extending the partial_selection, we will always complete the pair before
             committing.
           - When the mask indicates that we can continue extending beyond a pair
             (i.e., partial_selection already has at least two cards and there are
             still legal card indices < NUM_CARDS), we keep adding cards instead of
             committing, so that we actually play the longest tractor the environment
             deems legal from our current partial selection.
        3. If we cannot start any pair/tractor, we play the smallest non-point trump
           card we are allowed to play.
        4. If even that is not available, we fall back to low non-trump, non-point
           cards.
    - When we are following:
        * Reconstructs the current trick, lead suit, current best action, and whether
          our team (seats 0 and 2) is currently winning the trick.
        * If our team is winning and our teammate's action is strong enough to be
          trusted, we try to dump point cards (10, K, 5) while avoiding splitting pairs.
          Otherwise we discard the weakest non-point card, preferring non-trump.
        * If our team is losing, we:
            - First try to beat the current best on-suit with the strongest on-suit
              card we can.
            - Failing that, try to overtrump a non-trump best action using relatively
              low/cheap trump, avoiding point-carrying trump when possible.
            - If we cannot win, discard low, non-point, non-trump cards and prefer not
              to break pairs unless forced.
    The module is non-trainable and only produces logits consistent with the action
    mask; the learner never optimizes its parameters.
    """

    # --------- basic helpers ---------

    def _decode_observation(self, obs_vec: torch.Tensor) -> dict[str, Any]:
        """Decode flat observation vector into structured components.

        Layout must match TractorEnv.encode_state.
        """
        x = obs_vec.detach().cpu().numpy()
        offset = 0

        # Hand encoding.
        hand_counts = x[offset : offset + NUM_CARDS]
        offset += NUM_CARDS

        # Trump suit one-hot.
        trump_block = x[offset : offset + NUM_SUITS]
        trump_suit_idx = int(trump_block.argmax())
        offset += NUM_SUITS

        # Current trick: other 3 players, relative to acting player.
        current_trick_blocks: list[list[int]] = []
        for _ in range(NUM_PLAYERS - 1):
            block = x[offset : offset + NUM_CARDS]
            offset += NUM_CARDS
            cards = [i for i, v in enumerate(block) if v > 0.5]
            current_trick_blocks.append(cards)

        # Lead player (relative index 0..3; 0 is us).
        lead_block = x[offset : offset + NUM_PLAYERS]
        lead_rel_index = int(lead_block.argmax())
        offset += NUM_PLAYERS

        # Partial selection (our still-uncommitted cards this trick).
        partial_block = x[offset : offset + NUM_CARDS]
        offset += NUM_CARDS
        partial_selection = [i for i, v in enumerate(partial_block) if v > 0.5]

        return {
            "hand_counts": hand_counts,
            "trump_suit_idx": trump_suit_idx,
            "current_trick_blocks": current_trick_blocks,
            "lead_rel_index": lead_rel_index,
            "partial_selection": partial_selection,
        }

    @staticmethod
    def _is_trump(card: Card, trump_suit_idx: int) -> bool:
        return card.rank in TRUMP_RANKS or card.suit.value == trump_suit_idx

    @staticmethod
    def _card_points(card: Card) -> int:
        if card.rank == Rank.FIVE:
            return 5
        if card.rank in (Rank.TEN, Rank.KING):
            return 10
        return 0

    def _current_trick_info(
        self, obs_info: dict[str, Any]
    ) -> tuple[bool, bool, bool, Action | None, Action | None, Action | None]:
        """Compute high-level info about the current trick.

        Returns:
            is_lead: True if no one has played to this trick yet.
            points_on_table: True if any points are currently in the trick.
            team_winning: True if the best action so far belongs to us or teammate.
            best_action: Current best Action, if any.
            teammate_action: Our teammate's committed Action, if any.
            lead_action: Action of the player who led this trick, if any.
        """
        trump_suit_idx = obs_info["trump_suit_idx"]
        current_trick_blocks: list[list[int]] = obs_info["current_trick_blocks"]
        lead_rel_index: int = obs_info["lead_rel_index"]

        # actions[0] is us (we have not committed yet), 1 and 3 are opponents, 2 is teammate.
        actions: list[Action | None] = [None for _ in range(NUM_PLAYERS)]
        for rel_offset in range(1, NUM_PLAYERS):
            cards = current_trick_blocks[rel_offset - 1]
            if cards:
                actions[rel_offset] = Action([Card.from_index(i) for i in cards])

        is_lead = all(len(block) == 0 for block in current_trick_blocks)

        # If no one has played yet, there is no best_action or lead_action.
        if is_lead:
            return True, False, False, None, None, None

        # Determine lead_suit from the lead player's action.
        lead_action = actions[lead_rel_index]
        if lead_action is None:
            raise RuntimeError("Lead player has no action in current trick encoding")

        trump_suit = Card.from_index(0).suit.__class__(trump_suit_idx)
        lead_suit = lead_action.get_suit(trump_suit)
        if lead_suit is None:
            # Fallback: first card's effective suit.
            lead_suit = (
                trump_suit if lead_action.cards[0].rank in TRUMP_RANKS
                else lead_action.cards[0].suit
            )

        best_action: Action | None = None
        best_class: ActionClass | None = None
        best_index: int | None = None

        points_on_table = False
        for rel_idx, act in enumerate(actions):
            if act is None:
                continue
            for c in act.cards:
                if self._card_points(c) > 0:
                    points_on_table = True
                    break
            cls = act.classify(lead_suit, trump_suit)
            if best_class is None or best_class < cls:
                best_class = cls
                best_action = act
                best_index = rel_idx

        team_winning = best_index in (0, 2)
        teammate_action = actions[2]
        return is_lead, points_on_table, team_winning, best_action, teammate_action, lead_action

    # --------- scoring rules ---------

    def _score_lead_card(
        self,
        card: Card,
        hand_counts,
        trump_suit_idx: int,
    ) -> float:
        """Score a single card when we are leading a trick.

        Approximates 1a/1b/1d/1e without void tracking.
        """
        is_trump = self._is_trump(card, trump_suit_idx)
        pts = self._card_points(card)

        # 1a. Prefer non-trump aces.
        if card.rank == Rank.ACE and not is_trump:
            return 1000.0

        # 1b. Prefer strongest pairs/tractors – approximated by "we have at least 2 of this card".
        if int(round(hand_counts[card.get_index()])) >= 2:
            return 800.0 + float(card.rank.value)

        # 1d. Smallest trump card not worth points.
        if is_trump and pts == 0:
            return 600.0 - float(card.rank.value)

        # 1e. Fallback: small preference for low non-trump, non-point cards.
        base = 100.0
        if pts > 0:
            base -= 20.0
        if is_trump:
            base -= 10.0
        return base - float(card.rank.value)

    def _select_action_for_single_obs(
        self, obs_vec: torch.Tensor, action_mask: torch.Tensor
    ) -> int:
        """Select a single discrete action index for one observation."""
        obs_info = self._decode_observation(obs_vec)
        hand_counts = obs_info["hand_counts"]
        trump_suit_idx = obs_info["trump_suit_idx"]
        partial_selection = obs_info["partial_selection"]

        (
            is_lead,
            _points_on_table,
            team_winning,
            best_action,
            teammate_action,
            lead_action,
        ) = self._current_trick_info(obs_info)

        valid_indices = [i for i, v in enumerate(action_mask.tolist()) if v == 1.0]
        if not valid_indices:
            raise RuntimeError("No valid actions available under action mask")

        # If only commit is valid, just commit.
        if valid_indices == [NUM_CARDS]:
            return NUM_CARDS

        # Prefer playing cards over committing when both are available.
        candidate_cards = [i for i in valid_indices if i < NUM_CARDS]
        if not candidate_cards:
            return NUM_CARDS

        # If we have already selected exactly one card, try to complete a pair
        # of that rank before doing anything else (always play pair over non-pair).
        if len(partial_selection) == 1:
            selected = Card.from_index(partial_selection[0])
            same_rank_candidates = [
                idx
                for idx in candidate_cards
                if idx not in partial_selection
                and Card.from_index(idx).rank == selected.rank
            ]
            if same_rank_candidates:
                # Mask already ensures legality; choose deterministic lowest index.
                return min(same_rank_candidates)

        # If we already have a pair or longer selected and there are still legal card
        # choices (excluding commit and already-selected cards), prefer to extend the
        # current multi-card action instead of committing immediately. This makes us
        # actually play the maximal tractor that the environment considers legal.
        if len(partial_selection) >= 2:
            extend_candidates = [
                idx for idx in candidate_cards if idx not in partial_selection
            ]
            if extend_candidates:
                def _effective_rank_local(c: Card) -> int:
                    if c.rank in TRUMP_RANKS:
                        return 100 + c.rank.value
                    return c.rank.value

                # Choose the highest-ranked extension deterministically.
                best_extend_idx = max(
                    extend_candidates,
                    key=lambda idx: _effective_rank_local(Card.from_index(idx)),
                )
                return best_extend_idx

        scores: dict[int, float] = {}

        if is_lead:
            # Case 1: starting the trick.
            for idx in candidate_cards:
                card = Card.from_index(idx)
                scores[idx] = self._score_lead_card(card, hand_counts, trump_suit_idx)
        else:
            # Case 2: following.
            trump_suit = Card.from_index(0).suit.__class__(trump_suit_idx)

            def _effective_rank(c: Card) -> int:
                if c.rank in TRUMP_RANKS:
                    return 100 + c.rank.value
                return c.rank.value

            lead_suit = None
            if lead_action is not None:
                lead_suit = lead_action.get_suit(trump_suit)

            best_card: Card | None = None
            if best_action is not None and best_action.action_type == ActionType.SINGLE:
                best_card = best_action.cards[0]

            if team_winning:
                # 2a: team winning.
                strong_teammate = False
                if teammate_action is not None and lead_suit is not None:
                    t_cls = teammate_action.classify(lead_suit, trump_suit)
                    # Define "strong teammate" exactly per user heuristic:
                    # - On non-trump tricks: either any trump action, or a high pair
                    #   (T or higher) in the non-trump suit.
                    # - On trump tricks: actions whose high_rank is at least 2 for
                    #   singles/tractors, or at least K for pairs.
                    if t_cls.suit_type == ActionSuitType.TRUMP:
                        # Any trump play on an off-suit trick is strong.
                        strong_teammate = True
                    elif lead_suit != trump_suit:
                        # Non-trump trick: look for high pairs in the lead suit.
                        if (
                            t_cls.suit_type == ActionSuitType.SAME
                            and t_cls.action_type == ActionType.PAIR
                            and t_cls.high_rank >= Rank.TEN.value
                        ):
                            strong_teammate = True
                    else:
                        # Trump trick: use rank thresholds.
                        if t_cls.action_type in (ActionType.SINGLE, ActionType.TRACTOR):
                            # 2 or higher as a single/tractor is considered strong.
                            if t_cls.high_rank >= Rank.TWO.value:
                                strong_teammate = True
                        elif t_cls.action_type == ActionType.PAIR:
                            # K pair or higher on trump trick is strong.
                            if t_cls.high_rank >= Rank.KING.value:
                                strong_teammate = True

                for idx in candidate_cards:
                    card = Card.from_index(idx)
                    pts = self._card_points(card)
                    count = int(round(hand_counts[idx]))
                    split_pair_penalty = 5.0 if count > 1 else 0.0

                    if strong_teammate and pts > 0:
                        # 2a.i: dump points; avoid splitting pairs if possible.
                        scores[idx] = 900.0 - split_pair_penalty + float(pts)
                    else:
                        # Otherwise discard weakest non-point card.
                        base = 200.0
                        if pts > 0:
                            base -= 50.0
                        if self._is_trump(card, trump_suit_idx):
                            base -= 30.0
                        scores[idx] = base + float(card.rank.value)
            else:
                # 2b: team losing.
                for idx in candidate_cards:
                    card = Card.from_index(idx)
                    pts = self._card_points(card)
                    base = 0.0

                    can_beat_on_suit = False
                    can_beat_with_trump = False

                    if best_card is not None and lead_suit is not None:
                        # 2b.i: can we beat on-suit?
                        if (
                            (card.rank not in TRUMP_RANKS and card.suit == lead_suit)
                            or (
                                card.rank in TRUMP_RANKS
                                and lead_suit == trump_suit
                            )
                        ):
                            if _effective_rank(card) > _effective_rank(best_card):
                                can_beat_on_suit = True

                        # 2b.iii/iv: can we overtrump a non-trump best?
                        if not self._is_trump(best_card, trump_suit_idx) and self._is_trump(
                            card, trump_suit_idx
                        ):
                            can_beat_with_trump = True

                    if can_beat_on_suit:
                        base = 800.0 + float(_effective_rank(card))
                    elif can_beat_with_trump:
                        base = 700.0 + float(_effective_rank(card))
                        if pts > 0:
                            base -= 30.0
                    else:
                        base = 300.0
                        if pts > 0:
                            base -= 80.0
                        if self._is_trump(card, trump_suit_idx):
                            base -= 40.0
                        base -= float(_effective_rank(card))

                    count = int(round(hand_counts[idx]))
                    if count > 1 and base < 800.0:
                        base -= 5.0

                    scores[idx] = base

        best_idx = max(scores.keys(), key=lambda k: (scores[k], -k))
        return int(best_idx)

    # --------- RLlib interface ---------

    def _compute_logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute logits from observations and action mask using the heuristic."""
        obs_vec = obs["observations"]
        action_mask = obs["action_mask"]
        if obs_vec.dim() == 1:
            obs_vec = obs_vec.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        batch_size = obs_vec.shape[0]
        logits = torch.full_like(action_mask, -1e10)

        for i in range(batch_size):
            best_idx = self._select_action_for_single_obs(obs_vec[i], action_mask[i])
            logits[i, best_idx] = 0.0

        return logits

    def forward_inference(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        logits = self._compute_logits(obs)
        return {"action_dist_inputs": logits}

    def forward_exploration(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        logits = self._compute_logits(obs)
        return {"action_dist_inputs": logits}

    def forward_train(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        obs = input_dict["obs"]
        logits = self._compute_logits(obs)
        batch_size = logits.shape[0]
        vf_preds = torch.zeros(
            batch_size, dtype=torch.float32, device=logits.device
        )
        return {"action_dist_inputs": logits, "vf_preds": vf_preds}


