"""FastAPI server for DeTractor web interface."""

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from web.game_manager import GameSession, PlayerType

app = FastAPI(title="DeTractor - Tractor Game Visualizer")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Active game sessions keyed by websocket
sessions: dict[WebSocket, GameSession] = {}


def _serialize_completed_trick(hand) -> list:
    """Serialize a completed Hand (trick) for the client."""
    trick = [None] * 4
    for i, a in enumerate(hand.actions):
        if a:
            trick[i] = {
                "player": i,
                "cards": [
                    {"index": int(c.get_index()), "suit": c.suit.name, "rank": c.rank.name, "str": str(c)}
                    for c in a.cards
                ],
            }
    return trick


@app.get("/")
async def root():
    """Serve the main game page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/checkpoints")
async def get_checkpoints():
    """Return list of available PPO checkpoints."""
    from web.game_manager import GameSession
    return {"checkpoints": GameSession.get_available_checkpoints()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for game communication."""
    await websocket.accept()
    session: GameSession | None = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "configure":
                # Configure and start a new game
                players_config = message.get("players", ["human", "heuristic", "heuristic", "heuristic"])
                player_types = [PlayerType.from_string(p) for p in players_config]

                session = GameSession(player_types)
                sessions[websocket] = session

                # Send initial game state
                state = session.get_state_for_client()
                human_seats = session.get_human_seats()
                await websocket.send_json({
                    "type": "game_start",
                    "state": state,
                    "human_seats": human_seats,
                })

                # If it's a bot's turn, let them play
                await run_bot_turns(websocket, session)

            elif msg_type == "play":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                action_indices = message.get("cards", [])
                prev_history_len = len(session.game.hand_history)
                thinking = session.play_human_action(action_indices)
                trick_completed = len(session.game.hand_history) > prev_history_len

                if trick_completed:
                    completed = session.game.hand_history[-1]
                    await websocket.send_json({
                        "type": "trick_complete",
                        "trick": _serialize_completed_trick(completed),
                    })
                    await asyncio.sleep(1.5)

                # Send updated state
                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": {
                        "player": session.last_player,
                        "cards": action_indices,
                    },
                    "thinking": thinking,
                })

                # Run bot turns
                await run_bot_turns(websocket, session)

            elif msg_type == "undo":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                success = session.undo()
                if success:
                    state = session.get_state_for_client()
                    await websocket.send_json({
                        "type": "state_update",
                        "state": state,
                        "last_action": None,
                        "thinking": None,
                    })
                else:
                    await websocket.send_json({"type": "error", "message": "Cannot undo"})

            elif msg_type == "clear_selection":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                session.clear_selection()
                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": None,
                    "thinking": None,
                })

            elif msg_type == "get_suggestion":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                player = message.get("player")  # optional: specific seat
                bot_type = message.get("bot_type", "heuristic")
                checkpoint = message.get("checkpoint")  # optional: specific PPO checkpoint
                suggestion = session.get_bot_suggestion(player, bot_type, checkpoint)
                await websocket.send_json({
                    "type": "suggestion",
                    "suggestion": suggestion,
                })

            elif msg_type == "export":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                history = session.export_history()
                await websocket.send_json({
                    "type": "history",
                    "history": history,
                })

            elif msg_type == "import":
                history = message.get("history", [])
                session = GameSession.from_history(history)
                sessions[websocket] = session

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "game_start",
                    "state": state,
                    "human_seats": session.get_human_seats(),
                })

            elif msg_type == "set_hand":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                player = message.get("player", 0)
                card_indices = message.get("cards", [])
                session.set_player_hand(player, card_indices)

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": None,
                    "thinking": None,
                })

            elif msg_type == "set_trick_card":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                player = message.get("player")
                card_indices = message.get("cards", [])
                session.set_trick_action(player, card_indices)

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": None,
                    "thinking": None,
                })

            elif msg_type == "set_lead_player":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                player = message.get("player", 0)
                session.set_lead_player(player)

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": None,
                    "thinking": None,
                })

            elif msg_type == "sandbox_play":
                if session is None:
                    await websocket.send_json({"type": "error", "message": "No game in progress"})
                    continue

                player = message.get("player")
                card_indices = message.get("cards", [])

                prev_history_len = len(session.game.hand_history)
                session.sandbox_play(player, card_indices)
                trick_completed = len(session.game.hand_history) > prev_history_len

                if trick_completed:
                    completed = session.game.hand_history[-1]
                    await websocket.send_json({
                        "type": "trick_complete",
                        "trick": _serialize_completed_trick(completed),
                    })
                    await asyncio.sleep(1.5)

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "state_update",
                    "state": state,
                    "last_action": {
                        "player": player,
                        "cards": card_indices,
                    },
                    "thinking": None,
                })

            elif msg_type == "start_sandbox":
                trump_suit = message.get("trump_suit", "SPADES")
                lead_player = message.get("lead_player", 0)
                my_seats = message.get("my_seats", [0])
                session = GameSession.create_sandbox_mode(trump_suit, lead_player, my_seats)
                sessions[websocket] = session

                state = session.get_state_for_client()
                await websocket.send_json({
                    "type": "game_start",
                    "state": state,
                    "human_seats": my_seats,
                    "sandbox_mode": True,
                    "my_seats": my_seats,
                })

    except WebSocketDisconnect:
        if websocket in sessions:
            del sessions[websocket]
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})


async def run_bot_turns(websocket: WebSocket, session: GameSession):
    """Run bot turns until it's a human's turn or game is over."""
    while not session.is_game_over() and session.is_bot_turn():
        # Delay so user can see each bot play
        await asyncio.sleep(1.0)

        prev_history_len = len(session.game.hand_history)
        thinking, action_cards = session.play_bot_turn()
        trick_completed = len(session.game.hand_history) > prev_history_len

        if trick_completed:
            completed = session.game.hand_history[-1]
            await websocket.send_json({
                "type": "trick_complete",
                "trick": _serialize_completed_trick(completed),
            })
            await asyncio.sleep(1.5)

        state = session.get_state_for_client()
        await websocket.send_json({
            "type": "state_update",
            "state": state,
            "last_action": {
                "player": session.last_player,
                "cards": [int(c.get_index()) for c in action_cards],
            },
            "thinking": thinking,
        })

    # Check if game is over
    if session.is_game_over():
        await websocket.send_json({
            "type": "game_over",
            "state": session.get_state_for_client(),
            "winner": "defenders" if session.game.defender_points >= 80 else "attackers",
        })
