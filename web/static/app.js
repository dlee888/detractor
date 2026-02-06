// DeTractor - Tractor Game

const SUITS = {
    CLUBS: { symbol: '\u2663', color: 'black' },
    DIAMONDS: { symbol: '\u2666', color: 'red' },
    HEARTS: { symbol: '\u2665', color: 'red' },
    SPADES: { symbol: '\u2660', color: 'black' }
};
const SUIT_ORDER = ['CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES'];
const RANK_ORDER = ['THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN', 'JACK', 'QUEEN', 'KING', 'ACE', 'TWO'];
const RANK_DISPLAY = { THREE: '3', FOUR: '4', FIVE: '5', SIX: '6', SEVEN: '7', EIGHT: '8', NINE: '9', TEN: '10', JACK: 'J', QUEEN: 'Q', KING: 'K', ACE: 'A', TWO: '2', BJ: 'B', RJ: 'R' };

function cardToIndex(suit, rank) {
    if (rank === 'BJ') return 52;
    if (rank === 'RJ') return 53;
    return RANK_ORDER.indexOf(rank) * 4 + SUIT_ORDER.indexOf(suit);
}

function indexToCard(idx) {
    if (idx === 52) return { rank: 'BJ', suit: 'CLUBS', color: 'black' };
    if (idx === 53) return { rank: 'RJ', suit: 'HEARTS', color: 'red' };
    const rankIdx = Math.floor(idx / 4);
    const suitIdx = idx % 4;
    return { rank: RANK_ORDER[rankIdx], suit: SUIT_ORDER[suitIdx], color: SUITS[SUIT_ORDER[suitIdx]].color };
}

function formatCard(idx) {
    const c = indexToCard(idx);
    if (c.rank === 'BJ') return 'BJ';
    if (c.rank === 'RJ') return 'RJ';
    return RANK_DISPLAY[c.rank] + SUITS[c.suit].symbol;
}

class TractorGame {
    constructor() {
        this.ws = null;
        this.state = null;
        this.humanSeats = [];
        this.sandboxMode = false;
        this.mySeats = [0];       // which players you control (sandbox mode)
        this.showAllCards = true;
        // History
        this.tricks = [];
        this.currentTrick = [];
        // Card picker state
        this.pickerMode = null;    // 'hand', 'play'
        this.pickerPlayer = 0;
        this.pickerSelection = new Set();
        this.handSetupDone = false;
        this.handsToSetup = [];    // queue of seats to set up hands for
        this.initElements();
        this.initEventListeners();
    }

    initElements() {
        this.lobbyScreen = document.getElementById('lobby');
        this.gameScreen = document.getElementById('game');
        this.gameOverScreen = document.getElementById('game-over');
        this.startBtn = document.getElementById('start-btn');
        this.sandboxStartBtn = document.getElementById('sandbox-start-btn');
        this.playerSelects = [0,1,2,3].map(i => document.getElementById('player' + i));
        this.trumpSelect = document.getElementById('trump-select');
        this.leadSelect = document.getElementById('lead-select');
        this.mySeatCheckboxes = document.querySelectorAll('.my-seat-cb');
        this.virtualMode = document.getElementById('virtual-mode');
        this.sandboxModeDiv = document.getElementById('sandbox-mode');
        this.trumpSuitEl = document.getElementById('trump-suit');
        this.defenderPointsEl = document.getElementById('defender-points');
        this.attackerPointsEl = document.getElementById('attacker-points');
        this.undoBtn = document.getElementById('undo-btn');
        this.suggestBtn = document.getElementById('suggest-btn');
        this.quitBtn = document.getElementById('quit-btn');
        this.playerHands = [0,1,2,3].map(i => document.getElementById('player-' + i + '-hand'));
        this.playerAreas = [0,1,2,3].map(i => document.getElementById('player-' + i + '-area'));
        this.trickSlots = document.querySelectorAll('.trick-slot');
        this.turnIndicator = document.getElementById('turn-indicator');
        this.currentTurnEl = document.getElementById('current-turn');
        this.commitBtn = document.getElementById('commit-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.historyContent = document.getElementById('history-content');
        this.thinkingContent = document.getElementById('thinking-content');
        this.cardPicker = document.getElementById('card-picker');
        this.cardGrid = document.getElementById('card-grid');
        this.pickerTitle = document.getElementById('picker-title');
        this.pickerDone = document.getElementById('picker-done');
        this.gameResult = document.getElementById('game-result');
        this.finalDefender = document.getElementById('final-defender');
        this.finalAttacker = document.getElementById('final-attacker');
        this.playAgainBtn = document.getElementById('play-again-btn');
        this.suggestionModal = document.getElementById('suggestion-modal');
        this.suggestionContent = document.getElementById('suggestion-content');
    }

    initEventListeners() {
        this.startBtn.addEventListener('click', () => this.startGame());
        this.sandboxStartBtn.addEventListener('click', () => this.startSandbox());
        this.undoBtn.addEventListener('click', () => this.sendUndo());
        this.suggestBtn.addEventListener('click', () => this.requestSuggestion());
        this.quitBtn.addEventListener('click', () => this.quitGame());
        this.commitBtn.addEventListener('click', () => this.commitSelection());
        this.clearBtn.addEventListener('click', () => this.clearSelection());
        this.playAgainBtn.addEventListener('click', () => this.showScreen('lobby'));
        document.querySelectorAll('.modal-close').forEach(btn => btn.addEventListener('click', () => this.hideModal()));
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                const mode = e.target.dataset.mode;
                this.virtualMode.classList.toggle('hidden', mode !== 'virtual');
                this.sandboxModeDiv.classList.toggle('hidden', mode !== 'sandbox');
            });
        });
        this.pickerDone.addEventListener('click', () => this.onPickerDone());
        document.getElementById('show-all-toggle').addEventListener('change', (e) => {
            this.showAllCards = e.target.checked;
            this.renderState();
        });
        // Toggle checkpoint selector visibility based on bot type
        const hintBotSelect = document.getElementById('hint-bot-select');
        const checkpointSelect = document.getElementById('checkpoint-select');
        hintBotSelect.addEventListener('change', () => {
            checkpointSelect.classList.toggle('hidden', hintBotSelect.value !== 'ppo');
        });
        // Load available checkpoints
        this.loadCheckpoints();
    }

    async loadCheckpoints() {
        try {
            const resp = await fetch('/api/checkpoints');
            const data = await resp.json();
            const select = document.getElementById('checkpoint-select');
            select.innerHTML = '';
            if (data.checkpoints && data.checkpoints.length > 0) {
                data.checkpoints.forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    select.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = '(no checkpoints)';
                select.appendChild(opt);
            }
        } catch (e) {
            console.error('Failed to load checkpoints:', e);
        }
    }

    getSelectedSeats() {
        const seats = [];
        this.mySeatCheckboxes.forEach(cb => {
            if (cb.checked) seats.push(parseInt(cb.value));
        });
        return seats.length > 0 ? seats : [0];
    }

    connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws`);
        this.ws.onmessage = (e) => this.handleMessage(JSON.parse(e.data));
        this.ws.onerror = (e) => console.error('WS error', e);
        this.ws.onclose = () => {
            console.warn('WebSocket closed');
            this.currentTurnEl.textContent = 'Disconnected';
            this.turnIndicator.classList.remove('your-turn');
        };
    }

    handleMessage(msg) {
        console.log('Received:', msg);
        if (msg.type === 'game_start') {
            this.humanSeats = msg.human_seats || [0];
            this.sandboxMode = msg.sandbox_mode || false;
            if (this.sandboxMode) {
                this.mySeats = msg.my_seats || [0];
            }
            this.state = msg.state;
            this.showScreen('game');
            this.renderState();
            if (this.sandboxMode) {
                this.handSetupDone = false;
                this.handsToSetup = [...this.mySeats];
                this.openNextHandSetup();
            }
        } else if (msg.type === 'trick_complete') {
            this.showCompletedTrick(msg.trick);
        } else if (msg.type === 'state_update') {
            if (msg.last_action?.cards?.length > 0) this.addMoveToHistory(msg.last_action);
            if (msg.thinking) this.showThinking(msg.thinking);
            this.state = msg.state;
            this.renderState();
            // In sandbox mode, auto-open picker for other players' turns
            if (this.sandboxMode && this.handSetupDone) {
                this.checkSandboxTurn();
            }
        } else if (msg.type === 'game_over') {
            this.state = msg.state;
            this.showGameOver(msg.winner);
        } else if (msg.type === 'suggestion') {
            this.showSuggestion(msg.suggestion);
        } else if (msg.type === 'error') {
            alert('Error: ' + msg.message);
        }
    }

    startGame() {
        this.sandboxMode = false;
        this.clearHistory();
        this.connect();
        const wait = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                clearInterval(wait);
                this.ws.send(JSON.stringify({ type: 'configure', players: this.playerSelects.map(s => s.value) }));
            }
        }, 100);
    }

    startSandbox() {
        this.sandboxMode = true;
        this.mySeats = this.getSelectedSeats();
        this.clearHistory();
        this.connect();
        const wait = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                clearInterval(wait);
                this.ws.send(JSON.stringify({
                    type: 'start_sandbox',
                    trump_suit: this.trumpSelect.value,
                    lead_player: parseInt(this.leadSelect.value),
                    my_seats: this.mySeats,
                }));
            }
        }, 100);
    }

    // --- Sandbox mode flow ---

    openNextHandSetup() {
        if (this.handsToSetup.length === 0) {
            this.handSetupDone = true;
            setTimeout(() => this.checkSandboxTurn(), 200);
            return;
        }
        const seat = this.handsToSetup.shift();
        this.pickerMode = 'hand';
        this.pickerPlayer = seat;
        this.pickerSelection = new Set();
        this.showCardPicker(`Set hand for P${seat}`);
    }

    checkSandboxTurn() {
        if (!this.state || this.state.game_over) return;
        const cur = this.state.current_player;
        if (this.mySeats.includes(cur)) {
            // It's one of your seats' turn - play from hand normally
            return;
        }
        // It's another player's turn - open picker for their play
        this.openOtherPlayerPicker(cur);
    }

    openOtherPlayerPicker(playerIdx) {
        this.pickerMode = 'play';
        this.pickerPlayer = playerIdx;
        this.pickerSelection = new Set();
        this.showCardPicker(`What did P${playerIdx} play?`);
    }

    onPickerDone() {
        const cards = [...this.pickerSelection];
        if (this.pickerMode === 'hand') {
            // Finished setting hand for this seat
            this.ws.send(JSON.stringify({ type: 'set_hand', player: this.pickerPlayer, cards }));
            this.cardPicker.classList.add('hidden');
            // Set up next seat's hand, or finish
            setTimeout(() => this.openNextHandSetup(), 200);
        } else if (this.pickerMode === 'play') {
            // Submit another player's play
            if (cards.length === 0) return; // don't submit empty
            this.ws.send(JSON.stringify({ type: 'sandbox_play', player: this.pickerPlayer, cards }));
            this.cardPicker.classList.add('hidden');
        }
    }

    // --- Card Picker ---

    showCardPicker(title) {
        this.cardGrid.innerHTML = '';
        this.cardPicker.querySelectorAll('.joker-row').forEach(r => r.remove());
        this.pickerTitle.textContent = title || 'Select cards';

        for (const suit of SUIT_ORDER) {
            for (const rank of RANK_ORDER) {
                const idx = cardToIndex(suit, rank);
                const el = this.createPickerCard(idx, suit, rank);
                this.cardGrid.appendChild(el);
            }
        }
        const jokerRow = document.createElement('div');
        jokerRow.className = 'joker-row';
        jokerRow.appendChild(this.createPickerCard(52, 'CLUBS', 'BJ'));
        jokerRow.appendChild(this.createPickerCard(53, 'HEARTS', 'RJ'));
        this.cardGrid.parentNode.insertBefore(jokerRow, this.pickerDone);
        this.cardPicker.classList.remove('hidden');
    }

    createPickerCard(idx, suit, rank) {
        const el = document.createElement('div');
        el.className = 'card ' + SUITS[suit].color;
        el.dataset.idx = idx;
        if (this.pickerSelection.has(idx)) el.classList.add('in-hand');
        el.innerHTML = `<span class="rank">${RANK_DISPLAY[rank]}</span><span class="suit">${SUITS[suit].symbol}</span>`;
        el.addEventListener('click', () => this.togglePickerCard(idx, el));
        return el;
    }

    togglePickerCard(idx, el) {
        if (this.pickerSelection.has(idx)) {
            this.pickerSelection.delete(idx);
            el.classList.remove('in-hand');
        } else {
            this.pickerSelection.add(idx);
            el.classList.add('in-hand');
        }
        // For hand setup, send live updates so state renders
        if (this.pickerMode === 'hand') {
            const cards = [...this.pickerSelection];
            this.ws.send(JSON.stringify({ type: 'set_hand', player: this.pickerPlayer, cards }));
        }
    }

    // --- Screens ---

    showScreen(name) {
        this.lobbyScreen.classList.remove('active');
        this.gameScreen.classList.remove('active');
        this.gameOverScreen.classList.remove('active');
        document.getElementById(name === 'game-over' ? 'game-over' : name).classList.add('active');
    }

    // --- Rendering ---

    getTeamLabel(pi) {
        if (!this.state) return '';
        const host = this.state.host;
        const defTeam = [host, (host + 2) % 4];
        return defTeam.includes(pi) ? 'Def' : 'Atk';
    }

    renderState() {
        if (!this.state) return;
        const s = SUITS[this.state.trump_suit];
        this.trumpSuitEl.innerHTML = `<span style="color:${s.color === 'red' ? 'var(--red)' : 'var(--black)'}">${s.symbol}</span>`;
        this.defenderPointsEl.textContent = this.state.defender_points;
        this.attackerPointsEl.textContent = this.state.attacker_points;

        const cur = this.state.current_player;
        const isMyTurn = this.sandboxMode ? this.mySeats.includes(cur) : this.humanSeats.includes(cur);
        this.turnIndicator.classList.toggle('your-turn', isMyTurn);
        if (this.state.game_over) {
            this.currentTurnEl.textContent = 'Game Over';
        } else if (this.sandboxMode) {
            this.currentTurnEl.textContent = this.mySeats.includes(cur) ? `P${cur} (Your turn)` : `P${cur}'s turn`;
        } else {
            this.currentTurnEl.textContent = `P${cur}${this.humanSeats.includes(cur) ? ' (You)' : ''}`;
        }

        this.playerAreas.forEach((a, i) => {
            if (!a) return;
            a.classList.toggle('current-turn', i === cur && !this.state.game_over);
            const label = a.querySelector('.player-label');
            if (label) {
                const team = this.getTeamLabel(i);
                const isMe = this.sandboxMode ? this.mySeats.includes(i) : this.humanSeats.includes(i);
                label.textContent = `P${i} [${team}]${isMe ? ' (You)' : ''}`;
                label.className = 'player-label ' + (team === 'Def' ? 'def-label' : 'atk-label');
            }
        });

        this.renderHands();
        this.renderTrick();

        // Show play controls only when it's your turn in the right mode
        const playControls = document.getElementById('play-controls');
        if (playControls) {
            const showControls = this.sandboxMode ? isMyTurn : this.humanSeats.includes(cur);
            playControls.classList.toggle('hidden', !showControls);
        }
        this.commitBtn.disabled = !this.state.can_commit;
    }

    renderHands() {
        const legal = new Set(this.state.legal_cards);
        const partial = new Set(this.state.partial_selection);
        const cur = this.state.current_player;

        this.state.hands.forEach((hand, pi) => {
            const el = this.playerHands[pi];
            if (!el) return;
            el.innerHTML = '';

            const isMe = this.sandboxMode ? this.mySeats.includes(pi) : this.humanSeats.includes(pi);
            const isTurn = this.sandboxMode
                ? (pi === cur && this.mySeats.includes(pi))
                : (pi === cur && this.humanSeats.includes(cur));
            const showFaceUp = isMe || this.showAllCards;

            hand.forEach(c => {
                if (showFaceUp) {
                    const small = !isMe;
                    const cardEl = this.createCardElement(c, isTurn && legal.has(c.index), partial.has(c.index), small);
                    if (isTurn) cardEl.addEventListener('click', () => this.selectCard(c.index));
                    el.appendChild(cardEl);
                } else {
                    el.appendChild(this.createCardBack());
                }
            });
        });
    }

    renderTrick() {
        this.trickSlots.forEach(slot => {
            const pi = parseInt(slot.dataset.player);
            const t = this.state.trick[pi];
            slot.innerHTML = '';
            if (t?.cards) {
                t.cards.forEach(c => slot.appendChild(this.createCardElement(c, false, false, false)));
            }
        });
    }

    showCompletedTrick(trickData) {
        this.trickSlots.forEach(slot => {
            const pi = parseInt(slot.dataset.player);
            slot.innerHTML = '';
            if (trickData[pi]?.cards) {
                trickData[pi].cards.forEach(c => slot.appendChild(this.createCardElement(c, false, false, false)));
            }
        });
    }

    createCardElement(card, isLegal, isSelected, small) {
        const el = document.createElement('div');
        const isJoker = card.rank === 'BJ' || card.rank === 'RJ';
        const suit = SUITS[card.suit];
        el.className = 'card ' + (isJoker ? (card.rank === 'RJ' ? 'red' : 'black') : (suit?.color || 'black'));
        if (small) el.classList.add('small');
        if (this.isTrump(card)) el.classList.add('trump');
        if (isLegal) el.classList.add('legal');
        if (isSelected) el.classList.add('selected');
        if (isJoker) {
            el.innerHTML = `<span class="rank">${card.rank === 'RJ' ? 'R' : 'B'}</span><span class="suit">J</span>`;
        } else {
            el.innerHTML = `<span class="rank">${RANK_DISPLAY[card.rank]}</span><span class="suit">${suit.symbol}</span>`;
        }
        return el;
    }

    createCardBack() {
        const el = document.createElement('div');
        el.className = 'card card-back small';
        return el;
    }

    isTrump(c) {
        return c.rank === 'BJ' || c.rank === 'RJ' || c.rank === 'TWO' || c.suit === this.state?.trump_suit;
    }

    // --- Actions ---

    selectCard(idx) {
        if (!this.state.legal_cards.includes(idx)) return;
        this.ws.send(JSON.stringify({ type: 'play', cards: [idx] }));
    }

    clearSelection() {
        this.ws.send(JSON.stringify({ type: 'clear_selection' }));
    }

    commitSelection() {
        if (!this.state.can_commit) return;
        this.ws.send(JSON.stringify({ type: 'play', cards: [54] }));
    }

    sendUndo() { this.ws.send(JSON.stringify({ type: 'undo' })); }
    requestSuggestion() {
        const botType = document.getElementById('hint-bot-select').value;
        const msg = { type: 'get_suggestion', bot_type: botType };
        if (botType === 'ppo') {
            const checkpoint = document.getElementById('checkpoint-select').value;
            if (checkpoint) msg.checkpoint = checkpoint;
        }
        this.ws.send(JSON.stringify(msg));
    }

    // --- Modals ---

    showSuggestion(sug) {
        this.suggestionContent.innerHTML = sug.action === 'commit'
            ? '<p>Play your selected cards now!</p>'
            : sug.action === 'error'
            ? `<p>${sug.reason}</p>`
            : `<p>Play:</p><div class="suggested-card">${sug.card}</div><p class="suggestion-reason">${sug.reason || ''}</p>`;
        this.suggestionModal.classList.remove('hidden');
    }
    hideModal() { this.suggestionModal.classList.add('hidden'); }

    showThinking(thinking) {
        if (!this.thinkingContent || !thinking) return;
        const cards = thinking.cards_played ? thinking.cards_played.join(', ') : '';
        const reason = thinking.reason || '';
        const type = thinking.player_type || '';
        this.thinkingContent.innerHTML =
            `<div class="thinking-entry">` +
            `<span class="thinking-type">${type}</span> ` +
            `<span class="thinking-cards">${cards}</span>` +
            (reason ? `<div class="thinking-reason">${reason}</div>` : '') +
            `</div>`;
    }

    // --- Game flow ---

    quitGame() {
        if (confirm('Quit game?')) {
            this.ws?.close();
            this.showScreen('lobby');
        }
    }

    showGameOver(winner) {
        const defTeam = [this.state.host, (this.state.host + 2) % 4];
        const humanDef = this.humanSeats.some(s => defTeam.includes(s));
        const humanAtk = this.humanSeats.some(s => !defTeam.includes(s));
        const won = (winner === 'defenders' && humanDef) || (winner === 'attackers' && humanAtk);
        this.gameResult.textContent = won ? 'You Win!' : 'You Lose';
        this.gameResult.className = 'result ' + (won ? 'win' : 'lose');
        this.finalDefender.textContent = this.state.defender_points;
        this.finalAttacker.textContent = this.state.attacker_points;
        this.showScreen('game-over');
    }

    // --- History ---

    addMoveToHistory(action) {
        const cards = action.cards.filter(c => c < 54).map(idx => formatCard(idx));
        if (cards.length === 0) return;
        this.currentTrick.push({ player: action.player, cards });
        if (this.currentTrick.length === 4) {
            this.tricks.push([...this.currentTrick]);
            this.currentTrick = [];
        }
        this.renderHistory();
    }

    renderHistory() {
        if (this.tricks.length === 0 && this.currentTrick.length === 0) {
            this.historyContent.innerHTML = '<span style="color:var(--muted)">No moves yet</span>';
            return;
        }
        let html = '';
        this.tricks.forEach((trick, ti) => {
            html += `<div class="trick-group">`;
            html += `<div class="trick-header">Trick ${ti + 1}</div>`;
            trick.forEach(m => {
                html += `<div class="history-entry"><span class="player">P${m.player}</span> ${m.cards.join(' ')}</div>`;
            });
            html += `</div>`;
        });
        if (this.currentTrick.length > 0) {
            html += `<div class="trick-group current">`;
            html += `<div class="trick-header">Trick ${this.tricks.length + 1}</div>`;
            this.currentTrick.forEach(m => {
                html += `<div class="history-entry"><span class="player">P${m.player}</span> ${m.cards.join(' ')}</div>`;
            });
            html += `</div>`;
        }
        this.historyContent.innerHTML = html;
        this.historyContent.scrollTop = this.historyContent.scrollHeight;
    }

    clearHistory() {
        this.tricks = [];
        this.currentTrick = [];
        this.renderHistory();
    }
}

document.addEventListener('DOMContentLoaded', () => { window.game = new TractorGame(); });
