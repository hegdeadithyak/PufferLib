import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import chess
except ImportError as e:
    raise ImportError(
        "pufferlib.alphazero requires the python-chess package. "
        "Install the project dependencies or run `pip install chess`."
    ) from e


CHESS_BOARD_OBS_SIZE = 12 * 64 + 7
CHESS_ACTION_SPACE_SIZE = 64 * 73
CHESS_OBS_SIZE = CHESS_BOARD_OBS_SIZE + CHESS_ACTION_SPACE_SIZE

QUEEN_DIRECTIONS = (
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
)

KNIGHT_DIRECTIONS = (
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
)

UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def encode_move(move, turn):
    from_sq = move.from_square
    to_sq = move.to_square
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(to_sq)
    to_file = chess.square_file(to_sq)
    dr = to_rank - from_rank
    df = to_file - from_file

    if move.promotion in UNDERPROMOTION_PIECES:
        forward = 1 if turn == chess.WHITE else -1
        if dr != forward or df < -1 or df > 1:
            return -1
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        return from_sq * 73 + 64 + piece_idx * 3 + (df + 1)

    for direction_idx, (rank_step, file_step) in enumerate(QUEEN_DIRECTIONS):
        for distance in range(1, 8):
            if dr == rank_step * distance and df == file_step * distance:
                return from_sq * 73 + direction_idx * 7 + (distance - 1)

    for direction_idx, (rank_step, file_step) in enumerate(KNIGHT_DIRECTIONS):
        if dr == rank_step and df == file_step:
            return from_sq * 73 + 56 + direction_idx

    return -1


def legal_action_map(board):
    actions = {}
    for move in board.legal_moves:
        action = encode_move(move, board.turn)
        if action >= 0:
            actions[action] = move
    return actions


def encode_board(board):
    obs = np.zeros(CHESS_OBS_SIZE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        obs[plane * 64 + square] = 1.0

    aux = 12 * 64
    obs[aux + 0] = float(board.has_kingside_castling_rights(chess.WHITE))
    obs[aux + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    obs[aux + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    obs[aux + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    obs[aux + 4] = chess.square_file(board.ep_square) / 7.0 if board.ep_square is not None else -1.0
    obs[aux + 5] = board.halfmove_clock / 100.0
    obs[aux + 6] = 0.0 if board.turn == chess.WHITE else 1.0

    mask_offset = CHESS_BOARD_OBS_SIZE
    for action in legal_action_map(board):
        obs[mask_offset + action] = 1.0

    return obs


def terminal_value(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.bn1(self.conv1(x))
        return F.relu(x + residual)


class ChessAlphaZeroNet(nn.Module):
    def __init__(self, channels=256, blocks=19):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(19, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, CHESS_ACTION_SPACE_SIZE)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc0 = nn.Linear(8 * 8, channels)
        self.value_fc1 = nn.Linear(channels, 1)

    def _obs_to_planes(self, obs):
        obs = obs.float()
        board = obs[:, :12 * 64].reshape(obs.shape[0], 12, 8, 8)
        aux = obs[:, 12 * 64:CHESS_BOARD_OBS_SIZE].reshape(obs.shape[0], 7, 1, 1)
        aux = aux.expand(-1, -1, 8, 8)
        return torch.cat([board, aux], dim=1)

    def forward(self, obs):
        x = self._obs_to_planes(obs)
        x = self.tower(self.stem(x))

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_fc(policy.flatten(1))

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = F.relu(self.value_fc0(value.flatten(1)))
        value = torch.tanh(self.value_fc1(value)).squeeze(-1)
        return policy, value


@dataclass
class Edge:
    action: int
    move: chess.Move
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: object = None

    @property
    def q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class Node:
    def __init__(self):
        self.edges = {}
        self.expanded = False


class AlphaZeroMCTS:
    def __init__(self, model, device, simulations, c_puct, dirichlet_alpha, root_noise):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise = root_noise

    @torch.no_grad()
    def evaluate(self, board):
        obs_np = encode_board(board)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        logits, value = self.model(obs)
        mask = torch.from_numpy(obs_np[CHESS_BOARD_OBS_SIZE:]).to(self.device, dtype=torch.bool)
        logits = logits[0].masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=0).detach().cpu().numpy()
        return probs, float(value.item())

    def expand(self, node, board):
        actions = legal_action_map(board)
        if not actions:
            node.expanded = True
            return 0.0

        priors, value = self.evaluate(board)
        prior_sum = float(sum(priors[action] for action in actions))
        uniform = 1.0 / len(actions)
        node.edges = {
            action: Edge(
                action=action,
                move=move,
                prior=float(priors[action] / prior_sum) if prior_sum > 0 else uniform,
            )
            for action, move in actions.items()
        }
        node.expanded = True
        return value

    def add_root_noise(self, root):
        if not root.edges or self.root_noise <= 0:
            return

        actions = list(root.edges)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, eta in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = (1.0 - self.root_noise) * edge.prior + self.root_noise * float(eta)

    def select_edge(self, node):
        total_visits = sum(edge.visit_count for edge in node.edges.values())
        sqrt_visits = math.sqrt(max(1, total_visits))

        best_score = -float("inf")
        best_edge = None
        for edge in node.edges.values():
            u = self.c_puct * edge.prior * sqrt_visits / (1 + edge.visit_count)
            score = edge.q + u
            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def simulate(self, board, node):
        value = terminal_value(board)
        if value is not None:
            return value

        if not node.expanded:
            return self.expand(node, board)

        edge = self.select_edge(node)
        if edge.child is None:
            edge.child = Node()

        board.push(edge.move)
        value = -self.simulate(board, edge.child)
        board.pop()

        edge.visit_count += 1
        edge.value_sum += value
        return value

    def run(self, board, add_noise=True):
        root = Node()
        self.expand(root, board)
        if add_noise:
            self.add_root_noise(root)

        for _ in range(self.simulations):
            self.simulate(board, root)

        return root


@dataclass
class ReplaySample:
    fen: str
    actions: np.ndarray
    probs: np.ndarray
    value: float


def visit_policy(root, temperature):
    actions = []
    visits = []
    for action, edge in root.edges.items():
        actions.append(action)
        visits.append(edge.visit_count)

    actions = np.asarray(actions, dtype=np.int64)
    visits = np.asarray(visits, dtype=np.float32)
    if len(actions) == 0:
        return actions, visits

    if temperature <= 1e-6:
        probs = np.zeros_like(visits, dtype=np.float32)
        probs[int(np.argmax(visits))] = 1.0
        return actions, probs

    visits = np.power(visits, 1.0 / temperature)
    total = float(visits.sum())
    if total <= 0:
        probs = np.full_like(visits, 1.0 / len(visits), dtype=np.float32)
    else:
        probs = visits / total
    return actions, probs.astype(np.float32)


def sample_action(actions, probs, rng):
    if len(actions) == 0:
        return -1
    return int(rng.choices(list(actions), weights=list(probs), k=1)[0])


def random_endgame(rng):
    for _ in range(200):
        board = chess.Board.empty()
        strong = chess.WHITE if rng.random() < 0.5 else chess.BLACK
        weak = not strong
        occupied = set()

        strong_king = rng.randrange(64)
        occupied.add(strong_king)
        board.set_piece_at(strong_king, chess.Piece(chess.KING, strong))

        edge_squares = [
            sq for sq in range(64)
            if chess.square_rank(sq) in (0, 7) or chess.square_file(sq) in (0, 7)
        ]
        weak_king = rng.choice(edge_squares)
        if weak_king in occupied:
            continue
        if chess.square_distance(strong_king, weak_king) <= 1:
            continue
        occupied.add(weak_king)
        board.set_piece_at(weak_king, chess.Piece(chess.KING, weak))

        pieces = [chess.QUEEN] if rng.random() < 0.5 else [chess.ROOK, chess.ROOK]
        for piece_type in pieces:
            square = rng.randrange(64)
            tries = 0
            while square in occupied and tries < 100:
                square = rng.randrange(64)
                tries += 1
            if square in occupied:
                break
            occupied.add(square)
            board.set_piece_at(square, chess.Piece(piece_type, strong))
        else:
            board.turn = strong
            board.castling_rights = 0
            board.ep_square = None
            board.halfmove_clock = 0
            board.fullmove_number = 1
            if board.is_valid():
                return board

    return chess.Board()


def new_game(curriculum, rng):
    if curriculum == "endgame":
        return random_endgame(rng)
    if curriculum == "mixed" and rng.random() < 0.5:
        return random_endgame(rng)
    return chess.Board()


def play_game(model, cfg, device, rng):
    mcts = AlphaZeroMCTS(
        model=model,
        device=device,
        simulations=cfg["mcts_simulations"],
        c_puct=cfg["c_puct"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        root_noise=cfg["root_dirichlet_fraction"],
    )
    board = new_game(cfg["curriculum"], rng)
    trajectory = []

    for ply in range(cfg["max_game_plies"]):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break

        temperature = cfg["temperature"] if ply < cfg["temperature_moves"] else 0.0
        root = mcts.run(board, add_noise=True)
        actions, probs = visit_policy(root, temperature)
        action = sample_action(actions, probs, rng)
        if action < 0:
            break

        trajectory.append((board.fen(), board.turn, actions, probs))
        board.push(root.edges[action].move)

    outcome = board.outcome(claim_draw=True)
    winner = outcome.winner if outcome is not None else None

    samples = []
    for fen, turn, actions, probs in trajectory:
        if winner is None:
            value = 0.0
        else:
            value = 1.0 if winner == turn else -1.0
        samples.append(ReplaySample(
            fen=fen,
            actions=actions.astype(np.int16),
            probs=probs.astype(np.float32),
            value=value,
        ))

    return samples, winner


def lr_for_step(step, cfg):
    lr = cfg["learning_rate"]
    if step >= cfg["lr_drop_3"]:
        return lr * 0.001
    if step >= cfg["lr_drop_2"]:
        return lr * 0.01
    if step >= cfg["lr_drop_1"]:
        return lr * 0.1
    return lr


def train_batch(model, optimizer, replay, cfg, device, rng):
    batch = [replay[rng.randrange(len(replay))] for _ in range(cfg["batch_size"])]
    obs_np = np.stack([encode_board(chess.Board(sample.fen)) for sample in batch])
    obs = torch.from_numpy(obs_np).to(device)
    values = torch.tensor([sample.value for sample in batch], dtype=torch.float32, device=device)

    logits, pred_values = model(obs)
    log_probs = F.log_softmax(logits, dim=1)

    policy_losses = []
    for idx, sample in enumerate(batch):
        actions = torch.as_tensor(sample.actions.astype(np.int64), device=device)
        probs = torch.as_tensor(sample.probs, dtype=torch.float32, device=device)
        policy_losses.append(-(probs * log_probs[idx, actions]).sum())

    policy_loss = torch.stack(policy_losses).mean()
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


def save_checkpoint(model, optimizer, args, update, games, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{update:016d}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args,
        "update": update,
        "self_play_games": games,
    }, path)
    return path


def load_checkpoint(model, optimizer, path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _alphazero_config(args):
    cfg = dict(args.get("alphazero", {}))
    defaults = {
        "training_steps": 700_000,
        "batch_size": 4096,
        "replay_buffer_size": 500_000,
        "min_replay_size": 4096,
        "self_play_games_per_iteration": 1,
        "train_steps_per_iteration": 1,
        "mcts_simulations": 800,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "root_dirichlet_fraction": 0.25,
        "temperature": 1.0,
        "temperature_moves": 30,
        "max_game_plies": 512,
        "channels": 256,
        "blocks": 19,
        "learning_rate": 0.2,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "max_grad_norm": 5.0,
        "lr_drop_1": 100_000,
        "lr_drop_2": 300_000,
        "lr_drop_3": 500_000,
        "curriculum": "standard",
        "seed": args.get("train", {}).get("seed", 42),
        "checkpoint_interval": args.get("checkpoint_interval", 200),
        "device": "auto",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def train(env_name, args=None):
    if env_name != "chess":
        raise ValueError("puffer alphazero currently supports the chess env only")

    cfg = _alphazero_config(args or {})
    rng = random.Random(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    use_cuda = torch.cuda.is_available() and cfg["device"] in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ChessAlphaZeroNet(channels=cfg["channels"], blocks=cfg["blocks"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    update = 0
    games = 0
    load_path = (args or {}).get("load_model_path")
    if load_path == "latest":
        root = os.path.join((args or {}).get("checkpoint_dir", "checkpoints"), env_name, "alphazero")
        candidates = []
        for dirpath, _, filenames in os.walk(root):
            candidates.extend(os.path.join(dirpath, name) for name in filenames if name.endswith(".pt"))
        if not candidates:
            raise FileNotFoundError(f"No AlphaZero checkpoints found under {root}")
        load_path = max(candidates, key=os.path.getctime)
    if load_path:
        checkpoint = load_checkpoint(model, optimizer, load_path, device)
        update = int(checkpoint.get("update", 0))
        games = int(checkpoint.get("self_play_games", 0))
        print(f"Loaded AlphaZero checkpoint {load_path}")

    replay = deque(maxlen=cfg["replay_buffer_size"])
    run_id = str(int(1000 * time.time()))
    checkpoint_dir = os.path.join(
        (args or {}).get("checkpoint_dir", "checkpoints"),
        env_name,
        "alphazero",
        run_id,
    )

    print(
        f"AlphaZero chess on {device}: updates={cfg['training_steps']} "
        f"batch={cfg['batch_size']} sims={cfg['mcts_simulations']} "
        f"channels={cfg['channels']} blocks={cfg['blocks']} "
        f"curriculum={cfg['curriculum']}"
    )

    last_log = time.time()
    last_path = ""
    while update < cfg["training_steps"]:
        model.eval()
        game_samples = 0
        for _ in range(cfg["self_play_games_per_iteration"]):
            samples, _ = play_game(model, cfg, device, rng)
            replay.extend(samples)
            game_samples += len(samples)
            games += 1

        if len(replay) < cfg["min_replay_size"]:
            if time.time() - last_log > 1.0:
                print(
                    f"az warmup games={games} replay={len(replay)}/"
                    f"{cfg['min_replay_size']} latest_game_positions={game_samples}"
                )
                last_log = time.time()
            continue

        model.train()
        metrics = {}
        for _ in range(cfg["train_steps_per_iteration"]):
            if update >= cfg["training_steps"]:
                break
            lr = lr_for_step(update, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            metrics = train_batch(model, optimizer, replay, cfg, device, rng)
            update += 1

        if cfg["checkpoint_interval"] > 0 and update % cfg["checkpoint_interval"] == 0:
            last_path = save_checkpoint(model, optimizer, args, update, games, checkpoint_dir)

        if time.time() - last_log > 1.0:
            print(
                f"az update={update}/{cfg['training_steps']} games={games} "
                f"replay={len(replay)} lr={optimizer.param_groups[0]['lr']:.4g} "
                f"loss={metrics.get('loss', float('nan')):.4f} "
                f"policy={metrics.get('policy_loss', float('nan')):.4f} "
                f"value={metrics.get('value_loss', float('nan')):.4f}"
            )
            last_log = time.time()

    if not last_path:
        last_path = save_checkpoint(model, optimizer, args, update, games, checkpoint_dir)
    print(f"Saved AlphaZero checkpoint {last_path}")
    return last_path
