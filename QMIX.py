### Imports ###

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
import gym
import numpy as np
import torch
from PIL import Image
import os
from IPython.display import display, Image as IPImage
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

## Uncomment if you'd like to use your personal Google Drive to store outputs
## from your runs. You can find some hooks to Google Drive commented
## throughout the rest of this code.
# from google.colab import drive

### Environment setup ###

## Swap between the 3 layouts here:
#layout = "cramped_room"
layout = "coordination_ring"
# layout = "counter_circuit_o_1order"
IS_CRAMPED = (layout == "cramped_room")
IS_RING = (layout == "coordination_ring")
IS_CIRCUIT = (layout == "counter_circuit_o_1order")

## Reward shaping is disabled by default; i.e., only the sparse rewards are
## included in the reward returned by the enviornment).  If you'd like to do
## reward shaping (recommended to make the task much easier to solve), this
## data structure provides access to a built-in reward-shaping mechanism within
## the Overcooked environment.  You can, of course, do your own reward shaping
## in lieu of, or in addition to, using this structure. The shaped rewards
## provided by this structure will appear in a different place (see below)
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
from collections import deque
def needs_swap(env) -> bool:
    # Overcooked flips agent roles on reset; info arrays must be swapped when this is 1
    return bool(getattr(env, "agent_idx", 0))
def align_pair_by_env(pair, env):
    if pair is None: return pair
    return (pair[1], pair[0]) if needs_swap(env) else (pair[0], pair[1])
def get_obs_pair(obs_dict):
    # Your env returns dict; features are already aligned for obs order
    return obs_dict["both_agent_obs"][0], obs_dict["both_agent_obs"][1]
def count_delivery(R, info) -> int:
    # Prefer explicit counters; else detect +20 spike robustly
    if isinstance(info, dict) and "num_delivered" in info:
        return int(info["num_delivered"])
    return int(float(R) >= 10.0)
def shaped_team_reward(info, env, scale: float = 1.0) -> float:
    # Mild shaped reward for stability; excludes the +20 soup delivery
    rs = info.get("shaped_r_by_agent")
    if rs is None: return 0.0
    r0, r1 = align_pair_by_env(rs, env)
    return float(scale * 0.5 * (r0 + r1))

def likely_legal_interact(feats: np.ndarray) -> bool:
    """
    Use featurized obs to guess if Interact makes sense right now.
    Works with the documented 96-D features:
      - if holding onion/soup/dish -> interact can place/use
      - if closest_* distances are (±1,0) or (0,±1) -> adjacent
    We don't need exact indices; we check the 'closest_*' 2D deltas blocks.
    """
    # crude but effective: any of the dx,dy in these feature slots equals ±1 with the other 0
    # player block starts; from the spec: orientation(4), held_obj(4), then nearest deltas etc.
    # safest generic check: look for any exact [±1,0] or [0,±1] in the vector
    v = feats.astype(np.int32)
    for i in range(0, len(v)-1, 1):
        dx, dy = v[i], v[i+1]
        if (abs(dx) == 1 and dy == 0) or (abs(dy) == 1 and dx == 0):
            return True
    return False
def heuristic_nudge(obs_feat: np.ndarray) -> int:
    """
    Minimal bias: move along the axis with larger |dx| toward any nearby target.
    We scan for any (dx,dy) pair with finite nonzero offsets and step closer.
    Fallback to random move if nothing stands out.
    """
    v = obs_feat
    best = None; best_manh = 1e9; best_dxdy = (0,0)
    # scan all adjacent pairs as candidate (dx,dy). In 96-D layout, dx/dy pairs are plentiful.
    for i in range(0, len(v)-1):
        dx, dy = v[i], v[i+1]
        m = abs(dx) + abs(dy)
        if 0 < m < best_manh and m < 6:   # ignore faraway junk; cap radius
            best_manh = m; best_dxdy = (dx, dy)
            best = i
    if best is None:
        return np.random.choice([1,2,3,4])
    dx, dy = best_dxdy
    if abs(dx) >= abs(dy):
        return 4 if dx > 0 else 3  # right vs left
    else:
        return 2 if dy > 0 else 1  # down vs up

def maybe_bias_moves(step, a0, a1, o0, o1):
    if step < 10_000 and np.random.rand() < 0.25:
        a0 = heuristic_nudge(o0) if np.random.rand() < 0.6 else a0
        a1 = heuristic_nudge(o1) if np.random.rand() < 0.6 else a1
    return a0, a1
INTERACT = 5
def mask_interact(obs0: np.ndarray, obs1: np.ndarray, a0: int, a1: int):
    """If Interact is unlikely, replace with a random move."""
    if a0 == INTERACT and not likely_legal_interact(obs0):
        a0 = np.random.choice([1,2,3,4])   # NSEW
    if a1 == INTERACT and not likely_legal_interact(obs1):
        a1 = np.random.choice([1,2,3,4])
    return int(a0), int(a1)

def compute_shaped_rewards(
    info: dict,
    env,
    step: int,
    *,
    sparse_R: float,
    early_shape_steps: int = 50_000,
    shape_scale_max: float = 6.0,
    shape_scale_min: float = 1.0,
    extra_event_bonus: float = 5.0
):
    """
    Returns:
      R_team: float        # the sparse team reward (unchanged)
      r0_total: float      # per-agent 0: shaped component (scaled)
      r1_total: float      # per-agent 1: shaped component (scaled)
      shaped_hit: int      # 1 if any non-zero shaping happened on this step, else 0
    """
    # Scale schedule: big at start, smaller later
    shape_scale = shape_scale_max if step < early_shape_steps else shape_scale_min

    # Default no shaping
    r0, r1 = 0.0, 0.0

    # Built-in shaping from the env (e.g., onion in pot, dish pickup, soup pickup)
    rs = info.get("shaped_r_by_agent")
    if rs is not None:
        # env can swap player roles; align to current index
        if getattr(env, "agent_idx", 0):
            # env.agent_idx == 1 means swap order
            r0, r1 = float(rs[1]), float(rs[0])
        else:
            r0, r1 = float(rs[0]), float(rs[1])

    # Optional: bonus for clearly attributable key events if your env surfaces them
    # (Guarded to not crash when keys are missing.)
    # Example keys you might add in a custom wrapper:
    #   info["soup_plate_pickup_by_agent"] in {0,1}
    #   info["served_by_agent"] in {0,1}
    a_plate = info.get("soup_plate_pickup_by_agent", None)
    if a_plate in (0, 1):
        if a_plate == 0: r0 += extra_event_bonus
        else:            r1 += extra_event_bonus

    a_serve = info.get("served_by_agent", None)
    if a_serve in (0, 1):
        # small extra nudge on top of +20 sparse (don’t overpower)
        if a_serve == 0: r0 += extra_event_bonus * 0.5
        else:            r1 += extra_event_bonus * 0.5

    # Scale shaping
    r0_total = r0 * shape_scale
    r1_total = r1 * shape_scale

    shaped_hit = int((r0_total != 0.0) or (r1_total != 0.0))
    return float(sparse_R), float(r0_total), float(r1_total), shaped_hit

# ===== QMIX components =====
class AgentQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=128):
        super().__init__()
        self.embed_role = nn.Embedding(2, 8)         # 2 roles → 8 dims
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 8, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, act_dim)
        )
    def forward(self, obs, role_id):  # obs: (B,obs_dim), role_id: (B,)
        e = self.embed_role(role_id)  # (B,8)
        x = torch.cat([obs, e], dim=-1)
        return self.net(x)

class MonotonicMixer(nn.Module):
    """
    QMIX mixer: Q_tot = w1 * Q1 + w2 * Q2 + b, with w>=0 enforced by softplus.
    Hypernet produces weights from global state s (concat of both obs here).
    """
    def __init__(self, state_dim, n_agents=2, hidden=64):
        super().__init__()
        self.n = n_agents
        self.state_dim = state_dim
        self.w1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n))
        self.b1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, q_agents, state):
        # q_agents: (B, n_agents) selected action-values
        # state:    (B, state_dim)
        w = F.softplus(self.w1(state))           # (B, n_agents), nonnegative
        b = self.b1(state)                       # (B, 1)
        qtot = (w * q_agents).sum(dim=1, keepdim=True) + b
        return qtot.squeeze(-1)                  # (B,)

class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)
    def push(self, *x):
        self.buf.append(tuple(x))
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        # each item: (s, o0, o1, a0, a1, r, s_next, o0n, o1n, done)
        trans = list(zip(*batch))
        return [np.array(x) for x in trans]
    def __len__(self):
        return len(self.buf)

class QMIX:
    def __init__(self, obs_dim, act_dim, gamma=0.99, lr=3e-4, tau=0.01, n_agents=2):
        self.n = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau

        # Agent Q nets (shared weights for both agents keeps symmetry; switch to separate if you like)
        self.q = AgentQ(obs_dim, act_dim).to(device)
        self.q_tgt = AgentQ(obs_dim, act_dim).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        # fixed role ids for the two agents
        self.role0 = torch.tensor([0], dtype=torch.long, device=device)  # shape (1,)
        self.role1 = torch.tensor([1], dtype=torch.long, device=device)

        # Mixer over global state = concat(o0, o1)
        state_dim = 2 * obs_dim
        self.mixer = MonotonicMixer(state_dim).to(device)
        self.mixer_tgt = MonotonicMixer(state_dim).to(device)
        self.mixer_tgt.load_state_dict(self.mixer.state_dict())

        self.opt = torch.optim.Adam(list(self.q.parameters()) + list(self.mixer.parameters()), lr=lr)

        # ε-greedy
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 5e-3  # tune if needed

    def act_pair(self, o0, o1):
        # ε-greedy per agent
        a = []
        for o in [o0, o1]:
            if np.random.rand() < self.eps:
                a.append(np.random.randint(self.act_dim))
            else:
                with torch.no_grad():
                    q0 = self.q(torch.as_tensor(o0[None, :], dtype=torch.float32, device=device), self.role0)
                    q1 = self.q(torch.as_tensor(o1[None, :], dtype=torch.float32, device=device), self.role1)
                    a.append(int(q0.argmax(1).item()));
                    a.append(int(q1.argmax(1).item()))
        return a[0], a[1]

    def soft_update(self, net, tgt):
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def train_step(self, replay: Replay, batch_size=128):
        if len(replay) < batch_size:
            return None

        s, o0, o1, a0, a1, r, s2, o0n, o1n, done = replay.sample(batch_size)

        # tensors
        o0  = torch.as_tensor(o0,  dtype=torch.float32, device=device)
        o1  = torch.as_tensor(o1,  dtype=torch.float32, device=device)
        a0  = torch.as_tensor(a0,  dtype=torch.long,    device=device)
        a1  = torch.as_tensor(a1,  dtype=torch.long,    device=device)
        r   = torch.as_tensor(r,   dtype=torch.float32, device=device)
        o0n = torch.as_tensor(o0n, dtype=torch.float32, device=device)
        o1n = torch.as_tensor(o1n, dtype=torch.float32, device=device)
        done= torch.as_tensor(done,dtype=torch.float32, device=device)

        s   = torch.as_tensor(s,   dtype=torch.float32, device=device)   # (B, 2*obs)
        s2  = torch.as_tensor(s2,  dtype=torch.float32, device=device)

        # batch role ids: (B,)
        B = o0.shape[0]
        rid0 = torch.zeros(B, dtype=torch.long, device=device)  # all 0s
        rid1 = torch.ones(B, dtype=torch.long, device=device)  # all 1s

        # current Q for chosen actions
        q0 = self.q(o0, rid0).gather(1, a0.unsqueeze(1)).squeeze(1)
        q1 = self.q(o1, rid1).gather(1, a1.unsqueeze(1)).squeeze(1)
        q_agents = torch.stack([q0, q1], dim=1)  # (B,2)
        q_tot = self.mixer(q_agents, s)  # (B,)

        # Target Q: max over next actions per agent (Double-Q optional)
        with torch.no_grad():
            q0n_all = self.q_tgt(o0n, rid0)  # (B,A)
            q1n_all = self.q_tgt(o1n, rid1)
            a0n = q0n_all.argmax(dim=1)
            a1n = q1n_all.argmax(dim=1)
            q0n = q0n_all.gather(1, a0n.unsqueeze(1)).squeeze(1)
            q1n = q1n_all.gather(1, a1n.unsqueeze(1)).squeeze(1)
            qn_agents = torch.stack([q0n, q1n], dim=1)
            qn_tot = self.mixer_tgt(qn_agents, s2)
            target = r + self.gamma * (1.0 - done) * qn_tot

        loss = F.smooth_l1_loss(q_tot, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q.parameters()) + list(self.mixer.parameters()), 10.0)
        self.opt.step()

        # targets
        self.soft_update(self.q, self.q_tgt)
        self.soft_update(self.mixer, self.mixer_tgt)

        # decay epsilon
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        return float(loss.item())

def train_qmix(
    env,
    steps_total=600_000,
    start_learning=10_000,   # start updates sooner
    batch_size=256,
    log_every=5000,
    early_shape_steps=20_000  # stronger shaping early on
):

    # --- init env / agent / replay ---
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]
    act_dim = env.action_space.n

    agent = QMIX(obs_dim, act_dim, gamma=0.99, lr=3e-4, tau=0.01)
    # speed up ε decay so we don’t explore forever
    agent.eps_min = 0.10
    agent.eps_decay = 3e-3

    replay = Replay(cap=200_000)

    # rolling debug counters
    hits = 0
    interacts = 0
    adj = 0
    best_soups = -1.0
    for step in range(1, steps_total + 1):
        # ---- pick actions (no masking during training) ----

        a0, a1 = agent.act_pair(o0, o1)
        MASK_PROB = 0.3  # 30% of the time, clean up obviously useless Interact
        if np.random.rand() < MASK_PROB:
            a0, a1 = mask_interact(o0, o1, a0, a1)  # remember your fix: returns (a0,a1)
        a0, a1 = maybe_bias_moves(step, a0, a1, o0, o1)
        # ---- step env ----
        next_obs, R, done, info = env.step([a0, a1])

        R_team, r0_shape, r1_shape, shaped_hit = compute_shaped_rewards(
            info, env, step=step, sparse_R=float(R),
            early_shape_steps=20_000, shape_scale_max=6.0, shape_scale_min=1.0
        )

        # Option A (simple): fold both agents’ shaping into a single scalar
        r_team_shaped = R_team + 0.5 * (r0_shape + r1_shape)

        # push to replay (unchanged structure), using r_team_shaped as 'r'
        o0n, o1n = get_obs_pair(next_obs)
        state = np.concatenate([o0, o1], axis=-1)
        state_nxt = np.concatenate([o0n, o1n], axis=-1)
        replay.push(state, o0, o1, a0, a1, r_team_shaped, state_nxt, o0n, o1n, float(done))

        # book-keeping for logs
        hits += shaped_hit

        # advance
        o0, o1 = o0n, o1n
        if done:
            obs = env.reset()
            o0, o1 = get_obs_pair(obs)

        # ---- learn ----
        loss = None
        if step >= start_learning:
            loss = agent.train_step(replay, batch_size=batch_size)

        # ---- periodic eval & logs ----
        if step % log_every == 0:
            # print shaping/behavior diagnostics over last window
            shaped_hit_rate = hits / (log_every + 1e-9)
            interact_rate   = interacts / (2 * log_every + 1e-9)
            adj_rate        = adj / (2 * log_every + 1e-9)
            print(f"[step {step}] eps={agent.eps:.3f} "
                  f"loss={(loss if loss is not None else float('nan')):.3f} "
                  f"shaped_hit_rate={shaped_hit_rate:.3f} "
                  f"interact_rate={interact_rate:.3f} "
                  f"adj_rate={adj_rate:.3f}")
            hits = interacts = adj = 0

            # greedy eval with a light interact mask for stability
            mean_ret, mean_soups = eval_qmix(agent, env, episodes=20)
            print(f"           eval_return={mean_ret:.1f} soups/ep={mean_soups:.2f}")

            # IMPORTANT: eval used the same env; reset before resuming training
            obs = env.reset()
            o0, o1 = get_obs_pair(obs)
            best_soups = max(best_soups, mean_soups)
            if best_soups == mean_soups and mean_ret!=0:
                torch.save({
                    "q": agent.q.state_dict(),
                    "mixer": agent.mixer.state_dict()
                }, f"overcooked_{layout}_qmix.pt")
                print(f"Checkpoint saved to overcooked_{layout}_qmix.pt")
    return agent


@torch.no_grad()
def eval_qmix(agent: QMIX, env, episodes=20):
    """Greedy eval; apply a light interact mask to avoid dumb wall-pokes."""
    eps_bak = agent.eps
    agent.eps = 0.0
    rets, soups = [], 0
    for _ in range(episodes):
        obs = env.reset()
        o0, o1 = get_obs_pair(obs)
        done, ep_ret = False, 0.0
        while not done:
            a0, a1 = agent.act_pair(o0, o1)
            # eval-only stability: fuzzy, probabilistic masking
            a0, a1 = mask_interact(o0, o1, int(a0), int(a1))
            obs, R, done, info = env.step([a0, a1])
            ep_ret += float(R)
            soups += count_delivery(float(R), info)
            o0, o1 = get_obs_pair(obs)
        rets.append(ep_ret)
    agent.eps = eps_bak
    return float(np.mean(rets)), soups / float(episodes)

agent = train_qmix(env,
                   steps_total=600_000,
                   start_learning=5_000,
                   batch_size=128,
                   log_every=5000)
