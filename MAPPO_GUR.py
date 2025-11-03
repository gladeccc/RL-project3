### Imports ###

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import warnings
warnings.filterwarnings("ignore")
import gym
import numpy as np
import torch
from PIL import Image
import os
from IPython.display import display, Image as IPImage
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt
from gym.vector import AsyncVectorEnv

## Uncomment if you'd like to use your personal Google Drive to store outputs
## from your runs. You can find some hooks to Google Drive commented
## throughout the rest of this code.
# from google.colab import drive

### Environment setup ###

## Swap between the 3 layouts here:


## Reward shaping is disabled by default; i.e., only the sparse rewards are
## included in the reward returned by the enviornment).  If you'd like to do
## reward shaping (recommended to make the task much easier to solve), this
## data structure provides access to a built-in reward-shaping mechanism within
## the Overcooked environment.  You can, of course, do your own reward shaping
## in lieu of, or in addition to, using this structure. The shaped rewards
## provided by this structure will appear in a different place (see below)
reward_shaping = {
    # "NEAR_POT_REWARD": 1,
    # "NEAR_ONION_REWARD": 1,
    # "NEAR_DISH_REWARD": 1,
    "ONION_PICKUP_REWARD": 1,
    "DISH_PICKUP_REWARD": 1,
    "SOUP_PICKUP_REWARD": 5,
    "PLACEMENT_IN_POT_REW": 4,
    "SOUP_DELIVERY_REWARD_SHAPING":5,
    "PICKUP_WRONG_OBJ_PEN":-1,
    "USEFUL_INTERACT_REWARD": 2,
    "INVALID_INTERACT_PENALTY": -2,
    "STEP_COST": -0.01
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
def get_obs_pair(obs_dict):
    # Your env returns dict; features are already aligned for obs order
    return obs_dict["both_agent_obs"][0], obs_dict["both_agent_obs"][1]

def needs_swap(env) -> bool:
    # Overcooked flips agent roles on reset; info arrays must be swapped when this is 1
    return bool(getattr(env, "agent_idx", 0))

def align_pair_by_env(pair, env):
    if pair is None: return pair
    return (pair[1], pair[0]) if needs_swap(env) else (pair[0], pair[1])

def shaped_team_reward(info, env, scale: float = 1.0) -> float:
    # Mild shaped reward for stability; excludes the +20 soup delivery
    rs = info.get("shaped_r_by_agent")
    if rs is None: return 0.0
    r0, r1 = align_pair_by_env(rs, env)
    return float(scale * 0.5 * (r0 + r1))

def count_delivery(R, info) -> int:
    # Prefer explicit counters; else detect +20 spike robustly
    if isinstance(info, dict) and "num_delivered" in info:
        return int(info["num_delivered"])
    return int(float(R) >= 10.0)

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

def maybe_bias_actions(step_idx: int, upd_idx: int, a0: int, a1: int, o0: np.ndarray, o1: np.ndarray):
    """For the first ~10 updates, with small prob, overwrite actions with heuristic moves."""
    if upd_idx <= 10 and np.random.rand() < 0.30:
        a0 = heuristic_nudge(o0) if np.random.rand() < 0.7 else a0
        a1 = heuristic_nudge(o1) if np.random.rand() < 0.7 else a1
    return a0, a1

# ====== MAPPO networks (shared actor, centralized critic) ======
def mlp(sizes, act=nn.Tanh):
    layers=[]
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2: layers.append(act())
    return nn.Sequential(*layers)

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

def mask_interact(obs0: np.ndarray, obs1: np.ndarray, a0: int, a1: int):
    """If Interact is unlikely, replace with a random move."""
    if a0 == INTERACT and not likely_legal_interact(obs0):
        a0 = np.random.choice([1,2,3,4])   # NSEW
    if a1 == INTERACT and not likely_legal_interact(obs1):
        a1 = np.random.choice([1,2,3,4])
    return a0, a1

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

def plot_training_metrics(results_dict, metric="reward",save_dir=None, filename=None, dpi=200, fmt="png",
                          title_prefix="MAPPO training performance"):
    """
    results_dict: dict like {
        'cramped': {'reward': list, 'soups': list},
        'ring': {'reward': list, 'soups': list},
        'circuit': {'reward': list, 'soups': list}
    }
    metric: 'reward' or 'soups'
    """
    plt.figure(figsize=(8,5))
    for name, data in results_dict.items():
        y = data['reward'] if metric == "reward" else data['soups']
        x = np.arange(len(y)) * 10  # one point every 10 updates
        plt.plot(x, y, label=name)
    plt.xlabel("Training updates")
    plt.ylabel("Mean " + ("return" if metric=="reward" else "soups/episode"))
    plt.title(f"MAPPO training performance ({metric})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    saved_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = f"{filename}_{metric}_curve"
        saved_path = os.path.join(save_dir, f"{filename}.{fmt}")
        plt.savefig(saved_path, dpi=dpi, format=fmt, bbox_inches="tight")
    return saved_path

# ---------- helpers to build vector envs ----------
def make_overcooked_env(layout, reward_shaping, horizon):
    def _init():
        mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
        print(mdp.rew_shaping_params)
        base = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        return gym.make("Overcooked-v0", base_env=base, featurize_fn=base.featurize_state_mdp)
    return _init

def build_vec_env(layouts, copies_per_layout=2):
    env_fns = []
    layout_ids = []
    for l in layouts:
        for _ in range(copies_per_layout):
            env_fns.append(make_overcooked_env(l, reward_shaping, horizon))
            layout_ids.append(l)
    vec = AsyncVectorEnv(env_fns)
    layout_ids = np.array(layout_ids)  # shape (N,)
    return vec, layout_ids

def get_layout_norm(layout_name: str, env, pool: dict):
    if layout_name not in pool:
        obs = env.reset()
        o0, _ = get_obs_pair(obs)
        pool[layout_name] = ObsNorm(dim=o0.shape[0])  # your ObsNorm class
    return pool[layout_name]

def augment_obs(obs_vec, agent_idx, use_norm=False, obsnorm=None):
    agent_id = np.array([1, 0], dtype=np.float32) if agent_idx == 0 else np.array([0, 1], dtype=np.float32)
    return np.concatenate([obs_vec, agent_id], axis=-1)

def load_mappo_ckpt(file,layout):
    ckpt, path_used = None, None
    if os.path.exists(file):
        ckpt = torch.load(file, map_location="cpu")
        path_used = file
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for layout={layout}. Tried: {file}")

    actor_sd  = ckpt["actor"]
    critic_sd = ckpt["critic"]
    obsnorm = None
    if all(k in ckpt for k in ["obsnorm_m","obsnorm_s","obsnorm_n"]):
        m, s, n = ckpt["obsnorm_m"], ckpt["obsnorm_s"], ckpt["obsnorm_n"]
        obsnorm = ObsNorm(dim=m.shape[0])
        obsnorm.m, obsnorm.s, obsnorm.n = m, s, float(n)

    print(f"Loaded checkpoint: {path_used} | has_norm_stats={obsnorm is not None}")
    return actor_sd, critic_sd, obsnorm

def visualization_using_existing_flow(agent,layout, use_norm=False, obsnorm=None):
    ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
    featurize_fn = ae.env.featurize_state_mdp

    actor_sd, critic_sd, obsnorm = load_mappo_ckpt(layout)
    agent.actor.load_state_dict(actor_sd)
    agent.critic.load_state_dict(critic_sd)
    class VizPolicy(NNPolicy):
        def __init__(self, actor, featurize_fn, use_norm=False, obsnorm=None):
            super().__init__()
            self.actor = actor.eval()
            self.featurize_fn = featurize_fn
            self.use_norm = use_norm
            self.obsnorm = obsnorm

        def state_policy(self, state, agent_index):
            feats = self.featurize_fn(state)[agent_index]  # 96
            if self.use_norm and self.obsnorm is not None:
                feats = self.obsnorm.apply(feats)
            feats = augment_obs(feats, agent_index)
            x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = self.actor(x).squeeze(0).cpu().numpy()
            probs = np.exp(logits - logits.max());
            probs /= (probs.sum() + 1e-8)
            return probs

        def multi_state_policy(self, states, agent_indices):
            return [self.state_policy(s, i) for s, i in zip(states, agent_indices)]

    policy0 = StudentPolicy(agent.actor)
    policy1 = StudentPolicy(agent.actor)
    agent_pair = AgentPair(AgentFromPolicy(policy0), AgentFromPolicy(policy1))

    trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
    print("len(trajs):", len(trajs))
    img_dir = f"imgs/{layout}/"
    ipython_display = True
    StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir, ipython_display=ipython_display)


class Actor(nn.Module):
    """
    Input:  x        [B, T, obs_dim]  (during update) or [B, 1, obs_dim] (during rollout)
            h0       [1, B, hid]      initial hidden
    Output: logits   [B, T, act_dim]
            hT       [1, B, hid]
    """
    def __init__(self, obs_dim, act_dim, hid=128):
        super().__init__()
        self.inp = mlp([obs_dim, 128])      # small pre-MLP (Tanh inside your mlp)
        self.rnn = nn.GRU(128, hid, batch_first=True)
        self.head = nn.Linear(hid, act_dim)

    def forward(self, x, h0):
        # x: [B, T, obs_dim]
        b, t, d = x.shape
        x = self.inp(x.view(b*t, d)).view(b, t, -1)
        y, hT = self.rnn(x, h0)             # y: [B, T, hid]
        logits = self.head(y)                # [B, T, act_dim]
        return logits, hT

class CentralCritic(nn.Module):
    # Takes concatenated [obs0, obs1] and outputs team value
    def __init__(self, joint_dim):
        super().__init__()
        self.v = mlp([joint_dim, 256, 256, 1])
    def forward(self, joint_obs):
        return self.v(joint_obs).squeeze(-1)

# ----- action masking: only press Interact when facing something useful -----
# Action mapping we’ve been using: 0:stay, 1:up, 2:down, 3:left, 4:right, 5:interact
INTERACT = 5



@dataclass
class PPOCfg:
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-4
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_env_steps: int = 4096  # env steps/update
    minibatch_size: int = 512
    opt_iters: int = 8
    total_updates: int = 1500

    shaping_scale: float = 1.0     # mild nudge only

# ====== Centralized-critic PPO trainer that fits your env ======
class PPOMulti:
    def __init__(self, obs_dim, act_dim, cfg: PPOCfg, hid=128):
        obs_dim += 2
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, hid).to(device)
        self.critic = CentralCritic(2*obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr
        )

    def rnn_forward_with_masks(actor, obs_seq, h0, masks):
        """
        obs_seq: [1, T, D]
        h0:      [1, 1, H]
        masks:   [1, T]  (1=延续同一回合, 0=新回合第一步)
        return:  logits [1, T, A]
        """
        B, T, D = obs_seq.shape
        h = h0
        logits_list = []
        for t in range(T):
            m_t = masks[:, t].view(1, 1, 1)  # [1,1,1]
            h = h * m_t
            x_t = obs_seq[:, t:t + 1, :]  # [1,1,D]
            logit_t, h = actor(x_t, h)
            logits_list.append(logit_t)  # [1,1,A]
        return torch.cat(logits_list, dim=1)  # [1,T,A]

    def dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs_np, h_np):
        x = torch.as_tensor(obs_np[None, None, :], dtype=torch.float32, device=device)
        h = torch.as_tensor(h_np, dtype=torch.float32, device=device)
        logits, hT = self.actor(x, h)  # [1,1,A]
        last_logits = logits[:, -1, :]  # [1,A]
        d = self.dist(last_logits)
        a = d.sample()
        return int(a.item()), float(d.log_prob(a).item()), hT.detach().cpu().numpy(), last_logits.squeeze(
            0).cpu().numpy()

    @torch.no_grad()
    def value(self, joint_np):
        j = torch.as_tensor(joint_np, dtype=torch.float32, device=device)
        return self.critic(j).cpu().numpy()

    def update(self, batch):
        # ---- helpers ----
        def to_t(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=device)

        def rnn_forward_with_masks(actor, obs_seq, h0, masks):
            """
            obs_seq: [1, T, D]
            h0:      [1, 1, H]
            masks:   [1, T]  (1=同一回合延续, 0=新回合第一步)
            return:  logits [1, T, A]
            """
            B, T, D = obs_seq.shape
            h = h0
            outs = []
            for t in range(T):
                m_t = masks[:, t].view(1, 1, 1)  # [1,1,1]
                h = h * m_t  # 在新回合清零隐藏态
                x_t = obs_seq[:, t:t + 1, :]  # [1,1,D]
                logit_t, h = actor(x_t, h)  # GRU 前向
                outs.append(logit_t)  # [1,1,A]
            return torch.cat(outs, dim=1)  # [1,T,A]

        # ---- build tensors ----
        T = batch["obs0"].shape[0]
        obs0 = to_t(batch["obs0"]).unsqueeze(0)  # [1,T,D]
        obs1 = to_t(batch["obs1"]).unsqueeze(0)
        act0 = to_t(batch["act0"], torch.int64).unsqueeze(0)  # [1,T]
        act1 = to_t(batch["act1"], torch.int64).unsqueeze(0)
        old0 = to_t(batch["logp0"]).unsqueeze(0)  # [1,T]
        old1 = to_t(batch["logp1"]).unsqueeze(0)
        adv = to_t(batch["adv"]).unsqueeze(0)  # [1,T]
        ret = to_t(batch["ret"]).unsqueeze(0)  # [1,T]
        jobs = to_t(batch["joint"])  # [T,2D]
        masks = to_t(batch["mask"]).unsqueeze(0)  # [1,T]
        h0_0 = to_t(batch["h0_0"])  # [1,1,H]
        h0_1 = to_t(batch["h0_1"])

        # ---- RNN forward with per-step mask resets ----
        logits0 = rnn_forward_with_masks(self.actor, obs0, h0_0, masks)  # [1,T,A]
        logits1 = rnn_forward_with_masks(self.actor, obs1, h0_1, masks)

        d0 = torch.distributions.Categorical(logits=logits0.squeeze(0))  # [T,A]
        d1 = torch.distributions.Categorical(logits=logits1.squeeze(0))

        logp0 = d0.log_prob(act0.squeeze(0))  # [T]
        logp1 = d1.log_prob(act1.squeeze(0))  # [T]

        ratio0 = torch.exp(logp0 - old0.squeeze(0))
        ratio1 = torch.exp(logp1 - old1.squeeze(0))

        m = masks.squeeze(0)  # [T]
        adv0 = adv.squeeze(0) * m
        adv1 = adv.squeeze(0) * m

        clip = self.cfg.clip
        pi_loss0 = -torch.mean(torch.min(ratio0 * adv0,
                                         torch.clamp(ratio0, 1 - clip, 1 + clip) * adv0))
        pi_loss1 = -torch.mean(torch.min(ratio1 * adv1,
                                         torch.clamp(ratio1, 1 - clip, 1 + clip) * adv1))
        ent = torch.mean(d0.entropy() + d1.entropy())

        v = self.critic(jobs)  # [T]
        v_loss = self.cfg.vf_coef * F.mse_loss(v * m, ret.squeeze(0) * m)

        pi_loss = pi_loss0 + pi_loss1 - self.cfg.ent_coef * ent

        self.opt.zero_grad()
        (pi_loss + v_loss).backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                 self.cfg.max_grad_norm)
        self.opt.step()
class MinimalRNNPolicy:
    """
    A minimal policy wrapper for Overcooked's AgentFromPolicy.
    - No inheritance from NNPolicy (avoids .body/.head expectations).
    - Keeps a per-agent GRU hidden state across calls.
    - Accepts optional obsnorm and augment_obs usage.
    """
    def __init__(self, actor: Actor, featurize_fn, use_norm=False, obsnorm=None, device="cpu"):
        self.actor = actor.eval()
        self.featurize_fn = featurize_fn
        self.use_norm = use_norm
        self.obsnorm = obsnorm
        self.device = device
        self.h = {}  # maps agent_index -> np.array([1,1,H])

    def _ensure_h(self, agent_index: int, hid_size: int):
        if agent_index not in self.h:
            self.h[agent_index] = np.zeros((1, 1, hid_size), np.float32)

    def state_policy(self, state, agent_index):
        # 1) featurize
        feats = self.featurize_fn(state)[agent_index]        # shape [obs_dim]
        if self.use_norm and self.obsnorm is not None:
            feats = self.obsnorm.apply(feats)
        feats = augment_obs(feats, agent_index)              # +2 agent id

        # 2) forward through your GRU actor with persistent hidden state
        hid_size = self.actor.rnn.hidden_size
        self._ensure_h(agent_index, hid_size)

        x = torch.as_tensor(feats, dtype=torch.float32, device=self.device).view(1, 1, -1)
        h0 = torch.as_tensor(self.h[agent_index], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, hT = self.actor(x, h0)                   # [1,1,A], [1,1,H]
        self.h[agent_index] = hT.cpu().numpy()

        # 3) return probabilities
        logits = logits.squeeze(0).squeeze(0).cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-8
        return probs

    def multi_state_policy(self, states, agent_indices):
        return [self.state_policy(s, i) for s, i in zip(states, agent_indices)]
def logp_from_logits_np(logits_np, a_idx: int) -> float:
    # logits_np: shape [A], numpy
    t = torch.as_tensor(logits_np[None, :], dtype=torch.float32)
    d = torch.distributions.Categorical(logits=t)
    a = torch.tensor([a_idx], dtype=torch.int64)
    return float(d.log_prob(a).item())

class ObsNorm:
    def __init__(self, dim, eps=1e-8):
        self.m = np.zeros(dim, np.float32)
        self.s = np.ones(dim,  np.float32)
        self.n = eps
    def update(self, x):
        self.n += 1
        delta = x - self.m
        self.m += delta / self.n
        self.s += delta * (x - self.m)
    def apply(self, x):
        var = np.clip(self.s / max(self.n-1, 1), 1e-3, 1e9)
        return (x - self.m) / np.sqrt(var)
def visulize(agent, ae, layout, horizon=400, use_norm=False, obsnorm=None,
             img_root="imgs/GUR", ipython_display=False):
    # Build evaluator for this layout to get the correct featurizer
    featurize_fn = ae.env.featurize_state_mdp  # do NOT use global base_env

    class StudentPolicy(NNPolicy):
        """
        Wraps the trained shared actor. Returns a probability vector over 6 actions.
        """
        def __init__(self, actor: Actor):
            super().__init__()
            self.actor = actor.eval()  # inference mode

        def state_policy(self, state, agent_index):
            # 1) 96-D features from the evaluator env
            feats = featurize_fn(state)[agent_index]
            # 2) If you trained with normalization, apply it here
            if use_norm and (obsnorm is not None):
                feats = obsnorm.apply(feats)
            # 3) Append the 2-D agent one-hot used in training → 98-D
            feats = augment_obs(feats, agent_index)

            x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            assert x.shape[-1] == agent.actor.body[0].in_features, \
                f"Feature dim mismatch: got {x.shape[-1]}, expected {agent.actor.body[0].in_features}"
            with torch.no_grad():
                logits = agent.actor(x).squeeze(0).cpu().numpy()
            probs = np.exp(logits - logits.max()); probs /= (probs.sum() + 1e-8)
            return probs

        def multi_state_policy(self, states, agent_indices):
            return [self.state_policy(s, i) for s, i in zip(states, agent_indices)]

    policy0 = StudentPolicy(agent.actor)
    policy1 = StudentPolicy(agent.actor)
    pair = AgentPair(AgentFromPolicy(policy0), AgentFromPolicy(policy1))

    trajs = ae.evaluate_agent_pair(pair, num_games=1)
    out_dir = os.path.join(img_root, layout)
    os.makedirs(out_dir, exist_ok=True)
    StateVisualizer().display_rendered_trajectory(
        trajs, img_directory_path=out_dir, ipython_display=ipython_display
    )
    print("len(trajs):", len(trajs), "| saved to:", out_dir)
def compute_adv_ret_from_time_steps(rews, vals, dones, *, gamma=0.99, lam=0.95, last_v=None, last_done=None):
    rews = np.asarray(rews, dtype=np.float32)
    vals = np.asarray(vals, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T = len(rews)

    if last_v is None:
        last_v = 0.0
    if last_done is None:
        last_done = 1.0  # 默认为终局，行为与旧实现一致


    next_vals = np.empty_like(vals, dtype=np.float32)
    if T > 1:
        next_vals[:-1] = vals[1:]
    next_vals[-1] = float(last_v)


    next_mask = 1.0 - np.concatenate([dones[1:], np.array([last_done], np.float32)])

    deltas = rews + gamma * next_vals * next_mask - vals

    adv = np.zeros_like(rews, dtype=np.float32)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * lam * next_mask[t] * gae
        adv[t] = gae

    ret = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

def build_batch_from_time_steps(step_obs0, step_obs1,
                                step_act0, step_act1,
                                step_logp0, step_logp1,
                                step_joint,
                                adv, ret):
    obs = np.vstack([np.asarray(step_obs0, dtype=np.float32),
                     np.asarray(step_obs1, dtype=np.float32)])  # [2T, obs_dim]
    act = np.concatenate([np.asarray(step_act0, dtype=np.int64),
                          np.asarray(step_act1, dtype=np.int64)], axis=0)  # [2T]
    logp = np.concatenate([np.asarray(step_logp0, dtype=np.float32),
                           np.asarray(step_logp1, dtype=np.float32)], axis=0)
    # 广播
    adv_b = np.concatenate([adv, adv], axis=0).astype(np.float32)  # [2T]
    ret_b = np.concatenate([ret, ret], axis=0).astype(np.float32)  # [2T]
    joint = np.vstack([np.asarray(step_joint, dtype=np.float32),
                       np.asarray(step_joint, dtype=np.float32)])  # [2T, 2*obs_dim]
    return {
        "obs": obs, "act": act, "logp": logp,
        "adv": adv_b, "ret": ret_b, "joint_obs": joint
    }
def train_mappo(env, updates=2000, rollout_steps=2048, shaping_scale=1.0):
    hid = 128
    h0 = np.zeros((1, 1, hid), dtype=np.float32)  # agent 0 hidden
    h1 = np.zeros((1, 1, hid), dtype=np.float32)  # agent 1 hidden
    # Probe dims
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]; act_dim = env.action_space.n

    # # Running normalizer
    # obsnorm = ObsNorm(obs_dim)

    agent = PPOMulti(obs_dim, act_dim, PPOCfg(
        total_updates=updates, rollout_env_steps=rollout_steps, shaping_scale=shaping_scale
    ))

    best_soups = -1.0
    rewards_log, soups_log = [], []
    ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
    for upd in range(1, agent.cfg.total_updates + 1):
        # simple entropy schedule (optional but helps)
        if upd <= int(0.6 * agent.cfg.total_updates):
            agent.cfg.ent_coef = 0.02
        elif upd <= int(0.9 * agent.cfg.total_updates):
            agent.cfg.ent_coef = 0.01
        else:
            agent.cfg.ent_coef = 0.005

        buf = {k: [] for k in ["obs","act","logp","rew","done","val","joint_obs"]}
        steps = 0
        step_obs0, step_obs1 = [], []
        step_act0, step_act1 = [], []
        step_logp0, step_logp1 = [], []
        step_joint, step_val, step_rew, step_done = [], [], [], []
        step_mask = []  # 1 alive step, 0 if this step is the first step of an episode
        init_h0_0 = h0.copy()
        init_h0_1 = h1.copy()
        while steps < agent.cfg.rollout_env_steps:
            o0_aug = augment_obs(o0, agent_idx=0)
            o1_aug = augment_obs(o1, agent_idx=1)
            # x0 = torch.as_tensor(o0_aug[None, :], dtype=torch.float32, device=device)
            # x1 = torch.as_tensor(o1_aug[None, :], dtype=torch.float32, device=device)
            # sample with RNN
            a0, lp0_samp, h0, logits0 = agent.act(o0_aug, h0)
            a1, lp1_samp, h1, logits1 = agent.act(o1_aug, h1)
            # with torch.no_grad():
            #     dist0 = agent.dist(agent.actor(x0))
            #     dist1 = agent.dist(agent.actor(x1))
            #     a0 = int(dist0.sample().item())
            #     a1 = int(dist1.sample().item())

            if upd <= 50 and np.random.rand() < 0.1:
                a0_exec, a1_exec = mask_interact(o0, o1, a0, a1)
                a0_exec, a1_exec = maybe_bias_actions(steps, upd, a0_exec, a1_exec, o0, o1)
            else:
                a0_exec, a1_exec = a0, a1

            lp0 = logp_from_logits_np(logits0, a0_exec)
            lp1 = logp_from_logits_np(logits1, a1_exec)
            # recompute logp for executed actions from the same dists
            # with torch.no_grad():
            #     lp0 = float(dist0.log_prob(torch.tensor(a0_exec, device=device)).cpu().item())
            #     lp1 = float(dist1.log_prob(torch.tensor(a1_exec, device=device)).cpu().item())

            # critic sees NORMALIZED joint
            # joint = np.concatenate([o0n, o1n], axis=-1)
            joint = np.concatenate([o0_aug, o1_aug], axis=-1)
            v = agent.value(joint[None,:])[0]

            # step env
            obs, R, done, info = env.step([a0_exec, a1_exec])

            # sparse + mild shaped reward
            shape_scale = 8.0 if upd <= 200 else agent.cfg.shaping_scale
            r = float(R) + shaped_team_reward(info, env, scale=shape_scale)

            step_obs0.append(o0_aug);
            step_obs1.append(o1_aug)
            step_act0.append(a0_exec);
            step_act1.append(a1_exec)
            step_logp0.append(lp0);
            step_logp1.append(lp1)
            step_joint.append(joint);
            step_val.append(v)
            step_rew.append(r);
            step_done.append(float(done))
            step_mask.append(1.0)  # default alive
            # store NORMALIZED obs so training matches sampling distribution
            # for ob_n, ac, lp in [(o0n, a0_exec, lp0), (o1n, a1_exec, lp1)]:
            for ob, ac, lp in [(o0_aug, a0_exec, lp0), (o1_aug, a1_exec, lp1)]:
                # buf["obs"].append(ob_n)
                buf["obs"].append(ob)
                buf["act"].append(ac)
                buf["logp"].append(lp)
                buf["rew"].append(r)
                buf["done"].append(float(done))
                buf["val"].append(v)
                buf["joint_obs"].append(joint)

            # advance
            o0, o1 = get_obs_pair(obs)
            steps += 1
            if done:
                step_mask[-1] = 0.0
                obs = env.reset()
                o0, o1 = get_obs_pair(obs)
                h0[:] = 0.0;
                h1[:] = 0.0

        # shaped-hit print (unchanged)
        shaped_hits = sum(1 for rr in buf["rew"] if (rr != 0.0 and rr < 20.0))
        print(f"[upd {upd}] Regular： shaped_hit_rate={shaped_hits / len(buf['rew']):.3f}")

        #for k in buf: buf[k] = np.asarray(buf[k], dtype=np.float32)
        last_joint = np.concatenate([o0_aug, o1_aug], axis=-1)
        last_v = agent.value(last_joint[None, :])[0]
        last_done = float(done)  # 1.0 if episode ended, else 0.0
        # === GAE ===
        adv, ret = compute_adv_ret_from_time_steps(
            step_rew, step_val, step_done,
            gamma=agent.cfg.gamma, lam=agent.cfg.lam,
            last_v=last_v, last_done=last_done
        )

        # batch = build_batch_from_time_steps(
        #     step_obs0, step_obs1, step_act0, step_act1,
        #     step_logp0, step_logp1, step_joint, adv, ret
        # )
        batch = {
            "obs0": np.asarray(step_obs0, np.float32),
            "obs1": np.asarray(step_obs1, np.float32),
            "act0": np.asarray(step_act0, np.int64),
            "act1": np.asarray(step_act1, np.int64),
            "logp0": np.asarray(step_logp0, np.float32),
            "logp1": np.asarray(step_logp1, np.float32),
            "adv": adv.astype(np.float32),
            "ret": ret.astype(np.float32),
            "joint": np.asarray(step_joint, np.float32),
            "mask": np.asarray(step_mask, np.float32),
            "h0_0": init_h0_0.astype(np.float32),
            "h0_1": init_h0_1.astype(np.float32),
        }
        agent.update(batch)

        if upd % 10 == 0:
            mean_ret, mean_soups = eval_soups(agent, env, episodes=30)
            rewards_log.append(mean_ret)
            soups_log.append(mean_soups)
            print(f"[upd {upd}] return≈ {mean_ret:.1f}  soups≈ {mean_soups:.2f}  "
                  f"ratio≈ {mean_ret / (20 * max(1e-6, mean_soups)):.2f}")
            obs = env.reset()
            o0, o1 = get_obs_pair(obs)
            if best_soups < mean_soups:
                best_soups = mean_soups
                torch.save({
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                }, f"overcooked_{layout}_GUR.pt")
                print(f"Checkpoint saved to overcooked_{layout}_GUR.pt")

    print(f"Best soups/ep observed: {best_soups:.2f}")
    return agent, rewards_log, soups_log

def train_mappo_norm(env,  obsnorm, updates=2000, rollout_steps=2048, shaping_scale=1.0):
    hid = 128
    h0 = np.zeros((1, 1, hid), dtype=np.float32)  # agent 0 hidden
    h1 = np.zeros((1, 1, hid), dtype=np.float32)  # agent 1 hidden
    # Probe dims
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]; act_dim = env.action_space.n

    agent = PPOMulti(obs_dim, act_dim, PPOCfg(
        total_updates=updates, rollout_env_steps=rollout_steps, shaping_scale=shaping_scale
    ))

    best_soups = -1.0
    rewards_log, soups_log = [], []
    ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
    for upd in range(1, agent.cfg.total_updates + 1):
        # # simple entropy schedule (optional but helps)
        # if upd <= int(0.3 * agent.cfg.total_updates):
        #     agent.cfg.ent_coef = 0.03
        # elif upd <= int(0.7 * agent.cfg.total_updates):
        #     agent.cfg.ent_coef = 0.015
        # else:
        #     agent.cfg.ent_coef = 0.007

        buf = {k: [] for k in ["obs","act","logp","rew","done","val","joint_obs"]}
        steps = 0
        step_obs0, step_obs1 = [], []
        step_act0, step_act1 = [], []
        step_logp0, step_logp1 = [], []
        step_joint, step_val, step_rew, step_done = [], [], [], []
        step_mask = []  # 1 alive step, 0 if this step is the first step of an episode
        init_h0_0 = h0.copy()
        init_h0_1 = h1.copy()
        while steps < agent.cfg.rollout_env_steps:
            # update stats BEFORE using
            obsnorm.update(o0); obsnorm.update(o1)
            o0n = obsnorm.apply(o0)
            o1n = obsnorm.apply(o1)
            o0_aug = augment_obs(o0n, agent_idx=0)
            o1_aug = augment_obs(o1n, agent_idx=1)
            # sample actions from NORMALIZED inputs
            a0, lp0_samp, h0, logits0 = agent.act(o0_aug, h0)
            a1, lp1_samp, h1, logits1 = agent.act(o1_aug, h1)


            if upd <= 50 and np.random.rand() < 0.1:
                a0_exec, a1_exec = mask_interact(o0, o1, a0, a1)
                a0_exec, a1_exec = maybe_bias_actions(steps, upd, a0_exec, a1_exec, o0, o1)
            else:
                a0_exec, a1_exec = a0, a1
            a0e, a1e = a0_exec, a1_exec  # keep whatever ring/circuit tweaks you already do
            # recompute logp for executed actions from the same dists
            lp0 = logp_from_logits_np(logits0, a0_exec)
            lp1 = logp_from_logits_np(logits1, a1_exec)
            # critic sees NORMALIZED joint
            joint = np.concatenate([o0_aug, o1_aug], axis=-1)
            # joint = np.concatenate([o0, o1], axis=-1)
            v = agent.value(joint[None,:])[0]

            # step env
            obs, R, done, info = env.step([a0e, a1e])
            # if "shaped_info_by_agent" in info:
            #     print("shaped_info_by_agent:", info["shaped_info_by_agent"])
            # elif "shaped_r_by_agent" in info:
            #     print("shaped_r_by_agent:", info["shaped_r_by_agent"])
            # print("info keys:", info.keys())
            # sparse + mild shaped reward
            shape_scale = 8.0 if upd <= 200 else agent.cfg.shaping_scale
            r = float(R) + shaped_team_reward(info, env, scale=shape_scale)

            # store NORMALIZED obs so training matches sampling distribution
            for ob_n, ac, lp in [(o0_aug, a0e, lp0), (o1_aug, a1e, lp1)]:
            # for ob, ac, lp in [(o0, a0_exec, lp0), (o1, a1_exec, lp1)]:
                buf["obs"].append(ob_n)
                # buf["obs"].append(ob)
                buf["act"].append(ac)
                buf["logp"].append(lp)
                buf["rew"].append(r)
                buf["done"].append(float(done))
                buf["val"].append(v)
                buf["joint_obs"].append(joint)
            # push time-step
            step_obs0.append(o0_aug);
            step_obs1.append(o1_aug)
            step_act0.append(a0e);
            step_act1.append(a1e)
            step_logp0.append(lp0);
            step_logp1.append(lp1)
            step_joint.append(joint);
            step_val.append(v)
            step_rew.append(r);
            step_done.append(float(done))
            step_mask.append(1.0)  # default alive

            # advance
            o0, o1 = get_obs_pair(obs)
            steps += 1
            if done:
                step_mask[-1] = 0.0
                obs = env.reset()
                o0, o1 = get_obs_pair(obs)
                h0[:] = 0.0;
                h1[:] = 0.0

        # shaped-hit print (unchanged)
        shaped_hits = sum(1 for rr in buf["rew"] if (rr != 0.0 and rr < 20.0))
        print(f"[upd {upd}] Norm：shaped_hit_rate={shaped_hits / len(buf['rew']):.3f}")
        last_joint = np.concatenate([o0_aug, o1_aug], axis=-1)
        last_v = agent.value(last_joint[None, :])[0]
        last_done = float(done)  # 1.0 if episode ended, else 0.0
        # === GAE ===
        adv, ret = compute_adv_ret_from_time_steps(
            step_rew, step_val, step_done,
            gamma=agent.cfg.gamma, lam=agent.cfg.lam,
            last_v=last_v, last_done=last_done
        )

        batch = {
            "obs0": np.asarray(step_obs0, np.float32),
            "obs1": np.asarray(step_obs1, np.float32),
            "act0": np.asarray(step_act0, np.int64),
            "act1": np.asarray(step_act1, np.int64),
            "logp0": np.asarray(step_logp0, np.float32),
            "logp1": np.asarray(step_logp1, np.float32),
            "adv": adv.astype(np.float32),
            "ret": ret.astype(np.float32),
            "joint": np.asarray(step_joint, np.float32),
            "mask": np.asarray(step_mask, np.float32),
            "h0_0": init_h0_0.astype(np.float32),
            "h0_1": init_h0_1.astype(np.float32),
        }
        agent.update(batch)

        if upd % 10 == 0:
            # visulize(agent, ae, layout)
            mean_ret, mean_soups = eval_soups_norm(agent, env, obsnorm,episodes=30)
            rewards_log.append(mean_ret)
            soups_log.append(mean_soups)
            print(f"[upd {upd}] return≈ {mean_ret:.1f}  soups≈ {mean_soups:.2f}  "
                  f"ratio≈ {mean_ret / (20 * max(1e-6, mean_soups)):.2f}")
            obs = env.reset()
            o0, o1 = get_obs_pair(obs)
            if best_soups < mean_soups:
                best_soups = mean_soups
                torch.save({
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "obsnorm_m": obsnorm.m,
                    "obsnorm_s": obsnorm.s,
                    "obsnorm_n": obsnorm.n
                }, f"overcooked_{layout}_GUR_norm.pt")
                print(f"Checkpoint saved to overcooked_{layout}_GUR_norm.pt")

    print(f"Best soups/ep observed: {best_soups:.2f}")
    return agent, rewards_log, soups_log

@torch.no_grad()
def eval_soups(agent, env, episodes=20, hid=128):
    rets, soups = [], 0
    for _ in range(episodes):
        obs = env.reset()
        o0, o1 = get_obs_pair(obs)
        h0 = np.zeros((1,1,hid), np.float32)
        h1 = np.zeros((1,1,hid), np.float32)
        done, ep_ret = False, 0.0
        while not done:
            # Augment with agent id to match training input
            o0_aug = augment_obs(o0, agent_idx=0)
            o1_aug = augment_obs(o1, agent_idx=1)

            x0 = torch.as_tensor(o0_aug[None, None, :], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1_aug[None, None, :], dtype=torch.float32, device=device)
            h0_t = torch.as_tensor(h0, dtype=torch.float32, device=device)
            h1_t = torch.as_tensor(h1, dtype=torch.float32, device=device)
            logits0, h0_next = agent.actor(x0, h0_t)
            logits1, h1_next = agent.actor(x1, h1_t)

            # Stochastic eval to match your agent.act() behavior
            pi0 = torch.distributions.Categorical(logits=logits0[0, -1])
            pi1 = torch.distributions.Categorical(logits=logits1[0, -1])
            a0 = int(pi0.sample().item())
            a1 = int(pi1.sample().item())
            a0, a1 = mask_interact(o0, o1, a0, a1)

            obs, R, done, info = env.step([a0, a1])
            ep_ret += float(R)
            soups += count_delivery(float(R), info)
            o0, o1 = get_obs_pair(obs)
            h0 = h0_next.detach().cpu().numpy()
            h1 = h1_next.detach().cpu().numpy()
        rets.append(ep_ret)
    return float(np.mean(rets)), soups / float(episodes)

@torch.no_grad()
def eval_soups_norm(agent, env, obsnorm, episodes=20, hid=128):
    rets, soups = [], 0
    for _ in range(episodes):
        obs = env.reset()
        o0, o1 = obs["both_agent_obs"]
        h0 = np.zeros((1, 1, hid), np.float32)
        h1 = np.zeros((1, 1, hid), np.float32)
        done, ep_ret = False, 0.0
        while not done:

            o0n = obsnorm.apply(o0);  o1n = obsnorm.apply(o1)
            o0_aug = augment_obs(o0n, 0)
            o1_aug = augment_obs(o1n, 1)


            x0 = torch.as_tensor(o0_aug[None, None, :], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1_aug[None, None, :], dtype=torch.float32, device=device)
            h0_t = torch.as_tensor(h0, dtype=torch.float32, device=device)
            h1_t = torch.as_tensor(h1, dtype=torch.float32, device=device)
            logits0, h0_next = agent.actor(x0, h0_t)   # logits0: [1, 1, A]
            logits1, h1_next = agent.actor(x1, h1_t)

            pi0 = torch.distributions.Categorical(logits=logits0[0, -1])
            pi1 = torch.distributions.Categorical(logits=logits1[0, -1])
            a0 = int(pi0.sample().item())
            a1 = int(pi1.sample().item())
            a0, a1 = mask_interact(o0, o1, a0, a1)


            obs, R, done, info = env.step([a0, a1])
            ep_ret += float(R)
            soups += count_delivery(float(R), info)
            o0, o1 = obs["both_agent_obs"]
            h0 = h0_next.detach().cpu().numpy()
            h1 = h1_next.detach().cpu().numpy()
        rets.append(ep_ret)

    mean_reward = float(np.mean(rets))
    mean_soups = soups / float(episodes)
    return mean_reward, mean_soups

### All of the remaining code in this notebook is solely for using the
### built-in Overcooked state visualizer on a trained agent, so that you can see
### a graphical rendering of what your agents are doing. It is not
### necessary to use this.

# The below code is a partcular way to rollout episodes in a format
# compatible with the built-in state visualizer.
# ====== Policy wrapper compatible with AgentEvaluator/StateVisualizer ======
class StudentPolicy(NNPolicy):
    """
    Wraps the trained shared actor. state_policy must return a probability vector
    over 6 actions for the requested agent index.
    """
    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor.eval()  # inference mode

    def state_policy(self, state, agent_index):
        # base_env.featurize_state_mdp gives features for both; we only need the one for agent_index
        feats = base_env.featurize_state_mdp(state)[agent_index]
        x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(x).squeeze(0).cpu().numpy()
        # Convert logits to a valid probability simplex for the visualizer
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-8
        return probs

    def multi_state_policy(self, states, agent_indices):
        return [self.state_policy(s, i) for s, i in zip(states, agent_indices)]
def visualization_using_existing_flow(agent,ae, layout, use_norm=False, obsnorm=None):

    featurize_fn = ae.env.featurize_state_mdp

    # NOTE: your loader signature is (file, layout). Pass the checkpoint path explicitly.
    file = f"overcooked_{layout}_GUR_norm.pt" if use_norm else f"overcooked_{layout}_GUR.pt"
    actor_sd, critic_sd, loaded_norm = load_mappo_ckpt(file, layout)
    agent.actor.load_state_dict(actor_sd)
    agent.critic.load_state_dict(critic_sd)

    if use_norm and loaded_norm is not None:
        obsnorm = loaded_norm

    policy0 = MinimalRNNPolicy(agent.actor, featurize_fn, use_norm=use_norm, obsnorm=obsnorm, device=device)
    policy1 = MinimalRNNPolicy(agent.actor, featurize_fn, use_norm=use_norm, obsnorm=obsnorm, device=device)
    agent_pair = AgentPair(AgentFromPolicy(policy0), AgentFromPolicy(policy1))

    trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
    print("len(trajs):", len(trajs))
    img_dir = f"imgs/{layout}/"
    StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir, ipython_display=True)
# ====== Train MAPPO ======
def sweep_layout(layout):
    IS_CRAMPED = (layout == "cramped_room")
    IS_RING = (layout == "coordination_ring")
    IS_CIRCUIT = (layout == "counter_circuit_o_1order")
    # Build the environment.  Do not modify!
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    ckpt = None
    return env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt

Layouts=["cramped_room","coordination_ring","counter_circuit_o_1order"]
# Layouts=["counter_circuit_o_1order"]
results = {}
for layout in Layouts:
    print(f"layout is {layout}")
    env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt =sweep_layout(layout)
    agent, rewards_log, soups_log = train_mappo(env, updates=2000, rollout_steps=2048, shaping_scale=1.0)
    if ckpt:
        agent.actor.load_state_dict(ckpt["actor"]); agent.critic.load_state_dict(ckpt["critic"])
    results[layout.split('_')[0]] = {"reward": rewards_log, "soups": soups_log}
# Plot both metrics
plot_training_metrics(results, metric="reward",
                      save_dir="plots", filename="mappo_reward_GRU")
plot_training_metrics(results, metric="soups",
                      save_dir="plots", filename="mappo_soups_GRU")

results = {}
obsnorm_by_layout = {}
for layout in Layouts:
    print(f"layout is {layout}")
    env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt =sweep_layout(layout)
    norm = get_layout_norm(layout, env, obsnorm_by_layout)
    agent, rewards_log, soups_log = train_mappo_norm(env, norm, updates=2000, rollout_steps=2048, shaping_scale=1.0)
    if ckpt:
        agent.actor.load_state_dict(ckpt["actor"]); agent.critic.load_state_dict(ckpt["critic"])
    results[layout.split('_')[0]] = {"reward": rewards_log, "soups": soups_log}

# Plot both metrics
plot_training_metrics(results, metric="reward",
                      save_dir="plots", filename="mappo_reward_norm_GRU")
plot_training_metrics(results, metric="soups",
                      save_dir="plots", filename="mappo_soups_norm_GRU")
# layout="counter_circuit_o_1order"
# mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
# base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
# env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
# # agent, rewards_log, soups_log = train_mappo(env, updates=3, rollout_steps=2048, shaping_scale=1.0)
# # Create an empty agent structure (so it has the same architecture)
# obs = env.reset()
# o0, o1 = get_obs_pair(obs)
# obs_dim = o0.shape[0] + 2    # +2 if you used augment_obs
# act_dim = env.action_space.n
# agent = PPOMulti(obs_dim, act_dim, PPOCfg())
#
# # Load checkpoint weights and normalization statistics
# file=f"overcooked_{layout}_norm.pt"
# actor_sd, critic_sd, obsnorm = load_mappo_ckpt(file)
# agent.actor.load_state_dict(actor_sd)
# agent.critic.load_state_dict(critic_sd)
# visualization_using_existing_flow(agent, layout="counter_circuit_o_1order", use_norm=True, obsnorm=obsnorm)