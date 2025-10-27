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
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5
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

class Actor(nn.Module):
    # Produces logits over 6 discrete actions for ONE agent
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = mlp([obs_dim, 128, 128])
        self.logits = nn.Linear(128, act_dim)
    def forward(self, x):
        return self.logits(self.body(x))

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
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_env_steps: int = 2048  # env steps/update
    minibatch_size: int = 256
    opt_iters: int = 4
    total_updates: int = 1500

    shaping_scale: float = 1.0     # mild nudge only

# ====== Centralized-critic PPO trainer that fits your env ======
class PPOMulti:
    def __init__(self, obs_dim, act_dim, cfg: PPOCfg):
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = CentralCritic(2*obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr
        )

    def dist(self, logits):
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs_np):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        d = self.dist(self.actor(x))
        a = d.sample()
        return a.cpu().numpy(), d.log_prob(a).cpu().numpy()

    @torch.no_grad()
    def value(self, joint_np):
        j = torch.as_tensor(joint_np, dtype=torch.float32, device=device)
        return self.critic(j).cpu().numpy()

    def update(self, batch):
        cfg = self.cfg
        obs  = torch.tensor(batch["obs"],  dtype=torch.float32, device=device)
        act  = torch.tensor(batch["act"],  dtype=torch.int64,   device=device)
        adv  = torch.tensor(batch["adv"],  dtype=torch.float32, device=device)
        ret  = torch.tensor(batch["ret"],  dtype=torch.float32, device=device)
        lpo  = torch.tensor(batch["logp"], dtype=torch.float32, device=device)
        jobs = torch.tensor(batch["joint_obs"], dtype=torch.float32, device=device)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.opt_iters):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch_size):
                mb = idx[s:s+cfg.minibatch_size]
                d = self.dist(self.actor(obs[mb]))
                logp = d.log_prob(act[mb])
                ratio = torch.exp(logp - lpo[mb])
                clipped = torch.clamp(ratio, 1-cfg.clip, 1+cfg.clip) * adv[mb]
                pi_loss = -(torch.min(ratio*adv[mb], clipped).mean() + cfg.ent_coef*d.entropy().mean())

                v = self.critic(jobs[mb])
                v_loss = cfg.vf_coef * F.mse_loss(v, ret[mb])

                self.opt.zero_grad()
                (pi_loss + v_loss).backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm
                )
                self.opt.step()

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



def train_mappo(env, updates=2000, rollout_steps=2048, shaping_scale=1.0):
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

        while steps < agent.cfg.rollout_env_steps:
            # update stats BEFORE using
            # obsnorm.update(o0); obsnorm.update(o1)
            # o0n = obsnorm.apply(o0)
            # o1n = obsnorm.apply(o1)

            # sample actions from NORMALIZED inputs
            # x0 = torch.as_tensor(o0n[None,:], dtype=torch.float32, device=device)
            # x1 = torch.as_tensor(o1n[None,:], dtype=torch.float32, device=device)
            x0 = torch.as_tensor(o0[None, :], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                dist0 = agent.dist(agent.actor(x0))
                dist1 = agent.dist(agent.actor(x1))
                a0 = int(dist0.sample().item())
                a1 = int(dist1.sample().item())

            if upd <= 300 and np.random.rand() < 0.7:
                a0_exec, a1_exec = mask_interact(o0, o1, a0, a1)
                a0_exec, a1_exec = maybe_bias_actions(steps, upd, a0_exec, a1_exec, o0, o1)
            else:
                a0_exec, a1_exec = a0, a1

            # recompute logp for executed actions from the same dists
            with torch.no_grad():
                lp0 = float(dist0.log_prob(torch.tensor(a0_exec, device=device)).cpu().item())
                lp1 = float(dist1.log_prob(torch.tensor(a1_exec, device=device)).cpu().item())

            # critic sees NORMALIZED joint
            # joint = np.concatenate([o0n, o1n], axis=-1)
            joint = np.concatenate([o0, o1], axis=-1)
            v = agent.value(joint[None,:])[0]

            # step env
            obs, R, done, info = env.step([a0_exec, a1_exec])

            # sparse + mild shaped reward
            shape_scale = 8.0 if upd <= 200 else agent.cfg.shaping_scale
            r = float(R) + shaped_team_reward(info, env, scale=shape_scale)

            # store NORMALIZED obs so training matches sampling distribution
            # for ob_n, ac, lp in [(o0n, a0_exec, lp0), (o1n, a1_exec, lp1)]:
            for ob, ac, lp in [(o0, a0_exec, lp0), (o1, a1_exec, lp1)]:
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
                obs = env.reset()
                o0, o1 = get_obs_pair(obs)

        # shaped-hit print (unchanged)
        shaped_hits = sum(1 for rr in buf["rew"] if (rr != 0.0 and rr < 20.0))
        print(f"[upd {upd}] shaped_hit_rate={shaped_hits / len(buf['rew']):.3f}")

        for k in buf: buf[k] = np.asarray(buf[k], dtype=np.float32)

        # === GAE ===
        g, l = agent.cfg.gamma, agent.cfg.lam
        rews, vals, dones = buf["rew"], buf["val"], buf["done"]
        T = len(rews)
        next_vals = np.concatenate([vals[1:], np.array([0.0], dtype=np.float32)])
        next_mask = 1.0 - dones
        deltas = rews + g*next_vals*next_mask - vals

        adv = np.zeros_like(rews); gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + g*l*next_mask[t]*gae
            adv[t] = gae
        ret = adv + vals
        adv = (adv - adv.mean())/(adv.std()+1e-8)

        batch = {
            "obs": buf["obs"],
            "act": buf["act"].astype(np.int64),
            "logp": buf["logp"],
            "adv": adv,
            "ret": ret,
            "joint_obs": buf["joint_obs"]
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
                    # "obsnorm_m": obsnorm.m,
                    # "obsnorm_s": obsnorm.s,
                    # "obsnorm_n": obsnorm.n
                }, f"overcooked_{layout}.pt")
                print(f"Checkpoint saved to overcooked_{layout}.pt")

    print(f"Best soups/ep observed: {best_soups:.2f}")
    return agent, rewards_log, soups_log

def train_mappo_norm(env, updates=2000, rollout_steps=2048, shaping_scale=1.0):
    # Probe dims
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]; act_dim = env.action_space.n

    # # Running normalizer
    obsnorm = ObsNorm(obs_dim)

    agent = PPOMulti(obs_dim, act_dim, PPOCfg(
        total_updates=updates, rollout_env_steps=rollout_steps, shaping_scale=shaping_scale
    ))

    best_soups = -1.0
    rewards_log, soups_log = [], []
    for upd in range(1, agent.cfg.total_updates + 1):
        # # simple entropy schedule (optional but helps)
        if upd <= int(0.3 * agent.cfg.total_updates):
            agent.cfg.ent_coef = 0.03
        elif upd <= int(0.7 * agent.cfg.total_updates):
            agent.cfg.ent_coef = 0.015
        else:
            agent.cfg.ent_coef = 0.007

        buf = {k: [] for k in ["obs","act","logp","rew","done","val","joint_obs"]}
        steps = 0

        while steps < agent.cfg.rollout_env_steps:
            # update stats BEFORE using
            obsnorm.update(o0); obsnorm.update(o1)
            o0n = obsnorm.apply(o0)
            o1n = obsnorm.apply(o1)

            # sample actions from NORMALIZED inputs
            x0 = torch.as_tensor(o0n[None,:], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1n[None,:], dtype=torch.float32, device=device)
            # x0 = torch.as_tensor(o0[None, :], dtype=torch.float32, device=device)
            # x1 = torch.as_tensor(o1[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                dist0 = agent.dist(agent.actor(x0))
                dist1 = agent.dist(agent.actor(x1))
                a0 = int(dist0.sample().item())
                a1 = int(dist1.sample().item())

            if upd <= 300 and np.random.rand() < 0.7:
                a0_exec, a1_exec = mask_interact(o0, o1, a0, a1)
                a0_exec, a1_exec = maybe_bias_actions(steps, upd, a0_exec, a1_exec, o0, o1)
            else:
                a0_exec, a1_exec = a0, a1

            # recompute logp for executed actions from the same dists
            with torch.no_grad():
                lp0 = float(dist0.log_prob(torch.tensor(a0_exec, device=device)).cpu().item())
                lp1 = float(dist1.log_prob(torch.tensor(a1_exec, device=device)).cpu().item())

            # critic sees NORMALIZED joint
            joint = np.concatenate([o0n, o1n], axis=-1)
            # joint = np.concatenate([o0, o1], axis=-1)
            v = agent.value(joint[None,:])[0]

            # step env
            obs, R, done, info = env.step([a0_exec, a1_exec])

            # sparse + mild shaped reward
            shape_scale = 8.0 if upd <= 200 else agent.cfg.shaping_scale
            r = float(R) + shaped_team_reward(info, env, scale=shape_scale)

            # store NORMALIZED obs so training matches sampling distribution
            for ob_n, ac, lp in [(o0n, a0_exec, lp0), (o1n, a1_exec, lp1)]:
            # for ob, ac, lp in [(o0, a0_exec, lp0), (o1, a1_exec, lp1)]:
                buf["obs"].append(ob_n)
                # buf["obs"].append(ob)
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
                obs = env.reset()
                o0, o1 = get_obs_pair(obs)

        # shaped-hit print (unchanged)
        shaped_hits = sum(1 for rr in buf["rew"] if (rr != 0.0 and rr < 20.0))
        print(f"[upd {upd}] shaped_hit_rate={shaped_hits / len(buf['rew']):.3f}")

        for k in buf: buf[k] = np.asarray(buf[k], dtype=np.float32)

        # === GAE ===
        g, l = agent.cfg.gamma, agent.cfg.lam
        rews, vals, dones = buf["rew"], buf["val"], buf["done"]
        T = len(rews)
        next_vals = np.concatenate([vals[1:], np.array([0.0], dtype=np.float32)])
        next_mask = 1.0 - dones
        deltas = rews + g*next_vals*next_mask - vals

        adv = np.zeros_like(rews); gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + g*l*next_mask[t]*gae
            adv[t] = gae
        ret = adv + vals
        adv = (adv - adv.mean())/(adv.std()+1e-8)

        batch = {
            "obs": buf["obs"],
            "act": buf["act"].astype(np.int64),
            "logp": buf["logp"],
            "adv": adv,
            "ret": ret,
            "joint_obs": buf["joint_obs"]
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
                    "obsnorm_m": obsnorm.m,
                    "obsnorm_s": obsnorm.s,
                    "obsnorm_n": obsnorm.n
                }, f"overcooked_{layout}.pt")
                print(f"Checkpoint saved to overcooked_{layout}.pt")

    print(f"Best soups/ep observed: {best_soups:.2f}")
    return agent, rewards_log, soups_log

@torch.no_grad()
def eval_soups(agent, env, episodes=20):
    rets, soups = [], 0
    for _ in range(episodes):
        obs = env.reset()
        o0, o1 = get_obs_pair(obs)
        done, ep_ret = False, 0.0
        while not done:
            a0, _ = agent.act(o0[None, :]);
            a1, _ = agent.act(o1[None, :])
            obs, R, done, info = env.step([int(a0[0]), int(a1[0])])
            ep_ret += float(R)
            soups += count_delivery(float(R), info)
            o0, o1 = get_obs_pair(obs)
        rets.append(ep_ret)
    return float(np.mean(rets)), soups / float(episodes)

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

    if IS_CIRCUIT:
        ckpt = torch.load("overcooked_counter_circuit_o_1order_MAPPO.pt", map_location="cpu")
        print("overcooked_counter_circuit_o_1order.pt loaded")
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        print("Loaded weights from cramped_room.")
    return env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt

Layouts=["cramped_room","coordination_ring","counter_circuit_o_1order"]
results = {}
for layout in Layouts:
    print(f"layout is {layout}")
    env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt =sweep_layout(layout)
    agent, rewards_log, soups_log = train_mappo(env, updates=1200, rollout_steps=1024, shaping_scale=1.0)
    if ckpt:
        agent.actor.load_state_dict(ckpt["actor"]); agent.critic.load_state_dict(ckpt["critic"])
    results[layout.split('_')[0]] = {"reward": rewards_log, "soups": soups_log}
# Plot both metrics
plot_training_metrics(results, metric="reward",
                      save_dir="plots", filename="mappo_reward")
plot_training_metrics(results, metric="soups",
                      save_dir="plots", filename="mappo_soups")

results = {}
for layout in Layouts:
    print(f"layout is {layout}")
    env, base_env, IS_CIRCUIT, IS_CRAMPED,IS_RING, ckpt =sweep_layout(layout)
    agent, rewards_log, soups_log = train_mappo_norm(env, updates=1200, rollout_steps=1024, shaping_scale=1.0)
    if ckpt:
        agent.actor.load_state_dict(ckpt["actor"]); agent.critic.load_state_dict(ckpt["critic"])
    results[layout.split('_')[0]] = {"reward": rewards_log, "soups": soups_log}
# Plot both metrics
plot_training_metrics(results, metric="reward",
                      save_dir="plots", filename="mappo_reward_norm")
plot_training_metrics(results, metric="soups",
                      save_dir="plots", filename="mappo_soups_norm")

# ====== Visualization using the existing flow ======
# from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
# from overcooked_ai_py.agents.benchmarking import AgentEvaluator
# from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
#
# policy0 = StudentPolicy(agent.actor)
# policy1 = StudentPolicy(agent.actor)
# agent0 = AgentFromPolicy(policy0)
# agent1 = AgentFromPolicy(policy1)
# agent_pair = AgentPair(agent0, agent1)
#
# ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
# trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
# print("len(trajs):", len(trajs))
#
# img_dir = "imgs/"
# ipython_display = True
# StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir, ipython_display=ipython_display)
