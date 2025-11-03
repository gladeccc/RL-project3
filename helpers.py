import matplotlib.pyplot as plt
import gym
import torch.nn as nn
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from MAPPO import *
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
    if upd_idx <= 100 and np.random.rand() < 0.50:
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
    blocked = [False, False]
    if a0 == 'INTERACT' and not likely_legal_interact(obs0):
        a0 = np.random.choice([1,2,3,4])
        blocked[0] = True   # NSEW
    if a1 == 'INTERACT' and not likely_legal_interact(obs1):
        a1 = np.random.choice([1,2,3,4])
        blocked[1] = True
    if blocked[0] or blocked[1]:
        print("interact blocked:", blocked)
    return a0, a1

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

def get_layout_norm(layout_name: str, env, pool: dict):
    if layout_name not in pool:
        obs = env.reset()
        o0, _ = get_obs_pair(obs)
        pool[layout_name] = ObsNorm(dim=o0.shape[0])  # your ObsNorm class
    return pool[layout_name]

def augment_obs(obs_vec, agent_idx, use_norm=False, obsnorm=None):
    agent_id = np.array([1, 0], dtype=np.float32) if agent_idx == 0 else np.array([0, 1], dtype=np.float32)
    return np.concatenate([obs_vec, agent_id], axis=-1)

def compute_adv_ret_from_time_steps(rews, vals, dones, gamma=0.99, lam=0.95):
    rews  = np.asarray(rews,  dtype=np.float32)
    vals  = np.asarray(vals,  dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T = len(rews)
    next_vals = np.concatenate([vals[1:], np.array([0.0], np.float32)])
    next_mask = 1.0 - dones
    deltas = rews + gamma * next_vals * next_mask - vals
    adv = np.zeros_like(rews, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
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
    adv_b = np.concatenate([adv, adv], axis=0).astype(np.float32)  # [2T]
    ret_b = np.concatenate([ret, ret], axis=0).astype(np.float32)  # [2T]
    joint = np.vstack([np.asarray(step_joint, dtype=np.float32),
                       np.asarray(step_joint, dtype=np.float32)])  # [2T, 2*obs_dim]
    return {
        "obs": obs, "act": act, "logp": logp,
        "adv": adv_b, "ret": ret_b, "joint_obs": joint
    }

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