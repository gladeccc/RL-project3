from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt


reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5
}
horizon = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlp(sizes, act=nn.Tanh):
    layers=[]
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            layers.append(act())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = mlp([obs_dim, 128, 128])
        self.logits = nn.Linear(128, act_dim)
    def forward(self, x):
        return self.logits(self.body(x))

class CentralCritic(nn.Module):
    def __init__(self, joint_dim):
        super().__init__()
        self.v = mlp([joint_dim, 256, 256, 1])
    def forward(self, joint_obs):
        return self.v(joint_obs).squeeze(-1)
def augment_obs(obs_vec, agent_idx, use_norm=False, obsnorm=None):
    agent_id = np.array([1, 0], dtype=np.float32) if agent_idx == 0 else np.array([0, 1], dtype=np.float32)
    return np.concatenate([obs_vec, agent_id], axis=-1)
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
INTERACT=5
def mask_interact(obs0: np.ndarray, obs1: np.ndarray, a0: int, a1: int):
    """If Interact is unlikely, replace with a random move."""
    blocked = [False, False]
    if a0 == INTERACT and not likely_legal_interact(obs0):
        a0 = np.random.choice([1,2,3,4]);blocked[0] = True   # NSEW
    if a1 == INTERACT and not likely_legal_interact(obs1):
        a1 = np.random.choice([1,2,3,4]);blocked[0] = True
    if any(blocked): print("interact blocked:", blocked)
    return a0, a1

def load_trained_model(layout):
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    obs = env.reset()
    obs0, obs1 = obs["both_agent_obs"]
    obs_dim = obs0.shape[0]
    act_dim = env.action_space.n

    # Instantiate models
    actor = Actor(obs_dim, act_dim).to(device)
    critic = CentralCritic(2*obs_dim).to(device)

    # Load checkpoint
    ckpt = torch.load(f"overcooked_{layout}.pt", map_location=device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    actor.eval()
    critic.eval()
    return env, actor

    print(f"Loaded checkpoint {layout} successfully.")

def count_delivery(R, info) -> int:
    # Prefer explicit counters; else detect +20 spike robustly
    if isinstance(info, dict) and "num_delivered" in info:
        return int(info["num_delivered"])
    return int(float(R) >= 10.0)
def get_obs_pair(obs_dict):
    # Your env returns dict; features are already aligned for obs order
    return obs_dict["both_agent_obs"][0], obs_dict["both_agent_obs"][1]

def evaluate(agent, env, episodes=100, smooth_window=10, Norm=False, save_path=None, show=True):
    total_returns, total_soups = [], 0
    for ep in range(episodes):
        obs = env.reset()
        o0, o1 = obs["both_agent_obs"]
        done, ep_ret, ep_soups = False, 0.0, 0
        while not done:
            o0_aug = augment_obs(o0, agent_idx=0)
            o1_aug = augment_obs(o1, agent_idx=1)
            x0 = torch.as_tensor(o0_aug[None, :], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1_aug[None, :], dtype=torch.float32, device=device)
            # Stochastic eval to match your agent.act() behavior
            a0 = int(agent.dist(agent.actor(x0)).sample().item())
            a1 = int(agent.dist(agent.actor(x1)).sample().item())

            # Optional hygiene consistent with training
            a0, a1 = mask_interact(o0, o1, a0, a1)

            obs, R, done, info = env.step([a0, a1])
            ep_ret += float(R)
            ep_soups += count_delivery(float(R), info)
            o0, o1 = get_obs_pair(obs)
        total_returns.append(ep_ret)
        total_soups += ep_soups
        print(f"Episode {ep+1}: return={ep_ret:.1f}, soups={ep_soups}")
    mean_ret = np.mean(total_returns)
    mean_soups = total_soups / episodes
    print(f"\nAverage return: {mean_ret:.1f}, average soups/episode: {mean_soups:.2f}")
    # Compute simple moving average (SMA) for smoothing
    if smooth_window is None or smooth_window < 1:
        smooth_window = 1
    if smooth_window > 1 and len(total_returns) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        sma = np.convolve(np.array(total_returns, dtype=np.float32), kernel, mode="valid")
        sma_x = np.arange(smooth_window - 1, len(total_returns))
    else:
        sma = None
        sma_x = None

    # Plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(total_returns) + 1), total_returns, label="Episode Return")
    if sma is not None:
        plt.plot(sma_x + 1, sma, linewidth=2, label=f"SMA({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Per-episode Return")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return total_returns

def _is_adjacent(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1

def _get_static_stations(mdp):
    # Pull station coordinates from the MDP (layout graph)
    # These helpers exist on OvercookedGridworld
    onions = set(mdp.get_onion_dispenser_locations())
    dishes = set(mdp.get_dish_dispenser_locations())
    servs  = set(mdp.get_serving_locations())
    pots   = set([p[0] if isinstance(p, tuple) else p for p in mdp.get_pot_locations()])
    return onions, dishes, servs, pots

def _held_symbol(player_obj):
    # player_obj: env.unwrapped.state.players[i]
    try:
        h = player_obj.held_object
        if h is None:         return "."
        t = getattr(h, "name", None) or getattr(h, "state", None) or str(h)
        t = str(t).lower()
        if "onion" in t:      return "o"
        if "dish" in t and "soup" not in t: return "d"
        if "soup" in t:       return "S"    # plated soup or soup object
        return "?"
    except Exception:
        return "."

def _pot_status(base_env):
    # Try to summarize pot states. Works with overcooked_ai >= 1.2 style states.
    try:
        state = base_env.state
        pots = base_env.mdp.get_pots_pos_and_states(state)
        out = []
        for pos, pst in pots.items():
            # pst keys usually: 'cook_time', 'num_onions', 'has_water', 'ready'
            num = int(pst.get("num_onions", 0))
            ready = bool(pst.get("ready", False))
            ct = pst.get("cook_time", None)
            if ready:
                tag = f"{pos}:READY"
            elif ct is not None and ct > 0:
                tag = f"{pos}:cook{ct:02d}/{base_env.mdp.start_cook_time}"
            else:
                tag = f"{pos}:onions{num}"
            out.append(tag)
        return "|".join(out) if out else "-"
    except Exception:
        return "-"  # graceful fallback if API differs

def trace_episode(env, actor, steps=400, mask_interact=False):
    """
    Run one episode and print a compact timeline.
    - Shows agent positions, adjacency to [O]nion [P]ot [D]ish [V]Serve,
      held items, pot status, and events (3rd onion, cook start, ready, serve).
    - If mask_interact=True, applies your interact hygiene during rollout.
    """
    base_env = env.unwrapped.base_env
    mdp = base_env.mdp
    onions, dishes, servs, pots = _get_static_stations(mdp)

    obs = env.reset()
    o0, o1 = obs["both_agent_obs"]
    done = False

    deliveries = 0
    prev_num_onions = 0
    cooking_active = False
    served_steps = []

    print("\nT |  A(x,y)  aAdj h |  B(x,y)  bAdj h |   Pot(s)                | Events")
    print("---+-------------------+-------------------+------------------------+-----------------")

    for t in range(steps):
        # Positions and holds from true env state for correctness
        st = base_env.state
        A_pos = tuple(st.players_pos[0])
        B_pos = tuple(st.players_pos[1])
        A_hold = _held_symbol(st.players[0])
        B_hold = _held_symbol(st.players[1])

        # Adjacency flags
        def adj_flags(pos):
            aO = any(_is_adjacent(pos, p) for p in onions)
            aP = any(_is_adjacent(pos, p) for p in pots)
            aD = any(_is_adjacent(pos, p) for p in dishes)
            aV = any(_is_adjacent(pos, p) for p in servs)  # V for deliVery slot
            s = "".join([c if f else "." for c, f in zip("OPDV", [aO, aP, aD, aV])])
            return s

        A_adj = adj_flags(A_pos)
        B_adj = adj_flags(B_pos)

        # Sample actions from actor
        with torch.no_grad():
            x0 = torch.as_tensor(o0[None, :], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1[None, :], dtype=torch.float32, device=device)
            logits0 = actor(x0)
            logits1 = actor(x1)

            # Optional: discourage 'stay' on narrow maps while tracing
            # STAY = 0
            # logits0[..., STAY] -= 0.5; logits1[..., STAY] -= 0.5

            d0 = Categorical(logits=logits0)
            d1 = Categorical(logits=logits1)
            a0 = int(d0.sample().item())
            a1 = int(d1.sample().item())

        # Optional interact hygiene
        if mask_interact:
            def _likely_legal_interact(feats):
                v = feats.astype(np.int32)
                for i in range(0, len(v)-1):
                    dx, dy = v[i], v[i+1]
                    if (abs(dx) == 1 and dy == 0) or (abs(dy) == 1 and dx == 0):
                        return True
                return False
            INTERACT = 5
            if a0 == INTERACT and not _likely_legal_interact(o0): a0 = np.random.choice([1,2,3,4])
            if a1 == INTERACT and not _likely_legal_interact(o1): a1 = np.random.choice([1,2,3,4])

        # Step env
        obs, R, done, info = env.step([a0, a1])
        o0, o1 = obs["both_agent_obs"]

        # Pot/serve events
        pots_str = _pot_status(base_env)
        ev = []
        # Detect onion count rising to 3 on any pot
        try:
            # parse num_onions from pots_str best-effort
            if "onions" in pots_str:
                # crude scan: ...:onionsN
                import re
                counts = [int(m.group(1)) for m in re.finditer(r"onions(\d+)", pots_str)]
                if counts:
                    cur = max(counts)
                    if cur != prev_num_onions:
                        if cur == 3:
                            ev.append("3rd-ONION")
                        prev_num_onions = cur
        except Exception:
            pass

        # Detect cook start/ready via pot status
        if "cook" in pots_str and not cooking_active:
            cooking_active = True
            ev.append("COOK-START")
        if "READY" in pots_str:
            ev.append("READY")

        # Detect delivery by reward spike
        if float(R) >= 10.0:
            deliveries += 1
            served_steps.append(t)
            ev.append(f"SERVE(+20) total={deliveries}")

        # Print timeline row
        print(f"{t:02d}| A{A_pos!s:>7} {A_adj} {A_hold} | B{B_pos!s:>7} {B_adj} {B_hold} | {pots_str:<22} | {';'.join(ev)}")

        if done:
            break

    print("\nEpisode done. Deliveries:", deliveries, "| Served at steps:", served_steps)

if __name__ == "__main__":
    layout = "cramped_room"
    #layout = "coordination_ring"
    #layout = "counter_circuit_o_1order"
    env, actor = load_trained_model(layout)
    evaluate(actor, env, episodes=20)
    # trace_episode(env, actor)