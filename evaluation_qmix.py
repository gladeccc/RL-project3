# ====== QMIX EVALUATION (load .pt and roll out) ======
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ----- config -----
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
}
horizon = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- tiny helpers -----
def get_obs_pair(obs_dict):
    return obs_dict["both_agent_obs"][0], obs_dict["both_agent_obs"][1]

def count_delivery(R, info):
    # Robust: +20 for soup, but tolerate variants
    return int(float(R) >= 10.0)

# Optional hygiene at eval time only
INTERACT = 5
def likely_legal_interact(feats: np.ndarray) -> bool:
    v = feats.astype(np.int32)
    # best-effort: look for any adjacent (±1,0) or (0,±1) deltas in featurization
    for i in range(0, len(v) - 1):
        dx, dy = v[i], v[i+1]
        if (abs(dx) == 1 and dy == 0) or (abs(dy) == 1 and dx == 0):
            return True
    return False

def mask_interact_eval(o0, o1, a0, a1):
    if a0 == INTERACT and not likely_legal_interact(o0): a0 = np.random.choice([1,2,3,4])
    if a1 == INTERACT and not likely_legal_interact(o1): a1 = np.random.choice([1,2,3,4])
    return int(a0), int(a1)

# ----- QMIX nets (must match what you trained) -----
class AgentQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, act_dim)
        )
    def forward(self, obs):  # (B, obs_dim)
        return self.net(obs) # (B, act_dim)

class MonotonicMixer(nn.Module):
    # Q_tot = sum_i softplus(w_i(s))*Q_i + b(s)
    def __init__(self, state_dim, n_agents=2, hidden=64):
        super().__init__()
        self.n = n_agents
        self.w1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n))
        self.b1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, q_agents, state):
        w = F.softplus(self.w1(state))    # (B,n), nonnegative
        b = self.b1(state)                # (B,1)
        return (w * q_agents).sum(dim=1, keepdim=True).add(b).squeeze(-1)

# ----- loader -----
def load_qmix_for_eval(layout: str, ckpt_path: str):
    # build env
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    # probe dims
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]
    act_dim = env.action_space.n

    # build nets
    q = AgentQ(obs_dim, act_dim).to(device).eval()
    mixer = MonotonicMixer(state_dim=2*obs_dim).to(device).eval()

    # load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    q.load_state_dict(ckpt["q"])
    mixer.load_state_dict(ckpt["mixer"])

    return env, q, mixer

# ----- evaluation -----
@torch.no_grad()
def evaluate_qmix_checkpoint(layout: str,
                             ckpt_path: str,
                             episodes: int = 30,
                             use_interact_mask: bool = True,
                             greedy: bool = True):
    env, q, _mixer = load_qmix_for_eval(layout, ckpt_path)

    total_returns = []
    total_soups = 0

    for ep in range(episodes):
        obs = env.reset()
        o0, o1 = get_obs_pair(obs)
        done = False
        ep_ret = 0.0
        ep_soups = 0

        while not done:
            # greedy or ε=0 equivalent
            q0 = q(torch.as_tensor(o0[None, :], dtype=torch.float32, device=device))  # (1,A)
            q1 = q(torch.as_tensor(o1[None, :], dtype=torch.float32, device=device))
            a0 = int(q0.argmax(dim=1).item())
            a1 = int(q1.argmax(dim=1).item())

            if use_interact_mask:
                a0, a1 = mask_interact_eval(o0, o1, a0, a1)

            obs, R, done, info = env.step([a0, a1])
            ep_ret += float(R)
            ep_soups += count_delivery(R, info)
            o0, o1 = get_obs_pair(obs)

        total_returns.append(ep_ret)
        total_soups += ep_soups
        print(f"Episode {ep+1:02d}: return={ep_ret:.1f}, soups={ep_soups}")

    mean_ret = float(np.mean(total_returns)) if total_returns else 0.0
    mean_soups = total_soups / float(episodes)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Layout: {layout}")
    print(f"Average return: {mean_ret:.1f} | Average soups/episode: {mean_soups:.2f}")

    return {"mean_return": mean_ret, "mean_soups": mean_soups}

if __name__ == "__main__":
    layout = "cramped_room"            # or "coordination_ring", "counter_circuit_o_1order"
    ckpt = f"overcooked_{layout}_qmix.pt"

    metrics = evaluate_qmix_checkpoint(layout, ckpt, episodes=30, use_interact_mask=True)
    print(metrics)