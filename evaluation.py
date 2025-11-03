from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import gym, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from MAPPO import *
from helpers import *

# --- utility: get the device of a module safely ---
def module_device(module: torch.nn.Module):
    """Return the device where a module's parameters live."""
    return next(module.parameters()).device

def load_trained_model(layout, Norm=False):
    """
    Load the trained Overcooked model with augmented input dimension = 98.
    Reads actor, critic, and (optionally) ObsNorm statistics from the checkpoint.
    """
    # Build environment
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    # Load checkpoint
    ckpt_path = f"overcooked_{layout}_norm.pt" if Norm else f"overcooked_{layout}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild ObsNorm if needed (constructor takes dimension)
    obsnorm = None
    if Norm:
        obsnorm = ObsNorm(98)
        obsnorm.m = ckpt["obsnorm_m"]
        obsnorm.s = ckpt["obsnorm_s"]
        obsnorm.n = ckpt["obsnorm_n"]

    # Build models with augmented dimension
    obs_dim = 98                   # fixed augmented dim
    act_dim = env.action_space.n   # discrete action count

    actor = Actor(obs_dim, act_dim).to(device)
    critic = CentralCritic(2 * obs_dim).to(device)

    # Load weights strictly (must match keys and shapes)
    actor.load_state_dict(ckpt["actor"], strict=True)
    critic.load_state_dict(ckpt["critic"], strict=True)

    actor.eval(); critic.eval()
    print(f"Loaded {layout} model with obs_dim={obs_dim}, Norm={Norm}")
    return env, actor, obsnorm

def get_obs_pair(obs_dict):
    """Return the two agents' observations from env output."""
    return obs_dict["both_agent_obs"][0], obs_dict["both_agent_obs"][1]

def _eval_one_layout(env, actor, obsnorm, episodes=100):
    """Evaluate a single layout and return per-episode returns."""
    returns = []
    dev = module_device(actor)
    actor.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            o0, o1 = obs["both_agent_obs"]
            done, ep_ret = False, 0.0

            while not done:
                # Optional normalization if provided
                if obsnorm is not None:
                    o0 = obsnorm.apply(o0)
                    o1 = obsnorm.apply(o1)

                # Use the same augmentation pipeline as training
                o0_aug = augment_obs(o0, agent_idx=0)
                o1_aug = augment_obs(o1, agent_idx=1)

                x0 = torch.as_tensor(o0_aug[None, :], dtype=torch.float32, device=dev)
                x1 = torch.as_tensor(o1_aug[None, :], dtype=torch.float32, device=dev)

                # Stochastic evaluation to mirror training behavior
                a0 = int(Categorical(logits=actor(x0)).sample().item())
                a1 = int(Categorical(logits=actor(x1)).sample().item())

                # Optional hygiene to keep actions legal/consistent
                a0, a1 = mask_interact(o0, o1, a0, a1)

                obs, R, done, info = env.step([a0, a1])
                ep_ret += float(R)
                o0, o1 = get_obs_pair(obs)

            returns.append(ep_ret)
    return returns

def evaluate_across_layouts(layouts, episodes=100, smooth_window=10, Norm=False,
                            save_path=None, show=False):
    """
    Evaluate the loaded model on multiple layouts and plot them in one figure.
    Returns: dict {layout: list_of_returns}
    """
    results = {}

    for layout in layouts:
        env, actor, obsnorm = load_trained_model(layout, Norm=Norm)
        rets = _eval_one_layout(env, actor, obsnorm, episodes=episodes)
        results[layout] = rets
        try:
            env.close()
        except Exception:
            pass

    # Plot all layouts in one figure
    plt.figure(figsize=(10, 5.5))
    for layout, rets in results.items():
        x = np.arange(1, len(rets) + 1)
        plt.plot(x, rets, linewidth=1, alpha=0.35, label=f"{layout} (raw)")
        if smooth_window and smooth_window > 1 and len(rets) >= smooth_window:
            kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
            sma = np.convolve(np.asarray(rets, dtype=np.float32), kernel, mode="valid")
            sma_x = np.arange(smooth_window, len(rets) + 1)
            plt.plot(sma_x, sma, linewidth=2, label=f"{layout} SMA({smooth_window})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Evaluation Returns Across Layouts")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return results

if __name__ == "__main__":
    layouts = [ "coordination_ring", "counter_circuit_o_1order"]
    results = evaluate_across_layouts(
        layouts=layouts,
        episodes=100,
        smooth_window=1,
        Norm=False,
        save_path="eval_multi_layouts.png",
        show=False
    )
