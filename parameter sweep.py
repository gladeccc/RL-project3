# =========================
# MAPPO hyperparameter tuning
# =========================
import copy
import itertools
import time
from MAPPO import *
from helpers import *

def clone_cfg(cfg: PPOCfg) -> PPOCfg:
    """Deep-copy PPO configuration."""
    return copy.deepcopy(cfg)

def apply_shift_to_cfg(cfg: PPOCfg, param: str, base_value, shift, mode: str):
    """Return value after applying multiplicative or additive shift."""
    if mode == "mul":
        return base_value * shift
    elif mode == "add":
        return base_value + shift
    else:
        raise ValueError(f"Unknown mode: {mode}")

def build_env_and_norm(layout: str):
    """Build env and per-layout normalizer using your existing helpers."""
    env, base_env, IS_CIRCUIT, IS_CRAMPED, IS_RING, ckpt = sweep_layout(layout)
    obsnorm = get_layout_norm(layout, env, pool={})
    return env, obsnorm

def run_training_for_cfg(layout: str,
                         cfg: PPOCfg,
                         norm: bool = True,
                         updates: int = 400,
                         rollout_steps: int = 1024,
                         shaping_scale: float = 1.0,
                         eval_episodes: int = 30):
    """
    Train for a short budget and return training logs.
    Uses your existing train_mappo / train_mappo_norm functions.
    """
    env, obsnorm = build_env_and_norm(layout)
    if norm:
        agent, rewards_log, soups_log = train_mappo_norm(
            env, obsnorm, updates=updates, rollout_steps=rollout_steps, shaping_scale=shaping_scale
        )
    else:
        agent, rewards_log, soups_log = train_mappo(
            env, updates=updates, rollout_steps=rollout_steps, shaping_scale=shaping_scale
        )

    # Optionally do a final eval using your eval functions (the logs already include periodic evals)
    if norm:
        final_ret, final_soups = eval_soups_norm(agent, env, obsnorm, episodes=eval_episodes)
    else:
        final_ret, final_soups = eval_soups(agent, env, episodes=eval_episodes)

    return {
        "rewards_log": rewards_log,
        "soups_log": soups_log,
        "final_ret": float(final_ret),
        "final_soups": float(final_soups),
    }

def tune_hyperparams(layout: str,
                     shifts: dict,
                     base_cfg: PPOCfg = None,
                     mode: str = "mul",
                     norm: bool = True,
                     updates: int = 400,
                     rollout_steps: int = 1024,
                     shaping_scale: float = 1.0,
                     eval_episodes: int = 30,
                     grid: bool = False,
                     save_dir: str = "plots",
                     tag: str = None):
    """
    Hyperparameter tuning/sensitivity.

    Args:
        layout: Overcooked layout name (e.g., "cramped_room").
        shifts: dict of {param_name: [shift_values...]}. Values are multiplicative factors if mode="mul",
                or additive deltas if mode="add".
                Example (mul): {"lr": [0.3, 1.0, 3.0], "clip": [0.5, 1.0, 1.5]}
                Example (add): {"lam": [-0.05, 0.0, 0.05]}
        base_cfg: starting PPOCfg. If None, use PPOCfg().
        mode: "mul" or "add" (how to interpret shift values).
        norm: whether to use normalized training path (train_mappo_norm).
        updates: training updates per run (keep small for sweep).
        rollout_steps: rollout steps per update.
        shaping_scale: shaped reward scale during training.
        eval_episodes: episodes for final evaluation.
        grid: if False, sweep one factor at a time; if True, run full cartesian product.
        save_dir: where to save plots.
        tag: optional run tag appended to filenames.

    Returns:
        results_dict for plotting, plus a summary list of runs with final metrics.
    """
    base = base_cfg if base_cfg is not None else PPOCfg()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = tag or f"{layout}_{mode}_{timestamp}"

    # Helper to set an attribute safely
    def with_param(cfg_in: PPOCfg, name: str, new_value):
        cfg_out = clone_cfg(cfg_in)
        if not hasattr(cfg_out, name):
            raise AttributeError(f"PPOCfg has no attribute '{name}'")
        setattr(cfg_out, name, type(getattr(cfg_out, name))(new_value))
        return cfg_out

    results_for_plot = {}  # key -> {'reward': list, 'soups': list}
    leaderboard = []       # list of dicts summarizing final metrics
    run_idx = 0

    if not grid:
        # One-factor-at-a-time sensitivity
        for param_name, shift_list in shifts.items():
            base_value = getattr(base, param_name)
            for shift in shift_list:
                run_idx += 1
                new_value = apply_shift_to_cfg(base, param_name, base_value, shift, mode)
                cfg_try = with_param(base, param_name, new_value)

                run_key = f"{param_name}={new_value:.6g}"
                print(f"[{run_idx}] {run_key} | updates={updates}, rollout_steps={rollout_steps}")

                out = run_training_for_cfg(
                    layout, cfg_try, norm=norm, updates=updates,
                    rollout_steps=rollout_steps, shaping_scale=shaping_scale,
                    eval_episodes=eval_episodes
                )

                # Store the whole curves for plotting
                results_for_plot[run_key] = {
                    "reward": out["rewards_log"],
                    "soups":  out["soups_log"],
                }
                leaderboard.append({
                    "key": run_key,
                    "param": param_name,
                    "value": float(new_value),
                    "final_ret": out["final_ret"],
                    "final_soups": out["final_soups"],
                })
    else:
        # Full factorial grid over all provided params
        names = list(shifts.keys())
        shift_lists = [shifts[n] for n in names]
        for combo in itertools.product(*shift_lists):
            cfg_try = clone_cfg(base)
            label_parts = []
            for name, shift in zip(names, combo):
                base_value = getattr(cfg_try, name)
                new_value = apply_shift_to_cfg(cfg_try, name, base_value, shift, mode)
                cfg_try = with_param(cfg_try, name, new_value)
                label_parts.append(f"{name}={new_value:.6g}")
            run_key = ",".join(label_parts)

            run_idx += 1
            print(f"[{run_idx}] {run_key} | updates={updates}, rollout_steps={rollout_steps}")

            out = run_training_for_cfg(
                layout, cfg_try, norm=norm, updates=updates,
                rollout_steps=rollout_steps, shaping_scale=shaping_scale,
                eval_episodes=eval_episodes
            )
            results_for_plot[run_key] = {
                "reward": out["rewards_log"],
                "soups":  out["soups_log"],
            }
            leaderboard.append({
                "key": run_key,
                "param": ",".join(names),
                "value": ",".join([f"{v}" for v in combo]),
                "final_ret": out["final_ret"],
                "final_soups": out["final_soups"],
            })

    # === Plot curves using your existing helper ===
    # reward curve
    plot_training_metrics(
        results_for_plot, metric="reward",
        save_dir=save_dir, filename=f"{tag}_reward"
    )
    # soups curve
    plot_training_metrics(
        results_for_plot, metric="soups",
        save_dir=save_dir, filename=f"{tag}_soups"
    )

    # Sort leaderboard by final soups then reward
    leaderboard.sort(key=lambda d: (d["final_soups"], d["final_ret"]), reverse=True)

    print("\n=== Leaderboard (top 10 by final soups, then reward) ===")
    for row in leaderboard[:10]:
        print(f"{row['key']:40s} | soups={row['final_soups']:.3f} | reward={row['final_ret']:.1f}")

    return results_for_plot, leaderboard

if __name__ == "__main__":
    shifts = {
        "lr": [0.3, 1.0, 3.0],
        "clip": [0.75, 1.0, 1.25],
        "opt_iters": [0.5, 1.0, 2.0],  # will be multiplied and cast back to int
    }
    results, board = tune_hyperparams(
        layout="cramped_room",
        shifts=shifts,
        base_cfg=PPOCfg(),  # or your tuned base
        mode="mul",
        norm=True,
        updates=400,  # keep small for sweeps
        rollout_steps=1024,
        shaping_scale=1.0,
        eval_episodes=30,
        grid=False,  # one-factor sweeps
        save_dir="plots",
        tag="cramped_sweep"
    )
