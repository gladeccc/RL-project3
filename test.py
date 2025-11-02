def train_mappo(env, layout_name, obsnorm, updates=2000, rollout_steps=2048, shaping_scale=1.0):
    obs = env.reset()
    o0, o1 = get_obs_pair(obs)
    obs_dim = o0.shape[0]; act_dim = env.action_space.n

    agent = PPOMulti(obs_dim, act_dim, PPOCfg(
        total_updates=updates, rollout_env_steps=rollout_steps, shaping_scale=shaping_scale
    ))

    rewards_log, soups_log = [], []
    best_soups = -1.0

    for upd in range(1, agent.cfg.total_updates + 1):
        buf = {k: [] for k in ["obs","act","logp","rew","done","val","joint_obs"]}
        steps = 0
        while steps < agent.cfg.rollout_env_steps:
            # update & apply using the per-layout normalizer
            obsnorm.update(o0); obsnorm.update(o1)
            o0n = obsnorm.apply(o0)
            o1n = obsnorm.apply(o1)

            x0 = torch.as_tensor(o0n[None,:], dtype=torch.float32, device=device)
            x1 = torch.as_tensor(o1n[None,:], dtype=torch.float32, device=device)
            with torch.no_grad():
                d0 = agent.dist(agent.actor(x0)); d1 = agent.dist(agent.actor(x1))
                a0 = int(d0.sample().item()); a1 = int(d1.sample().item())

            a0e, a1e = a0, a1  # keep whatever ring/circuit tweaks you already do
            with torch.no_grad():
                lp0 = float(d0.log_prob(torch.tensor(a0e, device=device)).cpu().item())
                lp1 = float(d1.log_prob(torch.tensor(a1e, device=device)).cpu().item())

            joint = np.concatenate([o0n, o1n], axis=-1)
            v = agent.value(joint[None,:])[0]

            obs, R, done, info = env.step([a0e, a1e])
            r = float(R) + shaped_team_reward(info, env, scale=(2.0 if upd <= 200 else 1.0))

            for ob_n, ac, lp in [(o0n, a0e, lp0), (o1n, a1e, lp1)]:
                buf["obs"].append(ob_n); buf["act"].append(ac); buf["logp"].append(lp)
                buf["rew"].append(r); buf["done"].append(float(done))
                buf["val"].append(v); buf["joint_obs"].append(joint)

            o0, o1 = get_obs_pair(obs)
            steps += 1
            if done:
                obs = env.reset(); o0, o1 = get_obs_pair(obs)

        for k in buf: buf[k] = np.asarray(buf[k], dtype=np.float32)
        # ... your GAE + agent.update(...) unchanged ...

        if upd % 10 == 0:
            mean_ret, mean_soups = eval_soups_norm(agent, env, obsnorm, episodes=30)
            rewards_log.append(mean_ret); soups_log.append(mean_soups)
            # ... checkpoint logic ...
    return agent, rewards_log, soups_log