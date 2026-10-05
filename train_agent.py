"""
train_agent.py — AVIP 2026 | AI/ML Task 9: The "Gladiator" Grid World
Tabular Q-Learning agent that learns the shortest safe path through the arena.

Run:  python train_agent.py
Produces:
  models/q_table.npy, models/q_table.csv   the learned Q-table
  outputs/training_log.csv                 per-episode results
  outputs/training_progress.png            win rate, path efficiency, steps, reward
  outputs/policy_map.png                   learned values + policy arrows + final path
  outputs/untrained_agent.gif              random agent before training
  outputs/trained_agent.gif                fully trained agent solving the arena
  docs/replay_data.js                      data for the live web replay (GitHub Pages)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import ListedColormap

from gridworld_env import ACTION_ARROWS, ACTION_NAMES, ARENA, GladiatorGridWorld

# ----------------------------------------------------------- hyper-parameters
EPISODES = 5000
ALPHA = 0.1            # learning rate: how much each new experience updates Q
GAMMA = 0.95           # discount: how much future rewards matter
EPS_START, EPS_END = 1.0, 0.01
EPS_DECAY_EPISODES = 3500   # explore a lot early, exploit what was learned later
CHECKPOINTS = [1, 50, 100, 200, 300, 500, 1000, 2000, 3500, 5000]
SEED = 42

TILE_COLOURS = {".": "#d9dee3", "#": "#3a4450", "T": "#e0457b", "S": "#7fb8e6", "G": "#f2b33d"}


def epsilon_at(ep):
    frac = min(ep / EPS_DECAY_EPISODES, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_rollout(env, Q, max_steps=60):
    """Follow the best-known action from every tile. Returns path + outcome."""
    s = env.reset()
    path, outcome = [env.to_pos(s)], "timeout"
    for _ in range(max_steps):
        s, _, done, event = env.step(int(np.argmax(Q[s])))
        path.append(env.to_pos(s))
        if done:
            outcome = event
            break
    return path, outcome


def random_rollout(env, rng, max_steps=40):
    s = env.reset()
    path, outcome = [env.to_pos(s)], "timeout"
    for _ in range(max_steps):
        s, _, done, event = env.step(int(rng.integers(4)))
        path.append(env.to_pos(s))
        if done:
            outcome = event
            break
    return path, outcome


# ----------------------------------------------------------- training loop
def train(env):
    rng = np.random.default_rng(SEED)
    Q = np.zeros((env.n_states, env.n_actions))
    shortest = env.shortest_path_length()
    log, snapshots = [], []

    for ep in range(1, EPISODES + 1):
        eps = epsilon_at(ep)
        s, total, done = env.reset(), 0, False
        while not done:
            # epsilon-greedy: explore with probability eps, otherwise exploit
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                best = np.flatnonzero(Q[s] == Q[s].max())   # break ties randomly
                a = int(rng.choice(best))
            s2, r, done, event = env.step(a)

            # Q-Learning update (Bellman equation)
            target = r if done and event != "timeout" else r + GAMMA * np.max(Q[s2])
            Q[s, a] += ALPHA * (target - Q[s, a])
            s, total = s2, total + r

        win = event == "goal"
        log.append({"episode": ep, "epsilon": round(eps, 4), "outcome": event,
                    "win": int(win), "steps": env.steps, "total_reward": total,
                    "path_efficiency": round(shortest / env.steps, 4) if win else 0.0})

        if ep in CHECKPOINTS:
            path, outcome = greedy_rollout(env, Q)
            snapshots.append({"episode": ep, "outcome": outcome,
                              "path": [list(map(int, p)) for p in path]})
            print(f"Episode {ep:>5} | eps {eps:.2f} | greedy agent: {outcome:<7} "
                  f"in {len(path) - 1:>2} moves")

    return Q, pd.DataFrame(log), snapshots, shortest


# ----------------------------------------------------------- plots
def plot_progress(log, shortest):
    w = 100
    roll = log.rolling(w, min_periods=1)
    wins = log[log["win"] == 1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.plot(log["episode"], roll["win"].mean() * 100, color="#2a9d8f")
    ax.set(title=f"Win rate (rolling {w} episodes)", xlabel="Episode", ylabel="Win rate %", ylim=(0, 102))
    ax2 = ax.twinx()
    ax2.plot(log["episode"], log["epsilon"], color="#999", ls="--", lw=1)
    ax2.set_ylabel("Exploration rate ε (dashed)", color="#777")

    ax = axes[0, 1]
    eff = wins.set_index("episode")["path_efficiency"].rolling(w, min_periods=1).mean() * 100
    ax.plot(eff.index, eff.values, color="#e76f51")
    ax.axhline(100, color="#264653", ls=":", label="100% = shortest possible path")
    ax.set(title="Path efficiency on winning episodes (rolling)", xlabel="Episode",
           ylabel="Shortest path ÷ steps taken (%)", ylim=(0, 105))
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    steps = wins.set_index("episode")["steps"].rolling(w, min_periods=1).mean()
    ax.plot(steps.index, steps.values, color="#264653")
    ax.axhline(shortest, color="#e76f51", ls=":", label=f"Shortest path = {shortest} moves")
    ax.set(title="Steps needed to reach the goal (rolling)", xlabel="Episode", ylabel="Steps")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(log["episode"], roll["total_reward"].mean(), color="#8e44ad")
    ax.set(title="Total reward per episode (rolling)", xlabel="Episode", ylabel="Reward")

    fig.suptitle("Gladiator Grid World — Q-Learning Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/training_progress.png", dpi=150)
    plt.close()


def draw_arena(ax, env):
    codes = {k: i for i, k in enumerate(TILE_COLOURS)}
    img = np.vectorize(codes.get)(env.grid)
    ax.imshow(img, cmap=ListedColormap(list(TILE_COLOURS.values())), vmin=0, vmax=len(codes) - 1)
    ax.set_xticks(np.arange(-0.5, env.n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for (r, c), ch in np.ndenumerate(env.grid):
        if ch in "SGT":
            ax.text(c, r, {"S": "S", "G": "G", "T": "✕"}[ch], ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white" if ch == "T" else "#1b2a38")


def plot_policy(env, Q, final_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.8))

    # Left: learned state values V(s) = max_a Q(s, a)
    V = Q.max(axis=1).reshape(env.n_rows, env.n_cols)
    # Goal and trap tiles end the episode, so Q-Learning never gives them a
    # value of their own: hide them from the colour scale and draw them as tiles.
    masked = np.ma.masked_where(np.isin(env.grid, ["#", "T", "G"]), V)
    im = ax1.imshow(masked, cmap="viridis")
    for (r, c), ch in np.ndenumerate(env.grid):
        if ch == "#":
            ax1.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#3a4450"))
        elif ch == "T":
            ax1.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#e0457b"))
            ax1.text(c, r, "✕", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        elif ch == "G":
            ax1.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="#f2b33d"))
            ax1.text(c, r, "G", ha="center", va="center", color="#1b2a38", fontsize=13, fontweight="bold")
        else:
            ax1.text(c, r, ACTION_ARROWS[int(np.argmax(Q[env.to_state((r, c))]))],
                     ha="center", va="center", color="white", fontsize=13)
    ax1.set_title("Learned values V(s) and best action per tile")
    ax1.tick_params(labelbottom=False, labelleft=False, length=0)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    # Right: arena with the final greedy path
    draw_arena(ax2, env)
    ys, xs = zip(*final_path)
    ax2.plot(xs, ys, "-o", color="#2ec4b6", lw=3, ms=6)
    ax2.set_title(f"Trained agent's path: {len(final_path) - 1} moves")
    plt.tight_layout()
    plt.savefig("outputs/policy_map.png", dpi=150)
    plt.close()


def make_gif(env, path, outcome, filename, title):
    fig, ax = plt.subplots(figsize=(5.6, 6))
    draw_arena(ax, env)
    trail, = ax.plot([], [], "-", color="#2ec4b6", lw=3, alpha=0.7)
    agent, = ax.plot([], [], "o", color="#0b6e66", ms=20, markeredgecolor="white", mew=2)
    caption = ax.set_title("")
    hold = 6   # extra frames at the end so the result is readable

    def update(i):
        i = min(i, len(path) - 1)
        ys, xs = zip(*path[: i + 1])
        trail.set_data(xs, ys)
        agent.set_data([xs[-1]], [ys[-1]])
        status = f"step {i}/{len(path) - 1}"
        if i == len(path) - 1:
            status = {"goal": f"GOAL in {i} moves!", "trap": f"Hit a trap after {i} moves",
                      "timeout": f"Lost after {i} moves"}[outcome]
        caption.set_text(f"{title}\n{status}")
        return trail, agent, caption

    anim = animation.FuncAnimation(fig, update, frames=len(path) + hold, interval=250, blit=False)
    anim.save(filename, writer=animation.PillowWriter(fps=4))
    plt.close()


def save_q_table(env, Q):
    np.save("models/q_table.npy", Q)
    rows = []
    for s in range(env.n_states):
        r, c = env.to_pos(s)
        rows.append({"state": s, "row": r, "col": c, "tile": env.grid[r, c],
                     **{f"Q_{ACTION_NAMES[a]}": round(Q[s, a], 3) for a in range(4)},
                     "best_action": ACTION_NAMES[int(np.argmax(Q[s]))]
                     if env.grid[r, c] in ".S" else "-"})
    pd.DataFrame(rows).to_csv("models/q_table.csv", index=False)


def export_web_data(env, Q, snapshots, shortest, log):
    best = [int(np.argmax(Q[s])) for s in range(env.n_states)]
    data = {
        "arena": ARENA,
        "shortest": shortest,
        "episodes": len(log),
        "finalWinRate": round(log.tail(500)["win"].mean() * 100, 1),
        "values": [round(float(v), 2) for v in Q.max(axis=1)],
        "bestAction": best,
        "snapshots": snapshots,
    }
    with open("docs/replay_data.js", "w", encoding="utf-8") as f:
        f.write("window.REPLAY = " + json.dumps(data) + ";\n")


def main():
    for d in ("models", "outputs", "docs"):
        os.makedirs(d, exist_ok=True)
    env = GladiatorGridWorld()

    # Before training: a random agent, for comparison
    rpath, routcome = random_rollout(env, np.random.default_rng(7))
    make_gif(env, rpath, routcome, "outputs/untrained_agent.gif", "Untrained agent (random moves)")

    Q, log, snapshots, shortest = train(env)
    log.to_csv("outputs/training_log.csv", index=False)
    save_q_table(env, Q)

    path, outcome = greedy_rollout(env, Q)
    moves = len(path) - 1
    first_win = log.loc[log["win"] == 1, "episode"].min()
    summary = {
        "episodes": EPISODES,
        "shortest_possible_path": shortest,
        "trained_agent_outcome": outcome,
        "trained_agent_moves": moves,
        "found_optimal_path": bool(outcome == "goal" and moves == shortest),
        "first_win_episode": int(first_win),
        "win_rate_first_500_pct": round(log.head(500)["win"].mean() * 100, 1),
        "win_rate_last_500_pct": round(log.tail(500)["win"].mean() * 100, 1),
        "trap_rate_first_500_pct": round((log.head(500)["outcome"] == "trap").mean() * 100, 1),
        "trap_rate_last_500_pct": round((log.tail(500)["outcome"] == "trap").mean() * 100, 1),
    }
    with open("outputs/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    plot_progress(log, shortest)
    plot_policy(env, Q, path)
    make_gif(env, path, outcome, "outputs/trained_agent.gif", "Trained Q-Learning agent")
    export_web_data(env, Q, snapshots, shortest, log)

    env.reset()
    print("\nFinal path ('*'):\n" + env.render_ascii(path))
    print("\nSaved Q-table, plots, GIFs and docs/replay_data.js")


if __name__ == "__main__":
    main()
