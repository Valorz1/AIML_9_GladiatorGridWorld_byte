"""
play_agent.py — watch the trained agent run the arena in your terminal.

Run:  python play_agent.py            (trained agent, uses models/q_table.npy)
      python play_agent.py --random   (untrained agent making random moves)
"""
import os
import sys
import time

import numpy as np

from gridworld_env import ACTION_NAMES, GladiatorGridWorld

if os.name == "nt":
    os.system("")   # enable colours in the Windows terminal

COLOURS = {"#": "\033[90m", "T": "\033[91m", "G": "\033[93m", "S": "\033[94m",
           "*": "\033[96m", "A": "\033[1;92m"}
RESET = "\033[0m"


def coloured(text):
    return "".join(f"{COLOURS[ch]}{ch}{RESET}" if ch in COLOURS else ch for ch in text)


def main():
    random_mode = "--random" in sys.argv
    env = GladiatorGridWorld()
    if not random_mode:
        if not os.path.exists("models/q_table.npy"):
            sys.exit("models/q_table.npy not found. Run `python train_agent.py` first.")
        Q = np.load("models/q_table.npy")
    rng = np.random.default_rng()

    s, path, done, total = env.reset(), [env.start], False, 0
    while not done:
        a = int(rng.integers(4)) if random_mode else int(np.argmax(Q[s]))
        s, r, done, event = env.step(a)
        total += r
        path.append(env.to_pos(s))
        os.system("cls" if os.name == "nt" else "clear")
        title = "UNTRAINED (random) agent" if random_mode else "TRAINED Q-Learning agent"
        print(f"=== GLADIATOR GRID WORLD — {title} ===\n")
        print(coloured(env.render_ascii(path)))
        print(f"\nStep {env.steps:>3} | action: {ACTION_NAMES[a]:<5} | reward: {r:>4} | total: {total}")
        time.sleep(float(os.environ.get("DELAY", 0.25)))

    result = {"goal": "🏆 REACHED THE GOAL", "trap": "💀 STEPPED ON A TRAP",
              "timeout": "⌛ RAN OUT OF MOVES"}[event]
    print(f"\n{result} in {env.steps} moves (shortest possible: {env.shortest_path_length()})")


if __name__ == "__main__":
    main()
