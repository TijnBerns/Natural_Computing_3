import matplotlib.pyplot as plt
import numpy as np


class Agent():
    def __init__(self, v: float, pos: int) -> None:
        self.v = v
        self.pos = pos
        self.best = pos


def f(x: int):
    return x ** 2


def simple_PSO(n_iter: int, agent, w: float, a1: float, a2: float, r1: float, r2: float):
    best_pos = [agent.pos]
    all_pos = [agent.pos]

    for _ in range(n_iter):
        # update velocity
        agent.v = w * agent.v + a1 * r1 * \
            (agent.best - agent.pos) + a2 * r2 * (agent.best - agent.pos)
        # agent.v = w * agent.v + (agent.best - agent.pos) * (a1 * r1 + a2 * r2)

        # update position
        agent.pos = agent.pos + agent.v

        # update_local_best
        if f(agent.pos) < f(agent.best):
            agent.best = agent.pos

        best_pos.append(agent.best)
        all_pos.append(agent.pos)

    return np.array(best_pos), np.array(all_pos)


def plot_positions(pos: np.array, title: str = ""):
    x = np.linspace(-30, 30, 1000)

    plt.figure(figsize=(7, 7))
    plt.plot(x, f(x), label="f(x)", zorder=0)
    plt.scatter(pos[:-1], f(pos[:-1]), color="tab:orange", zorder=2)
    plt.plot(pos, f(pos), color="tab:orange", label="Agent trajectory", zorder=2)
    plt.scatter(pos[len(pos) - 1], f(pos[len(pos) - 1]),
                color="tab:red", label="Final position", zorder=4)
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def exercise_2():
    n_iter = 25
    init_v = 10
    init_pos = 20

    _, all_pos_1 = simple_PSO(n_iter, Agent(
        init_v, init_pos), 0.5, 1.5, 1.5, 0.5, 0.5)
    _, all_pos_2 = simple_PSO(n_iter, Agent(
        init_v, init_pos), 0.7, 1.5, 1.5, 1., 1.)

    plot_positions(all_pos_1, "Simple PSO with paramter setting 1")
    plot_positions(all_pos_2, "Simple PSO with paramter setting 2")


if __name__ == "__main__":
    exercise_2()

# %%
