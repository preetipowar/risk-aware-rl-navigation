import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# -------------------------
# ENVIRONMENT SETUP
# -------------------------
grid_size = 5

start = (0, 0)
goal = (4, 4)

obstacles = [(0,3), (1,3), (4,0), (4,1)]
risk_zones = [(1,1), (2,1), (3,2)]
Q_normal = np.zeros((grid_size, grid_size, 4))
Q_risk = np.zeros((grid_size, grid_size, 4))
actions = [0, 1, 2, 3]  # up, down, left, right

# Q-table
Q = np.zeros((grid_size, grid_size, 4))

# Hyperparameters
alpha = 0.1
gamma = 0.9
episodes = 3000

# -------------------------
# FUNCTIONS
# -------------------------
def get_next_state(state, action):
    x, y = state
    new_x, new_y = x, y

    if action == 0: new_x -= 1
    elif action == 1: new_x += 1
    elif action == 2: new_y -= 1
    elif action == 3: new_y += 1

   
    if new_x < 0 or new_x >= grid_size or new_y < 0 or new_y >= grid_size:
        return state

    
    if (new_x, new_y) in obstacles:
        return state

    return (new_x, new_y)


def get_reward(state, agent_type="normal"):
    if state in obstacles:
        return -10
    elif state in risk_zones:
        if agent_type == "risk":
            return -50   # 🔥 stronger penalty
        else:
            return -1
    elif state == goal:
        return 100
    else:
        return -1


# -------------------------
# TRAINING
# -------------------------
for ep in range(episodes):

    # -------- NORMAL AGENT --------
    state = start
    while state != goal:
        x, y = state

        if random.uniform(0,1) < 0.03:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q_normal[x,y])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state, "normal")

        nx, ny = next_state

        Q_normal[x,y,action] += alpha * (
            reward + gamma * np.max(Q_normal[nx,ny]) - Q_normal[x,y,action]
        )

        state = next_state

    # -------- RISK-AWARE AGENT --------
    state = start
    while state != goal:
        x, y = state

        if random.uniform(0,1) < 0.03:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q_risk[x,y])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state, "risk")

        nx, ny = next_state

        Q_risk[x,y,action] += alpha * (
            reward + gamma * np.max(Q_risk[nx,ny]) - Q_risk[x,y,action]
        )

        state = next_state
def get_path(Q_table):
    state = start
    path = [state]

    steps = 0
    while state != goal and steps < 50:
        x, y = state
        action = np.argmax(Q_table[x,y])
        next_state = get_next_state(state, action)

        if next_state == state:
            break

        state = next_state
        path.append(state)
        steps += 1

    return path


path_normal = get_path(Q_normal)
path_risk = get_path(Q_risk)

print("Normal Path:", path_normal)
print("Risk-Aware Path:", path_risk)
# -------------------------
# VALUE MAP
# -------------------------
value_map = np.max(Q_normal, axis=2)

print("\nValue Map:")
print(value_map)

plt.figure(figsize=(6,5))
plt.imshow(value_map)
plt.colorbar(label="Value")
plt.title("Agent Value Map (Brain)")

for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, round(value_map[i, j], 1),
                 ha='center', va='center', color='black')

plt.show()

# -------------------------
# GENERALIZATION TEST
# -------------------------
print("\n--- GENERALIZATION TEST ---")

# new environment
obstacles = [(1,1), (2,2), (3,3)]
risk_zones = [(0,2), (2,1), (4,3)]

state = start
path = [state]

steps = 0
prev_state = None
Q_test = Q_risk   # or Q_normal
while state != goal and steps < 50:
    x, y = state

    q_values = Q[x,y].copy()

    # avoid going back
    if prev_state is not None:
        for a in range(4):
            if get_next_state(state, a) == prev_state:
                q_values[a] -= 100

    # small exploration
    if random.uniform(0,1) < 0.02:
        action = random.randint(0,3)
    else:
        action = np.argmax(q_values)

    next_state = get_next_state(state, action)

    # ensure movement
    attempts = 0
    while next_state == state and attempts < 10:
        action = random.randint(0,3)
        next_state = get_next_state(state, action)
        attempts += 1

    prev_state = state
    state = next_state

    path.append(state)
    steps += 1

print("Generalization Path:", path)

# -------------------------
# ANIMATION
# -------------------------
from matplotlib.colors import ListedColormap

import imageio  # add this at top of file

def animate_comparison(path1, path2, obstacles, risk_zones):
    plt.ion()
    fig, ax = plt.subplots()

    frames = []  # 🔥 NEW: store frames

    max_len = max(len(path1), len(path2))

    for i in range(max_len):
        ax.clear()

        # --- Draw grid background (white)
        ax.set_xlim(-0.5, grid_size-0.5)
        ax.set_ylim(grid_size-0.5, -0.5)
        ax.set_facecolor("white")

        # --- Draw grid lines
        for x in range(grid_size):
            for y in range(grid_size):
                rect = plt.Rectangle((y-0.5, x-0.5), 1, 1,
                                     fill=False, edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)

        # --- Obstacles (BLACK)
        for ox, oy in obstacles:
            rect = plt.Rectangle((oy-0.5, ox-0.5), 1, 1,
                                 color='black')
            ax.add_patch(rect)

        # --- Risk zones (ORANGE with transparency)
        for rx, ry in risk_zones:
            rect = plt.Rectangle((ry-0.5, rx-0.5), 1, 1,
                                 color='orange', alpha=0.5)
            ax.add_patch(rect)

        # --- Goal (GREEN)
        gx, gy = goal
        rect = plt.Rectangle((gy-0.5, gx-0.5), 1, 1,
                             color='green')
        ax.add_patch(rect)

        # --- Agent A (Normal → BLUE CIRCLE)
        if i < len(path1):
            x1, y1 = path1[i]
            circle = plt.Circle((y1, x1), 0.3, color='blue', alpha=0.9)
            ax.add_patch(circle)

        # --- Agent B (Risk-aware → RED SQUARE)
        if i < len(path2):
            x2, y2 = path2[i]
            rect = plt.Rectangle((y2-0.3, x2-0.3), 0.6, 0.6,
                                 color='red', alpha=0.9)
            ax.add_patch(rect)

        ax.set_title("Blue = Normal (Circle) | Red= Risk-Aware (Square)")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        fig.canvas.draw()
        fig.canvas.flush_events()

        # 🔥 NEW: capture frame (backend-safe)
        image = np.array(fig.canvas.renderer.buffer_rgba())
        image = image[:, :, :3]  # remove alpha
        frames.append(image)

        plt.pause(0.5)

    # 🔥 NEW: save GIF (no outputs folder needed)
    imageio.mimsave("animation.gif", frames, fps=2)

    plt.ioff()
    plt.show()
    
obstacles = [(0,3), (1,3), (4,0), (4,1)]
risk_zones = [(1,1), (2,1), (3,2)]
print("\nStarting animation...")
animate_comparison(path_normal, path_risk, obstacles, risk_zones)