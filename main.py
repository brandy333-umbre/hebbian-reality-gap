"""
AIAB 2026 - Light Following Robot Simulation v3
Arena: 74cm x 29cm, random starting angles
Light at left end, robot spawns in right half.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# ============================================================
# ARENA AND ROBOT PARAMETERS
# ============================================================

ARENA_WIDTH = 74.0
ARENA_HEIGHT = 29.0
ROBOT_RADIUS = 5.5
SENSOR_SPREAD = 0.4
MAX_SPEED = 2.0
WHEEL_BASE = 10.0
TIMESTEPS = 300

LIGHT_POS = np.array([2.0, 14.5])

SPAWN_X_MIN = 37.0 + ROBOT_RADIUS
SPAWN_X_MAX = 74.0 - ROBOT_RADIUS
SPAWN_Y_MIN = ROBOT_RADIUS
SPAWN_Y_MAX = 29.0 - ROBOT_RADIUS

SENSOR_MAX_RANGE = 74.0
AMBIENT_LIGHT = 0.05

POP_SIZE = 50
GENERATIONS = 300
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.3
TOURNAMENT_SIZE = 5
NUM_RUNS = 5
ELITISM = 2

N_INPUTS = 2
N_HIDDEN = 5
N_OUTPUTS = 2


# ============================================================
# NEURAL NETWORK
# ============================================================

def count_weights():
    return (N_INPUTS * N_HIDDEN) + N_HIDDEN + (N_HIDDEN * N_OUTPUTS) + N_OUTPUTS

def forward_pass(inputs, genome):
    idx = 0
    W1 = genome[idx:idx + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN)
    idx += N_INPUTS * N_HIDDEN
    b1 = genome[idx:idx + N_HIDDEN]
    idx += N_HIDDEN
    W2 = genome[idx:idx + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS)
    idx += N_HIDDEN * N_OUTPUTS
    b2 = genome[idx:idx + N_OUTPUTS]

    hidden = np.tanh(inputs @ W1 + b1)
    output = np.tanh(hidden @ W2 + b2)
    motor_output = (output + 1.0) / 2.0
    return motor_output, inputs, hidden


# ============================================================
# SIMULATION
# ============================================================

def sensor_reading(robot_pos, robot_angle, sensor_offset, light_pos):
    sensor_angle = robot_angle + sensor_offset
    to_light = light_pos - robot_pos
    distance = np.linalg.norm(to_light)

    if distance < 0.001:
        return 1.0

    light_angle = np.arctan2(to_light[1], to_light[0])
    angle_diff = light_angle - sensor_angle
    angle_diff = abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

    angular_factor = max(0.0, np.cos(angle_diff))
    distance_factor = max(0.0, 1.0 - (distance / SENSOR_MAX_RANGE))
    distance_factor = distance_factor ** 2

    reading = angular_factor * distance_factor + AMBIENT_LIGHT
    return min(1.0, max(0.0, reading))

def simulate_robot(genome, render=False):
    robot_pos = np.array([
        np.random.uniform(SPAWN_X_MIN, SPAWN_X_MAX),
        np.random.uniform(SPAWN_Y_MIN, SPAWN_Y_MAX)
    ])
    robot_angle = np.random.uniform(0, 2 * np.pi)

    total_fitness = 0.0
    positions = [robot_pos.copy()]

    for t in range(TIMESTEPS):
        left_sensor = sensor_reading(robot_pos, robot_angle, SENSOR_SPREAD, LIGHT_POS)
        right_sensor = sensor_reading(robot_pos, robot_angle, -SENSOR_SPREAD, LIGHT_POS)

        inputs = np.array([left_sensor, right_sensor])
        motors, _, _ = forward_pass(inputs, genome)
        left_motor = motors[0] * MAX_SPEED
        right_motor = motors[1] * MAX_SPEED

        v = (left_motor + right_motor) / 2.0
        omega = (right_motor - left_motor) / WHEEL_BASE

        robot_angle += omega
        new_x = robot_pos[0] + v * np.cos(robot_angle)
        new_y = robot_pos[1] + v * np.sin(robot_angle)

        if new_x < ROBOT_RADIUS:
            new_x = ROBOT_RADIUS
        if new_x > ARENA_WIDTH - ROBOT_RADIUS:
            new_x = ARENA_WIDTH - ROBOT_RADIUS
        if new_y < ROBOT_RADIUS:
            new_y = ROBOT_RADIUS
        if new_y > ARENA_HEIGHT - ROBOT_RADIUS:
            new_y = ARENA_HEIGHT - ROBOT_RADIUS

        robot_pos[0] = new_x
        robot_pos[1] = new_y

        dist_to_light = np.linalg.norm(robot_pos - LIGHT_POS)
        total_fitness += max(0, 1.0 - dist_to_light / SENSOR_MAX_RANGE)

        if render:
            positions.append(robot_pos.copy())

    fitness = total_fitness / TIMESTEPS

    if render:
        return fitness, positions
    return fitness

def evaluate(genome):
    return np.mean([simulate_robot(genome) for _ in range(NUM_RUNS)])


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def run_evolution():
    n_weights = count_weights()
    population = np.random.randn(POP_SIZE, n_weights) * 0.5

    best_fitness_history = []
    avg_fitness_history = []

    print(f"Arena: {ARENA_WIDTH}cm x {ARENA_HEIGHT}cm")
    print(f"Light: ({LIGHT_POS[0]}, {LIGHT_POS[1]})")
    print(f"Spawn: x=[{SPAWN_X_MIN:.1f}, {SPAWN_X_MAX:.1f}], y=[{SPAWN_Y_MIN:.1f}, {SPAWN_Y_MAX:.1f}]")
    print(f"Starting angle: RANDOM (0-360 degrees)")
    print(f"Genome: {n_weights} weights | Pop: {POP_SIZE} | Gens: {GENERATIONS}")
    print("-" * 50)

    for gen in range(GENERATIONS):
        fitnesses = np.array([evaluate(g) for g in population])

        best_idx = np.argmax(fitnesses)
        best_fitness_history.append(fitnesses[best_idx])
        avg_fitness_history.append(np.mean(fitnesses))

        if gen % 10 == 0:
            print(f"Gen {gen:4d} | Best: {fitnesses[best_idx]:.4f} | Avg: {np.mean(fitnesses):.4f}")

        # Next generation
        new_pop = []
        sorted_idx = np.argsort(fitnesses)[::-1]
        for i in range(ELITISM):
            new_pop.append(population[sorted_idx[i]].copy())

        while len(new_pop) < POP_SIZE:
            # Tournament selection
            t1 = np.random.choice(POP_SIZE, TOURNAMENT_SIZE, replace=False)
            t2 = np.random.choice(POP_SIZE, TOURNAMENT_SIZE, replace=False)
            p1 = population[t1[np.argmax(fitnesses[t1])]].copy()
            p2 = population[t2[np.argmax(fitnesses[t2])]].copy()

            # Crossover
            point = np.random.randint(1, len(p1))
            child = np.concatenate([p1[:point], p2[point:]])

            # Mutation
            mask = np.random.random(len(child)) < MUTATION_RATE
            child[mask] += np.random.randn(np.sum(mask)) * MUTATION_STRENGTH
            new_pop.append(child)

        population = np.array(new_pop)

    fitnesses = np.array([evaluate(g) for g in population])
    best_idx = np.argmax(fitnesses)
    print("-" * 50)
    print(f"Done. Best fitness: {fitnesses[best_idx]:.4f}")

    return population[best_idx], best_fitness_history, avg_fitness_history


# ============================================================
# VISUALISATION
# ============================================================

def plot_fitness(best_history, avg_history):
    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label='Best fitness', linewidth=2)
    plt.plot(avg_history, label='Average fitness', linewidth=1, alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations (Random Start Angles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fitness_curve.png', dpi=150)
    plt.show()
    print("Saved: fitness_curve.png")

def plot_trajectory(genome, n_trials=5):
    plt.figure(figsize=(12, 5))
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i in range(n_trials):
        fitness, positions = simulate_robot(genome, render=True)
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], color=colors[i % len(colors)],
                alpha=0.6, linewidth=1.5, label=f'Trial {i+1} (fitness={fitness:.3f})')
        plt.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i % len(colors)], markersize=8)
        plt.plot(positions[-1, 0], positions[-1, 1], 's', color=colors[i % len(colors)], markersize=8)

    plt.plot(LIGHT_POS[0], LIGHT_POS[1], '*', color='yellow', markersize=20,
            markeredgecolor='orange', markeredgewidth=2, label='Light source')
    plt.axvspan(SPAWN_X_MIN, SPAWN_X_MAX, alpha=0.1, color='green', label='Spawn area')

    plt.xlim(0, ARENA_WIDTH)
    plt.ylim(0, ARENA_HEIGHT)
    plt.xlabel('X position (cm)')
    plt.ylabel('Y position (cm)')
    plt.title('Robot Trajectory - Random Start Angles (o=start, square=end)')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=150)
    plt.show()
    print("Saved: trajectory.png")


# ============================================================
# SAVE / LOAD
# ============================================================

def save_weights(genome, filename='best_weights.json'):
    data = {
        'weights': genome.tolist(),
        'architecture': {'n_inputs': N_INPUTS, 'n_hidden': N_HIDDEN, 'n_outputs': N_OUTPUTS},
        'arena': {'width': ARENA_WIDTH, 'height': ARENA_HEIGHT, 'light_pos': LIGHT_POS.tolist()}
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")

def load_weights(filename='best_weights.json'):
    with open(filename, 'r') as f:
        return np.array(json.load(f)['weights'])


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=== Light Following Robot Evolution v3 ===\n")

    best_genome, best_history, avg_history = run_evolution()
    save_weights(best_genome)
    plot_fitness(best_history, avg_history)
    plot_trajectory(best_genome)