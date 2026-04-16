"""
AIAB 2026 - Light Following Robot Simulation
Evolves neural network controllers using a genetic algorithm.
Controllers can later be transferred to the physical OpenEduBot.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# SIMULATION PARAMETERS - tweak these to match your real robot
# ============================================================

ARENA_SIZE = 200.0  # arena is 200x200 units
ROBOT_RADIUS = 5.0  # robot body radius
SENSOR_SPREAD = 0.4  # angle (radians) of each sensor from centre line (~23 degrees)
MAX_SPEED = 3.0  # max wheel speed per timestep
WHEEL_BASE = 10.0  # distance between wheels
TIMESTEPS = 500  # how long each trial lasts
LIGHT_POS = np.array([150.0, 150.0])  # light source position

# Sensor model - adjust based on your real robot measurements
SENSOR_MAX_RANGE = 150.0  # distance at which sensor reads ~0
AMBIENT_LIGHT = 0.05  # baseline reading with no light (adjust from your data)

# GA parameters
POP_SIZE = 50
GENERATIONS = 300
MUTATION_RATE = 0.1  # probability of mutating each weight
MUTATION_STRENGTH = 0.3  # std dev of gaussian noise added
TOURNAMENT_SIZE = 5
NUM_RUNS = 5  # evaluate each controller multiple times for robustness
ELITISM = 2  # keep top N from each generation

# Neural network architecture
N_INPUTS = 2  # left and right light sensor
N_HIDDEN = 5  # hidden neurons
N_OUTPUTS = 2  # left and right motor speed


# ============================================================
# NEURAL NETWORK
# ============================================================

def count_weights():
    """Total number of weights and biases in the network."""
    return (N_INPUTS * N_HIDDEN) + N_HIDDEN + (N_HIDDEN * N_OUTPUTS) + N_OUTPUTS


def decode_weights(genome):
    """Unpack a flat genome array into weight matrices and bias vectors."""
    idx = 0
    W1 = genome[idx:idx + N_INPUTS * N_HIDDEN].reshape(N_INPUTS, N_HIDDEN)
    idx += N_INPUTS * N_HIDDEN
    b1 = genome[idx:idx + N_HIDDEN]
    idx += N_HIDDEN
    W2 = genome[idx:idx + N_HIDDEN * N_OUTPUTS].reshape(N_HIDDEN, N_OUTPUTS)
    idx += N_HIDDEN * N_OUTPUTS
    b2 = genome[idx:idx + N_OUTPUTS]
    return W1, b1, W2, b2


def forward_pass(inputs, genome):
    """
    Run the neural network and return motor outputs.
    Also returns activations for Hebbian learning later.
    """
    W1, b1, W2, b2 = decode_weights(genome)

    # Hidden layer
    hidden_pre = inputs @ W1 + b1
    hidden = np.tanh(hidden_pre)

    # Output layer
    output_pre = hidden @ W2 + b2
    output = np.tanh(output_pre)

    # Scale from [-1, 1] to [0, 1] for motor speeds
    motor_output = (output + 1.0) / 2.0

    return motor_output, inputs, hidden


# ============================================================
# ROBOT SIMULATION
# ============================================================

def sensor_reading(robot_pos, robot_angle, sensor_offset, light_pos):
    """
    Calculate what a single light sensor reads.
    Returns a value between 0 (no light) and 1 (maximum light).
    """
    # Sensor direction in world coordinates
    sensor_angle = robot_angle + sensor_offset

    # Vector from robot to light
    to_light = light_pos - robot_pos
    distance = np.linalg.norm(to_light)

    if distance < 0.001:
        return 1.0

    # Angle from sensor direction to light
    light_angle = np.arctan2(to_light[1], to_light[0])
    angle_diff = abs(light_angle - sensor_angle)
    # Normalise to [0, pi]
    angle_diff = abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

    # Angular sensitivity - sensors have a field of view
    # Cosine falloff: max response when light is straight ahead of sensor
    angular_factor = max(0.0, np.cos(angle_diff))

    # Distance falloff - inverse square (clamped)
    distance_factor = max(0.0, 1.0 - (distance / SENSOR_MAX_RANGE))
    distance_factor = distance_factor ** 2  # makes it more realistic

    reading = angular_factor * distance_factor + AMBIENT_LIGHT
    return min(1.0, max(0.0, reading))


def simulate_robot(genome, render=False):
    """
    Run one trial of the robot in the arena.
    Returns fitness (higher = better light following).
    """
    # Random starting position and angle
    robot_pos = np.array([
        np.random.uniform(20, ARENA_SIZE - 20),
        np.random.uniform(20, ARENA_SIZE - 20)
    ])
    robot_angle = np.random.uniform(0, 2 * np.pi)

    total_sensor = 0.0
    positions = [robot_pos.copy()]

    for t in range(TIMESTEPS):
        # Read sensors
        left_sensor = sensor_reading(robot_pos, robot_angle, SENSOR_SPREAD, LIGHT_POS)
        right_sensor = sensor_reading(robot_pos, robot_angle, -SENSOR_SPREAD, LIGHT_POS)

        inputs = np.array([left_sensor, right_sensor])

        # Get motor commands from neural network
        motors, _, _ = forward_pass(inputs, genome)
        left_motor = motors[0] * MAX_SPEED
        right_motor = motors[1] * MAX_SPEED

        # Differential drive kinematics
        v = (left_motor + right_motor) / 2.0
        omega = (right_motor - left_motor) / WHEEL_BASE

        robot_angle += omega
        robot_pos[0] += v * np.cos(robot_angle)
        robot_pos[1] += v * np.sin(robot_angle)

        # Wall bouncing
        if robot_pos[0] < ROBOT_RADIUS:
            robot_pos[0] = ROBOT_RADIUS
            robot_angle = np.pi - robot_angle
        if robot_pos[0] > ARENA_SIZE - ROBOT_RADIUS:
            robot_pos[0] = ARENA_SIZE - ROBOT_RADIUS
            robot_angle = np.pi - robot_angle
        if robot_pos[1] < ROBOT_RADIUS:
            robot_pos[1] = ROBOT_RADIUS
            robot_angle = -robot_angle
        if robot_pos[1] > ARENA_SIZE - ROBOT_RADIUS:
            robot_pos[1] = ARENA_SIZE - ROBOT_RADIUS
            robot_angle = -robot_angle

        # Accumulate fitness: reward being close to light
        dist_to_light = np.linalg.norm(robot_pos - LIGHT_POS)
        total_sensor += max(0, 1.0 - dist_to_light / SENSOR_MAX_RANGE)

        if render:
            positions.append(robot_pos.copy())

    fitness = total_sensor / TIMESTEPS

    if render:
        return fitness, positions
    return fitness


def evaluate(genome):
    """Average fitness over multiple runs for robustness."""
    fitnesses = [simulate_robot(genome) for _ in range(NUM_RUNS)]
    return np.mean(fitnesses)


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def initialise_population():
    """Create random population."""
    n_weights = count_weights()
    return np.random.randn(POP_SIZE, n_weights) * 0.5


def tournament_select(population, fitnesses):
    """Select an individual using tournament selection."""
    indices = np.random.choice(POP_SIZE, TOURNAMENT_SIZE, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx].copy()


def crossover(parent1, parent2):
    """Single-point crossover."""
    point = np.random.randint(1, len(parent1))
    child = np.concatenate([parent1[:point], parent2[point:]])
    return child


def mutate(genome):
    """Gaussian mutation."""
    mask = np.random.random(len(genome)) < MUTATION_RATE
    genome[mask] += np.random.randn(np.sum(mask)) * MUTATION_STRENGTH
    return genome


def run_evolution():
    """Main evolutionary loop."""
    population = initialise_population()

    best_fitness_history = []
    avg_fitness_history = []

    print(f"Genome size: {count_weights()} weights")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}")
    print("-" * 50)

    for gen in range(GENERATIONS):
        # Evaluate all individuals
        fitnesses = np.array([evaluate(genome) for genome in population])

        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        avg_fitness = np.mean(fitnesses)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        if gen % 10 == 0:
            print(f"Gen {gen:4d} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f}")

        # Create next generation
        new_population = []

        # Elitism - keep the best
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(ELITISM):
            new_population.append(population[sorted_indices[i]].copy())

        # Fill rest with offspring
        while len(new_population) < POP_SIZE:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = np.array(new_population)

    # Final evaluation
    fitnesses = np.array([evaluate(genome) for genome in population])
    best_idx = np.argmax(fitnesses)
    best_genome = population[best_idx]

    print("-" * 50)
    print(f"Evolution complete. Best fitness: {fitnesses[best_idx]:.4f}")

    return best_genome, best_fitness_history, avg_fitness_history


# ============================================================
# VISUALISATION
# ============================================================

def plot_fitness(best_history, avg_history):
    """Plot fitness over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label='Best fitness', linewidth=2)
    plt.plot(avg_history, label='Average fitness', linewidth=1, alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fitness_curve.png', dpi=150)
    plt.show()
    print("Saved: fitness_curve.png")


def plot_trajectory(genome, n_trials=3):
    """Show the robot's path in the arena."""
    plt.figure(figsize=(8, 8))

    colors = ['blue', 'green', 'red']
    for i in range(n_trials):
        fitness, positions = simulate_robot(genome, render=True)
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], color=colors[i % len(colors)],
                 alpha=0.6, linewidth=1, label=f'Trial {i + 1} (fitness={fitness:.3f})')
        # Mark start and end
        plt.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i % len(colors)], markersize=8)
        plt.plot(positions[-1, 0], positions[-1, 1], 's', color=colors[i % len(colors)], markersize=8)

    # Draw light source
    plt.plot(LIGHT_POS[0], LIGHT_POS[1], '*', color='yellow', markersize=20,
             markeredgecolor='orange', markeredgewidth=2, label='Light source')

    plt.xlim(0, ARENA_SIZE)
    plt.ylim(0, ARENA_SIZE)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Robot Trajectory (o=start, square=end)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=150)
    plt.show()
    print("Saved: trajectory.png")


# ============================================================
# SAVE / LOAD WEIGHTS (for transfer to real robot)
# ============================================================

def save_weights(genome, filename='best_weights.json'):
    """Save evolved weights to JSON for transfer to the robot."""
    data = {
        'weights': genome.tolist(),
        'architecture': {
            'n_inputs': N_INPUTS,
            'n_hidden': N_HIDDEN,
            'n_outputs': N_OUTPUTS
        }
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved weights to {filename}")


def load_weights(filename='best_weights.json'):
    """Load previously saved weights."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data['weights'])


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=== Light Following Robot Evolution ===\n")

    # Run evolution
    best_genome, best_history, avg_history = run_evolution()

    # Save the best weights
    save_weights(best_genome)

    # Plot results
    plot_fitness(best_history, avg_history)
    plot_trajectory(best_genome)

    print("\nDone! Next steps:")
    print("1. Check fitness_curve.png - fitness should increase over generations")
    print("2. Check trajectory.png - robot should move toward the light")
    print("3. Transfer best_weights.json to the real robot")

