"""
Run this on your computer to generate the robot controller script.
It reads best_weights.json and produces robot_controller.py with
the weights hardcoded, ready to upload to the Pico.
"""

import json


def generate_robot_script(weights_file='best_weights.json', output_file='robot_controller.py'):
    with open(weights_file, 'r') as f:
        data = json.load(f)

    weights = data['weights']
    arch = data['architecture']

    weights_str = '[' + ', '.join(f'{w:.6f}' for w in weights) + ']'

    script = f'''"""
AIAB 2026 - Evolved Neural Controller for OpenEduBot
Weights evolved in simulation, transferred to real robot.
Architecture: {arch['n_inputs']} -> {arch['n_hidden']} -> {arch['n_outputs']}
"""

from EduBot_CP import wheelBot
import time
import board
import math
from analogio import AnalogIn

# ============================================================
# CONFIGURATION
# ============================================================

HEBBIAN_ENABLED = False
HEBBIAN_RATE = 0.01
WEIGHT_CLIP = 5.0
MOTOR_SCALE = 30
TIMESTEPS = 500
LOG_INTERVAL = 10

N_INPUTS = {arch['n_inputs']}
N_HIDDEN = {arch['n_hidden']}
N_OUTPUTS = {arch['n_outputs']}

weights = {weights_str}

# ============================================================
# HARDWARE SETUP
# ============================================================

s1 = AnalogIn(board.GP27)
s2 = AnalogIn(board.GP26)
bot = wheelBot(board_type="pico")
bot.stop()

def get_intensity(pin):
    return 1 - (pin.value / 65536)

# ============================================================
# NEURAL NETWORK
# ============================================================

def tanh(x):
    if x > 10:
        return 1.0
    if x < -10:
        return -1.0
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def forward_pass(inputs, w):
    idx = 0

    W1 = []
    for i in range(N_INPUTS):
        row = []
        for j in range(N_HIDDEN):
            row.append(w[idx])
            idx += 1
        W1.append(row)

    b1 = w[idx:idx + N_HIDDEN]
    idx += N_HIDDEN

    W2 = []
    for i in range(N_HIDDEN):
        row = []
        for j in range(N_OUTPUTS):
            row.append(w[idx])
            idx += 1
        W2.append(row)

    b2 = w[idx:idx + N_OUTPUTS]

    hidden = []
    for j in range(N_HIDDEN):
        val = b1[j]
        for i in range(N_INPUTS):
            val += inputs[i] * W1[i][j]
        hidden.append(tanh(val))

    output = []
    for j in range(N_OUTPUTS):
        val = b2[j]
        for i in range(N_HIDDEN):
            val += hidden[i] * W2[i][j]
        output.append(tanh(val))

    motor_output = [(o + 1.0) / 2.0 for o in output]

    return motor_output, inputs, hidden

# ============================================================
# HEBBIAN LEARNING
# ============================================================

def apply_hebbian(w, inputs, hidden, outputs, eta):
    idx = 0

    for i in range(N_INPUTS):
        for j in range(N_HIDDEN):
            w[idx] += eta * inputs[i] * hidden[j]
            w[idx] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, w[idx]))
            idx += 1

    idx += N_HIDDEN

    for i in range(N_HIDDEN):
        for j in range(N_OUTPUTS):
            w[idx] += eta * hidden[i] * outputs[j]
            w[idx] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, w[idx]))
            idx += 1

# ============================================================
# MAIN LOOP
# ============================================================

print("=== Evolved Controller ===")
print("Hebbian:", HEBBIAN_ENABLED)
if HEBBIAN_ENABLED:
    print("Learning rate:", HEBBIAN_RATE)
print("Timesteps:", TIMESTEPS)
print("-" * 40)
print("step, sensor_L, sensor_R, motor_L, motor_R")

time.sleep(3)

try:
    for t in range(TIMESTEPS):
        left_sensor = get_intensity(s1)
        right_sensor = get_intensity(s2)
        inputs = [left_sensor, right_sensor]

        motors, inp, hidden = forward_pass(inputs, weights)

        if HEBBIAN_ENABLED:
            apply_hebbian(weights, inp, hidden, motors, HEBBIAN_RATE)

        left_speed = motors[0] * MOTOR_SCALE
        right_speed = motors[1] * MOTOR_SCALE

        bot.motorOn(bot.mA, "f", left_speed)
        bot.motorOn(bot.mB, "f", right_speed)

        if t % LOG_INTERVAL == 0:
            print(t, ",", left_sensor, ",", right_sensor, ",", left_speed, ",", right_speed)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopped")

bot.stop()
print("=== Done ===")
'''

    with open(output_file, 'w') as f:
        f.write(script)

    print(f"Generated {output_file} with {len(weights)} weights")
    print(f"Motor direction: 'f' (forward)")
    print(f"Board type: pico")
    print(f"Upload {output_file} to your Pico via Thonny")


if __name__ == '__main__':
    generate_robot_script()