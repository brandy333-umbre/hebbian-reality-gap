import random

for trial in range(5):
    x = round(random.uniform(5.5, 31.5), 1)
    y = round(random.uniform(5.5, 23.5), 1)
    angle = random.randint(0, 360)
    print(f"Trial {trial+1}: x={x}cm, y={y}cm, angle={angle} degrees")