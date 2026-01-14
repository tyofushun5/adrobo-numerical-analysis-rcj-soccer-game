import numpy as np
import math

N = 50_000
SEED = 42
min_distance = 0.18

np.random.seed(SEED)

robot_x, robot_y, ball_x, ball_y = [], [], [], []
attempts = 0
clip_x = 0
clip_y = 0

while len(robot_x) < N:
    attempts += 1

    # robot: uniform
    rx = np.random.uniform(0.4, 1.5)
    ry = np.random.uniform(0.4, 1.8)

    # ball: normal -> clip
    bx_raw = np.random.normal(0.95, 0.18)
    by_raw = np.random.normal(1.1, 0.23)

    if bx_raw < 0.4 or bx_raw > 1.5:
        clip_x += 1
    if by_raw < 0.4 or by_raw > 1.8:
        clip_y += 1

    bx = max(0.4, min(1.5, bx_raw))
    by = max(0.4, min(1.8, by_raw))

    # min distance constraint
    dist = math.hypot(rx - bx, ry - by)
    if dist >= min_distance:
        robot_x.append(rx); robot_y.append(ry)
        ball_x.append(bx);  ball_y.append(by)

robot_x = np.array(robot_x); robot_y = np.array(robot_y)
ball_x  = np.array(ball_x);  ball_y  = np.array(ball_y)

print("mean/std (ddof=0)")
print("robot x:", robot_x.mean(), robot_x.std(ddof=0))
print("robot y:", robot_y.mean(), robot_y.std(ddof=0))
print("ball  x:", ball_x.mean(),  ball_x.std(ddof=0))
print("ball  y:", ball_y.mean(),  ball_y.std(ddof=0))

accept_rate = N / attempts
print("accept rate:", accept_rate)
print("clip rate x:", clip_x / attempts)
print("clip rate y:", clip_y / attempts)
