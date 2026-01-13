import os
import random
import time

import numpy as np
import pybullet as p

from environment import Environment
from object.robot import MyRobot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "default_model", "default_model")
MAX_STEPS = 10000
AI_MAGNITUDE = 21.0
USER_MAGNITUDE = 21.0
USER_ROTATE_SCALE = 2.5
SCORE_LIMIT = 5
SLEEP_SEC = 0
SEED = None


def load_model(model_path, env):
    try:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(model_path, env=env)
    except Exception as exc:
        try:
            from stable_baselines3 import PPO

            print(f"RecurrentPPO load failed: {exc}")
            print("Falling back to PPO.load(...)")
            return PPO.load(model_path, env=env)
        except Exception:
            raise


def update_held_keys(held_keys, events):
    for code, state in events.items():
        if state & p.KEY_WAS_RELEASED:
            held_keys.discard(code)
        if state & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            held_keys.add(code)


def key_down(held_keys, key):
    return ord(key) in held_keys or ord(key.upper()) in held_keys


def keyboard_action(held_keys):
    move_dir = 0
    if key_down(held_keys, "i"):
        move_dir += 1
    if key_down(held_keys, "k"):
        move_dir -= 1

    rotate = 0
    if key_down(held_keys, "j"):
        rotate += 1
    if key_down(held_keys, "l"):
        rotate -= 1

    rotate = max(-1, min(1, rotate))

    if move_dir != 0:
        angle_deg = 0.0 if move_dir > 0 else 180.0
        move = True
    else:
        angle_deg = 0.0
        move = False

    return angle_deg, rotate, move


def create_user_robot(create_position):
    user_agent = MyRobot(create_position, mode="enemy")
    user_pos = [create_position[0] + 1.0, create_position[1] + 2.0, create_position[2] + 0.1]
    user_id = user_agent.create(user_pos)
    p.changeVisualShape(user_id, -1, rgbaColor=[1.0, 1.0, 1.0, 1.0])
    return user_agent, user_id


def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    env = Environment(
        create_position=[4.0, 0.0, 0.0],
        max_steps=MAX_STEPS,
        magnitude=AI_MAGNITUDE,
        gui=True,
    )
    if hasattr(p, "COV_ENABLE_KEYBOARD_SHORTCUTS"):
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)

    model = load_model(MODEL_PATH, env=env)

    observation, info = env.reset()
    state = None
    episode_start = True

    user_agent, user_agent_id = create_user_robot(env.cp)

    user_score = 0
    ai_score = 0

    print("Controls: I/K move, J/L rotate, ESC to quit.")
    esc_key = getattr(p, "B3G_ESCAPE", 27)
    held_keys = set()

    while True:
        keys = p.getKeyboardEvents()
        update_held_keys(held_keys, keys)
        if keys.get(esc_key, 0) & p.KEY_WAS_TRIGGERED:
            break

        angle_deg, rotate, move = keyboard_action(held_keys)
        rotate *= USER_ROTATE_SCALE
        user_mag = USER_MAGNITUDE if move else 0.0
        user_agent.action(user_agent_id, angle_deg=angle_deg, rotate=rotate, magnitude=user_mag)

        action, state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = env.step(action)

        goal = None
        if env.hit_ids:
            if env.hit_ids[env.unit.court.enemy_goal_line_idx] == env.unit.ball_id:
                ai_score += 1
                goal = "AI"
            elif env.hit_ids[env.unit.court.my_goal_line_idx] == env.unit.ball_id:
                user_score += 1
                goal = "USER"

        if goal or terminated or truncated:
            if goal:
                print(f"Goal: {goal}  score USER {user_score} - AI {ai_score}")
            else:
                print(f"Reset: out/time  score USER {user_score} - AI {ai_score}")

            observation, info = env.reset()
            state = None
            episode_start = True
            user_agent, user_agent_id = create_user_robot(env.cp)

            if SCORE_LIMIT > 0 and (user_score >= SCORE_LIMIT or ai_score >= SCORE_LIMIT):
                break
        else:
            episode_start = False

        if SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)

    env.close()


if __name__ == "__main__":
    main()
