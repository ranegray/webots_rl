"""E-puck RL driver using Q-learning to navigate a track."""

import random
import math
from controller import Robot
from controller import Supervisor

# Supervisor interface and simulation timing
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck_node = robot.getFromDef("e-puck")
epuck_translation = epuck_node.getField("translation")
epuck_rotation = epuck_node.getField("rotation")

# Infrared proximity sensors
ps = []
psNames = ["ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"]
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(timestep)

# Wheels set to pure velocity control
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Pose sensors for rough localization
gps = robot.getDevice("gps")
compass = robot.getDevice("compass")
gps.enable(timestep)
compass.enable(timestep)

# Action set available to the controller
ACTIONS = ["HARD_LEFT", "SOFT_LEFT", "STRAIGHT_FAST", "SOFT_RIGHT", "HARD_RIGHT"]

# Q-learning hyperparameters we tuned empirically
alpha = 0.1
gamma = 0.95
epsilon = 1.0  # high initial exploration keeps it from getting stuck early
MAX_STEPS_PER_EPISODE = 5000  # long episodes give it time to learn

Q = {}

def read_sensors_and_pose():
    ir_vals = [s.getValue() for s in ps]
    pos = gps.getValues()
    x = pos[0]
    z = pos[2]
    mag = compass.getValues()
    theta = math.atan2(mag[0], mag[2])
    return {"ir": ir_vals, "x": x, "y": z, "theta": theta}


def bucket_sensor_val(val):
    if val > 800.0:
        return 2  # sensors report imminent collision
    if val > 150.0:
        return 1  # close enough to matter
    return 0  # comfortably far


def discretize_state(raw_obs):
    ir_vals = raw_obs["ir"]

    # Focus on six forward/side sensors and ignore the rear pair.
    b0 = bucket_sensor_val(ir_vals[0])
    b1 = bucket_sensor_val(ir_vals[1])
    b2 = bucket_sensor_val(ir_vals[2])
    b5 = bucket_sensor_val(ir_vals[5])
    b6 = bucket_sensor_val(ir_vals[6])
    b7 = bucket_sensor_val(ir_vals[7])
    
    return (b0, b1, b2, b5, b6, b7)


def compute_reward(state, action):
    (b0, b1, b2, b5, b6, b7) = state

    # Crashing ends the episode immediately
    
    if b0 == 2 or b1 == 2 or b2 == 2 or b5 == 2 or b6 == 2 or b7 == 2:
         return -100.0, True, "crash"
         
    # Strong penalty for crowding obstacles head-on
    if b7 == 1 or b0 == 1:
        return -5.0, False, "too close front"
        
    # Side scraping is discouraged as well
    if b1 == 1 or b2 == 1 or b5 == 1 or b6 == 1:
        return -2.0, False, "too close side"

    # Clear corridor: reward steady forward motion and allow gentle turns
    if state == (0, 0, 0, 0, 0, 0):
        if action == "STRAIGHT_FAST":
            return +2.0, False, "clear and fast"
        elif action == "SOFT_LEFT" or action == "SOFT_RIGHT":
            return +0.5, False, "clear and turning"
        else:  # HARD_LEFT or HARD_RIGHT
            return -1.0, False, "spinning in clear"

    # Otherwise issue a mild penalty to keep it moving purposefully
    return -0.5, False, "generic bad state"


def set_wheel_speeds_for_action(action):
    spd = 6.28
    if action == "HARD_LEFT":
        l, r = 0.2 * spd, 0.8 * spd
    elif action == "SOFT_LEFT":
        l, r = 0.6 * spd, 1.0 * spd
    elif action == "STRAIGHT_FAST":
        l, r = 1.0 * spd, 1.0 * spd
    elif action == "SOFT_RIGHT":
        l, r = 1.0 * spd, 0.6 * spd
    elif action == "HARD_RIGHT":
        l, r = 0.8 * spd, 0.2 * spd
    else:
        l, r = 0.0, 0.0
    left_motor.setVelocity(l)
    right_motor.setVelocity(r)


def reset_robot_to_start():
    initial_trans = [-1.6, 0, 0]
    initial_rot = [0, 0, 1, 0]
    epuck_translation.setSFVec3f(initial_trans)
    epuck_rotation.setSFRotation(initial_rot)
    epuck_node.resetPhysics()
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    robot.step(timestep)
    raw_obs = read_sensors_and_pose()
    return discretize_state(raw_obs)


def choose_action(state, Q, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    best = ACTIONS[0]
    max_q = -1e9
    for a in ACTIONS:
        q = Q.get((state, a), 0.0)
        if q > max_q:
            max_q = q
            best = a
    return best


def update_q(s, a, r, s_next):
    old_q = Q.get((s, a), 0.0)
    next_max = max([Q.get((s_next, a2), 0.0) for a2 in ACTIONS])
    Q[(s, a)] = old_q + alpha * (r + gamma * next_max - old_q)


def run_episode(training=True, eps=0.1):
    state = reset_robot_to_start()
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = choose_action(state, Q, eps if training else 0.0)

        set_wheel_speeds_for_action(action)

        if robot.step(timestep) == -1:
            done = True
            break

        raw_obs = read_sensors_and_pose()
        next_state = discretize_state(raw_obs)

        reward, done, event = compute_reward(next_state, action)

        if training:
            update_q(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        steps += 1

    return total_reward, steps


if __name__ == "__main__":
    episodes = 10000  # need thousands of episodes for stable behavior
    print(f"Starting training for {episodes} episodes...")

    for e in range(episodes):
        rew, steps = run_episode(training=True, eps=epsilon)

        if e % 50 == 0:  # periodic progress report
            print(
                f"Episode {e}: Reward={rew:.2f}, Steps={steps}, Epsilon={epsilon:.3f}"
            )

        # Gradually decay exploration while keeping a non-zero floor to avoid local minima
        if epsilon > 0.05:
            epsilon *= 0.9995

    print("Training finished.")
    print("Final Q-Table size:", len(Q))

    print("\nRunning greedy evaluation episodes...")
    for e in range(10):
        rew, steps = run_episode(training=False, eps=0.0)
        print(f"Eval Episode {e}: Reward={rew:.2f}, Steps={steps}")
