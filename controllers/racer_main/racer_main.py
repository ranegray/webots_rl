"""
racer_main controller (Simple RL - Solution 2, v3 - Final Fix)
"""

import random
import math
import pickle
import os
from controller import Robot
from controller import Supervisor

# --- Setup ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

epuck_node = robot.getFromDef('e-puck')
epuck_translation = epuck_node.getField('translation')
epuck_rotation = epuck_node.getField('rotation')

# Sensors
ps = []
psNames = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(timestep)

# Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# GPS & Compass
gps = robot.getDevice('gps')
compass = robot.getDevice('compass')
gps.enable(timestep)
compass.enable(timestep)

# Actions
ACTIONS = ["HARD_LEFT", "SOFT_LEFT", "STRAIGHT_FAST", "SOFT_RIGHT", "HARD_RIGHT"]

# RL Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0 
MAX_STEPS_PER_EPISODE = 5000

Q = {}


def read_sensors_and_pose():
    ir_vals = [s.getValue() for s in ps]
    pos = gps.getValues()
    x = pos[0]
    z = pos[2]
    mag = compass.getValues()
    theta = math.atan2(mag[0], mag[2])
    return { "ir": ir_vals, "x": x, "y": z, "theta": theta }

def bucket_sensor_val(val):
    if val > 800.0: return 2 # Danger
    if val > 150.0: return 1 # Warning
    return 0 # Clear

def discretize_state(raw_obs):
    ir_vals = raw_obs["ir"]
    
    # Buckets for each sensor
    # ps0: front-right
    # ps1: side-right (front)
    # ps2: side-right (back)
    # ps5: side-left (back)
    # ps6: side-left (front)
    # ps7: front-left
    
    b0 = bucket_sensor_val(ir_vals[0])
    b1 = bucket_sensor_val(ir_vals[1])
    b2 = bucket_sensor_val(ir_vals[2])
    b5 = bucket_sensor_val(ir_vals[5])
    b6 = bucket_sensor_val(ir_vals[6])
    b7 = bucket_sensor_val(ir_vals[7])
    
    return (b0, b1, b2, b5, b6, b7)

def compute_reward(state, action):
    (b0, b1, b2, b5, b6, b7) = state

    if b0 == 2 or b1 == 2 or b2 == 2 or b5 == 2 or b6 == 2 or b7 == 2:
         return -100.0, True, "crash"
         
    if b7 == 1 or b0 == 1:
        return -5.0, False, "too close front"
        
    if b1 == 1 or b2 == 1 or b5 == 1 or b6 == 1:
        return -2.0, False, "too close side"

    if state == (0, 0, 0, 0, 0, 0):
        if action == "STRAIGHT_FAST":
            return +2.0, False, "clear and fast"
        elif action == "SOFT_LEFT" or action == "SOFT_RIGHT":
            return +0.5, False, "clear and turning"
        else:
            return -1.0, False, "spinning in clear"

    return -0.5, False, "generic bad state"

def set_wheel_speeds_for_action(action):
    spd = 6.28
    if action == "HARD_LEFT": l, r = 0.2*spd, 0.8*spd
    elif action == "SOFT_LEFT": l, r = 0.6*spd, 1.0*spd
    elif action == "STRAIGHT_FAST": l, r = 1.0*spd, 1.0*spd
    elif action == "SOFT_RIGHT": l, r = 1.0*spd, 0.6*spd
    elif action == "HARD_RIGHT": l, r = 0.8*spd, 0.2*spd
    else: l, r = 0.0, 0.0
    left_motor.setVelocity(l)
    right_motor.setVelocity(r)

def reset_robot_to_start(): #Resets robot in world to do another episode after termination or time out
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

def choose_action(state, Q, eps): #selects an action factoring in the current epsilon
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

def update_q(s, a, r, s_next): # updates q table
    old_q = Q.get((s, a), 0.0)
    next_max = max([Q.get((s_next, a2), 0.0) for a2 in ACTIONS])
    Q[(s, a)] = old_q + alpha * (r + gamma * next_max - old_q)

def run_episode(training=True, eps=0.1): #resets for another episode
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
    
    Q_TABLE_FILENAME = "q_table.pkl"
    DO_TRAINING = True  # <-- SET TO False TO SKIP TRAINING AND JUST EVALUATE

    #Loading Q-Table to play eval
    if os.path.exists(Q_TABLE_FILENAME):
        print(f"Loading existing Q-Table from {Q_TABLE_FILENAME}...")
        with open(Q_TABLE_FILENAME, "rb") as f:
            Q = pickle.load(f)
        print(f"Loaded {len(Q)} entries.")
    else:
        print("No Q-Table found, starting fresh.")

    if DO_TRAINING:
        episodes = 10000 
        print(f"Starting Training (v3) for {episodes} episodes...")
        
        for e in range(episodes): 
            rew, steps = run_episode(training=True, eps=epsilon)
            
            if e % 50 == 0:
                print(f"Episode {e}: Reward={rew:.2f}, Steps={steps}, Epsilon={epsilon:.3f}")
            
            if epsilon > 0.05: #setting minimum level of exploration at 5%
                epsilon *= 0.9995 
                
        print("Training finished.")

        print(f"Saving Q-Table to {Q_TABLE_FILENAME}...")
        with open(Q_TABLE_FILENAME, "wb") as f:
            pickle.dump(Q, f)
        print("Save complete.")
        # -----------------------------
    
    else:
        print("Skipping training, moving directly to evaluation.")
    
    
    print("Final Q-Table size:", len(Q))
    
    print("\nRunning greedy evaluation episodes...")
    for e in range(10):
        # This part now uses the Q-table that was either just trained or loaded
        rew, steps = run_episode(training=False, eps=0.0)
        print(f"Eval Episode {e}: Reward={rew:.2f}, Steps={steps}")
