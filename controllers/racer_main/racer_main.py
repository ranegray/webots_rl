"""racer_main controller."""

import random
import math
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

# --- Constants & Waypoints ---
# (x, y, angle) from your list. Note: We treat 'y' as Webots 'z'.
WAYPOINTS = [
    (0, 0, 1.57),
    (1.12, -1.41, 0),
    (-0.28, -2.31, 1.57),
    (-1.43, -1.85, 0.85),
    (-2.6, -1.52, 1.57),
    (-3.1, -0.75, 0),
    (-1.55, -0.075, 1.57),
    (0, 0, 1.57)
]

# Pre-calculate segment vectors and lengths for lap progress
SEGMENTS = []
TOTAL_TRACK_LENGTH = 0.0
for i in range(len(WAYPOINTS) - 1):
    p1 = WAYPOINTS[i]
    p2 = WAYPOINTS[i+1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    # Calculate segment angle (tangent)
    angle = math.atan2(dy, dx)
    SEGMENTS.append({
        'p1': p1,
        'p2': p2,
        'length': length,
        'angle': angle,
        'start_dist': TOTAL_TRACK_LENGTH
    })
    TOTAL_TRACK_LENGTH += length

ACTIONS = ["HARD_LEFT", "SOFT_LEFT", "STRAIGHT_FAST", "SOFT_RIGHT", "HARD_RIGHT"]

# RL Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.2
MAX_STEPS_PER_EPISODE = 10000

Q = {}

# --- Helper Functions ---

def normalize_angle(angle):
    """Wraps angle to -pi..pi"""
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

def read_sensors_and_pose():
    """
    Returns raw observation: IR values, position (x,z), and heading.
    """
    ir_vals = [s.getValue() for s in ps]
    
    pos = gps.getValues()
    # Webots coordinates: x is x, z is y (depth)
    x = pos[0]
    z = pos[2] 

    mag = compass.getValues()
    # Heading in Webots (usually relative to North/X-axis depending on setup)
    # atan2(x, z) matches Webots North usually
    theta = math.atan2(mag[0], mag[2])
    
    return { "ir": ir_vals, "x": x, "y": z, "theta": theta }

def get_closest_segment_index(x, y):
    """Finds which track segment (0..N) the robot is closest to."""
    best_idx = 0
    min_dist = float('inf')
    
    for i, seg in enumerate(SEGMENTS):
        p1 = seg['p1']
        p2 = seg['p2']
        
        # Project point onto line segment
        px = p2[0] - p1[0]
        py = p2[1] - p1[1]
        norm = px*px + py*py
        if norm == 0: continue
        
        u = ((x - p1[0]) * px + (y - p1[1]) * py) / float(norm)
        
        # Clamp u to segment bounds
        u = max(0, min(1, u))
        
        # Closest point on segment
        cx = p1[0] + u * px
        cy = p1[1] + u * py
        
        dist = math.sqrt((x-cx)**2 + (y-cy)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            
    return best_idx

def which_track_segment(raw_obs):
    return get_closest_segment_index(raw_obs['x'], raw_obs['y'])

def heading_error_bucket(raw_obs):
    """
    Compare robot heading to the 'ideal' heading of the current track segment.
    """
    seg_idx = which_track_segment(raw_obs)
    target_angle = SEGMENTS[seg_idx]['angle']
    
    # Alternatively, interpolate target angle based on position, 
    # but segment tangent is fine for this resolution.
    
    # Note: We might need to offset Webots compass theta 
    # to match standard math theta (0 = East). 
    # Assuming read_sensors_and_pose handles coordinate frame standard.
    
    # Rotate robot theta by pi/2 if compass North is Z axis 
    # to match math atan2(dy, dx) logic if needed. 
    # For now, we use simple difference.
    
    diff = normalize_angle(raw_obs['theta'] - target_angle)
    
    # Bucket into 5 states: -2 (Far Left), -1, 0 (Good), 1, 2 (Far Right)
    if diff < -0.6: return -2
    if diff < -0.2: return -1
    if diff < 0.2:  return 0
    if diff < 0.6:  return 1
    return 2

def proximity_bucket(raw_obs, sensor="front"):
    ir_vals = raw_obs["ir"]
    # ps7(left-front), ps0(right-front)
    # ps5(left), ps2(right)
    val = 0.0
    if sensor == "front":
        val = max(ir_vals[0], ir_vals[7])
    elif sensor == "left":
        val = ir_vals[5]
    elif sensor == "right":
        val = ir_vals[2]
        
    # Tuning for e-puck IR sensors
    if val > 200.0: return 2 # Danger
    if val > 90.0:  return 1 # Warning
    return 0 # Clear

def discretize_state(raw_obs):
    h_err = heading_error_bucket(raw_obs)
    l_prox = proximity_bucket(raw_obs, "left")
    f_prox = proximity_bucket(raw_obs, "front")
    r_prox = proximity_bucket(raw_obs, "right")
    seg_id = which_track_segment(raw_obs)
    
    return (h_err, l_prox, f_prox, r_prox, seg_id)

def get_lap_progress(raw_obs):
    """Returns 0.0 -> 1.0 (or >1.0 if multiple laps)"""
    x, y = raw_obs['x'], raw_obs['y']
    idx = get_closest_segment_index(x, y)
    seg = SEGMENTS[idx]
    
    # Project onto current segment to get local progress
    px = seg['p2'][0] - seg['p1'][0]
    py = seg['p2'][1] - seg['p1'][1]
    norm = px*px + py*py
    u = 0.0
    if norm > 0:
        u = ((x - seg['p1'][0]) * px + (y - seg['p1'][1]) * py) / float(norm)
        u = max(0, min(1, u))
        
    current_dist = seg['start_dist'] + (u * seg['length'])
    progress = current_dist / TOTAL_TRACK_LENGTH
    
    return progress

def check_crash(raw_obs):
    # Threshold ~1000 implies very close to wall
    if any(v > 1000.0 for v in raw_obs["ir"]):
        return True
    return False

def compute_reward(raw_obs, prev_progress, current_progress, crashed):
    r = -0.01 # Living penalty
    
    if crashed:
        return -1000.0, True, "crash"
        
    # Reward for moving forward along track
    progress_delta = current_progress - prev_progress

    if progress_delta < -0.5: 
            # Forward wrap-around (0.99 -> 0.01)
            # Means we finished a lap (Good!)
            progress_delta += 1.0
    elif progress_delta > 0.5:
        # Backward wrap-around (0.01 -> 0.99)
        # Means we went backwards across start line (Bad!)
        progress_delta -= 1.0
    
    if progress_delta > 0:
        r += progress_delta * 2000.0 # Scale up progress reward
        
    done = False
    event = None
    
    # If we completed a lap (simplified check)
    # Real logic would track 'laps_completed' counter
    if current_progress > 0.98:
         # r += 100.0
         pass 

    return r, done, event

def set_wheel_speeds_for_action(action):
    # Base speed
    spd = 6.28
    
    if action == "HARD_LEFT":
        l, r = 0.2*spd, 0.8*spd
    elif action == "SOFT_LEFT":
        l, r = 0.6*spd, 1.0*spd
    elif action == "STRAIGHT_FAST":
        l, r = 1.0*spd, 1.0*spd
    elif action == "SOFT_RIGHT":
        l, r = 1.0*spd, 0.6*spd
    elif action == "HARD_RIGHT":
        l, r = 0.8*spd, 0.2*spd
    else:
        l, r = 0.0, 0.0

    left_motor.setVelocity(l)
    right_motor.setVelocity(r)

def reset_robot_to_start():
    initial_trans = [0, 0, 0] # x, y(height), z
    initial_rot = [0, 0, 1, 0] # Axis-Angle
    
    epuck_translation.setSFVec3f(initial_trans)
    epuck_rotation.setSFRotation(initial_rot)
    epuck_node.resetPhysics()
    
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    robot.step(timestep)
    
    raw_obs = read_sensors_and_pose()
    return discretize_state(raw_obs), get_lap_progress(raw_obs)

def choose_action(state, Q, eps):
    if random.random() < eps:
        return random.choice(ACTIONS)
    # Greedy
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
    state, progress = reset_robot_to_start()
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = choose_action(state, Q, eps if training else 0.0)
        
        set_wheel_speeds_for_action(action)
        robot.step(timestep)
        
        raw_obs = read_sensors_and_pose()
        next_state = discretize_state(raw_obs)
        next_progress = get_lap_progress(raw_obs)
        crashed = check_crash(raw_obs)
        
        reward, done, event = compute_reward(raw_obs, progress, next_progress, crashed)
        
        if training:
            update_q(state, action, reward, next_state)
            
        state = next_state
        progress = next_progress
        total_reward += reward
        steps += 1
        
    return total_reward, steps

# --- Main Loop ---
if __name__ == "__main__":
    episodes = 3000
    print("Starting Training...")
    for e in range(episodes):
        rew, steps = run_episode(training=True, eps=epsilon)
        print(f"Episode {e}: Reward={rew:.2f}, Steps={steps}")
        
        # Decay epsilon
        if epsilon > 0.05:
            epsilon *= 0.99
