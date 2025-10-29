"""racer_main controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
# from controller import Robot

# create the Robot instance.
# robot = Robot()

# get the time step of the current world.
# timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
# while robot.step(timestep) != -1:
# Read the sensors:
# Enter here functions to read sensor data, like:
#  val = ds.getValue()

# Process sensor data here.

# Enter here functions to send actuator commands, like:
#  motor.setPosition(10.0)
# pass

# Enter here exit cleanup code.

### Q-learning racer controller ###
import random
import math

ACTIONS = ["HARD_LEFT", "SOFT_LEFT", "STRAIGHT_FAST", "SOFT_RIGHT", "HARD_RIGHT"]

# Hyperparams
alpha = 0.1  # learning rate
gamma = 0.95  # discount
epsilon = 0.2  # exploration rate during training
TIME_STEP_MS = 32
MAX_STEPS_PER_EPISODE = 1000  # safety cap

# Q-table: dict with keys (state_tuple, action_str)
Q = {}

# --------------- Helper functions -----------------


def reset_robot_to_start():
    """
    Reset pose to (x0,y0,theta0), reset wheel speeds to 0, etc.
    Return the *discretized* initial state.
    """
    # TODO: call Supervisor API if you’re using Supervisor controller,
    # or manually set position if allowed.
    # Then read sensors once and build state.
    raw_obs = read_sensors_and_pose()
    s = discretize_state(raw_obs)
    return s


def choose_action(state, Q, epsilon):
    if random.random() < epsilon:
        # explore
        return random.choice(ACTIONS)
    else:
        # exploit best known action
        best_a = None
        best_q = -1e9
        for a in ACTIONS:
            q = Q.get((state, a), 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a


def set_wheel_speeds_for_action(action):
    """
    Map each discrete action to left/right wheel speeds (rad/s).
    Tune these.
    """
    if action == "HARD_LEFT":
        left, right = 1.0, 5.0
    elif action == "SOFT_LEFT":
        left, right = 3.0, 5.0
    elif action == "STRAIGHT_FAST":
        left, right = 5.0, 5.0
    elif action == "SOFT_RIGHT":
        left, right = 5.0, 3.0
    elif action == "HARD_RIGHT":
        left, right = 5.0, 1.0
    else:
        left, right = 0.0, 0.0

    # TODO: actually command the motors through Webots API here
    # left_motor.setVelocity(left)
    # right_motor.setVelocity(right)


def apply_action_and_step(action):
    """
    1. Send wheel speeds.
    2. Step Webots forward exactly one control tick.
    3. Read sensors.
    Returns (raw_obs, dt_seconds).
    """
    set_wheel_speeds_for_action(action)

    # advance simulation one step
    # robot.step(TIME_STEP_MS)

    raw_obs = read_sensors_and_pose()

    dt_seconds = TIME_STEP_MS / 1000.0
    return raw_obs, dt_seconds


def read_sensors_and_pose():
    """
    Grab IR distances, robot (x,y,theta), maybe linear velocity.
    Return a dict of continuous/raw values.
    """
    # TODO:
    # ir_vals = [ir[i].getValue() for i in range(num_ir)]
    # x,y,theta = gps.getValues(), imu.getRollPitchYaw() or supervisor pose
    # return { "ir": ir_vals, "x": x, "y": y, "theta": theta }
    return {}


def discretize_state(raw_obs):
    """
    Turn raw_obs into a small tuple.
    Example buckets:
      - heading_bucket: {-2,-1,0,1,2}
      - left/front/right close: {0,1,2}
      - segment_id: {0..N-1}
    """
    heading_err = heading_error_bucket(raw_obs)
    left_close = proximity_bucket(raw_obs, sensor="left")
    front_close = proximity_bucket(raw_obs, sensor="front")
    right_close = proximity_bucket(raw_obs, sensor="right")
    segment_id = which_track_segment(raw_obs)

    return (heading_err, left_close, front_close, right_close, segment_id)


def heading_error_bucket(raw_obs):
    """
    Compare robot heading to local tangent of track.
    Return -2,-1,0,1,2.
    """
    # TODO: compute angle diff = wrap(robot_theta - ideal_track_theta)
    angle_diff = 0.0
    if angle_diff < -0.6:
        return -2
    if angle_diff < -0.2:
        return -1
    if angle_diff < 0.2:
        return 0
    if angle_diff < 0.6:
        return 1
    return 2


def proximity_bucket(raw_obs, sensor="front"):
    """
    Take IR sensor reading and bucket it.
    Example: 0 = clear, 1 = kinda close, 2 = danger close.
    """
    # TODO: pick which IR index maps to 'front', 'left', 'right'
    dist_val = 0.0
    if dist_val < 0.2:
        return 2  # dangerously close
    if dist_val < 0.5:
        return 1  # kinda close
    return 0  # clear


def which_track_segment(raw_obs):
    """
    Based on (x,y), figure out which stretch of track you're on.
    Helps the agent know progress around the loop.
    """
    # TODO: divide the loop path into N segments by angle or by arc length.
    return 0


def get_lap_progress(raw_obs):
    """
    Returns a float 0.0 -> 1.0 estimating % of lap completed.
    Could be based on projection of (x,y) onto centerline spline.
    """
    # TODO
    return 0.0


def check_crash(raw_obs):
    """
    True if we hit wall / went off track.
    Simple version: robot's |x,y| is outside track bounds,
    OR any proximity sensor is "danger close" for multiple consecutive steps.
    """
    # TODO
    return False


def compute_reward(raw_obs, lap_progress, crashed):
    """
    Shaping:
    -1 each step (time penalty)
    -100 if crash
    +1000 if lap_progress >= 1.0 (finished)
    """
    # base time penalty
    r = -1.0
    done = False
    event = None

    if crashed:
        r += -100.0
        done = True
        event = "crash"

    if lap_progress >= 1.0:
        r += 1000.0
        done = True
        event = "finish"

    return r, done, event


def update_q(Q, s, a, r, s_next, alpha, gamma):
    old_q = Q.get((s, a), 0.0)
    max_next = max(Q.get((s_next, a2), 0.0) for a2 in ACTIONS)
    td_target = r + gamma * max_next
    td_error = td_target - old_q
    Q[(s, a)] = old_q + alpha * td_error


# --------------- Episode loop -----------------


def run_episode(Q, training=True, epsilon=0.2):
    s = reset_robot_to_start()
    done = False

    total_reward = 0.0
    elapsed_time = 0.0
    steps = 0
    event = None

    while not done:
        # 1. pick action
        a = choose_action(s, Q, epsilon if training else 0.0)

        # 2. apply
        raw_obs, dt = apply_action_and_step(a)
        elapsed_time += dt
        steps += 1

        # 3. observe next state
        s_next = discretize_state(raw_obs)

        # 4. figure out what happened
        lap_progress = get_lap_progress(raw_obs)
        crashed = check_crash(raw_obs)
        r, done, event_local = compute_reward(raw_obs, lap_progress, crashed)
        if event_local is not None:
            event = event_local

        # 5. learn
        if training:
            update_q(Q, s, a, r, s_next, alpha, gamma)

        total_reward += r
        s = s_next

        # 6. timeout safety
        if steps >= MAX_STEPS_PER_EPISODE and not done:
            done = True
            event = "timeout"

    # Package episode metrics
    return {
        "total_reward": total_reward,
        "elapsed_time": elapsed_time,
        "steps": steps,
        "event": event,
        "success": (event == "finish"),
    }
