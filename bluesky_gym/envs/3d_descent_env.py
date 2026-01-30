import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

import gymnasium as gym
from gymnasium import spaces


# =========================
# Constants
# =========================

ALT_MEAN = 1500
ALT_STD = 3000
ALT_MEAN = 0
ALT_STD = 5

X_MEAN = 0
X_STD = 100
VX_MEAN = 0
VX_STD = 10

Y_MEAN = 0
Y_STD = 100
VY_MEAN = 0
VY_STD = 10

ACTION_2_MS_VERT = 12.5
ACTION_2_MS_HORIZ = 8.0

ALT_DIF_REWARD_SCALE = -5 / 3000
LANDING_REWARD = 100
LANDING_DISTANCE_THRESHOLD = 5  # km
CRASH_PENALTY = -100

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

AC_SPD = 150
ACTION_FREQUENCY = 30

MAX_ALT_RENDER = 5000
MAX_X_RANGE = 50  # km
MAX_Y_RANGE = 50  # km
MAX_DISTANCE = 180  # km


# =========================
# Environment
# =========================

class DescentEnvXYZ(gym.Env):
    """
    3D (x, y, z) landing environment

    x : horizontal movement (distance along x-axis)
    y : vertical movement across the screen (distance along y-axis)
    z : altitude

    z is visualized via transparency (lower = darker, higher = lighter)
    """

    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "alt": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "landing_zone_x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "landing_zone_y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
            }
        )

        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky
        if bs.sim is None:
            bs.init(mode="sim", detached=True)

        bs.scr = ScreenDummy()
        bs.stack.stack("DT 1;FF")

        self.total_reward = 0
        self.final_altitude = 0

        # Agent position in 3D space (x, y, z)
        self.x = 0.0
        self.y = 0.0
        self.altitude = 0.0
        
        # Agent velocities
        self.vx = 0.0
        self.vy = 0.0
        self.alt = 0.0
        
        # Agent heading (in degrees)
        self.heading = 0.0
        
        # Landing zone position (x, y) at altitude 0
        self.landing_zone_x = 0.0
        self.landing_zone_y = 0.0

        self.window = None
        self.clock = None
        self.font = None

    # =========================
    # Observation
    # =========================

    def _get_obs(self):
        # Get altitude from BlueSky
        # self.altitude = bs.traf.alt[0]
        self.alt = bs.traf.vs[0]
        self.altitude = bs.traf.alt[0]


        # Calculate relative positions to landing zone
        rel_x = self.x - self.landing_zone_x
        rel_y = self.y - self.landing_zone_y

        obs_target_alt = np.array([((self.target_alt- ALT_MEAN)/ALT_STD)])


        obs = {
            "x": np.array([(rel_x - X_MEAN) / X_STD]),
            "y": np.array([(rel_y - Y_MEAN) / Y_STD]),
            "z": np.array([(self.altitude - ALT_MEAN) / ALT_STD]),
            "vx": np.array([(self.vx - VX_MEAN) / VX_STD]),
            "vy": np.array([(self.vy - VY_MEAN) / VY_STD]),
            "alt": np.array([(self.alt - alt_MEAN) / alt_STD]),
            "landing_zone_x": np.array([(self.landing_zone_x - X_MEAN) / X_STD]),
            "landing_zone_y": np.array([(self.landing_zone_y - Y_MEAN) / Y_STD]),
            "target_altitude": obs_target_alt,
        }

        return obs

    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "total_reward": self.total_reward,
            "final_altitude": self.final_altitude
        }

    # =========================
    # Reward
    # =========================

    def _get_reward(self):
        # Calculate distance to landing zone
        dx = self.x - self.landing_zone_x
        dy = self.y - self.landing_zone_y
        distance_to_landing_zone = np.sqrt(dx**2 + dy**2)

        # Check if crashed (altitude <= 0)
        if self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = -100
            self.total_reward += reward
            return reward, True

        # Check if successfully landed (low altitude and close to landing zone)
        if self.altitude <= 100 and distance_to_landing_zone <= LANDING_DISTANCE_THRESHOLD:
            reward = LANDING_REWARD
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

        # Ongoing flight reward (penalize altitude deviation from target and distance from landing zone)
        # Target altitude decreases as we approach landing zone
        target_alt = max(0, distance_to_landing_zone * 10)  # Lower target altitude when closer
        alt_penalty = abs(self.altitude - target_alt) * ALT_DIF_REWARD_SCALE
        
        # Small penalty for being far from landing zone
        distance_penalty = distance_to_landing_zone * -0.1

        alt_diff_penalty = abs(self.target_alt - self.altitude) * ALT_DIF_REWARD_SCALE
        
        reward = alt_penalty + distance_penalty + alt_diff_penalty
        self.total_reward += reward
        return reward, False

    # =========================
    # Action
    # =========================

    def _get_action(self, act):
        act = np.asarray(act).squeeze()

        # Action 0: vertical speed (z-axis/altitude)
        alt_cmd = act[0] * ACTION_2_MS_VERT
        
        # Action 1: horizontal speed (x-axis)
        vx_cmd = act[1] * ACTION_2_MS_HORIZ
        
        # Action 2: vertical screen movement (y-axis)
        vy_cmd = act[2] * ACTION_2_MS_HORIZ

        # Apply vertical speed command to BlueSky
        if alt_cmd >= 0:
            bs.traf.selalt[0] = 100000
            bs.traf.selvs[0] = alt_cmd
        else:
            bs.traf.selalt[0] = 0
            bs.traf.selvs[0] = alt_cmd

        # Update velocities for x and y axes
        self.vx = vx_cmd
        self.vy = vy_cmd
        self.alt = alt_cmd
        
        # Update heading based on velocity direction
        if abs(self.vx) > 0.01 or abs(self.vy) > 0.01:
            heading_rad = np.arctan2(self.vy, self.vx)
            self.heading = np.degrees(heading_rad)

    # =========================
    # Reset
    # =========================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF,TARGET_ALT_DIF)

        self.total_reward = 0
        self.final_altitude = 0

        # Randomly place landing zone at some location with altitude 0
        self.landing_zone_x = np.random.uniform(-MAX_X_RANGE/2, MAX_X_RANGE/2)
        self.landing_zone_y = np.random.uniform(-MAX_Y_RANGE/2, MAX_Y_RANGE/2)

        # Randomly initialize agent position
        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.x = np.random.uniform(-MAX_X_RANGE/2, MAX_X_RANGE/2)
        self.y = np.random.uniform(-MAX_Y_RANGE/2, MAX_Y_RANGE/2)
        self.altitude = alt_init

        # Initialize velocities
        self.vx = 0.0
        self.vy = 0.0
        self.alt = 0.0
        self.heading = 0.0

        # Create aircraft in BlueSky at initial position
        # Convert x, y to lat/lon for BlueSky (using a reference point)
        REF_LAT = 52.0
        REF_LON = 4.0
        NM2KM = 1.852
        
        # Convert x, y (km) to approximate lat/lon offset
        lat_offset = self.x / 111.0  # Rough conversion: 1 degree lat ≈ 111 km
        lon_offset = self.y / (111.0 * np.cos(np.radians(REF_LAT)))
        
        init_lat = REF_LAT + lat_offset
        init_lon = REF_LON + lon_offset

        bs.traf.cre("KL001", actype="A320", acalt=alt_init, acspd=AC_SPD, aclat=init_lat, aclon=init_lon)
        bs.traf.swvnav[0] = False

        obs = self._get_obs()
        info = {"total_reward": self.total_reward}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # =========================
    # Step
    # =========================

    def step(self, action):
        self._get_action(action)

        # Time step in seconds (ACTION_FREQUENCY steps per action)
        # BlueSky sim runs at 1 Hz by default, so each step is 1 second
        dt = 1.0  # Each BlueSky step is 1 second

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            
            # Update position based on velocity
            # vx and vy are in m/s, x and y are in km
            # So we need to convert: distance = velocity * time
            # distance in km = (velocity in m/s) * (time in s) / 1000
            self.x += self.vx * dt / 1000.0  # Convert m to km
            self.y += self.vy * dt / 1000.0  # Convert m to km
            
            # Update altitude from BlueSky
            self.altitude = bs.traf.alt[0]
            
            # Update aircraft position in BlueSky based on x, y movements
            # Convert x, y (km) back to lat/lon
            REF_LAT = 52.0
            REF_LON = 4.0
            lat_offset = self.x / 111.0  # 1 degree lat ≈ 111 km
            lon_offset = self.y / (111.0 * np.cos(np.radians(REF_LAT)))
            new_lat = REF_LAT + lat_offset
            new_lon = REF_LON + lon_offset
            
            # Update aircraft position in BlueSky
            if len(bs.traf.id) > 0:
                try:
                    bs.traf.lat[0] = new_lat
                    bs.traf.lon[0] = new_lon
                except (IndexError, AttributeError):
                    pass

            if self.render_mode == "human":
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = {"total_reward": self.total_reward}

        if terminated:
            for acid in bs.traf.id:
                try:
                    idx = bs.traf.id2idx(acid)
                    bs.traf.delete(idx)
                except (ValueError, AttributeError):
                    pass

        return obs, reward, terminated, False, info

    # =========================
    # Rendering
    # =========================

    def _altitude_to_red_color(self, alt):
        """Convert altitude to red color value.
        Lower altitude = dark red, higher altitude = light red
        """
        alt_norm = np.clip(alt / MAX_ALT_RENDER, 0, 1)
        # Lower altitude = dark red (low value, e.g., 50)
        # Higher altitude = light red (high value, e.g., 255)
        red_value = int(50 + 205 * alt_norm)
        return red_value

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            # Enable per-pixel alpha for transparency
            self.window.set_alpha(None)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        # Fill entire background with blue
        canvas.fill((135,206,235))

        # Convert world coordinates to screen coordinates
        # Center of screen is (0, 0) in world coordinates
        center_x = self.window_width / 2
        center_y = self.window_height / 2
        
        # Scale factors
        scale_x = self.window_width / MAX_X_RANGE
        scale_y = self.window_height / MAX_Y_RANGE

        # Draw landing zone (same shape as waypoints in plan_waypoint_env.py)
        landing_zone_screen_x = int(center_x + self.landing_zone_x * scale_x)
        landing_zone_screen_y = int(center_y + self.landing_zone_y * scale_y)
        
        # Draw landing zone as two circles (filled inner circle and outline outer circle)
        landing_zone_color = (0, 0, 0)  # Black
        # Inner filled circle
        pygame.draw.circle(
            canvas,
            landing_zone_color,
            (landing_zone_screen_x, landing_zone_screen_y),
            radius=4,
            width=0
        )
        # Outer outline circle (radius based on landing distance threshold)
        pygame.draw.circle(
            canvas,
            landing_zone_color,
            (landing_zone_screen_x, landing_zone_screen_y),
            radius=int((LANDING_DISTANCE_THRESHOLD / MAX_X_RANGE) * self.window_width),
            width=2
        )

        # Draw aircraft position
        aircraft_screen_x = int(center_x + self.x * scale_x)
        aircraft_screen_y = int(center_y + self.y * scale_y)

        # Use stored heading (updated in _get_action based on velocity)
        heading_deg = self.heading

        # Get red color value based on altitude (dark red at 0, lighter as altitude increases)
        red_value = self._altitude_to_red_color(self.altitude)
        aircraft_color = (red_value, 0, 0)  # Red color, brightness based on altitude

        # Draw aircraft as a line (same as plan_waypoint_env.py)
        ac_length = 8
        # Convert length to screen coordinates
        ac_length_screen = (ac_length / MAX_X_RANGE) * self.window_width
        heading_end_x = np.cos(np.deg2rad(heading_deg)) * ac_length_screen
        heading_end_y = np.sin(np.deg2rad(heading_deg)) * ac_length_screen

        pygame.draw.line(
            canvas,
            aircraft_color,
            (aircraft_screen_x, aircraft_screen_y),
            (aircraft_screen_x + heading_end_x, aircraft_screen_y - heading_end_y),
            width=4
        )

        # Draw heading line (longer line showing heading direction)
        heading_length = 50
        heading_length_screen = (heading_length / MAX_X_RANGE) * self.window_width
        heading_end_x = np.cos(np.deg2rad(heading_deg)) * heading_length_screen
        heading_end_y = np.sin(np.deg2rad(heading_deg)) * heading_length_screen

        pygame.draw.line(
            canvas,
            (0, 0, 0),  # Black heading line
            (aircraft_screen_x, aircraft_screen_y),
            (aircraft_screen_x + heading_end_x, aircraft_screen_y - heading_end_y),
            width=1
        )

        # Draw altitude text on screen
        if self.font is None:
            self.font = pygame.font.Font(None, 36)
        
        altitude_text = f"Target Alt: {self.target_alt:.0f} m, Altitude: {self.altitude:.0f} m"
        text_surface = self.font.render(altitude_text, True, (0, 0, 0))  # Black text
        canvas.blit(text_surface, (10, 10))  # Position at top-left corner

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        bs.stack.stack("quit")
