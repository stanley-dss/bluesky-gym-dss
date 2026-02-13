"""
BlueSky GUI variant of the 3D Descent with Conflict Resolution environment.

Identical logic to 3d_descent_cr_env.py but renders in the BlueSky QtGL GUI
instead of pygame. Requires a running BlueSky server (python BlueSky.py).
"""
from pathlib import Path
import numpy as np
import glob
import random
import time

import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky.core.walltime import Timer

import gymnasium as gym
from gymnasium import spaces


# =========================
# Constants (same as 3d_descent_cr_env.py)
# =========================

ALT_MEAN = 1500
ALT_STD = 3000

AC_SPD = 150
ACTION_FREQUENCY = 20

ACTION_2_MS = 12.5
D_HEADING = 45

ALT_DIF_REWARD_SCALE = -5 / 3000
REACH_REWARD = 100
CRASH_PENALTY = -100
RWY_ALT_DIF_REWARD_SCALE = -5 / 300

ALT_MIN = 2000
ALT_MAX = 4000
TARGET_ALT_DIF = 500

VZ_MEAN = 0
VZ_STD = 5

NUM_WAYPOINTS = 1
DISTANCE_MARGIN = 5
WAYPOINT_DISTANCE_MIN = 75
WAYPOINT_DISTANCE_MAX = 300

INTRUSION_DISTANCE = 5
VERTICAL_MARGIN = 1000 * 0.3048

INTRUSION_PENALTY = -20
NM2KM = 1.852

# Agent color (red) for BlueSky GUI
AGENT_COLOR = (255, 0, 0)

# Seconds to sleep per sim step so the GUI can follow along.
# DT=1 means each step is 1 s of sim time. With 20 steps per action,
# GUI_STEP_DELAY=0.05 gives ~1 s wall time per action (≈20× speedup).
GUI_STEP_DELAY = 0.05


# =========================
# Environment
# =========================

class DescentEnvCR3DBluesky(gym.Env):
    """
    3D descent + conflict resolution environment that renders in the BlueSky
    QtGL GUI instead of pygame.

    Usage:
        1. Start BlueSky server+GUI:  python BlueSky.py
        2. Then in your script:
            bs.init(mode='sim', detached=False)
            bs.net.connect()
            env = gym.make('DescentEnvCR3D-Bluesky-v0', render_mode='bluesky', ...)
    """

    metadata = {"render_modes": ["bluesky"], "render_fps": 120}

    def __init__(self, render_mode=None, scenario_path=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)
        self.rng = np.random.default_rng(seed=1)

        # Handle scenario path
        self.scenario_files = []
        if scenario_path is not None:
            path = Path(scenario_path)
            if path.is_dir():
                self.scenario_files = sorted(glob.glob(str(path / "*.scn")))
                if not self.scenario_files:
                    raise ValueError(f"No .scn files found in directory: {scenario_path}")
                init_scenario = self.scenario_files[0]
            elif path.is_file():
                self.scenario_files = [str(path)]
                init_scenario = str(path)
            else:
                raise ValueError(f"scenario_path is neither a file nor directory: {scenario_path}")

            self.intruder_ids = self._infer_intruder_ids(scn_path=init_scenario)
            self.num_intruders = len(self.intruder_ids)
            self.ac_lat, self.ac_long = self._get_scen_lat_lon(scn_path=init_scenario)
        else:
            self.num_intruders = 0
            self.intruder_ids = []
            self.ac_lat, self.ac_long = 52., 4.

        self.current_scenario = None

        self.observation_space = spaces.Dict(
            {
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "cos_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "sin_difference": spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "waypoint_reached": spaces.Box(0, 1, shape=(NUM_WAYPOINTS,), dtype=np.float64),
                "altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "altitude_difference": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64),
                "z_difference_speed": spaces.Box(-np.inf, np.inf, shape=(self.num_intruders,), dtype=np.float64)
            }
        )
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize BlueSky — expect it to already be initialized in networked mode
        # by the calling script (bs.init(mode='sim', detached=False); bs.net.connect())
        if bs.sim is None:
            bs.init(mode="sim", detached=True)

        # Do NOT replace bs.scr with ScreenDummy — keep the default ScreenIO
        # so that aircraft data is published to the BlueSky GUI over ZMQ.
        # Only set DT (no FF) so the GUI can keep up.
        bs.stack.stack("DT 1")

        # Logging variables
        self.total_reward = 0
        self.final_altitude = 0
        self.total_intrusions = 0

        # Episode state
        self.reached = False
        self.landed = False

        # Initialize observation variables
        self.altitude = 0.0
        self.vz = 0.0
        self.ac_hdg = 0.0
        self.wpt_dis = 0.0
        self.wpt_qdr = []
        self.wpt_reach = [0]
        self.wpt_cos = 0.0
        self.wpt_sin = 0.0
        self.drift = 0.0
        self.target_alt = 0.0
        self.wpt_lat = 0.0
        self.wpt_lon = 0.0

        self.prev_hdg = None

    # =========================
    # Scenario loading
    # =========================

    def _read_scn_commands(self, scn_path: str):
        cmds = []
        for raw in Path(scn_path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ">" not in line:
                continue
            _, cmd = line.split(">", 1)
            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd.startswith("DEL "):
                continue
            cmds.append(cmd)
        return cmds

    def _get_scen_lat_lon(self, scn_path: str):
        cmds = self._read_scn_commands(scn_path)
        for cmd in cmds:
            cmd_split = cmd.split()
            if cmd_split[0] == "PAN":
                lat = cmd_split[1].split(',')[0]
                lon = cmd_split[1].split(',')[1]
                return float(lat), float(lon)
        print("Scenario file doesn't pan to lat/long")

    def _load_scenario(self, scn_path: str):
        for cmd in self._read_scn_commands(scn_path):
            bs.stack.stack(cmd)

    def _infer_intruder_ids(self, scn_path: str):
        ids = []
        for cmd in self._read_scn_commands(scn_path):
            if cmd.startswith("CRE "):
                parts = cmd.split()
                if len(parts) >= 2:
                    acid = parts[1].split(",")[0].strip()
                    ids.append(acid)
        return ids

    # =========================
    # Observation
    # =========================

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        self.ac_hdg = bs.traf.hdg[ac_idx]

        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
            self.wpt_lat, self.wpt_lon
        )

        self.wpt_dis = wpt_dis * NM2KM
        self.wpt_qdr = [wpt_qdr]

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.wpt_cos = np.cos(np.deg2rad(drift))
        self.wpt_sin = np.sin(np.deg2rad(drift))
        self.drift = drift

        self.vz = bs.traf.vs[0]
        self.altitude = bs.traf.alt[0]

        obs_altitude = np.array([(self.altitude - ALT_MEAN) / ALT_STD])
        obs_target_alt = np.array([(self.target_alt - ALT_MEAN) / ALT_STD])

        wpt_reach_arr = np.array(self.wpt_reach, dtype=np.float64)
        mask = (wpt_reach_arr - 1) * -1

        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.altitude_difference = []
        self.x_difference_speed = []
        self.y_difference_speed = []
        self.z_difference_speed = []

        intruder_ids = self.intruder_ids[:self.num_intruders]

        for acid in intruder_ids:
            int_idx = bs.traf.id2idx(acid)

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )

            self.intruder_distance.append(int_dis * NM2KM)

            alt_dif = bs.traf.alt[int_idx] - self.altitude
            vz_dif = bs.traf.vs[int_idx] - self.vz

            self.altitude_difference.append(alt_dif)
            self.z_difference_speed.append(vz_dif)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = -np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)

        obs = {
            "waypoint_distance": mask * np.array([self.wpt_dis]) / WAYPOINT_DISTANCE_MAX,
            "cos_difference": mask * np.array([self.wpt_cos]),
            "sin_difference": mask * np.array([self.wpt_sin]),
            "waypoint_reached": wpt_reach_arr,
            "altitude": obs_altitude,
            "target_altitude": obs_target_alt,
            "vz": np.array([(self.vz - VZ_MEAN) / VZ_STD]),
            "intruder_distance": np.array(self.intruder_distance) / WAYPOINT_DISTANCE_MAX,
            "cos_difference_pos": np.array(self.cos_bearing),
            "sin_difference_pos": np.array(self.sin_bearing),
            "altitude_difference": np.array(self.altitude_difference) / ALT_STD,
            "x_difference_speed": np.array(self.x_difference_speed) / AC_SPD,
            "y_difference_speed": np.array(self.y_difference_speed) / AC_SPD,
            "z_difference_speed": np.array(self.z_difference_speed)
        }

        return obs

    def _get_info(self):
        return {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "final_altitude": self.final_altitude,
            "scenario_file": self.current_scenario
        }

    # =========================
    # Reward
    # =========================

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0

        intruder_ids = self.intruder_ids[:self.num_intruders]

        for acid in intruder_ids:
            int_idx = bs.traf.id2idx(acid)
            _, int_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )

            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY

        return reward

    def _get_reward(self):
        d = self.wpt_dis

        if self.altitude <= 0:
            reward = CRASH_PENALTY
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

        alpha = np.clip(d / WAYPOINT_DISTANCE_MAX, 0.0, 1.0)

        desired_alt = alpha * self.target_alt

        alt_error = alpha * abs(self.altitude - self.target_alt)
        altitude_penalty = ALT_DIF_REWARD_SCALE * alt_error

        distance_penalty = -0.02 * d

        vz_penalty = -0.01 * abs(self.vz)

        if self.prev_hdg is not None:
            d_hdg = fn.bound_angle_positive_negative_180(self.ac_hdg - self.prev_hdg)
            hdg_penalty = -0.001 * abs(d_hdg) * alpha
        else:
            hdg_penalty = 0.0

        intruder_penalty = self._check_intrusion()

        if d <= DISTANCE_MARGIN and not self.reached:
            self.reached = True
            self.wpt_reach = [1]

        if self.reached and self.altitude <= 100:
            reward = REACH_REWARD
            self.final_altitude = self.altitude
            self.total_reward += reward
            self.landed = True
            return reward, True

        if self.reached and self.altitude > 100:
            reward = RWY_ALT_DIF_REWARD_SCALE * self.altitude
            self.final_altitude = self.altitude
            self.total_reward += reward
            return reward, True

        reward = altitude_penalty + distance_penalty + vz_penalty + hdg_penalty + intruder_penalty
        self.total_reward += reward

        return reward, False

    # =========================
    # Action
    # =========================

    def _get_action(self, act):
        alt_action = act[0] * ACTION_2_MS

        if alt_action >= 0:
            bs.traf.selalt[0] = 1000000
            bs.traf.selvs[0] = alt_action
        else:
            bs.traf.selalt[0] = 0
            bs.traf.selvs[0] = alt_action

        self.prev_hdg = self.ac_hdg

        hdg_action = (self.ac_hdg + act[1] * D_HEADING) % 360
        bs.stack.stack(f"HDG KL001 {hdg_action}")

    # =========================
    # BlueSky GUI sync
    # =========================

    def _sync_bluesky_gui(self):
        """Sync simulation state to the BlueSky GUI.

        The normal bs.sim.run() loop calls these three functions each iteration,
        but bs.sim.step() does not. We call them here so that the ScreenIO
        publishes ACDATA to the GUI and processes any incoming network messages.
        A small sleep throttles the sim so the GUI can visually follow along.
        """
        Timer.update_timers()
        bs.net.update()
        bs.scr.update()
        time.sleep(GUI_STEP_DELAY)

    # =========================
    # Reset
    # =========================

    def _generate_waypoint(self, acid="KL001"):
        wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
        wpt_hdg_init = np.random.randint(0, 359)

        ac_idx = bs.traf.id2idx(acid)

        self.wpt_lat, self.wpt_lon = fn.get_point_at_distance(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init
        )
        self.wpt_reach = [0]
        self.wpt_qdr = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0
        self.final_altitude = 0
        self.total_intrusions = 0
        self.reached = False
        self.landed = False

        if self.scenario_files:
            self.current_scenario = random.choice(self.scenario_files)
            self.ac_lat, self.ac_long = self._get_scen_lat_lon(scn_path=self.current_scenario)

        alt_init = np.random.randint(ALT_MIN, ALT_MAX)
        self.target_alt = alt_init + np.random.randint(-TARGET_ALT_DIF, TARGET_ALT_DIF)

        bs.traf.cre(
            "KL001", actype="A320", acalt=alt_init, acspd=AC_SPD,
            aclat=self.ac_lat + random.sample([-1, 1], 1)[0] * random.randint(25, 75) / 100,
            aclon=self.ac_long + random.sample([-1, 1], 1)[0] * random.randint(25, 75) / 100
        )
        bs.traf.swvnav[0] = False

        # Color the RL agent aircraft green so it stands out in the GUI
        if self.render_mode == "bluesky":
            bs.scr.custacclr['KL001'] = AGENT_COLOR

        if self.current_scenario is not None:
            self._load_scenario(self.current_scenario)
            self.intruder_ids = self._infer_intruder_ids(self.current_scenario)
        else:
            self.intruder_ids = []

        self._generate_waypoint()

        # Pan the GUI to the scenario location and let it catch up
        if self.render_mode == "bluesky":
            bs.stack.stack(f"PAN {self.ac_lat},{self.ac_long}")
            bs.stack.stack("ZOOM 0.4")
            # Process the PAN/ZOOM commands and publish initial aircraft data
            bs.sim.step()
            for _ in range(5):
                self._sync_bluesky_gui()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # =========================
    # Step
    # =========================

    def step(self, action):
        self._get_action(action)

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()

            if self.render_mode == "bluesky":
                self._sync_bluesky_gui()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()

        if terminated:
            for acid in bs.traf.id:
                try:
                    idx = bs.traf.id2idx(acid)
                    bs.traf.delete(idx)
                except (ValueError, AttributeError):
                    pass

        return obs, reward, terminated, False, info

    def close(self):
        bs.stack.stack("HOLD")
