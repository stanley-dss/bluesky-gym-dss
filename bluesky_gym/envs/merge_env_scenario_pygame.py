"""
Scenario-based Merge environment with pygame rendering.

Intruder aircraft positions and headings are taken from .scn scenario files
(scenarios_kord_merge_standard/), but their speeds and altitudes are overridden
to AC_SPD / AC_ALT so that the observation distribution exactly matches the
training setup of MergeEnv-v0 (all aircraft at 100 kt, FL100).

Use this env to:
  - Visually verify that a trained MergeEnv-v0 model behaves correctly
    on KORD-style scenario traffic
  - Run offline evaluations with scenario-based intruder positions

WHY THE BLUESKY VERSION LOOKS WRONG
------------------------------------
MergeEnv-Bluesky-v0 keeps the scenario's original 320 kt intruder speeds.
The trained model expects intruders at ~100 kt (training distribution), so the
relative-velocity features (vx_r, vy_r) are 2-4x larger than anything seen
during training, causing erratic behaviour.  This env fixes that by resetting
every intruder to AC_SPD before the first observation is computed.
"""
from pathlib import Path
import glob
import random
import re

import numpy as np
import pygame

import bluesky as bs
from bluesky.core import simtime as _simtime
from bluesky.stack.stackbase import Stack
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces


# =====================================================================
# Constants  — must match MergeEnv-v0 training values exactly
# =====================================================================

DISTANCE_MARGIN    = 10     # NM – waypoint capture radius (kwikqdrdist returns NM)
REACH_REWARD       = 1
DRIFT_PENALTY      = -0.1
INTRUSION_PENALTY  = -1
INTRUSION_DISTANCE = 4      # NM

SPAWN_DISTANCE_MIN = 80     # km from MERGE (inside KORD funnel range)
SPAWN_DISTANCE_MAX = 160

FUNNEL_AXIS = 270           # degrees – ownship approaches from the west
D_HEADING   = 15
D_SPEED     = 20

# All aircraft forced to these values to match training distribution
AC_SPD = 100   # knots
AC_ALT = 10000 # feet

NM2KM    = 1.852
MpS2Kt   = 1.94384

ACTION_FREQUENCY = 10
NUM_AC_STATE     = 5

# Pygame display
MAX_DISPLAY_KM = 600   # half-width of display in km; covers full funnel + KORD


# =====================================================================
# Environment
# =====================================================================

class MergeEnvScenarioPygame(gym.Env):
    """
    Merge environment that reads intruder positions from scenario files
    and renders with pygame.  All aircraft are reset to AC_SPD / AC_ALT
    so the observations match the MergeEnv-v0 training distribution.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, scenario_path=None):
        self.window_width  = 750
        self.window_height = 500
        self.window_size   = (self.window_width, self.window_height)

        # --- Observation / action spaces (identical to MergeEnv-v0) ----
        self.observation_space = spaces.Dict({
            "cos(drift)":    spaces.Box(-1, 1,           shape=(1,),           dtype=np.float64),
            "sin(drift)":    spaces.Box(-1, 1,           shape=(1,),           dtype=np.float64),
            "airspeed":      spaces.Box(-np.inf, np.inf, shape=(1,),           dtype=np.float64),
            "waypoint_dist": spaces.Box(-np.inf, np.inf, shape=(1,),           dtype=np.float64),
            "faf_reached":   spaces.Box(0, 1,            shape=(1,),           dtype=np.float64),
            "x_r":           spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "y_r":           spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "vx_r":          spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "vy_r":          spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "cos(track)":    spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "sin(track)":    spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
            "distances":     spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
        })
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # --- Scenario files ---------------------------------------------
        self.scenario_files = []
        if scenario_path is not None:
            p = Path(scenario_path)
            if p.is_dir():
                self.scenario_files = sorted(glob.glob(str(p / "*.scn")))
                if not self.scenario_files:
                    raise ValueError(f"No .scn files found in: {scenario_path}")
            elif p.is_file():
                self.scenario_files = [str(p)]
            else:
                raise ValueError(f"scenario_path not found: {scenario_path}")

        self.current_scenario = None

        # MERGE point (FAF) and destination (KORD)
        self.wpt_lat = None
        self.wpt_lon = None
        self.rwy_lat = None
        self.rwy_lon = None

        # --- BlueSky (headless detached mode, same as training) ---------
        if bs.sim is None:
            bs.init(mode='sim', detached=True)
        bs.scr = ScreenDummy()
        # DT=1 matches the training distribution for MergeEnv-Bluesky-v0.
        # FF so the headless sim still runs as fast as possible.
        bs.stack.stack('DT 1;FF')

        # --- Episode state ----------------------------------------------
        self.wpt_reach        = 0
        self.total_reward     = 0.0
        self.average_drift    = []
        self.total_intrusions = 0
        self.faf_reached      = 0
        self.drift            = 0.0
        self.waypoint_dist    = 0.0

        # Pygame
        self.window = None
        self.clock  = None

        # Funnel guide lines parsed from the scenario file
        self.funnel_lines = []   # list of (name, lat1, lon1, lat2, lon2)

    # =================================================================
    # Scenario helpers
    # =================================================================

    def _read_scn_commands(self, scn_path: str):
        """Return stripped command strings (timing prefix removed)."""
        cmds = []
        for raw in Path(scn_path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ">" not in line:
                continue
            _, cmd = line.split(">", 1)
            cmd = cmd.strip()
            if cmd:
                cmds.append(cmd)
        return cmds

    def _parse_merge_coords(self, scn_path: str):
        """Extract MERGE and KORD coordinates from scenario comments."""
        text = Path(scn_path).read_text()
        merge_lat = merge_lon = dest_lat = dest_lon = None

        m = re.search(r'#\s*MERGE\s*=\s*\(([0-9.-]+),\s*([0-9.-]+)\)', text)
        if m:
            merge_lat, merge_lon = float(m.group(1)), float(m.group(2))

        m2 = re.search(r'#\s*KORD\s*=\s*\(([0-9.-]+),\s*([0-9.-]+)\)', text)
        if m2:
            dest_lat, dest_lon = float(m2.group(1)), float(m2.group(2))

        # Fallback: first two ADDWPT pairs
        if merge_lat is None or dest_lat is None:
            addwpts = re.findall(r'ADDWPT\s+([0-9.-]+),\s*([0-9.-]+)', text)
            if len(addwpts) >= 2:
                if merge_lat is None:
                    merge_lat, merge_lon = float(addwpts[0][0]), float(addwpts[0][1])
                if dest_lat is None:
                    dest_lat, dest_lon  = float(addwpts[1][0]), float(addwpts[1][1])

        return merge_lat, merge_lon, dest_lat, dest_lon

    def _parse_intruder_init(self, scn_path: str):
        """
        Parse intruder initial conditions from CRE commands.
        Returns list of (acid, ac_type, lat, lon, hdg).
        Speed and altitude from the scenario are intentionally ignored;
        they are set to AC_SPD / AC_ALT during _load_scenario_normalized.
        """
        intruders = []
        for cmd in self._read_scn_commands(scn_path):
            upper = cmd.upper()
            if not upper.startswith("CRE "):
                continue
            rest = cmd[4:]  # strip "CRE "
            fields = rest.split(",")
            if len(fields) < 5:
                continue
            acid = fields[0].strip()
            if acid == "KL001":
                continue
            ac_type = fields[1].strip()
            try:
                lat = float(fields[2])
                lon = float(fields[3])
                hdg = float(fields[4])
            except ValueError:
                continue
            intruders.append((acid, ac_type, lat, lon, hdg))
        return intruders

    def _parse_funnel_lines(self, scn_path: str):
        """Parse LINE commands from a scenario file.
        Returns list of (name, lat1, lon1, lat2, lon2).
        """
        result = []
        for cmd in self._read_scn_commands(scn_path):
            if not cmd.upper().startswith("LINE "):
                continue
            parts = cmd[5:].split(",")   # strip "LINE "
            if len(parts) < 5:
                continue
            name = parts[0].strip()
            try:
                lat1, lon1 = float(parts[1]), float(parts[2])
                lat2, lon2 = float(parts[3]), float(parts[4])
            except ValueError:
                continue
            result.append((name, lat1, lon1, lat2, lon2))
        return result

    def _load_scenario_timed(self, scn_path: str):
        """Schedule scenario commands into BlueSky's timed queue.

        Mirrors MergeEnvBluesky._load_scenario exactly so that the wave
        timing and aircraft parameters are identical between the two envs.
        Requires bs.sim.simt to be reset to 0 before this call (done in reset()).
        """
        from bluesky.stack.simstack import readscn
        t_offset = bs.sim.simt  # 0.0 after simt reset in reset()
        scn_path = str(Path(scn_path).resolve())
        for cmdtime, cmdline in readscn(scn_path):
            upper = cmdline.upper()
            if upper.startswith("DEL ") or upper.startswith("RTF"):
                continue
            if upper.startswith("CRE "):
                parts = cmdline[4:].split(",")
                if len(parts) >= 6 and parts[0].strip().upper() != "KL001":
                    parts[5] = str(AC_ALT)
                    if len(parts) >= 7:
                        parts[6] = str(AC_SPD)
                    cmdline = "CRE " + ",".join(parts)
            tokens = upper.split()
            if (len(tokens) >= 2 and tokens[1] in ("SPD", "ALT")
                    and tokens[0] != "KL001"):
                continue
            scheduled_time = t_offset + cmdtime
            ins = next(
                (i for i, t in enumerate(Stack.scentime) if t >= scheduled_time),
                len(Stack.scentime),
            )
            Stack.scentime.insert(ins, scheduled_time)
            Stack.scencmd.insert(ins, cmdline)

    def _load_scenario_normalized(self, scn_path: str):
        """
        Create intruder aircraft at their scenario-defined positions/headings
        but at AC_SPD and AC_ALT to match the MergeEnv-v0 training distribution.
        Each intruder is given waypoints: MERGE → KORD.
        """
        intruders = self._parse_intruder_init(scn_path)
        for acid, ac_type, lat, lon, hdg in intruders:
            bs.traf.cre(
                acid, actype=ac_type,
                aclat=lat, aclon=lon, achdg=int(hdg),
                acalt=AC_ALT, acspd=AC_SPD
            )
            bs.stack.stack(f"{acid} ADDWPT {self.wpt_lat:.6f},{self.wpt_lon:.6f}")
            bs.stack.stack(f"{acid} ADDWPT {self.rwy_lat:.6f},{self.rwy_lon:.6f}")
        bs.stack.stack("reso off")

    # =================================================================
    # Observation
    # =================================================================

    def _get_obs(self):
        ac_idx = bs.traf.id2idx("KL001")
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        ac_hdg = bs.traf.hdg[ac_idx]

        # Bearing and distance to current waypoint
        if self.wpt_reach == 0:
            wpt_qdr, wpt_dist = bs.tools.geo.kwikqdrdist(ac_lat, ac_lon, self.wpt_lat, self.wpt_lon)
        else:
            wpt_qdr, wpt_dist = bs.tools.geo.kwikqdrdist(ac_lat, ac_lon, self.rwy_lat, self.rwy_lon)

        drift = fn.bound_angle_positive_negative_180(ac_hdg - wpt_qdr)
        self.drift         = drift
        self.waypoint_dist = wpt_dist

        cos_drift = np.array([np.cos(np.deg2rad(drift))])
        sin_drift = np.array([np.sin(np.deg2rad(drift))])
        airspeed  = np.array([bs.traf.tas[ac_idx]])

        vx_own = np.cos(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
        vy_own = np.sin(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]

        # Collect all other aircraft
        other_lats, other_lons, other_hdgs, other_tas = [], [], [], []
        for acid in bs.traf.id:
            if acid == "KL001":
                continue
            idx = bs.traf.id2idx(acid)
            other_lats.append(bs.traf.lat[idx])
            other_lons.append(bs.traf.lon[idx])
            other_hdgs.append(bs.traf.hdg[idx])
            other_tas.append(bs.traf.tas[idx])

        n_others = len(other_lats)

        x_r_out       = np.zeros(NUM_AC_STATE)
        y_r_out       = np.zeros(NUM_AC_STATE)
        vx_r_out      = np.zeros(NUM_AC_STATE)
        vy_r_out      = np.zeros(NUM_AC_STATE)
        cos_track_out = np.zeros(NUM_AC_STATE)
        sin_track_out = np.zeros(NUM_AC_STATE)
        dist_out      = np.full(NUM_AC_STATE, 9999.0)

        if n_others > 0:
            lats_arr = np.array(other_lats)
            lons_arr = np.array(other_lons)
            hdgs_arr = np.array(other_hdgs)
            tas_arr  = np.array(other_tas)

            dists_nm = np.ravel(
                bs.tools.geo.kwikdist_matrix(ac_lat, ac_lon, lats_arr, lons_arr)
            )
            sorted_idx = np.argsort(dists_nm)
            n_use = min(n_others, NUM_AC_STATE)

            for slot, si in enumerate(sorted_idx[:n_use]):
                si = int(si)
                brg, dist_nm = bs.tools.geo.kwikqdrdist(
                    ac_lat, ac_lon, lats_arr[si], lons_arr[si]
                )
                dist_out[slot] = dists_nm[si]

                x_r_out[slot] = (dist_nm * NM2KM * 1000) * np.cos(np.deg2rad(brg))
                y_r_out[slot] = (dist_nm * NM2KM * 1000) * np.sin(np.deg2rad(brg))

                vx_int = np.cos(np.deg2rad(hdgs_arr[si])) * tas_arr[si]
                vy_int = np.sin(np.deg2rad(hdgs_arr[si])) * tas_arr[si]
                vx_r_out[slot] = vx_int - vx_own
                vy_r_out[slot] = vy_int - vy_own

                track = np.arctan2(vy_int - vy_own, vx_int - vx_own)
                cos_track_out[slot] = np.cos(track)
                sin_track_out[slot] = np.sin(track)

        return {
            "cos(drift)":    cos_drift,
            "sin(drift)":    sin_drift,
            "airspeed":      airspeed,
            "waypoint_dist": np.array([wpt_dist / 250.0]),
            "faf_reached":   np.array([float(self.wpt_reach)]),
            "x_r":           x_r_out   / 1_000_000,
            "y_r":           y_r_out   / 1_000_000,
            "vx_r":          vx_r_out  / 150.0,
            "vy_r":          vy_r_out  / 150.0,
            "cos(track)":    cos_track_out,
            "sin(track)":    sin_track_out,
            "distances":     dist_out  / 250.0,
        }

    def _get_info(self):
        return {
            "total_reward":     self.total_reward,
            "faf_reach":        self.faf_reached,
            "average_drift":    float(np.mean(self.average_drift)) if self.average_drift else 0.0,
            "total_intrusions": self.total_intrusions,
            "scenario_file":    self.current_scenario,
        }

    # =================================================================
    # Reward
    # =================================================================

    def _get_reward(self):
        reach_reward, done = self._check_waypoint()
        drift_reward       = self._check_drift()
        intrusion_reward   = self._check_intrusion()
        reward = reach_reward + drift_reward + intrusion_reward
        self.total_reward += reward
        return reward, done

    def _check_waypoint(self):
        reward, done = 0, 0
        if self.waypoint_dist < DISTANCE_MARGIN and self.wpt_reach != 1:
            self.wpt_reach   = 1
            self.faf_reached = 1
            reward += REACH_REWARD
        elif self.waypoint_dist < 2 * DISTANCE_MARGIN and self.wpt_reach == 1:
            self.faf_reached = 2
            done = 1
        return reward, done

    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift))
        self.average_drift.append(drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx("KL001")
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        reward = 0.0
        for acid in bs.traf.id:
            if acid == "KL001":
                continue
            int_idx = bs.traf.id2idx(acid)
            _, int_dis = bs.tools.geo.kwikqdrdist(
                ac_lat, ac_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        return reward

    # =================================================================
    # Action
    # =================================================================

    def _get_action(self, action):
        ac_idx = bs.traf.id2idx("KL001")
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] + dh)
        speed_new   = (bs.traf.cas[ac_idx] + dv) * MpS2Kt
        bs.stack.stack(f"HDG KL001 {heading_new}")
        bs.stack.stack(f"SPD KL001 {speed_new}")

    # =================================================================
    # Reset
    # =================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.wpt_reach        = 0
        self.total_reward     = 0.0
        self.average_drift    = []
        self.total_intrusions = 0
        self.faf_reached      = 0

        bs.traf.reset()

        # Reset sim time to 0 so wave-timed scenario commands schedule correctly,
        # identical to MergeEnvBluesky.reset().
        # simtime.reset() resets _clock.t so simt advances correctly from 0 in
        # every episode (not just the first). setdt(1.0) restores DT=1 because
        # simtime.reset() resets _clock.dt to the default 0.05 s.
        _simtime.reset()
        _simtime.setdt(1.0)
        bs.sim.simt = 0.0
        Stack.scentime.clear()
        Stack.scencmd.clear()

        # Choose scenario and extract coordinates
        if self.scenario_files:
            self.current_scenario = random.choice(self.scenario_files)
            self.wpt_lat, self.wpt_lon, self.rwy_lat, self.rwy_lon = \
                self._parse_merge_coords(self.current_scenario)
        else:
            self.wpt_lat, self.wpt_lon = 41.9742, -90.5000   # MERGE
            self.rwy_lat, self.rwy_lon = 41.9742, -87.9073   # KORD

        # Spawn ownship west of MERGE, heading east
        bearing_to_merge = random.uniform(FUNNEL_AXIS - D_HEADING, FUNNEL_AXIS + D_HEADING)
        dist_to_merge    = random.uniform(SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX)
        rlat, rlon = fn.get_point_at_distance(
            self.wpt_lat, self.wpt_lon, dist_to_merge, bearing_to_merge
        )
        hdg_init = fn.bound_angle_positive_negative_180(bearing_to_merge - 180)

        bs.traf.cre(
            "KL001", actype="A320", acspd=AC_SPD,
            aclat=rlat, aclon=rlon, achdg=hdg_init, acalt=AC_ALT
        )
        own_idx = bs.traf.id2idx("KL001")
        bs.traf.swvnav[own_idx] = False

        # Load intruders using the same wave-timed approach as MergeEnvBluesky
        # so that intruder count and positions match exactly between the two envs.
        if self.current_scenario is not None:
            self._load_scenario_timed(self.current_scenario)
            self.funnel_lines = self._parse_funnel_lines(self.current_scenario)
        else:
            self.funnel_lines = []
        # Disable conflict resolution so intruders don't maneuver to avoid KL001.
        # The training env (MergeEnvBluesky) always queues reso off in reset().
        bs.stack.stack("reso off")

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    # =================================================================
    # Step
    # =================================================================

    def step(self, action):
        self._get_action(action)

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                self._get_obs()
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()

        return obs, reward, terminated, False, info

    # =================================================================
    # Pygame rendering  (centered on MERGE point, matching merge_env.py)
    # =================================================================

    def _render_frame(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("MergeEnv – scenario pygame")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 500  # km — matches merge_env.py

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))  # sky blue

        circle_x = self.window_width  / 2
        circle_y = self.window_height / 2

        # --- MERGE point (centre) — identical to merge_env.py ---
        pygame.draw.circle(canvas, (255, 255, 255),
                           (circle_x, circle_y), radius=4, width=0)
        pygame.draw.circle(canvas, (255, 255, 255),
                           (circle_x, circle_y),
                           radius=(DISTANCE_MARGIN / max_distance) * self.window_width,
                           width=2)

        # --- Approach axis lines (dynamic version of merge_env.py's hardcoded bearing=180) ---
        if self.rwy_lat is not None:
            rwy_faf_qdr, rwy_faf_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, self.rwy_lat, self.rwy_lon
            )

            # Position of KORD on screen (same formula as merge_env.py for RWY)
            x_kord = circle_x + (np.cos(np.deg2rad(rwy_faf_qdr)) * (rwy_faf_dis * NM2KM) / max_distance) * self.window_width
            y_kord = circle_y - (np.sin(np.deg2rad(rwy_faf_qdr)) * (rwy_faf_dis * NM2KM) / max_distance) * self.window_height

            heading_length = 5000  # km — same value as merge_env.py

            # Black centreline from MERGE toward KORD (merge_env.py draws toward bearing 180)
            he_x = ((np.cos(np.deg2rad(rwy_faf_qdr)) * heading_length) / max_distance) * self.window_width
            he_y = ((np.sin(np.deg2rad(rwy_faf_qdr)) * heading_length) / max_distance) * self.window_width
            pygame.draw.line(canvas, (0, 0, 0),
                             (circle_x, circle_y),
                             (circle_x + he_x / 2, circle_y - he_y / 2),
                             width=2)

            # Green funnel boundary lines at ±135° from KORD bearing
            # (merge_env.py uses 180±135; here we use rwy_faf_qdr±135 for the scenario geometry)
            he_x_l = ((np.cos(np.deg2rad(rwy_faf_qdr + 135)) * heading_length) / max_distance) * self.window_width
            he_y_l = ((np.sin(np.deg2rad(rwy_faf_qdr + 135)) * heading_length) / max_distance) * self.window_width
            pygame.draw.line(canvas, (3, 252, 11),
                             (circle_x, circle_y),
                             (circle_x + he_x_l / 2, circle_y - he_y_l / 2),
                             width=4)

            he_x_r = ((np.cos(np.deg2rad(rwy_faf_qdr - 135)) * heading_length) / max_distance) * self.window_width
            he_y_r = ((np.sin(np.deg2rad(rwy_faf_qdr - 135)) * heading_length) / max_distance) * self.window_width
            pygame.draw.line(canvas, (3, 252, 11),
                             (circle_x, circle_y),
                             (circle_x + he_x_r / 2, circle_y - he_y_r / 2),
                             width=4)

            # White approach line from KORD extending in KORD direction
            # (merge_env.py draws from RWY toward bearing 180 from centre)
            pygame.draw.line(canvas, (255, 255, 255),
                             (x_kord, y_kord),
                             (circle_x + he_x / 2, circle_y - he_y / 2),
                             width=4)

        # --- Ownship (KL001) — identical to merge_env.py ---
        ac_idx = bs.traf.id2idx("KL001")
        if ac_idx >= 0:
            ac_length = 8
            heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width
            heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length) / max_distance) * self.window_width

            own_qdr, own_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]
            )
            x_pos = circle_x + (np.cos(np.deg2rad(own_qdr)) * (own_dis * NM2KM) / max_distance) * self.window_width
            y_pos = circle_y - (np.sin(np.deg2rad(own_qdr)) * (own_dis * NM2KM) / max_distance) * self.window_height

            pygame.draw.line(canvas, (0, 0, 0),
                             (x_pos, y_pos),
                             (x_pos + heading_end_x / 2, y_pos - heading_end_y / 2),
                             width=4)

            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length) / max_distance) * self.window_width
            heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length) / max_distance) * self.window_width
            pygame.draw.line(canvas, (0, 0, 0),
                             (x_pos, y_pos),
                             (x_pos + heading_end_x, y_pos - heading_end_y),
                             width=1)

        # --- Intruder aircraft — identical to merge_env.py ---
        # Color is based on distance from MERGE (same as merge_env.py)
        ac_length = 3
        for acid in bs.traf.id:
            if acid == "KL001":
                continue
            idx = bs.traf.id2idx(acid)
            int_hdg = bs.traf.hdg[idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length) / max_distance) * self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length) / max_distance) * self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, bs.traf.lat[idx], bs.traf.lon[idx]
            )
            color = (220, 20, 60) if int_dis < INTRUSION_DISTANCE else (80, 80, 80)

            x_pos = circle_x + (np.cos(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_width
            y_pos = circle_y - (np.sin(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_height

            pygame.draw.line(canvas, color,
                             (x_pos, y_pos),
                             (x_pos + heading_end_x, y_pos - heading_end_y),
                             width=4)

            heading_length = 10
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length) / max_distance) * self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length) / max_distance) * self.window_width
            pygame.draw.line(canvas, color,
                             (x_pos, y_pos),
                             (x_pos + heading_end_x, y_pos - heading_end_y),
                             width=1)

            pygame.draw.circle(canvas, color, (x_pos, y_pos),
                               radius=(INTRUSION_DISTANCE * NM2KM / max_distance) * self.window_width,
                               width=2)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        bs.stack.stack("quit")
