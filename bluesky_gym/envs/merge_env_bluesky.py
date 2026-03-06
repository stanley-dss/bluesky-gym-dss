"""
BlueSky GUI variant of the Merge environment.

Identical reward/observation logic to merge_env.py but:
  - renders in the BlueSky QtGL GUI instead of pygame
  - loads intruder traffic from .scn scenario files (e.g. scenarios_kord_merge_standard/)
  - handles a dynamic number of intruders (pads observation to NUM_AC_STATE)

Usage (same bootstrap as DescentEnvCR3DBluesky):
    bs.init(mode='sim', detached=False)
    bs.net.connect()
    env = gym.make(
        'MergeEnv-Bluesky-v0',
        render_mode='bluesky',
        scenario_path='scenarios_kord_merge_standard/',
    )
"""
from pathlib import Path
import glob
import random
import re
import time

import numpy as np
import pygame
import bluesky as bs
import bluesky_gym.envs.common.functions as fn
from bluesky.core.walltime import Timer
from bluesky.core import simtime as _simtime
from bluesky.stack.stackbase import Stack

import gymnasium as gym
from gymnasium import spaces

# =====================================================================
# Constants  (matching merge_env.py)
# =====================================================================

DISTANCE_MARGIN   = 10     # km  – waypoint capture radius
REACH_REWARD      = 1
DRIFT_PENALTY     = -0.1
INTRUSION_PENALTY = -1
INTRUSION_DISTANCE = 4     # NM – conflict detection radius

# Ownship spawn distance from MERGE point (km)
SPAWN_DISTANCE_MIN = 80
SPAWN_DISTANCE_MAX = 160

# Bearing spread around funnel axis (±deg) for ownship spawn
D_HEADING = 15
D_SPEED   = 20

# Funnel approach axis (westward from MERGE, aircraft fly east toward MERGE)
FUNNEL_AXIS = 270

AC_SPD   = 100    # knots — must match training distribution
AC_ALT   = 10000  # feet  — must match training distribution
NM2KM    = 1.852
MpS2Kt   = 1.94384

ACTION_FREQUENCY = 10
NUM_AC_STATE     = 5   # intruder slots in observation

# Agent aircraft color in BlueSky GUI  (red)
AGENT_COLOR = (255, 0, 0)

# Wall-clock sleep per sim step so the GUI can keep up
GUI_STEP_DELAY = 0.05


# =====================================================================
# Environment
# =====================================================================

class MergeEnvBluesky(gym.Env):
    """
    Merge environment rendered in the BlueSky QtGL GUI.
    Intruder traffic is loaded from .scn scenario files.
    """

    metadata = {"render_modes": ["bluesky", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, scenario_path=None, verbose=False):

        # --- Observation / action spaces (identical to MergeEnv) --------
        self.observation_space = spaces.Dict({
            "cos(drift)":    spaces.Box(-1, 1,          shape=(1,),          dtype=np.float64),
            "sin(drift)":    spaces.Box(-1, 1,          shape=(1,),          dtype=np.float64),
            "airspeed":      spaces.Box(-np.inf, np.inf, shape=(1,),          dtype=np.float64),
            "waypoint_dist": spaces.Box(-np.inf, np.inf, shape=(1,),          dtype=np.float64),
            "faf_reached":   spaces.Box(0, 1,           shape=(1,),          dtype=np.float64),
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

        # MERGE point (FAF) and destination (runway / KORD)
        self.wpt_lat = None
        self.wpt_lon = None
        self.rwy_lat = None
        self.rwy_lon = None

        # --- BlueSky init -----------------------------------------------
        # Expect the caller to have already run bs.init(mode='sim', detached=False)
        # and bs.net.connect().  Fall back to detached mode otherwise.
        if bs.sim is None:
            print("Fallback to detached sim")
            bs.init(mode="sim", detached=True)

        # DT=1 matches the training distribution for MergeEnv-Bluesky-v0.
        # We do NOT use FF here (or call bs.traf.reset() in reset()) because in
        # networked mode the server notifies the GUI of any traf.reset(), and the
        # GUI sends back a HOLD command that races with our FF re-queue and wins,
        # leaving the sim paused.  Instead we delete aircraft individually
        # (see reset()), exactly like DescentEnvCR3DBluesky, so the sim never
        # leaves its running state.
        bs.stack.stack("DT 1")

        self.verbose = verbose

        # --- Episode state ---------------------------------------------
        self.wpt_reach       = 0
        self.total_reward    = 0.0
        self.average_drift   = []
        self.total_intrusions = 0
        self.faf_reached     = 0

        # Cached observation intermediates
        self.drift        = 0.0
        self.waypoint_dist = 0.0

        # --- Pygame overlay (shown alongside the BlueSky GUI) ----------
        self.window_width  = 750
        self.window_height = 500
        self.window_size   = (self.window_width, self.window_height)
        self.window        = None
        self.clock         = None

    # =================================================================
    # Scenario helpers
    # =================================================================

    def _parse_merge_coords(self, scn_path: str):
        """
        Extract MERGE (FAF) and destination (KORD) coordinates from scenario.
        Looks for comment lines:
            # MERGE = (lat, lon)
            # KORD  = (lat, lon)
        Falls back to the first two ADDWPT coordinate pairs if comments absent.
        """
        text = Path(scn_path).read_text()
        merge_lat = merge_lon = dest_lat = dest_lon = None

        m = re.search(r'#\s*MERGE\s*=\s*\(([0-9.-]+),\s*([0-9.-]+)\)', text)
        if m:
            merge_lat, merge_lon = float(m.group(1)), float(m.group(2))

        m2 = re.search(r'#\s*KORD\s*=\s*\(([0-9.-]+),\s*([0-9.-]+)\)', text)
        if m2:
            dest_lat, dest_lon = float(m2.group(1)), float(m2.group(2))

        # Fallback: first two ADDWPT pairs in the file
        if merge_lat is None or dest_lat is None:
            addwpts = re.findall(r'ADDWPT\s+([0-9.-]+),\s*([0-9.-]+)', text)
            if len(addwpts) >= 2:
                if merge_lat is None:
                    merge_lat, merge_lon = float(addwpts[0][0]), float(addwpts[0][1])
                if dest_lat is None:
                    dest_lat, dest_lon  = float(addwpts[1][0]), float(addwpts[1][1])

        return merge_lat, merge_lon, dest_lat, dest_lon

    def _load_scenario(self, scn_path: str):
        """Schedule scenario commands into BlueSky's timed queue.

        Preserves the wave timing from the scenario file so aircraft spawn
        gradually during the episode, matching what BlueSky GUI shows when
        loading the same .scn file directly.  Requires bs.sim.simt to have
        been reset to 0 before this call (done in reset()).
        """
        from bluesky.stack.simstack import readscn
        t_offset = bs.sim.simt  # 0.0 after reset clears simt
        # Resolve to absolute path so readscn doesn't prepend BlueSky's scenario dir
        scn_path = str(Path(scn_path).resolve())
        for cmdtime, cmdline in readscn(scn_path):
            upper = cmdline.upper()
            if upper.startswith("DEL ") or upper.startswith("RTF"):
                continue
            # Normalize intruder CRE speeds/altitudes to match training distribution.
            # Scenario files use ~320 kt; the model expects AC_SPD=100 kt.
            if upper.startswith("CRE "):
                parts = cmdline[4:].split(",")
                if len(parts) >= 6 and parts[0].strip().upper() != "KL001":
                    parts[5] = str(AC_ALT)          # altitude (ft)
                    if len(parts) >= 7:
                        parts[6] = str(AC_SPD)      # speed (kt)
                    cmdline = "CRE " + ",".join(parts)
            # Skip per-aircraft SPD/ALT overrides for non-KL001 intruders.
            # Scenario files follow each CRE with "ACID SPD 320" and "ACID ALT 16000"
            # which would override the normalization above.
            tokens = upper.split()
            if (len(tokens) >= 2 and tokens[1] in ("SPD", "ALT")
                    and tokens[0] != "KL001"):
                continue
            scheduled_time = t_offset + cmdtime
            # Insert at the correct sorted position in the scenario queue
            ins = next(
                (i for i, t in enumerate(Stack.scentime) if t >= scheduled_time),
                len(Stack.scentime),
            )
            Stack.scentime.insert(ins, scheduled_time)
            Stack.scencmd.insert(ins, cmdline)

    # =================================================================
    # Pygame rendering  (mirrors merge_env_scenario_pygame._render_frame)
    # =================================================================

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("MergeEnv – bluesky (pygame overlay)")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # drain pygame event queue so the window stays responsive
        for event in pygame.event.get():
            pass

        max_distance = 500  # km — matches merge_env.py

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))  # sky blue

        cx = self.window_width  / 2
        cy = self.window_height / 2

        # --- MERGE point (centre) ---
        pygame.draw.circle(canvas, (255, 255, 255), (cx, cy), radius=4, width=0)
        pygame.draw.circle(canvas, (255, 255, 255), (cx, cy),
                           radius=int((DISTANCE_MARGIN / max_distance) * self.window_width),
                           width=2)

        # --- Approach axis lines (MERGE → KORD) ---
        if self.rwy_lat is not None:
            rwy_qdr, rwy_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, self.rwy_lat, self.rwy_lon
            )
            x_kord = cx + (np.cos(np.deg2rad(rwy_qdr)) * (rwy_dis * NM2KM) / max_distance) * self.window_width
            y_kord = cy - (np.sin(np.deg2rad(rwy_qdr)) * (rwy_dis * NM2KM) / max_distance) * self.window_height

            hl = 5000  # km — same as merge_env.py
            he_x = ((np.cos(np.deg2rad(rwy_qdr)) * hl) / max_distance) * self.window_width
            he_y = ((np.sin(np.deg2rad(rwy_qdr)) * hl) / max_distance) * self.window_width

            # Black centreline MERGE → KORD direction
            pygame.draw.line(canvas, (0, 0, 0),
                             (cx, cy), (cx + he_x / 2, cy - he_y / 2), width=2)

            # Green funnel boundaries at ±135°
            for sign in (+1, -1):
                ang = rwy_qdr + sign * 135
                fx = ((np.cos(np.deg2rad(ang)) * hl) / max_distance) * self.window_width
                fy = ((np.sin(np.deg2rad(ang)) * hl) / max_distance) * self.window_width
                pygame.draw.line(canvas, (3, 252, 11),
                                 (cx, cy), (cx + fx / 2, cy - fy / 2), width=4)

            # White approach line from KORD to centreline
            pygame.draw.line(canvas, (255, 255, 255),
                             (x_kord, y_kord), (cx + he_x / 2, cy - he_y / 2), width=4)

        # --- Ownship (KL001) ---
        ac_idx = bs.traf.id2idx("KL001")
        if ac_idx >= 0:
            own_qdr, own_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]
            )
            x_pos = cx + (np.cos(np.deg2rad(own_qdr)) * (own_dis * NM2KM) / max_distance) * self.window_width
            y_pos = cy - (np.sin(np.deg2rad(own_qdr)) * (own_dis * NM2KM) / max_distance) * self.window_height

            hdg = bs.traf.hdg[ac_idx]
            for ac_len, lw in [(8, 4), (10, 1)]:
                hx = ((np.cos(np.deg2rad(hdg)) * ac_len) / max_distance) * self.window_width
                hy = ((np.sin(np.deg2rad(hdg)) * ac_len) / max_distance) * self.window_width
                end = (x_pos + hx / 2, y_pos - hy / 2) if lw == 4 else (x_pos + hx, y_pos - hy)
                pygame.draw.line(canvas, (0, 0, 0), (x_pos, y_pos), end, width=lw)

        # --- Intruders ---
        for acid in bs.traf.id:
            if acid == "KL001":
                continue
            idx = bs.traf.id2idx(acid)
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(
                self.wpt_lat, self.wpt_lon, bs.traf.lat[idx], bs.traf.lon[idx]
            )
            color = (220, 20, 60) if int_dis < INTRUSION_DISTANCE else (80, 80, 80)

            x_pos = cx + (np.cos(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_width
            y_pos = cy - (np.sin(np.deg2rad(int_qdr)) * (int_dis * NM2KM) / max_distance) * self.window_height

            int_hdg = bs.traf.hdg[idx]
            for ac_len, lw in [(3, 4), (10, 1)]:
                hx = ((np.cos(np.deg2rad(int_hdg)) * ac_len) / max_distance) * self.window_width
                hy = ((np.sin(np.deg2rad(int_hdg)) * ac_len) / max_distance) * self.window_width
                pygame.draw.line(canvas, color, (x_pos, y_pos), (x_pos + hx, y_pos - hy), width=lw)

            pygame.draw.circle(canvas, color, (x_pos, y_pos),
                               radius=int((INTRUSION_DISTANCE * NM2KM / max_distance) * self.window_width),
                               width=2)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    # =================================================================
    # Observation
    # =================================================================

    def _get_obs(self):
        ac_idx = bs.traf.id2idx("KL001")
        ac_lat = bs.traf.lat[ac_idx]
        ac_lon = bs.traf.lon[ac_idx]
        ac_hdg = bs.traf.hdg[ac_idx]

        # Drift toward current waypoint
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

        # Collect all other aircraft currently present in the simulation
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

        # Padded output arrays (zeros for unused slots)
        x_r_out       = np.zeros(NUM_AC_STATE)
        y_r_out       = np.zeros(NUM_AC_STATE)
        vx_r_out      = np.zeros(NUM_AC_STATE)
        vy_r_out      = np.zeros(NUM_AC_STATE)
        cos_track_out = np.zeros(NUM_AC_STATE)
        sin_track_out = np.zeros(NUM_AC_STATE)
        dist_out      = np.full(NUM_AC_STATE, 9999.0)

        if n_others > 0:
            hdgs_arr = np.array(other_hdgs)
            tas_arr  = np.array(other_tas)

            # Compute distances from ownship to each intruder using kwikqdrdist.
            # kwikdist_matrix is avoided because the compiled _cgeo version has a
            # different calling convention (all-vs-all pairwise) and returns (1,1)
            # when called with a scalar vs array, breaking distance computation.
            brgs_dists = [
                bs.tools.geo.kwikqdrdist(ac_lat, ac_lon, lat, lon)
                for lat, lon in zip(other_lats, other_lons)
            ]
            dists_nm = np.array([d[1] for d in brgs_dists])
            sorted_idx = np.argsort(dists_nm)
            n_use = min(n_others, NUM_AC_STATE)

            for slot, si in enumerate(sorted_idx[:n_use]):
                si = int(si)   # guarantee plain Python int for indexing
                brg, dist_nm = brgs_dists[si]
                dist_out[slot] = dist_nm

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
        reach_reward     = self._check_waypoint()
        drift_reward     = self._check_drift()
        intrusion_reward = self._check_intrusion()
        reward = reach_reward[0] + drift_reward + intrusion_reward
        self.total_reward += reward
        return reward, reach_reward[1]

    def _check_waypoint(self):
        reward, done = 0, 0
        if self.waypoint_dist < DISTANCE_MARGIN and self.wpt_reach != 1:
            self.wpt_reach  = 1
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
    # BlueSky GUI sync
    # =================================================================

    def _sync_bluesky_gui(self):
        """Publish state to the GUI and throttle wall-clock speed."""
        Timer.update_timers()
        bs.net.update()
        bs.scr.update()
        time.sleep(GUI_STEP_DELAY)

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

        # Delete all aircraft individually so the sim never enters HOLD state.
        # Calling bs.traf.reset() in networked mode (detached=False) causes the
        # server to notify the GUI, which responds with a HOLD command.  That HOLD
        # races with any FF re-queue we add, leaving the sim paused → actions
        # change heading but lat/lon never updates ("spinning in circles").
        # This matches the pattern used by DescentEnvCR3DBluesky.
        for acid in list(bs.traf.id):
            try:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)
            except (ValueError, AttributeError):
                pass

        # Reset sim time to 0 so wave-timed scenario commands schedule correctly.
        # simtime.reset() resets _clock.t (the internal Decimal accumulator) so
        # that simt advances correctly from 0 in every episode, not just the first.
        # setdt(1.0) restores DT=1 because simtime.reset() resets _clock.dt to
        # the default 0.05 s.
        _simtime.reset()
        _simtime.setdt(1.0)
        bs.sim.simt = 0.0
        Stack.scentime.clear()
        Stack.scencmd.clear()

        # Pick a scenario and extract merge/destination coordinates
        if self.scenario_files:
            self.current_scenario = random.choice(self.scenario_files)
            self.wpt_lat, self.wpt_lon, self.rwy_lat, self.rwy_lon = \
                self._parse_merge_coords(self.current_scenario)
        else:
            # Fallback: hardcoded KORD merge/runway coordinates
            self.wpt_lat, self.wpt_lon = 41.9742, -90.5000   # MERGE
            self.rwy_lat, self.rwy_lon = 41.9742, -87.9073   # KORD

        # Spawn ownship west of the MERGE point (inside the funnel)
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

        # Color ownship red so it stands out in the GUI
        if self.render_mode == "bluesky":
            bs.scr.custacclr["KL001"] = AGENT_COLOR

        # Load intruder traffic from scenario (timed CRE commands auto-schedule)
        if self.current_scenario is not None:
            self._load_scenario(self.current_scenario)
        bs.stack.stack("reso off")

        # Capture the initial observation at simt=0 BEFORE any sim step,
        # matching the training distribution (render_mode=None never steps here).
        obs  = self._get_obs()
        info = self._get_info()
        if self.verbose:
            self._debug_obs("RESET", obs)

        # Pan GUI to the MERGE area and flush pending stack commands for display.
        # The sim step here is for GUI purposes only; obs was already captured above.
        if self.render_mode == "bluesky":
            bs.stack.stack(f"PAN {self.wpt_lat},{self.wpt_lon}")
            bs.stack.stack("ZOOM 0.15")
            bs.sim.step()
            for _ in range(5):
                self._sync_bluesky_gui()

        if self.render_mode in ("bluesky", "human"):
            self._render_frame()

        return obs, info

    # =================================================================
    # Step
    # =================================================================

    def _debug_obs(self, tag, obs, action=None):
        ac_idx = bs.traf.id2idx("KL001")
        cas_kt = bs.traf.cas[ac_idx] * MpS2Kt if ac_idx >= 0 else -1
        alt_ft = bs.traf.alt[ac_idx] / 0.3048 if ac_idx >= 0 else -1
        hdg    = bs.traf.hdg[ac_idx] if ac_idx >= 0 else -1
        traf_ids = list(bs.traf.id)
        print(f"\n[DBG {tag}] mode={self.render_mode} simt={bs.sim.simt:.1f} "
              f"sim.state={bs.sim.state} ntraf={bs.traf.ntraf} "
              f"hdg={hdg:.1f} cas={cas_kt:.1f}kt alt={alt_ft:.0f}ft")
        print(f"  traf.id      : n={len(traf_ids)}/{bs.traf.ntraf}  ids={traf_ids[:10]}")
        if action is not None:
            print(f"  action       : {action}")
        print(f"  cos(drift)   : {obs['cos(drift)'][0]:.4f}")
        print(f"  sin(drift)   : {obs['sin(drift)'][0]:.4f}")
        print(f"  airspeed     : {obs['airspeed'][0]:.4f} m/s")
        print(f"  waypoint_dist: {obs['waypoint_dist'][0]:.4f}")
        print(f"  faf_reached  : {obs['faf_reached'][0]:.1f}")
        print(f"  distances    : {np.round(obs['distances'], 3)}")

    def step(self, action):
        if self.verbose:
            self._debug_obs("STEP_IN", self._get_obs(), action=action)

        self._get_action(action)

        for _ in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "bluesky":
                self._sync_bluesky_gui()
            if self.render_mode in ("bluesky", "human"):
                self._render_frame()

        obs = self._get_obs()
        reward, terminated = self._get_reward()
        info = self._get_info()
        if self.verbose:
            self._debug_obs("STEP_OUT", obs)

        if terminated:
            for acid in list(bs.traf.id):
                try:
                    idx = bs.traf.id2idx(acid)
                    bs.traf.delete(idx)
                except (ValueError, AttributeError):
                    pass

        return obs, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        bs.stack.stack("HOLD")
