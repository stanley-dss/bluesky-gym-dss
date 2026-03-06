"""
Visualize a trained RL agent in the BlueSky QtGL GUI.

This script:
  1. Starts a BlueSky server (no auto-spawned sim nodes) in a background thread
  2. Runs the RL simulation as the only sim node (so the GUI displays its data)
  3. Launches the BlueSky QtGL GUI as a client subprocess

Usage:
  # Descent environment (default)
  python visualize_bluesky.py
  python visualize_bluesky.py --env descent --model_path models/DescentEnvCR3D-v0/DescentEnvCR3D-v0_SAC

  # Merge environment
  python visualize_bluesky.py --env merge
  python visualize_bluesky.py --env merge --model_path models/MergeEnv-v0/MergeEnv-v0_SAC --scenario_path scenarios_kord_merge_standard/

The RL agent aircraft (KL001) appears in red on the BlueSky radar display.
Intruder aircraft appear in their default color.
"""
import argparse
import sys
import subprocess
import threading
import time

import zmq
import bluesky as bs

# Initialize BlueSky as a networked simulation node BEFORE importing gym/envs,
# so that when the environment checks `if bs.sim is None` it finds an already-
# initialized (networked) BlueSky and skips the detached-mode fallback.
bs.init(mode='sim', detached=False)


def _start_server():
    """Start a minimal BlueSky server in a background thread.

    Unlike the default server started by BlueSky.py, this one does NOT
    auto-spawn a sim node subprocess.  Our script IS the only sim node,
    so the GUI (connected as a client) will display our data.
    """
    from bluesky.network.server import Server

    server = Server(discovery=False)

    # Monkey-patch addnodes to a no-op so run() doesn't spawn a subprocess.
    # The server's run() calls self.addnodes() on startup — we skip that
    # because our process is already the sim node.
    server.addnodes = lambda **kwargs: None

    def _run():
        try:
            server.run()
        except Exception as e:
            print(f"Server error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # Give the server time to bind its ZMQ ports
    time.sleep(1.0)
    return server


def _launch_gui():
    """Launch the BlueSky QtGL GUI as a client subprocess."""
    proc = subprocess.Popen(
        [sys.executable, '-m', 'bluesky', '--client', 'localhost'],
        # Inherit stdout/stderr so we can see any GUI errors
    )
    # Give the GUI time to connect to the server
    time.sleep(3.0)
    return proc


# ── Bootstrap: server → connect node → launch GUI ──────────────────────
server = _start_server()
bs.net.connect()
gui_proc = _launch_gui()

# Now safe to import gym / RL libs (they may trigger env registration)
import gymnasium as gym
from stable_baselines3 import SAC

import bluesky_gym
bluesky_gym.register_envs()


# ── Environment configs ──────────────────────────────────────────────────
ENV_CONFIGS = {
    "descent": {
        "gym_id":        "DescentEnvCR3D-Bluesky-v0",
        "model_path":    "models/DescentEnvCR3D-v0/DescentEnvCR3D-v0_SAC",
        "scenario_path": "scenarios_kord/scenario_001.scn",
        "pan_lat":       None,
        "pan_lon":       None,
        "zoom":          0.4,
    },
    "merge": {
        "gym_id":        "MergeEnv-Bluesky-v0",
        "model_path":    "models/MergeEnv-Bluesky-v0/MergeEnv-Bluesky-v0_SAC",
        "scenario_path": "scenarios_kord_merge_standard/",
        "pan_lat":       None,
        "pan_lon":       None,
        "zoom":          0.15,
    },
}


def _print_episode_result(env_type, i, n_episodes, total_reward, steps, info):
    base = (f"  Episode {i + 1}/{n_episodes}: "
            f"reward = {total_reward:.2f}, steps = {steps}, "
            f"intrusions = {info.get('total_intrusions', 'N/A')}")
    if env_type == "descent":
        alt = info.get('final_altitude', None)
        alt_str = f"{alt:.0f} m" if alt is not None else "N/A"
        print(base + f", final_alt = {alt_str}")
    else:
        faf = info.get('faf_reach', 'N/A')
        drift = info.get('average_drift', None)
        drift_str = f"{drift:.3f} rad" if drift is not None else "N/A"
        print(base + f", faf_reach = {faf}, avg_drift = {drift_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL agent in the BlueSky GUI"
    )
    parser.add_argument(
        "--env", type=str, default="descent", choices=["descent", "merge"],
        help="Environment to visualize: 'descent' or 'merge' (default: descent)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained SB3 model (.zip). Defaults to the standard path for the chosen env."
    )
    parser.add_argument(
        "--scenario_path", type=str, default=None,
        help="Scenario file or directory. Defaults to the standard path for the chosen env."
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--pan_lat", type=float, default=None,
        help="Latitude to pan the GUI to on startup (overrides env default)"
    )
    parser.add_argument(
        "--pan_lon", type=float, default=None,
        help="Longitude to pan the GUI to on startup (overrides env default)"
    )
    parser.add_argument(
        "--zoom", type=float, default=None,
        help="Zoom level for the GUI (overrides env default)"
    )
    args = parser.parse_args()

    cfg = ENV_CONFIGS[args.env]

    # Override defaults with any explicitly provided CLI arguments
    model_path    = args.model_path    or cfg["model_path"]
    scenario_path = args.scenario_path or cfg["scenario_path"]
    pan_lat       = args.pan_lat       if args.pan_lat is not None else cfg["pan_lat"]
    pan_lon       = args.pan_lon       if args.pan_lon is not None else cfg["pan_lon"]
    zoom          = args.zoom          if args.zoom    is not None else cfg["zoom"]

    # Optional manual GUI pan/zoom (env also pans automatically on reset)
    if pan_lat is not None and pan_lon is not None:
        bs.stack.stack(f"PAN {pan_lat},{pan_lon}")
        bs.stack.stack(f"ZOOM {zoom}")

    # Create the BlueSky-GUI variant of the environment
    env = gym.make(
        cfg["gym_id"],
        render_mode="bluesky",
        scenario_path=scenario_path,
    )

    # Load the trained model
    model = SAC.load(model_path, env=env)

    print(f"Environment  : {cfg['gym_id']}")
    print(f"Model        : {model_path}")
    print(f"Scenario     : {scenario_path}")
    print(f"Episodes     : {args.episodes}")
    print("Watch the BlueSky GUI — the agent aircraft (KL001) is red.\n")

    for i in range(args.episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        _print_episode_result(args.env, i, args.episodes, total_reward, steps, info)

        # Brief pause between episodes so the GUI isn't jarring
        time.sleep(1.0)

    env.close()
    bs.net.close()

    # Clean up GUI subprocess
    gui_proc.terminate()
    gui_proc.wait()
    print("\nDone.")


if __name__ == "__main__":
    main()
