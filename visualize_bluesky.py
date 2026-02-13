"""
Visualize a trained RL agent in the BlueSky QtGL GUI.

This script:
  1. Starts a BlueSky server (no auto-spawned sim nodes) in a background thread
  2. Runs the RL simulation as the only sim node (so the GUI displays its data)
  3. Launches the BlueSky QtGL GUI as a client subprocess

Usage:
  python visualize_bluesky.py
  python visualize_bluesky.py --model_path models/DescentEnvCR3D-v0/DescentEnvCR3D-v0_SAC
  python visualize_bluesky.py --scenario_path scenarios_kord/scenario_001.scn --episodes 5

The RL agent aircraft (KL001) appears in green on the BlueSky radar display.
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL agent in the BlueSky GUI"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/DescentEnvCR3D-v0/DescentEnvCR3D-v0_SAC",
        help="Path to trained SB3 model (.zip)"
    )
    parser.add_argument(
        "--scenario_path", type=str,
        default="scenarios_kord/scenario_001.scn",
        help="Scenario file or directory"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run"
    )
    args = parser.parse_args()

    # Create the BlueSky-GUI variant of the environment
    env = gym.make(
        "DescentEnvCR3D-Bluesky-v0",
        render_mode="bluesky",
        scenario_path=args.scenario_path,
    )

    # Load the trained model
    model = SAC.load(args.model_path, env=env)

    print(f"Running {args.episodes} episodes with model: {args.model_path}")
    print(f"Scenario: {args.scenario_path}")
    print("Watch the BlueSky GUI — the agent aircraft (KL001) is red.\n")

    for i in range(args.episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            total_reward += reward
            steps += 1

        print(f"  Episode {i + 1}/{args.episodes}: "
              f"reward = {total_reward:.2f}, steps = {steps}, "
              f"intrusions = {info.get('total_intrusions', 'N/A')}, "
              f"final_alt = {info.get('final_altitude', 'N/A'):.0f} m")

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
