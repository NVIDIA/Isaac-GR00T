import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.pop("DISPLAY", None)

# Suppress GLFW warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

import gymnasium as gym
import robocasa, robosuite
import gr00t_wbc.control.envs.robocasa.sync_env

print("Imports OK, creating env...", flush=True)
env = gym.make("gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc", enable_render=False)
obs, info = env.reset()
print("Env OK! Obs keys:", list(obs.keys()), flush=True)

print("Sampling action...", flush=True)
action = env.action_space.sample()
print("Stepping...", flush=True)
next_obs, reward, term, trunc, info = env.step(action)
print("Step OK! Reward:", reward, "Term:", term, flush=True)

print("Closing env...", flush=True)
env.close()
print("ALL GOOD", flush=True)
