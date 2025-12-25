import ctypes
import os
from pathlib import Path
import subprocess


CREATE_NVIDIA_ICD_JSON = """
cat <<EOF >/usr/share/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF
"""

CREATE_10_NVIDIA_JSON = """
cat <<EOF >/usr/share/glvnd/egl_vendor.d/10_nvidia.json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
EOF
"""


def check_uv_installation():
    try:
        output = subprocess.check_output(["uv", "--version"], text=True, stderr=subprocess.DEVNULL)
        version_str = output.strip()
        print(f"✓ uv is installed: {version_str}")

        # Parse version (format: "uv 0.9.8" or similar)
        version_parts = version_str.split()[1].split(".")
        version_tuple = tuple(int(part) for part in version_parts[:3])

        if version_tuple < (0, 8, 14):
            raise RuntimeError(
                f"uv version {version_str} is too old. Expected at least v0.9.8. "
                f"Please run `uv self update` to update."
            )
    except FileNotFoundError:
        raise RuntimeError(
            "uv is not installed or not in PATH. "
            "Please run `curl -LsSf https://astral.sh/uv/install.sh | sh` to install uv."
        )


def _test_nvidia_driver_installation():
    try:
        cmd = "find /usr -type f -name 'libEGL_nvidia.so*' -o -name 'libGLX_nvidia.so*' -o -name 'libGLESv*.so*' 2>/dev/null"
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        if not output:
            raise RuntimeError("Necessary EGL/GLX/GLES libraries are not installed")
        print("✓ EGL/GLX/GLES libraries found")
        return True
    except Exception:
        return False


def _test_egl():
    try:
        from OpenGL import EGL

        # Initialize EGL
        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("Failed to get EGL display.")

        if not EGL.eglInitialize(display, None, None):
            raise RuntimeError("Failed to initialize EGL.")

        print("EGL initialized successfully.")

        # Choose an EGL config
        attrib_list = [
            EGL.EGL_SURFACE_TYPE,
            EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE,
            EGL.EGL_OPENGL_BIT,
            EGL.EGL_NONE,
        ]
        num_configs = ctypes.c_int()
        configs = (EGL.EGLConfig * 1)()

        if not EGL.eglChooseConfig(display, attrib_list, configs, 1, ctypes.byref(num_configs)):
            raise RuntimeError("Failed to choose EGL config.")

        if num_configs.value == 0:
            raise RuntimeError("No suitable EGL configs found.")

        print("EGL config chosen successfully.")

        # Create a Pbuffer surface (headless rendering)
        pbuffer_attribs = [EGL.EGL_WIDTH, 1, EGL.EGL_HEIGHT, 1, EGL.EGL_NONE]
        surface = EGL.eglCreatePbufferSurface(display, configs[0], pbuffer_attribs)
        if surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("Failed to create Pbuffer surface.")

        print("EGL Pbuffer surface created successfully.")

        # Create an OpenGL context
        context = EGL.eglCreateContext(display, configs[0], EGL.EGL_NO_CONTEXT, None)
        if context == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("Failed to create EGL context.")

        print("EGL context created successfully.")

        # Make the context current
        if not EGL.eglMakeCurrent(display, surface, surface, context):
            raise RuntimeError("Failed to make EGL context current.")

        print("EGL context made current successfully.")

        # Clean up
        EGL.eglDestroySurface(display, surface)
        EGL.eglDestroyContext(display, context)
        EGL.eglTerminate(display)

        print("✓ EGL headless test completed successfully")
        return True

    except Exception as e:
        print(f"✗ EGL test failed: {e}")
        return False


def check_vulkan_installation():
    if not Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists():
        print("Creating /usr/share/vulkan/icd.d/nvidia_icd.json ...")
        subprocess.run(CREATE_NVIDIA_ICD_JSON, shell=True)
    print("✓ Vulkan installation is OK")


def check_egl_installation():
    assert _test_nvidia_driver_installation(), (
        "Necessary EGL/GLX/GLES libraries are not installed. Reinstall the NVIDIA driver or set NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility when launching the container"
    )

    assert Path("/usr/share/glvnd/egl_vendor.d").exists(), (
        "EGL is not installed, please run `apt install libegl1 -y`"
    )
    if not Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json").exists():
        print("Creating /usr/share/glvnd/egl_vendor.d/10_nvidia.json ...")
        subprocess.run(CREATE_10_NVIDIA_JSON, shell=True)
    assert _test_egl(), "EGL test failed"


def check_robocasa_environments():
    groot_path = os.path.join(os.path.dirname(__file__), "../..")
    python_exec = os.path.join(
        os.path.dirname(__file__), "../../gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python3"
    )
    python_script = (
        "import gymnasium as gym\n"
        "import robocasa.utils.gym_utils.gymnasium_groot\n"
        "env = gym.make('robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env', enable_render=True)\n"
        "env.reset()\n"
        "env.step(env.action_space.sample())\n"
        "print('Env OK:', type(env))"
    )
    cmd = f'PYTHONPATH={groot_path} {python_exec} -c "{python_script}"'
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    assert "Env OK:" in output, f"Failed to check robocasa environment:\n{cmd}\n{output}"
    print("✓ RoboCasa environment is installed")


def check_robocasa_gr1_tabletop_tasks_environments():
    groot_path = os.path.join(os.path.dirname(__file__), "../..")
    python_exec = os.path.join(
        os.path.dirname(__file__),
        "../../gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python3",
    )
    python_script = (
        "import gymnasium as gym\n"
        "import robocasa.utils.gym_utils.gymnasium_groot\n"
        "env = gym.make('gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env', enable_render=True)\n"
        "env.reset()\n"
        "env.step(env.action_space.sample())\n"
        "print('Env OK:', type(env))"
    )
    cmd = f'PYTHONPATH={groot_path} {python_exec} -c "{python_script}"'
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    assert "Env OK:" in output, f"Failed to check robocasa environment:\n{cmd}\n{output}"
    print("✓ RoboCasa GR1 Tabletop Tasks environment is installed")


def check_g1_locomanipulation_environment():
    groot_path = os.path.join(os.path.dirname(__file__), "../..")
    python_exec = os.path.join(
        os.path.dirname(__file__),
        "../../gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python3",
    )
    python_script = (
        "import gymnasium as gym\n"
        "from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv\n"
        "env = gym.make('gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc', onscreen=False, offscreen=True)\n"
        "env.reset()\n"
        "env.step(env.action_space.sample())\n"
        "print('Env OK:', type(env))"
    )
    cmd = f'PYTHONPATH={groot_path} {python_exec} -c "{python_script}"'
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    assert "Env OK:" in output, f"Failed to check G1 LocoManipulation environment:\n{cmd}\n{output}"
    print("✓ G1 LocoManipulation environment is installed")


def check_simpler_env_environments():
    groot_path = os.path.join(os.path.dirname(__file__), "../..")
    python_exec = os.path.join(
        os.path.dirname(__file__), "../../gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python3"
    )
    python_script = (
        "import gymnasium as gym\n"
        "from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs\n"
        "register_simpler_envs()\n"
        "env = gym.make('simpler_env_google/google_robot_pick_coke_can')\n"
        "env.reset()\n"
        "env.step(env.action_space.sample())\n"
        "print('Env OK:', type(env))"
    )
    cmd = f'PYTHONPATH={groot_path} {python_exec} -c "{python_script}"'
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    assert "Env OK:" in output, f"Failed to check SimplerEnv environment:\n{cmd}\n{output}"
    print("✓ SimplerEnv environment is installed")


def check_libero_environments():
    groot_path = os.path.join(os.path.dirname(__file__), "../..")
    python_exec = os.path.join(
        os.path.dirname(__file__), "../../gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python3"
    )
    assert os.path.exists(os.path.expanduser("~/.libero")), (
        "Config folder is missing. Please rerun the setup script gr00t/eval/sim/LIBERO/setup_libero.sh"
    )

    python_script = (
        "import gymnasium as gym\n"
        "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs\n"
        "register_libero_envs()\n"
        "env = gym.make('libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket')\n"
        "env.reset()\n"
        "env.step(env.action_space.sample())\n"
        "env.close()\n"
        "print('Env OK:', type(env))"
    )
    cmd = f'PYTHONPATH={groot_path} {python_exec} -c "{python_script}"'
    output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    assert "Env OK:" in output, f"Failed to check Libero environment:\n{cmd}\n{output}"
    print("✓ LIBERO environment is installed")


if __name__ == "__main__":
    check_uv_installation()
    check_vulkan_installation()
    check_egl_installation()
    check_robocasa_environments()
    check_robocasa_gr1_tabletop_tasks_environments()
    check_g1_locomanipulation_environment()
    check_simpler_env_environments()
    check_libero_environments()
