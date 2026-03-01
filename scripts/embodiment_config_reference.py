#!/usr/bin/env python3
"""
GR00T Embodiment Configuration Reference and Debugger

Provides utilities to:
- List all available embodiments and their configurations
- Show state/action modality groups for each embodiment
- Validate embodiment configurations
- Help debug configuration issues
- Generate configuration templates for new embodiments

Usage:
    python scripts/embodiment_config_reference.py --list
    python scripts/embodiment_config_reference.py --show unitree_g1
    python scripts/embodiment_config_reference.py --validate /path/to/config.py
    python scripts/embodiment_config_reference.py --template new_robot
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add workspace to path if not already in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig
except ImportError as e:
    print("Error: Could not import GR00T modules. Make sure you're in the GR00T environment.")
    print(f"Details: {e}")
    sys.exit(1)


class EmbodimentConfigReference:
    """Reference and debugging tool for embodiment configurations."""

    def __init__(self):
        """Initialize the reference tool."""
        self.modality_configs = MODALITY_CONFIGS
        self.embodiments = list(EmbodimentTag)

    def list_embodiments(self) -> None:
        """List all available embodiments."""
        print("\n" + "=" * 80)
        print("📋 Available GR00T Embodiments")
        print("=" * 80 + "\n")

        for embodiment in self.embodiments:
            tag = embodiment.value
            doc = embodiment.__doc__ or "No description available"
            doc = doc.strip().split("\n")[0]

            # Check if configuration is available
            has_config = tag in self.modality_configs
            status = "✅" if has_config else "⚠️"

            print(f"{status} {tag:<30} - {doc}")

        print("\n" + "-" * 80)
        print(f"Total: {len(self.embodiments)} embodiments")
        print("✅ = Configuration available, ⚠️ = Configuration not found")
        print("-" * 80 + "\n")

    def show_embodiment(self, embodiment_tag: str) -> None:
        """Show detailed configuration for a specific embodiment.

        Args:
            embodiment_tag: The embodiment tag to show
        """
        # Find matching embodiment
        matching = [e for e in self.embodiments if e.value == embodiment_tag]
        if not matching:
            print(f"\n❌ Error: Embodiment '{embodiment_tag}' not found")
            self._suggest_embodiments(embodiment_tag)
            return

        embodiment = matching[0]
        print("\n" + "=" * 80)
        print(f"🤖 Embodiment Configuration: {embodiment_tag}")
        print("=" * 80 + "\n")

        # Show description
        if embodiment.__doc__:
            doc = embodiment.__doc__.strip()
            print(f"Description:\n   {doc}\n")

        # Show configuration if available
        if embodiment_tag not in self.modality_configs:
            print(f"⚠️  No modality configuration found for '{embodiment_tag}'")
            print("\nThis embodiment may be a placeholder for custom configurations.")
            print("See '--template' option to generate a configuration template.\n")
            return

        config = self.modality_configs[embodiment_tag]
        self._show_modality_details(config)

    def show_all_embodiments_summary(self) -> None:
        """Show summary table of all embodiments with modality-group counts."""
        print("\n" + "=" * 100)
        print("🤖 Embodiment Configuration Summary")
        print("=" * 100 + "\n")

        # Create summary table
        data = []
        for embodiment in self.embodiments:
            tag = embodiment.value
            if tag not in self.modality_configs:
                data.append(
                    {
                        "Embodiment": tag,
                        "State Groups": "—",
                        "Action Groups": "—",
                        "Videos": "—",
                        "Status": "No Config",
                    }
                )
                continue

            config = self.modality_configs[tag]
            
            # Get state dimension
            state_dim = 0
            if "state" in config and hasattr(config["state"], "modality_keys"):
                state_dim = len(config["state"].modality_keys)
            
            # Get action dimension  
            action_dim = 0
            if "action" in config and hasattr(config["action"], "modality_keys"):
                action_dim = len(config["action"].modality_keys)
            
            # Get number of video streams
            num_videos = 0
            if "video" in config and hasattr(config["video"], "modality_keys"):
                num_videos = len(config["video"].modality_keys)

            data.append(
                {
                    "Embodiment": tag,
                    "State Groups": state_dim,
                    "Action Groups": action_dim,
                    "Videos": num_videos,
                    "Status": "✓",
                }
            )

        # Print as formatted table
        if data:
            headers = ["Embodiment", "State Groups", "Action Groups", "Videos", "Status"]
            
            # Calculate column widths
            col_widths = {}
            for h in headers:
                col_widths[h] = max(len(h), max(len(str(row.get(h, ""))) for row in data))
            
            # Print header
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            print(header_line)
            print("-" * len(header_line))

            # Print rows
            for row in data:
                row_line = " | ".join(str(row.get(h, "—")).ljust(col_widths[h]) for h in headers)
                print(row_line)

        print("\n" + "=" * 100 + "\n")

    def _show_modality_details(self, config: Dict[str, Any]) -> None:
        """Show detailed modality configuration."""
        print("Configuration Details:")
        print("-" * 80)

        # State modality
        print("\n📊 STATE MODALITY:")
        if "state" in config:
            state_config = config["state"]
            if hasattr(state_config, "modality_keys") and state_config.modality_keys:
                print(f"   Keys: {', '.join(state_config.modality_keys)}")
            if hasattr(state_config, "delta_indices") and state_config.delta_indices:
                print(f"   Delta Indices: {state_config.delta_indices}")
            state_groups = len(state_config.modality_keys) if hasattr(state_config, "modality_keys") else 0
            print(f"   Groups: {state_groups}")
        else:
            print("   No state configuration")

        # Action modality
        print("\n🎮 ACTION MODALITY:")
        if "action" in config:
            action_config = config["action"]
            if hasattr(action_config, "modality_keys") and action_config.modality_keys:
                print(f"   Keys: {', '.join(action_config.modality_keys)}")
            if hasattr(action_config, "delta_indices") and action_config.delta_indices:
                print(f"   Delta Indices: {action_config.delta_indices}")
            if hasattr(action_config, "action_configs") and action_config.action_configs:
                print(f"   Action Configs: {len(action_config.action_configs)} defined")
                for i, ac in enumerate(action_config.action_configs):
                    if hasattr(ac, "rep") and hasattr(ac, "type"):
                        print(f"      [{i}] rep={ac.rep.name if hasattr(ac.rep, 'name') else ac.rep}, type={ac.type.name if hasattr(ac.type, 'name') else ac.type}")
            action_groups = len(action_config.modality_keys) if hasattr(action_config, "modality_keys") else 0
            print(f"   Groups: {action_groups}")
        else:
            print("   No action configuration")

        # Video modality
        print("\n🎬 VIDEO MODALITY:")
        if "video" in config:
            video_config = config["video"]
            if hasattr(video_config, "modality_keys") and video_config.modality_keys:
                print(f"   Streams: {', '.join(video_config.modality_keys)}")
            if hasattr(video_config, "delta_indices") and video_config.delta_indices:
                print(f"   Delta Indices: {video_config.delta_indices}")
        else:
            print("   No video configuration")

        print("\n" + "-" * 80 + "\n")

    def _suggest_embodiments(self, tag: str) -> None:
        """Suggest similar embodiment names."""
        import difflib

        options = [e.value for e in self.embodiments]
        suggestions = difflib.get_close_matches(tag, options, n=3, cutoff=0.6)

        if suggestions:
            print("\nDid you mean one of these?")
            for suggestion in suggestions:
                print(f"   • {suggestion}")

    def generate_template(self, robot_name: str) -> None:
        """Generate a configuration template for a new robot.

        Args:
            robot_name: Name of the new robot/embodiment
        """
        print("\n" + "=" * 80)
        print(f"📝 Configuration Template: {robot_name}")
        print("=" * 80 + "\n")

        template = f'''
"""Configuration for {robot_name} embodiment."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Define the configuration for {robot_name}
{robot_name.lower()}_config = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "camera_1",      # Change to your camera names
            # "camera_2",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm",           # Change to your state modality names
            "gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 8)),  # Adjust based on action dims
        modality_keys=[
            "arm",           # Change to match action modalities
            "gripper",
        ],
        action_configs=[
            # arm (example: 6-DOF)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,  # or ABSOLUTE
                type=ActionType.NON_EEF,            # or EEF
                format=ActionFormat.DEFAULT,
            ),
            # gripper (example: 1-DOF)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
}}


# Register the configuration
register_modality_config("{robot_name.lower()}", {robot_name.lower()}_config)
'''

        print("Example configuration:")
        print("-" * 80)
        print(template)
        print("-" * 80)

        print("\nNext steps:")
        print("1. Customize the camera_names, state keys, and action keys for your robot")
        print("2. Update the state and action dimensions based on your robot")
        print("3. Register the configuration using register_modality_config()")
        print("4. Use --validate to check your configuration")
        print("\nFor more details, see:")
        print("  - getting_started/finetune_new_embodiment.md")
        print("  - gr00t/data/types.py (ModalityConfig reference)")
        print()

    def validate_config_file(self, config_path: str) -> None:
        """Validate a configuration file.

        Args:
            config_path: Path to configuration file
        """
        import importlib.util

        path = Path(config_path)
        if not path.exists():
            print(f"\n❌ Error: File not found: {config_path}")
            return

        print(f"\n🔍 Validating: {config_path}")
        print("=" * 80)

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("config_module", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            print("✅ File loaded successfully")
            print("\nAvailable configurations:")

            # Look for modality configs in the module
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, dict) and "video" in obj or "state" in obj or "action" in obj:
                    print(f"\n   • {name}")
                    if isinstance(obj, dict):
                        print(f"      Keys: {list(obj.keys())}")

        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            print("\nCommon issues:")
            print("  - Missing imports (ModalityConfig, ActionConfig, etc.)")
            print("  - Syntax errors in the file")
            print("  - Incorrect indentation")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GR00T Embodiment Configuration Reference and Debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/embodiment_config_reference.py --list
  python scripts/embodiment_config_reference.py --all
  python scripts/embodiment_config_reference.py --show unitree_g1
  python scripts/embodiment_config_reference.py --template my_robot
  python scripts/embodiment_config_reference.py --validate config.py
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available embodiments",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show summary of all embodiments with modality-group counts",
    )
    parser.add_argument(
        "--show",
        type=str,
        metavar="EMBODIMENT",
        help="Show detailed configuration for a specific embodiment",
    )
    parser.add_argument(
        "--template",
        type=str,
        metavar="ROBOT_NAME",
        help="Generate configuration template for a new robot",
    )
    parser.add_argument(
        "--validate",
        type=str,
        metavar="CONFIG_FILE",
        help="Validate a configuration file",
    )

    args = parser.parse_args()

    # Create reference tool
    reference = EmbodimentConfigReference()

    # Handle commands
    if args.list:
        reference.list_embodiments()
    elif args.all:
        reference.show_all_embodiments_summary()
    elif args.show:
        reference.show_embodiment(args.show)
    elif args.template:
        reference.generate_template(args.template)
    elif args.validate:
        reference.validate_config_file(args.validate)
    else:
        # Default: show summary
        reference.list_embodiments()

    sys.exit(0)


if __name__ == "__main__":
    main()
