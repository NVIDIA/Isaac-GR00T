import json
from pathlib import Path
import tempfile
import unittest

from scripts.lerobot_conversion.convert_v3_to_v2 import convert_episodes_metadata


class ConvertEpisodesMetadataTest(unittest.TestCase):
    def test_omits_null_stat_leaves_and_preserves_available_stats(self) -> None:
        records = [
            {
                "episode_index": 7,
                "length": 12,
                "stats/action/count": [12],
                "stats/observation.images.rgb.cam_left_wrist/count": None,
                "stats/observation.images.rgb.cam_left_wrist/mean": None,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            convert_episodes_metadata(root, records)

            stats_path = root / "meta" / "episodes_stats.jsonl"
            payload = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["episode_index"], 7)
        self.assertEqual(payload["stats"]["action"]["count"], [12])
        self.assertNotIn("observation.images.rgb.cam_left_wrist", payload["stats"])


if __name__ == "__main__":
    unittest.main()
