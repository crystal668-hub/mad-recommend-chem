import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.environment import load_project_environment


class ProjectEnvironmentTests(unittest.TestCase):
    def test_declared_values_replace_or_remove_inherited_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "REPLACED_KEY=fresh\nREMOVED_KEY\nEMPTY_KEY=\n",
                encoding="utf-8",
            )
            inherited = {
                "REPLACED_KEY": "stale",
                "REMOVED_KEY": "stale",
                "EMPTY_KEY": "stale",
                "UNRELATED_KEY": "keep",
            }
            with patch.dict(os.environ, inherited, clear=True):
                loaded = load_project_environment(env_path)

                self.assertEqual(loaded, env_path)
                self.assertEqual(os.environ["REPLACED_KEY"], "fresh")
                self.assertNotIn("REMOVED_KEY", os.environ)
                self.assertEqual(os.environ["EMPTY_KEY"], "")
                self.assertEqual(os.environ["UNRELATED_KEY"], "keep")

    def test_missing_file_does_not_change_environment(self):
        with patch.dict(os.environ, {"EXISTING_KEY": "keep"}, clear=True):
            loaded = load_project_environment(Path("missing-project.env"))

            self.assertIsNone(loaded)
            self.assertEqual(os.environ["EXISTING_KEY"], "keep")


if __name__ == "__main__":
    unittest.main()
