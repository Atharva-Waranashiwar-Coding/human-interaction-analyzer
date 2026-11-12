"""Never run destructive fixture cleanup against the user's SQLite database."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_ROOT = tempfile.TemporaryDirectory(prefix="interaction-tests-")
os.environ["DATABASE_PATH"] = str(Path(TEST_ROOT.name) / "test.db")
os.environ["UPLOAD_ROOT"] = str(Path(TEST_ROOT.name) / "uploads")
