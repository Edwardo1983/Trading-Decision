import os
import sys
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "main"
SRC = ROOT / "src"
TESTS = ROOT / "tests"
LOCAL_TEMP = (ROOT / ".tmp_pytest_root").resolve()
LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(LOCAL_TEMP)
os.environ["TEMP"] = str(LOCAL_TEMP)
os.environ["TMP"] = str(LOCAL_TEMP)
tempfile.tempdir = str(LOCAL_TEMP)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


@pytest.fixture
def tmp_path():
    path = LOCAL_TEMP / f"case-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
