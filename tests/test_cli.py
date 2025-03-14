from mps_motion.__main__ import app
from typer.testing import CliRunner
import pytest


@pytest.mark.xfail
def test_cli(TEST_FILENAME):
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", TEST_FILENAME])
    assert result.exit_code == 0
