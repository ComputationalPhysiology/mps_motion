from mps_motion.__main__ import app
from typer.testing import CliRunner


def test_cli(TEST_FILENAME):
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", TEST_FILENAME])
    assert result.exit_code == 0
