import subprocess as sp


def test_cli(TEST_FILENAME):
    sp.check_call(["python", "-m", "mps_motion", TEST_FILENAME])
