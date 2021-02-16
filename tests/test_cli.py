import subprocess as sp


def main(TEST_FILENAME):
    sp.check_call("python", "-m", "mps_motion_tracking", TEST_FILENAME)
