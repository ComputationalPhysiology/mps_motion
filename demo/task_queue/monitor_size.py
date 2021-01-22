import argparse
import datetime
import os
from collections import namedtuple
from pathlib import Path

File = namedtuple("File", ["path", "size", "mtime"])


def get_parser():
    descr = "Check size of folder and delete oldest files if size is a above a certain limit"
    uasage = "python monitor_size.py folder --limit 10 \n"

    parser = argparse.ArgumentParser()

    parser.description = descr
    parser.usage = uasage

    parser.add_argument(
        action="store",
        dest="directory",
        type=str,
        help="Directory to check",
    )
    parser.add_argument(
        "--limit",
        action="store",
        dest="limit",
        default="3",
        type=int,
        help="Size limit in GB for when starting to delte. Default 3GB",
    )
    return parser


def main(directory, limit):

    limit_bytes = limit * 1e9
    directory = Path(directory)
    assert directory.is_dir(), f"Invalid directory {directory}"
    q = []
    for root, dirs, files in os.walk(directory):
        root = Path(root)
        for f in files:
            p = root.joinpath(f)
            stat = p.stat()
            time = datetime.datetime.fromtimestamp(stat.st_mtime)
            q.append(File(p, stat.st_size, time))

    size = sum(map(lambda x: x.size, q)) / 1e9
    print(f"Total size of folder '{directory}' is {size} GB")
    # Sort files based one the time they where modified last
    q.sort(key=lambda x: x.mtime, reverse=True)
    while sum(map(lambda x: x.size, q)) > limit_bytes:
        f = q.pop().path
        print(f"Delete file {f}")
        f.unlink()


if __name__ == "__main__":
    kwargs = vars(get_parser().parse_args())
    main(**kwargs)
