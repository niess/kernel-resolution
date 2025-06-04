#! /usr/bin/env python3
import argparse
import gzip
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

from kernel_resolution import Processor


ARGS = None

PREFIX = Path(__file__).parent

plt.style.use(PREFIX / "paper.mplstyle")


def load(path: Path):
    """Load Monte Carlo deposits."""

    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def main():
    """Apply a Gaussian resolution to the Monte Carlo deposits."""

    data = load(ARGS.path)
    processor = Processor(
        coefficients = (9.9E-04, 8.47E-05, 0.612E-03),
        xmin = 0.0,
        xmax = 2.0,
        nx = 12001
    )

    iterator = processor.iterator(
        data.deposits,
        weights = data.rate * data.events / data.deposits.size,
        events = data.events
    )
    for _ in tqdm(iterator, desc="processing deposits", leave=False):
        pass

    density = processor.export()

    plt.figure()
    plt.plot(density.x, density.y[:,0], "k")
    plt.xlim(processor.xmin, processor.xmax)
    plt.yscale("log")
    plt.xlabel("energy (MeV)")
    plt.ylabel("rate (Hz / MeV)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("path",
        type=Path,
        help="deposits file to process"
    )

    ARGS = parser.parse_args()
    main()
