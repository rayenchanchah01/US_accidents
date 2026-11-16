"""Reproduce the complete full-data study and static dashboard."""
import argparse
from pathlib import Path
from accidents.prepare import prepare
from accidents.train import train
from accidents.export import export
from accidents.report import report
from accidents.story import export_story


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Full US Accidents CSV or single-CSV ZIP')
    p.add_argument('--tune-rows', type=int, default=600_000)
    p.add_argument('--validation-rows', type=int, default=160_000)
    args = p.parse_args()
    prepare(args.source)
    train(tune_rows=args.tune_rows, val_rows=args.validation_rows)
    export()
    export_story(args.source)
    report()
