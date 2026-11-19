"""Run the study on any US Accidents CSV, one stage at a time or end to end.

Stages: clean (prepare the dataset), train (select and refit the model, test set untouched),
test (score the test set, then export the dashboard data and reports), all.
The dashboard runs these same stages from its Pipeline panel.
"""
import argparse
from pathlib import Path
from accidents.prepare import prepare
from accidents.train import train, test
from accidents.export import export
from accidents.report import report
from accidents.story import export_story


def run(stage, source=None, tune_rows=600_000, val_rows=160_000):
    if stage in ('clean', 'all'):
        prepare(source)
    if stage in ('train', 'all'):
        train(tune_rows=tune_rows, val_rows=val_rows, evaluate=False)
    if stage in ('test', 'all'):
        test()
        export()
        export_story(source)
        report()
    print(f'Stage "{stage}" finished.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='US Accidents CSV or single-CSV ZIP')
    p.add_argument('--stage', choices=['clean', 'train', 'test', 'all'], default='all')
    p.add_argument('--tune-rows', type=int, default=600_000)
    p.add_argument('--validation-rows', type=int, default=160_000)
    args = p.parse_args()
    run(args.stage, args.source, args.tune_rows, args.validation_rows)
