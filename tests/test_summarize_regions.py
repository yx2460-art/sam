import csv
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_regions import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, directory, broken=False):
        rows=[]
        for method in ('conflict','random'):
            for location,n,h in [('all',100,10),('near_gt_boundary',20,8),
                                 ('away_from_gt_boundary',80,2),('near_window_edge',50,5),
                                 ('away_from_window_edge',50,5)]:
                rows.append(dict(method=method,region='all_roi',location=location,pixels=n,
                                 removed_true_foreground=h,removed_false_foreground=0,
                                 added_true_foreground=0,added_false_foreground=0,relabeled_foreground=0))
        if broken:
            rows[-1]['removed_true_foreground']=4
        with (directory/'changes.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_reports_share_and_area_rate_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            self.fixture(directory)
            text=summarize(directory)
            self.assertIn('share=80.00%, rate=40.00%',text)
            self.assertIn('share=20.00%, rate=2.50%',text)
            with self.assertRaises(FileExistsError):
                summarize(directory)

    def test_rejects_inconsistent_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            self.fixture(directory,broken=True)
            with self.assertRaises(ValueError):
                summarize(directory)
            self.assertFalse((directory/'location_summary.txt').exists())
