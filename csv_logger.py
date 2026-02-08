import csv
import os


class CSVLogger:
    def __init__(self, path):
        self.path = path
        self._fieldnames = None

    def log(self, row):
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        file_exists = os.path.isfile(self.path)
        write_header = not file_exists

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
