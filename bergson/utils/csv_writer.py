import csv


class CSVWriter:
    """CSV writer that no-ops when disabled."""

    def __init__(self, path: str, columns: list[str], enabled: bool = True):
        self.path = path
        if enabled:
            self._file = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(columns)
        else:
            self._file = None
            self._writer = None

    def writerow(self, *args):
        if self._writer is None or self._file is None:
            return
        self._writer.writerow([*args])
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
