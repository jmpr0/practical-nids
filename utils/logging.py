import json
import sys
import os

class FileOutputDuplicator:
  """A class that duplicates stdout/stderr output to a file."""

  def __init__(self, duplicate, fname, mode='w'):
    self.file = open(fname, mode)
    self.duplicate = duplicate

  def __del__(self):
    """Ensure the file is closed properly when the object is destroyed."""
    if self.file and not self.file.closed:
      self.file.close()

  def write(self, data):
    """Write data to both file and duplicate stream."""
    self.file.write(data)
    self.duplicate.write(data)
    self.flush()

  def flush(self):
    """Flush both file and duplicate output stream."""
    self.file.flush()
    self.duplicate.flush()

class DiskLogger:
  """Handles logging to disk, including argument saving and stdout/stderr redirection."""

  def __init__(self, results_path):
    """Initialize with the results path where logs will be stored."""
    self.results_path = results_path
    os.makedirs(results_path, exist_ok=True)

  def log_args(self, args, filename='args.log'):
    """Save experiment arguments to a JSON file."""
    log_path = os.path.join(self.results_path, filename)
    try:
      with open(log_path, 'w') as f:
        json.dump(args, f, indent=2)
    except Exception as e:
      print(f"Error saving args log: {e}", file=sys.stderr)

  def log_stdout(self, filename='stdout.log'):
    """Redirect stdout to a file while maintaining console output."""
    sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(self.results_path, filename))

  def log_stderr(self, filename='stderr.log'):
    """Redirect stderr to a file while maintaining console output."""
    sys.stderr = FileOutputDuplicator(sys.stderr, os.path.join(self.results_path, filename))
