# project_logger.py
import logging  # The built-in Python module for logging events in your application.
import csv  # For writing log data into CSV files in a structured, tabular format.
from pathlib import Path  # Modern way to handle file paths across different operating systems.
from datetime import datetime  # To timestamp log entries with the current date and time.

class CSVLogHandler(logging.Handler):
    def __init__(self, module_name: str):
        super().__init__()  
        self.module_name = module_name  # Store the module name from function in module
        self.log_file = Path(f"logs/{module_name}_logs.csv")  
        self._ensure_log_directory_exists()  
        self._initialize_csv()  # Set up the CSV file with headers so it’s ready for logging.

    # Why `_ensure_log_directory_exists`? 
    # We need to guarantee the 'logs/' directory exists before writing the CSV file. Without this,
    # attempting to write to a non-existent directory would raise an error. This method makes the
    # logger robust and self-contained. Could we skip it? Yes, if we manually created the directory
    # beforehand, but that’s less user-friendly and error-prone.
    def _ensure_log_directory_exists(self):
        """Ensure the 'logs/' directory exists and is writable, raising an error if it fails."""
        try:
            # Why `self.log_file.parent.mkdir`? 
            # `self.log_file` is a `Path` object (e.g., "logs/module_logs.csv"). The `.parent`
            # attribute comes from `pathlib.Path` and gives us the directory part ("logs/").
            # `mkdir(exist_ok=True)` creates the directory if it’s missing and does nothing if it
            # already exists, avoiding unnecessary errors.
            self.log_file.parent.mkdir(exist_ok=True)
            # Why test writability? 
            # Creating a directory doesn’t guarantee we can write to it (e.g., due to permissions).
            # This test ensures the directory is usable.
            test_file = self.log_file.parent / "test_write.tmp"  # Create a temporary file path.
            with open(test_file, "w") as f:
                f.write("test")  # Write something small to verify permissions.
            test_file.unlink()  # Remove the test file to keep the directory clean.
        except Exception as e:
            raise RuntimeError(f"Failed to initialize logging directory: {e}")  # Inform the user if setup fails.

    # Why `_initialize_csv`? 
    # This clears the old log file (if it exists) and sets up headers. We do this to start fresh
    # each time the logger is initialized, ensuring no stale data lingers and the CSV is properly
    # structured from the start.
    def _initialize_csv(self):
        """Clear the CSV file if it exists and write the column headers."""
        try:
            # How do we clear the CSV? 
            # Opening the file in "w" (write) mode overwrites it completely, effectively clearing
            # it. If the file didn’t exist, it’s created. This is simpler than checking for existence
            # and deleting manually.
            with open(self.log_file, mode="w", newline="") as file:  # "newline='' " avoids extra blank lines on Windows.
                writer = csv.writer(file)  # Use the CSV module to write rows cleanly.
                # Why these headers? 
                # They define the structure of our log data: timestamp, level, message, module, and
                # traceback. This makes the CSV readable and consistent.
                writer.writerow(["timestamp", "level", "message", "module", "traceback"])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CSV log file: {e}")  # Handle file creation errors gracefully.

    # Why `emit`? 
    # This is the core method of a logging handler, required by `logging.Handler`. It’s called
    # whenever a log message is emitted (e.g., via `logger.info()`), and we override it to write
    # to our CSV file.
    def emit(self, record):
        """Write a log record to the CSV file in a structured format."""
        try:
            # What’s `record.levelname`? 
            # `record` is a `LogRecord` object automatically created by the logging system. It has
            # attributes like `levelname` (e.g., "INFO", "ERROR"), `message`, and `exc_info`,
            # provided by the `logging` module when a log event occurs.
            log_entry = [
                datetime.now().isoformat(),  # Current time in ISO format (e.g., "2025-03-21T12:00:00").
                record.levelname,  # Log level from the `LogRecord` (e.g., "DEBUG", "ERROR").
                record.getMessage(),  # The actual log message, processed by the logger.
                self.module_name,  # Tag the entry with the module name for context.
                record.exc_info and self.format(record),  # Include traceback if an exception occurred, else None.
            ]
            with open(self.log_file, mode="a", newline="") as file:  # "a" appends to the file without overwriting.
                writer = csv.writer(file)
                writer.writerow(log_entry)  # Add the log entry as a new row.
        except Exception as e:
            # Why catch exceptions here? 
            # If logging fails (e.g., file permissions change mid-run), we don’t want the app to crash.
            # Printing the error keeps the app running while alerting us to the issue.
            print(f"Failed to write log to CSV: {e}")

# Why define `setup_logger` last? 
# Order doesn’t strictly matter in Python for function definitions (as long as they’re defined
# before use), but we put it after `CSVLogHandler` because it *uses* the class. This is a logical
# flow: define the tool (`CSVLogHandler`), then show how to use it (`setup_logger`). It’s more
# about readability than necessity.
def setup_logger() -> logging.Logger:
    """Set up and return a configured logger for the current module."""
    module_name = __name__  # `__name__` is a built-in variable giving the module’s name (e.g., "project_logger").
    logger = logging.getLogger(module_name)  # Get or create a logger instance for this module.
    logger.setLevel(logging.DEBUG)  # Capture all log levels from DEBUG upward.

    # Why clear handlers? 
    # If this logger was used before (e.g., in a long-running app), it might have old handlers
    # causing duplicate logs. Clearing them ensures a clean slate.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Why add `CSVLogHandler` here? 
    # This ties our custom handler to the logger, directing log output to our CSV file.
    csv_handler = CSVLogHandler(module_name)
    csv_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))  # Define log string format.
    logger.addHandler(csv_handler)  # Attach the handler to the logger.

    return logger  # Return the logger for use in the application.