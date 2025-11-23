import sqlite3


class AutoTrainDB:
    """
    A class to manage job records in a SQLite database.

    Attributes:
    -----------
    db_path : str
        The path to the SQLite database file.
    conn : sqlite3.Connection
        The SQLite database connection object (thread-safe).
    c : sqlite3.Cursor
        The SQLite database cursor object.

    Methods:
    --------
    __init__(db_path):
        Initializes the database connection and creates the jobs table if it does not exist.

    create_jobs_table():
        Creates the jobs table in the database if it does not exist.

    add_job(pid):
        Adds a new job with the given process ID (pid) to the jobs table.

    get_running_jobs():
        Retrieves a list of all running job process IDs (pids) from the jobs table.

    delete_job(pid):
        Deletes the job with the given process ID (pid) from the jobs table.

    close():
        Closes the database connection.

    __enter__(), __exit__():
        Context manager support for automatic resource cleanup.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.c = self.conn.cursor()
        self.create_jobs_table()

    def create_jobs_table(self):
        self.c.execute(
            """CREATE TABLE IF NOT EXISTS jobs
            (id INTEGER PRIMARY KEY, pid INTEGER)"""
        )
        self.conn.commit()

    def add_job(self, pid):
        self.c.execute("INSERT INTO jobs (pid) VALUES (?)", (pid,))
        self.conn.commit()

    def get_running_jobs(self):
        self.c.execute("""SELECT pid FROM jobs""")
        running_pids = self.c.fetchall()
        running_pids = [pid[0] for pid in running_pids]
        return running_pids

    def delete_job(self, pid):
        self.c.execute("DELETE FROM jobs WHERE pid=?", (pid,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context manager."""
        self.close()
