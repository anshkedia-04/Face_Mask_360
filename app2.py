"""Compatibility entrypoint mirroring app.py for deployments targeting app2.py."""

from facemask_app.ui import main

if __name__ == "__main__":
    main(default_page="Mark Attendance")
