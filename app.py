#!/usr/bin/env python
"""WSGI entrypoint.

Delegates entirely to the application factory (cognitive_mirror.factory)
so gunicorn/Procfile/Dockerfile (`app:app`) and local `python app.py`
both boot the *same* app — auth, entries, Mirror, Sherlock Lens, the
stateless /predict demo endpoint, health, and metrics.

The previous version of this file defined a second, parallel Flask app
with only /predict and /review routes and no database — that's been
retired in favor of the factory so there's a single source of truth.
"""

import os
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognitive_mirror.factory import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))


def _find_available_port(start_port: int = 5000) -> int:
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if port == 5000:
        port = _find_available_port(port)
    app.run(debug=app.config.get("DEBUG", True), port=port)
