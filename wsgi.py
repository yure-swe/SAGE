# =============================================================================
# wsgi.py — Gunicorn Entry Point
# =============================================================================
# This file is what Gunicorn uses to serve the app on Hostinger VPS.
#
# Start command (Hostinger VPS):
#   gunicorn --workers 2 --bind 0.0.0.0:5000 wsgi:app
#
# As a systemd service (see deployment guide):
#   ExecStart=/path/to/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:5000 wsgi:app
#
# Worker count = 2 for Hostinger KVM 2 (2GB RAM).
# Keep this low — each worker loads all .pkl models into memory (~300-600MB).
# =============================================================================

from app import app

if __name__ == "__main__":
    app.run()
