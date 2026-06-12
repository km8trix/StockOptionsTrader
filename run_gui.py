"""
Launch the web GUI for the Trading System
"""
from __future__ import annotations

import os
import sys

# 1. Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 2. Build the app via the application factory
from gui.app import create_app, get_server_config

if __name__ == '__main__':
    host, port, debug = get_server_config()

    print("\n" + "=" * 70)
    print("Stock Options Trading System - Web GUI")
    print("=" * 70)
    print("\n🌐 Starting Modular Flask server...\n")
    print(f"📱 Open your browser and go to: http://{host}:{port}")
    print("📱 Press CTRL+C to stop the server\n")
    print("=" * 70 + "\n")

    # 3. Run the application.
    # load_dotenv=False is load-bearing: Flask's run() otherwise calls
    # cli.load_dotenv() (python-dotenv is installed) and silently injects the
    # repo-root .env — including real ETRADE_CONSUMER_KEY/SECRET — into
    # os.environ, even when the operator explicitly stripped ETRADE_* vars.
    # E*TRADE credentials must only enter the process via explicit operator
    # action; the launcher must never bulk-load secret files behind their back.
    create_app().run(host=host, port=port, debug=debug, load_dotenv=False)
