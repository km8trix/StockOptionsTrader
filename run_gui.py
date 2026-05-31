"""
Launch the web GUI for the Trading System
"""

import sys
import os

# 1. Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 2. Explicitly import the app from the gui package
from gui.app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Stock Options Trading System - Web GUI")
    print("="*70)
    print("\n🌐 Starting Modular Flask server...\n")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("📱 Press CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    # 3. Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)