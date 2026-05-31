# gui/app.py
from flask import Flask, jsonify
from flask_cors import CORS
import traceback

# Import the blueprints
from gui.routes.views import views_bp
from gui.routes.api_analysis import analysis_bp
from gui.routes.api_backtest import backtest_bp 
from gui.routes.api_trading import trading_bp    

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Register Blueprints
app.register_blueprint(views_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(backtest_bp)
app.register_blueprint(trading_bp)

# Global Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Stock Options Trading System - Web GUI")
    print("="*70)
    print("\n🌐 Starting Modular Flask server...\n")
    print("📱 Access the GUI at: http://localhost:5001")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)