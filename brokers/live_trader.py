# brokers/live_trader.py
import requests
from requests_oauthlib import OAuth1Session # Required for E*TRADE authentication
from datetime import datetime
from core.models import Asset, AssetType, Order, OrderType
import json

class LiveEtradeBroker:
    """Live execution broker for E*TRADE API"""
    
    def __init__(self, consumer_key, consumer_secret, access_token, access_secret, account_id_key):
        # E*TRADE uses OAuth 1.0a
        self.session = OAuth1Session(consumer_key, consumer_secret, access_token, access_secret)
        self.base_url = "https://api.etrade.com/v1"
        self.account_id_key = account_id_key
        
    def place_order(self, asset: Asset, order_type: OrderType, quantity: int, limit_price: float | None) -> str:
        """Translates system orders into live brokerage API payloads"""
        
        # Determine action (BUY/SELL)
        action = "BUY" if order_type == OrderType.BUY else "SELL"
        
        # Build the order payload for the E*TRADE API
        order_payload = {
            "PlaceOrderRequest": {
                "orderType": "EQ",
                "clientOrderId": f"ORD-{int(datetime.now().timestamp())}",
                "Order": [{
                    "allOrNone": False,
                    "priceType": "LIMIT" if limit_price else "MARKET",
                    "orderTerm": "GOOD_FOR_DAY",
                    "Instrument": [{
                        "Product": {
                            "securityType": "EQ",
                            "symbol": asset.symbol
                        },
                        "orderAction": action,
                        "quantityType": "QUANTITY",
                        "quantity": quantity
                    }]
                }]
            }
        }

        if limit_price:
            order_payload["PlaceOrderRequest"]["Order"][0]["limitPrice"] = limit_price

        url = f"{self.base_url}/accounts/{self.account_id_key}/orders/place.json"
        
        # Execute the real trade
        response = self.session.post(url, json=order_payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("PlaceOrderResponse", {}).get("OrderIds", [{}])[0].get("orderId", "UNKNOWN")
        else:
            raise Exception(f"Live trade failed: {response.text}")

    def get_portfolio_status(self) -> dict:
        """Fetches real-time portfolio balances and positions from the broker"""
        
        # 1. Fetch live balances
        balance_url = f"{self.base_url}/accounts/{self.account_id_key}/balance.json"
        balance_resp = self.session.get(balance_url).json()
        cash = balance_resp.get("BalanceResponse", {}).get("Computed", {}).get("cashAvailableForInvestment", 0)
        
        # 2. Fetch live positions
        positions_url = f"{self.base_url}/accounts/{self.account_id_key}/portfolio.json"
        pos_resp = self.session.get(positions_url).json()
        
        live_positions = []
        for pos in pos_resp.get("PortfolioResponse", {}).get("AccountPortfolio", [{}])[0].get("Position", []):
            live_positions.append({
                'symbol': pos["Product"]["symbol"],
                'quantity': pos["quantity"],
                'current_price': pos["marketValue"] / pos["quantity"] if pos["quantity"] > 0 else 0,
                'pnl': pos["totalGain"],
                'pnl_pct': pos["totalGainPct"]
            })

        return {
            'timestamp': datetime.now().isoformat(),
            'cash': cash,
            'portfolio_value': balance_resp.get("BalanceResponse", {}).get("Computed", {}).get("RealTimeValues", {}).get("totalAccountValue", 0),
            'positions': live_positions,
            'pending_orders': 0 # Would require fetching from /orders endpoint
        }