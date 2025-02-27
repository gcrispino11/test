import json
from CITool import *
import pprint as pp
import pprint
import requests
#from Singleton import *
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta,timezone
import numpy as np

"""
Market Tag ID Lookup Table
=============================
Tag ID	Description
80	FX
81	FX FX-Major
82	FX AUD-Crosses
83	FX CHF-Crosses
84	FX EUR-Crosses
85	FX GBP-Crosses
86	FX Scandies-Crosses
87	FX JPY-Crosses
88	FX EM-Europe
89	FX EM-Asia
90	Indices
91	Indices UK
92	Indices US
93	Indices Europe
94	Indices Asia
95	Indices Australia
96	Commodities
97	Commodities Energy
98	Commodities Grain
99	Commodities Soft
100	Commodities Other
101	Commodities Options
102	Equities
103	Equities UK
104	Equities US
105	Equities Europe
106	Equities Asia
107	Equities Austria
108	Equities Belgium
109	Equities Canada
110	Equities Denmark
111	Equities France
112	Equities Germany
113	Equities Ireland
114	Equities Italy
115	Equities Netherlands
116	Equities Norway
117	Equities Poland
118	Equities Portugal
119	Equities Spain
120	Equities Sweden
121	Equities Switzerland
122	Equities Finland
123	Sectors
124	Sectors UK
125	Metals
126	Bonds
127	Interest Rates
128	iShares
129	iShares UK
130	iShares US
131	iShares Asia
132	iShares Australia
133	iShares Emerging-Markets
134	Options
135	Options UK 100
136	Options Germany 30
137	Options US SP 500
138	Options Wall Street
139	Options Australia 200
140	Options US Crude Oil
141	Options GBP/USD
142	Options EUR/USD
143	Options AUD/USD
144	Options Gold
145	Equities Australia
146	Popular
147	Popular Spreads
150	Popular Australia
"""


class COrderList:
    def __init__(self, orders):
        if len(orders) <= 0:
            self.orders = []
            return
        self.orders = orders

    def select_orders_by_marketID(self, marketID):
        orders = []
        for order in self.orders:
            if order["MarketId"] == marketID:
                orders.append(order)
        return orders


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class API(metaclass=_Singleton):
    """API class to interact with City Index API."""
    
    OP_BUY = "buy"
    OP_SELL = "sell"

    def __init__(self, isLive=True):
        self.isLive = isLive
        self.uid = None
        self.password = None
        self.session = None

        if isLive:
            self.APIURL = 'https://ciapi.cityindex.com/TradingAPI'
            self.base_url = "https://ciapi.cityindex.com/TradingAPI"
        else:
            #self.APIURL = "https://ciapipreprod.cityindextest9.co.uk/TradingApi/"
            self.APIURL = 'https://ciapi.cityindex.com/TradingAPI'
            
        # Explicitly load the .env file in the current directory
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
        
        # Print the environment variables to check if they are loaded
        print(f"USER_ID: {os.getenv('USER_ID')}")
        print(f"USER_PASSWORD: {os.getenv('USER_PASSWORD')}")
        self.uid = os.getenv('USER_ID')
        self.password = os.getenv('USER_PASSWORD')

        if not self.uid or not self.password:
            self.uid = input("Enter City Index account ID: ")
            self.password = input("Enter City Index password: ")

    def login(self):
        """Login to CityIndex"""
        data = {'Password': self.password,
                'AppVersion': '1',
                'AppComments': 'LoginFromPython',
                'Username': self.uid,
                'AppKey': 'cipythonAPP'}
        url = self.APIURL + '/session'
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, json.dumps(data), headers=headers)
        if response.status_code != 200:
            print("Failed Login " + str(response.status_code))
            return False

        self.login_resp = response.json()
        self.session = response.json()["Session"]
        return True

    def logout(self):
        """Logout of Cityindex"""
        data = {"Username": self.uid, "Session": self.session}
        url = self.APIURL + '/session/deleteSession?Username=' + self.uid + '&Session=' + self.session
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, json.dumps(data), headers=headers)
        if response.status_code != 200:
            print("Failed Logout " + str(response.status_code))
            return False

        return True

    """
            Account Information
    ===============================

    """

    def get_trading_account_info(self):
        """
        Get User's ClientAccountId and a list of their TradingAccounts
        :return: [dictionary] AccountInformationResponseDTO
        """
        data = {"Username": self.uid, "Session": self.session}
        url = self.APIURL + "/useraccount/ClientAndTradingAccount"
        response = requests.get(url, data)

        if response.status_code != 200:
            print("Error retrieving Trading Acc info: " + str(response.status))
            return False

        self.trading_account_info = response.json()
        return self.trading_account_info

    """
            Margin
    ===============================
    """

    def get_client_account_margin(self):
        """
        Retrieves the current margin values for the logged-in client account.
        :return: ApiClientAccountMarginResponseDTO
        """
        url = self.APIURL + '/margin/clientaccountmargin?Username=' + self.uid + '&Session=' + self.session
        response = requests.get(url)
        if response.status_code != 200:
            print("Error getting clientaccountmargin : " + str(response.status_code))
            return False
        self.client_account_margin = response.json()
        return self.client_account_margin

    """
            Market
    ===============================
    """

    def get_full_market_info(self, tagId="0"):
        """
        Return market information
        :param tagId: [string] market Tag IDs
        :return: [Dictionary] market_info[UnderlyingRicCode]
        """

        url = self.APIURL + "/market/fullsearchwithtags?Username=" + self.uid + "&Session=" + self.session + "&maxResults=200&tagId=" + tagId
        response = requests.get(url)
        if response.status_code != 200:
            return False

        response = json.loads(response.text)
        self.market_info = {}
        for symbol in response["MarketInformation"]:
            self.market_info[symbol["UnderlyingRicCode"]] = symbol

        return self.market_info

    #"""
    #        Price History
    #===============================
    #"""
    #
    #def get_pricebar_history(self, symbol, interval="HOUR", span="1", pricebars="65535", priceType="BID"):
    #    """
    #    Get historic price bars for the specified market in OHLC (open, high, low, close) format,
    #    suitable for plotting in candlestick charts. Returns price bars in ascending order up to the current time.
    #    When there are no prices for a particular time period, no price bar is returned.
    #    Thus, it can appear that the array of price bars has "gaps", i.e.
    #    the gap between the date & time of each price bar might not be equal to interval x span.
    #
    #    :param symbol: ricCode (not the marketTagID)
    #    :param interval: [string] (TICK, MINUTE, HOUR, DAY, WEEK)
    #    :param span: [string] (1, 2, 3, 5, 10, 15, 30 MINUTE) and (1, 2, 4, 8 HOUR) TICK, DAY and WEEK must be supplied with a span of 1
    #    :param pricebars: [string] number of pricebars in string
    #    :param priceType: [string] BID, MID, ASK
    #    :return: GetPriceBarResponseDTO
    #    """
    #    url = self.APIURL + "/market/" + str(
    #        self.market_info[symbol][
    #            "MarketId"]) + "/barhistory?Username=" + self.uid + "&Session=" + self.session + "&interval=" + interval + "&span=" + span + "&PriceBars=" + pricebars + "&PriceType=" + priceType
    #    response = requests.get(url)
    #    if (response.status_code != 200):
    #        print("GetPriceBarHistory: HTTP Error " + str(response.status_code))
    #        return False
    #
    #    return response.json()
    #
    #"""
    #        Trades and Orders
    #===============================
    #"""
    def get_pricebar_history(self, symbol, interval="HOUR", span="4", pricebars="1000"):
        """Enhanced price bar history retrieval with robust SSL handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = f"{self.APIURL}/market/{self.market_info[symbol]['MarketId']}/barhistory"
                params = {
                    'Username': self.uid,
                    'Session': self.session,
                    'interval': interval,
                    'span': span,
                    'PriceBars': pricebars,
                    'PriceType': 'BID'
                }
                
                session = requests.Session()
                session.verify = True
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=3,
                    pool_connections=100,
                    pool_maxsize=100
                )
                session.mount('https://', adapter)
                
                response = session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    self.login()  # Refresh session
                    
            except requests.exceptions.SSLError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return self.get_cached_data(symbol)
                
        return self.get_cached_data(symbol)
    
   
        
    def is_session_expired(self):
        """Check if current session needs refresh"""
        if not hasattr(self, 'session_start_time'):
            return True
        session_age = time.time() - self.session_start_time
        return session_age > 3600  # Refresh after 1 hour
        
    def list_markets(self):
        url = f"{self.base_url}/v2/market/list"
        headers = self.get_headers()
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to list markets: {str(e)}")
            return None
            
    def get_headers(self):
        """
        Generate headers for API requests.
        
        Returns:
            dict: Headers with authentication details.
        """
        if not self.session:
            raise ValueError("Session ID is not available. Please log in first.")
        return {
            "Authorization": f"Bearer {self.session}",
            "Content-Type": "application/json"
        }
    def get_market_info(self, symbol):
        """
        Fetch specific market information using MarketId or Symbol.
        """
        if not hasattr(self, "market_info") or not self.market_info:
            self.get_full_market_info()  # Popola i dati di mercato se non sono già disponibili
    
        for market in self.market_info.values():
            if market["Name"] == symbol or market["UnderlyingRicCode"] == symbol:
                return market
    
        print(f"Market info not found for symbol: {symbol}")
        return None
        
    def get_market_information_v2(self, symbol, client_account_id):
        """
        Fetch market information for the specified symbol.
        
        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            client_account_id (int): The client account identifier.
        
        Returns:
            dict: Market information if successful, None otherwise.
        """
        url = f"{self.base_url}/v2/market/information"
        params = {
            "symbol": symbol,
            "clientAccountId": client_account_id
        }
        headers = self.get_headers()  # Ensure session ID is included in headers
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:  # Unauthorized
                print("Session expired. Refreshing session...")
                self.login()  # Re-login to refresh session
                headers = self.get_headers()  # Get updated headers
                response = requests.get(url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            else:
                print(f"Failed to fetch market information for {symbol}: {str(e)}")
                return None
        except Exception as e:
            print(f"Error fetching market information for {symbol}: {str(e)}")
            return None
                    
    def simulate_trade_order(self, symbol, cmd, qty, data):
        url = self.APIURL + '/order/simulate/newtradeorder?Username=' + self.uid + "&Session=" + self.session
        
        trade_data = {
            "OcoOrder": None,
            "Applicability": None,
            "Direction": "SELL" if cmd == self.OP_SELL else "BUY",
            "BidPrice": float(data["Bid"]),
            "AuditId": str(data["AuditId"]),
            "AutoRollover": True,
            "MarketId": int(self.market_info[symbol]["MarketId"]),
            "isTrade": True,
            "OfferPrice": float(data["Offer"]),
            "Quantity": float(qty),
            "QuoteId": None,
            "TradingAccountId": int(self.trading_account_info["TradingAccounts"][0]["TradingAccountId"]),
            "PositionMethodId": 2
        }
        
        print(f"Trade Data: {trade_data}")
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, json.dumps(trade_data), headers=headers)
        return response.json() if response.status_code == 200 else False



    def orders_total(self):
        url = self.APIURL + '/order/openpositions'
        data = {"Username": self.uid, "Session": self.session,
                "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                "maxResults": 10000}
        response = requests.get(url, data)
        if response.status_code != 200:
            print("ListOpenPositions Error: " + str(response.status_code))
            return False

        return len(response.json()["OpenPositions"])

    def get_orders(self):
        url = self.APIURL + '/order/openpositions'
        data = {
            "Username": self.uid,
            "Session": self.session,
            "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
            "maxResults": 10000
        }
        try:
            response = requests.get(url, params=data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_json = response.json()
    
            # Pass the raw list of orders to COrderList
            if "OpenPositions" in response_json:
                return COrderList(response_json["OpenPositions"])
            elif isinstance(response_json, list):  # In case it's directly a list
                return COrderList(response_json)
            else:
                print("Unexpected response format.")
                return COrderList([])
    
        except requests.RequestException as e:
            print(f"HTTP Error: {e}")
            return COrderList([])
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return COrderList([])
    

    def get_order_history(self):
        url = self.APIURL + '/order/tradehistory'
        data = {"Username": self.uid, "Session": self.session,
                "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                "maxResults": 10000}
        response = requests.get(url, data)
        if response.status_code != 200:
            print("ListOpenPositions Error: " + str(response.status_code))
            return False
        orders = COrderList(response.json()["TradeHistory"])
        return orders

    def get_order(self, orderId):
        """

        :param orderId:
        :return:
        """
        url = self.APIURL + "/order/" + str(orderId) + "?UserName=" + self.uid + "&Session=" + self.session
        print("Get Order URL : " + url)
        response = requests.get(url)
        if response.status_code != 200:
            print("Get Order Error: " + str(response.status_code))
            return False
        pp.pprint(response.json())
        return response.json()

    def close_order(self, symbol, order, data):
        """

        :return:
        """
        orderID = order["OrderId"]
        cmd = order["Direction"]
        qty = order["Orders"][0]["Quantity"]

        if cmd == self.OP_BUY:
            oppcmd = self.OP_SELL
        else:
            oppcmd = self.OP_BUY

        data = {
            "PositionMethodId": None,
            "BidPrice": data["Bid"],
            "OfferPrice": data["Offer"],
            "AuditId": data["AuditId"],
            "MarketId": self.market_info[symbol]["MarketId"],
            "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
            "Direction": oppcmd,  # must have!
            "Quantity": qty,
            "Close": [orderID]
        }
        url = self.APIURL + '/order/newtradeorder?Username=' + self.uid + "&Session=" + self.session
        headers = {'Content-type': 'application/json'}
        pp.pprint(data)
        response = requests.post(url, json.dumps(data), headers=headers)
        if response.status_code != 200:
            print("Error Update Trade : " + str(response.status_code))
            print("Reason: " + response.reason)
            print("URL " + url)
            return False
        pp.pprint(response.json())
        return response.json()

    def modify_order(self, symbol, order, stoploss=0.0, takeprofit=0.0, Guaranteed=False):
        orderID = order["OrderId"]
        qty = order["Orders"][0]["Quantity"]
        IfDone = order["Orders"][0]["IfDone"]
        cmd = order["Direction"]

        if cmd == self.OP_BUY:
            oppcmd = self.OP_SELL
        else:
            oppcmd = self.OP_BUY

        stoploss = round(stoploss, self.market_info[symbol]["PriceDecimalPlaces"])
        takeprofit = round(takeprofit, self.market_info[symbol]["PriceDecimalPlaces"])

        data = {
            "MarketId": self.market_info[symbol]["MarketId"],
            "OrderId": orderID,
            "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
            "IfDone": [],
            "Direction": cmd  # must have!
        }

        if stoploss > 0.0:
            stopLossData = {"Stop": {
                "TriggerPrice": stoploss,
                # "OrderId" : order["Orders"][0]["IfDone"][0]["Stop"]["OrderId"],
                "Direction": oppcmd,
                "Quantity": qty
                # "ParentOrderId":order["Orders"][0]["IfDone"][]
            }}

            for stoplimitorder in IfDone:
                if stoplimitorder["Stop"] is not None:
                    stopLossData["Stop"]["OrderId"] = stoplimitorder["Stop"]["OrderId"]

            data["IfDone"].append(stopLossData)

        if takeprofit > 0.0:
            limitData = {"Limit": {
                "TriggerPrice": takeprofit,
                "Direction": cmd,
                "Quantity": qty
                # "ParentOrderId":orderID
            }}
            for stoplimitorder in IfDone:
                if stoplimitorder["Limit"] is not None:
                    limitData["Limit"]["OrderId"] = stoplimitorder["Limit"]["OrderId"]
            data["IfDone"].append(limitData)

        url = self.APIURL + '/order/updatetradeorder?Username=' + self.uid + "&Session=" + self.session
        headers = {'Content-type': 'application/json'}
        pp.pprint(data)
        response = requests.post(url, json.dumps(data), headers=headers)
        if response.status_code != 200:
            print("Error Update Trade : " + str(response.status_code))
            print("Reason: " + response.reason)
            print("URL " + url)
            return False
        pp.pprint(response.json())
        return response.json()

    def send_market_order(self, symbol, cmd, qty, data, stoploss=0.0, takeprofit=0.0, Guaranteed=True):
        try:
            logging.info(f"Placing market order: {symbol} | {cmd} | Qty: {qty} | SL: {stoploss} | TP: {takeprofit}")
            
            # Round values according to symbol decimals
            stoploss = round(stoploss, self.market_info[symbol]["PriceDecimalPlaces"])
            takeprofit = round(takeprofit, self.market_info[symbol]["PriceDecimalPlaces"])
            qty = round(qty, 0)
            
            # Validate quantity
            if qty < self.market_info[symbol]['WebMinSize']:
                logging.warning(f"Order qty {qty} is below WebMinSize [{self.market_info[symbol]['WebMinSize']}]")
                return False
                
            # Get current market prices
            price = data["Offer"] if cmd == "BUY" else data["Bid"]
            
            url = self.APIURL + '/order/newtradeorder?Username=' + self.uid + "&Session=" + self.session
            order_data = {
                "IfDone": [],
                "Direction": cmd,
                "Quantity": qty,
                "MarketId": self.market_info[symbol]["MarketId"],
                "BidPrice": data["Bid"],
                "OfferPrice": data["Offer"],
                "AuditId": data["AuditId"],
                "AutoRollover": True,
                "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                "PositionMethodId": 2,
                "Currency": self.market_info[symbol]["MarketSizesCurrencyCode"],
                "isTrade": True
            }
    
            # Add stop loss if specified
            if stoploss > 0.0:
                sl_direction = "SELL" if cmd == "BUY" else "BUY"
                order_data["IfDone"].append({
                    "Stop": {
                        "TriggerPrice": stoploss,
                        "Direction": sl_direction,
                        "MarketId": self.market_info[symbol]["MarketId"],
                        "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                        "Quantity": qty,
                        "IsGuaranteed": Guaranteed
                    }
                })
    
            # Add take profit if specified
            if takeprofit > 0.0:
                tp_direction = "SELL" if cmd == "BUY" else "BUY"
                order_data["IfDone"].append({
                    "Limit": {
                        "TriggerPrice": takeprofit,
                        "Direction": tp_direction,
                        "MarketId": self.market_info[symbol]["MarketId"],
                        "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                        "Quantity": qty
                    }
                })
    
            headers = {'Content-type': 'application/json'}
            response = requests.post(url, json.dumps(order_data), headers=headers)
        
            if response.status_code == 200:
                order_response = response.json()
                logging.info(f"Order placed successfully:")
                logging.info(f"Symbol: {symbol}")
                logging.info(f"Direction: {cmd}")
                logging.info(f"Quantity: {qty}")
                logging.info(f"Entry Price: {price}")
                logging.info(f"Stop Loss: {stoploss}")
                logging.info(f"Take Profit: {takeprofit}")
                logging.info(f"Order ID: {order_response.get('OrderId', 'N/A')}")
                return order_response
            else:
                logging.error(f"Order failed - Status: {response.status_code}")
                logging.error(f"Error message: {response.text}")
                return False
    
        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            return False
    
    

    def cross_rate(self, symbol, cmd, data):
       
        # Extract just the signal if cmd is a dictionary
        direction = cmd.get('signal') if isinstance(cmd, dict) else cmd
        
        # If no valid direction, default to "BUY"
        if not direction:
            direction = "BUY"
        
        qty = self.market_info[symbol]["WebMinSize"]
        simulatedOrder = self.simulate_trade_order(symbol, direction, qty, data)
        # Convert direction dictionary to string if needed
        if isinstance(cmd, dict):
            cmd = cmd.get('signal', 'BUY')  # Default to BUY if no signal
            
        # Convert float32 values to standard Python floats
        clean_data = {
            "Bid": float(data["Bid"]),
            "Offer": float(data["Offer"]),
            "AuditId": str(data["AuditId"])
        }
        
        qty = float(self.market_info[symbol]["WebMinSize"])
        return self.simulate_trade_order(symbol, cmd, qty, clean_data)

    
class Context:
    lightstreamer_url = 'https://push.cityindex.com'

    def __init__(self, symbol, ea_param, time_interval, time_span, handle_data, bars_count="65000"):
        self.high = []
        self.low = []
        self.close = []
        self.open = []
        self.time = []
        self.data = {}
        self.indicators = []
        self.clientAccountMarginData = {}
        self.EAParam = ea_param
        self.symbol = symbol
        self.TimeInterval = time_interval
        self.TimeSpan = time_span
        self.BarsCount = bars_count
        self.handle_data = handle_data
        self.LS_PRICE_DATA_ADAPTER = "PRICES"
        self.LS_PRICE_ID = "PRICE."
        self.LS_PRICE_SCHEMA = ["MarketId", "AuditId", "Bid",
                                "Offer", "Change", "Direction",
                                "High", "Low", "Price", "StatusSummary",
                                "TickDate"]
        self.LS_MARGIN_DATA_ADAPTER = "CLIENTACCOUNTMARGIN"
        self.LS_MARGIN_ID = "CLIENTACCOUNTMARGIN"
        self.LS_MARGIN_SCHEMA = ["Cash",
                                 "CurrencyId",
                                 "CurrencyISO",
                                 "Margin",
                                 "MarginIndicator",
                                 "NetEquity",
                                 "OpenTradeEquity",
                                 "TradeableFunds",
                                 "PendingFunds",
                                 "TradingResource",
                                 "TotalMarginRequirement"]

    def init_data(self):
        api = API()
        self.LS_PRICE_ID = self.LS_PRICE_ID + str(api.market_info[self.symbol]["MarketId"])
        self.clientAccountMarginData = api.get_client_account_margin()

        priceBars = False
        while not priceBars:
            priceBars = api.get_pricebar_history(self.symbol,
                                                 self.TimeInterval,
                                                 self.TimeSpan,
                                                 self.BarsCount)

        for counter, price in enumerate(priceBars["PriceBars"]):
            self.high.insert(0, price["High"])
            self.low.insert(0, price["Low"])
            self.close.insert(0, price["Close"])
            self.open.insert(0, price["Open"])
            self.time.insert(0, wcfDate2Sec(price["BarDate"]))

        partialBar = priceBars["PartialPriceBar"]

        self.high.insert(0, partialBar["High"])
        self.low.insert(0, partialBar["Low"])
        self.close.insert(0, partialBar["Close"])
        self.open.insert(0, partialBar["Open"])
        self.time.insert(0, wcfDate2Sec(partialBar["BarDate"]))

    def update_data(self):
        api = API()
        last_time_sec = self.time[0];
        time_diff = int((time.time() - last_time_sec) / intervalUnitSec(self))

        priceBars = False
        while not priceBars:
            priceBars = api.get_pricebar_history(self.symbol,
                                                 self.TimeInterval,
                                                 self.TimeSpan,
                                                 str(time_diff + 2))

        partialBar = priceBars["PartialPriceBar"]
        curTime = wcfDate2Sec(partialBar["BarDate"])

        self.high[0] = partialBar["High"]
        self.low[0] = partialBar["Low"]
        self.open[0] = partialBar["Open"]
        self.close[0] = partialBar["Close"]
        self.time[0] = curTime

        for counter, price in enumerate(priceBars["PriceBars"]):
            time_sec = wcfDate2Sec(price["BarDate"])
            time_diff = int((time_sec - self.time[1]) / intervalUnitSec(self))

            if time_diff > 0:
                self.high.insert(1, price["High"])
                self.low.insert(1, price["Low"])
                self.close.insert(1, price["Close"])
                self.open.insert(1, price["Open"])
                self.time.insert(1, time_sec)

    def prepare_data(self, data):
        tableNo = data["_tableIdx_"]
        if tableNo == 1:
            data['TickDate'] = wcfDate2Sec(data['TickDate'])
            if data['Offer']:
                data['Offer'] = float(data['Offer'])
            else:
                if len(self.data) > 0:
                    if self.data["Offer"]:
                        data['Offer'] = self.data["Offer"]

            if data['Bid']:
                data['Bid'] = float(data['Bid'])
            else:
                if len(self.data) > 0:
                    if self.data["Bid"]:
                        data['Bid'] = self.data["Bid"]
            self.data = data

            self.update_data()

            for counter, indicator in enumerate(self.indicators):
                indicator.onCalculate(self, len(self.time))
            self.handle_data(self, data)
        elif tableNo == 2:
            if data["Cash"]:
                self.clientAccountMarginData["Cash"] = data["Cash"]
            if data["CurrencyISO"]:
                self.clientAccountMarginData["CurrencyISO"] = data["CurrencyISO"]
            if data["CurrencyId"]:
                self.clientAccountMarginData["CurrencyId"] = data["CurrencyId"]
            if data["Margin"]:
                self.clientAccountMarginData["Margin"] = data["Margin"]
            if data["MarginIndicator"]:
                self.clientAccountMarginData["MarginIndicator"] = data["MarginIndicator"]
            if data["NetEquity"]:
                self.clientAccountMarginData["NetEquity"] = data["NetEquity"]
            if data["OpenTradeEquity"]:
                self.clientAccountMarginData["OpenTradeEquity"] = data["OpenTradeEquity"]
            if data["PendingFunds"]:
                self.clientAccountMarginData["PendingFunds"] = data["PendingFunds"]
            if data["TotalMarginRequirement"]:
                self.clientAccountMarginData["TotalMarginRequirement"] = data["TotalMarginRequirement"]
            if data["TradeableFunds"]:
                self.clientAccountMarginData["TradeableFunds"] = data["TradeableFunds"]
            if data["TradingResource"]:
                self.clientAccountMarginData["TradingResource"] = data["TradingResource"]

class TradingStrategy:
    def __init__(self, api, symbol, capital=500, risk_percent=3):
        self.api = api
        self.symbol = symbol
        self.capital = capital
        self.risk_percent = risk_percent
        self.timeframe = "15"
        self.position = None
        self.event_log = []  # Initialize event_log in constructor
        
        # Initialize API connection
        if not self.api.session:
            self.api.login()
            
        # Load market info
        self.api.get_full_market_info("81")
        self.api.get_full_market_info("146")
        
        # Map symbol
        for market_key, market_data in self.api.market_info.items():
            if market_data['UnderlyingRicCode'] == self.symbol:
                self.symbol_mapping = market_key
                self.market_id = market_data['MarketId']
                break
    
    def calculate_kama(self, data, n=10, fast=2, slow=30):
        close = np.array([bar['Close'] for bar in data])
        change = np.abs(close[1:] - close[:-1])
        volatility = np.array([np.sum(change[max(0, i-n+1):i+1]) for i in range(len(change))])
        er = np.abs(close[n:] - close[:-n]) / volatility[n-1:]
        
        fast_sc = 2/(fast+1)
        slow_sc = 2/(slow+1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = np.zeros_like(close)
        kama[n-1] = close[n-1]
        
        for i in range(n, len(close)):
            kama[i] = kama[i-1] + sc[i-n] * (close[i] - kama[i-1])
        
        return kama

    
    def calculate_bb_width(self, data, period=20):
        close = np.array([bar['Close'] for bar in data])
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (upper - lower) / sma
    
    def calculate_adx(self, data, period=14):
        high = np.array([bar['High'] for bar in data])
        low = np.array([bar['Low'] for bar in data])
        close = np.array([bar['Close'] for bar in data])
        
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        
        pos_dm = np.where((up > down) & (up > 0), up, 0)
        neg_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr_smooth = self.smooth(tr, period)
        pos_dm_smooth = self.smooth(pos_dm, period)
        neg_dm_smooth = self.smooth(neg_dm, period)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        pos_di = 100 * pos_dm_smooth / (tr_smooth + eps)
        neg_di = 100 * neg_dm_smooth / (tr_smooth + eps)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + eps)
        adx = self.smooth(dx, period)
        
        return adx[-1]

    
    def calculate_choppiness(self, data, period=14):
        high = np.array([bar['High'] for bar in data])
        low = np.array([bar['Low'] for bar in data])
        close = np.array([bar['Close'] for bar in data])
        
        tr_sum = np.sum([np.max([high[i] - low[i],
                                abs(high[i] - close[i-1]),
                                abs(low[i] - close[i-1])])
                        for i in range(1, period+1)])
        
        range_high = np.max(high[-period:])
        range_low = np.min(low[-period:])
        
        chop = 100 * np.log10(tr_sum / (range_high - range_low)) / np.log10(period)
        return chop
    
    def check_conditions(self, data):
        indicators = {
            'kama': self.calculate_kama(data),
            'bb_width': self.calculate_bb_width(data) * 100,
            'adx': self.calculate_adx(data),
            'chop': self.calculate_choppiness(data)
        }
        
        current_price = data[-1]['Close']
        
        valid_conditions = (
            indicators['bb_width'] <= 7 and
            indicators['adx'] >= 50 and
            indicators['chop'] <= 50
        )
        
        if valid_conditions:
            if current_price > indicators['kama'][-1]:
                return 'BUY'
            elif current_price < indicators['kama'][-1]:
                return 'SELL'
        
        return None
    
    def calculate_atr(self, data, period=14):
        tr = []
        for i in range(1, len(data)):
            high = data[i]['High']
            low = data[i]['Low']
            prev_close = data[i-1]['Close']
            tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        return np.mean(tr[-period:])
        
    def smooth(self, data, period):
        """
        Exponential smoothing function for indicator calculations
        """
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    def execute_trade(self, signal, data):
        # Initialize trading account info first
        self.api.trading_account_info = self.api.get_trading_account_info()
    
        current_bar = data[-1]
        entry_price = current_bar['Close']
        market_info = self.api.market_info[self.symbol]
        
        # Calculate ATR and price levels first
        atr = self.calculate_atr(data)
        stop_loss = round(entry_price - (atr * 2.5) if signal == "BUY" else entry_price + (atr * 2.5), 5)
        take_profit = round(entry_price + (atr * 2.5) if signal == "BUY" else entry_price - (atr * 2.5), 5)
        
        # Determine if forex or stock
        is_forex = self.symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'EURJPY']
        
        if is_forex:
            qty = max(self.calculate_position_size(entry_price, stop_loss), market_info.get('WebMinSize', 1000))
            market_data = self._prepare_forex_market_data(entry_price, market_info)
        else:
            if not self._is_stock_market_open():
                print(f"Market closed for {self.symbol}. Order queued for next session.")
                return None
            
            qty = max(self.calculate_position_size(entry_price, stop_loss), 100)
            market_data = self._prepare_stock_market_data(entry_price, market_info)
        
        order = self.api.send_market_order(
            symbol=self.symbol,
            cmd=signal,
            qty=qty,
            data=market_data,
            stoploss=stop_loss,
            takeprofit=take_profit
        )
        
        if order:
            self._log_trade(order, entry_price, qty)
        
        return order
    
    def _log_trade(self, order, entry_price, qty):
        timestamp = datetime.now(timezone.utc)
        print(f"[{timestamp}] Trade executed: {self.symbol} | Price: {entry_price} | Quantity: {qty}")
        
        if not hasattr(self, 'event_log'):
            self.event_log = []
            
        self.event_log.append({
            'timestamp': timestamp,
            'event': 'Trade',
            'symbol': self.symbol,
            'details': order
        })

    def _prepare_forex_market_data(self, price, market_info):
        return {
            "Bid": price - 0.0001,
            "Offer": price + 0.0001,
            "AuditId": str(int(time.time())),
            "MarketId": market_info["MarketId"],
            "TradingAccountId": self.api.trading_account_info["TradingAccounts"][0]["TradingAccountId"]
        }
    
    def _prepare_stock_market_data(self, price, market_info):
        return {
            "Bid": price - 0.01,
            "Offer": price + 0.01,
            "AuditId": str(int(time.time())),
            "MarketId": market_info["MarketId"],
            "TradingAccountId": self.api.trading_account_info["TradingAccounts"][0]["TradingAccountId"]
        }
    
    def _is_stock_market_open(self):
        now = datetime.now(timezone.utc)
        # US Market hours (UTC)
        market_open = now.replace(hour=13, minute=30, second=0)  # 9:30 AM EST
        market_close = now.replace(hour=20, minute=0, second=0)  # 4:00 PM EST
        return market_open <= now <= market_close and now.weekday() < 5
    
    def is_market_closed(self):
        market_info = self.api.market_info[self.symbol]
        current_timestamp = time.time()
        trading_start = wcfDate2Sec(market_info['TradingStartTimeUtc'])
        trading_end = wcfDate2Sec(market_info['TradingEndTimeUtc'])
        
        return current_timestamp < trading_start or current_timestamp > trading_end
    
    def send_limit_order(self, symbol, cmd, qty, price, stoploss=0.0, takeprofit=0.0):
        """Send a pending limit order"""
        market_data = {
            "Bid": price - 0.0001,
            "Offer": price + 0.0001,
            "AuditId": str(int(time.time()))
        }
        
        data = {
            "OcoOrder": None,
            "Direction": cmd,
            "BidPrice": market_data["Bid"],
            "AuditId": market_data["AuditId"],
            "AutoRollover": False,
            "MarketId": self.market_info[symbol]["MarketId"],
            "OrderTypeId": 3,  # Limit order type
            "OfferPrice": market_data["Offer"],
            "Quantity": qty,
            "TriggerPrice": price,
            "TradingAccountId": self.trading_account_info["TradingAccounts"][0]["TradingAccountId"]
        }
        
        url = self.APIURL + '/order/newtradeorder'
        headers = {'Content-type': 'application/json'}
        response = requests.post(url, json.dumps(data), headers=headers)
        
        return response.json() if response.status_code == 200 else False
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk parameters"""
        risk_amount = self.capital * (self.risk_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        position_size = risk_amount / risk_per_unit
        
        # Round to nearest 1000 units
        position_size = round(position_size / 1000) * 1000
        
        # Ensure minimum position size
        return max(position_size, 1000)
    
    def update_position(self, current_bar):
        if not self.position:
            return
            
        current_price = current_bar['Close']
        
        if self.position['type'] == 'BUY':
            if current_price <= self.position['stop_loss'] or current_price >= self.position['take_profit']:
                self.close_position()
        else:
            if current_price >= self.position['stop_loss'] or current_price <= self.position['take_profit']:
                self.close_position()
    
    def close_position(self):
        self.position = None
        
   
            
if __name__ == "__main__":
    load_dotenv()
    api = API(isLive=True)
    
    #pp = pprint.PrettyPrinter()
    #
    #print("Test Login...")
    #if api.login():
    #    print("\tSuccess!\n\n")
    #else:
    #    print("\tFailed\n\n")
    #    exit()
    #
    #print("Test Trading Account Information...")
    #if api.get_trading_account_info():
    #    pp.pprint(api.trading_account_info)
    #    print("\tSuccess!")
    #else:
    #    print("\tFailed\n\n")
    #    api.logout()
    #    exit()
    #
    #print("Test Margin Info...")
    #if api.get_client_account_margin():
    #    pp.pprint(api.client_account_margin)
    #    print("\tSuccess!")
    #else:
    #    print("\tFailed\n\n")
    #    api.logout()
    #    exit()
    #
    #print("Test Get Order History...")
    #orders = api.get_order_history()
    #if orders:
    #    pp.pprint(orders.orders)
    #    print("\tSuccess!")
    #else:
    #    print("\tFailed\n\n")
    #    api.logout()
    #    exit()
    #    
    #print("Test get FX-major (81) market info...")
    #if api.get_full_market_info("81"):
    #    pp.pprint(api.market_info)
    #    print("\tSuccess!")
    #else:
    #    print("\tFailed\n\n")
    #    api.logout()
    #    exit()
    #
    #
    #    
    #print("Test get popolar(146) market info...")
    #if api.get_full_market_info("146"):
    #    pp.pprint(api.market_info)
    #    print("\tSuccess!")
    #else:
    #    print("\tFailed\n\n")
    #    api.logout()
    #    exit()
    #
    
    # System checks
    print("Performing system checks...")
    if not api.login():
        print("Login failed")
        exit()
    
    symbols = ["USDJPY", "EURUSD", "GBPUSD", "EURJPY", "USDCHF", "USDCAD", 
               "NVDA.NB", "GOOGL.NB", "AMZN.NB", "AMC.N"]
               
    def print_market_conditions(symbol, price, bb_width, adx, chop, data):
        conditions_met = []
        kama = strategy.calculate_kama(data)
        trend = "BULLISH" if price > kama[-1] else "BEARISH"
        
        if bb_width <= 7:
            conditions_met.append("BB Width ✓")
        if adx >= 50:
            conditions_met.append("ADX ✓")
        if chop <= 50:
            conditions_met.append("Choppiness ✓")
            
        print(f"\n{symbol} Price: {price:.5f}")
        print(f"BB Width: {bb_width:.2f}")
        print(f"ADX: {adx:.2f}")
        print(f"Choppiness: {chop:.2f}")
        
        if len(conditions_met) >= 2:
            print(f"🎯 HIGH POTENTIAL - Conditions met: {', '.join(conditions_met)}")
            print(f"💹 TREND: {trend} | KAMA: {kama[-1]:.5f}")
            if trend == "BULLISH":
                print("⬆️ LONG SETUP")
            else:
                print("⬇️ SHORT SETUP")
    
        
                    
    strategies = {}
    for symbol in symbols:
        strategies[symbol] = TradingStrategy(api=api, symbol=symbol, capital=500, risk_percent=3)
    
    while True:
        try:
            for symbol, strategy in strategies.items():
                data = api.get_pricebar_history(symbol, "MINUTE", "15", "100", "bid")
                
                if data and 'PriceBars' in data:
                    current_price = data['PriceBars'][-1]['Close']
                    bb_width = strategy.calculate_bb_width(data['PriceBars']) * 100
                    adx = strategy.calculate_adx(data['PriceBars'])
                    chop = strategy.calculate_choppiness(data['PriceBars'])
                    kama = strategy.calculate_kama(data['PriceBars'])
                    
                    print_market_conditions(symbol, current_price, bb_width, adx, chop, data['PriceBars'])
                    
                    # Automatic Trade Execution
                    if bb_width <= 7 and chop <= 50:
                        if current_price > kama[-1]:
                            print(f"🔔 EXECUTING LONG TRADE: {symbol}")
                            order = strategy.execute_trade("BUY", data['PriceBars'])
                            print(f"Order placed: {order}")
                        elif current_price < kama[-1]:
                            print(f"🔔 EXECUTING SHORT TRADE: {symbol}")
                            order = strategy.execute_trade("SELL", data['PriceBars'])
                            print(f"Order placed: {order}")
                            
                    # Position Management
                    if strategy.position:
                        strategy.update_position(data['PriceBars'][-1])
            
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            api.logout()
            break
