import time
from datetime import datetime, timezone, timedelta
import logging
import json
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from CIApiorin import API ,Context
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from collections import deque
import os
import keras_tuner
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import pprint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurazione del logger
logging.basicConfig(
    filename="trading.log",  # Salva i log in un file
    level=logging.INFO,  # Livello di log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataStorage:
    def __init__(self, db_file):
        # Initialize connection to the database
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # Create a table if it does not exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_data
            (id INTEGER PRIMARY KEY, timestamp TEXT, action TEXT, symbol TEXT, details TEXT)
        ''')
        self.conn.commit()

    def save_data(self, data):
        # Save trading data into the database
        self.cursor.execute('''
            INSERT INTO trading_data (timestamp, action, symbol, details)
            VALUES (?, ?, ?, ?)
        ''', (data['timestamp'], data['action'], data['symbol'], str(data['details'])))
        self.conn.commit()
 
    def close_connection(self):
        # Close the connection to the database
        self.conn.close()

        
class TradingStrategy:
    # Timeframe settings
    INTERVAL = "MINUTE"  # Options: "MINUTE", "HOUR", "DAY"
    SPAN = "5"          # Time period (e.g., "1", "5", "15", "30")
    BARS = "50"         # Number of candles to analyze
    DEFAULT_THRESHOLDS = {
        'bb_width': 0.015,
        'adx': 18,
        'chop': 45
    }
    
    TRADING_SESSIONS = {
        'london_open': 8,
        'newyork_open': 14,
        'asia_open': 22
    }
    
    def __init__(self, api, symbols, capital=500, risk_percent=2):
        # Core initialization
        self.api = api
        self.api.trading_account_info = self.api.get_trading_account_info()
        self.symbols = symbols
        self.capital = capital
        self.risk_percent = risk_percent
    
        # Initialize components
        self.trailing_stop = {}
        self.market_data = {}
        self.historical_data = {}
        self.open_positions = {}
        self.event_log = []
        self.signal_history = {}
        
        # Add this configuration at startup
        tf.config.run_functions_eagerly(False)
        tf.keras.backend.clear_session()
        
        # Initialize storage and ML
        self.data_storage = DataStorage("trading_data.db")
        self.scaler = StandardScaler()
    
        #if os.path.exists("trading_model.h5"):
        #    self.model = tf.keras.models.load_model("trading_model.h5")
        #    print("Loaded pre-trained model.")
        #else:
        #    self.model = self.build_model()
        #    print("No pre-trained model found. Using a new model.")
    
        # Initialize markets
        self._initialize_markets()
    
        # Initialize market phase and last check time
        self.market_phase = 'neutral'
        self.last_phase_check = datetime.now() - timedelta(minutes=15)  # Inizializza last_phase_check
        self.error_log = []
        self.volatility_window = 20
        
    def get_client_account_margin(self):
        url = f"{self.base_url}/margin/clientaccountmargin?Username={self.username}&Session={self.session_id}"
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, verify=True, timeout=30)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt == max_retries - 1:
                    print(f"Connection failed after {max_retries} attempts. Refreshing session...")
                    self.login()  # Re-login to refresh session
                    return self.get_client_account_margin()  # Retry once more after login
                print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return None
    
    
    def is_market_open(self):
        """Check if Forex market is open"""
        current_time = datetime.now(timezone.utc)
        current_day = current_time.weekday()
    
        # Forex trades 24/5 - Only closed on weekends
        # Friday close at 21:00 UTC
        # Sunday open at 21:00 UTC
        if current_day == 5:  # Saturday
            return False
        elif current_day == 6:  # Sunday
            return current_time.hour >= 21
        elif current_day == 4:  # Friday
            return current_time.hour < 21
        else:  # Monday to Thursday
            return True
            
    def get_next_market_open(self):
        current = datetime.now(timezone.utc)
        if current.weekday() == 5:  # Saturday
            days_to_add = 1
            target_hour = 21
        elif current.weekday() == 6 and current.hour < 21:  # Sunday before market open
            days_to_add = 0
            target_hour = 21
        else:
            return None  # Market is open
    
        next_open = current.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if days_to_add:
            next_open += timedelta(days=days_to_add)
        return next_open
    
    def display_market_countdown(self):
        next_open = self.get_next_market_open()
        if not next_open:
            return False  # Market is open
            
        time_remaining = next_open - datetime.now(timezone.utc)
        hours = int(time_remaining.total_seconds() // 3600)
        minutes = int((time_remaining.total_seconds() % 3600) // 60)
        seconds = int(time_remaining.total_seconds() % 60)
        
        print(f"\rMarket closed | Opening in: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")
        return True  # Market is closed
        
    def train_model(self, historical_data, labels, epochs=50, batch_size=32):
        model_path = "trading_model.keras"
        
        # **Verifica l'uso della GPU e imposta XLA**
        tf.config.optimizer.set_jit(True)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {gpus}")
        else:
            print("⚠️ No GPU detected, training will be slower.")
        
        # **Se il modello esiste, caricalo**
        if os.path.exists(model_path):
            print("Loading existing model for fine-tuning...")
            self.model = tf.keras.models.load_model(model_path)
            # Add explicit fine-tuning call here
            self.fine_tune_model(historical_data, labels)
        else:
            print("No pre-trained model found. Creating a new model.")
            self.model = self.build_model()
            fine_tune_epochs = epochs

            
            # **Abilita esecuzione eager per debug**
            tf.config.run_functions_eagerly(False)  # 🚀 Disabilitato per performance migliori
            
            # **Pre-elaborazione dati più efficiente**
            X = np.array([self.prepare_ml_features(data) for data in historical_data], dtype=np.float32)
            X = X.reshape(-1, 50, 5)
            y = tf.keras.utils.to_categorical(labels, num_classes=3)
            
            # **Dividi training e validation set**
            split_index = int(0.8 * len(X))
            X_train, X_val = X[:split_index], X[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]
            
            # **Usa tf.data.Dataset per gestire i dati più velocemente**
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # **Calcola class weights per bilanciare le classi**
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(labels),
                y=labels
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            
            # **Definizione dei callback**
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # 🔥 Ridotto da 15 a 10 per evitare overfitting
                restore_best_weights=True
            )
            
            # **Compila il modello**
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # **Esegui l'addestramento**
            print("🚀 Starting training...")
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                class_weight=class_weight_dict,  # 🔥 Usa class_weight per bilanciare
                callbacks=[lr_scheduler, early_stopping]
            )
            
            # **Salva il modello aggiornato**
            self.model.save(model_path)
            print("✅ Model updated and saved successfully")
            
            return history
    
        
    def prepare_training_data(self, symbols, num_bars=1000):
        historical_data = []
        labels = []
    
        for symbol in symbols:
            price_data = self.api.get_pricebar_history(
                symbol=symbol,
                interval=self.INTERVAL,
                span=self.SPAN,
                pricebars=str(num_bars)
            )
    
            if price_data and "PriceBars" in price_data:
                bars = price_data["PriceBars"]
                formatted_data = []
    
                for i in range(len(bars) - 50):
                    sequence = bars[i:i+50]
                    formatted_data.append(sequence)
    
                    # Create labels based on price movement
                    current_close = float(sequence[-1]['Close'])
                    next_close = float(bars[i+50]['Close'])
                    if next_close > current_close * 1.001:
                        labels.append(2)  # BUY
                    elif next_close < current_close * 0.999:
                        labels.append(0)  # SELL
                    else:
                        labels.append(1)  # HOLD
    
                historical_data.extend(formatted_data)
    
        return historical_data, labels
    
            
    def fine_tune_model(self, historical_data, labels, epochs=1, batch_size=32):
        print("Starting fine-tuning process")
        print(f"Data shape: {len(historical_data)} samples")
        try:
            # Controlla se il modello è caricato
            if not hasattr(self, "model") or self.model is None:
                print("⚠️ No model found. Load or train a model before fine-tuning.")
                return
    
            # Congela i primi layer del modello per il fine-tuning
            for layer in self.model.layers[:5]:  # Puoi regolare il numero di layer
                layer.trainable = False
            
            # Prepara i dati
            X = np.array([self.prepare_ml_features(data) for data in historical_data])
            X = X.reshape(-1, 50, 5)
            y = tf.keras.utils.to_categorical(labels, num_classes=3)
    
            # Usa tf.data.Dataset per gestione ottimale dei dati
            dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(len(X)).prefetch(tf.data.AUTOTUNE)
    
            # Esegui il fine-tuning
            print("🚀 Fine-tuning in progress...")
            history = self.model.fit(
                dataset,
                epochs=epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2)
                ]
            )
    
            # Sblocca tutti i layer per il retraining completo (opzionale)
            for layer in self.model.layers:
                layer.trainable = True
    
            # Salva il modello aggiornato
            self.model.save("trading_model.keras")
            print("✅ Model fine-tuned and saved successfully.")
    
            # Visualizza i risultati del training
            #self.plot_training_history(history)
    
        except Exception as e:
            print(f"❌ Error during fine-tuning: {str(e)}")
    
    #Funzione per tracciare l'andamento dell'addestramento
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        # 📈 Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
        plt.legend()
        plt.title("📊 Model Accuracy")
    
        # 📉 Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history.get('val_loss', []), label='Validation Loss')
        plt.legend()
        plt.title("📉 Model Loss")
    
        plt.show()  
        
    def build_model(self):
        model = tf.keras.Sequential([
            # Enhanced input layer
            tf.keras.layers.LSTM(128, input_shape=(50, 5), return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Deeper architecture
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.LSTM(32),
            tf.keras.layers.BatchNormalization(),
            
            # Wider dense layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Optimized training configuration
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_ml_features(self, data):
        """Prepare features for ML prediction with proper formatting"""
        if not data or len(data) < 50:
            return np.zeros((1, 50, 5))  # Return default shaped array
            
        # Extract and normalize price data
        features = []
        for i in range(len(data)-50+1):
            window = data[-50:] 
            
            # Calculate technical features
            window_features = [
                [float(bar['Close']) for bar in window],
                [float(bar['Volume']) for bar in window],
                [float(bar['High']) - float(bar['Low']) for bar in window],
                [float(bar['Close']) - float(bar['Open']) for bar in window],
                [(float(bar['High']) + float(bar['Low'])) / 2 for bar in window]
            ]
            
            # Normalize features
            normalized_features = []
            for feature_set in window_features:
                if max(feature_set) - min(feature_set) != 0:
                    normalized = [(x - min(feature_set)) / (max(feature_set) - min(feature_set)) for x in feature_set]
                else:
                    normalized = [0] * len(feature_set)
                normalized_features.append(normalized)
                
            features = np.array(normalized_features).T
            
        return np.expand_dims(features, axis=0)  # Shape: (1, 50, 5)

    

    def get_ml_prediction(self, symbol, data):
        features = self.prepare_ml_features(data)
        print(f"Features for {symbol}: {features}")  # Log per verificare le features
        prediction = self.model.predict(features)
        print(f"Prediction for {symbol}: {prediction}")  # Log per verificare la predizione
        return np.argmax(prediction[0])


    def analyze_performance(self):
        """Analyze the performance of the executed trades."""
        profits = []
        for log in self.event_log:
            if log['action'] == 'CLOSE':
                profit = log['details'].get('profit', 0)
                profits.append(profit)
                print(f"Trade Closed - Profit: {profit:.2f}")
        
        total_profit = sum(profits)
        win_trades = len([p for p in profits if p > 0])
        total_trades = len(profits)
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"[PERFORMANCE] Total Profit: {total_profit:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {total_trades}")
        
        return total_profit
    

        
    def _initialize_markets(self):
        if not self.api.session:
            self.api.login()
            
        market_info = self.api.get_full_market_info("80")
        market_info = self.api.get_full_market_info("146")
        #print(f"Market Info: {market_info}")  # Log per verificare i dati di mercato
        
        self.market_data = {
            symbol: market_info[symbol.replace("/", "")] 
            for symbol in self.symbols 
            if symbol.replace("/", "") in market_info
        }
        #print(f"Market Data: {self.market_data}")  # Log per verificare i dati di mercato
        
        for symbol in self.symbols:
            price_data = self.api.get_pricebar_history(
                symbol=symbol,
                interval=self.INTERVAL,
                span=self.SPAN,
                pricebars=self.BARS
            )
            
            if price_data and "PriceBars" in price_data:
                recent_data = price_data["PriceBars"]
                self.historical_data[symbol] = {
                    'bb_width': [self.calculate_bb_width(recent_data)],
                    'adx': [self.calculate_adx(recent_data)],
                    'chop': [self.calculate_choppiness(recent_data)]
                }
    
    def log_action(self, action_type, symbol, details):
        """Log actions such as order execution, updates, and closures."""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'action': action_type,
            'symbol': symbol,
            'details': details
        }
        print(f"[LOG] {log_entry}")
        self.event_log.append(log_entry)
        self.data_storage.save_data(log_entry)

    def monitor_symbols(self):
        try:
            last_training = datetime.now()
            while True:
                current_time = datetime.now(timezone.utc)
                current_day = current_time.weekday()
                
                ## Market hours check with countdown
                #if current_day == 5 or (current_day == 6 and current_time.hour < 21):
                #    next_open = self.get_next_market_open()
                #    time_remaining = next_open - current_time
                #    hours = int(time_remaining.total_seconds() // 3600)
                #    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                #    seconds = int(time_remaining.total_seconds() % 60)
                #    print(f"\rMarket closed | Next open in: {hours:02d}:{minutes:02d}:{seconds:02d}", end="")
                #    time.sleep(1)
                #    continue
                #
                ## Weekly retraining check
                #if (datetime.now() - last_training).days >= 7:
                #    historical_data, labels = self.prepare_training_data(self.symbols)
                #    self.train_model(historical_data, labels)
                #    last_training = datetime.now()
                #
                #print(f"\nStarting monitoring cycle at {current_time}")
                
                # Market phase update (15-minute intervals)
                if (datetime.now() - self.last_phase_check).seconds > 900:
                    self.market_phase = self.check_market_phase()
                    self.last_phase_check = datetime.now()
                    print(f"Updated market phase: {self.market_phase}")
                
                # Symbol monitoring
                for symbol in self.symbols:
                    self.monitor_symbol(symbol)
                    self.verify_ai_interaction()
                
                self.analyze_performance()
                self.print_summary()
                print(f"Cycle complete. Waiting 60 seconds...")
                time.sleep(60)
    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        finally:
            self.cleanup()
    
        
                
    def monitor_symbol(self, symbol):
        """Monitor individual symbol price and conditions"""
        if symbol not in self.market_data:
            return
    
        # Fetch latest price data
        price_data = self.api.get_pricebar_history(
            symbol=symbol,
            interval="MINUTE",
            span="1",
            pricebars="1"
        )
    
        if price_data and "PartialPriceBar" in price_data:
            current_price = price_data["PartialPriceBar"]["Close"]
            print(f"{symbol}: {current_price}")
            
            # Log the price monitoring action
            self.log_action("MONITOR_PRICE", symbol, {
                'current_price': current_price,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
            current_data = self.fetch_latest_data(symbol)
            if symbol in self.open_positions:
                if self.monitor_position_profit(symbol, current_data):
                    print(f"Profit secured for {symbol} due to trend reversal")
    
    def get_validated_prices(self, symbol):
        """Fetch and validate real-time Bid/Offer prices"""
        price_data = self.api.get_pricebar_history(
            symbol=symbol,
            interval="MINUTE",
            span="1",
            pricebars="1"
        )
        
        if price_data and "PartialPriceBar" in price_data:
            partial_bar = price_data["PartialPriceBar"]
            close_price = float(partial_bar["Close"])
            
            # Enhanced pip value calculation
            if "JPY" in symbol:
                pip_value = 0.01
            elif ".NB" in symbol or "." in symbol:
                pip_value = 0.01
            else:
                pip_value = 0.0001 if len(str(close_price).split('.')[-1]) == 4 else 0.01
                
            spread = pip_value * 2  # Default 2 pip spread
            
            return {
                "Bid": close_price - spread,
                "Offer": close_price + spread,
                "Close": close_price
            }
        
        return None

    def calculate_sl_tp_levels(self, direction, current_price, atr):
        """Calculate Stop Loss and Take Profit levels with correct directional alignment"""
        multiplier_sl = 1.5
        multiplier_tp = 3.0
        
        if direction == "BUY":
            stop_loss = current_price - (atr * multiplier_sl)
            take_profit = current_price + (atr * multiplier_tp)
        else:  # SELL
            stop_loss = current_price + (atr * multiplier_sl)
            take_profit = current_price - (atr * multiplier_tp)
        
        return round(stop_loss, 5), round(take_profit, 5)


    #def execute_or_update_trade(self, symbol, direction, current_price, data):
    #    try:
    #        prices = self.get_validated_prices(symbol)
    #        if not prices:
    #            return False
    #
    #        atr = self.calculate_atr(data)
    #        stop_loss, take_profit = self.calculate_sl_tp_levels(direction, current_price, atr)
    #        
    #        # Add validation here
    #        if not self.validate_order_levels(direction, current_price, stop_loss, take_profit):
    #            print(f"[INFO] Invalid price levels for {symbol}: SL={stop_loss}, TP={take_profit}")
    #            return False
    #
    #        market_data = {
    #            "symbol": symbol,
    #            "cmd": direction,
    #            "qty": self.calculate_position_size(self.calculate_risk_amount()['RiskAmount'], atr, data, symbol),
    #            "data": {
    #                "AuditId": str(int(time.time())),
    #                "MarketId": self.api.market_info[symbol]["MarketId"],
    #                "TradingAccountId": self.api.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
    #                "Bid": prices["Bid"],
    #                "Offer": prices["Offer"]
    #            },
    #            "stoploss": stop_loss,
    #            "takeprofit": take_profit
    #        }
    #        
    #        order = self.api.send_market_order(**market_data)
    #        return self.handle_order_response(order, direction, symbol)
    #
    #    except Exception as e:
    #        print(f"[INFO] Trade execution error for {symbol}: {str(e)}")
    #        return False
    #
    #def validate_order_levels(self, direction, entry_price, sl_price, tp_price):
    #    if direction == "BUY":
    #        return sl_price < entry_price and tp_price > entry_price
    #    else:  # SELL
    #        return sl_price > entry_price and tp_price < entry_price
    #
    
    def handle_order_response(self, order, direction, symbol):
        """
        Handle the API order response and validate execution status
        """
        if not order:
            print(f"[INFO] No order response received for {symbol}")
            return False
            
        if order.get('Status') == 2:  # Success status
            print(f"[SUCCESS] {direction} order executed for {symbol}")
            
            # Log order details
            if 'Orders' in order and order['Orders']:
                order_details = order['Orders'][0]
                print(f"Order ID: {order_details.get('OrderId')}")
                print(f"Quantity: {order_details.get('Quantity')}")
                
                # Log stop loss and take profit
                if_done_orders = order_details.get('IfDone', [])
                for if_done in if_done_orders:
                    if 'Stop' in if_done and if_done['Stop']:
                        print(f"Stop Loss: {if_done['Stop'].get('TriggerPrice')}")
                    if 'Limit' in if_done and if_done['Limit']:
                        print(f"Take Profit: {if_done['Limit'].get('TriggerPrice')}")
            
            return True
            
        else:
            status_reason = order.get('StatusReason', 'Unknown')
            print(f"[INFO] Order not executed for {symbol}. Status: {order.get('Status')}, Reason: {status_reason}")
            return False
                                                
    def monitor_position_profit(self, symbol, current_data):
        position = self.open_positions[symbol]
        current_profit = self.calculate_current_profit(position)
        
        # Nuova logica di trailing stop
        if symbol in self.trailing_stop:
            ts_settings = self.trailing_stop[symbol]
            if current_profit >= ts_settings['activation'] * position['target_profit']:
                new_stop = self.calculate_trailing_stop(
                    position['direction'],
                    float(current_data[-1]['Close']),
                    ts_settings['distance']
                )
                if (position['direction'] == 'BUY' and new_stop > position['stop_loss']) or \
                (position['direction'] == 'SELL' and new_stop < position['stop_loss']):
                    self.api.update_order_stop_loss(position['OrderId'], new_stop)
                    self.log_action("TRAILING_STOP_UPDATE", symbol, {
                        "old_stop": position['stop_loss'],
                        "new_stop": new_stop
                    })
            
            # Use your existing API call
            if new_stop:
                self.api.update_order_stop_loss(position['OrderId'], new_stop)
            
            # Your existing trend reversal check
            new_signal = self.check_conditions(current_data, symbol)
            if self.detect_trend_reversal(position['direction'], new_signal):
                self.close_position(symbol)
                return True
        
        return False
        
    def calculate_trailing_stop(self, direction, current_price, trail_distance):
        """New helper function for trailing stop calculation"""
        return (current_price - trail_distance if direction == 'BUY' 
                else current_price + trail_distance)
                
    def detect_trend_reversal(self, current_direction, new_signal):
        """Detect if market direction is changing"""
        return (
            (current_direction == 'BUY' and new_signal == 'SELL') or
            (current_direction == 'SELL' and new_signal == 'BUY')
        )
        
    def fetch_latest_data(self, symbol):
        price_data = self.api.get_pricebar_history(
            symbol=symbol,
            interval="MINUTE",
            span="5",
            pricebars="50"
        )
        #print(f"Price Data for {symbol}: {price_data}")  # Log per verificare i dati storici
        
        if price_data and "PriceBars" in price_data:
            bars = price_data["PriceBars"]
            formatted_data = []
            
            for bar in bars:
                formatted_data.append({
                    'Open': bar['Open'],
                    'High': bar['High'],
                    'Low': bar['Low'],
                    'Close': bar['Close'],
                    'Volume': bar.get('TickCount', 0),
                    'Timestamp': bar['BarDate']
                })
                
            return formatted_data
        
        return None
        
    def optimize_base_position(self, base_size, market_phase):
        """Helper method to optimize the initial position size calculation"""
        phase_multiplier = {
            'trend': 1.2,
            'ranging': 0.8,
            'neutral': 1.0
        }
        
        return base_size * phase_multiplier[market_phase]
    
    def calculate_position_size(self, risk_amount, atr, data, symbol):
        """Calcola la dimensione della posizione con gestione avanzata del rischio
        
        Parametri:
        risk_amount (float): Capitale da rischiare per questa operazione
        atr (float): Valore ATR corrente
        data (list): Dati storici dei prezzi
        symbol (str): Simbolo del mercato
        
        Restituisce:
        int: Dimensione della posizione in lotti
        """
        try:
            # 1. Calcolo base della dimensione
            stop_loss_pips = atr * 4.0
            base_position_size = risk_amount / stop_loss_pips
    
            # 2. Fattore di volatilità regolato
            bb_width = self.calculate_bb_width(data) * 100
            # Applica una funzione logaritmica per smussare l'impatto
            volatility_factor = 1 + min(bb_width / 0.5, 2.0)  # Cap al 200% di aumento
            
            # 3. Adattamento alla fase del mercato
            market_phase_factor = 1.2 if self.market_phase == 'trend' else 0.8
    
            # 4. Fattore temporale (orario di trading)
            time_factor = 1.3 if self.is_london_newyork_overlap() else 0.7
    
            # 5. Considerazione dello spread
            current_spread = self.get_current_spread(data[-1])
            spread_factor = max(0.8, 1 - (current_spread / (atr * 0.1)))  # Riduce dimensione se spread > 10% ATR
    
            # 6. Calcolo della dimensione grezza
            raw_size = base_position_size * volatility_factor * market_phase_factor * time_factor * spread_factor
    
           # 7. Gestione esposizione e correlazione
            active_positions = len(self.open_positions)
            correlated_positions = self.count_correlated_positions(symbol)  # Use passed symbol parameter
            exposure_factor = 1 / (1 + (active_positions * 0.2) + (correlated_positions * 0.3))
            
            # 8. Limiti di sicurezza
            max_exposure = self.capital * 0.25  # Massimo 25% del capitale
            min_size = 500  # Dimensione minima
            max_size = 5000  # Dimensione massima
            
            # 9. Calcolo finale con limiti
            adjusted_size = raw_size * exposure_factor
            constrained_size = max(min_size, min(max_size, adjusted_size))
            
            # 10. Arrotondamento intelligente
            lot_step = 100  # Passo minimo del broker
            final_size = round(constrained_size / lot_step) * lot_step
    
            # Logging dettagliato
            print(f"\n{'='*40}")
            print("CALCOLO DIMENSIONE POSIZIONE:")
            print(f"Base: {base_position_size:.2f}")
            print(f"Fattori moltiplicativi:")
            print(f"• Volatilità ({bb_width:.2f}%): x{volatility_factor:.2f}")
            print(f"• Fase mercato ({self.market_phase}): x{market_phase_factor:.2f}")
            print(f"• Orario: x{time_factor:.2f}")
            print(f"• Spread ({current_spread*10000:.1f} pip): x{spread_factor:.2f}")
            print(f"• Esposizione (posizioni {active_positions}, correlate {correlated_positions}): x{exposure_factor:.2f}")
            print(f"Dimensione finale: {final_size}")
            print(f"{'='*40}\n")
    
            return int(final_size)
    
        except Exception as e:
            print(f"Errore nel calcolo della posizione: {str(e)}")
            return 1000  # Dimensione di fallback sicura
    
            
    def get_performance_multiplier(self, symbol):
        # Start with baseline multiplier
        multiplier = 1.0
        
        # Get recent win rate for symbol
        win_rate = self.get_symbol_win_rate(symbol)
        if win_rate > 0.6:
            multiplier *= 1.1
        elif win_rate < 0.4:
            multiplier *= 0.9
            
        return multiplier
    
    def get_market_state_multiplier(self, data):
        # Analyze recent price action
        trend = self.calculate_trend_strength([bar['Close'] for bar in data])
        volatility = self.calculate_bb_width(data)
        
        # Adjust multiplier based on market conditions
        multiplier = 1.0 + (trend * 0.1) + (volatility * 0.1)
        
        return min(multiplier, 1.25)
    
        
    def get_symbol_win_rate(self, symbol):
        """Calculate win rate for specific symbol"""
        total_trades = 0
        winning_trades = 0
        
        for log in self.event_log:
            if log['symbol'] == symbol and log['action'] == 'CLOSE':
                total_trades += 1
                if log['details'].get('profit', 0) > 0:
                    winning_trades += 1
        
        return winning_trades / total_trades if total_trades > 0 else 0.5

   

    def get_current_spread(self, price_data):
        """Calculate spread from price data with safety checks"""
        try:
            if isinstance(price_data, dict):
                if 'Bid' in price_data and 'Offer' in price_data:
                    return (float(price_data['Offer']) - float(price_data['Bid'])) / float(price_data['Bid'])
                elif 'Close' in price_data:
                    # Use typical spread value when Bid/Offer not available
                    return 0.0002  # 2 pips default spread
        except (KeyError, TypeError, ValueError):
            return 0.0002
        return 0.0002
    
    def count_correlated_positions(self, symbol):
        """Conta le posizioni correlate aperte"""
        correlation_groups = {
            'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'USDCHF'],
            'EUR': ['EURUSD', 'EURGBP', 'EURJPY'],
            'Commodities': ['XAUUSD', 'XAGUSD', 'OIL']
        }
        
        current_group = next((g for g in correlation_groups if symbol in correlation_groups[g]), None)
        return sum(1 for p in self.open_positions if p in correlation_groups.get(current_group, []))
        
    def get_correlation_adjustment(self):
        """New helper function for correlation-based position sizing"""
        active_positions = len(self.open_positions)
        return 1.0 / (1 + (active_positions * 0.1))  # Reduce size with more correlated positions

 

    def verify_ai_interaction(self):
        results = {}
        orders = self.api.get_orders()
        active_symbols = {order['MarketName'].replace('/', '') for order in orders.orders}
    
        for symbol in self.symbols:
            if symbol in active_symbols:
                print(f"Skip {symbol}: Active position found")
                continue
    
            # Fetch market data
            market_data = {
                'H4': self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="1000")['PriceBars'],
                'M15': self.api.get_pricebar_history(symbol, interval="MINUTE", span="15", pricebars="1000")['PriceBars'],
                'M1': self.api.get_pricebar_history(symbol, interval="MINUTE", span="1", pricebars="200")['PriceBars']
            }
    
            # Detect manipulation
            manipulation_data = self.detect_manipulation_candles(market_data['M1'])
    
            # Check initial signal
            initial_signal_data = self.check_conditions(market_data['M1'], symbol)
    
            ## Validate initial_signal_data
            #if not initial_signal_data:
            #    print(f"[INFO] No valid signal detected for {symbol}. Skipping...")
            #    continue
            #
            #print(f"\n=== PRIMA ANALISI: {symbol} ===")
            #print(f"Signal: {initial_signal_data}")
            #print(f"Manipulation detected: {manipulation_data}")
            #
            ## Wait and confirm signal
            #time.sleep(60)  # Wait 1 minute
            #confirmation_data = self.check_conditions(market_data['M1'], symbol)
            #
            ## Validate signal stability by comparing initial and confirmation data
            #if initial_signal_data == confirmation_data:
            #    print(f"[INFO] Signal confirmed for {symbol}. Proceeding with trade.")
            #else:
            #    print(f"[INFO] Signal unstable for {symbol}. Trade canceled.")
            #    continue
            #
            # Execute trade
            trade_result = self.execute_or_update_trade(
                symbol=symbol,
                direction=initial_signal_data["signal"],  # Extract direction from dictionary
                current_price=float(market_data['M1'][-1]['Close']),
                data=market_data['M1']
            )
    
            results[symbol] = {
                'signal': initial_signal_data,
                'result': trade_result
            }
    
        return results


    def calculate_risk_amount(self):
        """
        Calculate risk amount based on margin and risk percentage.
        Handles missing keys and invalid responses gracefully.
        """
        try:
            # Fetch margin information from the API
            margin_info = self.api.get_client_account_margin()
    
            # Log the raw API response for debugging
            print(f"[DEBUG] Raw Margin Info: {margin_info}")
    
            # Validate the response
            if not margin_info or not isinstance(margin_info, dict):
                print("[ERROR] Invalid margin info received from API.")
                return None
    
            # Safely extract values with default fallbacks
            available_balance = float(margin_info.get('Cash', 0))  # Default to 0 if missing
            used_margin = float(margin_info.get('Margin', 0))      # Default to 0 if missing
            tradeable_funds = float(margin_info.get('TradableFunds', 0))  # Default to 0 if missing
    
            # Log extracted values for debugging
            print(f"[DEBUG] Extracted Values - Cash: {available_balance}, Margin: {used_margin}, TradeableFunds: {tradeable_funds}")
    
            # Validate tradeable funds
            if tradeable_funds <= 0:
                print("[WARNING] Tradeable funds are insufficient or unavailable.")
                return None
    
            # Calculate risk amount
            risk_amount = available_balance * (self.risk_percent / 100)
    
            # Log the calculated values
            print("\n=== RISK CALCULATION ===")
            print(f"Available Balance: ${available_balance:.2f}")
            print(f"Used Margin: ${used_margin:.2f}")
            print(f"Tradeable Funds: ${tradeable_funds:.2f}")
            print(f"Risk Amount: ${risk_amount:.2f}")
    
            return {
                'Cash': available_balance,
                'Margin': used_margin,
                'TradeableFunds': tradeable_funds,
                'RiskAmount': risk_amount
            }
    
        except KeyError as ke:
            # Handle missing keys explicitly
            print(f"[ERROR] Missing key in margin info: {ke}")
            print(f"[DEBUG] Raw Margin Info: {margin_info}")
            return None
    
        except ValueError as ve:
            # Handle invalid numeric values
            print(f"[ERROR] Invalid numeric value in margin info: {ve}")
            print(f"[DEBUG] Raw Margin Info: {margin_info}")
            return None
    
        except Exception as e:
            # Catch-all for other exceptions
            print(f"[ERROR] Failed to calculate risk amount: {str(e)}")
            print(f"[DEBUG] Raw Margin Info: {margin_info}")
            return None
                    
    def get_market_state(self, symbol):
        """Get current market state based on indicators"""
        market_data = {
            'H4': self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="1000")['PriceBars'],
            'M1': self.api.get_pricebar_history(symbol, interval="MINUTE", span="1", pricebars="200")['PriceBars']
        }
        
        adx = self.calculate_adx(market_data['H4'])
        chop = self.calculate_choppiness(market_data['H4'])
        volatility = self.calculate_volatility(market_data['M1'])
        
        state = {
            'trend_strength': 'Strong' if adx > 25 else 'Weak',
            'market_type': 'Choppy' if chop > 50 else 'Trending',
            'volatility': 'High' if volatility > 1.2 else 'Low' if volatility < 0.8 else 'Medium'
        }
        
        return state
        
    def execute_or_update_trade(self, symbol, direction, current_price, data):
        try:
            logging.info(f"Executing trade: {symbol} | Direction: {direction} | Price: {current_price}")
    
            prices = self.get_validated_prices(symbol)
            if not prices:
                logging.error(f"Failed to get prices for {symbol}")
                return False
    
            atr = self.calculate_atr(data)
            if atr <= 0:
                logging.error(f"Invalid ATR for {symbol}")
                return False
    
            stop_loss, take_profit = self.calculate_sl_tp_levels(direction, current_price, atr)
            min_distance = self.get_minimum_distance(symbol)  # Nuovo metodo
    # Updated validation call with symbol parameter
            if not self.validate_order_prices(symbol, direction, current_price, stop_loss, take_profit):
                logging.info(f"Invalid price levels for {symbol}: Entry={current_price}, SL={stop_loss}, TP={take_profit}")
                return False
        
            # Resto del codice esistente...
            risk_amount = self.calculate_risk_amount()['RiskAmount']
            qty = self.calculate_position_size(risk_amount, atr, data, symbol)
    
            min_lot_size = self.api.market_info[symbol]['WebMinSize']
            max_lot_size = self.api.market_info[symbol]['MaxLongSize'] if direction == "BUY" else self.api.market_info[symbol]['MaxShortSize']
    
            qty = max(min(qty, max_lot_size), min_lot_size)
            qty = round(qty, 2)
    
            if qty < min_lot_size:
                logging.error(f"Quantity {qty} too small for {symbol}. Minimum: {min_lot_size}")
                return False
    
            margin_info = self.api.get_client_account_margin()
            tradeable_funds = float(margin_info.get('TradableFunds', 0))
    
            # **Creazione dati ordine**
            market_data = {
                "symbol": symbol,
                "cmd": direction,
                "qty": qty,
                "data": {
                    "AuditId": str(int(time.time())),
                    "MarketId": self.api.market_info[symbol]["MarketId"],
                    "TradingAccountId": self.api.trading_account_info["TradingAccounts"][0]["TradingAccountId"],
                    "Bid": prices["Bid"],
                    "Offer": prices["Offer"]
                },
                "stoploss": stop_loss,
                "takeprofit": take_profit
            }
    
            logging.info(f"Sending order: {market_data}")
    
            order = self.api.send_market_order(**market_data)
            result = self.handle_order_response(order, direction, symbol)
    
            logging.info(f"Order response: {result}")
            return result
    
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}", exc_info=True)
            return False
    
    def get_minimum_distance(self, symbol):
        """Get minimum price distance based on symbol characteristics"""
        return self.api.market_info[symbol].get('MinDistance', 0.0001)
    
    def validate_order_prices(self, symbol, direction, entry_price, sl_price, tp_price):
        """Validate order prices with proper symbol parameter"""
        min_distance = self.get_minimum_distance(symbol)
        
        if direction == "BUY":
            valid_levels = (sl_price < entry_price < tp_price)
            valid_distances = ((entry_price - sl_price) >= min_distance and 
                            (tp_price - entry_price) >= min_distance)
        else:  # SELL
            valid_levels = (tp_price < entry_price < sl_price)
            valid_distances = ((sl_price - entry_price) >= min_distance and 
                            (entry_price - tp_price) >= min_distance)
        
        return valid_levels and valid_distances

    
    def validate_order_levels(self, direction, entry_price, sl_price, tp_price):
        """
        Valida che i livelli di Stop Loss e Take Profit siano corretti rispetto alla direzione del trade.
        """
        if direction == "BUY":
            return sl_price < entry_price < tp_price
        elif direction == "SELL":
            return sl_price > entry_price > tp_price
        else:
            return False
    
    def optimize_trade_execution(self, symbol, signal, market_state):
        """Optimize trade execution based on market state and risk parameters"""
        
        # **Ottieni i dati di mercato e assicurati che sia una lista**
        market_data = self.api.get_pricebar_history(symbol, interval="MINUTE", span="1", pricebars="200")['PriceBars']
        
        if not isinstance(market_data, list) or len(market_data) == 0:
            print(f"[ERROR] Market data unavailable for {symbol}")
            return None
        
        latest_candle = market_data[-1]  # **Usiamo l'ultima candela**
        
        # **Calcola ATR**
        atr = self.calculate_atr(market_data)
        
        # **Ottieni il rischio disponibile**
        risk_amount = self.calculate_risk_amount()
        
        # **Calcola la dimensione base della posizione**
        base_size = self.calculate_position_size(risk_amount, atr, latest_candle, symbol)
        
        print(f"\n=== POSITION OPTIMIZATION: {symbol} ===")
        print(f"Base Position: {base_size}")
        print(f"ATR: {atr:.5f}")
        
        # **Moltiplicatori per le condizioni di mercato**
        multipliers = {
            'trend': 1.2 if market_state['trend_strength'] == 'Strong' else 0.8,
            'type': 0.8 if market_state['market_type'] == 'Choppy' else 1.0,
            'volatility': {'High': 0.8, 'Medium': 1.0, 'Low': 1.2}[market_state['volatility']]
        }
        
        # **Applica i moltiplicatori**
        final_size = int(base_size * multipliers['trend'] * multipliers['type'] * multipliers['volatility'])
        
        # **Verifica i limiti di posizione**
        min_lot_size = self.api.market_info[symbol]['WebMinSize']
        max_size = self.api.market_info[symbol]['MaxLongSize'] if signal == "BUY" else self.api.market_info[symbol]['MaxShortSize']
        
        final_size = max(min(final_size, max_size), min_lot_size)  # **Forza il valore nei limiti accettabili**
        
        # **DEBUG LOG**
        print(f"Multipliers Applied:")
        print(f" • Trend Strength: x{multipliers['trend']}")
        print(f" • Market Type: x{multipliers['type']}")
        print(f" • Volatility: x{multipliers['volatility']}")
        print(f"Final Position Size (adjusted to limits): {final_size}\n")
        
        return {
            'entry_type': 'MARKET',
            'position_size': final_size,
            'market_phase': self.market_phase,
            'risk_amount': risk_amount
        }
    

    def calculate_volatility(self, price_data, window=20):
        """Calculate market volatility from recent price data"""
        closes = [float(bar['Close']) for bar in price_data[-window:]]
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(window)
        
        print(f"\n=== VOLATILITY ANALYSIS ===")
        print(f"Current Volatility: {volatility:.4f}")
        print(f"Sample Size: {window} bars")
        
        # Normalize volatility to get optimal delay
        normalized_vol = min(max(volatility * 100, 0.5), 2.0)
        
        return normalized_vol
        
    def monitor_verification_intervals(self, symbol, price_data):
        """Monitor and adjust verification intervals based on real-time volatility"""
        volatility = self.calculate_volatility(price_data)
        optimal_delay = max(3, min(7, volatility * 2))
        
        print(f"\n=== VERIFICATION MONITORING: {symbol} ===")
        print(f"Base Delay: 3 seconds")
        print(f"Volatility Adjustment: +{(optimal_delay - 3):.1f} seconds")
        print(f"Final Verification Interval: {optimal_delay:.1f} seconds")
        
        # Market state classification
        market_state = {
            'LOW': volatility < 0.8,
            'MEDIUM': 0.8 <= volatility <= 1.2,
            'HIGH': volatility > 1.2
        }
        
        current_state = next(state for state, condition in market_state.items() if condition)
        print(f"Market State: {current_state}")
        
        return optimal_delay

    def conditions_match(self, initial, confirmation, threshold=0.05):
        """Verify if conditions remain stable with enhanced structure validation"""
        try:
            # Converti i segnali in dizionari se necessario
            initial = initial if isinstance(initial, dict) else {'signal': initial}
            confirmation = confirmation if isinstance(confirmation, dict) else {'signal': confirmation}
    
            # Debug: stampa la struttura completa
            print("\n=== CONDITIONS MATCH DEBUG ===")
            print("Initial Structure:")
            pprint.pprint(initial, depth=2)
            print("\nConfirmation Structure:")
            pprint.pprint(confirmation, depth=2)
    
            # 1. Verifica struttura di base
            base_checks = (
                isinstance(initial, dict) and 
                isinstance(confirmation, dict) and
                'signal' in initial and
                'signal' in confirmation
            )
            
            if not base_checks:
                print("[ERROR] Struttura del segnale non valida")
                print(f"Initial keys: {list(initial.keys())}")
                print(f"Confirmation keys: {list(confirmation.keys())}")
                return False
    
            # 2. Controllo manipolazione con gestione errori avanzata
            manipulation_stable = False
            try:
                if 'manipulation' in initial and 'manipulation' in confirmation:
                    manip_initial = initial['manipulation']
                    manip_confirmation = confirmation['manipulation']
                    
                    if isinstance(manip_initial, pd.Series) and isinstance(manip_confirmation, pd.Series):
                        manipulation_stable = manip_initial.equals(manip_confirmation)
                    else:
                        manipulation_stable = (str(manip_initial) == str(manip_confirmation))
            except Exception as e:
                print(f"[WARNING] Manipulation check error: {str(e)}")
                manipulation_stable = False
    
            # 3. Calcolo differenze indicatori con normalizzazione
            def safe_compare(a, b, name):
                try:
                    a_val = float(a) if a is not None else 1.0
                    b_val = float(b) if b is not None else 1.0
                    diff = abs(a_val - b_val) / max(abs(a_val), 1e-5)  # Evita divisione per zero
                    return diff < threshold
                except:
                    print(f"[WARNING] Errore nel confronto {name}")
                    return False
    
            indicator_checks = (
                safe_compare(initial.get('adx'), confirmation.get('adx'), 'ADX') and
                safe_compare(initial.get('rsi'), confirmation.get('rsi'), 'RSI') and
                safe_compare(initial.get('stoch_k'), confirmation.get('stoch_k'), 'Stochastic')
            )
    
            # 4. Controllo finale
            return (
                (initial['signal'] == confirmation['signal']) and
                indicator_checks and
                manipulation_stable
            )
    
        except Exception as e:
            print(f"[CRITICAL] Errore in conditions_match: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_open_positions(self):
        # Update open positions from API
        positions = self.api.get_open_positions()
        if positions:
            for position in positions:
                symbol = position['Symbol']
                self.open_positions[symbol] = position
        return self.open_positions

    def calculate_rsi(self, data, period=14):
        # Extract closing prices
        prices = np.array([float(bar['Close']) for bar in data])
        deltas = np.diff(prices)
        
        # Calculate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # First average gain and loss
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Calculate subsequent values
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        # Calculate RS and RSI
        rs = avg_gains[period:] / np.where(avg_losses[period:] != 0, avg_losses[period:], 1)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate((np.array([np.nan] * period), rsi))

    def calculate_signal_strength(self, data):
        """Calculate signal strength based on multiple indicators"""
        # Get technical indicators
        rsi = self.calculate_rsi(data)[-1]
        adx = float(self.calculate_adx(data))
        bb_width = float(self.calculate_bb_width(data))
        
        # Normalize indicators
        rsi_strength = abs(50 - rsi) / 50
        adx_strength = adx / 100
        volatility_strength = bb_width
        
        # Weighted average
        signal_strength = (
            rsi_strength * 0.3 + 
            adx_strength * 0.4 + 
            volatility_strength * 0.3
        )
        
        return min(max(signal_strength, 0), 1)  # Normalize between 0-1
        
    def get_current_price(self, symbol):
        price_data = self.api.get_pricebar_history(
            symbol=symbol,
            interval="MINUTE",
            span="1",
            pricebars="1"
        )
        
        if price_data and "PartialPriceBar" in price_data:
            return float(price_data["PartialPriceBar"]["Close"])
        elif price_data and "PriceBars" in price_data and price_data["PriceBars"]:
            return float(price_data["PriceBars"][-1]["Close"])
            
        return None

    def _calculate_signal_confidence(self, recent_data):
        prices = np.array([float(bar['Close']) for bar in recent_data])
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        trend_direction = np.sign(np.diff(prices))
        trend_consistency = np.abs(np.mean(trend_direction))
        confidence = (0.6 * (1 - volatility) + 0.4 * trend_consistency)
        return np.clip(confidence, 0, 1)

    

    def print_summary(self):
        """Print a summary of the most recent log actions."""
        print("\n=== Summary of Recent Actions ===")
        for log in self.event_log[-5:]:  # Show the last 10 actions
            print(f"{log['timestamp']} - {log['action']} - {log['symbol']} - {log['details']}")

    def calculate_kama(self, data, period=8, fast=2, slow=30):
        close = np.array([bar['Close'] for bar in data])
        change = np.abs(close[1:] - close[:-1])
        volatility = np.array([np.sum(change[max(0, i-period+1):i+1]) for i in range(len(change))])
        er = np.abs(close[period:] - close[:-period]) / volatility[period-1:]
    
        fast_sc = 2/(fast+1)
        slow_sc = 2/(slow+1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
        kama = np.zeros_like(close)
        kama[period-1] = close[period-1]
    
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i-period] * (close[i] - kama[i-1])
    
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
        
    def is_optimal_trading_hour(self):
        current_hour = datetime.now(timezone.utc).hour
        
        # Extended trading hours
        return (
            (8 <= current_hour < 16) or  # Main market hours
            (0 <= current_hour < 8)      # Asia session
        )
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        if period == 0:
            return 0
        
        prices = np.array([float(bar['Close']) for bar in data])
        
        if len(prices) < period:
            return 0
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def calculate_stochastic(self, data, period=14, smooth_k=3, smooth_d=3):
        """
        Calcola l'indicatore Stochastic Oscillator (%K e %D).
        
        Parametri:
        - data: Lista di dati OHLC (deve contenere i campi "High", "Low", "Close").
        - period: Numero di periodi per il calcolo dello Stochastic (default 14).
        - smooth_k: Periodi per la media mobile di %K (default 3).
        - smooth_d: Periodi per la media mobile di %D (default 3).
    
        Ritorna:
        - Lista di valori %K e %D per gli ultimi dati disponibili.
        """
    
        if len(data) < period:
            return [0]  # Se i dati sono insufficienti, restituiamo 0
    
        # Estrarre i valori High, Low e Close
        highs = np.array([float(bar['High']) for bar in data])
        lows = np.array([float(bar['Low']) for bar in data])
        closes = np.array([float(bar['Close']) for bar in data])
    
        # Calcolare il Lowest Low e l'Highest High per ogni finestra mobile di "period"
        lowest_lows = pd.Series(lows).rolling(window=period).min()
        highest_highs = pd.Series(highs).rolling(window=period).max()
    
        # Calcolare %K
        stoch_k = 100 * ((closes - lowest_lows) / (highest_highs - lowest_lows))
    
        # Applicare una media mobile a %K per ottenere una linea più liscia
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
    
        # Calcolare %D (SMA di %K)
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
    
        return stoch_k.fillna(0).tolist()  # Restituiamo la lista di valori %K    
          
    def detect_manipulation_candles(self, price_bars):
        """
        Detects manipulation candle patterns in price data with volume analysis.
        """
        df = pd.DataFrame(price_bars) if not isinstance(price_bars, pd.DataFrame) else price_bars.copy()
        
        # Base calculations
        df['body'] = df['Close'] - df['Open']
        df['body_abs'] = abs(df['body'])
        df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        
        # Volume analysis
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['high_volume'] = df['Volume'] > df['volume_sma'] * 1.5
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Pattern 1: Big Wick
        min_wick_ratio = 2
        df['Big_Wick'] = (
            ((df['upper_wick'] > min_wick_ratio * df['body_abs']) | 
            (df['lower_wick'] > min_wick_ratio * df['body_abs'])) & 
            df['high_volume']
        )
        
        # Pattern 2: Inside Bar
        df['Inside_Bar'] = False
        for i in range(1, len(df)):
            if (df.iloc[i]['High'] < df.iloc[i-1]['High'] and 
                df.iloc[i]['Low'] > df.iloc[i-1]['Low'] and 
                df.iloc[i]['high_volume']):
                df.loc[df.index[i], 'Inside_Bar'] = True
        
        # Pattern 3: Spinning Top
        df['Spinning_Top'] = (
            (df['body_abs'] < df['total_range'] * 0.1) & 
            (df['upper_wick'] > df['total_range'] * 0.4) & 
            (df['lower_wick'] > df['total_range'] * 0.4)
        )
        
        # Pattern 4: Large Candle
        df['Large_Candle'] = (
            (df['body_abs'] > df['total_range'] * 0.8) & 
            (df['total_range'] > df['total_range'].rolling(window=20).mean() * 1.5)
        )
        
        # Pattern 5: Kicker Patterns
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['Kicker_Bullish'] = (
            (df['gap'] < -df['total_range'].mean() * 0.3) & 
            (df['Close'] > df['Open'].shift(1)) & 
            (df['body_abs'] / df['total_range'] > 0.7)
        )
        df['Kicker_Bearish'] = (
            (df['gap'] > df['total_range'].mean() * 0.3) & 
            (df['Close'] < df['Open'].shift(1)) & 
            (df['body_abs'] / df['total_range'] > 0.7)
        )
        
        # Pattern 6: False Breakout
        df['False_Breakout'] = False
        for i in range(1, len(df)):
            if df.iloc[i-1]['Inside_Bar']:
                if (df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['Close'] < df.iloc[i-1]['High']) or \
                (df.iloc[i]['Low'] < df.iloc[i-1]['Low'] and df.iloc[i]['Close'] > df.iloc[i-1]['Low']):
                    df.loc[df.index[i], 'False_Breakout'] = True
        
        # Pattern mapping
        conditions = [
            df['Kicker_Bullish'],
            df['Kicker_Bearish'],
            df['False_Breakout'],
            df['Big_Wick'],
            df['Spinning_Top'],
            df['Large_Candle']
        ]
        choices = [
            'Bullish Kicker',
            'Bearish Kicker',
            'False Breakout',
            'Big Wick',
            'Spinning Top',
            'Liquidity Grab'
        ]
        df['Pattern'] = np.select(conditions, choices, default='Unknown')
        
        # Direction and strength assignment
        df['Direction'] = np.where(df['Kicker_Bullish'], 'BUY',
                                np.where(df['Kicker_Bearish'], 'SELL',
                                        np.where(df['False_Breakout'], 'Indecision',
                                                np.where(df['Big_Wick'] & (df['upper_wick'] > min_wick_ratio * df['body_abs']), 'SELL',
                                                        np.where(df['Big_Wick'] & (df['lower_wick'] > min_wick_ratio * df['body_abs']), 'BUY',
                                                                np.where(df['Spinning_Top'], 'Indecision',
                                                                        np.where(df['Large_Candle'] & (df['body'] > 0), 'BUY',
                                                                                np.where(df['Large_Candle'] & (df['body'] < 0), 'SELL', 'Unknown'))))))))
        
        df['Strength'] = np.where(df['volume_ratio'] > 1.5, 'Strong', 'Weak')
        
        # Add volume calculations
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume'] = df['Volume']
        
        return df[df['Pattern'] != 'Unknown'].fillna(0)

        
    def _log_market_analysis(self, symbol, data):
        """Enhanced market analysis logging"""
        print(f"\n=== MARKET ANALYSIS: {symbol} ===")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Trend Analysis
        print("\nTREND STATUS:")
        print(f"ADX (H4): {data['adx']:.2f} - {'Strong' if data['adx'] > 25 else 'Weak'} Trend")
        print(f"CHOP: {data['chop']:.2f} - {'Choppy' if data['chop'] > 50 else 'Trending'} Market")
        
        # Technical Indicators
        print("\nTECHNICAL INDICATORS:")
        print(f"RSI: {data['rsi']:.2f} - {'Overbought' if data['rsi'] > 70 else 'Oversold' if data['rsi'] < 30 else 'Neutral'}")
        print(f"Stochastic: {data['stoch_k']:.2f} - {'Overbought' if data['stoch_k'] > 80 else 'Oversold' if data['stoch_k'] < 20 else 'Neutral'}")
        
        # EMA Analysis
        print("\nEMA STRUCTURE:")
        print(f"EMA50: {data['ema50']:.5f}")
        print(f"EMA100: {data['ema100']:.5f}")
        print(f"EMA150: {data['ema150']:.5f}")
        
        # Calculate EMA trend
        ema_trend = "Uptrend" if data['ema50'] > data['ema100'] > data['ema150'] else "Downtrend" if data['ema50'] < data['ema100'] < data['ema150'] else "No Clear Trend"
        print(f"Trend Direction: {ema_trend}")
        
        # ML Analysis
        print("\nML ANALYSIS:")
        pred_class = np.argmax(data['pred_class'])
        confidence = np.max(data['pred_class']) * 100
        print(f"Prediction Class: {data['pred_class']} - {self._get_prediction_label(pred_class)}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Manipulation Detection
        print("\nMANIPULATION DETECTION:")
        if isinstance(data['manipulation'], pd.Series) and not data['manipulation'].empty:
            manip_data = data['manipulation']
            print(f"Pattern: {manip_data.get('Pattern', 'Unknown')}")
            print(f"Direction: {manip_data.get('Direction', 'Unknown')}")
            print(f"Strength: {manip_data.get('Strength', 'Unknown')}")
        else:
            print("No manipulation detected")
                
        
    
    def _get_prediction_label(self, pred_class):
        """Converts prediction class to readable label"""
        labels = {0: 'SELL Signal', 1: 'Neutral', 2: 'BUY Signal'}
        # Convert numpy array to integer
        pred_class = np.argmax(pred_class) if isinstance(pred_class, np.ndarray) else pred_class
        return labels.get(pred_class, 'Unknown')

    
    def _check_trading_conditions(self, data):
        """Checks all trading conditions"""
        return {
            'Strong Trend': data['adx'] > 25,
            'Clear Market Direction': data['chop'] < 50,
            'RSI Alignment': 30 < data['rsi'] < 70,
            'High ML Confidence': data['confidence'] > 0.95,
            'EMA Alignment': abs(data['ema50'] - data['ema100']) > 0.0001,
            'Volume Confirmation': True  # Add your volume logic here
        } 
        
    def check_conditions(self, data, symbol):
        """Main trading logic with optimized thresholds"""
        market_data = { 
            'H4': self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="1000")['PriceBars'],
            'M1': data
        }
        
        analysis_data = self._prepare_analysis_data(market_data)
        
        # Debug: Print analysis data
        print("\n=== DEBUG: ANALYSIS DATA ===")
        pprint.pprint(analysis_data)
        
        self._log_market_analysis(symbol, analysis_data)
        signal = self._generate_trading_signal(analysis_data)
        
        if signal:
            self._log_signal_details(symbol, analysis_data, signal)
        
        return signal
    
    def _prepare_analysis_data(self, market_data):
        """Prepara i dati di analisi con una struttura corretta."""
        last_bar = market_data['M1'][-1]  # Prende l'ultima candela dei dati M1
        """Prepara i dati di analisi con una struttura corretta."""
        try:
            return {
                'adx': self.calculate_adx(market_data['H4'], period=14),
                'chop': self.calculate_choppiness(market_data['H4'], period=14),
                'rsi': self.calculate_rsi(market_data['M1'])[-1],
                'stoch_k': self.calculate_stochastic(market_data['M1'])[-1],
                'ema50': self.calculate_ema(market_data['M1'], 50)[-1],
                'ema100': self.calculate_ema(market_data['M1'], 100)[-1],
                'ema150': self.calculate_ema(market_data['M1'], 150)[-1],
                'manipulation': self.detect_manipulation_candles(market_data['M1']).iloc[-1] if not self.detect_manipulation_candles(market_data['M1']).empty else None,
                'pred_class': self.model.predict(self.prepare_ml_features(market_data['M1']), verbose=0)[0],
                'confidence': np.max(self.model.predict(self.prepare_ml_features(market_data['M1']), verbose=0)[0]),
                'close': float(last_bar['Close']),  # Aggiunge il prezzo di chiusura
                'volume': market_data['M1'][-1].get('Volume', 0),  # Estrae il volume dell'ultima barra
                'volume_sma': np.mean([bar.get('Volume', 0) for bar in market_data['M1'][-20:]])  # Calcola la SMA del volume
            }
        except Exception as e:
            print(f"[ERROR] Failed to prepare analysis data: {str(e)}")
            return None
                
    #def _check_breakout_conditions(self, data):
    #    """Analyzes potential breakout setups"""
    #    
    #    # EMA compression check
    #    ema_spread = abs(data['ema50'] - data['ema100'])
    #    is_compressed = ema_spread < 0.0005
    #    
    #    # Volatility and momentum checks
    #    momentum_building = data['adx'] > 20 and data['adx'] < 25
    #    clear_market = data['chop'] < 30
    #    
    #    # Price action confirmation
    #    if is_compressed and momentum_building and clear_market:
    #        if data['stoch_k'] < 20:  # Potential upward breakout
    #            return 'UP'
    #        elif data['stoch_k'] > 80:  # Potential downward breakout
    #            return 'DOWN'
    #    
    #    return None
    #
    #def _generate_trading_signal(self, data):
    #    """
    #    Genera un segnale di trading con validazione integrata e gestione avanzata del trend.
    #    Versione 4.0 - Soluzione completa a segnali instabili e warning
    #    """
    #    signal = None
    #    THRESHOLDS = {
    #        'adx_strong': 30,  # Aumentato da 25 a 30 per trend più marcati
    #        'adx_developing': 20,
    #        'confidence_high': 0.85,  # Aumentato da 0.80
    #        'stoch_oversold': 20,     # Modificato da 25
    #        'stoch_overbought': 80,   # Modificato da 75
    #        'rsi_low': 35,           # Modificato da 40
    #        'rsi_high': 65           # Modificato da 60
    #    }
    #
    #    def normalize_manipulation(raw_manip):
    #        """Restituisce un oggetto standardizzato con tipi garantiti"""
    #        if isinstance(raw_manip, dict) or hasattr(raw_manip, 'to_dict'):
    #            manip_dict = raw_manip.to_dict() if hasattr(raw_manip, 'to_dict') else raw_manip
    #            return {
    #                'is_valid': True,
    #                'strength': manip_dict.get('Strength', 'No'),
    #                'direction': manip_dict.get('Direction', 'None'),
    #                'pattern': manip_dict.get('Pattern', 'Unknown')
    #            }
    #        return {'is_valid': False, 'strength': 'No', 'direction': 'None', 'pattern': 'Unknown'}
    #
    #    # Processa e normalizza tutti i dati
    #    manipulation = normalize_manipulation(data.get('manipulation'))
    #    adx = round(data['adx'], 2)
    #    confidence = round(data['confidence'], 4)
    #    ema_cross = data['ema50'] > data['ema100']
    #    trend_strength = 'Strong' if adx >= THRESHOLDS['adx_strong'] else 'Developing' if adx >= THRESHOLDS['adx_developing'] else 'Weak'    
    #    
    #    # 1. Logica di validazione integrata
    #    def is_valid_signal():
    #        """
    #        Verifica la validità del segnale combinando:
    #        - Condizioni di base
    #        - Manipolazione del mercato
    #        - Allineamento degli indicatori
    #        - Convalida del volume
    #        """
    #        # Condizioni fondamentali minime
    #        condizioni_base = all([
    #            confidence >= THRESHOLDS['confidence_high'],  # Confidenza minima 85%
    #            trend_strength in ['Strong', 'Developing'],   # Trend presente
    #            data['volume'] > data['volume_sma'] * 0.7     # Volume sopra il 70% della media
    #        ])
    #        
    #        # Se c'è una manipolazione forte confermata
    #        if manipulation['is_valid'] and manipulation['strength'] == 'Strong':
    #            return condizioni_base and all([
    #                manipulation['direction'] == ('BUY' if ema_cross else 'SELL'),  # Allineamento direzione
    #                abs(data['ema50'] - data['ema100']) > (data['close'] * 0.003), # Differenza EMA > 0.3%
    #                data['adx'] >= THRESHOLDS['adx_strong']                        # Trend forte
    #            ])
    #        
    #        # Validazione per segnali tecnici puri (senza manipolazione)
    #        return condizioni_base and all([
    #            # Condizioni per BUY
    #            (ema_cross and all([
    #                data['stoch_k'] <= THRESHOLDS['stoch_oversold'],      # Ipervenduto
    #                data['rsi'] <= THRESHOLDS['rsi_low'],                  # RSI basso
    #                data['close'] > data['upper_bb'] * 0.99                # Prezzo near banda superiore
    #            ])) or
    #            # Condizioni per SELL
    #            (not ema_cross and all([
    #                data['stoch_k'] >= THRESHOLDS['stoch_overbought'],     # Ipercomprato
    #                data['rsi'] >= THRESHOLDS['rsi_high'],                 # RSI alto
    #                data['close'] < data['lower_bb'] * 1.01                # Prezzo near banda inferiore
    #            ])),
    #            # Conferma volatilità
    #            (data['high'] - data['low']) > data['atr'] * 0.8           # Range giornaliero adeguato
    #        ])
    #        # 2. Generazione del segnale con validazione
    #        if is_valid_signal():
    #            signal = 'BUY' if ema_cross else 'SELL'
    #            if manipulation['is_valid'] and manipulation['strength'] == 'Strong':
    #                signal = manipulation['direction']  # Forza la direzione della manipolazione forte
    #        
    #        # 3. Preparazione output con logging integrato
    #        return {
    #            'signal': signal,
    #            'validation': {
    #                'is_valid': signal is not None,
    #                'confidence_level': 'High' if confidence >= 0.9 else 'Medium' if confidence >= 0.8 else 'Low',
    #                'trend_strength': trend_strength,
    #                'ema_alignment': 'Bullish' if ema_cross else 'Bearish',
    #                'manipulation_used': manipulation['is_valid']
    #            },
    #            'indicators': {
    #                'adx': adx,
    #                'rsi': round(data['rsi'], 2),
    #                'stoch_k': round(data['stoch_k'], 2),
    #                'ema_50_100': f"{data['ema50']:.5f} > {data['ema100']:.5f}" if ema_cross else f"{data['ema50']:.5f} < {data['ema100']:.5f}"
    #            }
    #        }
    #
    #test signal sell
    #def _generate_trading_signal(self, analysis_data):
    #    # Struttura dati completa per il test
    #    test_signal = {
    #        'signal': 'SELL',
    #        'validation': {
    #            'is_valid': True,
    #            'confidence_level': 'High',
    #            'trend_strength': 'Strong',
    #            'ema_alignment': 'Bullish',
    #            'manipulation_used': False
    #        },
    #        'indicators': {
    #            'adx': 35.0,
    #            'rsi': 45.0,
    #            'stoch_k': 25.0,
    #            'ema_50_100': "0.89932 > 0.89926",
    #            'confidence': 0.95
    #        },
    #        'manipulation': {
    #            'Pattern': 'Liquidity Grab',
    #            'Direction': 'SELL',
    #            'Strength': 'Strong'
    #        }
    #    }
    #    
    #    print("\n=== TEST SIGNAL GENERATED ===")
    #    print(json.dumps(test_signal, indent=2))
    #    
    #    return test_signal  # Restituisce un dizionario completo
        
    def _generate_trading_signal(self, data):
        """
        Generates a trading signal with integrated validation and advanced trend management.
        Version 5.0 - Introduces a weighted system to resolve indicator conflicts.
        
        Args:
            data (dict): Market data containing technical indicators
            
        Returns:
            dict: Trading signal with validation metrics and indicator values
        """
        # Constants moved to class level for better maintainability
        WEIGHTS = {
            'ema_alignment': 0.4,
            'adx': 0.3, 
            'stochastic': 0.15,
            'rsi': 0.15
        }
    
        THRESHOLDS = {
            'adx_strong': 20,
            'adx_developing': 15,
            'confidence_high': 0.6,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'rsi_low': 35,
            'rsi_high': 65
        }
    
        def normalize_manipulation(raw_manip):
            """
            Standardizes manipulation data with guaranteed types.
            
            Args:
                raw_manip (dict/object): Raw manipulation data
                
            Returns:
                dict: Normalized manipulation data
            """
            if isinstance(raw_manip, dict) or hasattr(raw_manip, 'to_dict'):
                manip_dict = raw_manip.to_dict() if hasattr(raw_manip, 'to_dict') else raw_manip
                return {
                    'is_valid': True,
                    'strength': manip_dict.get('Strength', 'No'),
                    'direction': manip_dict.get('Direction', 'None'),
                    'pattern': manip_dict.get('Pattern', 'Unknown')
                }
            return {'is_valid': False, 'strength': 'No', 'direction': 'None', 'pattern': 'Unknown'}
    
        # Score initialization
        buy_score = sell_score = 0
    
        # Data preprocessing
        manipulation = normalize_manipulation(data.get('manipulation'))
        adx = round(data['adx'], 2)
        confidence = round(data['confidence'], 4)
        ema_cross = data['ema50'] > data['ema100']
        trend_strength = ('Strong' if adx >= THRESHOLDS['adx_strong'] 
                        else 'Developing' if adx >= THRESHOLDS['adx_developing'] 
                        else 'Weak')
    
        # Score calculation based on EMA alignment
        if ema_cross:
            buy_score += WEIGHTS['ema_alignment']
        else:
            sell_score += WEIGHTS['ema_alignment']
    
        # ADX-based score adjustment
        if adx >= THRESHOLDS['adx_strong']:
            if ema_cross:
                buy_score += WEIGHTS['adx']
            else:
                sell_score += WEIGHTS['adx']
        elif adx >= THRESHOLDS['adx_developing']:
            weight_modifier = 0.5
            if ema_cross:
                buy_score += WEIGHTS['adx'] * weight_modifier
            else:
                sell_score += WEIGHTS['adx'] * weight_modifier
    
        # Stochastic oscillator analysis
        stoch_k = data.get('stoch_k', 0)
        if stoch_k < THRESHOLDS['stoch_oversold']:
            buy_score += WEIGHTS['stochastic']
        elif stoch_k >= THRESHOLDS['stoch_overbought']:
            sell_score += WEIGHTS['stochastic']
    
        # RSI analysis
        rsi = data.get('rsi', 0)
        if rsi < THRESHOLDS['rsi_low']:
            buy_score += WEIGHTS['rsi']
        elif rsi >= THRESHOLDS['rsi_high']:
            sell_score += WEIGHTS['rsi']
    
        # Signal determination
        signal = 'BUY' if buy_score > sell_score else 'SELL' if sell_score > buy_score else None
    
        return {
            'signal': signal,
            'validation': {
                'is_valid': signal is not None,
                'confidence_level': 'High' if confidence >= 0.9 else 'Medium' if confidence >= 0.8 else 'Low',
                'trend_strength': trend_strength,
                'ema_alignment': 'Bullish' if ema_cross else 'Bearish',
                'manipulation_used': manipulation['is_valid']
            },
            'indicators': {
                'adx': adx,
                'rsi': round(rsi, 2),
                'stoch_k': round(stoch_k, 2),
                'ema_50_100': f"{data.get('ema50', 0):.5f} > {data.get('ema100', 0):.5f}" if ema_cross 
                            else f"{data.get('ema50', 0):.5f} < {data.get('ema100', 0):.5f}"
            }
        }
    
                    
    def _log_breakout_signal(self, direction, data):
        """Detailed breakout signal logging"""
        print(f"\n=== BREAKOUT {direction} SIGNAL DETECTED ===")
        print(f"EMA Compression: {abs(data['ema50'] - data['ema100']):.6f}")
        print(f"ADX Development: {data['adx']:.2f}")
        print(f"Market Clarity (CHOP): {data['chop']:.2f}")
        print(f"Cycle Position (Stochastic): {data['stoch_k']:.2f}")
        print(f"ML Confidence: {data['confidence']:.2%}")
        
    def _collect_market_data(self, symbol):
        """Raccolta dati multi-timeframe"""
        return {
            'H4': self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="1000")['PriceBars'],
            'M15': self.api.get_pricebar_history(symbol, interval="MINUTE", span="15", pricebars="1000")['PriceBars'],
            'M1': self.api.get_pricebar_history(symbol, interval="MINUTE", span="1", pricebars="200")['PriceBars']
        }
    
    
    
    def _log_signal_details(self, symbol, data, signal):
        """Enhanced signal logging"""
        print(f"\n=== TRADE SIGNAL GENERATED: {symbol} {signal} ===")
        print(f"ADX Strength: {data['adx']:.2f}")
        print(f"Market Structure: {'Trending' if data['chop'] < 30 else 'Choppy'}")
        print(f"RSI: {data['rsi']:.2f}")
        print(f"Stochastic: {data['stoch_k']:.2f}")
        print(f"ML Confidence: {data['confidence']:.2%}")
        print(f"EMA Direction: {'Bullish' if data['ema50'] > data['ema100'] else 'Bearish'}")
    
    def _validate_manipulation_signal(self, manipulation, direction, required_strength):
        """Validates manipulation signals against required criteria"""
        if manipulation is None:
            return True
        return (manipulation['Signal'] == 'Reversal' and 
                manipulation['Direction'] == direction and 
                manipulation['Strength'] == required_strength)
    
    def _log_debug_info(self, symbol, pred_class, confidence, tech_indicators, emas, manipulation):
        """Centralized debug logging"""
        print(f"[DEBUG] {symbol} ML Prediction - Class: {pred_class}, Confidence: {confidence:.2f}")
        print(f"[DEBUG] {symbol} Technical Values - ADX: {tech_indicators['adx']:.2f}, RSI: {tech_indicators['rsi']:.2f}, Stochastic: {tech_indicators['stoch_k']:.2f}")
        print(f"[DEBUG] {symbol} EMAs - EMA50: {emas[50]:.5f}, EMA100: {emas[100]:.5f}, EMA150: {emas[150]:.5f}, EMA200: {emas[200]:.5f}")
        
        if manipulation is not None:
            print(f"[DEBUG] {symbol} Manipulation Signal - {manipulation['Signal']} ({manipulation['Pattern']}) - Direction: {manipulation['Direction']} - Strength: {manipulation['Strength']}")
    
    def _handle_signal_confirmation(self, signal, symbol, adx):
        """Handles signal confirmation and logging"""
        print(f"[INFO] {signal} signal detected for {symbol}")
        if adx > 30:
            print(f"[INFO] Strong trend detected - consider larger position size")
                
    
    def update_model_performance(self, symbol, prediction_class, confidence):
        """Updates model performance metrics and tracking"""
        timestamp = datetime.now()
        
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'predictions': deque(maxlen=50),
                'confidence_levels': deque(maxlen=50),
                'timestamps': deque(maxlen=50),
                'symbols': deque(maxlen=50)
            }
        
        # Add new performance data
        self.performance_metrics['predictions'].append(prediction_class)
        self.performance_metrics['confidence_levels'].append(confidence)
        self.performance_metrics['timestamps'].append(timestamp)
        self.performance_metrics['symbols'].append(symbol)
        
        # Calculate rolling accuracy
        if len(self.performance_metrics['confidence_levels']) >= 10:
            recent_accuracy = np.mean(np.array(self.performance_metrics['confidence_levels']))
            print(f"Rolling Model Accuracy: {recent_accuracy:.2%}")
        
        print(f"[DEBUG] Model performance updated for {symbol}")
    
    def analyze_neutral_signals(self, adx, rsi, chop, prediction_class, confidence):
        """Enhanced analysis for high-confidence neutral signals"""
        if prediction_class == 1 and confidence > 0.97:
            if adx > 25 and chop < 40:
                if rsi < 35:
                    return 'BUY'
                elif rsi > 65:
                    return 'SELL'
        return None
    
        
    
    def extract_price_features(self, data):
        prices = np.array([candle['Close'] for candle in data])
        return {
            'price_momentum': (prices[-1] - prices[-5]) / prices[-5],
            'price_volatility': np.std(prices[-20:]) / np.mean(prices[-20:]),
            'price_trend': self.calculate_trend_strength(prices[-30:])
        }
    
    def extract_technical_features(self, data):
        return {
            'kama': float(self.calculate_kama(data)[-1]),
            'bb_width': float(self.calculate_bb_width(data)),
            'adx': float(self.calculate_adx(data)),  # Remove [-1] indexing
            'volume_trend': self.analyze_volume_trend(data)
        }
    
    
    def extract_pattern_features(self, data):
        return {
            'candlestick_pattern': self.identify_candlestick_patterns(data[-5:]),
            'support_resistance': self.find_support_resistance(data),
            'market_structure': self.analyze_market_structure(data[-50:])
        }
    
    def extract_market_features(self, symbol):
        return {
            'market_phase': self.market_phase,
            'correlation_score': self.calculate_correlation_score(symbol),
            'volatility_regime': self.identify_volatility_regime(symbol)
        }
        
    def calculate_correlation_score(self, symbol, lookback=20):
        correlations = []
        reference_pairs = ['EURUSD', 'USDJPY', 'GBPUSD']
        
        # Get symbol data safely
        symbol_data = self.historical_data.get(symbol, [])
        if len(symbol_data) < lookback:
            return 0.5
            
        for ref_symbol in reference_pairs:
            if ref_symbol != symbol:
                ref_data = self.historical_data.get(ref_symbol, [])
                
                if len(ref_data) >= lookback:
                    # Extract prices
                    symbol_prices = np.array([candle['Close'] for candle in symbol_data])[-lookback:]
                    ref_prices = np.array([candle['Close'] for candle in ref_data])[-lookback:]
                    
                    # Calculate returns
                    symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
                    ref_returns = np.diff(ref_prices) / ref_prices[:-1]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(symbol_returns, ref_returns)[0,1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
        
        return np.mean(correlations) if correlations else 0.5
    

    def identify_candlestick_patterns(self, data):
        patterns = {
            'doji': 0,
            'hammer': 0,
            'engulfing': 0,
            'pinbar': 0
        }
        
        for candle in data:
            body = abs(candle['Close'] - candle['Open'])
            upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
            lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
            
            # Doji pattern
            if body <= (upper_wick + lower_wick) * 0.1:
                patterns['doji'] += 1
                
            # Hammer pattern
            if lower_wick > body * 2 and upper_wick < body * 0.5:
                patterns['hammer'] += 1
                
            # Pinbar pattern
            if upper_wick > body * 2 or lower_wick > body * 2:
                patterns['pinbar'] += 1
        
        # Calculate pattern strength (0-1)
        pattern_strength = sum(patterns.values()) / (len(data) * len(patterns))
        
        return pattern_strength
        
    def find_support_resistance(self, data, window=20):
        prices = np.array([candle['High'] for candle in data])
        lows = np.array([candle['Low'] for candle in data])
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(prices)-window):
            # Resistance
            if all(prices[i] > prices[i-window:i]) and all(prices[i] > prices[i+1:i+window]):
                resistance_levels.append(prices[i])
            # Support
            if all(lows[i] < lows[i-window:i]) and all(lows[i] < lows[i+1:i+window]):
                support_levels.append(lows[i])
        
        # Calculate strength based on touches and recency
        current_price = data[-1]['Close']
        levels = {
            'nearest_resistance': min(resistance_levels, default=current_price),
            'nearest_support': max(support_levels, default=current_price),
            'strength': len(resistance_levels) + len(support_levels)
        }
        
        return levels
        
    def analyze_market_structure(self, data):
        highs = np.array([candle['High'] for candle in data])
        lows = np.array([candle['Low'] for candle in data])
        closes = np.array([candle['Close'] for candle in data])
        
        # Identify swing points
        higher_highs = np.where(highs[1:-1] > highs[:-2])[0]
        higher_lows = np.where(lows[1:-1] > lows[:-2])[0]
        
        # Calculate market structure score
        trend_score = len(higher_highs) + len(higher_lows)
        momentum_score = (closes[-1] - closes[0]) / closes[0]
        
        structure = {
            'trend_strength': trend_score / len(data),
            'momentum': momentum_score,
            'structure_type': 'bullish' if momentum_score > 0 else 'bearish'
        }
        
        return structure
    
    def calculate_market_profile(self, data):
        """Calcola il profilo di mercato"""
        prices = np.array([float(bar['Close']) for bar in data])
        volumes = np.array([float(bar.get('Volume', 0)) for bar in data])
        
        # Calcola il Value Area
        total_volume = np.sum(volumes)
        poc_index = np.argmax(volumes)
        poc_price = prices[poc_index]
        
        # Calcola la distribuzione dei prezzi
        value_area = np.sum(volumes[volumes > np.percentile(volumes, 70)]) / total_volume
        
        return value_area
    
    def calculate_order_flow(self, data):
        """Analizza il flusso degli ordini"""
        delta = []
        for bar in data:
            close = float(bar['Close'])
            open_price = float(bar['Open'])
            volume = float(bar.get('Volume', 0))
            
            if close > open_price:
                delta.append(volume)
            else:
                delta.append(-volume)
        
        return np.mean(delta[-10:])  # Media degli ultimi 10 periodi
    
    def analyze_price_action(self, data, bars=3):
        """Analizza il price action"""
        recent_data = data[-bars:]
        
        # Calcola la forza del trend
        closes = [float(bar['Close']) for bar in recent_data]
        opens = [float(bar['Open']) for bar in recent_data]
        
        bullish_bars = sum(1 for c, o in zip(closes, opens) if c > o)
        bearish_bars = sum(1 for c, o in zip(closes, opens) if c < o)
        
        strength = max(bullish_bars, bearish_bars) / bars
        trend = 'bullish' if bullish_bars > bearish_bars else 'bearish'
        
        return {
            'trend': trend,
            'strength': strength
        }
    
    def analyze_market_correlations(self, symbol):
        """Analizza le correlazioni tra mercati"""
        correlations = []
        
        for other_symbol in self.symbols:
            if other_symbol != symbol:
                try:
                    symbol_data = np.array([float(bar['Close']) for bar in self.fetch_latest_data(symbol)])
                    other_data = np.array([float(bar['Close']) for bar in self.fetch_latest_data(other_symbol)])
                    
                    if len(symbol_data) == len(other_data):
                        correlation = np.corrcoef(symbol_data, other_data)[0, 1]
                        correlations.append(abs(correlation))
                except:
                    continue
        
        return np.mean(correlations) if correlations else 0.5
        
    def calculate_fractals(self, data, window=5):
        """Calculate Williams Fractals indicator"""
        highs = np.array([float(bar['High']) for bar in data])
        lows = np.array([float(bar['Low']) for bar in data])
        
        bullish_fractals = []
        bearish_fractals = []
        
        # Calculate fractals with a sliding window
        for i in range(window, len(data) - window):
            # Bullish fractal
            if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
            all(lows[i] < lows[i+j] for j in range(1, window+1)):
                bullish_fractals.append(1)
            else:
                bullish_fractals.append(0)
                
            # Bearish fractal
            if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
            all(highs[i] > highs[i+j] for j in range(1, window+1)):
                bearish_fractals.append(1)
            else:
                bearish_fractals.append(0)
        
        # Calculate fractal strength
        fractal_strength = sum(bullish_fractals) - sum(bearish_fractals)
        normalized_strength = fractal_strength / (len(bullish_fractals) + 1e-10)
        
        return normalized_strength
        
    def calculate_volume_profile(self, data):
        """Calculate volume profile analysis"""
        prices = np.array([float(bar['Close']) for bar in data])
        volumes = np.array([float(bar.get('Volume', 0)) for bar in data])
        
        # Create price bins
        bins = 10
        price_range = np.linspace(min(prices), max(prices), bins)
        
        # Calculate volume distribution
        volume_profile = np.zeros(bins-1)
        for i in range(len(prices)):
            bin_index = np.digitize(prices[i], price_range) - 1
            if 0 <= bin_index < bins-1:
                volume_profile[bin_index] += volumes[i]
        
        # Normalize the profile
        total_volume = np.sum(volume_profile)
        if total_volume > 0:
            normalized_profile = np.sum(volume_profile) / total_volume
        else:
            normalized_profile = 0
            
        return normalized_profile

    def calculate_momentum(self, data, period=5):
        closes = [float(bar['Close']) for bar in data]
        return (closes[-1] / closes[-period] - 1) * 100  
        
    def get_market_sentiment(self, symbol):
        """Get real-time sentiment score from news API"""
        # Implement your sentiment analysis here
        return 0.7  # Example value between 0-1
        
    def analyze_volume_trend(self, data, lookback=10):
        """Helper function for volume analysis"""
        recent_volumes = [float(bar['Volume']) for bar in data[-lookback:]]
        avg_volume = sum(recent_volumes[:-1]) / (lookback - 1)
        current_volume = recent_volumes[-1]
        
        # Return True if current volume is above average
        return current_volume > avg_volume * 0.9  # 10% above average
    
                    
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

    def calculate_dynamic_threshold(self, indicator_name, lookback_period=30):
        """Calculate dynamic threshold in decimal format (0-1 range)"""
        recent_data = []
        
        for symbol in self.symbols:
            if symbol in self.historical_data and indicator_name in self.historical_data[symbol]:
                value = self.historical_data[symbol][indicator_name]
                if isinstance(value, (list, np.ndarray)):
                    recent_data.extend(value[-lookback_period:])
                else:
                    recent_data.append(value)
    
        # Default values in decimal format
        default_values = {
            'bb_width': 0.15,  # 15%
            'adx': 0.25,      # 25%
            'chop': 0.45      # 45%
        }
    
        if not recent_data:
            return default_values.get(indicator_name, 0.0)
    
        volatility = np.std(recent_data) / np.mean(recent_data) if recent_data else 0.0
    
        # Dynamic configuration with decimal values
        dynamic_config = {
            'bb_width': {
                'percentile': 0.25 if volatility > 0.15 else 0.35,
                'min': 0.04,  # 4%
                'max': 0.25,  # 25%
                'time_adjust': 1.3 if self.is_asia_session() else 0.8
            },
            'adx': {
                'percentile': 0.15 if self.market_phase == 'trend' else 0.40,
                'min': 0.12,  # 12%
                'max': 0.30,  # 30%
                'time_adjust': 0.85 if 8 <= datetime.now().hour < 12 else 1.15
            },
            'chop': {
                'percentile': 0.55 if self.market_phase == 'ranging' else 0.35,
                'min': 0.30,  # 30%
                'max': 0.60,  # 60%
                'time_adjust': 1.1 if self.market_phase == 'ranging' else 0.9
            }
        }
    
        if indicator_name in dynamic_config:
            config = dynamic_config[indicator_name]
            raw_value = float(np.percentile(recent_data, config['percentile'])) / 100.0
            return float(np.clip(raw_value * config['time_adjust'], config['min'], config['max']))
        
        return default_values.get(indicator_name, 0.0)
    

    def calculate_trend_strength(self, prices, period=14):
        # Prices are already an array, no need for conversion
        prices = np.asarray(prices, dtype=float)
        
        # Directional movement
        up_moves = np.maximum(prices[1:] - prices[:-1], 0)
        down_moves = np.maximum(prices[:-1] - prices[1:], 0)
        
        # Trend momentum
        up_momentum = np.sum(up_moves[-period:])
        down_momentum = np.sum(down_moves[-period:])
        
        # Calculate strength ratio
        total_momentum = up_momentum + down_momentum
        if total_momentum == 0:
            return 0
            
        trend_strength = (up_momentum - down_momentum) / total_momentum
        return np.clip(trend_strength, -1, 1)
    
    def initialize_historical_data(self):
        """Initialize historical data for all symbols"""
        for symbol in self.symbols:
            # Get larger initial dataset
            h4_data = self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="100")
            
            if len(h4_data) >= 20:  # Minimum required for calculations
                adx = self.calculate_adx(h4_data, period=14)
                chop = self.calculate_chop(h4_data, period=14)
                
                self.historical_data[symbol] = {
                    'adx': adx[-10:],  # Store last 10 values
                    'chop': chop[-10:],
                    'price_data': h4_data[-20:]  # Store last 20 bars
                }
                print(f"[INFO] Initialized data for {symbol}: ADX={len(adx)}, CHOP={len(chop)}")
    
    def check_market_phase(self):
        """Enhanced market phase detection with multi-symbol analysis"""
        print("\nChecking market phase...")
        
        adx_values = []
        chop_values = []
    
        for symbol in self.symbols:
            h4_data = self.api.get_pricebar_history(symbol, interval="HOUR", span="4", pricebars="500")
            
            if 'PriceBars' in h4_data:
                price_bars = h4_data['PriceBars']
                print(f"\n=== Data Analysis for {symbol} ===")
                print(f"Total H4 bars collected: {len(price_bars)}")
                
                if len(price_bars) >= 20:
                    adx = self.calculate_adx(price_bars, period=14)
                    chop = self.calculate_choppiness(price_bars, period=14)
                    
                    print(f"Latest ADX value: {adx:.2f}")
                    print(f"Latest CHOP value: {chop:.2f}")
                    
                    adx_values.append(adx)
                    chop_values.append(chop)
                    
                    self.historical_data[symbol] = {
                        'adx': adx,
                        'chop': chop,
                        'volatility': self.identify_volatility_regime(symbol)
                    }
    
        if adx_values and chop_values:
            avg_adx = np.mean(adx_values)
            avg_chop = np.mean(chop_values)
            
            print(f"\nAverage ADX: {avg_adx:.2f}")
            print(f"Average CHOP: {avg_chop:.2f}")
            
            # Dynamic thresholds based on volatility
            trend_threshold = 22 if self.is_high_volatility() else 20
            range_threshold = 18 if self.is_high_volatility() else 16
            
            if avg_adx > trend_threshold and avg_chop < 45:
                print("Market phase: Trending")
                return 'trend'
            elif avg_adx < range_threshold and avg_chop > 50:
                print("Market phase: Ranging")
                return 'ranging'
        
        print("Market phase: Neutral")
        return 'neutral'
    
    def identify_volatility_regime(self, symbol, window=20):
        data = self.historical_data.get(symbol, [])
        if len(data) < window:
            return 'normal'
            
        prices = np.array([candle['Close'] for candle in data])
        returns = np.diff(np.log(prices))
        
        current_vol = np.std(returns[-window:])
        historical_vol = np.std(returns)
        
        high_vol_threshold = np.percentile(returns, 75)
        low_vol_threshold = np.percentile(returns, 25)
        
        if current_vol > high_vol_threshold:
            return 'high'
        elif current_vol < low_vol_threshold:
            return 'low'
        return 'normal'
    
    def is_high_volatility(self):
        """Check if majority of symbols are in high volatility regime"""
        high_vol_count = sum(1 for symbol in self.symbols 
                            if self.historical_data.get(symbol, {}).get('volatility') == 'high')
        return high_vol_count > len(self.symbols) / 2
    
    
    def time_based_adjustment(self, value, indicator):
        """Aggiusta i valori in base all'orario di trading"""
        current_hour = datetime.now().hour
        
        if indicator == 'adx':
            if self.is_london_newyork_overlap():
                return value * 0.9
            return value * 1.1
        
        elif indicator == 'bb_width':
            if self.is_asia_session():
                return value * 1.2
        return value

    def is_london_newyork_overlap(self):
        current_hour = datetime.now(timezone.utc).hour
        return 13 <= current_hour < 15
    
    def is_asia_session(self):
        current_hour = datetime.now(timezone.utc).hour
        return 22 <= current_hour < 24

    def cleanup(self):
        """Perform cleanup operations when monitoring stops"""
        try:
            # 1. Verifica esistenza attributi prima dell'uso
            shutdown_details = {'status': 'shutdown_started'}
            
            # 2. Gestione sicura del balance
            try:
                if hasattr(self.api, 'trading_account_info'):
                    accounts = self.api.trading_account_info.get('TradingAccounts', [])
                    if accounts:
                        shutdown_details['final_balance'] = accounts[0].get('Balance', 'n/a')
            except Exception as e:
                print(f"Error retrieving balance: {str(e)}")
    
            # 3. Chiusura connessione API sicura
            if hasattr(self.api, 'session') and hasattr(self.api.session, 'close'):
                try:
                    self.api.session.close()
                    shutdown_details['api_status'] = 'disconnected'
                except Exception as e:
                    print(f"Error closing API session: {str(e)}")
                    shutdown_details['api_status'] = 'disconnect_failed'
    
            # 4. Logging migliorato
            shutdown_data = {
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'SHUTDOWN',
                'symbol': 'SYSTEM',
                'details': shutdown_details
            }
    
            # 5. Salvataggio dati con fallback
            try:
                if hasattr(self, 'data_storage'):
                    self.data_storage.save_data(shutdown_data)
            except Exception as e:
                print(f"Failed to save shutdown data: {str(e)}")
                with open('emergency_shutdown.log', 'a') as f:
                    json.dump(shutdown_data, f)
    
        except Exception as e:
            print(f"Critical cleanup error: {str(e)}")
        
        finally:
            # 6. Chiusura connessione database con controllo
            if hasattr(self, 'data_storage'):
                try:
                    self.data_storage.close_connection()
                except Exception as e:
                    print(f"Database closure error: {str(e)}")
    
            # 7. Pulizia aggiuntiva per l'API
            if hasattr(self.api, 'logout'):
                try:
                    self.api.logout()
                except Exception as e:
                    print(f"Logout failed: {str(e)}")
                    
    
if __name__ == "__main__":
    load_dotenv()
    api = API(isLive=True)
    symbols = ["USDJPY", "EURUSD", "GBPUSD", "EURJPY", "USDCHF", "USDCAD", "AUDUSD"] #, "NVDA.NB", "GOOGL.NB", "AMZN.NB", "AMC.N"

    api.login()
    api.trading_account_info = api.get_trading_account_info()
    #pprint.pprint(api.trading_account_info)
    api.clientAccountMarginData = api.get_client_account_margin()
    pprint.pprint(api.clientAccountMarginData)
    
    strategy = TradingStrategy(api, symbols, capital=500, risk_percent=2)

    print("Preparing training data...")
    historical_data, labels = strategy.prepare_training_data(symbols, num_bars=1000)
    model_path = "trading_model.keras" 
    if os.path.exists(model_path):
        print("Loading existing model for update...")
        strategy.model = tf.keras.models.load_model(model_path)
        strategy.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        dummy_input = np.zeros((1, 50, 5))
        dummy_output = np.zeros((1, 3))
        strategy.model.evaluate(dummy_input, dummy_output, verbose=0)

    strategy.train_model(historical_data, labels, epochs=50, batch_size=32)
    strategy.initialize_historical_data()
    print("Starting monitoring...")
    strategy.monitor_symbols()


