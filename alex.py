#pip install MetaTrader5 pandas numpy tensorflow#
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#MetaTrader5: Para interagir com a plataforma de negociação MetaTrader 5.
pandas: Para manipulação de dados em forma de DataFrame.
numpy: Para operações numéricas.
MinMaxScaler: Para normalizar os dados entre 0 e 1.
TensorFlow Keras: Para criar e treinar um modelo de rede neural LSTM.#
# Conectar ao MetaTrader 5
if not mt5.initialize():
    print("Erro ao inicializar MetaTrader 5")
    mt5.shutdown()

# Função para coletar dados históricos
def get_historical_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    return pd.DataFrame(rates)

# Coletar dados do mini índice
symbol = "WIN$N"  # Mini Índice
data = get_historical_data(symbol, mt5.TIMEFRAME_M1, 1000)

# Preparar os dados
data['return'] = data['close'].pct_change()
data['target'] = np.where(data['return'].shift(-1) > 0, 1, 0)
data.dropna(inplace=True)

# Normalizar os dados
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'tick_volume']])

# Preparar os dados para a rede neural
X, y = [], []
look_back = 10  # Número de períodos anteriores a considerar
for i in range(len(data_scaled) - look_back):
    X.append(data_scaled[i:i + look_back])
    y.append(data['target'].iloc[i + look_back])

X, y = np.array(X), np.array(y)

# Dividir os dados em conjunto de treinamento e teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Criar o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Prever e executar trade
latest_data = data_scaled[-look_back:].reshape(1, look_back, -1)
prediction = model.predict(latest_data)
signal = 1 if prediction[0][0] > 0.5 else 0

# Função para executar trades
def execute_trade(signal):
    if signal == 1:
        # Comprar
        request = {
            "action": mt5.TRADE_ACTION_BUY,
            "symbol": symbol,
            "volume": 1,
            "type": mt5.ORDER_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": 0,
            "tp": 0,
            "deviation": 10,
            "magic": 0,
            "comment": "Compra com IA",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print("Compra executada" if result.retcode == mt5.TRADE_RETCODE_DONE else "Erro ao executar compra")
    elif signal == 0:
        # Vender
        request = {
            "action": mt5.TRADE_ACTION_SELL,
            "symbol": symbol,
            "volume": 1,
            "type": mt5.ORDER_SELL,
            "price": mt5.symbol_info_tick(symbol).bid,
            "sl": 0,
            "tp": 0,
            "deviation": 10,
            "magic": 0,
            "comment": "Venda com IA",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        print("Venda executada" if result.retcode == mt5.TRADE_RETCODE_DONE else "Erro ao executar venda")

execute_trade(signal)

# Desconectar do MetaTrader 5
mt5.shutdown()
Print("teste")
