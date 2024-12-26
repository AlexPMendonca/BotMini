import MetaTrader5 as mt5

# Inicialize a conexão com o terminal MetaTrader 5
if not mt5.initialize():
    print("Não foi possível iniciar o MetaTrader 5")
    mt5.shutdown()
else:
    print("Conexão com o MetaTrader 5 estabelecida com sucesso")



from datetime import datetime

# Obter dados históricos de ticks
symbol = "WIN$N"
timeframe = mt5.TIMEFRAME_M1  # 1 minuto
rates = mt5.copy_rates_from(symbol, timeframe, datetime(2019, 12, 2), 1000)


# Mostrar as primeiras linhas dos dados
for rate in rates[:5]:
 print(rate)


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Transformar os dados históricos em um DataFrame
df = pd.DataFrame(rates)

# Adicionar características, como a variação de preço (exemplo simples)
df['price_change'] = df['close'] - df['open']

# Criar variáveis independentes (X) e dependentes (y)
X = df[['open', 'high', 'low', 'close']]
y = np.where(df['price_change'] > 0, 1, 0)  # 1 = subir, 0 = descer

# Treinar o modelo
model = RandomForestClassifier()
model.fit(X, y)





def gerar_sinal(model, dados_atual):
    sinal = model.predict([dados_atual])
    return 'BUY' if sinal == 1 else 'SELL'



def enviar_ordem(symbol, sinal):
    lot = 0.1
    price = mt5.symbol_info_tick(symbol).ask if sinal == 'BUY' else mt5.symbol_info_tick(symbol).bid
    slippage = 10
    order_type = mt5.ORDER_TYPE_BUY if sinal == 'BUY' else mt5.ORDER_TYPE_SELL

    # Definir parâmetros da ordem
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "slippage": slippage,
        "magic": 234000,
        "comment": "Bot de IA",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC
    }

    # Enviar ordem
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Erro ao enviar ordem: {result.retcode}")
    else:
        print(f"Ordem {sinal} enviada com sucesso")


import time

while True:
    # Obter os dados mais recentes
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, datetime.now(), 1)
    dados_atual = rates[0][1:]  # Último preço (ou você pode usar mais características)

    # Gerar sinal com o modelo
    sinal = gerar_sinal(model, dados_atual)

    # Enviar ordem
    enviar_ordem(symbol, sinal)

    # Esperar antes de verificar novamente
    time.sleep(60)



mt5.shutdown()
#Alex P Mendonca
