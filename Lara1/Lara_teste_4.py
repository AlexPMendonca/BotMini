import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configurações do ativo e parâmetros
ticker = "WING25"  # Atualize para o contrato vigente
lote = 1
stop_loss = 200  # Em pontos
take_profit = 400  # Em pontos

# Conectar ao MetaTrader 5
if not mt5.initialize():
    print("Falha ao conectar ao MT5")
    exit()

def obter_dados_historicos():
    """ Obtém os últimos 2000 candles do Mini Índice para backtest """
    rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_M5, 0, 2000)
    if rates is None or len(rates) == 0:
        print("Erro ao obter dados históricos do MT5")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calcular_indicadores(df):
    """ Calcula indicadores técnicos """
    df['mm9'] = df['close'].rolling(9).mean()
    df['mm21'] = df['close'].rolling(21).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().apply(lambda x: max(x, 0)).rolling(14).mean() /
                                  df['close'].diff().apply(lambda x: abs(x)).rolling(14).mean())))
    df['volatilidade'] = df['close'].rolling(21).std()
    df.dropna(inplace=True)
    return df

def preparar_dados(df):
    """ Prepara os dados para treino e previsão """
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    X = df[['mm9', 'mm21', 'rsi', 'volatilidade']]
    y = df['target']
    return X, y

def treinar_modelo(X, y):
    """ Treina o modelo de Machine Learning """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_scaled, y)
    return modelo, scaler

def prever_sinal(modelo, scaler, df):
    """ Faz previsão da próxima movimentação """
    X_novo = df[['mm9', 'mm21', 'rsi', 'volatilidade']].iloc[-1:].values
    X_novo_scaled = scaler.transform(X_novo)
    previsao = modelo.predict(X_novo_scaled)[0]
    return 'compra' if previsao == 1 else 'venda'

def backtest(df, modelo, scaler):
    """ Realiza o backtest no histórico e exibe a quantidade de dias de negociação e percentual de ganho """
    saldo_inicial = 100  # Saldo inicial fictício
    saldo = saldo_inicial
    posicao = 0
    historico = []
    
    # Contar número de dias únicos no dataset
    df['date'] = df['time'].dt.date
    num_dias_negociacao = df['date'].nunique()
    num_dias_total = (df['date'].max() - df['date'].min()).days + 1
    
    for i in range(len(df) - 1):
        df_teste = df.iloc[:i+1]
        sinal = prever_sinal(modelo, scaler, df_teste)
        preco_entrada = df_teste.iloc[-1]['close']
        preco_saida = df.iloc[i+1]['close']
        
        if sinal == 'compra':
            posicao = preco_entrada
        elif sinal == 'venda' and posicao > 0:
            lucro = (preco_saida - posicao) * lote
            saldo += lucro
            posicao = 0
            historico.append(saldo)
    
    percentual_ganho = ((saldo - saldo_inicial) / saldo_inicial) * 100
    
    print(f"Saldo final: R$ {saldo:.2f}")
    print(f"Número de dias de negociação: {num_dias_negociacao}")
    print(f"Número total de dias no histórico: {num_dias_total}")
    print(f"Percentual de ganho: {percentual_ganho:.2f}%")
    
    # Plotar o saldo ao longo do tempo
    plt.plot(historico)
    plt.xlabel("Operações")
    plt.ylabel("Saldo")
    plt.title("Backtest do Bot")
    plt.show()

# Executar Backtest
df = obter_dados_historicos()
if df is not None:
    df = calcular_indicadores(df)
    X, y = preparar_dados(df)
    modelo, scaler = treinar_modelo(X, y)
    backtest(df, modelo, scaler)

mt5.shutdown()

######## Transforme o código em um robô de trading automático:
#import time

#while True:
 #   df = obter_dados_historicos()
  #  df = calcular_indicadores(df)
   # sinal = prever_sinal(modelo, scaler, df)
#
 #   preco_atual = df['close'].iloc[-1]

  #  if sinal == 'compra':
   #     mt5.order_send( ... )  # Envia ordem de compra
    #elif sinal == 'venda':
     #   mt5.order_send( ... )  # Envia ordem de venda

   # time.sleep(300)  # Espera 5 minutos até o próximo candle
