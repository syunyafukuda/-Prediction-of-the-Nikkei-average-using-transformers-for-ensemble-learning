# ライブラリの読み込み

from google.colab import drive
drive.mount('/content/drive')

%%shell
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
python -m pip install git+https://github.com/TA-Lib/ta-lib-python.git@TA_Lib-0.4.26

import talib as ta

# ライブラリのインポート
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter('ignore')

!pip install japanize-matplotlib
import japanize_matplotlib
!pip install yfinance
import yfinance as yfin
yfin.pdr_override()
!pip install mplfinance
import mplfinance as mpf
!pip install requests
import requests
!pip install plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
!pip install fredapi pandas_datareader
from fredapi import Fred
from pandas_datareader.data import DataReader
!pip install pdfminer.six
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.converter import PDFPageAggregator
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from io import StringIO

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# データの取得と目的変数の作成

# 日経平均株価データの取得
start = '1990-01-01'
end = datetime.now().strftime('2023-09-30')
data_master = data.get_data_yahoo('^N225', start, end)

# 前処理
df = data_master.copy()
df.drop(columns=['Adj Close'], inplace=True)
df['Target'] = df['Close'].shift(-1) > df['Close']
df['Target'] = df['Target'].astype(int)

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

# テクニカル指標系

# 実体部分の追加
df['Body'] = df['Open'] - df['Close']

# 日毎のRSI
df['RSI_9'] = ta.RSI(df['Close'], timeperiod=9) #9日
df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14) #14日
df['RSI_22'] = ta.RSI(df['Close'], timeperiod=22) #22日
df['RSI_42'] = ta.RSI(df['Close'], timeperiod=42) #42日
df['RSI_52'] = ta.RSI(df['Close'], timeperiod=52) #52日

# 週毎のRSI（週5日のマーケットを仮定）
df['RSI_9_week'] = ta.RSI(df['Close'], timeperiod=9*5) #9週
df['RSI_13_week'] = ta.RSI(df['Close'], timeperiod=13*5) #13週
df['RSI_26_week'] = ta.RSI(df['Close'], timeperiod=26*5)  # 26週

# 月毎のRSI（週5日のマーケットを仮定）
df['RSI_3_month'] = ta.RSI(df['Close'], timeperiod=13*5) # 3ヶ月（約13週）
df['RSI_6_month'] = ta.RSI(df['Close'], timeperiod=26*5) # 6ヶ月（約26週）
df['RSI_12_month'] = ta.RSI(df['Close'], timeperiod=52*5) # 12ヶ月（約52週）

# 日単位の単純移動平均
df['SMA_5_day'] = ta.SMA(df['Close'].rolling(window=5).mean())
df['SMA_25_day'] = ta.SMA(df['Close'].rolling(window=25).mean())
df['SMA_50_day'] = ta.SMA(df['Close'].rolling(window=50).mean())
df['SMA_75_day'] = ta.SMA(df['Close'].rolling(window=75).mean())
df['SMA_200_day'] = ta.SMA(df['Close'].rolling(window=200).mean())

# 週単位の単純移動平均（週5日のマーケットを仮定）
df['SMA_9_week'] = ta.SMA(df['Close'].rolling(window=9*5).mean())
df['SMA_13_week'] = ta.SMA(df['Close'].rolling(window=13*5).mean())
df['SMA_26_week'] = ta.SMA(df['Close'].rolling(window=26*5).mean())
df['SMA_50_week'] = ta.SMA(df['Close'].rolling(window=50*5).mean())
df['SMA_52_week'] = ta.SMA(df['Close'].rolling(window=52*5).mean())

# 月単位の単純移動平均（月平均21取引日を仮定）
df['SMA_6_month'] = ta.SMA(df['Close'].rolling(window=6*21).mean())
df['SMA_12_month'] = ta.SMA(df['Close'].rolling(window=12*21).mean())
df['SMA_24_month'] = ta.SMA(df['Close'].rolling(window=24*21).mean())
df['SMA_60_month'] = ta.SMA(df['Close'].rolling(window=60*21).mean())

# 日単位のEMA (Exponential Moving Average)
df['EMA_5_day'] = ta.EMA(df['Close'], timeperiod=5)
df['EMA_25_day'] = ta.EMA(df['Close'], timeperiod=25)
df['EMA_50_day'] = ta.EMA(df['Close'], timeperiod=50)
df['EMA_75_day'] = ta.EMA(df['Close'], timeperiod=75)
df['EMA_200_day'] = ta.EMA(df['Close'], timeperiod=200)

# 週単位のEMA (Exponential Moving Average)(週に5取引日と仮定）
df['EMA_9_week'] = ta.EMA(df['Close'], timeperiod=9*5)
df['EMA_13_week'] = ta.EMA(df['Close'], timeperiod=13*5)
df['EMA_26_week'] = ta.EMA(df['Close'], timeperiod=26*5)
df['EMA_50_week'] = ta.EMA(df['Close'], timeperiod=50*5)
df['EMA_52_week'] = ta.EMA(df['Close'], timeperiod=52*5)

# 月単位のEMA (Exponential Moving Average)(月に21取引日と仮定）
df['EMA_6_month'] = ta.EMA(df['Close'], timeperiod=6*21)
df['EMA_12_month'] = ta.EMA(df['Close'], timeperiod=12*21)
df['EMA_24_month'] = ta.EMA(df['Close'], timeperiod=24*21)
df['EMA_60_month'] = ta.EMA(df['Close'], timeperiod=60*21)

# 日単位のWMA (Weighted Moving Average)
df['WMA_5_day'] = ta.WMA(df['Close'], timeperiod=5)
df['WMA_25_day'] = ta.WMA(df['Close'], timeperiod=25)
df['WMA_50_day'] = ta.WMA(df['Close'], timeperiod=50)
df['WMA_75_day'] = ta.WMA(df['Close'], timeperiod=75)
df['WMA_200_day'] = ta.WMA(df['Close'], timeperiod=200)

# 週単位のWMA (Exponential Moving Average)(週に5取引日と仮定）
df['WMA_9_week'] = ta.WMA(df['Close'], timeperiod=9*5)
df['WMA_13_week'] = ta.WMA(df['Close'], timeperiod=13*5)
df['WMA_26_week'] = ta.WMA(df['Close'], timeperiod=26*5)
df['WMA_50_week'] = ta.WMA(df['Close'], timeperiod=50*5)
df['WMA_52_week'] = ta.WMA(df['Close'], timeperiod=52*5)

# 月単位のWMA (Exponential Moving Average)(月に21取引日と仮定）
df['WMA_6_month'] = ta.WMA(df['Close'], timeperiod=6*21)
df['WMA_12_month'] = ta.WMA(df['Close'], timeperiod=12*21)
df['WMA_24_month'] = ta.WMA(df['Close'], timeperiod=24*21)
df['WMA_60_month'] = ta.WMA(df['Close'], timeperiod=60*21)

# 一目均衡表の追加
def ichimoku(df, high_col='High', low_col='Low', close_col='Close'):
    # 転換線 (Conversion Line)
    nine_period_high = df[high_col].rolling(window=9).max()
    nine_period_low = df[low_col].rolling(window=9).min()
    df['ichimoku_Conversion_Line'] = (nine_period_high + nine_period_low) / 2

    # 基準線 (Base Line)
    twenty_six_period_high = df[high_col].rolling(window=26).max()
    twenty_six_period_low = df[low_col].rolling(window=26).min()
    df['ichimoku_Base_Line'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # 先行スパン A (Leading Span A)
    df['ichimoku_Leading_Span A'] = ((df['ichimoku_Conversion_Line'] + df['ichimoku_Base_Line']) / 2).shift(26)

    # 先行スパン B (Leading Span B)
    fifty_two_period_high = df[high_col].rolling(window=52).max()
    fifty_two_period_low = df[low_col].rolling(window=52).min()
    df['ichimoku_Leading_Span B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    # 遅行スパン (Lagging Span)
    df['ichimoku_Lagging_Span'] = df[close_col].shift(-26)

ichimoku(df)

# ボリンジャーバンドの追加
periods = [5,
           20,
           25,
           65, #13週
           252 #12か月
           ]

num_stds = [1, 2, 3]

for period in periods:
    for std in num_stds:
        upper, middle, lower = ta.BBANDS(df['Close'].values, timeperiod=period, nbdevup=std, nbdevdn=std)
        df[f'BB_Upper_{period}_std{std}'] = upper
        df[f'BB_Lower_{period}_std{std}'] = lower

# 'SMA_'で始まる列名を自動的に探してリストに格納
sma_columns = df.filter(like='SMA_').columns.tolist()

# 移動平均乖離率の追加
for sma_column in sma_columns:
    df[f'{sma_column}_DevRate'] = (df['Close'] - df[sma_column]) / df[sma_column] * 100

# 日毎MACDの追加
def add_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macd_signal, macd_hist = ta.MACD(df[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

add_macd(df)

# 週毎MACDの追加
def add_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    df_copy = df.copy()  # 元のDataFrameを変更しないためにコピーを作成
    macd, macd_signal, macd_hist = ta.MACD(df_copy[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df_copy['MACD_weekly'] = macd
    df_copy['MACD_Signal_weekly'] = macd_signal
    df_copy['MACD_Hist_weekly'] = macd_hist
    return df_copy  # 変更を加えたDataFrameを返す

# 週毎にリサンプリングして終値の平均を取る
df_weekly = df.resample('W').agg({'Close':'last'})  # 'W'は週毎、'last'は週の最後のデータを取る
df_weekly.fillna(method='ffill', inplace=True)  # 前方補間

# 新しい周期でMACDを計算
df_weekly_with_macd = add_macd(df_weekly, close_col='Close')

# インデックスをDatetimeIndexに変換
df.index = pd.to_datetime(df.index)

# 週の最後の日付を取得
week_last_day = df.resample('W').last().index

# 新しいカラムに週の最後の日付をセット
df['Week_last_day'] = df.index.to_series().apply(lambda x: week_last_day[week_last_day >= x][0])

# 元の日毎のデータに週の最後の日付を追加
df_weekly_with_macd['Week_last_day'] = df_weekly_with_macd.index

# インデックスを保存
original_index = df.index

df = pd.merge(df, df_weekly_with_macd[['Week_last_day', 'MACD_weekly', 'MACD_Signal_weekly', 'MACD_Hist_weekly']], on='Week_last_day', how='left')

# インデックスを再設定
df.index = original_index

df.drop('Week_last_day', axis=1, inplace=True)

# 月毎MACDの追加
def add_monthly_macd(df, close_col='Close', fastperiod=12, slowperiod=26, signalperiod=9):
    df_copy = df.copy()  # 元のDataFrameを変更しないためにコピーを作成
    macd, macd_signal, macd_hist = ta.MACD(df_copy[close_col], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    df_copy['MACD_monthly'] = macd
    df_copy['MACD_Signal_monthly'] = macd_signal
    df_copy['MACD_Hist_monthly'] = macd_hist
    return df_copy  # 変更を加えたDataFrameを返す

# 月毎にリサンプリングして終値を取る
df_monthly = df.resample('M').agg({'Close':'last'})  # 'M'は月毎、'last'は月の最後のデータを取る
df_monthly.fillna(method='ffill', inplace=True)  # 前方補間

# 新しい周期でMACDを計算
df_monthly_with_macd = add_monthly_macd(df_monthly, close_col='Close')

# 月の最後の日付を取得
month_last_day = df.resample('M').last().index

# 新しいカラムに月の最後の日付をセット
df['Month_last_day'] = df.index.to_series().apply(lambda x: month_last_day[month_last_day >= x][0])

# 元の日毎のデータに月の最後の日付を追加
df_monthly_with_macd['Month_last_day'] = df_monthly_with_macd.index

# インデックスを保存
original_index = df.index

# マージ
df = pd.merge(df, df_monthly_with_macd[['Month_last_day', 'MACD_monthly', 'MACD_Signal_monthly', 'MACD_Hist_monthly']], on='Month_last_day', how='left')

# インデックスを再設定
df.index = original_index

# 不要なカラムを削除
df.drop('Month_last_day', axis=1, inplace=True)

# 日毎のストキャスティクスの追加
def add_stochastic(df, high_col='High', low_col='Low', close_col='Close', fastk_period=5, slowk_period=3, slowd_period=3):
    slowk, slowd = ta.STOCH(df[high_col], df[low_col], df[close_col], fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
    df['Slow_Stochastic_K'] = slowk
    df['Slow_Stochastic_D'] = slowd

    fastk, fastd = ta.STOCHF(df[high_col], df[low_col], df[close_col], fastk_period=fastk_period, fastd_period=slowd_period)
    df['Fast_Stochastic_K'] = fastk
    df['Fast_Stochastic_D'] = fastd

add_stochastic(df)

# 週毎ストキャスティクスの追加
def add_weekly_stochastics(df, high_col='High', low_col='Low', close_col='Close',
                           fastk_period=5, fastd_period=3, slowk_period=5, slowd_period=3):
    df_copy = df.copy()

    # Fast Stochasticを計算
    fastk, fastd = ta.STOCHF(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                             fastk_period=fastk_period, fastd_period=fastd_period)
    df_copy['Fast_Stochastic_K_weekly'] = fastk
    df_copy['Fast_Stochastic_D_weekly'] = fastd

    # Slow Stochasticを計算
    slowk, slowd = ta.STOCH(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                            fastk_period=slowk_period, slowd_period=slowd_period)
    df_copy['Slow_Stochastic_K_weekly'] = slowk
    df_copy['Slow_Stochastic_D_weekly'] = slowd

    return df_copy

# 週毎にリサンプリングして各種データを取る
df_weekly = df.resample('W').agg({'High':'max', 'Low':'min', 'Close':'last'})
df_weekly.fillna(method='ffill', inplace=True)

# 週毎のストキャスティクスを計算
df_weekly_with_stochastics = add_weekly_stochastics(df_weekly)  # 関数名を修正

# 新しいカラムに週の最後の日付をセット
df_weekly_with_stochastics['Week_last_day'] = df_weekly_with_stochastics.index

# インデックスをDatetimeIndexに変換
df.index = pd.to_datetime(df.index)

# 週の最後の日付を取得
week_last_day = df.resample('W').last().index

# 新しいカラムに週の最後の日付をセット
df['Week_last_day'] = df.index.to_series().apply(lambda x: week_last_day[week_last_day >= x][0])

# 元の日毎のデータに週毎のストキャスティクスをマージ
df = pd.merge(df, df_weekly_with_stochastics[['Week_last_day', 'Fast_Stochastic_K_weekly', 'Fast_Stochastic_D_weekly', 'Slow_Stochastic_K_weekly', 'Slow_Stochastic_D_weekly']], on='Week_last_day', how='left')  # カラムを追加

# インデックスを再設定
df.index = original_index

# Week_last_dayカラムを削除
df.drop('Week_last_day', axis=1, inplace=True)

# 月毎ストキャスティクスの追加
def add_monthly_stochastics(df, high_col='High', low_col='Low', close_col='Close',
                            fastk_period=5, fastd_period=3, slowk_period=5, slowd_period=3):
    df_copy = df.copy()

    # Fast Stochasticを計算
    fastk, fastd = ta.STOCHF(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                             fastk_period=fastk_period, fastd_period=fastd_period)
    df_copy['Fast_Stochastic_K_monthly'] = fastk
    df_copy['Fast_Stochastic_D_monthly'] = fastd

    # Slow Stochasticを計算
    slowk, slowd = ta.STOCH(df_copy[high_col], df_copy[low_col], df_copy[close_col],
                            fastk_period=slowk_period, slowd_period=slowd_period)
    df_copy['Slow_Stochastic_K_monthly'] = slowk
    df_copy['Slow_Stochastic_D_monthly'] = slowd

    return df_copy

# 月毎にリサンプリングして各種データを取る
df_monthly = df.resample('M').agg({'High':'max', 'Low':'min', 'Close':'last'})
df_monthly.fillna(method='ffill', inplace=True)

# 月毎のストキャスティクスを計算
df_monthly_with_stochastics = add_monthly_stochastics(df_monthly)

# 新しいカラムに月の最後の日付をセット
df_monthly_with_stochastics['Month_last_day'] = df_monthly_with_stochastics.index

# 月の最後の日付を取得
month_last_day = df.resample('M').last().index

# 新しいカラムに月の最後の日付をセット
df['Month_last_day'] = df.index.to_series().apply(lambda x: month_last_day[month_last_day >= x][0])

# 元の日毎のデータに月毎のストキャスティクスをマージ
df = pd.merge(df, df_monthly_with_stochastics[['Month_last_day', 'Fast_Stochastic_K_monthly', 'Fast_Stochastic_D_monthly', 'Slow_Stochastic_K_monthly', 'Slow_Stochastic_D_monthly']], on='Month_last_day', how='left')

# インデックスを再設定
df.index = original_index

# Month_last_dayカラムを削除
df.drop('Month_last_day', axis=1, inplace=True)

from scipy.stats import spearmanr

# RCIの計算
def calc_rci(df, column, window):
    if len(df) < window:
        return np.nan
    ranks = df[column].rank()
    return 100 * (1 - 6 * sum((ranks - np.arange(window) - 1)**2) / (window * (window**2 - 1)))

# DataFrameにRCIを追加
def add_rci(df, column, window, suffix):
    rci_values = []
    for i in range(len(df)):
        if i < window:
            rci_values.append(np.nan)
            continue
        rci = calc_rci(df.iloc[i-window:i, :], column, window)
        rci_values.append(rci)
    df[f'RCI_{suffix}'] = rci_values

# 日単位
add_rci(df, 'Close', 9, '9_day')
add_rci(df, 'Close', 26, '26_day')

# 週単位（週5日のマーケットを仮定）
add_rci(df, 'Close', 9 * 5, '9_week')
add_rci(df, 'Close', 26 * 5, '26_week')

# 月単位（月平均21取引日を仮定）
add_rci(df, 'Close', 9 * 21, '9_month')
add_rci(df, 'Close', 26 * 21, '26_month')

# ATRを追加
df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

# CMFを追加
df['CMF'] = ta.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)

# ROCを追加
df['ROC_10'] = ta.ROC(df['Close'], timeperiod=10)
df['ROC_14'] = ta.ROC(df['Close'], timeperiod=14)
df['ROC_25'] = ta.ROC(df['Close'], timeperiod=25)

# CCIを追加
df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

# PASR（パラボリック）を追加
df['PSAR'] = ta.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)

# OBVを追加
df['OBV'] = ta.OBV(df['Close'], df['Volume'])

# データの整形と分割

# 2000年以降のデータのみを取得
df = df.loc['2000-01-01':]

df['ichimoku_Lagging_Span'].fillna(method='ffill', inplace=True)

# データをトレーニング、検証、テスト用に分割
train = df['2000-01-01':'2022-12-31']
val = df['2023-01-01':'2023-09-28']
test = df['2023-09-27':'2023-09-28']

スケーリングはMinMaxScaler

from sklearn.preprocessing import MinMaxScaler

feature_columns = [col for col in df.columns if col != 'Target']

# Instantiate a scaler for each feature
scalers = {}

# Create a copy of the dataframe for the scaled data
train_scaled = pd.DataFrame()
val_scaled = pd.DataFrame()

for feature in feature_columns:
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit on train data
    scaler.fit(train[feature].values.reshape(-1,1))

    # Transform on both train and val data
    train_scaled[feature] = scaler.transform(train[feature].values.reshape(-1,1)).reshape(-1)
    val_scaled[feature] = scaler.transform(val[feature].values.reshape(-1,1)).reshape(-1)
  
    # Save the scaler for later use
    scalers[feature] = scaler

from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define RMSE loss function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

n_features = train_scaled.shape[1]
n_steps = 1  # number of time steps - it could be more depending on your data

# Define a function for windowing the data
def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Apply the windowing function
X_train, y_train = create_dataset(train_scaled[feature_columns], train['Target'], time_steps=n_steps)
X_validation, y_validation = create_dataset(val_scaled[feature_columns], val['Target'], time_steps=n_steps)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# timeseriestransformerの導入

!git clone https://github.com/DanielAtKrypton/time_series_transformer.git

import sys
sys.path.append('/content/time_series_transformer')

"""
Decoder.py
This script hosts the Decoder class.
It performs the Decoder block from Attention is All You Need.
"""
import torch
import torch.nn as nn

from time_series_transformer.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadAttentionChunk,
    MultiHeadAttentionWindow
)
from time_series_transformer.positionwise_feed_forward import PositionwiseFeedForward


class Decoder(nn.Module):
    """Decoder block from Attention is All You Need.

    Apply two Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 **kwargs):
        """Initialize the Decoder block"""
        super().__init__(**kwargs)

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(
            d_model, q, v, h, attention_size=attention_size, **kwargs)
        self._encoderDecoderAttention = MHA(
            d_model, q, v, h, attention_size=attention_size, **kwargs)
        self._feedForward = PositionwiseFeedForward(d_model, **kwargs)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Decoder block.

        Apply the self attention block, add residual and normalize.
        Apply the encoder-decoder attention block, add residual and normalize.
        Apply the feed forward network, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        memory:
            Memory tensor with shape (batch_size, K, d_model)
            from encoder output.

        Returns
        -------
        x:
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x, mask="subsequent")
        x = self._dropout(x)
        x = self._layerNorm1(x + residual)

        # Encoder-decoder attention
        residual = x
        x = self._selfAttention(query=x, key=memory, value=memory)
        x = self._dropout(x)
        x = self._layerNorm2(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dropout(x)
        x = self._layerNorm3(x + residual)

        return x

"""
Encoder
"""
import torch
import torch.nn as nn

from time_series_transformer.multi_head_attention import (MultiHeadAttention,
                                   MultiHeadAttentionChunk,
                                   MultiHeadAttentionWindow)
from time_series_transformer.positionwise_feed_forward import PositionwiseFeedForward

class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 **kwargs):
        """Initialize the Encoder block"""
        super().__init__(**kwargs)

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size, **kwargs)
        self._feedForward = PositionwiseFeedForward(d_model, **kwargs)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dropout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dropout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map

"""
MultiHeadAttention
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from time_series_transformer.utils import generate_local_map_mask

def sqrt(value) -> torch.Tensor:
    return torch.sqrt(torch.tensor(float(value)))

class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._h = h
        self._attention_size = attention_size

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / sqrt(K)

        # Compute local map mask
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(
                K, self._attention_size, mask_future=False, device=queries.device)
            self._scores = self._scores.masked_fill(attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K), device=queries.device), diagonal=1).bool()
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    chunk_size:
        Size of chunks to apply attention on.
        Last one may be smaller (see :class:`torch.Tensor.chunk`).
        Default is 168.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 chunk_size: Optional[int] = 168,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._chunk_size = chunk_size

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._chunk_size, \
            self._chunk_size)), diagonal=1).bool(), requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask(self._chunk_size, \
                self._attention_size), requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]
        n_chunk = K // self._chunk_size

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(
            self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(
            self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(
            self._h, dim=-1), dim=0).chunk(n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(
            1, 2)) / sqrt(self._chunk_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(
                self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(
                self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(
            n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention


class MultiHeadAttentionWindow(MultiHeadAttention):
    """Multi Head Attention block with moving window.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks using a moving window.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    window_size:
        Size of the window used to extract chunks.
        Default is 168
    padding:
        Padding around each window. Padding will be applied to input sequence.
        Default is 168 // 4 = 42.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 window_size: Optional[int] = 168,
                 padding: Optional[int] = 168 // 4,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, attention_size, **kwargs)

        self._window_size = window_size
        self._padding = padding
        self._q = q
        self._v = v

        # Step size for the moving window
        self._step = self._window_size - 2 * self._padding

        # Score mask for decoder
        self._future_mask = nn.Parameter(torch.triu(torch.ones((self._window_size, \
            self._window_size)), diagonal=1).bool(), requires_grad=False)

        if self._attention_size is not None:
            self._attention_mask = nn.Parameter(generate_local_map_mask( \
                self._window_size, self._attention_size), requires_grad=False)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        batch_size = query.shape[0]

        # Apply padding to input sequence
        query = F.pad(query.transpose(1, 2), (self._padding,
                                              self._padding), 'replicate').transpose(1, 2)
        key = F.pad(key.transpose(1, 2), (self._padding,
                                          self._padding), 'replicate').transpose(1, 2)
        value = F.pad(value.transpose(1, 2), (self._padding,
                                              self._padding), 'replicate').transpose(1, 2)

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Divide Q, K and V using a moving window
        queries = queries.unfold(dimension=1, size=self._window_size, step=self._step).reshape(
            (-1, self._q, self._window_size)).transpose(1, 2)
        keys = keys.unfold(dimension=1, size=self._window_size, step=self._step).reshape(
            (-1, self._q, self._window_size)).transpose(1, 2)
        values = values.unfold(dimension=1, size=self._window_size, step=self._step).reshape(
            (-1, self._v, self._window_size)).transpose(1, 2)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(
            1, 2)) / sqrt(self._window_size)

        # Compute local map mask
        if self._attention_size is not None:
            self._scores = self._scores.masked_fill(
                self._attention_mask, float('-inf'))

        # Compute future mask
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(
                self._future_mask, float('-inf'))

        # Apply softmax
        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Fold chunks back
        attention = attention.reshape(
            (batch_size*self._h, -1, self._window_size, self._v))
        attention = attention[:, :, self._padding:-self._padding, :]
        attention = attention.reshape((batch_size*self._h, -1, self._v))

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

"""
PositionwiseFeedForward
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048):
        """Initialize the PFF block."""
        super().__init__()

        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.relu(self._linear1(x)))

"""
Transformer
"""
import torch
import torch.nn as nn

from time_series_transformer.decoder import Decoder
from time_series_transformer.encoder import Encoder
from time_series_transformer.utils import generate_original_PE, generate_regular_PE


class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 d_input: int = 1,
                 d_model: int = 32,
                 d_output: int = 1,
                 q: int = 4,
                 v: int = 4,
                 h: int = 4,
                 N: int = 4,
                 attention_size: int = 6,
                 dropout: float = 0.2,
                 chunk_mode: bool = None,
                 pe: str = None,
                 pe_period: int = 24):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                   q,
                                                   v,
                                                   h,
                                                   attention_size=attention_size,
                                                   dropout=dropout,
                                                   chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                   q,
                                                   v,
                                                   h,
                                                   attention_size=attention_size,
                                                   dropout=dropout,
                                                   chunk_mode=chunk_mode) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output

class CustomTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linear = nn.Linear(self._d_model, 1)  # 出力層の変更

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output

# モデルの構築と学習：Transformer

# モデルのインスタンス化
model = CustomTransformer(
    d_input=n_features,  
    d_model=64,  
    d_output=1,  
    q=4,
    v=4,
    h=4,
    N=4,
    attention_size=6,
    dropout=0.2,
    chunk_mode=None,
    pe=None,
    pe_period=24
)

model

# 損失関数とオプティマイザの定義
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0001)

best_accuracy_val = 0  
best_model_state = None  

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証
    model.eval()
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            val_outputs.append(outputs.squeeze())
            val_targets.append(targets)
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)
        val_preds = (val_outputs > 0.5).float()
        accuracy_val = accuracy_score(val_targets.numpy(), val_preds.numpy())
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy_val * 100:.2f}%')

        # 最高の検証精度を更新
        if accuracy_val > best_accuracy_val:
            best_accuracy_val = accuracy_val
            best_model_state = model.state_dict()

    # 訓練データに対する評価
    train_outputs = []
    train_targets = []
    with torch.no_grad():
        for batch in train_loader:
            inputs, targets = batch
            outputs = model(inputs)
            train_outputs.append(outputs.squeeze())
            train_targets.append(targets)
        train_outputs = torch.cat(train_outputs)
        train_targets = torch.cat(train_targets)
        train_preds = (train_outputs > 0.5).float()
        accuracy_train = accuracy_score(train_targets.numpy(), train_preds.numpy())
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {accuracy_train * 100:.2f}%')

        # トレーニングループの最後にこれを追加
        print(f'Best Validation Accuracy: {best_accuracy_val * 100:.2f}%')


# 最高のモデルの状態を保存
torch.save(best_model_state, 'best_model.pth')

# 訓練の予測結果
model.eval()
train_outputs = []
train_targets = []
with torch.no_grad():
    for batch in train_loader:
        inputs, targets = batch
        outputs = model(inputs)
        train_outputs.append(outputs.squeeze())
        train_targets.append(targets)
    train_outputs = torch.cat(train_outputs)
    train_targets = torch.cat(train_targets)

    # 出力とターゲットをnumpy配列に変換
    train_outputs_np = train_outputs.numpy()
    train_targets_np = train_targets.numpy()

    # 結果をデータフレームに保存
    train_results_df = pd.DataFrame({
        'Predicted_Probability': train_outputs_np,
        'Predicted_Class': (train_outputs_np > 0.5).astype(int),
        'True_Class': train_targets_np
    })

    # 結果をCSVファイルに保存（任意）
    # train_results_df.to_csv('training_results.csv', index=False)

# 結果のデータフレームを表示
print(train_results_df.head())

# 検証の予測結果
model.eval()
val_outputs = []
val_targets = []
with torch.no_grad():
    for batch in val_loader:
        inputs, targets = batch
        outputs = model(inputs)
        val_outputs.append(outputs.squeeze())
        val_targets.append(targets)
    val_outputs = torch.cat(val_outputs)
    val_targets = torch.cat(val_targets)

    # 出力とターゲットをnumpy配列に変換
    val_outputs_np = val_outputs.numpy()
    val_targets_np = val_targets.numpy()

    # 結果をデータフレームに保存
    results_df = pd.DataFrame({
        'Predicted_Probability': val_outputs_np,
        'Predicted_Class': (val_outputs_np > 0.5).astype(int),
        'True_Class': val_targets_np
    })

    # 結果をCSVファイルに保存（任意）
    #results_df.to_csv('validation_results.csv', index=False)

# 結果のデータフレームを表示
print(results_df.head())

tst_binary_df = pd.concat([train_results_df, results_df], ignore_index=True)
print(tst_binary_df)

df = df['2000-01-05':'2023-09-28']

df.drop(pd.Timestamp('2023-01-04'), inplace=True)

# dfの日付を取得
dates = df.index

# tst_binary_dfのインデックスを新しい日付に設定
tst_binary_df.index = dates

tst_binary_df.drop(columns=['Predicted_Class', 'True_Class'], inplace=True)

tst_binary_df.to_csv('YOURPATH')

# results_dfの行数を取得
number_of_rows = len(tst_binary_df)

# 行数を表示
print(number_of_rows)
