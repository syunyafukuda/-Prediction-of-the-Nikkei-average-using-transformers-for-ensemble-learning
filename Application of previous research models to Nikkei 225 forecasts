# ライブラリのインポート

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter('ignore')

# 必要なライブラリをインストールします
!pip install yfinance
import yfinance as yf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 日経平均のティッカーシンボルと日付範囲を指定します
ticker = "^N225"
start_date = "1990-01-01"
end_date = "2023-09-30"
# データを取得します
df = yf.download(ticker, start=start_date, end=end_date)
# 取得したデータを確認します
print(f"\n{ticker}")
print(df.head())

df = df.drop(columns=['Adj Close'])

# 新たな目的変数Targetを作成します。Targetは1日後のClose価格です
df['Target'] = df['Close'].shift(-1)
df = df.drop(pd.Timestamp('2023-09-29'))
df.head()

# 日経

## 標準化、分割、学習

train = df['2000-04-01':'2022-12-31']
validation = df['2023-01-01':]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# StandardScalerのインスタンスを作成します
scaler = StandardScaler()

# dfのデータを標準化します
scaled = scaler.fit_transform(df)

# 標準化したデータを再びデータフレームに変換します
scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

# 指定された日付でデータを訓練データと検証データに分割します
train_data = scaled_df['2000-04-01':'2022-12-31']
validation_data = scaled_df['2023-01-01':]

# 訓練データと検証データの最初の数行を表示して確認します
print('Training data:')
print(train_data.head())
print('\nValidation data:')
print(validation_data.head())

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 特徴量と目的変数を準備します
X_train = np.reshape(train_data.values[:, :-1], (train_data.shape[0], train_data.shape[1] - 1, 1))
y_train = train_data.values[:, -1]
X_validation = np.reshape(validation_data.values[:, :-1], (validation_data.shape[0], validation_data.shape[1] - 1, 1))
y_validation = validation_data.values[:, -1]

# Transformer モデルを構築します
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

model = build_model(X_train.shape[1:])

# モデルをコンパイルします
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error")

# モデルを訓練します
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation), verbose=2)

# 検証データでモデルを評価し、予測を行います
y_pred = model.predict(X_validation)

# 予測されたデータを元のスケールに戻します
y_pred_2d = y_pred.reshape(-1, 1)  # 予測を2D配列に変換します
X_validation_2d = X_validation.reshape(X_validation.shape[0], -1)  # X_validationを2D配列に変換します
y_pred_original_scale = scaler.inverse_transform(np.hstack((X_validation_2d, y_pred_2d)))[:, -1]

# 予測されたデータをデータフレームに変換します
predicted_target_df = pd.DataFrame(y_pred_original_scale, index=validation_data.index, columns=['Predicted_Target'])

# 予測されたデータの最初の数行を表示して確認します
print(predicted_target_df.head())

import plotly.graph_objects as go

# 'train' と 'validation' のデータを連結します
concatenated_data = pd.concat([train, validation])

# 新しいプロットオブジェクトを作成します
fig = go.Figure()

# 連結されたデータの実際の目標値をプロットします
fig.add_trace(go.Scatter(x=concatenated_data.index, y=concatenated_data['Target'],
                         mode='lines',
                         name='Actual Target',
                         line=dict(color='blue')))

# 予測期間のTransformerモデルの予測をプロットします
fig.add_trace(go.Scatter(x=predicted_target_df.index, y=predicted_target_df['Predicted_Target'],
                         mode='lines',
                         name='Transformer Prediction',
                         line=dict(color='red')))  # 色を赤に設定して区別します

# プロットのタイトルと軸ラベルを設定します
fig.update_layout(title='Transformer Model Forecast vs Actual Values',
                  xaxis_title='Date',
                  yaxis_title='Value')

# プロットを表示します
fig.show()

## ARIMAモデルでの予測

# 必要なライブラリをインストールします
!pip install pmdarima

# pmdarimaをインポートします
import pmdarima as pm

# 訓練データの 'Close' 列に対してARIMAモデルを訓練します
arima_model = pm.auto_arima(train_data['Close'],
                      seasonal=False,  # 時系列の季節性をFalseと定義します
                      stepwise=True,  # ステップワイズアプローチを使用します
                      suppress_warnings=True,  # 警告を抑制します
                      D=0, max_D=0,  # 最大の季節的な差分の順序を0とします
                      error_action="ignore")  # エラーを無視します

# モデルのサマリーを表示して確認します
arima_model.summary()

# 訓練されたモデルを使用してvalidation_dataの'Close'列の予測を行います
validation_forecast = arima_model.predict(n_periods=len(validation_data))

# 予測結果をデータフレームに変換し、インデックスを設定します
validation_forecast_df = pd.DataFrame(validation_forecast, index=validation_data.index, columns=['Predicted_Target'])

# 予測結果の最初の数行を表示して確認します
print(validation_forecast_df.head())

# 予測データのスケーリングを逆に戻すために、元のスケールで0を使ってダミーデータを作成します
dummy_features = np.zeros(shape=(len(validation_forecast), scaled_df.shape[1] - 1))
# 'Predicted_Target'列をダミーデータに追加します
inverse_transform_input = np.column_stack((dummy_features, validation_forecast))

# データのスケーリングを逆に戻します
inverse_transform_output = scaler.inverse_transform(inverse_transform_input)
# スケーリングを逆に戻した予測データを取得します
original_scale_forecast = inverse_transform_output[:, -1]

# スケーリングを逆に戻した予測データをデータフレームに変換します
original_scale_forecast_df = pd.DataFrame(original_scale_forecast, index=validation_data.index, columns=['Original_Scale_Predicted_Target'])
arima_df = original_scale_forecast_df

# データフレームの最初の数行を表示して確認します
print(original_scale_forecast_df.head())

import plotly.graph_objects as go

# 'PAST' と 'FUTURE' のデータを連結します
concatenated_data = pd.concat([train, validation])

# 新しいプロットオブジェクトを作成します
fig = go.Figure()

# 連結されたデータをプロットします
fig.add_trace(go.Scatter(x=concatenated_data.index, y=concatenated_data['Target'],
                         mode='lines',
                         name='PAST + FUTURE',
                         line=dict(color='blue')))

# 予測期間の'ARIMA'予測をプロットします
fig.add_trace(go.Scatter(x=validation_data.index, y=original_scale_forecast_df['Original_Scale_Predicted_Target'],
                         mode='lines',
                         name='ARIMA',
                         line=dict(color='green')))

# プロットのタイトルと軸ラベルを設定します
fig.update_layout(title='ARIMA Model Forecast vs Actual Values',
                  xaxis_title='Date',
                  yaxis_title='Value')

# プロットを表示します
fig.show()

## LSTMでの予測

Learning Rate：10−5GeneratorLength1, 3, 5, 7, 9
Hidden Layers：0
Neurons：64
Epochs：256

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 特徴量と目的変数を準備します
X_train = np.reshape(train_data.values[:, :-1], (train_data.shape[0], 1, train_data.shape[1] - 1))
y_train = train_data.values[:, -1]
X_validation = np.reshape(validation_data.values[:, :-1], (validation_data.shape[0], 1, validation_data.shape[1] - 1))
y_validation = validation_data.values[:, -1]

# LSTM モデルを構築します
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))  # Neurons
model.add(Dense(1))  # Hidden Layers (ただし、出力層は含まれています)
optimizer = Adam(lr=0.00001)  # Learning Rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# モデルを訓練します
model.fit(X_train, y_train, epochs=256, batch_size=1, validation_data=(X_validation, y_validation), verbose=2, shuffle=False)

# validation_dataを使用して予測を行います
y_pred = model.predict(X_validation)

# 予測されたデータを元のスケールに戻します
y_pred_original_scale = scaler.inverse_transform(np.hstack((X_validation.reshape(X_validation.shape[0], X_validation.shape[2]), y_pred)))[:, -1]

# 予測されたデータをデータフレームに変換します
predicted_target_df = pd.DataFrame(y_pred_original_scale, index=validation_data.index, columns=['Predicted_Target'])
lstm_df = predicted_target_df

# 予測されたデータの最初の数行を表示して確認します
print(predicted_target_df.head())

import plotly.graph_objects as go

# 'PAST' と 'FUTURE' のデータを連結します
concatenated_data = pd.concat([train, validation])

# 新しいプロットオブジェクトを作成します
fig = go.Figure()

# 連結されたデータをプロットします
fig.add_trace(go.Scatter(x=concatenated_data.index, y=concatenated_data['Target'],
                         mode='lines',
                         name='PAST + FUTURE',
                         line=dict(color='blue')))

# 予測期間の'LSTM'予測をプロットします
fig.add_trace(go.Scatter(x=validation_data.index, y=predicted_target_df['Predicted_Target'],
                         mode='lines',
                         name='LSTM',
                         line=dict(color='red')))  # 色を赤に設定して区別します

# プロットのタイトルと軸ラベルを設定します
fig.update_layout(title='LSTM Model Forecast vs Actual Values',
                  xaxis_title='Date',
                  yaxis_title='Value')

# プロットを表示します
fig.show()

## Transformerの定義

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

## Transformerでの予測

from sklearn.preprocessing import StandardScaler

# StandardScalerのインスタンスを作成します
scaler = StandardScaler()

# dfのデータを標準化します
scaled = scaler.fit_transform(df)

# 標準化したデータを再びデータフレームに変換します
scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

# 指定された日付でデータを訓練データと検証データに分割します
train = scaled_df['2000-01-01':'2022-12-31']
validation = scaled_df['2023-01-01':]

# 訓練データと検証データの最初の数行を表示して確認します
print('Training data:')
print(train.head())
print('\nValidation data:')
print(validation.head())

from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

feature_columns = [col for col in df.columns if col != 'Target']

# 特徴量と目的変数を準備します
X_train = np.reshape(train.values[:, :-1], (train.shape[0], 1, train.shape[1] - 1))
y_train = train.values[:, -1]
X_validation = np.reshape(validation.values[:, :-1], (validation.shape[0], 1, validation.shape[1] - 1))
y_validation = validation.values[:, -1]

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error

# あなたのTransformerモデルのインスタンス化
model = Transformer(
    d_input=5,
    d_model=256,
    d_output=1,
    q=9, #ハイパラ Q
    v=9, #ハイパラ V
    h=9, #ハイパラ H
    N=1,
    attention_size=9, #ハイパラ　Attention　Length
    dropout=0.2,
    chunk_mode=None,
    pe=None,
    pe_period=24
)

# 損失関数とオプティマイザの定義
criterion = nn.MSELoss()  # 平均二乗誤差損失
optimizer = optim.Adam(model.parameters(), lr=0.00001) #ハイパラ　学習率
num_epochs = 128

# 訓練ループ
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 検証
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(inputs), targets) for inputs, targets in val_loader) / len(val_loader)
    print(f'Epoch {epoch+1}, Val Loss: {val_loss.item()}')

# モデルを評価モードに設定します
model.eval()

# PyTorchテンソルに変換します
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)

# モデルを使用して予測を行います
with torch.no_grad():
    y_pred = model(X_validation_tensor)

# PyTorchテンソルからnumpy配列に変換し、余分な次元を削除します
y_pred_numpy = y_pred.squeeze().numpy()

# 予測されたデータを元のスケールに戻します
# 最初にX_validationとy_predを結合し、適切な形状に整形します
combined = np.hstack((X_validation.reshape(X_validation.shape[0], -1), y_pred_numpy.reshape(-1, 1)))
y_pred_original_scale = scaler.inverse_transform(combined)[:, -1]

# 予測されたデータをデータフレームに変換します
predicted_target_df = pd.DataFrame(y_pred_original_scale, index=validation.index, columns=['Predicted_Target'])
transformer_df = predicted_target_df

# 予測されたデータの最初の数行を表示して確認します
print(predicted_target_df.head())

train = df['2000-01-01':'2022-12-31']
validation = df['2023-01-01':]

import plotly.graph_objects as go

# 'train' と 'validation' のデータを連結します
concatenated_data = pd.concat([train, validation])

# 新しいプロットオブジェクトを作成します
fig = go.Figure()

# 連結されたデータの実際の目標値をプロットします
fig.add_trace(go.Scatter(x=concatenated_data.index, y=concatenated_data['Target'],
                         mode='lines',
                         name='Actual Target',
                         line=dict(color='blue')))

# 予測期間のTransformerモデルの予測をプロットします
fig.add_trace(go.Scatter(x=predicted_target_df.index, y=predicted_target_df['Predicted_Target'],
                         mode='lines',
                         name='Transformer Prediction',
                         line=dict(color='red')))  # 色を赤に設定して区別します

# プロットのタイトルと軸ラベルを設定します
fig.update_layout(title='Transformer Model Forecast vs Actual Values',
                  xaxis_title='Date',
                  yaxis_title='Value')

# プロットを表示します
fig.show()

## 全モデルの予測の可視化とRMSEの比較

import plotly.graph_objects as go

# 'train' と 'validation' のデータを連結します
concatenated_data = pd.concat([train, validation])

# 新しいプロットオブジェクトを作成します
fig = go.Figure()

# 連結されたデータの実際の目標値をプロットします
fig.add_trace(go.Scatter(x=concatenated_data.index, y=concatenated_data['Target'],
                         mode='lines',
                         name='PAST + FUTURE',
                         line=dict(color='blue')))

# 予測期間のARIMAモデルの予測をプロットします
fig.add_trace(go.Scatter(x=arima_df.index, y=arima_df['Original_Scale_Predicted_Target'],
                         mode='lines',
                         name='ARIMA Prediction',
                         line=dict(color='red')))  # 色を赤に設定して区別します

# 予測期間のLSTMモデルの予測をプロットします
fig.add_trace(go.Scatter(x=lstm_df.index, y=lstm_df['Predicted_Target'],
                         mode='lines',
                         name='LSTM Prediction',
                         line=dict(color='green')))  # 色を緑に設定して区別します

# 予測期間のTransformerモデルの予測をプロットします
fig.add_trace(go.Scatter(x=transformer_df.index, y=transformer_df['Predicted_Target'],
                         mode='lines',
                         name='Transformer Prediction',
                         line=dict(color='purple')))  # 色を紫に設定して区別します

# プロットのタイトルと軸ラベルを設定します
fig.update_layout(title='BBDC4 Predicts vs Actual',
                  xaxis_title='Date',
                  yaxis_title='Value')

# プロットを表示します
fig.show()

rmse_df = df['2023-01-01':]
merged_df = pd.merge(rmse_df[['Target']], arima_df[['Original_Scale_Predicted_Target']], left_index=True, right_index=True)

# 'Target'と'Original_Scale_Predicted_Target'列の間のRMSEを計算します
mse = mean_squared_error(merged_df['Target'], merged_df['Original_Scale_Predicted_Target'])
rmse = np.sqrt(mse)

print(f'Root Mean Square Error (RMSE): {rmse}')
arima_df

rmse_df = df['2023-01-01':]
merged_df = pd.merge(rmse_df[['Target']], lstm_df[['Predicted_Target']], left_index=True, right_index=True)

# 'Target'と'Predicted_Target'列の間のRMSEを計算します
mse = mean_squared_error(merged_df['Target'], merged_df['Predicted_Target'])
rmse = np.sqrt(mse)

print(f'Root Mean Square Error (RMSE): {rmse}')
lstm_df

rmse_df = df['2023-01-01':]
merged_df = pd.merge(rmse_df[['Target']], transformer_df[['Predicted_Target']], left_index=True, right_index=True)

# 'Target'と'Predicted_Target'列の間のRMSEを計算します
mse = mean_squared_error(merged_df['Target'], merged_df['Predicted_Target'])
rmse = np.sqrt(mse)

print(f'Root Mean Square Error (RMSE): {rmse}')
transformer_df
