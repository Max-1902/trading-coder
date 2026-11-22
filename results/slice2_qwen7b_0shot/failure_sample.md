# Failure sample: slice2_qwen7b_0shot

5 of 7 failures (seed 0).

## bollinger-00 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 92 bars, first at 78 (ref -1, got 0)

<details><summary>rules</summary>

- middle = SMA(18); upper = middle + 2.5 * PSTD(18); lower = middle - 2.5 * PSTD(18).
- When flat: go long when close > upper. Go short when close < lower.
- When long: exit to flat when close <= middle.
- When short: exit to flat when close >= middle.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat during warm-up (while the bands are undefined); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- PSTD(n): population standard deviation (divide by n) of the last n closes, including the current bar. Undefined for the first n-1 bars.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate the 18-bar SMA for the middle band
    middle_band = df['close'].rolling(window=18).mean()
    
    # Calculate the 18-bar population standard deviation for the upper and lower bands
    std_dev = df['close'].rolling(window=18).std(ddof=0)
    upper_band = middle_band + 2.5 * std_dev
    lower_band = middle_band - 2.5 * std_dev
    
    # Iterate over the DataFrame to determine positions
    for i in range(18, len(df)):
        if positions[i-1] == 0:  # Flat
            if df['close'][i] > upper_band[i]:
                positions[i] = 1  # Go long
            elif df['close'][i] < lower_band[i]:
                positions[i] = -1  # Go short
        elif positions[i-1] == 1:  # Long
            if df['close'][i] <= middle_band[i]:
                positions[i] = 0  # Exit to flat
        elif positions[i-1] == -1:  # Short
            if df['close'][i] >= middle_band[i]:
                positions[i] = 0  # Exit to flat
    
    return positions
```

## holding-00 (difficulty 3): `exception`

trend_201: IndexError: iloc cannot enlarge its target object (line 23)

<details><summary>rules</summary>

- Entry signal at bar t: RSI(12) < 20.
- When flat and the entry signal is true at bar t, the position is 1 on bars t to t+9 (10 bars including the entry bar) and 0 on bar t+10.
- Entry signals are ignored while the position is open and on the exit bar t+10; the earliest new entry is bar t+11.
- If the data ends before the holding period is over, keep the position until the last bar.
- Never go short.
- No entries while the signal is undefined (warm-up); the strategy starts flat.

Definitions:
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate RSI(12)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Initialize the position series with 0s
    positions = pd.Series(0, index=df.index)
    
    # Warm-up period for RSI
    warm_up_period = 12
    positions.iloc[:warm_up_period] = 0
    
    # Generate signals
    for i in range(warm_up_period, len(df)):
        if rsi.iloc[i] < 20:
            positions.iloc[i:i+10] = 1
            positions.iloc[i+10] = 0
    
    return positions
```

## ma_cross-00 (difficulty 1): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 88 bars, first at 1 (ref 0, got -1)

<details><summary>rules</summary>

- Long (1) when EMA(4) > EMA(90).
- Short (-1) when EMA(4) < EMA(90).
- Flat (0) when EMA(4) equals EMA(90).
- Flat during warm-up, until both EMA(4) and EMA(90) are defined.
- Each bar is decided on its own; there is no memory of earlier positions.

Definitions:
- EMA(n): exponential moving average of close with alpha = 2/(n+1), computed recursively as ema[t] = alpha*close[t] + (1-alpha)*ema[t-1], starting from ema[0] = close[0]. Treated as undefined for the first n-1 bars.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    ema4 = df['close'].ewm(span=4, adjust=False).mean()
    ema90 = df['close'].ewm(span=90, adjust=False).mean()
    
    signals = pd.Series(0, index=df.index)
    signals[ema4.isnull()] = 0
    signals[ema4 > ema90] = 1
    signals[ema4 < ema90] = -1
    
    return signals
```

## oscillator-00 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 113 bars, first at 17 (ref 1, got 0)

<details><summary>rules</summary>

- When flat: go long when %K(12) < 12.
- When long: exit to flat when %K(12) > 45.
- Never go short.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while %K(12) is undefined (warm-up); the strategy starts flat.

Definitions:
- Stochastic %K(n): 100 * (close - LL) / (HH - LL), where HH is the highest high and LL the lowest low of the last n bars including the current bar; %K = 50 when HH equals LL. Undefined for the first n-1 bars.

</details>

```python
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate the Stochastic %K(12)
    n = 12
    df['HH'] = df['high'].rolling(window=n).max()
    df['LL'] = df['low'].rolling(window=n).min()
    df['%K'] = 100 * (df['close'] - df['LL']) / (df['HH'] - df['LL'])
    
    # Initialize the position series with 0
    positions = pd.Series(0, index=df.index)
    
    # Warm-up period
    warm_up_period = n - 1
    positions.iloc[:warm_up_period] = 0
    
    # Generate signals
    for i in range(warm_up_period, len(df)):
        if positions.iloc[i-1] == 0:  # Flat
            if df['%K'].iloc[i] < 12:
                positions.iloc[i] = 1  # Go long
        elif positions.iloc[i-1] == 1:  # Long
            if df['%K'].iloc[i] > 45:
                positions.iloc[i] = 0  # Exit to flat
    
    return positions
```

## zscore-00 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 32 bars, first at 77 (ref 1, got 0)

<details><summary>rules</summary>

- Let m = SMA(20) and s = STD(20) of the close.
- When flat: go long when close < m - 2 * s.
- When long: exit to flat when close >= m - 1 * s.
- Never go short.
- Use the comparisons exactly as written (no division by s), so that s = 0 needs no special case.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat during warm-up (while m and s are undefined); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- STD(n): sample standard deviation (divide by n-1) of the last n closes, including the current bar. Undefined for the first n-1 bars.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    close = df['close']
    n = 20
    mean = close.rolling(window=n).mean()
    std = close.rolling(window=n).std(ddof=1)
    
    position = pd.Series(0, index=df.index)
    
    for i in range(n, len(df)):
        if position[i-1] == 0:
            if close[i] < mean[i] - 2 * std[i]:
                position[i] = 1
        elif position[i-1] == 1:
            if close[i] >= mean[i] - 1 * std[i]:
                position[i] = 0
    
    return position
```
