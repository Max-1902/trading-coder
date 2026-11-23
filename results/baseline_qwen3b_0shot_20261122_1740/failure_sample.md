# Failure sample: baseline_qwen3b_0shot_20261122_1740

30 of 83 failures (seed 0).

## bollinger-02 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 121 bars, first at 78 (ref 1, got 0)

<details><summary>rules</summary>

- middle = SMA(35); upper = middle + 2 * PSTD(35); lower = middle - 2 * PSTD(35).
- When flat: go long when close < lower.
- When long: exit to flat when close >= middle.
- Never go short.
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
    # Calculate the middle, upper, and lower Bollinger Bands
    middle = df['close'].rolling(window=35).mean()
    upper = middle + 2 * df['close'].rolling(window=35).std()
    lower = middle - 2 * df['close'].rolling(window=35).std()
    
    # Initialize the signals series
    signals = pd.Series(0, index=df.index)
    
    # Determine the signals based on the conditions
    for i in range(len(df)):
        if i < 34:
            # Warm-up period, stay flat
            signals[i] = 0
        elif df['close'].iloc[i] < lower.iloc[i]:
            # Go long when close is below the lower band
            signals[i] = 1
        elif df['close'].iloc[i] >= middle.iloc[i]:
            # Exit to flat when close is above the middle band
            signals[i] = 0
    
    return signals
```

## bollinger-06 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 153 bars, first at 44 (ref 1, got 0)

<details><summary>rules</summary>

- middle = SMA(25); upper = middle + 2.25 * PSTD(25); lower = middle - 2.25 * PSTD(25).
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
    # Calculate SMA and PSTD for the last 25 bars
    df['middle'] = df['close'].rolling(window=25).mean()
    df['upper'] = df['middle'] + 2.25 *df['close'].rolling(window=25).std()
    df['lower'] = df['middle'] - 2.25*df['close'].rolling(window=25).std()
    
    # Initialize the signals series
    signals = pd.Series(0, index=df.index)
    
    # Determine the signals based on the conditions
    for i in range(25, len(df)):
        if df['close'].iloc[i] > df['upper'].iloc[i]:
            signals.iloc[i] = 1
        elif df['close'].iloc[i] < df['lower'].iloc[i]:
            signals.iloc[i] = -1
    
    return signals
```

## bollinger-07 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 40 bars, first at 30 (ref 1, got 0)

<details><summary>rules</summary>

- middle = SMA(15); upper = middle + 2 * PSTD(15); lower = middle - 2 * PSTD(15).
- When flat: go long when close > upper.
- When long: exit to flat when close <= middle.
- Never go short.
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
    # Calculate SMA and PSTD for the last 15 bars
    df['middle'] = df['close'].rolling(window=15).mean()
    df['upper'] = df['middle'] + 2 * df['close'].rolling(window=15).std()
    df['lower'] = df['middle'] - 2 * df['close'].rolling(window=15).std()
    
    # Initialize the signals series
    signals = pd.Series(0, index=df.index)
    
    # Determine the signals based on the conditions
    for i in range(len(df)):
        if i < 14:
            # Warm-up period, stay flat
            signals[i] = 0
        elif df['close'].iloc[i] > df['upper'].iloc[i]:
            # Go long when close > upper
            signals[i] = 1
        elif df['close'].iloc[i] <= df['middle'].iloc[i]:
            # Exit to flat when close <= middle
            signals[i] = 0
    
    return signals
```

## bollinger-09 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 162 bars, first at 27 (ref -1, got 0)

<details><summary>rules</summary>

- middle = SMA(12); upper = middle + 1.75 * PSTD(12); lower = middle - 1.75 * PSTD(12).
- When flat: go long when close < lower. Go short when close > upper.
- When long: exit to flat when close >= middle.
- When short: exit to flat when close <= middle.
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
    
    # Calculate the SMA and PSTD for the last 12 bars
    sma_12 = df['close'].rolling(window=12).mean()
    pstd_12 = df['close'].rolling(window=12).std()
    
    # Calculate the upper and lower bands
    upper_band = sma_12 + 1.75 * pstd_12
    lower_band = sma_12 - 1.75 * pstd_12
    
    # Determine the position for each bar
    for i in range(len(df)):
        if i < 11:
            # Warm-up period, stay flat
            continue
        if df['close'].iloc[i] < lower_band.iloc[i]:
            positions.iloc[i] = 1  # Long
        elif df['close'].iloc[i] > upper_band.iloc[i]:
            positions.iloc[i] = -1  # Short
        elif df['close'].iloc[i] >= sma_12.iloc[i]:
            positions.iloc[i] = 0  # Exit long
        elif df['close'].iloc[i] <= sma_12.iloc[i]:
            positions.iloc[i] = 0  # Exit short
    
    return positions
```

## bollinger-10 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 220 bars, first at 44 (ref -1, got 0)

<details><summary>rules</summary>

- middle = SMA(40); upper = middle + 2.5 * PSTD(40); lower = middle - 2.5 * PSTD(40).
- When flat: go long when close < lower. Go short when close > upper.
- When long: exit to flat when close >= middle.
- When short: exit to flat when close <= middle.
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
    # Calculate the middle, upper, and lower Bollinger Bands
    middle = df['close'].rolling(window=40).mean()
    std_dev = df['close'].rolling(window=40).std()
    upper = middle + 2.5 * std_dev
    lower = middle - 2.5 * std_dev

    # Initialize the signals series
    signals = pd.Series(0, index=df.index)

    # Determine the signals based on the conditions
    for i in range(40, len(df)):
        if signals[i-1] == 0:
            if df['close'].iloc[i] < lower.iloc[i]:
                signals.iloc[i] = 1
            elif df['close'].iloc[i] > upper.iloc[i]:
                signals.iloc[i] = -1
        elif signals[i-1] == 1:
            if df['close'].iloc[i] >= middle.iloc[i]:
                signals.iloc[i] = 0
        elif signals[i-1] == -1:
            if df['close'].iloc[i] <= middle.iloc[i]:
                signals.iloc[i] = 0

    return signals
```

## bollinger-12 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 232 bars, first at 43 (ref -1, got 0)

<details><summary>rules</summary>

- middle = SMA(40); upper = middle + 2 * PSTD(40); lower = middle - 2 * PSTD(40).
- When flat: go long when close < lower. Go short when close > upper.
- When long: exit to flat when close >= middle.
- When short: exit to flat when close <= middle.
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
    # Calculate the middle, upper, and lower Bollinger Bands
    window = 40
    middle = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    
    # Initialize the signals series
    signals = pd.Series(0, index=df.index)
    
    # Warm-up period
    warm_up = window - 1
    for i in range(warm_up):
        signals[i] = 0
    
    # Main trading logic
    for i in range(warm_up, len(df)):
        if signals[i-1] == 0:
            if df['close'].iloc[i] < lower.iloc[i]:
                signals[i] = 1
            elif df['close'].iloc[i] > upper.iloc[i]:
                signals[i] = -1
        elif signals[i-1] == 1:
            if df['close'].iloc[i] >= middle.iloc[i]:
                signals[i] = 0
        elif signals[i-1] == -1:
            if df['close'].iloc[i] <= middle.iloc[i]:
                signals[i] = 0
    
    return signals
```

## breakout-01 (difficulty 2): `exception`

random_walk_101: KeyError: 12 (line 14)

<details><summary>rules</summary>

- Long (1) when close > prior 12-bar highest close.
- Flat (0) when close < prior 12-bar lowest close. Never go short.
- Otherwise keep the previous bar's position.
- Flat during warm-up (bars 0 to 11, while the 12-bar channel is undefined); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Calculate the 12-bar highest and lowest closes
    df['12_bar_high'] = df['close'].rolling(window=12).max()
    df['12_bar_low'] = df['close'].rolling(window=12).min()
    
    # Generate signals based on the conditions
    for i in range(12, len(df)):
        if df.at[i, 'close'] > df.at[i-1, '12_bar_high']:
            signals.at[i] = 1
        elif df.at[i, 'close'] < df.at[i-1, '12_bar_low']:
            signals.at[i] = -1
    
    return signals
```

## breakout-06 (difficulty 2): `wrong_signals`

8/9 series differ; e.g. random_walk_101: 34 bars, first at 53 (ref 1, got 0)

<details><summary>rules</summary>

- When flat: go long when close > prior 45-bar high.
- When long: exit to flat when close < prior 7-bar low.
- Never go short.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat during warm-up (bars 0 to 44, while the 45-bar channel is undefined); the strategy starts flat.

Definitions:
- Prior N-bar high: the highest `high` over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar low is the same with the lowest `low`.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Warm-up period
    for i in range(45):
        positions[i] = 0
    
    # Calculate the 45-bar high and 7-bar low for each bar
    df['45_bar_high'] = df['high'].rolling(window=45).max()
    df['7_bar_low'] = df['low'].rolling(window=7).min()
    
    # Apply the trading strategy
    for i in range(45, len(df)):
        if positions[i-1] == 0:
            if df['close'][i] > df['45_bar_high'][i-1]:
                positions[i] = 1
        elif positions[i-1] == 1:
            if df['close'][i] < df['7_bar_low'][i-1]:
                positions[i] = 0
    
    return positions
```

## breakout-07 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 126 bars, first at 43 (ref 1, got 0)

<details><summary>rules</summary>

- When flat: go long when close > prior 30-bar highest close.
- When long: exit to flat when close < prior 20-bar lowest close.
- Never go short.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat during warm-up (bars 0 to 29, while the 30-bar channel is undefined); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Calculate the 30-bar highest and lowest closes
    df['30_bar_high'] = df['close'].rolling(window=30).max()
    df['30_bar_low'] = df['close'].rolling(window=30).min()
    df['20_bar_low'] = df['close'].rolling(window=20).min()
    
    # Apply the strategy rules
    for i in range(30, len(df)):
        if signals[i-1] == 0:
            if df['close'].iloc[i] > df['30_bar_high'].iloc[i-1]:
                signals[i] = 1
        elif signals[i-1] == 1:
            if df['close'].iloc[i] < df['20_bar_low'].iloc[i-1]:
                signals[i] = 0
    
    return signals
```

## combo-03 (difficulty 3): `exception`

random_walk_101: AttributeError: 'numpy.ndarray' object has no attribute 'rolling' (line 16)

<details><summary>rules</summary>

- When flat: go long when close > SMA(120) and RSI(4) < 40. Go short when close < SMA(120) and RSI(4) > 60.
- When long: exit to flat when RSI(4) > 55 or close < SMA(120).
- When short: exit to flat when RSI(4) < 45 or close > SMA(120).
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while SMA(120) or RSI(4) is undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd
import math

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    position = pd.Series(0, index=df.index)
    
    # Calculate SMA(120)
    sma_120 = df['close'].rolling(window=120).mean()
    
    # Calculate RSI(4)
    def calculate_rsi(series, window=4):
        gains = np.where(series.diff() > 0, series.diff(), 0)
        losses = np.where(series.diff() < 0, -series.diff(), 0)
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi
    
    rsi_4 = calculate_rsi(df['close'], window=4)
    
    # Initialize warm-up flag
    warm_up = True
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        if warm_up:
            # Check if SMA(120) and RSI(4) are defined
            if i >= 119 and i >= 3:
                warm_up = False
                position[i] = 0
            continue
        
        # Long condition
        if position[i-1] == 0 and df['close'][i] > sma_120[i] and rsi_4[i] < 40:
            position[i] = 1
        
        # Short condition
        elif position[i-1] == 0 and df['close'][i] < sma_120[i] and rsi_4[i] > 60:
            position[i] = -1
        
        # Exit conditions
        elif position[i-1] == 1 and (rsi_4[i] > 55 or df['close'][i] < sma_120[i]):
            position[i] = 0
        
        elif position[i-1] == -1 and (rsi_4[i] < 45 or df['close'][i] > sma_120[i]):
            position[i] = 0
        
    
    return position
```

## combo-05 (difficulty 3): `exception`

random_walk_101: TypeError: Addition/subtraction of integers and integer-arrays with Timestamp is no longer supported.  Instead of adding/subtracting `n`, use `n * obj.freq` (line 13)

<details><summary>rules</summary>

- When flat: go long when close > SMA(60) and RSI(7) < 40. Go short when close < SMA(60) and RSI(7) > 60.
- When long: exit to flat when RSI(7) > 55 or close < SMA(60).
- When short: exit to flat when RSI(7) < 45 or close > SMA(60).
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while SMA(60) or RSI(7) is undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    position = pd.Series(0, index=df.index)
    
    # Calculate SMA(60) and RSI(7)
    sma_60 = df['close'].rolling(window=60).mean()
    rsi_7 = df['close'].diff().apply(lambda x: 100 - 100 / (1 + (x + 100) / (df['close'].diff().abs() + 100)))
    
    # Warm-up period
    warm_up = df.index[0] + 60
    position[warm_up:] = 0
    
    # Logic for determining positions
    for i in range(warm_up, len(df)):
        if position[i-1] == 0:
            if df['close'][i] > sma_60[i] and rsi_7[i] < 40:
                position[i] = 1
            elif df['close'][i] < sma_60[i] and rsi_7[i] > 60:
                position[i] = -1
        else:
            if position[i-1] == 1:
                if rsi_7[i] > 55 or df['close'][i] < sma_60[i]:
                    position[i] = 0
            elif position[i-1] == -1:
                if rsi_7[i] < 45 or df['close'][i] > sma_60[i]:
                    position[i] = 0
    
    return position
```

## combo-06 (difficulty 2): `wrong_signals`

4/9 series differ; e.g. random_walk_101: 1 bars, first at 59 (ref 1, got 0)

<details><summary>rules</summary>

- Long (1) when close > SMA(60) and SMA(6) > SMA(40).
- Never go short.
- Flat (0) in every other case.
- Flat during warm-up, until SMA(60), SMA(6) and SMA(40) are all defined.
- Each bar is decided on its own; there is no memory of earlier positions.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate SMA(60), SMA(6), and SMA(40)
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    df['SMA_6'] = df['close'].rolling(window=6).mean()
    df['SMA_40'] = df['close'].rolling(window=40).mean()
    
    # Initialize the signals series with 0
    signals = pd.Series(0, index=df.index)
    
    # Determine the signals based on the conditions
    for i in range(len(df)):
        if i < 60 or i < 6 or i < 40:
            continue  # Warm-up period
        if df['close'].iloc[i] > df['SMA_60'].iloc[i] and df['SMA_6'].iloc[i] > df['SMA_40'].iloc[i]:
            signals.iloc[i] = 1  # Long signal
    
    return signals
```

## holding-02 (difficulty 3): `exception`

random_walk_101: KeyError: 'gain' (line 9)

<details><summary>rules</summary>

- Entry signal at bar t: RSI(5) > 80.
- When flat and the entry signal is true at bar t, the position is -1 on bars t to t+19 (20 bars including the entry bar) and 0 on bar t+20.
- Entry signals are ignored while the position is open and on the exit bar t+20; the earliest new entry is bar t+21.
- If the data ends before the holding period is over, keep the position until the last bar.
- Never go long.
- No entries while the signal is undefined (warm-up); the strategy starts flat.

Definitions:
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    position = pd.Series(0, index=df.index)
    
    # Calculate the RSI(5) for each bar
    df['RSI'] = (100 - 100 / (1 + df['gain'].rolling(window=5).mean() / df['loss'].rolling(window=5).mean())).fillna(0)
    
    # Entry signal: RSI(5) > 80
    entry_signal = df['RSI'] > 80
    
    # Initialize the entry flag
    entry_flag = False
    
    # Iterate over the DataFrame to determine the position
    for i in range(len(df)):
        if not entry_flag and entry_signal[i]:
            position[i] = -1
            entry_flag = True
        elif entry_flag and i == len(df) - 1:
            position[i] = 0
        elif entry_flag and i == len(df) - 20:
            position[i] = 0
            entry_flag = False
    
    return position
```

## holding-04 (difficulty 3): `exception`

random_walk_101: KeyError: 11 (line 21)

<details><summary>rules</summary>

- Entry signal at bar t: close < prior 12-bar lowest close.
- When flat and the entry signal is true at bar t, the position is 1 on bars t to t+5 (6 bars including the entry bar) and 0 on bar t+6.
- Entry signals are ignored while the position is open and on the exit bar t+6; the earliest new entry is bar t+7.
- If the data ends before the holding period is over, keep the position until the last bar.
- Never go short.
- No entries while the signal is undefined (warm-up); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the target positions as flat
    signals = pd.Series(0, index=df.index)
    
    # Calculate the prior 12-bar lowest close
    prior_12_bar_low = df['close'].rolling(window=12).min()
    
    # Initialize the entry flag
    entry_flag = False
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        if i < 11:
            # Warm-up period
            continue
        
        # Check for new 12-bar low
        if df.at[i, 'close'] < prior_12_bar_low.at[i-1]:
            entry_flag = True
        
        # Apply the entry signal
        if entry_flag and signals.at[i-1] == 0:
            signals.at[i] = 1
            entry_flag = False
        
        # Apply the exit signal
        if signals.at[i] == 1 and i >= 11:
            signals.at[i+5] = 0
    
    return signals
```

## holding-05 (difficulty 3): `lookahead`

47/94 cuts changed past positions; e.g. random_walk_101 cut k=84 (spike) changed bar 84

<details><summary>rules</summary>

- Entry signal at bar t: the one-bar return is below -0.035.
- When flat and the entry signal is true at bar t, the position is 1 on bars t to t+2 (3 bars including the entry bar) and 0 on bar t+3.
- Entry signals are ignored while the position is open and on the exit bar t+3; the earliest new entry is bar t+4.
- If the data ends before the holding period is over, keep the position until the last bar.
- Never go short.
- No entries while the signal is undefined (warm-up); the strategy starts flat.

Definitions:
- One-bar return: r[t] = close[t] / close[t-1] - 1. Undefined at bar 0.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the target positions as flat
    signals = pd.Series(0, index=df.index)
    
    # Calculate one-bar returns
    one_bar_returns = df['close'].pct_change()
    
    # Initialize the entry flag
    entry_flag = False
    
    # Iterate over the DataFrame
    for i in range(1, len(df)):
        # Check for entry signal
        if not entry_flag and one_bar_returns[i] < -0.035:
            entry_flag = True
            signals[i] = 1
            # Set the entry bar as the first bar of the holding period
            signals[i-1] = 1
        
        # Check for exit signal
        if entry_flag and i >= 3:
            signals[i] = 0
            entry_flag = False
    
    return signals
```

## holding-08 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 275 bars, first at 7 (ref -1, got 0)

<details><summary>rules</summary>

- Entry signal at bar t: the one-bar return is above 0.02.
- When flat and the entry signal is true at bar t, the position is -1 on bars t to t+19 (20 bars including the entry bar) and 0 on bar t+20.
- Entry signals are ignored while the position is open and on the exit bar t+20; the earliest new entry is bar t+21.
- If the data ends before the holding period is over, keep the position until the last bar.
- Never go long.
- No entries while the signal is undefined (warm-up); the strategy starts flat.

Definitions:
- One-bar return: r[t] = close[t] / close[t-1] - 1. Undefined at bar 0.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the target positions as flat
    signals = pd.Series(0, index=df.index)
    
    # Calculate one-bar returns
    one_bar_returns = df['close'].pct_change()
    
    # Initialize the entry flag
    entry_flag = False
    
    # Iterate over the DataFrame
    for i in range(1, len(df)):
        # Check if the one-bar return is above 0.02
        if one_bar_returns[i] > 0.02:
            # Set the entry flag and mark the entry position
            entry_flag = True
            signals[i] = -1
            # Mark the exit position
            signals[i+1:i+20] = 0
        # If the entry flag is set and the position is open, mark the exit position
        elif entry_flag and signals[i] == -1:
            signals[i+1:i+20] = 0
            # Reset the entry flag
            entry_flag = False
    
    return signals
```

## oscillator-02 (difficulty 2): `exception`

random_walk_101: NameError: name 'n' is not defined (line 24)

<details><summary>rules</summary>

- When flat: go long when RSI(16) < 28. Go short when RSI(16) > 72.
- When long: exit to flat when RSI(16) > 52.
- When short: exit to flat when RSI(16) < 52.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while RSI(16) is undefined (warm-up); the strategy starts flat.

Definitions:
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate the RSI(16) indicator
    def rsi(series, n=16):
        change = series.diff()
        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)
        avg_gain = gain.rolling(window=n).mean()
        avg_loss = loss.rolling(window=n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate the RSI(16) for the given DataFrame
    rsi_values = rsi(df['close'])
    
    # Apply the trading strategy
    for i in range(len(rsi_values)):
        if i < n - 1:
            # Warm-up period, stay flat
            positions[i] = 0
        elif rsi_values[i] < 28:
            # Go long when RSI(16) < 28
            positions[i] = 1
        elif rsi_values[i] > 72:
            # Go short when RSI(16) > 72
            positions[i] = -1
        elif rsi_values[i] > 52 and positions[i] == 1:
            # Exit long when RSI(16) > 52
            positions[i] = 0
        elif rsi_values[i] < 52 and positions[i] == -1:
            # Exit short when RSI(16) < 52
            positions[i] = 0
    
    return positions
```

## oscillator-03 (difficulty 2): `exception`

random_walk_101: NameError: name 'window' is not defined (line 23)

<details><summary>rules</summary>

- When flat: go long when RSI(6) < 28.
- When long: exit to flat when RSI(6) > 45.
- Never go short.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while RSI(6) is undefined (warm-up); the strategy starts flat.

Definitions:
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate the RSI(6) indicator
    def rsi(series, window=6):
        change = series.diff()
        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi
    
    # Calculate the RSI(6) for the given DataFrame
    rsi_values = rsi(df['close'])
    
    # Apply the trading strategy
    for i in range(len(rsi_values)):
        if i < window - 1:
            # Warm-up period, stay flat
            positions[i] = 0
        elif rsi_values[i] < 28:
            # Oversold, go long
            positions[i] = 1
        elif rsi_values[i] > 45:
            # Oversold, exit to flat
            positions[i] = 0
    
    return positions
```

## oscillator-04 (difficulty 2): `exception`

random_walk_101: AttributeError: 'Series' object has no attribute 'ewindow' (line 11)

<details><summary>rules</summary>

- When flat: go long when RSI(21) < 35. Go short when RSI(21) > 65.
- When long: exit to flat when RSI(21) > 52.
- When short: exit to flat when RSI(21) < 52.
- Otherwise keep the current position.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- Flat while RSI(21) is undefined (warm-up); the strategy starts flat.

Definitions:
- RSI(n): change[t] = close[t] - close[t-1] (from bar 1); gain = max(change, 0), loss = max(-change, 0). avg_gain starts at avg_gain[1] = gain[1] and then avg_gain[t] = avg_gain[t-1] + (gain[t] - avg_gain[t-1]) / n; avg_loss likewise. RSI = 100 - 100 / (1 + avg_gain / avg_loss), and RSI = 100 when avg_loss is 0. RSI is undefined for bars 0 to n-1 (first defined at bar n).

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate RSI(21)
    def calculate_rsi(series, window=21):
        change = series.diff()
        avg_gain = change[1:].ewindow(window).mean()
        avg_loss = change[1:].ewindow(window).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        return rsi
    
    # Calculate RSI(21) for the entire DataFrame
    rsi_21 = calculate_rsi(df['close'], window=21)
    
    # Apply the trading strategy
    for i in range(len(rsi_21)):
        if positions[i] == 0:
            if rsi_21[i] < 35:
                positions[i] = 1
            elif rsi_21[i] > 65:
                positions[i] = -1
        elif positions[i] == 1:
            if rsi_21[i] > 52:
                positions[i] = 0
        elif positions[i] == -1:
            if rsi_21[i] < 52:
                positions[i] = 0
    
    return positions
```

## stop_tp-00 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 148 bars, first at 40 (ref 1, got 0)

<details><summary>rules</summary>

- When flat: go long when close crosses above SMA(35).
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.07) (stop-loss) or close >= entry_price * (1 + 0.08) (take-profit).
- Never go short.
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- x crosses above y at bar t when x[t] > y[t] and x[t-1] <= y[t-1], with all four values defined. x crosses below y when x[t] < y[t] and x[t-1] >= y[t-1].

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate the Simple Moving Average (SMA) of the last 35 closes
    df['SMA_35'] = df['close'].rolling(window=35).mean()
    
    # Initialize the position series with zeros
    position = pd.Series(0, index=df.index)
    
    # Initialize the entry price series with NaN
    entry_price = pd.Series(np.nan, index=df.index)
    
    # Iterate over the DataFrame to generate signals
    for i in range(len(df)):
        if position[i] == 0:
            # Check if the close crosses above the SMA
            if df['close'].iloc[i] > df['SMA_35'].iloc[i]:
                # Set the entry price to the current close
                entry_price.iloc[i] = df['close'].iloc[i]
                # Set the position to long
                position.iloc[i] = 1
        else:
            # Check if the close crosses below the entry price * (1 - 0.07)
            if df['close'].iloc[i] <= entry_price.iloc[i] * (1 - 0.07):
                # Exit to flat
                position.iloc[i] = 0
            # Check if the close crosses above the entry price * (1 + 0.08)
            elif df['close'].iloc[i] >= entry_price.iloc[i] * (1 + 0.08):
                # Exit to flat
                position.iloc[i] = 0
    
    return position
```

## stop_tp-01 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 298 bars, first at 19 (ref 0, got -1)

<details><summary>rules</summary>

- When flat: go long when close crosses above SMA(20). Go short when close crosses below SMA(20).
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.04) (stop-loss) or close >= entry_price * (1 + 0.08) (take-profit).
- When short: exit to flat when close >= entry_price * (1 + 0.04) (stop-loss) or close <= entry_price * (1 - 0.08) (take-profit).
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- x crosses above y at bar t when x[t] > y[t] and x[t-1] <= y[t-1], with all four values defined. x crosses below y when x[t] < y[t] and x[t-1] >= y[t-1].

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate the Simple Moving Average (SMA) of the last 20 closes
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Initialize the entry price series with NaN
    entry_prices = pd.Series(np.nan, index=df.index)
    
    # Iterate over the DataFrame to generate signals
    for i in range(len(df)):
        if i < 19:
            # Warm-up period, no signals
            continue
        
        # Check for entry conditions
        if df['close'].iloc[i] > df['SMA_20'].iloc[i] and positions.iloc[i-1] == 0:
            positions.iloc[i] = 1
            entry_prices.iloc[i] = df['close'].iloc[i]
        elif df['close'].iloc[i] < df['SMA_20'].iloc[i] and positions.iloc[i-1] == 0:
            positions.iloc[i] = -1
            entry_prices.iloc[i] = df['close'].iloc[i]
        
        # Check for exit conditions
        if positions.iloc[i] == 1 and df['close'].iloc[i] <= entry_prices.iloc[i] * (1 - 0.04):
            positions.iloc[i] = 0
        elif positions.iloc[i] == 1 and df['close'].iloc[i] >= entry_prices.iloc[i] * (1 + 0.08):
            positions.iloc[i] = 0
        elif positions.iloc[i] == -1 and df['close'].iloc[i] >= entry_prices.iloc[i] * (1 + 0.04):
            positions.iloc[i] = 0
        elif positions.iloc[i] == -1 and df['close'].iloc[i] <= entry_prices.iloc[i] * (1 - 0.08):
            positions.iloc[i] = 0
    
    return positions
```

## stop_tp-03 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 101 bars, first at 49 (ref 0, got 1)

<details><summary>rules</summary>

- When flat: go long when close crosses above SMA(50).
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.025) (stop-loss) or close >= entry_price * (1 + 0.12) (take-profit).
- Never go short.
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- x crosses above y at bar t when x[t] > y[t] and x[t-1] <= y[t-1], with all four values defined. x crosses below y when x[t] < y[t] and x[t-1] >= y[t-1].

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate the Simple Moving Average (SMA) of the last 50 closes
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Initialize the entry price to NaN
    entry_price = np.nan
    
    # Iterate over the DataFrame to generate signals
    for i in range(len(df)):
        # Check if the indicator is defined
        if i >= 49:
            # Check if the close crosses above the SMA
            if df['close'].iloc[i] > df['SMA_50'].iloc[i]:
                # If flat, go long
                if signals.iloc[i-1] == 0:
                    entry_price = df['close'].iloc[i]
                    signals.iloc[i] = 1
            # Check if the long position is open
            elif signals.iloc[i-1] == 1:
                # Check for stop-loss
                if df['close'].iloc[i] <= entry_price * (1 - 0.025):
                    signals.iloc[i] = 0
                # Check for take-profit
                elif df['close'].iloc[i] >= entry_price * (1 + 0.12):
                    signals.iloc[i] = 0
    
    return signals
```

## stop_tp-04 (difficulty 3): `lookahead`

47/94 cuts changed past positions; e.g. random_walk_101 cut k=105 (spike) changed bar 11

<details><summary>rules</summary>

- When flat: go long when close > prior 12-bar highest close.
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.07) (stop-loss) or close >= entry_price * (1 + 0.08) (take-profit).
- Never go short.
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Initialize the entry price to NaN
    entry_price = np.nan
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        # Check if the indicator is defined (warm-up)
        if i >= 11:
            # Check if the current bar is flat
            if signals[i-1] == 0:
                # Check if the current bar meets the entry condition
                if df['close'].iloc[i] > df['close'].iloc[i-12].max():
                    # Set the entry price
                    entry_price = df['close'].iloc[i]
                    # Set the signal to long
                    signals[i] = 1
            # Check if the current bar is long and meets the exit conditions
            elif signals[i-1] == 1:
                # Check if the current bar meets the stop-loss condition
                if df['close'].iloc[i] <= entry_price * (1 - 0.07):
                    # Set the signal to flat
                    signals[i] = 0
                # Check if the current bar meets the take-profit condition
                elif df['close'].iloc[i] >= entry_price * (1 + 0.08):
                    # Set the signal to flat
                    signals[i] = 0
    
    return signals
```

## stop_tp-06 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 138 bars, first at 44 (ref 1, got 0)

<details><summary>rules</summary>

- When flat: go long when close > prior 35-bar highest close.
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.07) (stop-loss) or close >= entry_price * (1 + 0.05) (take-profit).
- Never go short.
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Initialize the highest close for the last 35 bars
    highest_close = df['close'].rolling(window=35).max().shift(1)
    
    # Initialize the entry price
    entry_price = None
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        # Check if the indicator is defined
        if i >= 34:
            # Check if the current bar is flat
            if signals[i] == 0:
                # Check if the entry condition is met
                if df['close'].iloc[i] > highest_close.iloc[i]:
                    # Set the entry price
                    entry_price = df['close'].iloc[i]
                    # Set the signal to long
                    signals[i] = 1
            else:
                # Check if the exit conditions are met
                if df['close'].iloc[i] <= entry_price * (1 - 0.07):
                    # Set the signal to flat
                    signals[i] = 0
                elif df['close'].iloc[i] >= entry_price * (1 + 0.05):
                    # Set the signal to flat
                    signals[i] = 0
        
        # Update the highest close for the last 35 bars
        highest_close.iloc[i] = df['close'].iloc[i-34:i].max()
    
    return signals
```

## stop_tp-07 (difficulty 3): `lookahead`

40/66 cuts changed past positions; e.g. random_walk_101 cut k=0 (spike) changed bar 0

<details><summary>rules</summary>

- When flat: go long when close > prior 35-bar highest close. Go short when close < prior 35-bar lowest close.
- The entry price is the close of the entry bar.
- When long: exit to flat when close <= entry_price * (1 - 0.04) (stop-loss) or close >= entry_price * (1 + 0.08) (take-profit).
- When short: exit to flat when close >= entry_price * (1 + 0.04) (stop-loss) or close <= entry_price * (1 - 0.08) (take-profit).
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- Prior N-bar highest close: the highest close over the N bars before the current bar (bars t-N to t-1, not including bar t). Undefined for bars 0 to N-1. Prior N-bar lowest close is the same with the lowest close.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Initialize the signals series with zeros
    signals = pd.Series(0, index=df.index)
    
    # Initialize the highest and lowest close for the last 35 bars
    highest_35 = df['close'].rolling(window=35).max()
    lowest_35 = df['close'].rolling(window=35).min()
    
    # Initialize the entry price
    entry_price = np.nan
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        if np.isnan(entry_price):
            # If no entry price, check for entry conditions
            if df['close'].iloc[i] > highest_35.iloc[i-1]:
                signals.iloc[i] = 1
                entry_price = df['close'].iloc[i]
            elif df['close'].iloc[i] < lowest_35.iloc[i-1]:
                signals.iloc[i] = -1
                entry_price = df['close'].iloc[i]
        
        else:
            # If a position is open, check for exit conditions
            if signals.iloc[i] == 1:
                if df['close'].iloc[i] <= entry_price * (1 - 0.04):
                    signals.iloc[i] = 0
                elif df['close'].iloc[i] >= entry_price * (1 + 0.08):
                    signals.iloc[i] = 0
            elif signals.iloc[i] == -1:
                if df['close'].iloc[i] >= entry_price * (1 + 0.04):
                    signals.iloc[i] = 0
                elif df['close'].iloc[i] <= entry_price * (1 - 0.08):
                    signals.iloc[i] = 0
    
    return signals
```

## trailing-01 (difficulty 3): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 282 bars, first at 24 (ref 0, got -1)

<details><summary>rules</summary>

- When flat: go long when close crosses above SMA(25). Go short when close crosses below SMA(25).
- While long, track peak = the highest close since entry, including the entry bar and the current bar. Exit to flat when close < peak * (1 - 0.07).
- While short, track trough = the lowest close since entry, including the entry bar and the current bar. Exit to flat when close > trough * (1 + 0.07).
- Entry signals are ignored while a position is open.
- The position changes at most once per bar: on a bar where a position is exited, no new position is opened.
- No entries while the indicators needed for the entry signal are undefined (warm-up); the strategy starts flat.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- x crosses above y at bar t when x[t] > y[t] and x[t-1] <= y[t-1], with all four values defined. x crosses below y when x[t] < y[t] and x[t-1] >= y[t-1].

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate SMA(25)
    df['SMA_25'] = df['close'].rolling(window=25).mean()
    
    # Initialize the position series
    position = pd.Series(0, index=df.index)
    
    # Initialize peak and trough for long and short positions
    peak = pd.Series(0, index=df.index)
    trough = pd.Series(0, index=df.index)
    
    # Initialize flags for entry and exit
    entry_flag = False
    exit_flag = False
    
    # Iterate over the DataFrame
    for i in range(len(df)):
        if i < 24:
            continue  # Skip warm-up period
        
        # Check for entry conditions
        if position[i] == 0:
            if df['close'][i] > df['SMA_25'][i]:
                position[i] = 1
                peak[i] = df['close'][i]
                entry_flag = True
            elif df['close'][i] < df['SMA_25'][i]:
                position[i] = -1
                trough[i] = df['close'][i]
                entry_flag = True
        
        # Check for exit conditions
        if position[i] == 1:
            if df['close'][i] < peak[i] * (1 - 0.07):
                position[i] = 0
                exit_flag = True
        elif position[i] == -1:
            if df['close'][i] > trough[i] * (1 + 0.07):
                position[i] = 0
                exit_flag = True
        
        # Reset entry flag if exit is triggered
        if exit_flag:
            entry_flag = False
    
    return position
```

## vol_filter-01 (difficulty 2): `exception`

random_walk_101: KeyError: 14 (line 21)

<details><summary>rules</summary>

- Base signal: 1 when close > SMA(15), otherwise 0.
- The position equals the base signal when VOL(12) < 0.02, and 0 when VOL(12) >= 0.02.
- Never go short.
- Flat during warm-up, until both the moving average(s) and VOL(12) are defined.
- Each bar is decided on its own; there is no memory of earlier positions.

Definitions:
- SMA(n): the mean of the last n closes, including the current bar. Undefined for the first n-1 bars.
- One-bar return: r[t] = close[t] / close[t-1] - 1. Undefined at bar 0.
- VOL(m): sample standard deviation (divide by m-1) of the last m one-bar returns, including the current bar. First defined at bar m.

</details>

```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # Calculate the moving average (SMA) of the last 15 closes
    df['SMA_15'] = df['close'].rolling(window=15).mean()
    
    # Calculate the one-bar returns
    df['one_bar_return'] = df['close'].pct_change()
    
    # Calculate the standard deviation of the last 12 one-bar returns
    df['VOL_12'] = df['one_bar_return'].rolling(window=12).std()
    
    # Initialize the signal series
    df['signal'] = 0
    
    # Apply the strategy rules
    for i in range(len(df)):
        if i < 14:  # Warm-up period
            df.at[i, 'signal'] = 0
        elif df.at[i, 'close'] > df.at[i, 'SMA_15'] and df.at[i, 'VOL_12'] < 0.02:
            df.at[i, 'signal'] = 1
        elif df.at[i, 'close'] <= df.at[i, 'SMA_15'] or df.at[i, 'VOL_12'] >= 0.02:
            df.at[i, 'signal'] = 0
    
    return df['signal']
```

## zscore-01 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 67 bars, first at 76 (ref 1, got 0)

<details><summary>rules</summary>

- Let m = SMA(12) and s = STD(12) of the close.
- When flat: go long when close < m - 2 * s.
- When long: exit to flat when close >= m - 0.25 * s.
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
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate SMA(12) and STD(12) for each bar
    m = df['close'].rolling(window=12).mean()
    s = df['close'].rolling(window=12).std()
    
    # Warm-up period: do nothing
    if s.isnull().any() or m.isnull().any():
        return positions
    
    # Calculate the threshold for going long
    threshold = m - 2 * s
    
    # Apply the strategy rules
    for i in range(len(df)):
        if positions[i] == 0:
            if df['close'].iloc[i] < threshold:
                positions[i] = 1
        elif positions[i] == 1:
            if df['close'].iloc[i] >= m - 0.25 * s:
                positions[i] = 0
    
    return positions
```

## zscore-02 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 186 bars, first at 11 (ref -1, got 0)

<details><summary>rules</summary>

- Let m = SMA(12) and s = STD(12) of the close.
- When flat: go long when close < m - 1.25 * s. Go short when close > m + 1.25 * s.
- When long: exit to flat when close >= m - 1 * s.
- When short: exit to flat when close <= m + 1 * s.
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
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate the 12-bar SMA and STD
    m = df['close'].rolling(window=12).mean()
    s = df['close'].rolling(window=12).std()
    
    # Warm-up period: do nothing
    if s.isnull().any() or m.isnull().any():
        return positions
    
    # Calculate the thresholds
    threshold_up = m - 1.25 * s
    threshold_down = m + 1.25 * s
    
    # Apply the rules
    for i in range(len(df)):
        if positions[i] == 0:
            if df['close'].iloc[i] < threshold_down:
                positions[i] = 1
            elif df['close'].iloc[i] > threshold_up:
                positions[i] = -1
        elif positions[i] == 1:
            if df['close'].iloc[i] >= threshold_down:
                positions[i] = 0
        elif positions[i] == -1:
            if df['close'].iloc[i] <= threshold_up:
                positions[i] = 0
    
    return positions
```

## zscore-07 (difficulty 2): `wrong_signals`

9/9 series differ; e.g. random_walk_101: 237 bars, first at 93 (ref 1, got 0)

<details><summary>rules</summary>

- Let m = SMA(60) and s = STD(60) of the close.
- When flat: go long when close < m - 2 * s. Go short when close > m + 2 * s.
- When long: exit to flat when close >= m - 0.25 * s.
- When short: exit to flat when close <= m + 0.25 * s.
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
    # Initialize the position series with zeros
    positions = pd.Series(0, index=df.index)
    
    # Calculate the 60-bar SMA and STD
    m = df['close'].rolling(window=60).mean()
    s = df['close'].rolling(window=60).std()
    
    # Warm-up period: do nothing
    if s.isnull().any() or m.isnull().any():
        return positions
    
    # Calculate the thresholds
    threshold_long = m - 2 * s
    threshold_short = m + 2 * s
    threshold_exit_long = m - 0.25 * s
    threshold_exit_short = m + 0.25 * s
    
    # Apply the trading rules
    for i in range(len(df)):
        if positions[i] == 0:
            if df['close'].iloc[i] < threshold_long:
                positions[i] = 1
            elif df['close'].iloc[i] > threshold_short:
                positions[i] = -1
        elif positions[i] == 1:
            if df['close'].iloc[i] >= threshold_exit_long:
                positions[i] = 0
        elif positions[i] == -1:
            if df['close'].iloc[i] <= threshold_exit_short:
                positions[i] = 0
    
    return positions
```
