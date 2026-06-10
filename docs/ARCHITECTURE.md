# Architecture Notes

```text
news source + price source
          |
          v
dataset builder: ticker/date alignment
          |
          v
text preprocessing + feature engineering
          |
          v
model training + evaluation
          |
          v
prediction scores
          |
          v
signal generation
          |
          v
backtest engine
          |
          v
reports + Streamlit dashboard
```

## Key design decision

The rebuild should predict **future** movement from currently available news. The original project used same-day open/close labels, which is easier to implement but weaker as a trading-research setup.
