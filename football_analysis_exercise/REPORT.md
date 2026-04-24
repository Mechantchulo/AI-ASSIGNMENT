# Artificial Intelligence - Exercise 1

Dataset: International Football Results (`results.csv`)

## Step 1: Load the CSV

```python
import pandas as pd

df = pd.read_csv("results.csv")
df.head()
```

## Basic Exploration

### 1. How many matches are in the dataset?

```python
df.shape[0]
```

**Answer:** `49,287`

### 2. What is the earliest and latest year in the data?

```python
df["date"] = pd.to_datetime(df["date"])
df["date"].min(), df["date"].max()
```

```python
df["date"].dt.year.min(), df["date"].dt.year.max()
```

**Answer:** Earliest year is `1872`, latest year is `2026`.

### 3. How many unique countries (teams) are there?

```python
unique_teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
len(unique_teams)
```

**Answer:** `333`

### 4. Which team appears most frequently as home team?

```python
df["home_team"].value_counts().head(1)
```

**Answer:** `Brazil` (appears `614` times as home team).

## Goals Analysis

Create total goals:

```python
df["total_goals"] = df["home_score"] + df["away_score"]
```

### 5. What is the average number of goals per match?

```python
df["total_goals"].mean()
```

**Answer:** `2.938` goals per match (approximately).

### 6. What is the highest scoring match?

```python
max_goals = df["total_goals"].max()
df[df["total_goals"] == max_goals][["date", "home_team", "away_team", "home_score", "away_score", "total_goals"]]
```

**Answer:** `Australia 31 - 0 American Samoa` on `2001-04-11` (total `31` goals).

### 7. Are more goals scored at home or away?

```python
df["home_score"].sum(), df["away_score"].sum()
```

**Answer:** More goals are scored by home teams (`86,426`) than away teams (`58,192`).

### 8. What is the most common total goals value?

```python
df["total_goals"].mode()
```

**Answer:** `2`

## Match Results

Create match outcome:

```python
def match_result(row):
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    else:
        return "Draw"

df["result"] = df.apply(match_result, axis=1)
```

### 9. What percentage of matches are home wins?

```python
home_win_pct = (df["result"].value_counts()["Home Win"] / len(df)) * 100
home_win_pct
```

**Answer:** `48.91%` (approximately).

### 10. Does home advantage exist?

```python
df["result"].value_counts(normalize=True) * 100
```

**Answer:** Yes. Home wins (`48.91%`) are much higher than away wins (`28.23%`), indicating home advantage.

### 11. Which country/team has the most wins historically?

```python
winners = pd.Series(pd.NA, index=df.index, dtype="object")
winners[df["home_score"] > df["away_score"]] = df.loc[df["home_score"] > df["away_score"], "home_team"]
winners[df["away_score"] > df["home_score"]] = df.loc[df["away_score"] > df["home_score"], "away_team"]

winners.value_counts().head(10)
```

**Answer:** `Brazil` with `670` wins.

## Visualization

### Histogram of goals

```python
import matplotlib.pyplot as plt

df["total_goals"].hist(bins=15)
plt.title("Distribution of Goals Per Match")
plt.xlabel("Total Goals")
plt.ylabel("Number of Matches")
plt.show()
```

Saved figure: `plots/goals_histogram.png`

### Bar chart of match outcomes

```python
df["result"].value_counts().reindex(["Home Win", "Away Win", "Draw"]).plot(kind="bar")
plt.title("Match Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Number of Matches")
plt.show()
```

Saved figure: `plots/match_outcomes_bar.png`

### Top 10 teams by total wins

```python
winners.value_counts().head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Teams by Total Wins")
plt.xlabel("Wins")
plt.ylabel("Team")
plt.show()
```

Saved figure: `plots/top10_total_wins.png`

## Notes

- This report was computed from the current `results.csv` file in this folder.
- To regenerate results and plots, run:

```bash
python3 analysis.py
```
