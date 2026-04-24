import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load data
# If your results.csv is in another location, update this path.
df = pd.read_csv("results.csv")

df["date"] = pd.to_datetime(df["date"])

# Basic exploration
num_matches = df.shape[0]
earliest_date = df["date"].min()
latest_date = df["date"].max()
unique_teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True)).size
most_common_home_team = df["home_team"].value_counts().head(1)

# Goals analysis
df["total_goals"] = df["home_score"] + df["away_score"]

avg_goals = df["total_goals"].mean()
max_total_goals = df["total_goals"].max()
highest_scoring_match = df.loc[
    df["total_goals"] == max_total_goals,
    ["date", "home_team", "away_team", "home_score", "away_score", "total_goals"],
]

home_goals_total = df["home_score"].sum()
away_goals_total = df["away_score"].sum()
most_common_total_goals = df["total_goals"].mode().iloc[0]

# Match outcomes

def match_result(row):
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    else:
        return "Draw"


df["result"] = df.apply(match_result, axis=1)

result_counts = df["result"].value_counts()
home_win_percentage = result_counts["Home Win"] / num_matches * 100

winners = pd.Series(pd.NA, index=df.index, dtype="object")
winners[df["home_score"] > df["away_score"]] = df.loc[df["home_score"] > df["away_score"], "home_team"]
winners[df["away_score"] > df["home_score"]] = df.loc[df["away_score"] > df["home_score"], "away_team"]

team_total_wins = winners.dropna().value_counts()
most_historical_wins_team = team_total_wins.index[0]

# Print concise results
print("1) Matches in dataset:", num_matches)
print("2) Earliest date:", earliest_date.date(), "| Latest date:", latest_date.date())
print("3) Unique teams:", unique_teams)
print("4) Most frequent home team:", most_common_home_team.index[0], "(", int(most_common_home_team.iloc[0]), "matches)")
print("5) Average goals per match:", round(avg_goals, 3))
print("6) Highest scoring match:")
print(highest_scoring_match.to_string(index=False))
print("7) Goals scored at home vs away:", int(home_goals_total), "vs", int(away_goals_total))
print("8) Most common total goals value:", int(most_common_total_goals))
print("9) Home win percentage:", round(home_win_percentage, 2), "%")
print("10) Home advantage exists:", "Yes" if result_counts["Home Win"] > result_counts["Away Win"] else "No")
print("11) Team with most historical wins:", most_historical_wins_team, "(", int(team_total_wins.iloc[0]), "wins)")

# Visualizations
plt.figure(figsize=(8, 5))
df["total_goals"].hist(bins=15, edgecolor="black")
plt.title("Distribution of Goals Per Match")
plt.xlabel("Total Goals")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.savefig("plots/goals_histogram.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
result_counts.reindex(["Home Win", "Away Win", "Draw"]).plot(kind="bar", color=["#4CAF50", "#F44336", "#2196F3"])
plt.title("Match Outcomes")
plt.xlabel("Outcome")
plt.ylabel("Number of Matches")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/match_outcomes_bar.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
team_total_wins.head(10).sort_values().plot(kind="barh", color="#FF9800")
plt.title("Top 10 Teams by Total Wins")
plt.xlabel("Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("plots/top10_total_wins.png", dpi=150)
plt.close()
