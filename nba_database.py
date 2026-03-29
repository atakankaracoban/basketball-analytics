"""
NBA Database Builder
Creates a local SQLite database from NBA API data
Then demonstrates SQL queries
"""

import sqlite3
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    playergamelog
)
from nba_api.stats.static import players as nba_players
import time

DB_PATH = "nba_analytics.db"

# ============================================================
# DATABASE SETUP
# ============================================================

def create_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables(conn):
    """Create all database tables."""
    cursor = conn.cursor()

    # Players table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id   INTEGER PRIMARY KEY,
            full_name   TEXT NOT NULL,
            is_active   INTEGER
        )
    """)

    # Season stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS season_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id   INTEGER,
            player_name TEXT,
            season      TEXT,
            team        TEXT,
            gp          INTEGER,
            min         REAL,
            pts         REAL,
            reb         REAL,
            ast         REAL,
            stl         REAL,
            blk         REAL,
            tov         REAL,
            fg_pct      REAL,
            fg3_pct     REAL,
            ft_pct      REAL,
            age         REAL,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        )
    """)

    # Teams table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            season      TEXT,
            team_name   TEXT,
            team_id     INTEGER,
            gp          INTEGER,
            wins        INTEGER,
            losses      INTEGER,
            win_pct     REAL,
            pts         REAL,
            reb         REAL,
            ast         REAL,
            stl         REAL,
            blk         REAL,
            tov         REAL,
            fg_pct      REAL,
            fg3_pct     REAL
        )
    """)

    conn.commit()
    print("Tables created successfully.")

def populate_players(conn):
    """Load all NBA players into the database."""
    cursor = conn.cursor()

    # Check if already populated
    cursor.execute("SELECT COUNT(*) FROM players")
    if cursor.fetchone()[0] > 0:
        print("Players table already populated.")
        return

    all_players = nba_players.get_players()
    for p in all_players:
        cursor.execute("""
            INSERT OR IGNORE INTO players
            (player_id, full_name, is_active)
            VALUES (?, ?, ?)
        """, (p["id"], p["full_name"], int(p["is_active"])))

    conn.commit()
    print(f"Loaded {len(all_players)} players into database.")

def populate_season_stats(conn, season):
    """Load a full season of player stats into the database."""
    cursor = conn.cursor()

    # Check if this season already exists
    cursor.execute(
        "SELECT COUNT(*) FROM season_stats WHERE season = ?",
        (season,)
    )
    if cursor.fetchone()[0] > 0:
        print(f"Season {season} already in database.")
        return

    print(f"Fetching {season} stats from API...")
    time.sleep(0.5)

    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame"
    ).get_data_frames()[0]

    # Calculate percentages if not present
    if "FG_PCT" not in df.columns:
        df["FG_PCT"] = df["FGM"] / df["FGA"]
    if "FG3_PCT" not in df.columns:
        df["FG3_PCT"] = df["FG3M"] / df["FG3A"]
    if "FT_PCT" not in df.columns:
        df["FT_PCT"] = df["FTM"] / df["FTA"]

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO season_stats
            (player_id, player_name, season, team, gp, min,
             pts, reb, ast, stl, blk, tov,
             fg_pct, fg3_pct, ft_pct, age)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row["PLAYER_ID"]),
            row["PLAYER_NAME"],
            season,
            row.get("TEAM_ABBREVIATION", ""),
            int(row["GP"]),
            float(row["MIN"]),
            float(row["PTS"]),
            float(row["REB"]),
            float(row["AST"]),
            float(row["STL"]),
            float(row["BLK"]),
            float(row["TOV"]),
            float(row.get("FG_PCT", 0) or 0),
            float(row.get("FG3_PCT", 0) or 0),
            float(row.get("FT_PCT", 0) or 0),
            float(row.get("AGE", 0) or 0)
        ))

    conn.commit()
    print(f"Loaded {len(df)} player records for {season}.")

def populate_teams(conn, season):
    """Load team stats into database."""
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM teams WHERE season = ?", (season,)
    )
    if cursor.fetchone()[0] > 0:
        print(f"Team stats for {season} already in database.")
        return

    print(f"Fetching {season} team stats...")
    time.sleep(0.5)

    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense="Base"
    ).get_data_frames()[0]

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO teams
            (season, team_name, team_id, gp, wins, losses,
             win_pct, pts, reb, ast, stl, blk, tov,
             fg_pct, fg3_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            season,
            row["TEAM_NAME"],
            int(row["TEAM_ID"]),
            int(row["GP"]),
            int(row["W"]),
            int(row["L"]),
            float(row["W_PCT"]),
            float(row["PTS"] / row["GP"]),
            float(row["REB"] / row["GP"]),
            float(row["AST"] / row["GP"]),
            float(row["STL"] / row["GP"]),
            float(row["BLK"] / row["GP"]),
            float(row["TOV"] / row["GP"]),
            float(row["FGM"] / row["FGA"]),
            float(row["FG3M"] / row["FG3A"])
        ))

    conn.commit()
    print(f"Loaded team stats for {season}.")

# ============================================================
# SQL QUERIES — This is where you learn SQL
# ============================================================

def run_queries(conn):
    """
    Demonstrate SQL queries on our database.
    Each query teaches a new SQL concept.
    """

    print("\n" + "=" * 60)
    print("SQL QUERY DEMONSTRATIONS")
    print("=" * 60)

    # ── Query 1: Basic SELECT ───────────────────────────────
    # SELECT picks columns
    # FROM specifies the table
    # WHERE filters rows
    # ORDER BY sorts results
    # LIMIT restricts number of rows
    print("\n--- Query 1: Top 10 scorers this season ---")
    print("SQL: SELECT player_name, pts FROM season_stats")
    print("     WHERE season = '2025-26'")
    print("     ORDER BY pts DESC LIMIT 10\n")

    result = pd.read_sql("""
        SELECT player_name, pts, reb, ast, gp
        FROM season_stats
        WHERE season = '2025-26'
        ORDER BY pts DESC
        LIMIT 10
    """, conn)
    print(result.to_string(index=False))

    # ── Query 2: WHERE with multiple conditions ─────────────
    print("\n--- Query 2: Efficient high-volume scorers ---")
    print("SQL: SELECT ... WHERE pts > 20 AND fg_pct > 0.50\n")

    result = pd.read_sql("""
        SELECT player_name, pts, fg_pct, reb, ast
        FROM season_stats
        WHERE season = '2025-26'
          AND pts > 20
          AND fg_pct > 0.50
          AND gp >= 30
        ORDER BY pts DESC
    """, conn)
    print(result.to_string(index=False))

    # ── Query 3: GROUP BY and aggregate functions ───────────
    # GROUP BY collapses rows into groups
    # AVG(), MAX(), MIN(), SUM(), COUNT() work on groups
    print("\n--- Query 3: Average stats by age group ---")
    print("SQL: SELECT age, AVG(pts) GROUP BY age\n")

    result = pd.read_sql("""
        SELECT
            CAST(age AS INTEGER) as age_group,
            COUNT(*) as num_players,
            ROUND(AVG(pts), 1) as avg_pts,
            ROUND(AVG(reb), 1) as avg_reb,
            ROUND(AVG(ast), 1) as avg_ast
        FROM season_stats
        WHERE season = '2025-26'
          AND gp >= 20
        GROUP BY CAST(age AS INTEGER)
        ORDER BY age_group
    """, conn)
    print(result.to_string(index=False))

    # ── Query 4: JOIN — combining two tables ────────────────
    # JOIN connects rows from two tables based on a shared key
    # This is SQL's most powerful feature
    print("\n--- Query 4: JOIN players table with stats ---")
    print("SQL: SELECT ... FROM season_stats JOIN players ON ...\n")

    result = pd.read_sql("""
        SELECT
            p.full_name,
            s.season,
            s.pts,
            s.reb,
            s.ast,
            s.fg_pct
        FROM season_stats s
        JOIN players p ON s.player_id = p.player_id
        WHERE s.pts > 25
          AND s.season = '2025-26'
          AND s.gp >= 30
        ORDER BY s.pts DESC
    """, conn)
    print(result.to_string(index=False))

    # ── Query 5: Subquery ───────────────────────────────────
    # A query inside a query
    # Find players who score more than the league average
    print("\n--- Query 5: Players above league average scoring ---")
    print("SQL: WHERE pts > (SELECT AVG(pts) FROM ...)\n")

    result = pd.read_sql("""
        SELECT player_name, pts, reb, ast
        FROM season_stats
        WHERE season = '2025-26'
          AND gp >= 30
          AND pts > (
              SELECT AVG(pts)
              FROM season_stats
              WHERE season = '2025-26'
              AND gp >= 20
          )
        ORDER BY pts DESC
        LIMIT 15
    """, conn)
    print(result.to_string(index=False))

    # ── Query 6: Multi-season comparison ───────────────────
    # This is where the database really shines
    # Instant multi-year analysis — no API calls needed
    print("\n--- Query 6: Player improvement across seasons ---")
    print("SQL: Compare same player across multiple seasons\n")

    result = pd.read_sql("""
        SELECT
            player_name,
            season,
            pts,
            reb,
            ast,
            fg_pct
        FROM season_stats
        WHERE player_name IN (
            'Nikola Jokic',
            'Shai Gilgeous-Alexander',
            'Luka Doncic'
        )
        AND gp >= 20
        ORDER BY player_name, season
    """, conn)
    print(result.to_string(index=False))

    # ── Query 7: HAVING — filtering groups ─────────────────
    # HAVING is like WHERE but for GROUP BY results
    print("\n--- Query 7: Teams averaging 115+ points ---")
    print("SQL: GROUP BY team HAVING AVG(pts) > 115\n")

    result = pd.read_sql("""
        SELECT
            team_name,
            wins,
            losses,
            ROUND(win_pct, 3) as win_pct,
            ROUND(pts, 1) as pts_per_game,
            ROUND(fg_pct, 3) as fg_pct
        FROM teams
        WHERE season = '2025-26'
        ORDER BY win_pct DESC
        LIMIT 10
    """, conn)
    print(result.to_string(index=False))

    # ── Query 8: HAVING — filtering groups ─────────────────
    print("\n--- Query 8: Players who averaged more rebounds than assists ---")
    print("SQL: HAVING AVG(reb) > AVG(ast)\n")

    result = pd.read_sql("""
        SELECT
            player_name,
            AVG(reb) as avg_reb,
            AVG(ast) as avg_ast
        FROM season_stats
        WHERE season = '2025-26'
          AND gp >= 20
        GROUP BY player_name
        HAVING AVG(reb) > AVG(ast)
        ORDER BY avg_reb DESC
        LIMIT 10                 
    """, conn)
    print(result.to_string(index=False))

# ============================================================
# MAIN
# ============================================================

def main():
    print("NBA Analytics Database")
    print("=" * 40)

    # Connect to database
    conn = create_connection()
    print(f"Connected to {DB_PATH}")

    # Create tables
    create_tables(conn)

    # Populate with data
    populate_players(conn)

    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]
    for season in seasons:
        populate_season_stats(conn, season)
        populate_teams(conn, season)

    # Run SQL queries
    run_queries(conn)

    conn.close()
    print("\nDatabase connection closed.")
    print(f"Database saved as: {DB_PATH}")

if __name__ == "__main__":
    main()