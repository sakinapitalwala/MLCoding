import pandas as pd
from datetime import datetime


def analyze_bug_sla(df, sla_hours=24):
    print(df.columns)
    # Calculate resolution time
    df["resolution_time"] = (df["resolved_at"] - df["created_at"]).dt.total_seconds()/3600
    
    # SLA Breach Flag
    #df["sla_breached"] = df.apply(lambda row: (True if row["severity"] == "High" and row["resolution_time"] > sla_hours or pd.isna(row["resolved_at"]) else False), axis = 1)
    is_high_severity = df["severity"] == "High"
    is_overdue = (df["resolution_time"] > sla_hours) | df["resolved_at"].isna()
    df["sla_breached"] = is_high_severity & is_overdue
    # Domain Breach Rank
    domain_perc = (df["sla_breached"].groupby(df["domain"]).mean()) * 100

    rank = domain_perc.rank(
    method="dense",
    ascending=False
    ).astype(int)

    #rank_df = rank.rename("domain_breach_rank").reset_index()
    #rank_df = rank_df.rename(columns={"index": "domain"})
    #df["domain_breach_rank"] = df["domain"].map(
    #rank_df.set_index("domain")["domain_breach_rank"]
    #)

    df['domain_breach_rank'] = df['domain'].map(rank)
    return df
if __name__ == "__main__":
    
    data = {
    'bug_id': [101, 102, 103, 104, 105, 106],
    'domain': [
        'Synthesis',
        'Synthesis',
        'Timing',
        'Timing',
        'Place_Route',
        'Place_Route',
    ],
    'created_at': pd.to_datetime([
        '2026-09-01 08:00',
        '2026-09-01 09:00',
        '2026-09-01 10:00',
        '2026-09-01 11:00',
        '2026-09-01 12:00',
        '2026-09-01 13:00',
    ]),
    'resolved_at': pd.to_datetime([
        '2026-09-01 18:00',  # 10 hrs (Not Breached)
        '2026-09-03 09:00',  # 48 hrs (Breached)
        pd.NaT,  # Still Open (Breached)
        '2026-09-02 16:00',  # 29 hrs (Breached)
        '2026-09-01 17:00',  # 5 hrs (Not Breached)
        '2026-09-01 22:00',  # 9 hrs (Not Breached)
    ]),
    'severity': ['High', 'High', 'High', 'High', 'Low', 'High'],
    }

    df = pd.DataFrame(data)
    print(analyze_bug_sla(df, 24))