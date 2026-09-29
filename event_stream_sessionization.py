import pandas as pd
def sessionize_events(df, timeout_minutes = 30):
    df = df.copy()
    
    # 1. Sort chronologically per user
    df.sort_values(by = ['user_id', 'timestamp']).reset_index(drop = True)

    # 2. Calculate time gap in minutes from previous event per user
    prev_time = df.groupby("user_id")["timestamp"].shift(1)
    time_gap_mins = (df["timestamp"] - prev_time).dt.total_seconds()/60.0

    # 3. Flag new session boundaries (True if first event or gap > timeout)
    is_new_session = time_gap_mins.isna() | (time_gap_mins > timeout_minutes)

    # 4. Generate session_id using cumsum per user
    session_num  = is_new_session.groupby(df["user_id"]).cumsum()
    df["session_id"] = df["user_id"].astype(str) + "_s" + session_num.astype(str)

    # 5. Calculate session duration and event counts using transform
    session_min_time = df.groupby("session_id")["timestamp"].transform("min")
    session_max_time = df.groupby("session_id")["timestamp"].transform("max")

    df["session_duration_mins"] = (
        session_max_time - session_min_time
    ).dt.total_seconds() / 60.0

    df["session_event_count"] = df.groupby("session_id")["timestamp"].transform(
      "count"
    )

    return df

if __name__ == "__main__":
    data = {
    'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u1'],
    'timestamp': pd.to_datetime([
        '2026-09-23 10:00:00',
        '2026-09-23 10:15:00',  # Gap 15 mins (Same session)
        '2026-09-23 11:00:00',  # Gap 45 mins (New session!)
        '2026-09-23 10:00:00',
        '2026-09-23 10:10:00',  # Gap 10 mins (Same session)
        '2026-09-23 11:10:00',  # Gap 10 mins from prev u1 event (Same session as s2)
    ]),
    'event_type': ['search', 'click', 'checkout', 'login', 'click', 'download']
    }

    df = pd.DataFrame(data)
    print(sessionize_events(df, timeout_minutes = 30))