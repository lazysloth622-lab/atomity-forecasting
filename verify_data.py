from atomity_forecast.config import PATHS, CONFIG
from atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split

def main():
    print("Loading raw data...")
    df_readings, df_meta = load_raw_data(PATHS.data_dir)
    print(f"Readings shape: {df_readings.shape}, Metadata shape: {df_meta.shape}")

    print("Building targets...")
    targets_df = build_targets(df_readings)
    print(f"Targets shape: {targets_df.shape}")
    print(f"Columns: {list(targets_df.columns)}")

    train_df, holdout_df = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)
    print(f"Train split shape: {train_df.shape}")
    print(f"Holdout split shape: {holdout_df.shape}")

    # Integrity assertions
    assert len(holdout_df) == CONFIG.holdout_steps, "Holdout length mismatch"
    assert "active_vm_count" in targets_df.columns, "Missing active_vm_count"
    assert "aggregate_cpu_utilization" in targets_df.columns, "Missing aggregate_cpu_utilization"
    print("Data checks PASSED successfully.")

if __name__ == "__main__":
    main()