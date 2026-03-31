import numpy as np 
import pandas as pd

def get_data_quality_metrics_df(df):
    rows = []
    n_rows = len(df)

    # Identify duplicate rows once
    duplicate_mask = df.duplicated(keep=False)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / n_rows) * 100, 2)

        # Count how many duplicate rows involve this column
        duplicate_count = int(
            df.loc[duplicate_mask, col].notna().sum()
        )

        # Outlier count (numeric only)
        outlier_count = 0
        if col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(
                    ((series < lower) | (series > upper)).sum()
                )

        rows.append({
            "Column": col,
            "Null Count": null_count,
            "Duplicate Rows Involved": duplicate_count,
            "Outlier Count": outlier_count
        })

    return pd.DataFrame(rows)

def clean_dataset(df, null_threshold=0.30):
    df_cleaned = df.copy()
    n_rows = len(df_cleaned)

    # 1. Drop duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()

    # 2. Handle null values
    for col in df_cleaned.columns:
        null_ratio = df_cleaned[col].isna().mean()
        if null_ratio <= null_threshold and df_cleaned[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                skew_val = df_cleaned[col].skew()
                if pd.isna(skew_val) or abs(skew_val) > 0.5:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            else:
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])

    # 3. Remove outliers (IQR) only once
    numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
    mask = pd.Series([True] * len(df_cleaned), index=df_cleaned.index)
    
    for col in numeric_cols:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            col_mask = (df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)
            # Nulls are technically not outliers in this logic; if we want to preserve nulls:
            col_mask = col_mask | df_cleaned[col].isna()
            mask = mask & col_mask

    df_cleaned = df_cleaned[mask]
    return df_cleaned

import pandas as pd
import numpy as np

def _continuous_numeric_distribution(df, column_name):
    col = df[column_name].dropna()

    skewness = col.skew() if len(col) > 2 else np.nan

    if pd.isna(skewness):
        dist_type = "Insufficient data"
        suggestion = "Collect more data"
    elif skewness > 1:
        dist_type = "Highly Right Skewed"
        suggestion = "Apply log / sqrt transformation"
    elif skewness > 0.5:
        dist_type = "Moderately Right Skewed"
        suggestion = "Apply log or Box-Cox transformation"
    elif skewness < -1:
        dist_type = "Highly Left Skewed"
        suggestion = "Apply square / exponential transformation"
    elif skewness < -0.5:
        dist_type = "Moderately Left Skewed"
        suggestion = "Apply power transformation"
    else:
        dist_type = "Approximately Normal"
        suggestion = "No transformation required"

    return pd.DataFrame([
        {"Metric": "Column", "Value": column_name},
        {"Metric": "Skewness", "Value": round(skewness, 4)},
        {"Metric": "Distribution Type", "Value": dist_type},
        {"Metric": "Normalization Suggestion", "Value": suggestion}
    ])

def _discrete_numeric_distribution(df, column_name):
    col = df[column_name].dropna()
    value_counts = col.value_counts()

    top_pct = (value_counts.iloc[0] / len(col)) * 100 if len(col) > 0 else 0

    if top_pct > 70:
        balance = "Highly Concentrated"
        suggestion = "Consider treating as categorical"
    elif top_pct > 40:
        balance = "Moderately Concentrated"
        suggestion = "Check if ordinal encoding is appropriate"
    else:
        balance = "Well Distributed"
        suggestion = "No action required"

    return pd.DataFrame([
        {"Metric": "Column", "Value": column_name},
        {"Metric": "Unique Values", "Value": col.nunique()},
        {"Metric": "Dominant Value %", "Value": round(top_pct, 2)},
        {"Metric": "Distribution Type", "Value": balance},
        {"Metric": "Modeling Suggestion", "Value": suggestion}
    ])

def _categorical_distribution(df, column_name):
    col = df[column_name].dropna()
    value_counts = col.value_counts()

    top_pct = (value_counts.iloc[0] / len(col)) * 100 if len(col) > 0 else 0

    if top_pct > 70:
        balance = "Highly Dominated"
        suggestion = "Group rare categories or apply target encoding"
    elif top_pct > 40:
        balance = "Moderately Dominated"
        suggestion = "Consider frequency encoding"
    else:
        balance = "Well Balanced"
        suggestion = "One-hot encoding is safe"

    return pd.DataFrame([
        {"Metric": "Column", "Value": column_name},
        {"Metric": "Unique Categories", "Value": col.nunique()},
        {"Metric": "Dominant Category %", "Value": round(top_pct, 2)},
        {"Metric": "Category Balance", "Value": balance},
        {"Metric": "Encoding Suggestion", "Value": suggestion}
    ])

def _datetime_distribution(df, column_name):
    col = pd.to_datetime(df[column_name], errors="coerce").dropna()

    if col.empty:
        return pd.DataFrame([
            {"Metric": "Column", "Value": column_name},
            {"Metric": "Status", "Value": "No valid datetime values"}
        ])

    month_counts = (
        col.dt.month_name()
        .value_counts()
        .reindex([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        .fillna(0)
        .astype(int)
    )

    peak_month = month_counts.idxmax()

    rows = [{"Metric": "Column", "Value": column_name}]
    for month, count in month_counts.items():
        rows.append({"Metric": month, "Value": count})

    rows.append({
        "Metric": "Peak Activity Month",
        "Value": peak_month
    })

    return pd.DataFrame(rows)

import pandas as pd
import numpy as np

def get_distribution_insights_df(df, column_name, column_type):
    if column_type == "Continuous Numeric":
        return _continuous_numeric_distribution(df, column_name)

    elif column_type == "Discrete Numeric":
        return _discrete_numeric_distribution(df, column_name)

    elif column_type == "Categorical":
        return _categorical_distribution(df, column_name)

    elif column_type == "Date / Time":
        return _datetime_distribution(df, column_name)

    else:
        return pd.DataFrame([
            {"Metric": "Column", "Value": column_name},
            {"Metric": "Status", "Value": "Unsupported column type"}
        ])

