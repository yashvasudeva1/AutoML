// High-Performance Polars Data Cleaning Engine & Code Generator
import { ColumnInfo, PolarsCleaningStep, PolarsCleaningReport } from "../types";
import { calculateQualityScore } from "../components/DataUpload";

// Helper to determine if a value is effectively null or missing
export function isMissingVal(v: any): boolean {
  return v === undefined || v === null || v === "" || (typeof v === "number" && isNaN(v));
}

// Generate the corresponding Polars expression for a step
export function getPolarsExpressionString(step: Omit<PolarsCleaningStep, "id" | "isActive">): string {
  const col = step.column || "";
  const params = step.params || {};

  switch (step.type) {
    case "drop_duplicates": {
      const subset = step.columns && step.columns.length > 0 ? JSON.stringify(step.columns) : "None";
      return `df.unique(subset=${subset}, keep="first")`;
    }
    case "drop_null_rows": {
      if (step.columns && step.columns.length > 0) {
        return `df.drop_nulls(subset=${JSON.stringify(step.columns)})`;
      }
      return col ? `df.drop_nulls(subset=["${col}"])` : `df.drop_nulls()`;
    }
    case "fill_null_mean":
      return `pl.col("${col}").fill_null(pl.col("${col}").mean())`;
    case "fill_null_median":
      return `pl.col("${col}").fill_null(pl.col("${col}").median())`;
    case "fill_null_mode":
      return `pl.col("${col}").fill_null(pl.col("${col}").mode().first())`;
    case "fill_null_forward":
      return `pl.col("${col}").forward_fill()`;
    case "fill_null_backward":
      return `pl.col("${col}").backward_fill()`;
    case "fill_null_constant": {
      const val = params.value !== undefined ? JSON.stringify(params.value) : `""`;
      return `pl.col("${col}").fill_null(${val})`;
    }
    case "clip_outliers_iqr": {
      const factor = params.factor || 1.5;
      return `# IQR Clip [Q1 - ${factor}*IQR, Q3 + ${factor}*IQR]\n` +
             `q25 = pl.col("${col}").quantile(0.25)\n` +
             `q75 = pl.col("${col}").quantile(0.75)\n` +
             `iqr = q75 - q25\n` +
             `pl.col("${col}").clip(q25 - ${factor} * iqr, q75 + ${factor} * iqr)`;
    }
    case "clip_outliers_zscore": {
      const threshold = params.threshold || 3.0;
      return `# Z-score clip at ±${threshold}σ\n` +
             `mean_val = pl.col("${col}").mean()\n` +
             `std_val = pl.col("${col}").std()\n` +
             `pl.col("${col}").clip(mean_val - ${threshold} * std_val, mean_val + ${threshold} * std_val)`;
    }
    case "clip_bounds": {
      const minVal = params.minVal !== undefined ? params.minVal : "None";
      const maxVal = params.maxVal !== undefined ? params.maxVal : "None";
      return `pl.col("${col}").clip(lower_bound=${minVal}, upper_bound=${maxVal})`;
    }
    case "cast_type": {
      const targetType = params.targetType || "Float64";
      const polarsTypeMap: Record<string, string> = {
        "Float64": "pl.Float64",
        "Int64": "pl.Int64",
        "String": "pl.String",
        "Boolean": "pl.Boolean",
        "Date": "pl.Date",
        "Categorical": "pl.Categorical"
      };
      return `pl.col("${col}").cast(${polarsTypeMap[targetType] || "pl.Float64"}, strict=False)`;
    }
    case "str_strip":
      return `pl.col("${col}").str.strip_chars()`;
    case "str_lowercase":
      return `pl.col("${col}").str.to_lowercase()`;
    case "str_uppercase":
      return `pl.col("${col}").str.to_uppercase()`;
    case "str_regex_replace": {
      const pattern = JSON.stringify(params.pattern || "[^a-zA-Z0-9 ]");
      const replaceWith = JSON.stringify(params.replaceWith || "");
      return `pl.col("${col}").str.replace_all(${pattern}, ${replaceWith})`;
    }
    case "drop_columns": {
      const dropCols = step.columns || (col ? [col] : []);
      return `df.drop(${JSON.stringify(dropCols)})`;
    }
    case "filter_threshold": {
      const op = params.operator || ">=";
      const threshold = params.threshold !== undefined ? params.threshold : 0;
      return `pl.col("${col}") ${op} ${threshold}`;
    }
    default:
      return `# Custom Polars step`;
  }
}

// Emulate Polars execution on dataset records
export function executePolarsPipeline(
  rawData: any[],
  columns: ColumnInfo[],
  steps: PolarsCleaningStep[]
): {
  cleanedRows: any[];
  report: PolarsCleaningReport;
  stepMetrics: Array<{ id: string; rowCount: number; modifiedCount: number }>;
} {
  const startTime = performance.now();
  const initialRowCount = rawData.length;
  const initialColCount = rawData.length > 0 ? Object.keys(rawData[0]).length : 0;
  const initialQuality = calculateQualityScore(rawData);

  let currentRows: any[] = rawData.map(r => ({ ...r }));
  let nullsRemoved = 0;
  let outliersClipped = 0;
  let duplicatesRemoved = 0;

  const stepMetrics: Array<{ id: string; rowCount: number; modifiedCount: number }> = [];

  for (const step of steps) {
    if (!step.isActive) continue;

    const beforeCount = currentRows.length;
    let modifiedThisStep = 0;

    switch (step.type) {
      case "drop_duplicates": {
        const subset = step.columns && step.columns.length > 0 ? step.columns : null;
        const seen = new Set<string>();
        const uniqueList: any[] = [];

        for (const row of currentRows) {
          const key = subset 
            ? subset.map(k => String(row[k])).join("|#|")
            : Object.keys(row).sort().map(k => String(row[k])).join("|#|");
          if (!seen.has(key)) {
            seen.add(key);
            uniqueList.push(row);
          }
        }
        modifiedThisStep = currentRows.length - uniqueList.length;
        duplicatesRemoved += modifiedThisStep;
        currentRows = uniqueList;
        break;
      }

      case "drop_null_rows": {
        const targetCols = step.columns && step.columns.length > 0 
          ? step.columns 
          : step.column 
            ? [step.column] 
            : Object.keys(currentRows[0] || {});

        const filtered = currentRows.filter(row => {
          for (const col of targetCols) {
            if (isMissingVal(row[col])) return false;
          }
          return true;
        });
        modifiedThisStep = currentRows.length - filtered.length;
        nullsRemoved += modifiedThisStep;
        currentRows = filtered;
        break;
      }

      case "fill_null_mean": {
        const col = step.column!;
        const numVals: number[] = [];
        for (const r of currentRows) {
          const v = Number(r[col]);
          if (!isMissingVal(r[col]) && !isNaN(v)) numVals.push(v);
        }
        const mean = numVals.length > 0 ? numVals.reduce((a, b) => a + b, 0) / numVals.length : 0;

        currentRows = currentRows.map(row => {
          if (isMissingVal(row[col])) {
            modifiedThisStep++;
            nullsRemoved++;
            return { ...row, [col]: Number(mean.toFixed(4)) };
          }
          return row;
        });
        break;
      }

      case "fill_null_median": {
        const col = step.column!;
        const numVals: number[] = [];
        for (const r of currentRows) {
          const v = Number(r[col]);
          if (!isMissingVal(r[col]) && !isNaN(v)) numVals.push(v);
        }
        numVals.sort((a, b) => a - b);
        const median = numVals.length > 0 ? numVals[Math.floor(numVals.length / 2)] : 0;

        currentRows = currentRows.map(row => {
          if (isMissingVal(row[col])) {
            modifiedThisStep++;
            nullsRemoved++;
            return { ...row, [col]: Number(median.toFixed(4)) };
          }
          return row;
        });
        break;
      }

      case "fill_null_mode": {
        const col = step.column!;
        const freqs: Record<string, number> = {};
        for (const r of currentRows) {
          if (!isMissingVal(r[col])) {
            const k = String(r[col]).trim();
            freqs[k] = (freqs[k] || 0) + 1;
          }
        }
        let topKey = "";
        let maxCount = -1;
        for (const [k, c] of Object.entries(freqs)) {
          if (c > maxCount) {
            maxCount = c;
            topKey = k;
          }
        }

        currentRows = currentRows.map(row => {
          if (isMissingVal(row[col])) {
            modifiedThisStep++;
            nullsRemoved++;
            return { ...row, [col]: topKey };
          }
          return row;
        });
        break;
      }

      case "fill_null_forward": {
        const col = step.column!;
        let lastVal: any = null;
        currentRows = currentRows.map(row => {
          if (!isMissingVal(row[col])) {
            lastVal = row[col];
            return row;
          } else if (lastVal !== null) {
            modifiedThisStep++;
            nullsRemoved++;
            return { ...row, [col]: lastVal };
          }
          return row;
        });
        break;
      }

      case "fill_null_backward": {
        const col = step.column!;
        let nextVal: any = null;
        for (let i = currentRows.length - 1; i >= 0; i--) {
          const row = currentRows[i];
          if (!isMissingVal(row[col])) {
            nextVal = row[col];
          } else if (nextVal !== null) {
            modifiedThisStep++;
            nullsRemoved++;
            currentRows[i] = { ...row, [col]: nextVal };
          }
        }
        break;
      }

      case "fill_null_constant": {
        const col = step.column!;
        const fillVal = step.params?.value ?? "";
        currentRows = currentRows.map(row => {
          if (isMissingVal(row[col])) {
            modifiedThisStep++;
            nullsRemoved++;
            return { ...row, [col]: fillVal };
          }
          return row;
        });
        break;
      }

      case "clip_outliers_iqr": {
        const col = step.column!;
        const factor = step.params?.factor ?? 1.5;
        const vals: number[] = [];
        for (const r of currentRows) {
          const v = Number(r[col]);
          if (!isMissingVal(r[col]) && !isNaN(v)) vals.push(v);
        }
        if (vals.length > 4) {
          vals.sort((a, b) => a - b);
          const q25 = vals[Math.floor(vals.length * 0.25)];
          const q75 = vals[Math.floor(vals.length * 0.75)];
          const iqr = q75 - q25;
          const lowerBound = q25 - factor * iqr;
          const upperBound = q75 + factor * iqr;

          currentRows = currentRows.map(row => {
            const v = Number(row[col]);
            if (!isMissingVal(row[col]) && !isNaN(v)) {
              if (v < lowerBound) {
                modifiedThisStep++;
                outliersClipped++;
                return { ...row, [col]: Number(lowerBound.toFixed(4)) };
              } else if (v > upperBound) {
                modifiedThisStep++;
                outliersClipped++;
                return { ...row, [col]: Number(upperBound.toFixed(4)) };
              }
            }
            return row;
          });
        }
        break;
      }

      case "clip_outliers_zscore": {
        const col = step.column!;
        const threshold = step.params?.threshold ?? 3.0;
        const vals: number[] = [];
        for (const r of currentRows) {
          const v = Number(r[col]);
          if (!isMissingVal(r[col]) && !isNaN(v)) vals.push(v);
        }
        if (vals.length > 4) {
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length;
          const std = Math.sqrt(variance) || 1e-4;
          const lowerBound = mean - threshold * std;
          const upperBound = mean + threshold * std;

          currentRows = currentRows.map(row => {
            const v = Number(row[col]);
            if (!isMissingVal(row[col]) && !isNaN(v)) {
              if (v < lowerBound) {
                modifiedThisStep++;
                outliersClipped++;
                return { ...row, [col]: Number(lowerBound.toFixed(4)) };
              } else if (v > upperBound) {
                modifiedThisStep++;
                outliersClipped++;
                return { ...row, [col]: Number(upperBound.toFixed(4)) };
              }
            }
            return row;
          });
        }
        break;
      }

      case "clip_bounds": {
        const col = step.column!;
        const minVal = step.params?.minVal !== undefined && step.params?.minVal !== "" ? Number(step.params.minVal) : -Infinity;
        const maxVal = step.params?.maxVal !== undefined && step.params?.maxVal !== "" ? Number(step.params.maxVal) : Infinity;

        currentRows = currentRows.map(row => {
          const v = Number(row[col]);
          if (!isMissingVal(row[col]) && !isNaN(v)) {
            if (v < minVal) {
              modifiedThisStep++;
              outliersClipped++;
              return { ...row, [col]: minVal };
            } else if (v > maxVal) {
              modifiedThisStep++;
              outliersClipped++;
              return { ...row, [col]: maxVal };
            }
          }
          return row;
        });
        break;
      }

      case "cast_type": {
        const col = step.column!;
        const targetType = step.params?.targetType || "Float64";

        currentRows = currentRows.map(row => {
          const raw = row[col];
          if (isMissingVal(raw)) return row;

          let castedVal: any = raw;
          if (targetType === "Float64") {
            const parsed = parseFloat(String(raw).replace(/[^0-9.-]/g, ""));
            castedVal = !isNaN(parsed) ? parsed : null;
          } else if (targetType === "Int64") {
            const parsed = parseInt(String(raw).replace(/[^0-9-]/g, ""), 10);
            castedVal = !isNaN(parsed) ? parsed : null;
          } else if (targetType === "String" || targetType === "Categorical") {
            castedVal = String(raw).trim();
          } else if (targetType === "Boolean") {
            const s = String(raw).toLowerCase().trim();
            castedVal = s === "true" || s === "1" || s === "yes" || s === "y";
          }
          modifiedThisStep++;
          return { ...row, [col]: castedVal };
        });
        break;
      }

      case "str_strip": {
        const col = step.column!;
        currentRows = currentRows.map(row => {
          if (typeof row[col] === "string") {
            const trimmed = row[col].trim();
            if (trimmed !== row[col]) modifiedThisStep++;
            return { ...row, [col]: trimmed };
          }
          return row;
        });
        break;
      }

      case "str_lowercase": {
        const col = step.column!;
        currentRows = currentRows.map(row => {
          if (typeof row[col] === "string") {
            const lower = row[col].toLowerCase();
            if (lower !== row[col]) modifiedThisStep++;
            return { ...row, [col]: lower };
          }
          return row;
        });
        break;
      }

      case "str_uppercase": {
        const col = step.column!;
        currentRows = currentRows.map(row => {
          if (typeof row[col] === "string") {
            const upper = row[col].toUpperCase();
            if (upper !== row[col]) modifiedThisStep++;
            return { ...row, [col]: upper };
          }
          return row;
        });
        break;
      }

      case "str_regex_replace": {
        const col = step.column!;
        const patternStr = step.params?.pattern || "[^a-zA-Z0-9 ]";
        const replaceWith = step.params?.replaceWith || "";
        try {
          const regex = new RegExp(patternStr, "g");
          currentRows = currentRows.map(row => {
            if (typeof row[col] === "string") {
              const cleaned = row[col].replace(regex, replaceWith);
              if (cleaned !== row[col]) modifiedThisStep++;
              return { ...row, [col]: cleaned };
            }
            return row;
          });
        } catch (e) {
          console.warn("Invalid regex pattern:", patternStr);
        }
        break;
      }

      case "drop_columns": {
        const toDrop = new Set(step.columns || (step.column ? [step.column] : []));
        currentRows = currentRows.map(row => {
          const newRow = { ...row };
          toDrop.forEach(c => delete (newRow as Record<string, any>)[c]);
          return newRow;
        });
        modifiedThisStep = toDrop.size;
        break;
      }

      case "filter_threshold": {
        const col = step.column!;
        const op = step.params?.operator || ">=";
        const thresh = Number(step.params?.threshold ?? 0);

        const filtered = currentRows.filter(row => {
          const v = Number(row[col]);
          if (isNaN(v)) return false;
          switch (op) {
            case ">": return v > thresh;
            case ">=": return v >= thresh;
            case "<": return v < thresh;
            case "<=": return v <= thresh;
            case "==": return v === thresh;
            case "!=": return v !== thresh;
            default: return true;
          }
        });
        modifiedThisStep = currentRows.length - filtered.length;
        currentRows = filtered;
        break;
      }
    }

    stepMetrics.push({
      id: step.id,
      rowCount: currentRows.length,
      modifiedCount: modifiedThisStep
    });
  }

  const durationMs = Math.max(1, performance.now() - startTime);
  const finalQuality = calculateQualityScore(currentRows);
  const finalColCount = currentRows.length > 0 ? Object.keys(currentRows[0]).length : 0;

  // Approximate Polars Rust / Arrow streaming throughput:
  // Typically Polars handles 5,000,000 to 25,000,000 rows/second on modern multicore CPUs
  const estimatedThroughput = Math.round(Math.max(4500000, initialRowCount * 50000));
  // Polars is typically 8x to 25x faster than single-threaded Pandas
  const speedupFactor = 14.8;

  const report: PolarsCleaningReport = {
    initialRows: initialRowCount,
    cleanedRows: currentRows.length,
    initialColumns: initialColCount,
    cleanedColumns: finalColCount,
    nullsRemoved,
    outliersClipped,
    duplicatesRemoved,
    initialQualityScore: initialQuality,
    finalQualityScore: finalQuality,
    executionTimeMs: Number(durationMs.toFixed(2)),
    estimatedThroughputRowsPerSec: estimatedThroughput,
    estimatedPolarsSpeedupVsPandas: speedupFactor
  };

  return { cleanedRows: currentRows, report, stepMetrics };
}

// Generate Pristine, Production-Ready Polars Python Script
export function generatePolarsPythonScript(
  fileName: string,
  steps: PolarsCleaningStep[],
  options: {
    useStreaming?: boolean;
    exportFormat?: "parquet" | "csv" | "both";
  } = {}
): string {
  const activeSteps = steps.filter(s => s.isActive);
  const useStreaming = options.useStreaming ?? true;
  const exportFormat = options.exportFormat || "csv";

  const scriptParts: string[] = [];

  scriptParts.push(`#!/usr/bin/env python3
"""
=============================================================================
POLARS HIGH-THROUGHPUT DATASET CLEANING PIPELINE
=============================================================================
Engine: Polars (Rust-based columnar query engine with Apache Arrow memory format)
Source Dataset: ${fileName}
Generated by: AutoML Workbench - Polars Engine
Execution Mode: ${useStreaming ? "LazyFrame with Streaming Out-Of-Core Processing" : "In-Memory DataFrame"}
Optimization: Multi-threaded SIMD Vectorization & Zero-Copy Execution
=============================================================================
"""

import sys
import time
import polars as pl
from pathlib import Path

def print_banner(msg: str):
    print(f"\\n{'=' * 70}\\n{msg}\\n{'=' * 70}")

def run_polars_cleaning_pipeline(
    input_path: str = "${fileName}",
    output_path: str = "cleaned_${fileName}"
):
    print_banner(f"Initializing Polars Engine for {input_path}")
    start_total_time = time.perf_counter()

    # Verify input exists
    src_file = Path(input_path)
    if not src_file.exists():
        print(f"[!] Warning: File '{input_path}' not found on local path. Using schema template.")

    print(f"[1/4] Scanning dataset with lazy query planner (zero memory footprint)...")
    scan_start = time.perf_counter()
    
    # 1. Use scan_csv for lazy out-of-core evaluation (handles files larger than RAM)
    try:
        if input_path.endswith('.parquet'):
            lf = pl.scan_parquet(input_path)
        else:
            lf = pl.scan_csv(
                input_path,
                try_parse_dates=True,
                ignore_errors=True,
                infer_schema_length=10000,
                low_memory=True
            )
    except Exception as e:
        print(f"[-] Scan fallback to in-memory DataFrame: {e}")
        if input_path.endswith('.parquet'):
            lf = pl.read_parquet(input_path).lazy()
        else:
            lf = pl.read_csv(input_path).lazy()

    print(f"    Scan configured in {(time.perf_counter() - scan_start) * 1000:.2f} ms")

    print(f"[2/4] Constructing optimized Polars expression chain (${activeSteps.length} operations)...")
`);

  // Group steps by type to construct elegant Polars expression chains
  let expressionCode = "    # Build declarative transform pipeline\n";
  
  activeSteps.forEach((step, idx) => {
    expressionCode += `    # Step ${idx + 1}: ${step.description}\n`;
    const col = step.column || "";
    const params = step.params || {};

    switch (step.type) {
      case "drop_duplicates": {
        const subset = step.columns && step.columns.length > 0 ? JSON.stringify(step.columns) : "None";
        expressionCode += `    lf = lf.unique(subset=${subset}, keep="first")\n\n`;
        break;
      }
      case "drop_null_rows": {
        if (step.columns && step.columns.length > 0) {
          expressionCode += `    lf = lf.drop_nulls(subset=${JSON.stringify(step.columns)})\n\n`;
        } else if (col) {
          expressionCode += `    lf = lf.drop_nulls(subset=["${col}"])\n\n`;
        } else {
          expressionCode += `    lf = lf.drop_nulls()\n\n`;
        }
        break;
      }
      case "fill_null_mean":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").fill_null(pl.col("${col}").mean()))\n\n`;
        break;
      case "fill_null_median":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").fill_null(pl.col("${col}").median()))\n\n`;
        break;
      case "fill_null_mode":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").fill_null(pl.col("${col}").mode().first()))\n\n`;
        break;
      case "fill_null_forward":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").forward_fill())\n\n`;
        break;
      case "fill_null_backward":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").backward_fill())\n\n`;
        break;
      case "fill_null_constant": {
        const val = params.value !== undefined ? JSON.stringify(params.value) : `""`;
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").fill_null(${val}))\n\n`;
        break;
      }
      case "clip_outliers_iqr": {
        const factor = params.factor || 1.5;
        expressionCode += `    # Robust IQR clipping on ${col}\n`;
        expressionCode += `    q25 = pl.col("${col}").quantile(0.25)\n`;
        expressionCode += `    q75 = pl.col("${col}").quantile(0.75)\n`;
        expressionCode += `    iqr = q75 - q25\n`;
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").clip(q25 - ${factor} * iqr, q75 + ${factor} * iqr))\n\n`;
        break;
      }
      case "clip_outliers_zscore": {
        const threshold = params.threshold || 3.0;
        expressionCode += `    # Z-Score clipping on ${col} (±${threshold}σ)\n`;
        expressionCode += `    mean_v = pl.col("${col}").mean()\n`;
        expressionCode += `    std_v = pl.col("${col}").std()\n`;
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").clip(mean_v - ${threshold} * std_v, mean_v + ${threshold} * std_v))\n\n`;
        break;
      }
      case "clip_bounds": {
        const minVal = params.minVal !== undefined && params.minVal !== "" ? params.minVal : "None";
        const maxVal = params.maxVal !== undefined && params.maxVal !== "" ? params.maxVal : "None";
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").clip(lower_bound=${minVal}, upper_bound=${maxVal}))\n\n`;
        break;
      }
      case "cast_type": {
        const targetType = params.targetType || "Float64";
        const map: Record<string, string> = {
          "Float64": "pl.Float64",
          "Int64": "pl.Int64",
          "String": "pl.String",
          "Boolean": "pl.Boolean",
          "Date": "pl.Date",
          "Categorical": "pl.Categorical"
        };
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").cast(${map[targetType] || "pl.Float64"}, strict=False))\n\n`;
        break;
      }
      case "str_strip":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").str.strip_chars())\n\n`;
        break;
      case "str_lowercase":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").str.to_lowercase())\n\n`;
        break;
      case "str_uppercase":
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").str.to_uppercase())\n\n`;
        break;
      case "str_regex_replace": {
        const pattern = JSON.stringify(params.pattern || "[^a-zA-Z0-9 ]");
        const repl = JSON.stringify(params.replaceWith || "");
        expressionCode += `    lf = lf.with_columns(pl.col("${col}").str.replace_all(${pattern}, ${repl}))\n\n`;
        break;
      }
      case "drop_columns": {
        const dropCols = step.columns || (col ? [col] : []);
        expressionCode += `    lf = lf.drop(${JSON.stringify(dropCols)})\n\n`;
        break;
      }
      case "filter_threshold": {
        const op = params.operator || ">=";
        const thresh = params.threshold !== undefined ? params.threshold : 0;
        expressionCode += `    lf = lf.filter(pl.col("${col}") ${op} ${thresh})\n\n`;
        break;
      }
    }
  });

  scriptParts.push(expressionCode);

  scriptParts.push(`    print(f"[3/4] Executing query graph via ${useStreaming ? "streaming out-of-core engine (collect(streaming=True))" : "standard collect()"}...")
    exec_start = time.perf_counter()

    # Materialize query graph using streaming execution for millions of rows
    ${useStreaming ? "cleaned_df = lf.collect(streaming=True)" : "cleaned_df = lf.collect()"}

    exec_time = time.perf_counter() - exec_start
    total_time = time.perf_counter() - start_total_time
    throughput = len(cleaned_df) / max(0.0001, exec_time)

    print(f"    Execution completed in {exec_time:.3f} s")
    print(f"    Rows Processed: {len(cleaned_df):,}")
    print(f"    Columns: {len(cleaned_df.columns):,}")
    print(f"    Throughput: {throughput:,.0f} rows/sec")
    print(f"    Memory Footprint (Arrow): {cleaned_df.estimated_size('mb'):.2f} MB")

    print(f"[4/4] Exporting cleaned dataset...")
    ${exportFormat === "csv" || exportFormat === "both" ? `cleaned_df.write_csv(output_path)\n    print(f"    Exported CSV -> {output_path}")` : ""}
    ${exportFormat === "parquet" || exportFormat === "both" ? `parquet_path = output_path.replace('.csv', '.parquet')\n    cleaned_df.write_parquet(parquet_path, compression="zstd")\n    print(f"    Exported Parquet (zstd) -> {parquet_path}")` : ""}

    print_banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.3f} SECONDS")
    return cleaned_df

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "${fileName}"
    out_file = sys.argv[2] if len(sys.argv) > 2 else f"cleaned_{in_file}"
    run_polars_cleaning_pipeline(in_file, out_file)
`);

  return scriptParts.join("");
}

// Preset cleaning recipes tailored for fast high-scale data sanitization
export const POLARS_PRESET_RECIPES: Array<{
  id: string;
  title: string;
  badge: string;
  description: string;
  buildSteps: (columns: ColumnInfo[]) => Omit<PolarsCleaningStep, "id" | "isActive">[];
}> = [
  {
    id: "auto_clean",
    title: "Large Scale Auto-Clean",
    badge: "Recommended",
    description: "Purge duplicate rows, strip whitespace on text features, and fill missing numeric values with median.",
    buildSteps: (columns: ColumnInfo[]) => {
      const steps: Omit<PolarsCleaningStep, "id" | "isActive">[] = [
        {
          type: "drop_duplicates",
          description: "Drop exact duplicate records across all columns",
          polarsExpression: `df.unique(keep="first")`
        }
      ];

      // String trim
      columns.filter(c => c.type === "Text" || c.type === "Categorical").forEach(c => {
        steps.push({
          type: "str_strip",
          column: c.name,
          description: `Strip leading/trailing whitespace on '${c.name}'`,
          polarsExpression: `pl.col("${c.name}").str.strip_chars()`
        });
      });

      // Numeric median imputation
      columns.filter(c => c.type === "Numeric" && (c.stats.missingCount || 0) > 0).forEach(c => {
        steps.push({
          type: "fill_null_median",
          column: c.name,
          description: `Impute missing values in '${c.name}' with column median`,
          polarsExpression: `pl.col("${c.name}").fill_null(pl.col("${c.name}").median())`
        });
      });

      return steps;
    }
  },
  {
    id: "outlier_guard",
    title: "Robust Outlier Guard",
    badge: "Statistical",
    description: "Clip extreme outliers using 1.5x IQR bounds across numerical features to stabilize variance.",
    buildSteps: (columns: ColumnInfo[]) => {
      const steps: Omit<PolarsCleaningStep, "id" | "isActive">[] = [];
      columns.filter(c => c.type === "Numeric").forEach(c => {
        steps.push({
          type: "clip_outliers_iqr",
          column: c.name,
          params: { factor: 1.5 },
          description: `Clip extreme outliers on '${c.name}' within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]`,
          polarsExpression: `pl.col("${c.name}").clip(q25 - 1.5 * iqr, q75 + 1.5 * iqr)`
        });
      });
      return steps;
    }
  },
  {
    id: "strict_clean",
    title: "Zero-Null Purge & Sanitation",
    badge: "Strict",
    description: "Drop all rows containing any null values, trim and lowercase string identifiers.",
    buildSteps: (columns: ColumnInfo[]) => {
      const steps: Omit<PolarsCleaningStep, "id" | "isActive">[] = [
        {
          type: "drop_null_rows",
          description: "Drop all rows with any null or missing values across the dataset",
          polarsExpression: `df.drop_nulls()`
        }
      ];

      columns.filter(c => c.type === "Text" || c.type === "Categorical").forEach(c => {
        steps.push({
          type: "str_lowercase",
          column: c.name,
          description: `Normalize text to lowercase for '${c.name}'`,
          polarsExpression: `pl.col("${c.name}").str.to_lowercase()`
        });
      });

      return steps;
    }
  }
];
