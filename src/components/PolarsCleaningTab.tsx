import React, { useState, useMemo, useEffect } from "react";
import { DatasetSummary, ColumnInfo, PolarsCleaningStep, PolarsCleaningReport } from "../types";
import { 
  executePolarsPipeline, 
  generatePolarsPythonScript, 
  getPolarsExpressionString, 
  POLARS_PRESET_RECIPES 
} from "../utils/polarsEngine";
import { 
  Sparkles, Zap, Play, RotateCw, Trash2, ArrowUp, ArrowDown, 
  CheckCircle, AlertTriangle, FileCode, Copy, Download, 
  Layers, ShieldAlert, Cpu, Check, RefreshCw, BarChart2,
  Table, ChevronDown, ChevronUp, FileText, ArrowRight, CornerDownRight
} from "lucide-react";

interface PolarsCleaningTabProps {
  summary: DatasetSummary;
  rawData: any[] | null;
  onApplyCleanedData: (cleanedRows: any[]) => void;
  onRevertToOriginal: () => void;
  isDataCleaned: boolean;
}

export default function PolarsCleaningTab({
  summary,
  rawData,
  onApplyCleanedData,
  onRevertToOriginal,
  isDataCleaned
}: PolarsCleaningTabProps) {
  // Active dataset rows
  const activeRows = useMemo(() => {
    return rawData && rawData.length > 0 ? rawData : summary.sampleData || [];
  }, [rawData, summary.sampleData]);

  // Columns
  const numericCols = useMemo(() => {
    return summary.columns.filter(c => c.type === "Numeric").map(c => c.name);
  }, [summary.columns]);

  const textCols = useMemo(() => {
    return summary.columns.filter(c => c.type === "Text" || c.type === "Categorical").map(c => c.name);
  }, [summary.columns]);

  const allColNames = useMemo(() => {
    return summary.columns.map(c => c.name);
  }, [summary.columns]);

  // Steps in the cleaning pipeline
  const [pipelineSteps, setPipelineSteps] = useState<PolarsCleaningStep[]>([
    {
      id: "step_dedup",
      type: "drop_duplicates",
      description: "Drop exact duplicate records across all columns",
      polarsExpression: `df.unique(keep="first")`,
      isActive: true
    }
  ]);

  // Operation Builder Form State
  const [opCategory, setOpCategory] = useState<"nulls" | "outliers" | "strings" | "types" | "dedup" | "filter">("nulls");
  const [selectedCol, setSelectedCol] = useState<string>(allColNames[0] || "");
  const [nullAction, setNullAction] = useState<string>("median");
  const [constantValue, setConstantValue] = useState<string>("0");
  const [outlierMethod, setOutlierMethod] = useState<string>("iqr_1.5");
  const [stringAction, setStringAction] = useState<string>("strip");
  const [regexPattern, setRegexPattern] = useState<string>("[^a-zA-Z0-9 ]");
  const [regexReplace, setRegexReplace] = useState<string>("");
  const [targetType, setTargetType] = useState<string>("Float64");
  const [filterOp, setFilterOp] = useState<string>(">=");
  const [filterThresh, setFilterThresh] = useState<number>(0);

  // Sync selected column if columns change
  useEffect(() => {
    if (allColNames.length > 0 && !allColNames.includes(selectedCol)) {
      setSelectedCol(allColNames[0]);
    }
  }, [allColNames, selectedCol]);

  // Subtab for viewing: preview table vs python code vs benchmark
  const [viewMode, setViewMode] = useState<"preview" | "code" | "benchmark">("preview");

  // Execution Results
  const [cleanedResult, setCleanedResult] = useState<{
    cleanedRows: any[];
    report: PolarsCleaningReport;
    stepMetrics: Array<{ id: string; rowCount: number; modifiedCount: number }>;
  } | null>(null);

  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [appliedNotification, setAppliedNotification] = useState<boolean>(false);

  // Auto-run pipeline preview when pipeline steps or activeRows change
  useEffect(() => {
    if (activeRows.length > 0) {
      const res = executePolarsPipeline(activeRows, summary.columns, pipelineSteps);
      setCleanedResult(res);
    }
  }, [activeRows, pipelineSteps, summary.columns]);

  // Live expression preview in builder
  const liveBuilderExpression = useMemo(() => {
    const fakeStep: any = {
      type: "drop_duplicates",
      column: selectedCol,
      params: {}
    };

    if (opCategory === "nulls") {
      if (nullAction === "drop_row") fakeStep.type = "drop_null_rows";
      else if (nullAction === "mean") fakeStep.type = "fill_null_mean";
      else if (nullAction === "median") fakeStep.type = "fill_null_median";
      else if (nullAction === "mode") fakeStep.type = "fill_null_mode";
      else if (nullAction === "forward") fakeStep.type = "fill_null_forward";
      else if (nullAction === "backward") fakeStep.type = "fill_null_backward";
      else if (nullAction === "constant") {
        fakeStep.type = "fill_null_constant";
        fakeStep.params.value = constantValue;
      }
    } else if (opCategory === "outliers") {
      if (outlierMethod === "iqr_1.5") {
        fakeStep.type = "clip_outliers_iqr";
        fakeStep.params.factor = 1.5;
      } else if (outlierMethod === "iqr_3.0") {
        fakeStep.type = "clip_outliers_iqr";
        fakeStep.params.factor = 3.0;
      } else if (outlierMethod === "zscore_3") {
        fakeStep.type = "clip_outliers_zscore";
        fakeStep.params.threshold = 3.0;
      }
    } else if (opCategory === "strings") {
      if (stringAction === "strip") fakeStep.type = "str_strip";
      else if (stringAction === "lower") fakeStep.type = "str_lowercase";
      else if (stringAction === "upper") fakeStep.type = "str_uppercase";
      else if (stringAction === "regex") {
        fakeStep.type = "str_regex_replace";
        fakeStep.params.pattern = regexPattern;
        fakeStep.params.replaceWith = regexReplace;
      }
    } else if (opCategory === "types") {
      fakeStep.type = "cast_type";
      fakeStep.params.targetType = targetType;
    } else if (opCategory === "dedup") {
      fakeStep.type = "drop_duplicates";
    } else if (opCategory === "filter") {
      fakeStep.type = "filter_threshold";
      fakeStep.params.operator = filterOp;
      fakeStep.params.threshold = filterThresh;
    }

    return getPolarsExpressionString(fakeStep);
  }, [opCategory, selectedCol, nullAction, constantValue, outlierMethod, stringAction, regexPattern, regexReplace, targetType, filterOp, filterThresh]);

  // Add Step to Recipe
  const handleAddStep = (e: React.FormEvent) => {
    e.preventDefault();

    let newStep: PolarsCleaningStep;
    const stepId = `step_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`;

    if (opCategory === "nulls") {
      if (nullAction === "drop_row") {
        newStep = {
          id: stepId,
          type: "drop_null_rows",
          column: selectedCol,
          description: `Drop rows where '${selectedCol}' is null`,
          polarsExpression: `df.drop_nulls(subset=["${selectedCol}"])`,
          isActive: true
        };
      } else if (nullAction === "mean") {
        newStep = {
          id: stepId,
          type: "fill_null_mean",
          column: selectedCol,
          description: `Impute nulls in '${selectedCol}' with mean`,
          polarsExpression: `pl.col("${selectedCol}").fill_null(pl.col("${selectedCol}").mean())`,
          isActive: true
        };
      } else if (nullAction === "median") {
        newStep = {
          id: stepId,
          type: "fill_null_median",
          column: selectedCol,
          description: `Impute nulls in '${selectedCol}' with median`,
          polarsExpression: `pl.col("${selectedCol}").fill_null(pl.col("${selectedCol}").median())`,
          isActive: true
        };
      } else if (nullAction === "mode") {
        newStep = {
          id: stepId,
          type: "fill_null_mode",
          column: selectedCol,
          description: `Impute nulls in '${selectedCol}' with mode`,
          polarsExpression: `pl.col("${selectedCol}").fill_null(pl.col("${selectedCol}").mode().first())`,
          isActive: true
        };
      } else if (nullAction === "forward") {
        newStep = {
          id: stepId,
          type: "fill_null_forward",
          column: selectedCol,
          description: `Forward-fill null values in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").forward_fill()`,
          isActive: true
        };
      } else if (nullAction === "backward") {
        newStep = {
          id: stepId,
          type: "fill_null_backward",
          column: selectedCol,
          description: `Backward-fill null values in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").backward_fill()`,
          isActive: true
        };
      } else {
        newStep = {
          id: stepId,
          type: "fill_null_constant",
          column: selectedCol,
          params: { value: constantValue },
          description: `Fill nulls in '${selectedCol}' with constant '${constantValue}'`,
          polarsExpression: `pl.col("${selectedCol}").fill_null("${constantValue}")`,
          isActive: true
        };
      }
    } else if (opCategory === "outliers") {
      if (outlierMethod === "iqr_1.5") {
        newStep = {
          id: stepId,
          type: "clip_outliers_iqr",
          column: selectedCol,
          params: { factor: 1.5 },
          description: `Clip outliers on '${selectedCol}' using 1.5x IQR bounds`,
          polarsExpression: `pl.col("${selectedCol}").clip(q25 - 1.5 * iqr, q75 + 1.5 * iqr)`,
          isActive: true
        };
      } else if (outlierMethod === "iqr_3.0") {
        newStep = {
          id: stepId,
          type: "clip_outliers_iqr",
          column: selectedCol,
          params: { factor: 3.0 },
          description: `Clip extreme outliers on '${selectedCol}' using 3.0x IQR bounds`,
          polarsExpression: `pl.col("${selectedCol}").clip(q25 - 3.0 * iqr, q75 + 3.0 * iqr)`,
          isActive: true
        };
      } else {
        newStep = {
          id: stepId,
          type: "clip_outliers_zscore",
          column: selectedCol,
          params: { threshold: 3.0 },
          description: `Clip outliers on '${selectedCol}' using ±3.0σ Z-Score`,
          polarsExpression: `pl.col("${selectedCol}").clip(mean - 3 * std, mean + 3 * std)`,
          isActive: true
        };
      }
    } else if (opCategory === "strings") {
      if (stringAction === "strip") {
        newStep = {
          id: stepId,
          type: "str_strip",
          column: selectedCol,
          description: `Strip whitespace characters in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").str.strip_chars()`,
          isActive: true
        };
      } else if (stringAction === "lower") {
        newStep = {
          id: stepId,
          type: "str_lowercase",
          column: selectedCol,
          description: `Lowercase text in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").str.to_lowercase()`,
          isActive: true
        };
      } else if (stringAction === "upper") {
        newStep = {
          id: stepId,
          type: "str_uppercase",
          column: selectedCol,
          description: `Uppercase text in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").str.to_uppercase()`,
          isActive: true
        };
      } else {
        newStep = {
          id: stepId,
          type: "str_regex_replace",
          column: selectedCol,
          params: { pattern: regexPattern, replaceWith: regexReplace },
          description: `Regex replace pattern '${regexPattern}' in '${selectedCol}'`,
          polarsExpression: `pl.col("${selectedCol}").str.replace_all("${regexPattern}", "${regexReplace}")`,
          isActive: true
        };
      }
    } else if (opCategory === "types") {
      newStep = {
        id: stepId,
        type: "cast_type",
        column: selectedCol,
        params: { targetType },
        description: `Cast '${selectedCol}' to ${targetType}`,
        polarsExpression: `pl.col("${selectedCol}").cast(pl.${targetType})`,
        isActive: true
      };
    } else if (opCategory === "filter") {
      newStep = {
        id: stepId,
        type: "filter_threshold",
        column: selectedCol,
        params: { operator: filterOp, threshold: filterThresh },
        description: `Filter rows where '${selectedCol}' ${filterOp} ${filterThresh}`,
        polarsExpression: `pl.col("${selectedCol}") ${filterOp} ${filterThresh}`,
        isActive: true
      };
    } else {
      newStep = {
        id: stepId,
        type: "drop_duplicates",
        description: `Drop exact duplicate rows across all columns`,
        polarsExpression: `df.unique(keep="first")`,
        isActive: true
      };
    }

    setPipelineSteps(prev => [...prev, newStep]);
  };

  // Reorder steps
  const moveStep = (index: number, direction: "up" | "down") => {
    if (direction === "up" && index > 0) {
      setPipelineSteps(prev => {
        const copy = [...prev];
        const temp = copy[index - 1];
        copy[index - 1] = copy[index];
        copy[index] = temp;
        return copy;
      });
    } else if (direction === "down" && index < pipelineSteps.length - 1) {
      setPipelineSteps(prev => {
        const copy = [...prev];
        const temp = copy[index + 1];
        copy[index + 1] = copy[index];
        copy[index] = temp;
        return copy;
      });
    }
  };

  // Toggle step active
  const toggleStepActive = (id: string) => {
    setPipelineSteps(prev => prev.map(s => s.id === id ? { ...s, isActive: !s.isActive } : s));
  };

  // Delete step
  const removeStep = (id: string) => {
    setPipelineSteps(prev => prev.filter(s => s.id !== id));
  };

  // Load Preset Recipe
  const applyPreset = (presetId: string) => {
    const preset = POLARS_PRESET_RECIPES.find(p => p.id === presetId);
    if (!preset) return;
    const newSteps = preset.buildSteps(summary.columns).map((s, idx) => ({
      ...s,
      id: `preset_${presetId}_${idx}_${Date.now()}`,
      isActive: true
    }));
    setPipelineSteps(newSteps);
  };

  // Commit Cleaned Dataset to Workspace
  const handleApplyCleanedToWorkspace = () => {
    if (cleanedResult) {
      onApplyCleanedData(cleanedResult.cleanedRows);
      setAppliedNotification(true);
      setTimeout(() => setAppliedNotification(false), 3000);
    }
  };

  // Generated Python Script
  const pythonScript = useMemo(() => {
    return generatePolarsPythonScript(summary.fileName, pipelineSteps, {
      useStreaming: true,
      exportFormat: "both"
    });
  }, [summary.fileName, pipelineSteps]);

  // Copy Python Code
  const handleCopyCode = () => {
    navigator.clipboard.writeText(pythonScript);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  // Download Python Script (.py)
  const handleDownloadScript = () => {
    const blob = new Blob([pythonScript], { type: "text/x-python;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `polars_pipeline_${summary.fileName.replace(/[^a-zA-Z0-9]/g, "_")}.py`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Download Cleaned CSV
  const handleDownloadCsv = () => {
    if (!cleanedResult || cleanedResult.cleanedRows.length === 0) return;
    const headers = Object.keys(cleanedResult.cleanedRows[0]);
    const csvLines = [
      headers.join(","),
      ...cleanedResult.cleanedRows.map(row => 
        headers.map(h => {
          const val = row[h];
          if (val === null || val === undefined) return "";
          const str = String(val).replace(/"/g, '""');
          return str.includes(",") || str.includes("\n") || str.includes('"') ? `"${str}"` : str;
        }).join(",")
      )
    ];
    const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `cleaned_${summary.fileName.replace(/\.[^/.]+$/, "")}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const report = cleanedResult?.report;

  return (
    <div id="polars-cleaning-tab-content" className="space-y-6">
      
      {/* Polars Hero Banner */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 relative overflow-hidden">
        <div className="absolute top-0 right-0 p-5 opacity-10 pointer-events-none">
          <Zap className="h-32 w-32 text-sky-400" />
        </div>
        <div className="max-w-3xl space-y-2">
          <div className="flex items-center space-x-2.5">
            <span className="px-2 py-0.5 border border-sky-500/30 text-sky-400 bg-sky-500/10 font-mono text-[9px] font-bold uppercase tracking-wider rounded">
              Rust-Powered Polars Engine
            </span>
            <span className="text-[9px] font-mono text-[#73726C] border border-[#E5E2D9] bg-[#FAF9F5] px-2 py-0.5 rounded">
              Arrow Memory Format · SIMD Multi-threaded
            </span>
          </div>
          <h2 className="font-mono font-bold text-[#1F1E1D] text-sm uppercase tracking-wider">
            Large Scale Dataset Cleaning & Transform Pipeline
          </h2>
          <p className="text-xs text-[#73726C] leading-relaxed font-sans">
            Declaratively clean, normalize, impute, and deduplicate millions of records with near-zero latency. Built on the Polars query graph with lazy execution, out-of-core streaming, and automatic production Python script generation.
          </p>
        </div>
      </div>

      {/* Preset Action Cards */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] font-bold text-[#73726C] uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-sky-400" />
            Instant Polars Recipes
          </span>
          <span className="text-[10px] text-[#9E9D96] font-mono">1-Click Scale-Optimized Presets</span>
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          {POLARS_PRESET_RECIPES.map(preset => (
            <div
              key={preset.id}
              onClick={() => applyPreset(preset.id)}
              className="p-3.5 bg-[#FAF9F5] border border-[#E5E2D9] hover:border-sky-500/40 rounded transition-all cursor-pointer group flex flex-col justify-between"
            >
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-mono font-bold text-[#1F1E1D] text-xs group-hover:text-sky-400 transition-colors">
                    {preset.title}
                  </h4>
                  <span className="text-[8px] font-mono uppercase bg-[#FAF9F5] text-sky-400 border border-sky-500/20 px-1.5 py-0.2 rounded font-bold">
                    {preset.badge}
                  </span>
                </div>
                <p className="text-[10px] text-[#9E9D96] font-mono line-clamp-2 leading-relaxed">
                  {preset.description}
                </p>
              </div>

              <div className="pt-2 mt-2 border-t border-[#E5E2D9] flex items-center justify-between text-[9px] font-mono font-bold text-sky-400">
                <span>Load Recipe</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Grid: Recipe Builder & Active Pipeline */}
      <div className="grid lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Col (5 cols): Operation Configurator */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div className="flex items-center space-x-2">
                <Layers className="h-4 w-4 text-sky-400" />
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Add Polars Operation
                </h3>
              </div>
              <span className="text-[9px] font-mono text-[#9E9D96] uppercase">
                Vectorized Step
              </span>
            </div>

            {/* Category Selector Tabs */}
            <div className="grid grid-cols-3 gap-1.5 font-mono text-[10px]">
              <button
                type="button"
                onClick={() => setOpCategory("nulls")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "nulls"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Nulls / Impute
              </button>
              <button
                type="button"
                onClick={() => setOpCategory("outliers")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "outliers"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Outliers
              </button>
              <button
                type="button"
                onClick={() => setOpCategory("strings")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "strings"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Strings / Text
              </button>
              <button
                type="button"
                onClick={() => setOpCategory("types")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "types"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Schema / Cast
              </button>
              <button
                type="button"
                onClick={() => setOpCategory("dedup")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "dedup"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Deduplication
              </button>
              <button
                type="button"
                onClick={() => setOpCategory("filter")}
                className={`py-1.5 px-2 rounded border transition-colors ${
                  opCategory === "filter"
                    ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                    : "bg-[#FAF9F5] border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D]"
                }`}
              >
                Row Filter
              </button>
            </div>

            {/* Form Fields */}
            <form onSubmit={handleAddStep} className="space-y-3 font-mono text-[11px]">
              
              {/* Target Column (if applicable) */}
              {opCategory !== "dedup" && (
                <div className="space-y-1">
                  <label className="text-[10px] text-[#9E9D96] uppercase tracking-wider block font-bold">
                    Target Column
                  </label>
                  <select
                    value={selectedCol}
                    onChange={(e) => setSelectedCol(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                  >
                    {allColNames.map((c, idx) => (
                      <option key={idx} value={c}>
                        {c} ({summary.columns.find(col => col.name === c)?.type || "Any"})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Category-Specific Form Options */}
              {opCategory === "nulls" && (
                <div className="space-y-2">
                  <label className="text-[10px] text-[#9E9D96] uppercase tracking-wider block font-bold">
                    Imputation / Removal Strategy
                  </label>
                  <select
                    value={nullAction}
                    onChange={(e) => setNullAction(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                  >
                    <option value="median">Fill Null with Column Median (Numeric)</option>
                    <option value="mean">Fill Null with Column Mean (Numeric)</option>
                    <option value="mode">Fill Null with Mode / Most Frequent Value</option>
                    <option value="forward">Forward Fill (pl.col.forward_fill)</option>
                    <option value="backward">Backward Fill (pl.col.backward_fill)</option>
                    <option value="constant">Fill Null with Constant Literal Value</option>
                    <option value="drop_row">Drop Rows with Null in this column</option>
                  </select>

                  {nullAction === "constant" && (
                    <div className="pt-1">
                      <label className="text-[9px] text-[#9E9D96] block uppercase mb-1">Constant Literal</label>
                      <input
                        type="text"
                        value={constantValue}
                        onChange={(e) => setConstantValue(e.target.value)}
                        placeholder="e.g. 0, Unknown, Missing"
                        className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none"
                      />
                    </div>
                  )}
                </div>
              )}

              {opCategory === "outliers" && (
                <div className="space-y-2">
                  <label className="text-[10px] text-[#9E9D96] uppercase tracking-wider block font-bold">
                    Outlier Boundary Algorithm
                  </label>
                  <select
                    value={outlierMethod}
                    onChange={(e) => setOutlierMethod(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                  >
                    <option value="iqr_1.5">Clip to 1.5x IQR (Standard Tukey Fence)</option>
                    <option value="iqr_3.0">Clip to 3.0x IQR (Extreme Outliers Only)</option>
                    <option value="zscore_3">Clip to ±3.0σ (Standard Z-Score Normalization)</option>
                  </select>
                  <p className="text-[9px] text-[#9E9D96]">
                    Values outside bounds are clipped in-place without dropping data records.
                  </p>
                </div>
              )}

              {opCategory === "strings" && (
                <div className="space-y-2">
                  <label className="text-[10px] text-[#9E9D96] uppercase tracking-wider block font-bold">
                    String Sanitization
                  </label>
                  <select
                    value={stringAction}
                    onChange={(e) => setStringAction(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                  >
                    <option value="strip">Strip Whitespace (pl.col.str.strip_chars)</option>
                    <option value="lower">Lowercase (pl.col.str.to_lowercase)</option>
                    <option value="upper">Uppercase (pl.col.str.to_uppercase)</option>
                    <option value="regex">Regex Replace Special Characters</option>
                  </select>

                  {stringAction === "regex" && (
                    <div className="grid grid-cols-2 gap-2 pt-1">
                      <div>
                        <label className="text-[9px] text-[#9E9D96] block uppercase mb-1">Regex Pattern</label>
                        <input
                          type="text"
                          value={regexPattern}
                          onChange={(e) => setRegexPattern(e.target.value)}
                          placeholder="e.g. [^a-zA-Z0-9 ]"
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs font-mono"
                        />
                      </div>
                      <div>
                        <label className="text-[9px] text-[#9E9D96] block uppercase mb-1">Replacement</label>
                        <input
                          type="text"
                          value={regexReplace}
                          onChange={(e) => setRegexReplace(e.target.value)}
                          placeholder="e.g. empty or _"
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs font-mono"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {opCategory === "types" && (
                <div className="space-y-2">
                  <label className="text-[10px] text-[#9E9D96] uppercase tracking-wider block font-bold">
                    Cast Target Data Type
                  </label>
                  <select
                    value={targetType}
                    onChange={(e) => setTargetType(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                  >
                    <option value="Float64">Float64 (pl.Float64) - 64-bit IEEE float</option>
                    <option value="Int64">Int64 (pl.Int64) - 64-bit signed integer</option>
                    <option value="String">String (pl.String) - UTF-8 string</option>
                    <option value="Boolean">Boolean (pl.Boolean) - Binary true/false</option>
                    <option value="Categorical">Categorical (pl.Categorical) - Low memory dictionary</option>
                  </select>
                </div>
              )}

              {opCategory === "dedup" && (
                <div className="p-3 bg-white rounded border border-[#E5E2D9] space-y-1">
                  <p className="text-[10px] text-[#73726C] font-bold">Deduplicate Dataset</p>
                  <p className="text-[9px] text-[#9E9D96]">
                    Purges exact duplicate records across all columns, retaining the first occurrence.
                  </p>
                </div>
              )}

              {opCategory === "filter" && (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[10px] text-[#9E9D96] uppercase block mb-1">Operator</label>
                    <select
                      value={filterOp}
                      onChange={(e) => setFilterOp(e.target.value)}
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2 py-1.5 text-xs"
                    >
                      <option value=">">&gt; Greater than</option>
                      <option value=">=">&gt;= Greater or equal</option>
                      <option value="<">&lt; Less than</option>
                      <option value="<=">&lt;= Less or equal</option>
                      <option value="==">== Equal to</option>
                      <option value="!=">!= Not equal to</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-[#9E9D96] uppercase block mb-1">Threshold</label>
                    <input
                      type="number"
                      value={filterThresh}
                      onChange={(e) => setFilterThresh(Number(e.target.value))}
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-2 py-1.5 text-xs font-mono"
                    />
                  </div>
                </div>
              )}

              {/* Real-time Polars Expression Snippet */}
              <div className="p-2.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded space-y-1">
                <span className="text-[9px] text-[#9E9D96] uppercase font-bold block">
                  Generated Polars Expression:
                </span>
                <code className="text-[10px] font-mono text-sky-400 block whitespace-pre-wrap break-all">
                  {liveBuilderExpression}
                </code>
              </div>

              <button
                type="submit"
                className="w-full flex items-center justify-center space-x-2 bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-[11px] uppercase tracking-wider py-2 rounded transition-all cursor-pointer border border-sky-500/20"
              >
                <Zap className="h-3.5 w-3.5" />
                <span>Append Step to Polars Recipe</span>
              </button>
            </form>
          </div>
        </div>

        {/* Right Col (7 cols): Active Recipe Stack & Execution Summary */}
        <div className="lg:col-span-7 space-y-4">
          
          {/* Active Recipe Header */}
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div className="flex items-center space-x-2">
                <FileCode className="h-4 w-4 text-sky-400" />
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Active Cleaning Pipeline ({pipelineSteps.filter(s => s.isActive).length}/{pipelineSteps.length} Steps)
                </h3>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setPipelineSteps([])}
                  className="text-[9px] font-mono uppercase text-[#9E9D96] hover:text-rose-400 transition-colors"
                >
                  Clear All
                </button>
              </div>
            </div>

            {/* Pipeline Step List */}
            {pipelineSteps.length === 0 ? (
              <div className="p-8 text-center bg-white border border-[#E5E2D9] rounded font-mono space-y-2">
                <Layers className="h-6 w-6 mx-auto text-[#73726C]" />
                <p className="text-xs text-[#73726C] font-bold">No Operations Configured</p>
                <p className="text-[10px] text-[#9E9D96] max-w-sm mx-auto">
                  Select an operation from the builder or load a prebuilt recipe above to assemble your Polars cleaning pipeline.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[340px] overflow-y-auto pr-1">
                {pipelineSteps.map((step, idx) => (
                  <div
                    key={step.id}
                    className={`p-3 rounded border font-mono text-[11px] transition-all flex items-center justify-between gap-3 ${
                      step.isActive
                        ? "bg-white border-[#E5E2D9] text-[#1F1E1D]"
                        : "bg-white border-[#E5E2D9] text-[#73726C] opacity-60"
                    }`}
                  >
                    <div className="flex items-center space-x-2.5 min-w-0">
                      <input
                        type="checkbox"
                        checked={step.isActive}
                        onChange={() => toggleStepActive(step.id)}
                        className="rounded border-[#E5E2D9] text-sky-500 focus:ring-0 cursor-pointer h-3.5 w-3.5"
                      />
                      
                      <div className="space-y-0.5 min-w-0">
                        <div className="flex items-center space-x-2 flex-wrap">
                          <span className="text-[9px] bg-white text-[#73726C] border border-[#E5E2D9] px-1 rounded font-bold">
                            #{idx + 1}
                          </span>
                          <span className="font-bold text-[#1F1E1D] truncate">{step.description}</span>
                        </div>
                        <code className="text-[9px] text-sky-400/90 block truncate max-w-[340px]">
                          {step.polarsExpression}
                        </code>
                      </div>
                    </div>

                    <div className="flex items-center space-x-1 flex-shrink-0">
                      <button
                        onClick={() => moveStep(idx, "up")}
                        disabled={idx === 0}
                        className="p-1 hover:bg-[#F0EEE6] text-[#9E9D96] hover:text-[#73726C] rounded disabled:opacity-30 cursor-pointer"
                        title="Move Up"
                      >
                        <ArrowUp className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => moveStep(idx, "down")}
                        disabled={idx === pipelineSteps.length - 1}
                        className="p-1 hover:bg-[#F0EEE6] text-[#9E9D96] hover:text-[#73726C] rounded disabled:opacity-30 cursor-pointer"
                        title="Move Down"
                      >
                        <ArrowDown className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => removeStep(step.id)}
                        className="p-1 hover:bg-rose-500/20 text-[#9E9D96] hover:text-rose-400 rounded transition-colors cursor-pointer"
                        title="Delete Step"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Execution Metrics Bar */}
            {report && (
              <div className="pt-2 border-t border-[#E5E2D9] space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
                  <div className="p-2.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                    <span className="text-[9px] text-[#9E9D96] block uppercase">Rows Cleaned</span>
                    <span className="text-sm font-bold text-[#1F1E1D]">
                      {report.cleanedRows.toLocaleString()}
                    </span>
                    <span className="text-[8px] text-[#9E9D96] block">
                      {report.initialRows - report.cleanedRows > 0 ? `-${report.initialRows - report.cleanedRows} dropped` : "Zero dropped"}
                    </span>
                  </div>

                  <div className="p-2.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                    <span className="text-[9px] text-[#9E9D96] block uppercase">Nulls Handled</span>
                    <span className="text-sm font-bold text-sky-400">
                      {report.nullsRemoved.toLocaleString()}
                    </span>
                    <span className="text-[8px] text-[#9E9D96] block">Imputed / Dropped</span>
                  </div>

                  <div className="p-2.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                    <span className="text-[9px] text-[#9E9D96] block uppercase">Outliers Clipped</span>
                    <span className="text-sm font-bold text-sky-400">
                      {report.outliersClipped.toLocaleString()}
                    </span>
                    <span className="text-[8px] text-[#9E9D96] block">Variance bounded</span>
                  </div>

                  <div className="p-2.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                    <span className="text-[9px] text-[#9E9D96] block uppercase">Quality Score</span>
                    <div className="flex items-center space-x-1">
                      <span className="text-sm font-bold text-sky-400">
                        {report.finalQualityScore}%
                      </span>
                      {report.finalQualityScore > report.initialQualityScore && (
                        <span className="text-[9px] text-sky-400 font-bold">
                          (+{report.finalQualityScore - report.initialQualityScore}%)
                        </span>
                      )}
                    </div>
                    <span className="text-[8px] text-[#9E9D96] block">Schema health</span>
                  </div>
                </div>

                {/* Workspace Commit Action */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 bg-sky-500/5 border border-sky-500/20 rounded">
                  <div className="space-y-0.5 text-left">
                    <p className="font-mono text-xs font-bold text-[#1F1E1D] flex items-center gap-1.5">
                      <CheckCircle className="h-3.5 w-3.5 text-sky-400" />
                      Apply Cleaned Dataset to Active Workspace
                    </p>
                    <p className="text-[10px] text-[#73726C] font-sans">
                      Updates EDA distributions, hypothesis testing matrices, and AutoML models with this cleaned data.
                    </p>
                  </div>

                  <div className="flex items-center space-x-2">
                    {isDataCleaned && (
                      <button
                        onClick={onRevertToOriginal}
                        className="px-3 py-1.5 bg-white hover:bg-[#F0EEE6] text-[#73726C] hover:text-[#1F1E1D] border border-[#D5D1C5] font-mono font-bold text-[10px] uppercase rounded transition-colors cursor-pointer"
                        title="Revert to uploaded raw dataset"
                      >
                        Revert Original
                      </button>
                    )}

                    <button
                      onClick={handleApplyCleanedToWorkspace}
                      className="px-4 py-1.5 bg-sky-600 hover:bg-sky-500 text-white border border-sky-500/30 font-mono font-bold text-[10px] uppercase rounded transition-colors cursor-pointer flex items-center space-x-1.5"
                    >
                      <Zap className="h-3 w-3" />
                      <span>{appliedNotification ? "Workspace Updated" : "Commit Clean Data"}</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Lower Section: Tabs for Preview Table, Generated Polars Python Script, and Throughput Benchmark */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded overflow-hidden">
        {/* Sub-header navigation */}
        <div className="border-b border-[#E5E2D9] bg-white px-4 py-2 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center space-x-1 font-mono text-[10px]">
            <button
              onClick={() => setViewMode("preview")}
              className={`px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5 cursor-pointer ${
                viewMode === "preview"
                  ? "bg-white text-sky-400 font-bold border border-[#E5E2D9]"
                  : "text-[#73726C] hover:text-[#1F1E1D]"
              }`}
            >
              <Table className="h-3.5 w-3.5" />
              <span>Cleaned Data Preview ({cleanedResult?.cleanedRows.length || 0} Rows)</span>
            </button>

            <button
              onClick={() => setViewMode("code")}
              className={`px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5 cursor-pointer ${
                viewMode === "code"
                  ? "bg-white text-sky-400 font-bold border border-[#E5E2D9]"
                  : "text-[#73726C] hover:text-[#1F1E1D]"
              }`}
            >
              <FileCode className="h-3.5 w-3.5" />
              <span>Export Polars Python Script (.py)</span>
            </button>

            <button
              onClick={() => setViewMode("benchmark")}
              className={`px-3 py-1.5 rounded transition-colors flex items-center space-x-1.5 cursor-pointer ${
                viewMode === "benchmark"
                  ? "bg-white text-sky-400 font-bold border border-[#E5E2D9]"
                  : "text-[#73726C] hover:text-[#1F1E1D]"
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              <span>Polars vs Pandas Throughput Diagnostics</span>
            </button>
          </div>

          {/* Quick Download Actions */}
          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadCsv}
              className="flex items-center space-x-1 font-mono text-[9px] uppercase font-bold text-[#73726C] hover:text-white bg-white hover:bg-[#F0EEE6] border border-[#E5E2D9] px-2.5 py-1 rounded transition-colors cursor-pointer"
            >
              <Download className="h-3 w-3" />
              <span>Download CSV</span>
            </button>
            <button
              onClick={handleDownloadScript}
              className="flex items-center space-x-1 font-mono text-[9px] uppercase font-bold text-sky-400 hover:text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 px-2.5 py-1 rounded transition-colors cursor-pointer"
            >
              <FileCode className="h-3 w-3" />
              <span>Download .py</span>
            </button>
          </div>
        </div>

        {/* View Mode 1: Cleaned Data Preview Table */}
        {viewMode === "preview" && (
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-mono text-[#73726C]">
              <span>Showing first 12 records of processed dataframe</span>
              <span className="text-[10px] text-[#9E9D96]">
                {allColNames.length} columns · {cleanedResult?.cleanedRows.length || 0} total rows
              </span>
            </div>

            <div className="border border-[#E5E2D9] rounded overflow-x-auto bg-white max-h-[420px]">
              <table className="w-full font-mono text-[10px] text-left border-collapse">
                <thead className="sticky top-0 bg-[#FAF9F5] backdrop-blur border-b border-[#E5E2D9] text-[#73726C] uppercase tracking-wider">
                  <tr>
                    <th className="py-2.5 px-3 border-r border-[#E5E2D9]">#</th>
                    {allColNames.map((col, idx) => (
                      <th key={idx} className="py-2.5 px-3 border-r border-[#E5E2D9] whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-900 text-[#73726C]">
                  {cleanedResult?.cleanedRows.slice(0, 12).map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-[#FAF9F5] transition-colors">
                      <td className="py-2 px-3 text-[#73726C] border-r border-[#E5E2D9]">{rIdx + 1}</td>
                      {allColNames.map((col, cIdx) => {
                        const val = row[col];
                        const origVal = activeRows[rIdx]?.[col];
                        const wasModified = origVal !== undefined && origVal !== val;
                        return (
                          <td 
                            key={cIdx} 
                            className={`py-2 px-3 border-r border-[#E5E2D9] whitespace-nowrap ${
                              wasModified ? "bg-sky-500/10 text-sky-300 font-bold" : ""
                            }`}
                            title={wasModified ? `Cleaned from: ${origVal}` : undefined}
                          >
                            {val !== undefined && val !== null ? String(val) : <span className="text-[#73726C]">null</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* View Mode 2: Export Polars Python Script */}
        {viewMode === "code" && (
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#73726C]">Standalone Executable Python Pipeline</span>
              <button
                onClick={handleCopyCode}
                className="flex items-center space-x-1.5 text-[#73726C] hover:text-white bg-white hover:bg-[#F0EEE6] border border-[#E5E2D9] px-3 py-1 rounded transition-colors cursor-pointer text-[10px] uppercase font-bold"
              >
                {copiedCode ? <Check className="h-3.5 w-3.5 text-sky-400" /> : <Copy className="h-3.5 w-3.5" />}
                <span>{copiedCode ? "Copied to Clipboard!" : "Copy Code"}</span>
              </button>
            </div>

            <pre className="p-4 bg-[#FAF9F5] border border-[#E5E2D9] rounded text-[11px] font-mono text-[#73726C] overflow-x-auto max-h-[500px] leading-relaxed selection:bg-sky-100">
              <code>{pythonScript}</code>
            </pre>
          </div>
        )}

        {/* View Mode 3: Benchmark & Architecture Diagnostics */}
        {viewMode === "benchmark" && (
          <div className="p-5 space-y-5 font-mono">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-[#FAF9F5] border border-[#E5E2D9] rounded space-y-2">
                <span className="text-[10px] text-[#9E9D96] uppercase font-bold">Estimated Throughput</span>
                <p className="text-xl font-black text-sky-400">
                  {report?.estimatedThroughputRowsPerSec.toLocaleString()} rows/s
                </p>
                <p className="text-[9px] text-[#9E9D96]">
                  Zero-copy Apache Arrow SIMD multithreaded evaluation across all CPU cores.
                </p>
              </div>

              <div className="p-4 bg-[#FAF9F5] border border-[#E5E2D9] rounded space-y-2">
                <span className="text-[10px] text-[#9E9D96] uppercase font-bold">Speedup vs Pandas</span>
                <p className="text-xl font-black text-sky-400">
                  {report?.estimatedPolarsSpeedupVsPandas}x Faster
                </p>
                <p className="text-[9px] text-[#9E9D96]">
                  Eliminates Python Global Interpreter Lock (GIL) and Pandas object overhead.
                </p>
              </div>

              <div className="p-4 bg-[#FAF9F5] border border-[#E5E2D9] rounded space-y-2">
                <span className="text-[10px] text-[#9E9D96] uppercase font-bold">Memory Footprint</span>
                <p className="text-xl font-black text-sky-400">
                  ~60% Lower RAM
                </p>
                <p className="text-[9px] text-[#9E9D96]">
                  Out-of-core streaming chunks prevent Out-Of-Memory crashes on multi-GB datasets.
                </p>
              </div>
            </div>

            {/* Architecture Explanation */}
            <div className="p-4 bg-white border border-[#E5E2D9] rounded space-y-3">
              <h4 className="text-xs font-bold text-[#1F1E1D] uppercase tracking-wider flex items-center gap-2">
                <Zap className="h-4 w-4 text-sky-400" />
                Why Polars Excels at Large Scale Dataset Cleaning
              </h4>
              <div className="grid md:grid-cols-2 gap-4 text-[11px] text-[#73726C] font-sans leading-relaxed">
                <div className="space-y-1">
                  <span className="font-bold font-mono text-[#1F1E1D] block text-xs">1. Lazy Execution Graphs (pl.scan_csv)</span>
                  <p>
                    Rather than eagerly allocating large arrays in memory, Polars builds a logical execution plan, optimizing filter pushdown, column pruning, and dead code elimination before computing any row.
                  </p>
                </div>
                <div className="space-y-1">
                  <span className="font-bold font-mono text-[#1F1E1D] block text-xs">2. Streaming Out-Of-Core Engine</span>
                  <p>
                    When processing datasets exceeding available machine RAM, <code className="font-mono text-sky-400">collect(streaming=True)</code> processes files in cache-coherent batches, enabling gigabyte-scale data pipelines without crashing.
                  </p>
                </div>
                <div className="space-y-1">
                  <span className="font-bold font-mono text-[#1F1E1D] block text-xs">3. Apache Arrow Vectorization</span>
                  <p>
                    Columnar memory layouts allow modern CPUs to execute SIMD (Single Instruction Multiple Data) vector instructions over thousands of nulls, numeric values, and text tokens simultaneously.
                  </p>
                </div>
                <div className="space-y-1">
                  <span className="font-bold font-mono text-[#1F1E1D] block text-xs">4. Declarative Expressions (pl.col)</span>
                  <p>
                    Expressions like <code className="font-mono text-sky-400">pl.col().fill_null().clip()</code> are naturally parallelized across all available processor threads without manual multiprocessing locks.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
