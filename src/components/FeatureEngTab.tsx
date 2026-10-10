import React, { useState, useMemo } from "react";
import { DatasetSummary, ColumnInfo } from "../types";
import { 
  Wand2, Calculator, AlertTriangle, CheckCircle, Trash2, HelpCircle, 
  Sparkles, Layers, RefreshCw, Cpu, TrendingUp, Info
} from "lucide-react";

interface FeatureEngTabProps {
  summary: DatasetSummary;
  rawData: any[] | null;
  engineeredFeatures: Array<{ name: string; formula: string }>;
  onAddFeature: (featureName: string, formula: string, updatedRows: any[]) => void;
  onRemoveFeature: (featureName: string) => void;
}

type OperationType = "unary" | "binary";

export default function FeatureEngTab({
  summary,
  rawData,
  engineeredFeatures,
  onAddFeature,
  onRemoveFeature
}: FeatureEngTabProps) {
  // Creator form state
  const [featureName, setFeatureName] = useState<string>("");
  const [operationType, setOperationType] = useState<OperationType>("binary");
  
  // Columns state
  const numericCols = useMemo(() => {
    return summary.columns.filter(c => c.type === "Numeric").map(c => c.name);
  }, [summary.columns]);

  const [unaryCol, setUnaryCol] = useState<string>(numericCols[0] || "");
  const [unaryOp, setUnaryOp] = useState<string>("ln");

  const [binaryColA, setBinaryColA] = useState<string>(numericCols[0] || "");
  const [binaryColB, setBinaryColB] = useState<string>(numericCols[1] || numericCols[0] || "");
  const [binaryOp, setBinaryOp] = useState<string>("+");

  const [formError, setFormError] = useState<string | null>(null);

  // Synchronize dropdowns if columns change
  React.useEffect(() => {
    if (numericCols.length > 0) {
      if (!numericCols.includes(unaryCol)) setUnaryCol(numericCols[0]);
      if (!numericCols.includes(binaryColA)) setBinaryColA(numericCols[0]);
      if (!numericCols.includes(binaryColB)) setBinaryColB(numericCols[1] || numericCols[0]);
    }
  }, [numericCols, unaryCol, binaryColA, binaryColB]);

  // Real-time calculation helper
  const calculateValue = (row: any, opType: OperationType): { val: number | null; error?: string } => {
    if (opType === "unary") {
      const val = Number(row[unaryCol]);
      if (isNaN(val) || row[unaryCol] === undefined || row[unaryCol] === null || row[unaryCol] === "") {
        return { val: null };
      }

      switch (unaryOp) {
        case "ln":
          if (val <= 0) return { val: null, error: "x ≤ 0 (undefined log)" };
          return { val: Math.log(val) };
        case "sqrt":
          if (val < 0) return { val: null, error: "x < 0 (undefined sqrt)" };
          return { val: Math.sqrt(val) };
        case "square":
          return { val: val * val };
        case "abs":
          return { val: Math.abs(val) };
        case "standardize":
          const colStats = summary.columns.find(c => c.name === unaryCol)?.stats;
          const mean = colStats?.mean ?? 0;
          const stdDev = colStats?.stdDev ?? 1;
          if (stdDev === 0) return { val: 0 };
          return { val: (val - mean) / stdDev };
        default:
          return { val: null };
      }
    } else {
      const valA = Number(row[binaryColA]);
      const valB = Number(row[binaryColB]);

      if (
        isNaN(valA) || row[binaryColA] === undefined || row[binaryColA] === null || row[binaryColA] === "" ||
        isNaN(valB) || row[binaryColB] === undefined || row[binaryColB] === null || row[binaryColB] === ""
      ) {
        return { val: null };
      }

      switch (binaryOp) {
        case "+":
          return { val: valA + valB };
        case "-":
          return { val: valA - valB };
        case "*":
          return { val: valA * valB };
        case "/":
          if (valB === 0) return { val: null, error: "Division by 0" };
          return { val: valA / valB };
        case "^":
          return { val: Math.pow(valA, valB) };
        default:
          return { val: null };
      }
    }
  };

  // Live Formula Preview
  const formulaPreviewText = useMemo(() => {
    if (operationType === "unary") {
      switch (unaryOp) {
        case "ln": return `ln([${unaryCol}])`;
        case "sqrt": return `sqrt([${unaryCol}])`;
        case "square": return `[${unaryCol}]²`;
        case "abs": return `|[${unaryCol}]|`;
        case "standardize": return `([${unaryCol}] - mean) / stdDev`;
        default: return "";
      }
    } else {
      return `[${binaryColA}] ${binaryOp} [${binaryColB}]`;
    }
  }, [operationType, unaryCol, unaryOp, binaryColA, binaryColB, binaryOp]);

  // Real-time Preview Rows (First 5 records)
  const previewRows = useMemo(() => {
    if (!rawData || rawData.length === 0) return [];
    return rawData.slice(0, 5).map((row, idx) => {
      const result = calculateValue(row, operationType);
      return {
        idx: idx + 1,
        sourceA: operationType === "unary" ? row[unaryCol] : row[binaryColA],
        sourceB: operationType === "unary" ? null : row[binaryColB],
        result: result.val !== null ? Number(result.val.toFixed(4)) : null,
        error: result.error
      };
    });
  }, [rawData, operationType, unaryCol, unaryOp, binaryColA, binaryColB, binaryOp]);

  // Feature Creation Handler
  const handleCreateFeature = (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    // Form validations
    const cleanName = featureName.trim().replace(/\s+/g, "_");
    if (!cleanName) {
      setFormError("Please enter a valid feature name.");
      return;
    }

    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(cleanName)) {
      setFormError("Feature name must contain only letters, numbers, and underscores, and cannot start with a number.");
      return;
    }

    // Check for collisions
    const columnExists = summary.columns.some(c => c.name.toLowerCase() === cleanName.toLowerCase());
    if (columnExists) {
      setFormError(`A column named '${cleanName}' already exists in the dataset.`);
      return;
    }

    if (!rawData || rawData.length === 0) {
      setFormError("No dataset available to compute features on.");
      return;
    }

    // Calculate feature values for all rows
    const updatedRows = rawData.map(row => {
      const result = calculateValue(row, operationType);
      return {
        ...row,
        [cleanName]: result.val !== null ? Number(result.val.toFixed(6)) : ""
      };
    });

    onAddFeature(cleanName, formulaPreviewText, updatedRows);
    
    // Clear name state
    setFeatureName("");
  };

  // Multicollinearity Detection Logic
  // Flag pairs where absolute Pearson r >= 0.8
  const collinearPairs = useMemo(() => {
    const pairs: Array<{
      colA: string;
      colB: string;
      r: number;
      absR: number;
      severity: "high" | "critical";
      recommendation: string;
    }> = [];

    const numCols = summary.columns.filter(c => c.type === "Numeric");

    for (let i = 0; i < numCols.length; i++) {
      for (let j = i + 1; j < numCols.length; j++) {
        const colA = numCols[i].name;
        const colB = numCols[j].name;

        // Fetch r from computed correlation matrix
        const r = summary.correlation[colA]?.[colB] ?? 0;
        const absR = Math.abs(r);

        if (absR >= 0.80) {
          const severity = absR >= 0.95 ? "critical" : "high";
          
          // Determine advice based on which is engineered vs original
          const isAEngineered = engineeredFeatures.some(f => f.name === colA);
          const isBEngineered = engineeredFeatures.some(f => f.name === colB);

          let recommendation = "";
          if (isAEngineered && !isBEngineered) {
            recommendation = `Drop '${colA}' if it duplicates the variance of original feature '${colB}'.`;
          } else if (isBEngineered && !isAEngineered) {
            recommendation = `Drop '${colB}' if it duplicates the variance of original feature '${colA}'.`;
          } else {
            // Compare missing rates or unique counts to pick the cleaner column
            const missingA = numCols[i].stats.missingCount;
            const missingB = numCols[j].stats.missingCount;
            if (missingA > missingB) {
              recommendation = `Consider keeping '${colB}' and dropping '${colA}' due to better data completeness.`;
            } else {
              recommendation = `Consider keeping '${colA}' and dropping '${colB}' due to better data completeness.`;
            }
          }

          pairs.push({
            colA,
            colB,
            r,
            absR,
            severity,
            recommendation
          });
        }
      }
    }

    // Sort by absolute correlation descending
    return pairs.sort((a, b) => b.absR - a.absR);
  }, [summary.columns, summary.correlation, engineeredFeatures]);

  return (
    <div id="feature-eng-tab-content" className="space-y-6">
      
      {/* Tab Welcome Header */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 relative overflow-hidden">
        <div className="absolute top-0 right-0 p-5 opacity-10 pointer-events-none">
          <Layers className="h-28 w-28 text-sky-400" />
        </div>
        <div className="max-w-2xl space-y-2">
          <div className="flex items-center space-x-2">
            <Wand2 className="h-5 w-5 text-sky-400" />
            <h2 className="font-mono font-bold text-[#1F1E1D] text-sm uppercase tracking-wider">Feature Engineering Workbench</h2>
          </div>
          <p className="text-xs text-[#73726C] leading-relaxed font-sans">
            Enhance predictive modeling power by constructing new features. Cleanly execute unary and binary mathematical operations across your dataset columns, preview calculations in real time, and audit your feature matrix for problematic multicollinearity automatically.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-5 gap-6 items-start">
        {/* Left Column: Creator & Engineered Features Log */}
        <div className="lg:col-span-3 space-y-6">
          
          {/* Feature Creator Form */}
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div className="flex items-center space-x-2">
                <Calculator className="h-4 w-4 text-sky-400" />
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">Construct New Feature</h3>
              </div>
              <span className="text-[9px] font-mono bg-sky-500/10 text-sky-400 border border-sky-500/20 px-2 py-0.5 rounded font-bold uppercase">
                Active Compiler
              </span>
            </div>

            {numericCols.length === 0 ? (
              <div className="p-4 bg-rose-500/5 border border-rose-500/15 text-rose-400 rounded text-xs font-mono text-center">
                This dataset does not contain any Numeric columns required for mathematical engineering.
              </div>
            ) : (
              <form onSubmit={handleCreateFeature} className="space-y-5">
                {/* Feature Name */}
                <div className="space-y-1.5 font-mono text-[11px]">
                  <label className="font-bold text-[#73726C] block uppercase tracking-wider">New Column Identifier</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. sqft_price_ratio, log_income"
                    value={featureName}
                    onChange={(e) => setFeatureName(e.target.value)}
                    className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded px-3 py-2 text-xs focus:ring-1 focus:ring-sky-500 outline-none font-semibold transition-colors"
                  />
                  <p className="text-[9px] text-[#9E9D96]">Lowercase, digits, and underscores only. Must start with a letter.</p>
                </div>

                {/* Operation Type Selector */}
                <div className="space-y-2 font-mono text-[11px]">
                  <label className="font-bold text-[#73726C] block uppercase tracking-wider">Operation Type</label>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => setOperationType("binary")}
                      className={`py-2 px-3 rounded border text-center transition-all cursor-pointer ${
                        operationType === "binary"
                          ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-extrabold"
                          : "bg-[#FAF9F5] border-[#E5E2D9] text-[#9E9D96] hover:text-[#73726C]"
                      }`}
                    >
                      Binary Operation (Col A & Col B)
                    </button>
                    <button
                      type="button"
                      onClick={() => setOperationType("unary")}
                      className={`py-2 px-3 rounded border text-center transition-all cursor-pointer ${
                        operationType === "unary"
                          ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-extrabold"
                          : "bg-[#FAF9F5] border-[#E5E2D9] text-[#9E9D96] hover:text-[#73726C]"
                      }`}
                    >
                      Unary Transform (Single Column)
                    </button>
                  </div>
                </div>

                {/* Formula Configurator */}
                <div className="p-4 bg-white border border-[#E5E2D9] rounded space-y-4 font-mono text-[11px]">
                  <span className="font-bold text-[#73726C] block uppercase tracking-wider text-[10px]">
                    Configure Mathematical Mapping
                  </span>

                  {operationType === "unary" ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Unary Column */}
                      <div className="space-y-1.5">
                        <label className="text-[#9E9D96] uppercase block tracking-wider text-[10px]">Source Column</label>
                        <select
                          value={unaryCol}
                          onChange={(e) => setUnaryCol(e.target.value)}
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                        >
                          {numericCols.map((c, idx) => (
                            <option key={idx} value={c}>{c}</option>
                          ))}
                        </select>
                      </div>

                      {/* Unary Operator */}
                      <div className="space-y-1.5">
                        <label className="text-[#9E9D96] uppercase block tracking-wider text-[10px]">Transform Function</label>
                        <select
                          value={unaryOp}
                          onChange={(e) => setUnaryOp(e.target.value)}
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                        >
                          <option value="ln">Natural Log: ln(x)</option>
                          <option value="sqrt">Square Root: sqrt(x)</option>
                          <option value="square">Square: x²</option>
                          <option value="abs">Absolute Value: |x|</option>
                          <option value="standardize">Standardize: (x - mean) / stdDev</option>
                        </select>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                      {/* Binary Column A */}
                      <div className="space-y-1.5">
                        <label className="text-[#9E9D96] uppercase block tracking-wider text-[10px]">Column A</label>
                        <select
                          value={binaryColA}
                          onChange={(e) => setBinaryColA(e.target.value)}
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                        >
                          {numericCols.map((c, idx) => (
                            <option key={idx} value={c}>{c}</option>
                          ))}
                        </select>
                      </div>

                      {/* Binary Operator */}
                      <div className="space-y-1.5">
                        <label className="text-[#9E9D96] uppercase block tracking-wider text-[10px]">Operator</label>
                        <select
                          value={binaryOp}
                          onChange={(e) => setBinaryOp(e.target.value)}
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer text-center"
                        >
                          <option value="+">+</option>
                          <option value="-">-</option>
                          <option value="*">×</option>
                          <option value="/">÷</option>
                          <option value="^">^ (power)</option>
                        </select>
                      </div>

                      {/* Binary Column B */}
                      <div className="space-y-1.5">
                        <label className="text-[#9E9D96] uppercase block tracking-wider text-[10px]">Column B</label>
                        <select
                          value={binaryColB}
                          onChange={(e) => setBinaryColB(e.target.value)}
                          className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
                        >
                          {numericCols.map((c, idx) => (
                            <option key={idx} value={c}>{c}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  )}

                  {/* Real-time Formula Preview Line */}
                  <div className="pt-2 border-t border-[#E5E2D9] flex items-center justify-between text-[11px]">
                    <span className="text-[#9E9D96]">Live Formula Mapping:</span>
                    <span className="bg-white border border-[#E5E2D9] px-3 py-1 rounded text-sky-400 font-bold font-mono">
                      {formulaPreviewText}
                    </span>
                  </div>
                </div>

                {/* Live Real-time Math Preview Table */}
                <div className="space-y-2">
                  <span className="font-mono text-[10px] font-bold text-[#73726C] uppercase tracking-wider block">
                    Pre-compilation Math Verification (First 5 Rows)
                  </span>
                  
                  <div className="border border-[#E5E2D9] rounded overflow-x-auto">
                    <table className="w-full font-mono text-[10px] text-left border-collapse">
                      <thead>
                        <tr className="bg-white border-b border-[#E5E2D9] text-[#9E9D96] font-bold uppercase tracking-wider">
                          <th className="py-2 px-3">Idx</th>
                          <th className="py-2 px-3">
                            {operationType === "unary" ? unaryCol : binaryColA}
                          </th>
                          {operationType === "binary" && (
                            <>
                              <th className="py-2 px-3 text-center">Op</th>
                              <th className="py-2 px-3">{binaryColB}</th>
                            </>
                          )}
                          <th className="py-2 px-3 text-right text-sky-400">Result ({featureName || "Preview"})</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-neutral-900 text-[#73726C]">
                        {previewRows.map((row, idx) => (
                          <tr key={idx} className="hover:bg-[#FAF9F5]">
                            <td className="py-2 px-3 text-[#73726C]">{row.idx}</td>
                            <td className="py-2 px-3">{row.sourceA !== undefined ? String(row.sourceA) : "NaN"}</td>
                            {operationType === "binary" && (
                              <>
                                <td className="py-2 px-3 text-center text-[#9E9D96]">{binaryOp}</td>
                                <td className="py-2 px-3">{row.sourceB !== undefined ? String(row.sourceB) : "NaN"}</td>
                              </>
                            )}
                            <td className="py-2 px-3 text-right font-bold text-sky-400">
                              {row.error ? (
                                <span className="text-rose-500 italic font-normal text-[9px] inline-flex items-center gap-1" title={row.error}>
                                  <AlertTriangle className="h-2.5 w-2.5" /> {row.error}
                                </span>
                              ) : row.result !== null ? (
                                row.result
                              ) : (
                                <span className="text-[#73726C]">null</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Form Actions / Errors */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pt-2">
                  <div>
                    {formError && (
                      <p className="text-[10px] text-rose-400 bg-rose-500/10 border border-rose-500/20 px-3 py-1 rounded font-mono font-bold leading-relaxed inline-flex items-center gap-1.5">
                        <AlertTriangle className="h-3 w-3 shrink-0" /> {formError}
                      </p>
                    )}
                  </div>
                  
                  <button
                    type="submit"
                    className="flex items-center space-x-2 bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-[11px] uppercase tracking-wider px-5 py-2 rounded transition-all duration-150 cursor-pointer border border-sky-500/20 self-end"
                  >
                    <Wand2 className="h-3.5 w-3.5" />
                    <span>Compile & Insert Feature</span>
                  </button>
                </div>
              </form>
            )}
          </div>

          {/* Active Engineered Features Log */}
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div className="flex items-center space-x-2">
                <Layers className="h-4 w-4 text-sky-400" />
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">Engineered Feature Log</h3>
              </div>
              <span className="text-[10px] font-mono font-bold bg-[#FAF9F5] text-[#73726C] px-2 py-0.5 rounded border border-[#E5E2D9]">
                Count: {engineeredFeatures.length}
              </span>
            </div>

            {engineeredFeatures.length === 0 ? (
              <div className="p-4 bg-white border border-[#E5E2D9] text-[#9E9D96] rounded text-xs font-mono text-center py-6">
                No custom mathematical features have been compiled yet. Use the constructor form above to insert features.
              </div>
            ) : (
              <div className="divide-y divide-neutral-900 bg-white rounded border border-[#E5E2D9]">
                {engineeredFeatures.map((feat, idx) => (
                  <div key={idx} className="p-4 flex items-center justify-between font-mono text-[11px] hover:bg-[#FAF9F5] transition-colors">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="w-1.5 h-1.5 bg-sky-500 rounded-full inline-block" />
                        <span className="font-bold text-[#1F1E1D] text-xs text-sky-400">{feat.name}</span>
                        <span className="text-[9px] bg-sky-500/10 text-sky-400 border border-sky-500/25 px-1.5 py-0.2 rounded font-bold uppercase scale-90">
                          COMPILED
                        </span>
                      </div>
                      <p className="text-[#9E9D96] text-[10px]">
                        Formula: <span className="text-[#73726C] font-semibold">{feat.formula}</span>
                      </p>
                    </div>

                    <button
                      onClick={() => onRemoveFeature(feat.name)}
                      className="flex items-center space-x-1 border border-rose-500/10 hover:border-rose-500/30 bg-rose-500/5 hover:bg-rose-500/10 text-rose-400 font-bold uppercase text-[9px] px-2.5 py-1.5 rounded transition-colors cursor-pointer"
                      title={`Remove '${feat.name}' column from raw dataset`}
                    >
                      <Trash2 className="h-3 w-3" />
                      <span>Remove</span>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Multicollinearity Detector */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4 text-sky-500" />
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Multicollinearity Audit
                </h3>
              </div>
              
              {collinearPairs.length > 0 ? (
                <span className="text-[9px] font-mono bg-rose-500/10 text-rose-500 border border-rose-500/20 px-2 py-0.5 rounded uppercase font-semibold inline-flex items-center gap-1">
                  <AlertTriangle className="h-2.5 w-2.5" /> High Collinearity ({collinearPairs.length})
                </span>
              ) : (
                <span className="text-[9px] font-mono bg-sky-500/10 text-sky-600 border border-sky-500/20 px-2 py-0.5 rounded uppercase font-semibold inline-flex items-center gap-1">
                  <CheckCircle className="h-2.5 w-2.5" /> Low Collinearity
                </span>
              )}
            </div>

            <p className="text-[11px] text-[#73726C] leading-normal">
              High collinearity ($|r| \ge 0.80$) among numeric features indicates high statistical redundancy, which can inflates variance of coefficient estimates in regression and classification models.
            </p>

            {collinearPairs.length > 0 ? (
              <div className="space-y-3">
                {collinearPairs.map((pair, idx) => (
                  <div 
                    key={idx} 
                    className={`p-3.5 rounded border font-mono text-[11px] space-y-2.5 relative ${
                      pair.severity === "critical"
                        ? "bg-rose-500/5 border-rose-500/15"
                        : "bg-sky-500/5 border-sky-500/15"
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="space-y-0.5">
                        <div className="flex items-center space-x-1.5 flex-wrap">
                          <span className="font-bold text-[#1F1E1D]">{pair.colA}</span>
                          <span className="text-[#9E9D96]">↔</span>
                          <span className="font-bold text-[#1F1E1D]">{pair.colB}</span>
                        </div>
                        <p className="text-[9px] text-[#9E9D96]">
                          Pearson Correlation Coefficient
                        </p>
                      </div>

                      <div className="text-right">
                        <span className={`text-xs font-bold ${
                          pair.severity === "critical" ? "text-rose-400" : "text-sky-400"
                        }`}>
                          r = {pair.r.toFixed(3)}
                        </span>
                        <span className={`block text-[8px] font-bold uppercase tracking-wider mt-0.5 ${
                          pair.severity === "critical" ? "text-rose-500" : "text-sky-500"
                        }`}>
                          {pair.severity}
                        </span>
                      </div>
                    </div>

                    {/* Correlation Intensity Meter */}
                    <div className="space-y-1">
                      <div className="w-full bg-[#FAF9F5] h-1.5 rounded-full overflow-hidden border border-[#E5E2D9] flex justify-center">
                        <div 
                          className={`h-full rounded-full ${
                            pair.severity === "critical" ? "bg-rose-500" : "bg-sky-500"
                          }`}
                          style={{ width: `${pair.absR * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Recommendation Box */}
                    <div className="p-2 bg-white rounded border border-[#E5E2D9] flex items-start space-x-1.5 text-[10px] text-[#73726C] leading-relaxed">
                      <Info className="h-3.5 w-3.5 text-[#9E9D96] flex-shrink-0 mt-0.5" />
                      <p>{pair.recommendation}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 bg-sky-500/5 border border-sky-500/15 text-sky-400 rounded py-6 space-y-1.5 text-center font-mono">
                <CheckCircle className="h-5 w-5 text-sky-400 mx-auto" />
                <p className="text-xs font-bold">No High Collinearity Detected</p>
                <p className="text-[9px] text-[#9E9D96] leading-normal max-w-sm mx-auto">
                  All distinct numerical columns exhibit Pearson $|r| &lt; 0.80$. Your dataset is well-conditioned for stable machine learning model fitting.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
