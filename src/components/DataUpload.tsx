import React, { useState, useRef } from "react";
import * as XLSX from "xlsx";
import { 
  Upload, FileSpreadsheet, Loader2, Database, Zap, 
  Layers, TrendingUp, Check, ChevronDown, Sparkles, FileText, ArrowRight,
  AlertTriangle, ShieldAlert, Wrench, X, CheckCircle, AlertCircle, RefreshCw,
  Flower2, Home, Ship
} from "lucide-react";
import { SAMPLE_DATASETS } from "../data/samples";
import { BatchProcessingReport } from "../types";
import { processDatasetInBatches } from "../utils/batchProcessor";

interface DataUploadProps {
  onDataLoaded: (
    fileName: string, 
    rawData: any[], 
    dataQualityScore: number, 
    initialTab?: string,
    batchReport?: BatchProcessingReport
  ) => void;
  userGreetingName?: string;
}

export interface CsvValidationIssue {
  type: "missing_header" | "duplicate_header" | "jagged_row" | "blank_row";
  severity: "warning" | "error";
  title: string;
  description: string;
  count: number;
  details?: string[];
}

export interface CsvValidationReport {
  fileName: string;
  totalRawRows: number;
  expectedColCount: number;
  rawHeaders: string[];
  cleanHeaders: string[];
  issues: CsvValidationIssue[];
  hasWarnings: boolean;
  sanitizedRows: any[];
  rawRows: any[];
  qualityScore: number;
  targetMode?: string;
}

export function calculateQualityScore(rawData: any[]): number {
  if (!rawData || rawData.length === 0) return 100;
  
  const headers = Object.keys(rawData[0]);
  if (headers.length === 0) return 100;
  
  let totalCells = rawData.length * headers.length;
  let missingCount = 0;
  let totalConsistency = 0;
  
  headers.forEach((header) => {
    let colNumericCount = 0;
    let colDateCount = 0;
    let colTextCount = 0;
    let colCategoricalCount = 0;
    let colNonEmptyCount = 0;
    
    rawData.forEach((row) => {
      const val = row[header];
      if (val === undefined || val === null || val === "" || String(val).trim() === "") {
        missingCount++;
      } else {
        colNonEmptyCount++;
        const strVal = String(val).trim();
        
        const isNum = !isNaN(Number(strVal)) && !isNaN(parseFloat(strVal));
        const isDate = (strVal.includes("-") || strVal.includes("/")) && !isNaN(Date.parse(strVal));
        const isText = strVal.length > 50;
        
        if (isNum) {
          colNumericCount++;
        } else if (isDate) {
          colDateCount++;
        } else if (isText) {
          colTextCount++;
        } else {
          colCategoricalCount++;
        }
      }
    });
    
    if (colNonEmptyCount > 0) {
      const maxCount = Math.max(colNumericCount, colDateCount, colTextCount, colCategoricalCount);
      const colConsistency = maxCount / colNonEmptyCount;
      totalConsistency += colConsistency;
    } else {
      totalConsistency += 1.0;
    }
  });
  
  const missingRatio = missingCount / totalCells;
  const missingScore = (1 - missingRatio) * 100;
  const consistencyScore = (totalConsistency / headers.length) * 100;
  
  const finalScore = Math.round(0.5 * missingScore + 0.5 * consistencyScore);
  return Math.max(0, Math.min(100, finalScore));
}

// Pre-processing validator function that inspects raw 2D tabular rows
export function validateCsvStructure(
  rawGrid: any[][], 
  fileName: string, 
  targetMode?: string
): CsvValidationReport {
  if (!rawGrid || rawGrid.length === 0) {
    throw new Error("The uploaded file does not contain any readable tabular data.");
  }

  const rawHeadersRow = rawGrid[0] || [];
  const rawDataRows = rawGrid.slice(1);
  const issues: CsvValidationIssue[] = [];

  // 1. Validate Headers
  const cleanHeaders: string[] = [];
  const seenHeaderCounts: Record<string, number> = {};
  const missingHeaderIndices: number[] = [];
  const duplicateHeaderNames: string[] = [];

  rawHeadersRow.forEach((h: any, idx: number) => {
    let headerStr = h !== undefined && h !== null ? String(h).trim() : "";
    const isBlank = headerStr === "" || headerStr.toLowerCase().startsWith("__empty");
    
    if (isBlank) {
      missingHeaderIndices.push(idx + 1);
      headerStr = `Column_${idx + 1}`;
    }

    const lower = headerStr.toLowerCase();
    if (seenHeaderCounts[lower]) {
      seenHeaderCounts[lower]++;
      duplicateHeaderNames.push(headerStr);
      headerStr = `${headerStr}_${seenHeaderCounts[lower]}`;
    } else {
      seenHeaderCounts[lower] = 1;
    }

    cleanHeaders.push(headerStr);
  });

  if (missingHeaderIndices.length > 0) {
    issues.push({
      type: "missing_header",
      severity: "warning",
      title: "Missing or Unnamed Column Headers",
      description: `Detected ${missingHeaderIndices.length} column(s) without valid headers (columns: ${missingHeaderIndices.slice(0, 5).join(", ")}${missingHeaderIndices.length > 5 ? "..." : ""}).`,
      count: missingHeaderIndices.length,
      details: missingHeaderIndices.map(col => `Column #${col} was blank and auto-labeled 'Column_${col}'.`)
    });
  }

  if (duplicateHeaderNames.length > 0) {
    issues.push({
      type: "duplicate_header",
      severity: "warning",
      title: "Duplicate Column Header Names",
      description: `Found ${duplicateHeaderNames.length} duplicate header(s): ${Array.from(new Set(duplicateHeaderNames)).slice(0, 4).join(", ")}.`,
      count: duplicateHeaderNames.length,
      details: duplicateHeaderNames.map(name => `Column '${name}' appears multiple times.`)
    });
  }

  // 2. Validate Row Integrity & Detect Corrupted / Jagged Rows
  const expectedColCount = cleanHeaders.length;
  let blankRowsCount = 0;
  let jaggedRowsCount = 0;
  const jaggedRowDetails: string[] = [];
  const sanitizedRows: any[] = [];
  const rawRows: any[] = [];

  rawDataRows.forEach((row: any[], rIdx: number) => {
    const rowNum = rIdx + 2; // +1 for 0-index, +1 for header line
    
    // Check if row is completely blank
    const isBlank = !row || row.length === 0 || row.every(cell => cell === undefined || cell === null || String(cell).trim() === "");
    if (isBlank) {
      blankRowsCount++;
      return; // Skip blank row in sanitized rows
    }

    // Check if row is jagged (has fewer or more elements than headers)
    const isJagged = row.length !== expectedColCount;
    if (isJagged) {
      jaggedRowsCount++;
      if (jaggedRowDetails.length < 5) {
        jaggedRowDetails.push(`Line #${rowNum}: Expected ${expectedColCount} values, but row contained ${row.length} values.`);
      }
    }

    // Build raw row object
    const rawObj: Record<string, any> = {};
    const sanitizedObj: Record<string, any> = {};

    cleanHeaders.forEach((colName, cIdx) => {
      const val = row[cIdx];
      rawObj[colName] = val !== undefined ? val : "";
      sanitizedObj[colName] = val !== undefined && val !== null ? val : "";
    });

    rawRows.push(rawObj);
    sanitizedRows.push(sanitizedObj);
  });

  if (jaggedRowsCount > 0) {
    issues.push({
      type: "jagged_row",
      severity: "warning",
      title: "Corrupted or Jagged Rows Detected",
      description: `Found ${jaggedRowsCount} row(s) where column field counts do not align with the header schema.`,
      count: jaggedRowsCount,
      details: jaggedRowDetails
    });
  }

  if (blankRowsCount > 0) {
    issues.push({
      type: "blank_row",
      severity: "warning",
      title: "Empty or Ghost Rows",
      description: `Detected ${blankRowsCount} completely empty row(s) that will skew statistical evaluations.`,
      count: blankRowsCount,
      details: [`${blankRowsCount} empty trailing row(s) filtered from processing.`]
    });
  }

  const qualityScore = calculateQualityScore(sanitizedRows.length > 0 ? sanitizedRows : rawRows);

  return {
    fileName,
    totalRawRows: rawDataRows.length,
    expectedColCount,
    rawHeaders: rawHeadersRow.map((h: any) => String(h || "")),
    cleanHeaders,
    issues,
    hasWarnings: issues.length > 0,
    sanitizedRows: sanitizedRows.length > 0 ? sanitizedRows : rawRows,
    rawRows,
    qualityScore,
    targetMode
  };
}

export default function DataUpload({ onDataLoaded, userGreetingName }: DataUploadProps) {
  const [activeInputTab, setActiveInputTab] = useState<"file" | "paste">("file");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState("");
  const [selectedMode, setSelectedMode] = useState<"overview" | "eda" | "stats" | "polars" | "feature-eng" | "ml">("overview");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Pre-processing CSV Structure Validation Modal state
  const [validationReport, setValidationReport] = useState<CsvValidationReport | null>(null);
  const [autoFixHeaders, setAutoFixHeaders] = useState(true);
  const [autoPurgeBlanks, setAutoPurgeBlanks] = useState(true);
  const [autoPadJagged, setAutoPadJagged] = useState(true);

  // Batch Chunk Processing configuration
  const [enableBatchProcessing, setEnableBatchProcessing] = useState<boolean>(true);
  const [batchChunkSize, setBatchChunkSize] = useState<number>(500);
  const [batchProgress, setBatchProgress] = useState<{
    currentChunk: number;
    totalChunks: number;
    percentage: number;
    rowsProcessed: number;
    estimatedTokensSaved: number;
  } | null>(null);

  const modeOptions = [
    { id: "overview", label: "Overview & Matrix", desc: "Profile schema, null rates, and metrics" },
    { id: "eda", label: "Exploratory EDA", desc: "Correlations, scatter plots, and distributions" },
    { id: "stats", label: "Hypothesis Tests", desc: "T-Test, ANOVA, Chi-Square, Pearson r" },
    { id: "polars", label: "Polars Vector Clean", desc: "Vectorized out-of-core pipeline cleaning" },
    { id: "feature-eng", label: "Feature Engineering", desc: "Formula editor and math transformations" },
    { id: "ml", label: "AutoML Trainer", desc: "Benchmark estimators and cross-validation" }
  ];

  // Universal loader routing through batch chunking when enabled
  const finalizeAndLoadData = async (fName: string, rows: any[], qualityScore: number, mode: string) => {
    if (enableBatchProcessing && rows.length > batchChunkSize) {
      setIsLoading(true);
      try {
        const { datasetSummary, batchReport } = await processDatasetInBatches({
          rows,
          fileName: fName,
          chunkSize: batchChunkSize,
          onProgress: (p) => setBatchProgress(p)
        });
        onDataLoaded(fName, rows, datasetSummary.dataQualityScore ?? qualityScore, mode, batchReport);
      } catch (err: any) {
        console.error("Batch processing error, falling back to standard load:", err);
        onDataLoaded(fName, rows, qualityScore, mode);
      } finally {
        setIsLoading(false);
        setBatchProgress(null);
      }
    } else {
      onDataLoaded(fName, rows, qualityScore, mode);
    }
  };

  // Primary file parser: inspects 2D grid and triggers pre-processing check
  const processFile = (file: File) => {
    setIsLoading(true);
    setError(null);
    setValidationReport(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: "array" });
        
        if (workbook.SheetNames.length === 0) {
          throw new Error("The uploaded workbook does not contain any sheets.");
        }

        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Extract raw 2D grid to validate headers and row alignment
        const rawGrid: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });

        if (!rawGrid || rawGrid.length === 0 || rawGrid.every(r => r.length === 0)) {
          throw new Error("The uploaded spreadsheet is empty.");
        }

        // Run structural validation pre-processing step
        const report = validateCsvStructure(rawGrid, file.name, selectedMode);

        if (report.hasWarnings) {
          // Open Pre-processing Validation & Warnings step
          setValidationReport(report);
        } else {
          // 100% clean dataset -> Proceed directly to workbench via batch handler
          finalizeAndLoadData(file.name, report.sanitizedRows, report.qualityScore, selectedMode);
        }
      } catch (err: any) {
        console.error("CSV parse error:", err);
        setError(err.message || "Failed to parse file. Please verify CSV or Excel format.");
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setError("Error reading file from disk.");
      setIsLoading(false);
    };

    reader.readAsArrayBuffer(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const loadSampleDataset = (sampleName: string, csvContent: string, targetMode?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const workbook = XLSX.read(csvContent, { type: "string" });
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const rawGrid: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
      const report = validateCsvStructure(rawGrid, `${sampleName}.csv`, targetMode || selectedMode);

      finalizeAndLoadData(`${sampleName}.csv`, report.sanitizedRows, report.qualityScore, targetMode || selectedMode);
    } catch (err: any) {
      setError("Failed to load sample dataset.");
    } finally {
      setIsLoading(false);
    }
  };

  // Parse pasted CSV/TSV text with structural check
  const handleTextSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!textInput.trim()) return;

    setIsLoading(true);
    setError(null);
    try {
      const workbook = XLSX.read(textInput.trim(), { type: "string" });
      const worksheet = workbook.Sheets[workbook.SheetNames[0]];
      const rawGrid: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
      
      if (!rawGrid || rawGrid.length === 0) throw new Error("No tabular data detected in pasted input.");

      const report = validateCsvStructure(rawGrid, "pasted_dataset.csv", selectedMode);

      if (report.hasWarnings) {
        setValidationReport(report);
      } else {
        finalizeAndLoadData("pasted_dataset.csv", report.sanitizedRows, report.qualityScore, selectedMode);
      }
    } catch (err: any) {
      setError("Could not parse pasted tabular text as CSV/Excel. Try dropping a file instead.");
    } finally {
      setIsLoading(false);
    }
  };

  // Confirmation actions from Pre-Processing Validation screen
  const handleConfirmCleanedLoad = () => {
    if (!validationReport) return;
    const dataToLoad = autoFixHeaders && autoPadJagged ? validationReport.sanitizedRows : validationReport.rawRows;
    const score = calculateQualityScore(dataToLoad);
    finalizeAndLoadData(validationReport.fileName, dataToLoad, score, validationReport.targetMode || selectedMode);
    setValidationReport(null);
  };

  const handleConfirmRawLoad = () => {
    if (!validationReport) return;
    finalizeAndLoadData(validationReport.fileName, validationReport.rawRows, validationReport.qualityScore, validationReport.targetMode || selectedMode);
    setValidationReport(null);
  };

  const handleDismissReport = () => {
    setValidationReport(null);
  };

  return (
    <div id="data-uploader-container" className="max-w-4xl mx-auto space-y-8 py-6 w-full relative">
      
      {/* ========================================================================= */}
      {/* REAL-TIME BATCH PROCESSING OVERLAY */}
      {/* ========================================================================= */}
      {batchProgress && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 shadow-2xl max-w-md w-full space-y-4 font-mono text-xs animate-in fade-in zoom-in-95">
            <div className="flex items-center space-x-3 border-b border-[#E5E2D9] pb-3">
              <div className="w-9 h-9 rounded-xl bg-purple-50 border border-purple-200 text-purple-700 flex items-center justify-center flex-shrink-0">
                <Layers className="h-4 w-4" />
              </div>
              <div>
                <h4 className="font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Batch Processing Dataset
                </h4>
                <p className="text-[11px] text-[#73726C]">
                  Chunk Size: {batchChunkSize.toLocaleString()} rows / micro-batch
                </p>
              </div>
            </div>

            <div className="space-y-2.5">
              <div className="flex justify-between text-[11px]">
                <span className="text-[#1F1E1D] font-bold">
                  Processing Chunk {batchProgress.currentChunk} of {batchProgress.totalChunks}
                </span>
                <span className="text-purple-700 font-bold">
                  {batchProgress.percentage}%
                </span>
              </div>

              <div className="w-full h-2.5 bg-[#E5E2D9] rounded-full overflow-hidden">
                <div 
                  className="h-full bg-purple-600 transition-all duration-150 rounded-full"
                  style={{ width: `${batchProgress.percentage}%` }}
                />
              </div>

              <div className="flex justify-between text-[10px] text-[#73726C] pt-1">
                <span>Ingested: {batchProgress.rowsProcessed.toLocaleString()} rows</span>
                <span className="text-emerald-700 font-bold">
                  Saved ~{batchProgress.estimatedTokensSaved.toLocaleString()} tokens
                </span>
              </div>
            </div>

            <div className="p-2.5 rounded bg-purple-50/60 border border-purple-200/60 text-[10px] text-purple-900 leading-relaxed">
              Incremental chunking enforces strict compliance with LLM token quotas, eliminates memory bottlenecks, and builds robust running statistical digests.
            </div>
          </div>
        </div>
      )}
      
      {/* ========================================================================= */}
      {/* CSV PRE-PROCESSING STRUCTURAL VALIDATION MODAL */}
      {/* ========================================================================= */}
      {validationReport && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-[#E5E2D9] rounded-2xl shadow-xl max-w-2xl w-full overflow-hidden flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150">
            
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-[#E5E2D9] bg-[#FAF9F5] flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-700 flex items-center justify-center">
                  <ShieldAlert className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="font-serif text-base font-semibold text-[#1F1E1D]">
                    CSV Structure Validation & Pre-Processing
                  </h3>
                  <p className="text-xs text-[#73726C] font-mono">
                    File: {validationReport.fileName} ({validationReport.totalRawRows.toLocaleString()} rows inspected)
                  </p>
                </div>
              </div>

              <button
                onClick={handleDismissReport}
                className="p-1 text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#E8E5DC] transition-colors cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Modal Body: Detected Warnings */}
            <div className="p-6 space-y-5 overflow-y-auto flex-1 text-xs">
              
              <div className="p-3 bg-amber-50/70 border border-amber-200 rounded-xl text-amber-900 space-y-1">
                <div className="flex items-center space-x-2 font-semibold">
                  <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0" />
                  <span>Structural Warnings Detected Before Ingestion</span>
                </div>
                <p className="text-[11px] text-amber-800 leading-relaxed">
                  We scanned the dataset layout for schema consistency. Loading unformatted or corrupted rows can cause parsing failures or skewed statistics during downstream ML training.
                </p>
              </div>

              {/* Issues List */}
              <div className="space-y-3">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#73726C] font-semibold block">
                  Detailed Findings ({validationReport.issues.length} Issues):
                </span>

                {validationReport.issues.map((issue, idx) => (
                  <div key={idx} className="p-3.5 rounded-xl border border-[#E5E2D9] bg-[#FAF9F5] space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="h-3.5 w-3.5 text-amber-600" />
                        <h4 className="font-medium text-xs text-[#1F1E1D]">{issue.title}</h4>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-100 text-amber-800 border border-amber-200 font-semibold">
                        {issue.count} {issue.count === 1 ? "Occurrence" : "Occurrences"}
                      </span>
                    </div>
                    
                    <p className="text-xs text-[#73726C] leading-relaxed">
                      {issue.description}
                    </p>

                    {issue.details && issue.details.length > 0 && (
                      <div className="p-2 bg-white rounded-lg border border-[#E5E2D9] space-y-1 font-mono text-[10px] text-[#73726C]">
                        {issue.details.slice(0, 4).map((d, dIdx) => (
                          <div key={dIdx} className="truncate">• {d}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Pre-Processing Remediation Toggles */}
              <div className="p-4 rounded-xl bg-[#F0F9FF] border border-[#BAE6FD] space-y-3">
                <div className="flex items-center space-x-2 text-[#0284C7] font-semibold">
                  <Wrench className="h-4 w-4" />
                  <span>Automated Pre-Processing Sanitization:</span>
                </div>

                <div className="space-y-2 text-xs text-[#1F1E1D]">
                  <label className="flex items-center space-x-2.5 cursor-pointer">
                    <input 
                      type="checkbox"
                      checked={autoFixHeaders}
                      onChange={(e) => setAutoFixHeaders(e.target.checked)}
                      className="rounded text-[#0284C7] focus:ring-[#0284C7] cursor-pointer"
                    />
                    <span>Auto-assign default labels for blank headers (e.g. <code>Column_3</code>) and disambiguate duplicates</span>
                  </label>

                  <label className="flex items-center space-x-2.5 cursor-pointer">
                    <input 
                      type="checkbox"
                      checked={autoPadJagged}
                      onChange={(e) => setAutoPadJagged(e.target.checked)}
                      className="rounded text-[#0284C7] focus:ring-[#0284C7] cursor-pointer"
                    />
                    <span>Pad jagged/corrupted rows with null values to maintain rectangular schema</span>
                  </label>

                  <label className="flex items-center space-x-2.5 cursor-pointer">
                    <input 
                      type="checkbox"
                      checked={autoPurgeBlanks}
                      onChange={(e) => setAutoPurgeBlanks(e.target.checked)}
                      className="rounded text-[#0284C7] focus:ring-[#0284C7] cursor-pointer"
                    />
                    <span>Strip completely empty/blank rows from the dataframe</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Modal Actions */}
            <div className="px-6 py-3.5 border-t border-[#E5E2D9] bg-[#FAF9F5] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <button
                type="button"
                onClick={handleDismissReport}
                className="px-3.5 py-2 text-xs font-medium text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#E8E5DC] rounded-lg transition-colors cursor-pointer"
              >
                Choose Different File
              </button>

              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={handleConfirmRawLoad}
                  className="px-3 py-2 text-xs font-medium text-[#73726C] hover:text-[#1F1E1D] border border-[#E5E2D9] rounded-lg hover:bg-white transition-colors cursor-pointer"
                  title="Load without applying auto-healing fixes"
                >
                  Load As-Is (Raw)
                </button>

                <button
                  type="button"
                  onClick={handleConfirmCleanedLoad}
                  className="px-4 py-2 text-xs font-medium bg-[#0284C7] hover:bg-[#0369A1] text-white rounded-lg shadow-xs transition-colors cursor-pointer flex items-center space-x-1.5"
                >
                  <Check className="h-3.5 w-3.5" />
                  <span>Sanitize & Load Dataset</span>
                </button>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Studio Header */}
      <div className="text-center space-y-2.5">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#F0F9FF] border border-[#BAE6FD] rounded-full text-xs text-[#0284C7] font-medium">
          <Database className="h-3.5 w-3.5" />
          <span>AutoML & Data Science Studio</span>
        </div>
        <h2 className="font-serif text-3xl sm:text-4xl font-normal tracking-tight text-[#1F1E1D]">
          Import & Profile Dataset
        </h2>
        <p className="text-sm text-[#73726C] max-w-xl mx-auto leading-relaxed">
          Upload any CSV or Excel spreadsheet to compute distributions, verify hypotheses, wrangle out-of-core data with Polars, and train machine learning estimators.
        </p>
      </div>

      {/* Main Dataset Ingestion Workbench Card */}
      <div className="bg-white border border-[#E5E2D9] rounded-2xl shadow-sm overflow-hidden">
        
        {/* Ingestion Method Tabs: File Upload vs Raw Text Paste */}
        <div className="flex items-center justify-between px-6 pt-4 border-b border-[#E5E2D9] bg-[#FAF9F5]">
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveInputTab("file")}
              className={`px-4 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer flex items-center space-x-2 ${
                activeInputTab === "file"
                  ? "border-[#0284C7] text-[#0284C7] font-semibold"
                  : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
              }`}
            >
              <Upload className="h-3.5 w-3.5" />
              <span>File Upload</span>
            </button>
            <button
              onClick={() => setActiveInputTab("paste")}
              className={`px-4 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer flex items-center space-x-2 ${
                activeInputTab === "paste"
                  ? "border-[#0284C7] text-[#0284C7] font-semibold"
                  : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
              }`}
            >
              <FileText className="h-3.5 w-3.5" />
              <span>Paste Tabular Data</span>
            </button>
          </div>

          {/* Initial Focus View Selector */}
          <div className="hidden sm:flex items-center space-x-2 pb-2">
            <span className="text-[11px] text-[#73726C]">Initial view:</span>
            <select
              value={selectedMode}
              onChange={(e) => setSelectedMode(e.target.value as any)}
              className="bg-white border border-[#E5E2D9] rounded-lg px-2.5 py-1 text-xs text-[#1F1E1D] font-medium outline-none cursor-pointer"
            >
              {modeOptions.map(opt => (
                <option key={opt.id} value={opt.id}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Tab 1: Drag & Drop File Zone */}
        {activeInputTab === "file" ? (
          <div className="p-6 sm:p-8">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv, .xlsx, .xls"
              className="hidden"
            />

            <div
              id="drag-and-drop-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={triggerFileSelect}
              className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition-all cursor-pointer flex flex-col items-center justify-center space-y-4 ${
                isDragging
                  ? "border-[#0284C7] bg-[#F0F9FF] scale-[0.99]"
                  : "border-[#E5E2D9] hover:border-[#0284C7] hover:bg-[#F0F9FF]/40"
              }`}
            >
              {isLoading ? (
                <div className="py-4 flex flex-col items-center justify-center space-y-3">
                  <Loader2 id="loading-spinner" className="h-8 w-8 animate-spin text-[#0284C7]" />
                  <p className="text-sm font-medium text-[#1F1E1D]">
                    Running pre-processing structural validation & schema profiling...
                  </p>
                </div>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-[#F0F9FF] border border-[#BAE6FD] flex items-center justify-center text-[#0284C7] shadow-xs">
                    <Upload className="h-6 w-6" />
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm font-medium text-[#1F1E1D]">
                      <span className="text-[#0284C7] underline underline-offset-2">Click to browse</span> or drag and drop your spreadsheet here
                    </p>
                    <p className="text-xs text-[#73726C]">
                      Supports .csv, .xlsx, and .xls files up to 50MB
                    </p>
                  </div>

                  <div className="pt-2 flex flex-wrap items-center justify-center gap-2 text-[11px] text-[#73726C]">
                    <span className="px-2 py-0.5 rounded bg-[#FAF9F5] border border-[#E5E2D9] inline-flex items-center gap-1">
                      <CheckCircle className="h-3 w-3 text-emerald-600" />
                      Header Integrity Check
                    </span>
                    <span>·</span>
                    <span className="px-2 py-0.5 rounded bg-[#FAF9F5] border border-[#E5E2D9] inline-flex items-center gap-1">
                      <CheckCircle className="h-3 w-3 text-emerald-600" />
                      Corrupted Row Guard
                    </span>
                    <span>·</span>
                    <span className="px-2 py-0.5 rounded bg-[#FAF9F5] border border-[#E5E2D9] inline-flex items-center gap-1">
                      <CheckCircle className="h-3 w-3 text-emerald-600" />
                      Polars WASM Stream
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          /* Tab 2: Paste Raw Tabular Data */
          <div className="p-6 space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-[#1F1E1D]">
                Paste comma-separated (CSV) or tab-separated (TSV) records:
              </label>
              <textarea
                rows={6}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="id,age,income,category,target&#10;1,25,50000,Retail,1&#10;2,38,82000,Tech,0&#10;3,44,95000,Finance,1"
                className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-3.5 text-xs font-mono text-[#1F1E1D] focus:border-[#0284C7] focus:ring-1 focus:ring-[#0284C7] outline-none"
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-xs text-[#73726C]">
                First row must contain header column names.
              </span>
              <button
                type="button"
                onClick={() => handleTextSubmit()}
                disabled={!textInput.trim() || isLoading}
                className="px-4 py-2 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-lg cursor-pointer transition-colors shadow-xs disabled:opacity-50 flex items-center space-x-2"
              >
                {isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                <span>Validate & Import Pasted Data</span>
              </button>
            </div>
          </div>
        )}

        {/* Batch Processing Configuration Panel */}
        <div className="px-6 py-4 bg-[#FAF9F5] border-t border-[#E5E2D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-purple-50 border border-purple-200 text-purple-700 flex items-center justify-center flex-shrink-0">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono font-bold text-[#1F1E1D]">
                  Batch Processing Mode
                </span>
                <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-purple-100 text-purple-800 border border-purple-200">
                  Token Guard
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#73726C]">
                Process datasets in sequential micro-chunks to stay within token limits and eliminate memory spikes
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 font-mono text-xs">
            <label className="flex items-center space-x-2 cursor-pointer mr-2">
              <input
                type="checkbox"
                checked={enableBatchProcessing}
                onChange={(e) => setEnableBatchProcessing(e.target.checked)}
                className="rounded text-purple-600 focus:ring-purple-500 cursor-pointer"
              />
              <span className="text-xs font-bold text-[#1F1E1D]">Enable Batching</span>
            </label>

            {enableBatchProcessing && (
              <div className="flex items-center space-x-1 bg-white p-1 rounded border border-[#E5E2D9]">
                <span className="text-[10px] text-[#73726C] px-1 font-bold">Chunk:</span>
                {[250, 500, 1000, 2500].map(sz => (
                  <button
                    key={sz}
                    type="button"
                    onClick={() => setBatchChunkSize(sz)}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all cursor-pointer ${
                      batchChunkSize === sz
                        ? "bg-purple-600 text-white shadow-2xs"
                        : "text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6]"
                    }`}
                  >
                    {sz.toLocaleString()}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div id="upload-error-banner" className="p-3 bg-[#F0F9FF] text-[#0284C7] text-xs font-medium rounded-xl border border-[#BAE6FD] text-center">
          {error}
        </div>
      )}

      {/* Benchmark Sample Datasets Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-serif text-base font-normal text-[#1F1E1D]">
            Or load a benchmark sample dataset:
          </h3>
          <span className="text-xs text-[#73726C]">Pre-validated for instant exploration</span>
        </div>

        <div className="grid sm:grid-cols-3 gap-3.5">
          {SAMPLE_DATASETS.map((sample, idx) => (
            <button
              key={idx}
              id={`sample-card-${idx}`}
              onClick={() => loadSampleDataset(sample.name, sample.csv)}
              className="p-4 rounded-xl bg-white border border-[#E5E2D9] hover:border-[#0284C7] hover:bg-[#F0F9FF]/30 text-left transition-all cursor-pointer shadow-xs group space-y-2"
            >
              <div className="flex items-center justify-between">
                <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                  {idx === 0 ? <Flower2 className="h-4 w-4" /> : idx === 1 ? <Home className="h-4 w-4" /> : <Ship className="h-4 w-4" />}
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C]">
                  {idx === 0 ? "Classification" : idx === 1 ? "Regression" : "Ensemble"}
                </span>
              </div>
              <div>
                <h4 className="font-medium text-xs text-[#1F1E1D] group-hover:text-[#0284C7] transition-colors">
                  {sample.name}
                </h4>
                <p className="text-[11px] text-[#73726C] mt-0.5">
                  {idx === 0 ? "150 rows · 4 numerical features" : idx === 1 ? "Housing price prediction" : "Survival classification & null handling"}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Feature Capabilities Summary */}
      <div className="grid sm:grid-cols-3 gap-3.5 pt-2">
        <div className="p-4 rounded-xl bg-white border border-[#E5E2D9] space-y-1.5 shadow-xs">
          <div className="flex items-center space-x-2 text-[#0284C7]">
            <TrendingUp className="h-4 w-4" />
            <h4 className="text-xs font-medium text-[#1F1E1D]">Exploratory EDA</h4>
          </div>
          <p className="text-[11px] text-[#73726C] leading-relaxed">
            Automated correlation heatmaps, histogram distributions, bivariate trendlines, and outlier bounds.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-white border border-[#E5E2D9] space-y-1.5 shadow-xs">
          <div className="flex items-center space-x-2 text-[#0284C7]">
            <Zap className="h-4 w-4" />
            <h4 className="text-xs font-medium text-[#1F1E1D]">Polars Engine</h4>
          </div>
          <p className="text-[11px] text-[#73726C] leading-relaxed">
            Vectorized Rust query engine streaming out-of-core pipeline cleaning and automated code export.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-white border border-[#E5E2D9] space-y-1.5 shadow-xs">
          <div className="flex items-center space-x-2 text-[#0284C7]">
            <Layers className="h-4 w-4" />
            <h4 className="text-xs font-medium text-[#1F1E1D]">AutoML Tournament</h4>
          </div>
          <p className="text-[11px] text-[#73726C] leading-relaxed">
            Benchmark candidate estimators and auto-tune hyperparameters with K-fold cross-validation.
          </p>
        </div>
      </div>
    </div>
  );
}
