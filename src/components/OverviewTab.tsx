import React, { useState } from "react";
import { DatasetSummary } from "../types";
import { 
  RefreshCw, AlertTriangle, CheckCircle2, Shield, ShieldAlert, Lock, Check,
  Layers, Sparkles, Zap, ArrowRight, Database, RotateCw
} from "lucide-react";

interface OverviewTabProps {
  summary: DatasetSummary;
  onReset: () => void;
  onRebatch?: (chunkSize: number) => void;
}

export default function OverviewTab({ summary, onReset, onRebatch }: OverviewTabProps) {
  const cellCount = summary.rowCount * summary.colCount;
  
  const numericCount = summary.columns.filter(c => c.type === "Numeric").length;
  const categoricalCount = summary.columns.filter(c => c.type === "Categorical").length;
  const otherCount = summary.colCount - numericCount - categoricalCount;

  // Sensitive & PII Columns Detection
  const sensitiveColumns = summary.columns.filter(c => c.sensitivity?.isSensitive);
  const highRiskCount = sensitiveColumns.filter(c => c.sensitivity?.level === "High").length;
  const mediumRiskCount = sensitiveColumns.filter(c => c.sensitivity?.level === "Medium").length;
  const lowRiskCount = sensitiveColumns.filter(c => c.sensitivity?.level === "Low").length;

  return (
    <div id="overview-tab-content" className="space-y-6">
      
      {/* Metrics Row: Clean stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
        
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <p className="text-[11px] text-[#73726C] font-medium">Total Records</p>
          <p className="text-2xl font-serif font-normal text-[#1F1E1D] mt-1">
            {summary.rowCount.toLocaleString()}
          </p>
          <span className="text-[10px] text-[#9E9D96] mt-1 font-mono">sample observations</span>
        </div>

        <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <p className="text-[11px] text-[#73726C] font-medium">Feature Columns</p>
          <p className="text-2xl font-serif font-normal text-[#1F1E1D] mt-1">
            {summary.colCount}
          </p>
          <span className="text-[10px] text-[#9E9D96] mt-1 font-mono">{numericCount} numeric · {categoricalCount} cat</span>
        </div>

        <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <p className="text-[11px] text-[#73726C] font-medium">Matrix Cells</p>
          <p className="text-2xl font-serif font-normal text-[#1F1E1D] mt-1">
            {cellCount.toLocaleString()}
          </p>
          <span className="text-[10px] text-[#9E9D96] mt-1 font-mono">total data points</span>
        </div>

        {summary.dataQualityScore !== undefined && (
          <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs">
            <p className="text-[11px] text-[#73726C] font-medium">Quality Score</p>
            <div className="flex items-baseline space-x-1.5 mt-1">
              <p className={`text-2xl font-serif font-normal ${
                summary.dataQualityScore >= 80 
                  ? "text-sky-600" 
                  : summary.dataQualityScore >= 50 
                    ? "text-sky-600" 
                    : "text-rose-600"
              }`}>
                {summary.dataQualityScore}%
              </p>
              <span className={`text-[10px] font-medium ${
                summary.dataQualityScore >= 80 ? "text-sky-600" : "text-sky-600"
              }`}>
                {summary.dataQualityScore >= 80 ? "Healthy" : "Review"}
              </span>
            </div>
            <span className="text-[10px] text-[#9E9D96] mt-1 font-mono">schema integrity</span>
          </div>
        )}

        {/* Data Privacy & PII Stat Card */}
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs">
          <p className="text-[11px] text-[#73726C] font-medium">Data Privacy & PII</p>
          <div className="flex items-baseline space-x-1.5 mt-1">
            <p className={`text-2xl font-serif font-normal ${
              sensitiveColumns.length > 0 
                ? highRiskCount > 0 ? "text-rose-600" : "text-amber-700" 
                : "text-emerald-700"
            }`}>
              {sensitiveColumns.length}
            </p>
            <span className={`text-[10px] font-medium ${
              sensitiveColumns.length > 0 
                ? highRiskCount > 0 ? "text-rose-600" : "text-amber-700" 
                : "text-emerald-700"
            }`}>
              {sensitiveColumns.length > 0 ? (highRiskCount > 0 ? "High Risk" : "Identified") : "Clean"}
            </span>
          </div>
          <span className="text-[10px] text-[#9E9D96] mt-1 font-mono">
            {sensitiveColumns.length > 0 ? `${sensitiveColumns.length} sensitive vector(s)` : "no sensitive vectors"}
          </span>
        </div>

        <div className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between shadow-xs col-span-2 sm:col-span-1">
          <p className="text-[11px] text-[#73726C] font-medium">Source Dataset</p>
          <p className="text-sm font-medium text-[#1F1E1D] mt-1 truncate" title={summary.fileName}>
            {summary.fileName}
          </p>
          <button
            onClick={onReset}
            className="inline-flex items-center space-x-1 text-[11px] text-[#0284C7] hover:underline cursor-pointer mt-1"
          >
            <RefreshCw className="h-3 w-3" />
            <span>Change file</span>
          </button>
        </div>
      </div>

      {/* Dataset Schema Profile & Feature Ingestion Logs */}
      <div className="grid lg:grid-cols-3 gap-5">
        
        {/* Schema Breakdown & Governance Card */}
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
          <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
            Dataset Schema Profile
          </h3>
          
          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between text-[#73726C]">
              <span>Numeric Features</span>
              <span className="font-mono font-medium text-[#1F1E1D]">{numericCount}</span>
            </div>
            <div className="flex items-center justify-between text-[#73726C]">
              <span>Categorical Features</span>
              <span className="font-mono font-medium text-[#1F1E1D]">{categoricalCount}</span>
            </div>
            <div className="flex items-center justify-between text-[#73726C]">
              <span>Other (Date / Text)</span>
              <span className="font-mono font-medium text-[#1F1E1D]">{otherCount}</span>
            </div>
          </div>

          <div className="w-full bg-[#F0EEE6] h-2 rounded-full overflow-hidden flex">
            {numericCount > 0 && (
              <div 
                className="bg-[#0284C7] h-full" 
                style={{ width: `${(numericCount / summary.colCount) * 100}%` }} 
                title={`Numeric: ${numericCount}`}
              />
            )}
            {categoricalCount > 0 && (
              <div 
                className="bg-[#73726C] h-full" 
                style={{ width: `${(categoricalCount / summary.colCount) * 100}%` }} 
                title={`Categorical: ${categoricalCount}`}
              />
            )}
            {otherCount > 0 && (
              <div 
                className="bg-[#D5D1C5] h-full" 
                style={{ width: `${(otherCount / summary.colCount) * 100}%` }} 
                title={`Other: ${otherCount}`}
              />
            )}
          </div>

          {/* Privacy & Governance Section */}
          <div className="pt-3 border-t border-[#E5E2D9] space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-[#1F1E1D] flex items-center gap-1.5">
                <Shield className="h-3.5 w-3.5 text-[#0284C7]" />
                <span>Privacy & PII Audit</span>
              </span>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                sensitiveColumns.length > 0 
                  ? "bg-amber-100 text-amber-800 border border-amber-200" 
                  : "bg-emerald-50 text-emerald-800 border border-emerald-200"
              }`}>
                {sensitiveColumns.length > 0 ? `${sensitiveColumns.length} Sensitive` : "Verified Clean"}
              </span>
            </div>

            {sensitiveColumns.length > 0 ? (
              <div className="space-y-1.5 text-[11px] text-[#73726C]">
                <p>
                  Identified {sensitiveColumns.length} feature(s) containing personally identifiable information (PII). Consider applying tokenization or hashing before model export.
                </p>
                <div className="flex flex-wrap gap-1 pt-1">
                  {sensitiveColumns.map((col, cIdx) => (
                    <span 
                      key={cIdx} 
                      className={`px-1.5 py-0.5 rounded text-[9px] font-mono border font-medium ${
                        col.sensitivity?.level === "High" 
                          ? "bg-rose-50 text-rose-700 border-rose-200" 
                          : "bg-amber-50 text-amber-700 border-amber-200"
                      }`}
                    >
                      {col.name} ({col.sensitivity?.category})
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-[#73726C] leading-relaxed">
                Automated inference scanned column headers and pattern entropy: no high-risk identity, contact, or financial vectors found.
              </p>
            )}
          </div>

          {summary.dataQualityScore !== undefined && (
            <div className="pt-3 border-t border-[#E5E2D9] space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-[#73726C]">Integrity Index</span>
                <span className="font-mono font-medium text-sky-600">
                  {summary.dataQualityScore}%
                </span>
              </div>
              <p className="text-[11px] text-[#73726C] leading-relaxed">
                Calculated from cell completeness and column homogeneity across all observation rows.
              </p>
            </div>
          )}
        </div>

        {/* Feature List Table with Privacy Badges */}
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 lg:col-span-2 space-y-3 shadow-xs">
          <div className="flex items-center justify-between">
            <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
              Feature Ingestion Directory
            </h3>
            {sensitiveColumns.length > 0 && (
              <span className="text-[10px] font-mono text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                {sensitiveColumns.length} Privacy Flagged
              </span>
            )}
          </div>

          <div className="overflow-y-auto max-h-[220px] pr-1">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#E5E2D9] text-[#73726C] font-medium text-[11px]">
                  <th className="py-2 px-2">Column & Privacy</th>
                  <th className="py-2 px-2">Type</th>
                  <th className="py-2 px-2 text-right">Missing</th>
                  <th className="py-2 px-2 text-right">Unique</th>
                  <th className="py-2 px-2">Range / Values</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E5E2D9]/60 text-[#1F1E1D]">
                {summary.columns.map((col, idx) => {
                  const missingPercent = ((col.stats.missingCount / summary.rowCount) * 100).toFixed(1);
                  let detailsStr = "";
                  if (col.type === "Numeric") {
                    detailsStr = `[${col.stats.min} ... ${col.stats.max}]`;
                  } else if (col.stats.frequencies) {
                    const topVals = Object.keys(col.stats.frequencies).slice(0, 2);
                    detailsStr = topVals.join(", ");
                  }

                  return (
                    <tr key={idx} className="hover:bg-[#F0EEE6]/50">
                      <td className="py-2 px-2">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium font-mono text-xs">{col.name}</span>
                          
                          {/* Privacy Badge next to sensitive column */}
                          {col.sensitivity?.isSensitive && (
                            <span 
                              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono font-medium border shrink-0 ${
                                col.sensitivity.level === "High"
                                  ? "bg-rose-50 text-rose-700 border-rose-200"
                                  : col.sensitivity.level === "Medium"
                                    ? "bg-amber-50 text-amber-700 border-amber-200"
                                    : "bg-sky-50 text-sky-700 border-sky-200"
                              }`}
                              title={`Data Privacy: ${col.sensitivity.reason} (${col.sensitivity.level} Risk)`}
                            >
                              <Shield className="h-2.5 w-2.5" />
                              <span>PII: {col.sensitivity.category}</span>
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-2 px-2">
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#F0EEE6] text-[#73726C]">
                          {col.type}
                        </span>
                      </td>
                      <td className="py-2 px-2 text-right font-mono text-[11px]">
                        <span className={col.stats.missingCount > 0 ? "text-sky-600 font-medium" : "text-[#73726C]"}>
                          {col.stats.missingCount} ({missingPercent}%)
                        </span>
                      </td>
                      <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                        {col.stats.uniqueCount.toLocaleString()}
                      </td>
                      <td className="py-2 px-2 text-[#73726C] font-mono text-[11px] truncate max-w-[150px]" title={detailsStr}>
                        {detailsStr}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* BATCH PROCESSING & TOKEN CONSERVATION INSPECTOR */}
      {/* ========================================================================= */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs font-mono">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded bg-purple-500/10 border border-purple-500/20 text-purple-700 flex items-center justify-center">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="font-mono font-bold text-xs uppercase tracking-wider text-[#1F1E1D]">
                  Batch Processing & Token Conservation Engine
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-100 text-purple-800 border border-purple-200">
                  {summary.batchReport?.isBatched 
                    ? `Chunked: ${summary.batchReport.chunkSize} rows/batch` 
                    : "Single Ingestion"}
                </span>
              </div>
              <p className="text-[11px] text-[#73726C]">
                {summary.batchReport?.isBatched
                  ? `Processed across ${summary.batchReport.totalChunks} sequential micro-batches to guarantee strict adherence to LLM token limits`
                  : "Dataset ingested in a single batch. You can re-partition into sequential chunks below to optimize tokens."}
              </p>
            </div>
          </div>

          {/* Quick Re-Batch Action */}
          {onRebatch && (
            <div className="flex items-center space-x-1.5 bg-white p-1.5 rounded-lg border border-[#E5E2D9]">
              <span className="text-[10px] text-[#73726C] font-bold px-1">Re-Chunk:</span>
              {[250, 500, 1000, 2500].map(sz => (
                <button
                  key={sz}
                  type="button"
                  onClick={() => onRebatch(sz)}
                  className={`px-2 py-1 rounded text-[10px] font-bold transition-all cursor-pointer ${
                    summary.batchReport?.chunkSize === sz
                      ? "bg-purple-600 text-white shadow-2xs"
                      : "text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6]"
                  }`}
                  title={`Re-chunk dataset into ${sz} rows per batch`}
                >
                  {sz.toLocaleString()}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Batch KPI Cards */}
        {summary.batchReport && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-white rounded-lg border border-[#E5E2D9] space-y-1">
              <span className="text-[10px] uppercase text-[#73726C] font-bold">Total Chunks</span>
              <p className="font-bold text-base text-[#1F1E1D]">
                {summary.batchReport.totalChunks} Batches
              </p>
              <p className="text-[10px] text-purple-700">
                {summary.batchReport.chunkSize.toLocaleString()} rows / chunk
              </p>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#E5E2D9] space-y-1">
              <span className="text-[10px] uppercase text-[#73726C] font-bold">Estimated Tokens Saved</span>
              <p className="font-bold text-base text-emerald-700">
                ~{summary.batchReport.estimatedTokensSaved.toLocaleString()}
              </p>
              <p className="text-[10px] text-emerald-600 font-semibold">
                {summary.batchReport.tokenReductionPercent}% Token Reduction
              </p>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#E5E2D9] space-y-1">
              <span className="text-[10px] uppercase text-[#73726C] font-bold">Chunking Latency</span>
              <p className="font-bold text-base text-[#1F1E1D]">
                {summary.batchReport.processingDurationMs} ms
              </p>
              <p className="text-[10px] text-sky-600 font-semibold">
                {Math.round(summary.rowCount / Math.max(0.001, summary.batchReport.processingDurationMs / 1000)).toLocaleString()} rows/sec
              </p>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#E5E2D9] space-y-1">
              <span className="text-[10px] uppercase text-[#73726C] font-bold">Memory Footprint</span>
              <p className="font-bold text-base text-[#1F1E1D]">
                ~{summary.batchReport.peakMemoryMB} MB
              </p>
              <p className="text-[10px] text-[#73726C]">
                Zero Out-of-Memory Risk
              </p>
            </div>
          </div>
        )}

        {/* Chunk Details Table */}
        {summary.batchReport?.chunks && summary.batchReport.chunks.length > 0 && (
          <div className="bg-white rounded-lg border border-[#E5E2D9] overflow-hidden">
            <div className="px-3.5 py-2 bg-[#FAF9F5] border-b border-[#E5E2D9] flex items-center justify-between text-[10px] text-[#73726C] font-bold uppercase">
              <span>Sequential Chunk Ingestion Matrix</span>
              <span>{summary.batchReport.chunks.length} Micro-Batches</span>
            </div>
            <div className="overflow-x-auto max-h-48">
              <table className="w-full text-left border-collapse text-[11px]">
                <thead>
                  <tr className="border-b border-[#E5E2D9] text-[#73726C] text-[9px] uppercase bg-[#FAF9F5]/50">
                    <th className="py-1.5 px-3">Batch</th>
                    <th className="py-1.5 px-3">Row Interval</th>
                    <th className="py-1.5 px-3">Records</th>
                    <th className="py-1.5 px-3">Missing Cells</th>
                    <th className="py-1.5 px-3">Quality</th>
                    <th className="py-1.5 px-3 text-right">Chunk Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E5E2D9]/60 text-[#1F1E1D]">
                  {summary.batchReport.chunks.map((chk) => (
                    <tr key={chk.chunkIndex} className="hover:bg-[#FAF9F5]">
                      <td className="py-2 px-3 font-bold text-purple-700">
                        Chunk #{chk.chunkIndex}
                      </td>
                      <td className="py-2 px-3 text-[#73726C]">
                        Rows {chk.startRow.toLocaleString()} - {chk.endRow.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 font-semibold">
                        {chk.rowCount.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-[#73726C]">
                        {chk.nullCount > 0 ? `${chk.nullCount} nulls` : "0 nulls"}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                          chk.qualityScore >= 80 ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"
                        }`}>
                          {chk.qualityScore}%
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right text-[#73726C]">
                        {chk.processingTimeMs} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Outlier & Anomaly Analysis Card */}
      <div id="outliers-anomaly-section" className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E5E2D9] pb-3">
          <div>
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-[#0284C7]" />
              <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                Statistical Anomalies & Outlier Isolation
              </h3>
            </div>
            <p className="text-xs text-[#73726C] mt-0.5">
              Numerical variables exhibiting extreme variance evaluated using Interquartile Range (IQR) bounds and Z-Scores.
            </p>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#F0EEE6] text-[#73726C] self-start">
            Tukey Fences (1.5× IQR)
          </span>
        </div>

        {summary.outliers && summary.outliers.filter(o => o.isSignificant).length > 0 ? (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {summary.outliers.filter(o => o.isSignificant).map((outlier, oIdx) => {
              const colInfo = summary.columns.find(c => c.name === outlier.columnName);
              const meanVal = colInfo?.stats.mean ?? 0;
              const stdDevVal = colInfo?.stats.stdDev ?? 1;

              return (
                <div key={oIdx} className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-4 space-y-2.5 text-xs">
                  <div className="flex items-center justify-between">
                    <h4 className="font-mono font-medium text-xs text-[#1F1E1D]">{outlier.columnName}</h4>
                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-sky-600 font-medium">
                      {outlier.outlierCount} outliers ({outlier.outlierPercentage}%)
                    </span>
                  </div>

                  <div className="text-[11px] text-[#73726C] space-y-1 font-mono">
                    <p>Expected fence: [{outlier.lowerBound}, {outlier.upperBound}]</p>
                  </div>

                  <div className="space-y-1 pt-1 border-t border-[#E5E2D9]">
                    <span className="text-[10px] text-[#73726C]">Extreme Sample Points:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {outlier.extremeValues.slice(0, 4).map((v, vIdx) => {
                        const z = Math.abs((v - meanVal) / stdDevVal).toFixed(1);
                        return (
                          <span key={vIdx} className="px-1.5 py-0.5 bg-white border border-[#E5E2D9] rounded text-[10px] font-mono text-[#1F1E1D]" title={`Z-score: ${z}`}>
                            {v} <span className="text-[#9E9D96] text-[8px]">(z={z})</span>
                          </span>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-4 bg-[#F0EEE6]/50 border border-[#E5E2D9] text-xs text-[#73726C] rounded-xl text-center py-5 space-y-1">
            <CheckCircle2 className="h-4 w-4 text-sky-600 mx-auto" />
            <p className="font-medium text-[#1F1E1D]">Well-Conditioned Distribution</p>
            <p className="text-[11px]">No columns with statistically significant anomaly clusters detected.</p>
          </div>
        )}
      </div>

      {/* Comprehensive Feature Profiler Matrix with Privacy Badges */}
      <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-3.5 shadow-xs">
        <div className="border-b border-[#E5E2D9] pb-2.5 flex items-center justify-between">
          <div>
            <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
              Tabular Feature Profiler
            </h3>
            <p className="text-xs text-[#73726C]">
              Descriptive statistics, data privacy labels, and moments calculated across all columns.
            </p>
          </div>
          {sensitiveColumns.length > 0 && (
            <div className="flex items-center space-x-1.5 text-xs text-[#73726C]">
              <Shield className="h-3.5 w-3.5 text-amber-600" />
              <span className="text-[11px] font-mono">{sensitiveColumns.length} sensitive fields tagged</span>
            </div>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs min-w-[1000px]">
            <thead>
              <tr className="border-b border-[#E5E2D9] text-[#73726C] font-medium text-[11px]">
                <th className="py-2 px-2">Variable</th>
                <th className="py-2 px-2">Privacy</th>
                <th className="py-2 px-2">Type</th>
                <th className="py-2 px-2 text-right">Valid</th>
                <th className="py-2 px-2 text-right">Null</th>
                <th className="py-2 px-2 text-right">Mean</th>
                <th className="py-2 px-2 text-right">Median</th>
                <th className="py-2 px-2 text-right">Std Dev</th>
                <th className="py-2 px-2 text-right">Range</th>
                <th className="py-2 px-2 text-right">Skewness</th>
                <th className="py-2 px-2 text-right">Outliers</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2D9]/60 text-[#1F1E1D]">
              {summary.columns.map((col, idx) => {
                const s = col.stats;
                const isNum = col.type === "Numeric";
                const colOutliers = summary.outliers?.find(o => o.columnName === col.name);
                
                return (
                  <tr key={idx} className="hover:bg-[#F0EEE6]/50">
                    <td className="py-2 px-2 font-mono font-medium">
                      {col.name}
                    </td>
                    <td className="py-2 px-2">
                      {col.sensitivity?.isSensitive ? (
                        <span 
                          className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono font-medium border ${
                            col.sensitivity.level === "High"
                              ? "bg-rose-50 text-rose-700 border-rose-200"
                              : col.sensitivity.level === "Medium"
                                ? "bg-amber-50 text-amber-700 border-amber-200"
                                : "bg-sky-50 text-sky-700 border-sky-200"
                          }`}
                          title={col.sensitivity.reason}
                        >
                          <Shield className="h-2.5 w-2.5" />
                          <span>PII: {col.sensitivity.category}</span>
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono text-[#9E9D96]">
                          Public
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-2">
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#F0EEE6] text-[#73726C]">
                        {col.type}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                      {s.nonNullCount !== undefined ? s.nonNullCount : summary.rowCount - s.missingCount}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                      {s.missingCount > 0 ? <span className="text-amber-600 font-medium">{s.missingCount}</span> : "0"}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px]">
                      {isNum && s.mean !== undefined ? s.mean : <span className="text-[#9E9D96]">-</span>}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px]">
                      {isNum && s.median !== undefined ? s.median : <span className="text-[#9E9D96]">-</span>}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                      {isNum && s.stdDev !== undefined ? s.stdDev : <span className="text-[#9E9D96]">-</span>}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                      {isNum && s.range !== undefined ? s.range : <span className="text-[#9E9D96]">-</span>}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px] text-[#73726C]">
                      {isNum && s.skewness !== undefined ? s.skewness : <span className="text-[#9E9D96]">-</span>}
                    </td>
                    <td className="py-2 px-2 text-right font-mono text-[11px]">
                      {isNum && colOutliers && colOutliers.outlierCount > 0 ? (
                        <span className="text-[#0284C7] font-medium">{colOutliers.outlierCount}</span>
                      ) : <span className="text-[#9E9D96]">0</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Sample Preview */}
      <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-3 shadow-xs">
        <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
          <div>
            <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
              Observation Matrix Sample
            </h3>
            <p className="text-xs text-[#73726C]">
              First 10 records of active dataset.
            </p>
          </div>
          <span className="text-[11px] text-[#73726C] font-mono">
            {summary.rowCount.toLocaleString()} total records
          </span>
        </div>

        <div className="overflow-x-auto border border-[#E5E2D9] rounded-lg">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#73726C] font-mono text-[11px]">
                <th className="py-2 px-3">#</th>
                {summary.columns.map((col, idx) => (
                  <th key={idx} className="py-2 px-3 font-medium whitespace-nowrap">
                    <div className="flex items-center space-x-1.5">
                      <span>{col.name}</span>
                      {col.sensitivity?.isSensitive && (
                        <Shield className="h-3 w-3 text-amber-600 shrink-0" title={`PII: ${col.sensitivity.category}`} />
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E2D9]/60 text-[#1F1E1D]">
              {summary.sampleData.slice(0, 10).map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-[#F0EEE6]/30">
                  <td className="py-2 px-3 text-[#9E9D96] font-mono text-[11px]">{rowIdx + 1}</td>
                  {summary.columns.map((col, colIdx) => {
                    const cellVal = row[col.name];
                    return (
                      <td key={colIdx} className="py-2 px-3 font-mono text-[11px] whitespace-nowrap max-w-[180px] truncate">
                        {cellVal !== undefined && cellVal !== null ? String(cellVal) : <span className="text-[#9E9D96] italic">null</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
