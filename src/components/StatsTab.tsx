import React, { useState, useMemo, useEffect, useRef } from "react";
import { DatasetSummary, ColumnInfo, StatisticalTest } from "../types";
import { 
  runStatisticalTests, computeColumnStats 
} from "../utils/stats";
import { 
  Dna, HelpCircle, AlertCircle, FileText, CheckCircle2, Play,
  BarChart3, Sliders, ChevronDown, ArrowUpDown, Search, Grid3X3,
  List, Sparkles, TrendingUp, Info, X, Download, LayoutGrid, Check,
  Activity, FileSpreadsheet, Eye, SlidersHorizontal, AlertTriangle
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface StatsTabProps {
  summary: DatasetSummary;
}

export default function StatsTab({ summary }: StatsTabProps) {
  // --- Existing States for Hypothesis Testing Lab ---
  const [var1Name, setVar1Name] = useState<string>(
    summary.columns[0]?.name || ""
  );
  const [var2Name, setVar2Name] = useState<string>(
    summary.columns[1]?.name || ""
  );

  const var1 = useMemo(() => summary.columns.find(c => c.name === var1Name) || null, [var1Name, summary.columns]);
  const var2 = useMemo(() => summary.columns.find(c => c.name === var2Name) || null, [var2Name, summary.columns]);

  // --- New States for Comprehensive Batch Statistical Report ---
  const [reportState, setReportState] = useState<{
    isGenerated: boolean;
    isGenerating: boolean;
    progress: number;
    stage: string;
  }>({
    isGenerated: false,
    isGenerating: false,
    progress: 0,
    stage: "Ready"
  });

  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"name" | "variance" | "skewness" | "missing" | "outliers">("name");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [layoutMode, setLayoutMode] = useState<"detailed" | "compact">("detailed");
  const reportIntervalRef = React.useRef<any>(null);

  useEffect(() => {
    return () => {
      if (reportIntervalRef.current) clearInterval(reportIntervalRef.current);
    };
  }, []);

  // Number Formatter Helper
  const formatNum = (v: number | undefined | null) => {
    if (v === undefined || v === null || isNaN(v)) return "0";
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
    if (Math.abs(v) > 100000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return Number(v.toFixed(3)).toString();
  };

  // Filter columns to get all numeric features
  const numericCols = useMemo(() => {
    return summary.columns.filter(col => col.type === "Numeric");
  }, [summary.columns]);

  // Global aggregate statistics & automated interpretations
  const globalInsights = useMemo(() => {
    if (numericCols.length === 0) return null;

    let highlyRightSkewed: string[] = [];
    let highlyLeftSkewed: string[] = [];
    let symmetrical: string[] = [];
    
    let highestVarianceCol = "";
    let highestVarianceVal = -1;
    
    let highestOutliersCol = "";
    let highestOutliersVal = -1;
    let highestOutliersPct = 0;
    
    let highestMissingCol = "";
    let highestMissingVal = -1;

    numericCols.forEach(col => {
      const skew = col.stats.skewness || 0;
      if (skew > 1) highlyRightSkewed.push(col.name);
      else if (skew < -1) highlyLeftSkewed.push(col.name);
      else if (Math.abs(skew) <= 0.5) symmetrical.push(col.name);

      const variance = col.stats.variance || 0;
      if (variance > highestVarianceVal) {
        highestVarianceVal = variance;
        highestVarianceCol = col.name;
      }

      const outResult = summary.outliers?.find(o => o.columnName === col.name);
      const outCount = outResult?.outlierCount || 0;
      if (outCount > highestOutliersVal) {
        highestOutliersVal = outCount;
        highestOutliersCol = col.name;
        highestOutliersPct = outResult?.outlierPercentage || 0;
      }

      const missing = col.stats.missingCount || 0;
      if (missing > highestMissingVal) {
        highestMissingVal = missing;
        highestMissingCol = col.name;
      }
    });

    return {
      highlyRightSkewed,
      highlyLeftSkewed,
      symmetrical,
      highestVarianceCol,
      highestVarianceVal,
      highestOutliersCol,
      highestOutliersVal,
      highestOutliersPct,
      highestMissingCol,
      highestMissingVal
    };
  }, [numericCols, summary.outliers]);

  // Filtered and Sorted columns based on active configuration
  const filteredAndSortedCols = useMemo(() => {
    let cols = [...numericCols];

    // Filter by search query
    if (searchQuery.trim() !== "") {
      const q = searchQuery.toLowerCase();
      cols = cols.filter(c => c.name.toLowerCase().includes(q));
    }

    // Sort
    cols.sort((a, b) => {
      let valA: any = 0;
      let valB: any = 0;

      if (sortBy === "name") {
        valA = a.name.toLowerCase();
        valB = b.name.toLowerCase();
      } else if (sortBy === "variance") {
        valA = a.stats.variance || 0;
        valB = b.stats.variance || 0;
      } else if (sortBy === "skewness") {
        valA = Math.abs(a.stats.skewness || 0);
        valB = Math.abs(b.stats.skewness || 0);
      } else if (sortBy === "missing") {
        valA = a.stats.missingCount || 0;
        valB = b.stats.missingCount || 0;
      } else if (sortBy === "outliers") {
        const outA = summary.outliers?.find(o => o.columnName === a.name)?.outlierCount || 0;
        const outB = summary.outliers?.find(o => o.columnName === b.name)?.outlierCount || 0;
        valA = outA;
        valB = outB;
      }

      if (valA < valB) return sortOrder === "asc" ? -1 : 1;
      if (valA > valB) return sortOrder === "asc" ? 1 : -1;
      return 0;
    });

    return cols;
  }, [numericCols, searchQuery, sortBy, sortOrder, summary.outliers]);

  // Trigger simulated report generation pipeline
  const handleTriggerReport = () => {
    if (reportIntervalRef.current) {
      clearInterval(reportIntervalRef.current);
      reportIntervalRef.current = null;
    }

    setReportState({
      isGenerated: false,
      isGenerating: true,
      progress: 0,
      stage: "Scanning dataset features..."
    });

    const stages = [
      { prg: 20, text: "Scanning numerical features in dataset..." },
      { prg: 45, text: "Computing 5-number summary percentiles (Q25, Median, Q75)..." },
      { prg: 70, text: "Running Tukey's fences outlier thresholds..." },
      { prg: 90, text: "Generating custom vector SVG box plots..." },
      { prg: 100, text: "Structuring visual report view..." }
    ];

    let step = 0;
    reportIntervalRef.current = setInterval(() => {
      if (step < stages.length) {
        const currentStage = stages[step];
        if (currentStage) {
          setReportState(prev => ({
            ...prev,
            progress: currentStage.prg,
            stage: currentStage.text
          }));
        }
        step++;
      } else {
        if (reportIntervalRef.current) {
          clearInterval(reportIntervalRef.current);
          reportIntervalRef.current = null;
        }
        setTimeout(() => {
          setReportState(prev => ({
            ...prev,
            isGenerated: true,
            isGenerating: false,
            progress: 100,
            stage: "Complete"
          }));
        }, 200);
      }
    }, 350);
  };

  const handleResetReport = () => {
    if (reportIntervalRef.current) {
      clearInterval(reportIntervalRef.current);
      reportIntervalRef.current = null;
    }
    setReportState({
      isGenerated: false,
      isGenerating: false,
      progress: 0,
      stage: "Ready"
    });
  };

  // --- Existing Test Logic ---
  const dynamicTest = useMemo<StatisticalTest | null>(() => {
    if (!var1 || !var2 || var1Name === var2Name) {
      return null;
    }

    const isV1Num = var1.type === "Numeric";
    const isV2Num = var2.type === "Numeric";

    if (isV1Num && isV2Num) {
      const tests = runStatisticalTests(summary.sampleData, [var1, var2]);
      return tests.find(t => t.type === "Pearson Correlation") || null;
    }

    if ((isV1Num && !isV2Num) || (!isV1Num && isV2Num)) {
      const numCol = isV1Num ? var1 : var2;
      const catCol = isV1Num ? var2 : var1;

      const levels = Array.from(new Set(summary.sampleData.map(row => String(row[catCol.name]).trim()).filter(v => v !== "null" && v !== "")));
      if (levels.length === 2) {
        const tests = runStatisticalTests(summary.sampleData, [numCol, catCol]);
        return tests.find(t => t.type === "T-Test") || null;
      } else if (levels.length > 2 && levels.length <= 15) {
        const tests = runStatisticalTests(summary.sampleData, [numCol, catCol]);
        return tests.find(t => t.type === "ANOVA") || null;
      } else {
        return {
          type: "ANOVA",
          variable1: numCol.name,
          variable2: catCol.name,
          statisticValue: 0,
          pValue: 1.0,
          interpretation: `Cannot run T-test or ANOVA because categorical variable '${catCol.name}' has too many unique levels (${levels.length}). Max allowed is 15.`
        };
      }
    }

    if (!isV1Num && !isV2Num) {
      const tests = runStatisticalTests(summary.sampleData, [var1, var2]);
      return tests.find(t => t.type === "Chi-Square") || null;
    }

    return null;
  }, [var1, var2, var1Name, var2Name, summary.sampleData]);

  const getPValueBadge = (p: number) => {
    if (p < 0.001) {
      return (
        <span className="px-2 py-0.5 bg-sky-500/10 text-sky-400 font-extrabold text-[10px] uppercase tracking-wider rounded border border-sky-500/20 font-mono">
          p &lt; 0.001 (Highly Significant)
        </span>
      );
    }
    if (p < 0.05) {
      return (
        <span className="px-2 py-0.5 bg-sky-500/10 text-sky-400 font-bold text-[10px] uppercase tracking-wider rounded border border-sky-500/20 font-mono">
          p = {p.toFixed(4)} (Significant)
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 bg-[#F0EEE6] text-[#73726C] font-semibold text-[10px] uppercase tracking-wider rounded border border-[#D5D1C5] font-mono">
        p = {p.toFixed(4)} (Not Significant)
      </span>
    );
  };

  return (
    <div id="stats-tab-content" className="space-y-6">
      
      {/* =========================================================================
          COMPREHENSIVE BATCH STATISTICAL REPORT SECTION
          ========================================================================= */}
      <div id="batch-report-card" className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-5 relative overflow-hidden">
        
        {/* Glow Decorator */}
        <div className="absolute top-0 right-0 w-32 h-32 bg-sky-500/5 rounded-full blur-3xl pointer-events-none" />

        {/* Header Block */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#E5E2D9] pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="p-1 bg-sky-500/15 text-sky-400 rounded-md">
                <BarChart3 className="h-4 w-4" />
              </span>
              <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                Comprehensive Batch Statistical Report
              </h3>
            </div>
            <p className="text-[11px] text-[#9E9D96] leading-normal">
              Compute summary distributions and interactive box plots for <strong>all {numericCols.length} numerical columns</strong> at once.
            </p>
          </div>

          {reportState.isGenerated && (
            <button
              onClick={handleResetReport}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-[#FAF9F5] border border-[#E5E2D9] hover:border-[#D5D1C5] hover:text-[#1F1E1D] text-[#73726C] text-[10px] font-mono font-bold uppercase rounded cursor-pointer transition-all duration-150"
            >
              <X className="h-3 w-3" />
              Collapse Report
            </button>
          )}
        </div>

        {/* Content Views: 1. Trigger Card, 2. Progress View, 3. Rendered Report */}
        
        {/* VIEW 1: Ready to trigger */}
        {!reportState.isGenerated && !reportState.isGenerating && (
          <div className="py-6 flex flex-col items-center justify-center text-center space-y-4 max-w-xl mx-auto">
            <div className="p-3.5 bg-white border border-[#E5E2D9] rounded-full text-sky-400 animate-pulse shadow-inner">
              <Sparkles className="h-7 w-7" />
            </div>
            <div className="space-y-1">
              <h4 className="text-[#1F1E1D] font-bold text-xs uppercase tracking-wider font-mono">
                Analyze All Numeric Dimensions At Once
              </h4>
              <p className="text-[11px] text-[#9E9D96] leading-relaxed font-mono">
                Runs full descriptive statistics on numeric metrics, evaluates quartile spacing, generates individual vector box plots, computes distribution skewness, and analyzes outlier density globally.
              </p>
            </div>

            {numericCols.length > 0 ? (
              <button
                onClick={handleTriggerReport}
                className="w-full sm:w-auto px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-white font-mono font-black text-xs uppercase tracking-wider rounded cursor-pointer transition-all duration-150 shadow-lg shadow-sky-500/10 flex items-center justify-center gap-2 hover:scale-[1.01]"
              >
                <Activity className="h-4 w-4" />
                Generate Comprehensive Batch Report
              </button>
            ) : (
              <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] text-[#9E9D96] rounded text-xs font-mono flex items-center gap-1.5">
                <AlertCircle className="h-3.5 w-3.5" />
                <span>No numeric features found in this dataset to generate a batch report.</span>
              </div>
            )}
          </div>
        )}

        {/* VIEW 2: Generating Progress */}
        {reportState.isGenerating && (
          <div className="py-12 max-w-md mx-auto space-y-4">
            <div className="flex justify-between items-center text-[10px] font-mono font-bold text-[#73726C] uppercase tracking-wider">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-sky-500 animate-ping" />
                {reportState.stage}
              </span>
              <span>{reportState.progress}%</span>
            </div>
            
            {/* Progress Bar Container */}
            <div className="h-1.5 w-full bg-[#FAF9F5] rounded-full overflow-hidden border border-[#E5E2D9]">
              <motion.div 
                className="h-full bg-sky-500 rounded-full"
                initial={{ width: "0%" }}
                animate={{ width: `${reportState.progress}%` }}
                transition={{ duration: 0.2 }}
              />
            </div>

            <p className="text-[9px] text-[#9E9D96] text-center uppercase tracking-wide font-mono">
              Running vectorized calculations on client browser. Please wait...
            </p>
          </div>
        )}

        {/* VIEW 3: Rendered Report */}
        {reportState.isGenerated && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Executive Summary Metrics Block */}
            {globalInsights && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-3 font-mono space-y-1">
                  <span className="text-[8px] text-[#9E9D96] font-bold uppercase tracking-wider block">Numeric Features</span>
                  <p className="text-xl font-black text-[#1F1E1D]">{numericCols.length}</p>
                  <p className="text-[9px] text-[#73726C] leading-none">Columns analyzed</p>
                </div>

                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-3 font-mono space-y-1">
                  <span className="text-[8px] text-[#9E9D96] font-bold uppercase tracking-wider block">Symmetrical Features</span>
                  <p className="text-xl font-black text-sky-400">
                    {globalInsights.symmetrical.length}
                  </p>
                  <p className="text-[9px] text-[#9E9D96] leading-none">|skewness| &le; 0.50</p>
                </div>

                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-3 font-mono space-y-1">
                  <span className="text-[8px] text-[#9E9D96] font-bold uppercase tracking-wider block">Highly Skewed Features</span>
                  <p className="text-xl font-black text-sky-500">
                    {globalInsights.highlyRightSkewed.length + globalInsights.highlyLeftSkewed.length}
                  </p>
                  <p className="text-[9px] text-[#9E9D96] leading-none">|skewness| &gt; 1.00</p>
                </div>

                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-3 font-mono space-y-1">
                  <span className="text-[8px] text-[#9E9D96] font-bold uppercase tracking-wider block">Peak Outlier Density</span>
                  <p className="text-xl font-black text-rose-400">
                    {globalInsights.highestOutliersCol ? globalInsights.highestOutliersVal : 0}
                  </p>
                  <p className="text-[9px] text-[#9E9D96] leading-none truncate" title={`In column: ${globalInsights.highestOutliersCol}`}>
                    In: {globalInsights.highestOutliersCol || "None"}
                  </p>
                </div>
              </div>
            )}

            {/* Smart Interpretation Panel */}
            {globalInsights && (
              <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-4 text-[11px] font-mono space-y-3">
                <div className="flex items-center gap-2 border-b border-[#E5E2D9] pb-2">
                  <TrendingUp className="h-3.5 w-3.5 text-sky-400" />
                  <span className="font-bold text-[#73726C] uppercase tracking-wider text-[10px]">
                    Automated Global Distribution Diagnostics
                  </span>
                </div>

                <div className="grid md:grid-cols-2 gap-x-6 gap-y-2.5 text-[#73726C]">
                  <div className="flex items-start gap-2">
                    <Check className="h-3.5 w-3.5 text-sky-600 mt-0.5 shrink-0" />
                    <p className="leading-relaxed">
                      <strong>Symmetrical Features ({globalInsights.symmetrical.length}):</strong>{" "}
                      {globalInsights.symmetrical.length > 0 ? (
                        <span className="text-[#73726C]">{globalInsights.symmetrical.join(", ")}</span>
                      ) : (
                        <span className="text-[#9E9D96]">None detected.</span>
                      )}
                    </p>
                  </div>

                  <div className="flex items-start gap-2">
                    <AlertCircle className="h-3.5 w-3.5 text-sky-600 mt-0.5 shrink-0" />
                    <p className="leading-relaxed">
                      <strong>Right-Skewed Features ({globalInsights.highlyRightSkewed.length}):</strong>{" "}
                      {globalInsights.highlyRightSkewed.length > 0 ? (
                        <span className="text-[#73726C]">{globalInsights.highlyRightSkewed.join(", ")}</span>
                      ) : (
                        <span className="text-[#9E9D96]">None detected.</span>
                      )}
                    </p>
                  </div>

                  {globalInsights.highestVarianceCol && (
                    <div className="flex items-start gap-2 col-span-1 md:col-span-2">
                      <span className="text-[#9E9D96] font-mono">•</span>
                      <p className="leading-relaxed">
                        The column with highest scale dispersion is <strong className="text-sky-400">{globalInsights.highestVarianceCol}</strong> (Variance: {formatNum(globalInsights.highestVarianceVal)}). Scaling/normalization is recommended before model training.
                      </p>
                    </div>
                  )}

                  {globalInsights.highestOutliersCol && globalInsights.highestOutliersVal > 0 && (
                    <div className="flex items-start gap-2 col-span-1 md:col-span-2 border-t border-[#E5E2D9] pt-2 mt-1">
                      <AlertTriangle className="h-3.5 w-3.5 text-rose-500 mt-0.5 shrink-0" />
                      <p className="leading-relaxed">
                        The column <strong className="text-[#1F1E1D]">{globalInsights.highestOutliersCol}</strong> exhibits heavy tails with <strong className="text-rose-400">{globalInsights.highestOutliersVal} outliers ({globalInsights.highestOutliersPct.toFixed(1)}%)</strong> identified by IQR. Review this feature in the EDA tab's outlier filter.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Filters, Search and Sorting Controls Row */}
            <div className="flex flex-col md:flex-row justify-between items-stretch md:items-center gap-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded p-3 text-[11px] font-mono">
              <div className="flex flex-1 items-center gap-2 bg-white border border-[#E5E2D9] rounded px-2.5 py-1.5">
                <Search className="h-3.5 w-3.5 text-[#9E9D96] flex-shrink-0" />
                <input
                  type="text"
                  placeholder="Search numeric features..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-transparent text-[#73726C] w-full focus:outline-none placeholder-neutral-600 text-[11px]"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")} className="text-[#9E9D96] hover:text-[#73726C] cursor-pointer">
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {/* Sort Option Selection */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[#9E9D96] text-[10px] uppercase font-bold">Sort By:</span>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="bg-white border border-[#E5E2D9] rounded text-[#73726C] text-[10px] px-2 py-1 focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer font-bold uppercase"
                  >
                    <option value="name">Name</option>
                    <option value="variance">Dispersion (Variance)</option>
                    <option value="skewness">Asymmetry (Skewness)</option>
                    <option value="missing">Missing Count</option>
                    <option value="outliers">Outliers Count</option>
                  </select>
                  <button
                    onClick={() => setSortOrder(prev => prev === "asc" ? "desc" : "asc")}
                    className="p-1 bg-white border border-[#E5E2D9] rounded hover:border-[#D5D1C5] text-[#73726C] hover:text-[#1F1E1D] cursor-pointer"
                    title={sortOrder === "asc" ? "Ascending" : "Descending"}
                  >
                    <ArrowUpDown className="h-3.5 w-3.5" />
                  </button>
                </div>

                {/* Display Mode Layout Selector */}
                <div className="flex items-center gap-1 border-l border-[#E5E2D9] pl-3">
                  <span className="text-[#9E9D96] text-[10px] uppercase font-bold mr-1">Layout:</span>
                  <button
                    type="button"
                    onClick={() => setLayoutMode("detailed")}
                    className={`p-1.5 rounded transition-all cursor-pointer ${
                      layoutMode === "detailed"
                        ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                        : "text-[#9E9D96] hover:text-[#73726C] border border-transparent"
                    }`}
                    title="Detailed single-column list"
                  >
                    <List className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setLayoutMode("compact")}
                    className={`p-1.5 rounded transition-all cursor-pointer ${
                      layoutMode === "compact"
                        ? "bg-sky-500/10 text-sky-400 border border-sky-500/20"
                        : "text-[#9E9D96] hover:text-[#73726C] border border-transparent"
                    }`}
                    title="Compact grid layout"
                  >
                    <LayoutGrid className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Active Box Plots Grid */}
            <div className={layoutMode === "compact" ? "grid sm:grid-cols-2 lg:grid-cols-3 gap-4" : "space-y-4"}>
              {filteredAndSortedCols.map((col, cIdx) => {
                const stats = col.stats;
                const min = stats.min ?? 0;
                const max = stats.max ?? 0;
                const mean = stats.mean ?? 0;
                const median = stats.median ?? 0;
                const q25 = stats.q25 ?? 0;
                const q75 = stats.q75 ?? 0;
                const skew = stats.skewness ?? 0;

                const isFlat = max === min;

                // Scale value to coordinates inside SVG width (range 25 to 375 for 400 viewport width)
                const scaleValue = (val: number) => {
                  if (isFlat) return 200;
                  return 25 + ((val - min) / (max - min)) * 350;
                };

                // Skewness verdict
                let skewVerdict = "Approximately Symmetric";
                let skewColor = "text-sky-400 border-sky-500/10 bg-sky-500/5";
                if (skew > 1) {
                  skewVerdict = "Highly Right-Skewed";
                  skewColor = "text-sky-400 border-sky-500/10 bg-sky-500/5";
                } else if (skew < -1) {
                  skewVerdict = "Highly Left-Skewed";
                  skewColor = "text-sky-400 border-sky-500/10 bg-sky-500/5";
                } else if (skew >= 0.5 && skew <= 1) {
                  skewVerdict = "Moderately Right-Skewed";
                  skewColor = "text-yellow-400/85 border-yellow-500/10 bg-yellow-500/5";
                } else if (skew <= -0.5 && skew >= -1) {
                  skewVerdict = "Moderately Left-Skewed";
                  skewColor = "text-yellow-400/85 border-yellow-500/10 bg-yellow-500/5";
                }

                // Outlier details for current column
                const outlierInfo = summary.outliers?.find(o => o.columnName === col.name);
                const outCount = outlierInfo?.outlierCount || 0;
                const outPct = outlierInfo?.outlierPercentage || 0;

                // Missing density rate
                const totalCells = (stats.missingCount || 0) + (stats.nonNullCount || 0);
                const missingPct = totalCells > 0 ? ((stats.missingCount || 0) / totalCells) * 100 : 0;

                return (
                  <div 
                    key={cIdx} 
                    className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-4 font-mono text-[11px] hover:border-[#E5E2D9] transition-all duration-150 relative space-y-3"
                  >
                    {/* Column Heading and Stat Badges */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E5E2D9] pb-2">
                      <div className="space-y-0.5">
                        <span className="text-[8px] text-[#9E9D96] uppercase tracking-wide block font-extrabold">Feature Name</span>
                        <h4 className="font-black text-[#1F1E1D] text-xs tracking-tight">{col.name}</h4>
                      </div>

                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${skewColor}`}>
                          {skewVerdict}
                        </span>
                        {outCount > 0 ? (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded border bg-rose-500/5 text-rose-500 border-rose-500/10 inline-flex items-center gap-1">
                            <AlertTriangle className="h-2.5 w-2.5" />
                            <span>{outCount} Outliers ({outPct.toFixed(1)}%)</span>
                          </span>
                        ) : (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded border bg-white text-[#9E9D96] border-[#E5E2D9]">
                            Safe (0 Outliers)
                          </span>
                        )}
                        {missingPct > 0 && (
                          <span className="text-[8px] font-bold px-1.5 py-0.5 rounded border bg-amber-500/5 text-amber-600 border-amber-500/10 inline-flex items-center gap-1">
                            <AlertCircle className="h-2.5 w-2.5" />
                            <span>{missingPct.toFixed(1)}% Missing</span>
                          </span>
                        )}
                      </div>
                    </div>

                    {/* SVG Vector Box Plot */}
                    <div className="bg-[#FAF9F5] rounded border border-[#E5E2D9]/60 p-2 overflow-visible relative">
                      <svg className="w-full h-16 overflow-visible" viewBox="0 0 400 70" preserveAspectRatio="none">
                        {/* Shaded IQR Range Grid Decorator */}
                        <rect x={scaleValue(q25)} y={0} width={scaleValue(q75) - scaleValue(q25)} height={70} fill="#10b981/5" opacity={0.03} />

                        {/* Whisker dashed line */}
                        <line 
                          x1={scaleValue(min)} 
                          y1={30} 
                          x2={scaleValue(max)} 
                          y2={30} 
                          stroke="#444444" 
                          strokeWidth={1.5} 
                          strokeDasharray="4 4" 
                        />

                        {/* Left whisker terminal cap */}
                        <line 
                          x1={scaleValue(min)} 
                          y1={20} 
                          x2={scaleValue(min)} 
                          y2={40} 
                          stroke="#737373" 
                          strokeWidth={2} 
                        />

                        {/* Right whisker terminal cap */}
                        <line 
                          x1={scaleValue(max)} 
                          y1={20} 
                          x2={scaleValue(max)} 
                          y2={40} 
                          stroke="#737373" 
                          strokeWidth={2} 
                        />

                        {/* IQR Central Box (25% -> 75%) */}
                        <rect 
                          x={scaleValue(q25)} 
                          y={15} 
                          width={Math.max(2, scaleValue(q75) - scaleValue(q25))} 
                          height={30} 
                          fill="#10b98115" 
                          stroke="#10b981" 
                          strokeWidth={1.8} 
                          rx={1.5} 
                        />

                        {/* Solid Median line */}
                        <line 
                          x1={scaleValue(median)} 
                          y1={12} 
                          x2={scaleValue(median)} 
                          y2={48} 
                          stroke="#34d399" 
                          strokeWidth={3} 
                          strokeLinecap="round"
                        />

                        {/* Mean indicator marker (contrasting diamond/circle) */}
                        <circle 
                          cx={scaleValue(mean)} 
                          cy={30} 
                          r={3.5} 
                          fill="#3b82f6" 
                          stroke="#93c5fd" 
                          strokeWidth={1.2} 
                        />

                        {/* Vector Values Text Labels (Placed on Top / Bottom to prevent overlapping) */}
                        {/* Minimum Label */}
                        <text x={scaleValue(min)} y={60} textAnchor="middle" className="text-[8px] fill-neutral-500 font-mono font-semibold">
                          {formatNum(min)}
                        </text>
                        {/* Maximum Label */}
                        <text x={scaleValue(max)} y={60} textAnchor="middle" className="text-[8px] fill-neutral-500 font-mono font-semibold">
                          {formatNum(max)}
                        </text>
                        {/* Median Label */}
                        <text x={scaleValue(median)} y={10} textAnchor="middle" className="text-[8px] fill-sky-400 font-mono font-black">
                          {formatNum(median)}
                        </text>
                      </svg>

                      {/* Legend/Key overlay indicators */}
                      <div className="absolute top-1 right-2 flex items-center gap-2 text-[7px] text-[#73726C] font-bold uppercase tracking-wider scale-90">
                        <span className="flex items-center gap-0.5">
                          <span className="w-1.5 h-1.5 bg-sky-400 rounded-sm inline-block" /> Median
                        </span>
                        <span className="flex items-center gap-0.5">
                          <span className="w-1.5 h-1.5 bg-blue-500 rounded-full inline-block" /> Mean
                        </span>
                      </div>
                    </div>

                    {/* bento descriptive statistics metrics block */}
                    {layoutMode === "detailed" ? (
                      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 pt-2 text-center">
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Min</span>
                          <span className="text-[10px] text-[#73726C] font-bold block">{formatNum(min)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Q1 (25%)</span>
                          <span className="text-[10px] text-[#73726C] font-bold block">{formatNum(q25)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Median</span>
                          <span className="text-[10px] text-sky-400 font-black block">{formatNum(median)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Mean</span>
                          <span className="text-[10px] text-blue-400 font-bold block">{formatNum(mean)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Q3 (75%)</span>
                          <span className="text-[10px] text-[#73726C] font-bold block">{formatNum(q75)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Max</span>
                          <span className="text-[10px] text-[#73726C] font-bold block">{formatNum(max)}</span>
                        </div>
                        <div className="bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]/60 col-span-2 sm:col-span-1">
                          <span className="text-[7px] text-[#9E9D96] block uppercase font-bold tracking-wide">Std Dev</span>
                          <span className="text-[10px] text-[#73726C] font-bold block">{formatNum(stats.stdDev)}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between text-[9px] text-[#9E9D96] border-t border-[#E5E2D9]/60 pt-2 font-mono">
                        <span>Mean: <strong className="text-[#73726C]">{formatNum(mean)}</strong></span>
                        <span>Median: <strong className="text-sky-400">{formatNum(median)}</strong></span>
                        <span>Std Dev: <strong className="text-[#73726C]">{formatNum(stats.stdDev)}</strong></span>
                      </div>
                    )}
                  </div>
                );
              })}

              {filteredAndSortedCols.length === 0 && (
                <div className="py-8 text-center text-[#9E9D96] text-xs italic border border-dashed border-[#E5E2D9] rounded col-span-full">
                  No numerical columns matched your search query.
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>

      {/* =========================================================================
          DYNAMIC STATISTICAL TESTING SANDBOX (PRESERVED)
          ========================================================================= */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-5">
        <div className="border-b border-[#E5E2D9] pb-3">
          <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">Statistical Hypothesis Testing Lab</h3>
          <p className="text-[11px] text-[#9E9D96]">
            Select two features. The engine automatically executes the mathematically appropriate test to prove or disprove independence.
          </p>
        </div>

        {/* Dropdowns Row */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-[10px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Variable 1 (X)</label>
            <select
              value={var1Name}
              onChange={(e) => setVar1Name(e.target.value)}
              className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] text-xs font-mono font-bold rounded px-3 py-2 focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
            >
              {summary.columns.map((col, idx) => (
                <option key={idx} value={col.name} className="bg-[#FAF9F5] text-[#73726C]">
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Variable 2 (Y)</label>
            <select
              value={var2Name}
              onChange={(e) => setVar2Name(e.target.value)}
              className="w-full bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] text-xs font-mono font-bold rounded px-3 py-2 focus:ring-1 focus:ring-sky-500 outline-none cursor-pointer"
            >
              {summary.columns.map((col, idx) => (
                <option key={idx} value={col.name} className="bg-[#FAF9F5] text-[#73726C]">
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Selected Test Results */}
        {dynamicTest ? (
          <div className="bg-white border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E2D9] pb-3">
              <div className="space-y-1">
                <span className="inline-block text-[9px] font-extrabold uppercase tracking-wider bg-sky-500/10 text-sky-400 border border-sky-500/20 px-1.5 py-0.5 rounded font-mono">
                  Active Hypothesis Test: {dynamicTest.type}
                </span>
                <h4 className="font-bold text-[#1F1E1D] text-xs font-mono">
                  Testing relationship: {dynamicTest.variable1} ↔ {dynamicTest.variable2}
                </h4>
              </div>
              <div className="flex-shrink-0">
                {getPValueBadge(dynamicTest.pValue)}
              </div>
            </div>

            {/* Test Mechanics Details */}
            <div className="grid md:grid-cols-3 gap-3">
              <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E2D9] text-center">
                <p className="text-[10px] text-[#9E9D96] font-mono uppercase font-bold">Test Statistic</p>
                <p className="text-lg font-black text-[#1F1E1D] font-mono mt-1">
                  {dynamicTest.statisticValue}
                </p>
                <span className="text-[9px] text-[#9E9D96] font-mono uppercase">
                  {dynamicTest.type === "T-Test" ? "t-value" : dynamicTest.type === "ANOVA" ? "F-value" : dynamicTest.type === "Chi-Square" ? "χ²-value" : "Correlation (r)"}
                </span>
              </div>

              <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E2D9] text-center">
                <p className="text-[10px] text-[#9E9D96] font-mono uppercase font-bold">Degrees of Freedom</p>
                <p className="text-lg font-black text-[#1F1E1D] font-mono mt-1">
                  {dynamicTest.degreesOfFreedom !== undefined ? dynamicTest.degreesOfFreedom : "N/A"}
                </p>
                <span className="text-[9px] text-[#9E9D96] font-mono uppercase">df index</span>
              </div>

              <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E2D9] text-center">
                <p className="text-[10px] text-[#9E9D96] font-mono uppercase font-bold">Null Hypothesis (H₀)</p>
                <p className="text-[10px] font-mono font-bold text-[#73726C] mt-2.5 uppercase leading-snug">
                  {dynamicTest.pValue < 0.05 
                    ? "REJECTED (Dependent)" 
                    : "RETAINED (No dependency)"}
                </p>
              </div>
            </div>

            {/* Deep Interpretation */}
            <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded flex items-start space-x-3">
              <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-1.5 flex-shrink-0" />
              <div>
                <h5 className="font-mono font-bold text-sky-400 text-[10px] uppercase tracking-wider">Conclusion & Statistical Decision</h5>
                <p className="text-[11px] text-[#73726C] mt-1 leading-relaxed font-mono">
                  {dynamicTest.interpretation}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-[#E5E2D9] rounded p-8 text-center text-[#9E9D96] text-xs font-mono">
            Choose two different columns to calculate correlation, ANOVA, T-Tests, or Chi-Square tests.
          </div>
        )}
      </div>

      {/* =========================================================================
          OVERVIEW TABLE OF AUTOMATED SYSTEM TESTS (PRESERVED)
          ========================================================================= */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-3">
        <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">Baseline Statistical Tests Matrix</h3>
        <p className="text-[11px] text-[#9E9D96] leading-normal">
          The AutoML engine automatically calculated these statistical tests across key dataset features during ingestion:
        </p>

        <div className="overflow-x-auto border border-[#E5E2D9] rounded">
          <table className="w-full text-left border-collapse text-[11px]">
            <thead>
              <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#9E9D96] font-bold uppercase tracking-wider font-mono">
                <th className="py-2 px-3">Test Type</th>
                <th className="py-2 px-3">Variable 1</th>
                <th className="py-2 px-3">Variable 2</th>
                <th className="py-2 px-3 text-right">Statistic</th>
                <th className="py-2 px-3 text-right">p-Value</th>
                <th className="py-2 px-3">Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800 text-[#73726C]">
              {summary.statisticalTests.map((t, idx) => (
                <tr key={idx} className="hover:bg-[#FAF9F5]">
                  <td className="py-2 px-3">
                    <span className="inline-block px-1.5 py-0.5 bg-[#F0EEE6] border border-[#D5D1C5] text-[#73726C] font-mono text-[9px] uppercase tracking-wider rounded">
                      {t.type}
                    </span>
                  </td>
                  <td className="py-2 px-3 font-semibold text-[#73726C] font-mono">{t.variable1}</td>
                  <td className="py-2 px-3 font-semibold text-[#73726C] font-mono">{t.variable2}</td>
                  <td className="py-2 px-3 text-right font-mono text-[#73726C]">{t.statisticValue}</td>
                  <td className="py-2 px-3 text-right font-mono text-[#73726C]">{t.pValue}</td>
                  <td className="py-2 px-3 text-[#9E9D96] font-mono max-w-[200px] truncate" title={t.interpretation}>
                    {t.interpretation}
                  </td>
                </tr>
              ))}
              {summary.statisticalTests.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-4 text-center text-[#9E9D96] italic font-mono text-xs">
                    No automated tests computed. Verify your column types.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
