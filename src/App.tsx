import React, { useState, useEffect } from "react";
import DataUpload, { calculateQualityScore } from "./components/DataUpload";
import OverviewTab from "./components/OverviewTab";
import EdaTab from "./components/EdaTab";
import StatsTab from "./components/StatsTab";
import MlTab from "./components/MlTab";
import FeatureEngTab from "./components/FeatureEngTab";
import PolarsCleaningTab from "./components/PolarsCleaningTab";
import LandingPage from "./components/LandingPage";
import SettingsModal from "./components/SettingsModal";
import AdminHealthDashboard from "./components/AdminHealthDashboard";
import { DatasetSummary, AIAnalysisResult, ColumnInfo, TrainedModelExportInfo, BatchProcessingReport, AIModelConfig } from "./types";
import { computeColumnStats, computeCorrelationMatrix, runStatisticalTests, detectDatasetOutliers } from "./utils/stats";
import { inferColumnSensitivity } from "./utils/sensitivity";
import { generatePdfReport } from "./utils/pdf";
import { signOutGoogle } from "./utils/firebaseAuth";
import { processDatasetInBatches } from "./utils/batchProcessor";
import { 
  FileText, BarChart3, Binary, Settings, Sparkles, Download, 
  RefreshCw, Loader2, HelpCircle, LogOut, Zap, PanelLeft, 
  PanelLeftClose, Plus, Database, CheckCircle2,
  FileSpreadsheet, SlidersHorizontal, X, Shield, ArrowRight, Activity, Layers
} from "lucide-react";

// Client-Side In-Memory Cache for Dataset Analysis Results
// Prevents duplicate Gemini API calls for identical datasets
const memoryAnalysisCache = new Map<string, AIAnalysisResult>();

// Deterministic fingerprint key based on dataset signature
const getAnalysisCacheKey = (fileName: string, rowCount: number, colCount: number, qualityScore: number): string => {
  return `automl_analysis_cache_${fileName}_${rowCount}_${colCount}_${qualityScore}`;
};

// Retrieve cached analysis from Memory or localStorage
const getCachedAnalysis = (key: string): AIAnalysisResult | null => {
  // 1. In-memory check (0ms, 0 overhead)
  if (memoryAnalysisCache.has(key)) {
    return memoryAnalysisCache.get(key)!;
  }
  // 2. localStorage check (persists across page reloads)
  try {
    const raw = localStorage.getItem(key);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && (parsed.columnInsights || parsed.overallConclusions)) {
        memoryAnalysisCache.set(key, parsed);
        return parsed;
      }
    }
  } catch (err) {
    console.warn("Failed reading analysis from localStorage cache:", err);
  }
  return null;
};

// Persist analysis result into Memory and localStorage
const setCachedAnalysis = (key: string, data: AIAnalysisResult): void => {
  memoryAnalysisCache.set(key, data);
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (err) {
    console.warn("Failed saving analysis to localStorage cache:", err);
  }
};

export default function App() {
  // Pure Clean Light Mode only (No dark mode)
  useEffect(() => {
    document.documentElement.classList.remove("dark");
    try {
      localStorage.removeItem("theme_mode");
      localStorage.removeItem("theme_mode_workbench");
      localStorage.removeItem("theme_mode_setting");
      localStorage.removeItem("theme_mode_v2");
    } catch {}
  }, []);

  // Sidebar & Insights Drawer layout states
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);
  const [isInsightsOpen, setIsInsightsOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);

  // Session State
  const [userSession, setUserSession] = useState<string | null>(() => {
    try {
      return localStorage.getItem("automl_active_session");
    } catch {
      return null;
    }
  });

  // Admin View State
  const isAdmin = userSession?.toLowerCase() === "admin@gmail.com";
  const [isAdminView, setIsAdminView] = useState<boolean>(false);

  const handleLogout = async () => {
    try {
      localStorage.removeItem("automl_active_session");
      await signOutGoogle();
    } catch {}
    setUserSession(null);
    setIsAdminView(false);
    handleReset();
  };

  // State
  const [fileName, setFileName] = useState<string | null>(null);
  const [rawData, setRawData] = useState<any[] | null>(null);
  const [originalRawData, setOriginalRawData] = useState<any[] | null>(null);
  const [isDataCleaned, setIsDataCleaned] = useState<boolean>(false);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  
  // Tab control
  const [activeTab, setActiveTab] = useState<"overview" | "eda" | "stats" | "polars" | "feature-eng" | "ml">("overview");

  // Custom engineered features metadata
  const [engineeredFeatures, setEngineeredFeatures] = useState<Array<{ name: string; formula: string }>>([]);

  // Active Model & AI Provider State
  const [activeModelConfig, setActiveModelConfig] = useState<AIModelConfig | null>(null);

  const fetchActiveModelConfig = async () => {
    try {
      const res = await fetch("/api/admin/model-config");
      if (res.ok) {
        const data = await res.json();
        setActiveModelConfig(data);
      }
    } catch {}
  };

  useEffect(() => {
    fetchActiveModelConfig();
  }, [isAdminView]);

  // Gemini & Hugging Face AI Analysis
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [aiInsights, setAiInsights] = useState<AIAnalysisResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [isAnalysisFromCache, setIsAnalysisFromCache] = useState<boolean>(false);

  // Active Trained Model export data
  const [trainedModelInfo, setTrainedModelInfo] = useState<TrainedModelExportInfo | null>(null);

  // Centralised reusable function to recalculate descriptive stats, correlations, tests, outliers
  const rebuildSummary = (updatedRows: any[], customScore?: number) => {
    if (updatedRows.length === 0) return;
    
    const headers = Object.keys(updatedRows[0]);
    const cols: ColumnInfo[] = headers.map(h => {
      const samples = updatedRows
        .map(r => r[h])
        .filter(v => v !== undefined && v !== null && v !== "");
      
      let type: ColumnInfo["type"] = "Categorical";
      const numericValues = samples.map(v => Number(v)).filter(v => !isNaN(v));

      if (samples.length > 0 && numericValues.length === samples.length) {
        type = "Numeric";
      } else if (samples.some(v => String(v).includes("-") && !isNaN(Date.parse(String(v))))) {
        type = "Date";
      } else if (samples.some(v => String(v).length > 50)) {
        type = "Text";
      }

      const stats = computeColumnStats(updatedRows.map(r => r[h]), type);
      const sensitivity = inferColumnSensitivity(h, samples);

      return {
        name: h,
        type,
        stats,
        sensitivity
      };
    });

    const numericColNames = cols.filter(c => c.type === "Numeric").map(c => c.name);
    const correlation = computeCorrelationMatrix(updatedRows, numericColNames);
    const statisticalTests = runStatisticalTests(updatedRows, cols);
    const outliers = detectDatasetOutliers(updatedRows, cols);
    const sampleData = updatedRows.slice(0, 15);

    const quality = customScore !== undefined ? customScore : calculateQualityScore(updatedRows);

    const datasetSummary: DatasetSummary = {
      fileName: fileName || "Modified Dataset",
      rowCount: updatedRows.length,
      colCount: cols.length,
      columns: cols,
      correlation,
      statisticalTests,
      sampleData,
      dataQualityScore: quality,
      outliers,
      batchReport: summary?.batchReport
    };

    setRawData(updatedRows);
    setSummary(datasetSummary);
  };

  // Handler for committing Polars-cleaned dataset into workspace
  const handleApplyCleanedData = (cleanedRows: any[]) => {
    const newScore = calculateQualityScore(cleanedRows);
    setIsDataCleaned(true);
    rebuildSummary(cleanedRows, newScore);

    // Record telemetry for admin metrics
    try {
      const rawSize = cleanedRows.length * (summary?.colCount || 5) * 8;
      fetch("/api/admin/record-telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fileName: fileName || "Cleaned_Dataset.csv",
          rowCount: cleanedRows.length,
          colCount: summary?.colCount || 5,
          fileSizeBytes: rawSize,
          status: "Cleaned",
          user: userSession || "analyst@gmail.com",
          latencyMs: 140,
          cleaningSteps: ["Polars Pipeline Filtering", "Null Imputation", "Type Normalization"],
          qualityScoreBefore: summary?.dataQualityScore || 80,
          qualityScoreAfter: newScore
        })
      }).catch(() => {});
    } catch {}
  };

  // Handler for reverting to original uncleaned raw dataset
  const handleRevertToOriginal = () => {
    if (originalRawData && originalRawData.length > 0) {
      const origScore = calculateQualityScore(originalRawData);
      setIsDataCleaned(false);
      rebuildSummary(originalRawData, origScore);
    }
  };

  const handleAddFeature = (featureName: string, formula: string, updatedRows: any[]) => {
    setEngineeredFeatures(prev => [...prev, { name: featureName, formula }]);
    rebuildSummary(updatedRows);
  };

  const handleRemoveFeature = (featureName: string) => {
    if (!rawData) return;
    setEngineeredFeatures(prev => prev.filter(f => f.name !== featureName));
    
    const updatedRows = rawData.map(row => {
      const copy = { ...row };
      delete copy[featureName];
      return copy;
    });

    rebuildSummary(updatedRows);
  };

  // Re-chunk and re-batch the active dataset with user-selected chunk size
  const handleRebatchDataset = async (newChunkSize: number) => {
    if (!rawData || !summary) return;
    try {
      const { batchReport } = await processDatasetInBatches({
        rows: rawData,
        fileName: summary.fileName,
        chunkSize: newChunkSize,
      });
      setSummary(prev => prev ? ({ ...prev, batchReport }) : null);
    } catch (err) {
      console.error("Rebatch error:", err);
    }
  };

  // Parse raw dataset rows into structured DatasetSummary client-side
  const handleDataLoaded = async (
    fName: string, 
    parsedRows: any[], 
    dataQualityScore: number, 
    initialTab?: string,
    batchReport?: BatchProcessingReport
  ) => {
    setFileName(fName);
    setRawData(parsedRows);
    setOriginalRawData(parsedRows);
    setIsDataCleaned(false);
    setAiInsights(null);
    setAnalysisError(null);
    if (initialTab && ["overview", "eda", "stats", "polars", "feature-eng", "ml"].includes(initialTab)) {
      setActiveTab(initialTab as any);
    } else {
      setActiveTab("overview");
    }

    try {
      if (parsedRows.length === 0) {
        throw new Error("Uploaded dataset contains 0 records.");
      }

      const headers = Object.keys(parsedRows[0]);
      
      const cols: ColumnInfo[] = headers.map(h => {
        const samples = parsedRows
          .map(r => r[h])
          .filter(v => v !== undefined && v !== null && v !== "");
        
        let type: ColumnInfo["type"] = "Categorical";
        const numericValues = samples.map(v => Number(v)).filter(v => !isNaN(v));

        if (samples.length > 0 && numericValues.length === samples.length) {
          type = "Numeric";
        } else if (samples.some(v => String(v).includes("-") && !isNaN(Date.parse(String(v))))) {
          type = "Date";
        } else if (samples.some(v => String(v).length > 50)) {
          type = "Text";
        }

        const stats = computeColumnStats(parsedRows.map(r => r[h]), type);
        const sensitivity = inferColumnSensitivity(h, samples);

        return {
          name: h,
          type,
          stats,
          sensitivity
        };
      });

      const numericColNames = cols.filter(c => c.type === "Numeric").map(c => c.name);
      const correlation = computeCorrelationMatrix(parsedRows, numericColNames);
      const statisticalTests = runStatisticalTests(parsedRows, cols);
      const outliers = detectDatasetOutliers(parsedRows, cols);
      const sampleData = parsedRows.slice(0, 15);

      const datasetSummary: DatasetSummary = {
        fileName: fName,
        rowCount: parsedRows.length,
        colCount: cols.length,
        columns: cols,
        correlation,
        statisticalTests,
        sampleData,
        dataQualityScore,
        outliers,
        batchReport
      };

      setSummary(datasetSummary);

      // Check client-side memory or localStorage cache before firing Gemini API call
      const cacheKey = getAnalysisCacheKey(fName, parsedRows.length, cols.length, dataQualityScore);
      const cachedResult = getCachedAnalysis(cacheKey);

      if (cachedResult) {
        console.log(`[Client Cache Hit] Reusing cached analysis for '${fName}' - 0 Gemini tokens consumed`);
        setAiInsights(cachedResult);
        setIsAnalysisFromCache(true);
        setIsAnalyzing(false);
        return;
      }

      // Record uploaded dataset and system costing in Admin Panel (keep only name and system metrics, not whole dataset)
      try {
        const estPromptTokens = Math.round(550 + cols.length * 65);
        const estCandidateTokens = 280;
        const estTokens = estPromptTokens + estCandidateTokens;
        const estCost = ((estPromptTokens * 0.075) / 1000000) + ((estCandidateTokens * 0.30) / 1000000);
        const rawSizeBytes = parsedRows.length * cols.length * 8;

        fetch("/api/admin/record-telemetry", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fileName: fName,
            rowCount: parsedRows.length,
            colCount: cols.length,
            fileSizeBytes: rawSizeBytes,
            status: "Uploaded",
            promptTokens: estPromptTokens,
            candidateTokens: estCandidateTokens,
            tokensUsed: estTokens,
            costUsd: estCost,
            user: userSession || "analyst@automl.studio",
            latencyMs: 95,
            qualityScoreBefore: dataQualityScore
          })
        }).catch(err => console.warn("Admin telemetry recording skipped:", err));
      } catch (err) {
        console.warn("Failed to record upload telemetry:", err);
      }

      setIsAnalysisFromCache(false);
      setIsAnalyzing(true);
      const response = await fetch("/api/eda/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ datasetSummary, userEmail: userSession })
      });
      
      if (response.ok) {
        const aiData = await response.json();
        if (aiData.columnInsights) {
          setAiInsights(aiData);
          // Save to both Memory and localStorage
          setCachedAnalysis(cacheKey, aiData);
        }
      } else {
        const errorData = await response.json();
        setAnalysisError(errorData.error || "Gemini was unable to parse this file structure.");
      }
    } catch (err: any) {
      console.error(err);
      setAnalysisError(err.message || "Failed to parse dataframe structure.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Re-run Gemini analysis on-demand (bypassing client cache)
  const handleForceReAnalyze = async () => {
    if (!summary) return;
    const cacheKey = getAnalysisCacheKey(summary.fileName, summary.rowCount, summary.colCount, summary.dataQualityScore);
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const response = await fetch("/api/eda/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ datasetSummary: summary, userEmail: userSession })
      });
      if (response.ok) {
        const aiData = await response.json();
        if (aiData.columnInsights) {
          setAiInsights(aiData);
          setCachedAnalysis(cacheKey, aiData);
          setIsAnalysisFromCache(false);
        }
      } else {
        const errorData = await response.json();
        setAnalysisError(errorData.error || "Failed to re-analyze dataset.");
      }
    } catch (err: any) {
      setAnalysisError(err.message || "Failed to re-analyze dataset.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Reset app to file upload screen
  const handleReset = () => {
    setFileName(null);
    setRawData(null);
    setOriginalRawData(null);
    setIsDataCleaned(false);
    setSummary(null);
    setAiInsights(null);
    setAnalysisError(null);
    setEngineeredFeatures([]);
    setTrainedModelInfo(null);
    setActiveTab("overview");
  };

  // Trigger PDF Report Generation with comprehensive data and graphs from all tabs
  const handleExportPdf = () => {
    if (summary) {
      generatePdfReport(
        summary,
        aiInsights,
        trainedModelInfo?.results ?? null,
        trainedModelInfo?.modelName ?? null,
        trainedModelInfo,
        rawData,
        engineeredFeatures,
        isDataCleaned,
        originalRawData
      );
    }
  };

  if (!userSession) {
    return <LandingPage onLoginSuccess={(email) => {
      setUserSession(email);
      try {
        localStorage.setItem("automl_active_session", email);
      } catch {}
    }} />;
  }

  // If user is Admin and Admin Health View is activated, render the dedicated Admin Console
  if (isAdmin && isAdminView) {
    return (
      <AdminHealthDashboard 
        onBackToStudio={() => {
          setIsAdminView(false);
          fetchActiveModelConfig();
        }} 
        adminEmail={userSession}
        onModelConfigChanged={fetchActiveModelConfig}
      />
    );
  }

  // Active module title for breadcrumbs
  const tabTitles: Record<string, string> = {
    overview: "Overview & Matrix",
    eda: "Exploratory EDA",
    stats: "Hypothesis Tests",
    polars: "Polars Data Cleaning",
    "feature-eng": "Feature Engineering",
    ml: "AutoML Trainer"
  };

  return (
    <div className="flex h-screen bg-[#FAF9F5] text-[#1F1E1D] font-sans overflow-hidden">
      
      {/* Mobile Drawer Overlay Backdrop */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/25 backdrop-blur-xs z-40 md:hidden transition-opacity"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* ========================================================================= */}
      {/* 1. PROFESSIONAL WORKBENCH SIDEBAR */}
      {/* ========================================================================= */}
      <aside 
        className={`${
          isSidebarOpen 
            ? "translate-x-0 w-64" 
            : "-translate-x-full md:translate-x-0 md:w-16 w-0"
        } fixed md:static inset-y-0 left-0 z-50 transition-all duration-200 ease-in-out bg-[#F0EEE6] border-r border-[#E5E2D9] flex flex-col justify-between shrink-0 overflow-hidden`}
      >
        <div className="p-3.5 space-y-4 flex flex-col min-h-0 h-full">
          
          {/* Brand & New Analysis Button */}
          <div className="space-y-3">
            <div className="flex items-center justify-between px-1 pt-1">
              <div className="flex items-center space-x-2.5 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7] shrink-0">
                  <Database className="w-4 h-4" />
                </div>
                {isSidebarOpen && (
                  <div className="min-w-0">
                    <span className="font-serif text-sm font-medium tracking-tight text-[#1F1E1D] block truncate">
                      AutoML Studio
                    </span>
                    <span className="text-[10px] text-[#73726C] block -mt-0.5">
                      Analytics Workbench
                    </span>
                  </div>
                )}
              </div>

              {isSidebarOpen && (
                <button
                  onClick={() => setIsSidebarOpen(false)}
                  className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] rounded-md hover:bg-[#E8E5DC] transition-colors cursor-pointer"
                  title="Collapse sidebar"
                >
                  <PanelLeftClose className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* "+ New Dataset / Upload" */}
            <button
              onClick={handleReset}
              className={`w-full flex items-center rounded-xl bg-white border border-[#E5E2D9] hover:border-[#0284C7] text-xs font-medium text-[#1F1E1D] transition-all shadow-xs cursor-pointer group ${
                isSidebarOpen ? "justify-between px-3 py-2" : "justify-center p-2"
              }`}
              title="Upload New Dataset"
            >
              <div className="flex items-center space-x-2">
                <Plus className="h-4 w-4 text-[#0284C7]" />
                {isSidebarOpen && <span>New Dataset</span>}
              </div>
              {isSidebarOpen && (
                <span className="text-[10px] text-[#73726C] font-mono">
                  Import
                </span>
              )}
            </button>
          </div>

          {/* Admin System Health Button (Visible for admin@gmail.com) */}
          {isAdmin && (
            <div className="px-1">
              <button
                onClick={() => setIsAdminView(true)}
                className={`w-full flex items-center rounded-xl bg-[#F0F9FF] border border-[#BAE6FD] hover:bg-[#BAE6FD]/40 text-xs font-medium text-[#0284C7] transition-all shadow-xs cursor-pointer ${
                  isSidebarOpen ? "justify-between px-3 py-2" : "justify-center p-2"
                }`}
                title="Open Admin System Health Console"
              >
                <div className="flex items-center space-x-2">
                  <Shield className="h-4 w-4 text-[#0284C7]" />
                  {isSidebarOpen && <span>System Health & Ops</span>}
                </div>
                {isSidebarOpen && (
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#0284C7] text-white">
                    Admin
                  </span>
                )}
              </button>
            </div>
          )}

          {/* Recents / Active Dataset Status */}
          <div className="space-y-1.5 flex-1 overflow-y-auto pr-0.5">
            {isSidebarOpen && (
              <span className="px-2 text-[10px] font-medium uppercase tracking-wider text-[#9E9D96]">
                Workspace
              </span>
            )}

            {summary ? (
              <div className="space-y-1">
                {/* Active Dataset Item */}
                {isSidebarOpen && (
                  <div className="px-3 py-2.5 rounded-xl bg-white border border-[#E5E2D9] text-xs space-y-1.5 shadow-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-[#1F1E1D] truncate max-w-[130px]" title={summary.fileName}>
                        {summary.fileName}
                      </span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-sky-500/10 text-sky-600 font-medium">
                        Active
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-[10px] text-[#73726C]">
                      <span>{summary.rowCount.toLocaleString()} rows</span>
                      <span>·</span>
                      <span>{summary.colCount} cols</span>
                    </div>
                    {isDataCleaned && (
                      <div className="pt-0.5 flex items-center gap-1 text-[9px] text-[#0284C7] font-medium">
                        <Zap className="h-2.5 w-2.5" />
                        <span>Polars Cleaned</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Direct Tab Navigation Links */}
                <div className={`space-y-0.5 ${isSidebarOpen ? "pt-2" : "pt-1"}`}>
                  {isSidebarOpen && (
                    <span className="px-2 text-[10px] font-medium uppercase tracking-wider text-[#9E9D96]">
                      Modules
                    </span>
                  )}

                  <button
                    onClick={() => setActiveTab("overview")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "space-x-2.5 px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "overview"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="Overview & Matrix"
                  >
                    <FileText className="h-4 w-4 shrink-0" />
                    {isSidebarOpen && <span className="truncate">Overview & Matrix</span>}
                  </button>

                  <button
                    onClick={() => setActiveTab("eda")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "space-x-2.5 px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "eda"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="Exploratory EDA"
                  >
                    <BarChart3 className="h-4 w-4 shrink-0" />
                    {isSidebarOpen && <span className="truncate">Exploratory EDA</span>}
                  </button>

                  <button
                    onClick={() => setActiveTab("stats")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "space-x-2.5 px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "stats"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="Hypothesis Tests"
                  >
                    <Binary className="h-4 w-4 shrink-0" />
                    {isSidebarOpen && <span className="truncate">Hypothesis Tests</span>}
                  </button>

                  <button
                    onClick={() => setActiveTab("polars")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "justify-between px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "polars"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="Polars Data Cleaning"
                  >
                    <div className="flex items-center space-x-2.5 min-w-0">
                      <Zap className="h-4 w-4 shrink-0 text-[#0284C7]" />
                      {isSidebarOpen && <span className="truncate">Polars Cleaning</span>}
                    </div>
                    {isSidebarOpen && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-sky-500/10 text-sky-600 font-mono">
                        Fast
                      </span>
                    )}
                  </button>

                  <button
                    onClick={() => setActiveTab("feature-eng")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "space-x-2.5 px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "feature-eng"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="Feature Engineering"
                  >
                    <Settings className="h-4 w-4 shrink-0" />
                    {isSidebarOpen && <span className="truncate">Feature Engineering</span>}
                  </button>

                  <button
                    onClick={() => setActiveTab("ml")}
                    className={`w-full flex items-center rounded-lg text-xs font-medium transition-colors cursor-pointer text-left ${
                      isSidebarOpen ? "space-x-2.5 px-3 py-2" : "justify-center p-2"
                    } ${
                      activeTab === "ml"
                        ? "bg-white text-[#0284C7] shadow-xs font-semibold"
                        : "text-[#73726C] hover:bg-[#E8E5DC] hover:text-[#1F1E1D]"
                    }`}
                    title="AutoML Trainer"
                  >
                    <Sparkles className="h-4 w-4 shrink-0 text-[#0284C7]" />
                    {isSidebarOpen && <span className="truncate">AutoML Trainer</span>}
                  </button>
                </div>
              </div>
            ) : (
              isSidebarOpen && (
                <div className="px-2 py-4 text-center text-xs text-[#73726C] space-y-1">
                  <Database className="h-4 w-4 mx-auto opacity-50" />
                  <p>No active dataset</p>
                  <p className="text-[10px]">Import CSV or pick sample</p>
                </div>
              )
            )}
          </div>

          {/* Bottom Account & Settings Panel */}
          <div className="pt-3 border-t border-[#E5E2D9] space-y-2">
            
            {/* Quick Actions Bar */}
            <div className={`flex items-center ${isSidebarOpen ? "justify-between px-1" : "flex-col space-y-2"}`}>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => setIsSettingsOpen(true)}
                  className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#E8E5DC] transition-colors cursor-pointer text-xs flex items-center gap-1.5"
                  title="Studio Settings"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  {isSidebarOpen && <span className="text-[11px]">Settings</span>}
                </button>
              </div>

              {summary && isSidebarOpen && (
                <button
                  onClick={handleExportPdf}
                  className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#E8E5DC] transition-colors cursor-pointer flex items-center gap-1 text-[11px]"
                  title="Export PDF Report"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>PDF</span>
                </button>
              )}
            </div>

            {/* User profile row */}
            {isSidebarOpen ? (
              <div className="p-2 rounded-xl bg-white border border-[#E5E2D9] flex items-center justify-between shadow-xs">
                <div className="flex items-center space-x-2 min-w-0">
                  <div className="w-6 h-6 rounded-full bg-[#0284C7] text-white flex items-center justify-center font-serif text-[11px] font-bold shrink-0">
                    {userSession.charAt(0).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-[#1F1E1D] truncate" title={userSession}>
                      {userSession}
                    </p>
                    {isAdmin && (
                      <span className="text-[9px] font-mono text-[#0284C7] font-semibold block">
                        Administrator
                      </span>
                    )}
                  </div>
                </div>

                <button
                  onClick={handleLogout}
                  className="p-1 text-[#73726C] hover:text-rose-500 rounded transition-colors cursor-pointer"
                  title="Sign out"
                >
                  <LogOut className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : (
              <div className="flex justify-center">
                <div 
                  className="w-7 h-7 rounded-full bg-[#0284C7] text-white flex items-center justify-center font-serif text-xs font-bold cursor-pointer"
                  title={userSession}
                  onClick={() => setIsSidebarOpen(true)}
                >
                  {userSession.charAt(0).toUpperCase()}
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* ========================================================================= */}
      {/* 2. MAIN WORKBENCH CONTENT VIEW */}
      {/* ========================================================================= */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        
        {/* Top Header Chrome */}
        <header className="h-14 border-b border-[#E5E2D9] bg-white px-4 sm:px-6 flex items-center justify-between shrink-0 z-30 shadow-2xs">
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#F0EEE6] transition-colors cursor-pointer"
              title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
            >
              <PanelLeft className="h-4 w-4" />
            </button>

            <div className="flex items-center space-x-2 text-xs">
              <span className="font-serif font-medium text-sm text-[#1F1E1D]">AutoML Studio</span>
              {summary && (
                <>
                  <span className="text-[#9E9D96]">/</span>
                  <span className="text-[#0284C7] font-semibold">{tabTitles[activeTab]}</span>
                </>
              )}
            </div>
          </div>

          {/* Top Actions */}
          <div className="flex items-center space-x-2.5">
            {/* Active Engine Badge */}
            <button
              onClick={() => {
                if (isAdmin) {
                  setIsAdminView(true);
                }
              }}
              className={`px-2.5 py-1 text-xs rounded-lg transition-colors flex items-center space-x-1.5 shadow-2xs ${
                activeModelConfig?.activeProvider === "huggingface"
                  ? "bg-amber-50 border border-amber-200 text-amber-800 hover:bg-amber-100"
                  : "bg-sky-50 border border-sky-200 text-sky-800 hover:bg-sky-100"
              } ${isAdmin ? "cursor-pointer" : "cursor-default"}`}
              title={isAdmin ? "Active AI Engine: Click to open Admin AI Model Console" : "Active AI Engine"}
            >
              {activeModelConfig?.activeProvider === "huggingface" ? (
                <span className="text-xs">🤗</span>
              ) : (
                <Sparkles className="h-3 w-3 text-sky-600" />
              )}
              <span className="hidden sm:inline font-mono font-medium">
                {activeModelConfig?.activeProvider === "huggingface"
                  ? `HF: ${activeModelConfig?.hfModel ? (activeModelConfig.hfModel.includes("/") ? activeModelConfig.hfModel.split("/")[1] : activeModelConfig.hfModel) : "Open Source"}`
                  : "Gemini 3.8 Flash"}
              </span>
              {isAdmin && (
                <span className="text-[9px] px-1 py-0.2 rounded bg-white/70 text-[#73726C] border border-[#E5E2D9] uppercase font-bold">
                  Admin
                </span>
              )}
            </button>

            {isAdmin && (
              <button
                onClick={() => setIsAdminView(true)}
                className="px-2.5 py-1 text-xs font-medium text-[#0284C7] bg-[#F0F9FF] border border-[#BAE6FD] hover:bg-[#BAE6FD]/40 rounded-lg transition-colors cursor-pointer flex items-center space-x-1.5 shadow-2xs"
                title="View overall application health, dataset metrics, token usage & spend"
              >
                <Shield className="h-3.5 w-3.5 text-[#0284C7]" />
                <span className="hidden sm:inline font-mono">System Health</span>
              </button>
            )}

            {isAnalyzing && (
              <div className="flex items-center space-x-1.5 text-xs text-[#73726C] px-2.5 py-1 rounded-lg bg-[#F0F9FF] border border-[#BAE6FD]">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-[#0284C7]" />
                <span className="text-[11px] font-medium text-[#0284C7]">Profiling Schema...</span>
              </div>
            )}

            {!isAnalyzing && isAnalysisFromCache && summary && (
              <div className="hidden sm:flex items-center space-x-1.5 text-xs px-2.5 py-1 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800" title="Dataset analysis served instantly from client cache (0 Gemini API calls)">
                <Zap className="h-3 w-3 text-emerald-600" />
                <span className="text-[11px] font-mono font-medium">Cached (0 tokens)</span>
              </div>
            )}

            {summary && (
              <>
                <button
                  onClick={() => setIsInsightsOpen(!isInsightsOpen)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center space-x-1.5 border ${
                    isInsightsOpen
                      ? "bg-[#0284C7]/10 border-[#0284C7]/40 text-[#0284C7]"
                      : "border-[#E5E2D9] text-[#73726C] hover:bg-[#F0EEE6]"
                  }`}
                  title="Toggle Insights Panel"
                >
                  <Sparkles className="h-3.5 w-3.5 text-[#0284C7]" />
                  <span className="hidden sm:inline">Insights</span>
                </button>

                <button
                  onClick={handleExportPdf}
                  className="px-3 py-1.5 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-lg cursor-pointer transition-colors shadow-xs flex items-center space-x-1.5"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Export PDF</span>
                </button>

                <button
                  onClick={handleReset}
                  className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] border border-[#E5E2D9] rounded-lg hover:bg-[#F0EEE6] transition-colors cursor-pointer"
                  title="Upload Another Dataset"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              </>
            )}
          </div>
        </header>

        {/* Content Body */}
        <div className="flex-1 flex overflow-hidden relative">
          
          <main className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-5">
            {!summary ? (
              <div className="h-full flex flex-col justify-center">
                <DataUpload onDataLoaded={handleDataLoaded} userGreetingName={userSession} />
              </div>
            ) : (
              <div className="max-w-6xl mx-auto space-y-5 pb-16">
                
                {/* DATASET WORKBENCH HEADER BANNER */}
                <div className="bg-white border border-[#E5E2D9] rounded-2xl p-4 sm:p-5 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center space-x-3.5 min-w-0">
                    <div className="w-10 h-10 rounded-xl bg-[#F0F9FF] border border-[#BAE6FD] flex items-center justify-center text-[#0284C7] shrink-0">
                      <FileSpreadsheet className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center space-x-2">
                        <h2 className="font-serif text-base sm:text-lg font-medium text-[#1F1E1D] truncate" title={summary.fileName}>
                          {summary.fileName}
                        </h2>
                        {isDataCleaned && (
                          <span className="text-[10px] font-medium bg-[#F0F9FF] text-[#0284C7] border border-[#BAE6FD] px-2 py-0.5 rounded-md inline-flex items-center gap-1 shrink-0">
                            <Zap className="h-2.5 w-2.5" />
                            <span>Polars Cleaned</span>
                          </span>
                        )}
                        {summary.batchReport?.isBatched && (
                          <span className="text-[10px] font-medium bg-purple-50 text-purple-700 border border-purple-200 px-2 py-0.5 rounded-md inline-flex items-center gap-1 shrink-0 font-mono">
                            <Layers className="h-2.5 w-2.5" />
                            <span>Batch Processed ({summary.batchReport.totalChunks} Chunks)</span>
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-3 text-xs text-[#73726C] mt-0.5">
                        <span>{summary.rowCount.toLocaleString()} records</span>
                        <span>·</span>
                        <span>{summary.colCount} feature columns</span>
                        <span>·</span>
                        <span>Quality Score: <strong className="text-[#0284C7]">{summary.dataQualityScore}%</strong></span>
                      </div>
                    </div>
                  </div>

                  {/* Quick Action Buttons */}
                  <div className="flex items-center space-x-2 shrink-0 self-end sm:self-auto">
                    {isDataCleaned && (
                      <button
                        onClick={handleRevertToOriginal}
                        className="px-3 py-1.5 text-xs text-[#0284C7] hover:bg-[#F0F9FF] border border-[#BAE6FD] rounded-lg transition-colors cursor-pointer flex items-center gap-1.5"
                      >
                        <RefreshCw className="h-3 w-3" />
                        <span>Revert Clean</span>
                      </button>
                    )}

                    <button
                      onClick={handleReset}
                      className="px-3 py-1.5 text-xs text-[#73726C] hover:text-[#1F1E1D] border border-[#E5E2D9] rounded-lg hover:bg-[#F0EEE6] transition-colors cursor-pointer flex items-center gap-1.5"
                    >
                      <span>Change File</span>
                    </button>
                  </div>
                </div>

                {/* Horizontal Module Navigation Tab Bar */}
                <div className="flex items-center gap-1.5 border-b border-[#E5E2D9] pb-2 overflow-x-auto">
                  <button
                    onClick={() => setActiveTab("overview")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "overview"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <FileText className="h-3.5 w-3.5" />
                    <span>Overview & Matrix</span>
                  </button>

                  <button
                    onClick={() => setActiveTab("eda")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "eda"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <BarChart3 className="h-3.5 w-3.5" />
                    <span>Exploratory EDA</span>
                  </button>

                  <button
                    onClick={() => setActiveTab("stats")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "stats"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <Binary className="h-3.5 w-3.5" />
                    <span>Hypothesis Tests</span>
                  </button>

                  <button
                    onClick={() => setActiveTab("polars")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "polars"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <Zap className="h-3.5 w-3.5 text-[#0284C7]" />
                    <span>Polars Cleaning</span>
                  </button>

                  <button
                    onClick={() => setActiveTab("feature-eng")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "feature-eng"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <Settings className="h-3.5 w-3.5" />
                    <span>Feature Engineering</span>
                  </button>

                  <button
                    onClick={() => setActiveTab("ml")}
                    className={`px-3.5 py-2 text-xs font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
                      activeTab === "ml"
                        ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    <Sparkles className="h-3.5 w-3.5 text-[#0284C7]" />
                    <span>AutoML Trainer</span>
                  </button>
                </div>

                {/* Direct Tab Content Rendering */}
                <div className="pt-1">
                  {activeTab === "overview" && (
                    <OverviewTab summary={summary} onReset={handleReset} onRebatch={handleRebatchDataset} />
                  )}
                  {activeTab === "eda" && <EdaTab summary={summary} rawData={rawData} />}
                  {activeTab === "stats" && <StatsTab summary={summary} />}
                  {activeTab === "polars" && (
                    <PolarsCleaningTab
                      summary={summary}
                      rawData={rawData}
                      onApplyCleanedData={handleApplyCleanedData}
                      onRevertToOriginal={handleRevertToOriginal}
                      isDataCleaned={isDataCleaned}
                    />
                  )}
                  {activeTab === "feature-eng" && (
                    <FeatureEngTab
                      summary={summary}
                      rawData={rawData}
                      engineeredFeatures={engineeredFeatures}
                      onAddFeature={handleAddFeature}
                      onRemoveFeature={handleRemoveFeature}
                    />
                  )}
                  {activeTab === "ml" && (
                    <MlTab 
                      summary={summary} 
                      rawData={rawData}
                      onModelTrained={setTrainedModelInfo} 
                    />
                  )}
                </div>
              </div>
            )}
          </main>

          {/* 3. AUTOMATED INSIGHTS DRAWER */}
          {summary && (
            <aside 
              className={`${
                isInsightsOpen ? "w-80 border-l" : "w-0 border-l-0"
              } transition-all duration-200 ease-in-out bg-[#F0EEE6] border-[#E5E2D9] flex flex-col shrink-0 overflow-hidden relative z-20`}
            >
              <div className="p-4 space-y-4 flex flex-col h-full min-h-0">
                <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
                  <div className="flex items-center space-x-2">
                    <Sparkles className="h-4 w-4 text-[#0284C7]" />
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Automated Insights
                    </h3>
                  </div>
                  <button
                    onClick={() => setIsInsightsOpen(false)}
                    className="p-1 text-[#73726C] hover:text-[#1F1E1D] rounded-md hover:bg-[#E8E5DC] transition-colors cursor-pointer"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto pr-1 space-y-4">
                  {isAnalyzing ? (
                    <div className="py-12 flex flex-col items-center justify-center space-y-2 text-center">
                      <Loader2 className="h-5 w-5 animate-spin text-[#0284C7]" />
                      <p className="text-xs text-[#73726C]">
                        Synthesizing dataset insights...
                      </p>
                    </div>
                  ) : aiInsights ? (
                    <div className="space-y-4">
                      {/* Cache Status Badge */}
                      {isAnalysisFromCache && (
                        <div className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-[11px] font-mono">
                          <span className="inline-flex items-center gap-1 font-semibold">
                            <Zap className="h-3 w-3 text-emerald-600" />
                            Client Cache (0 Tokens)
                          </span>
                          <button
                            onClick={handleForceReAnalyze}
                            className="text-[10px] text-emerald-700 underline hover:text-emerald-900 cursor-pointer flex items-center gap-1 font-sans"
                            title="Force new Gemini API analysis"
                          >
                            <RefreshCw className="h-2.5 w-2.5" />
                            <span>Refresh</span>
                          </button>
                        </div>
                      )}

                      {/* Overall conclusions */}
                      <div className="space-y-2.5">
                        <span className="text-[10px] font-medium uppercase tracking-wider text-[#9E9D96]">
                          Statistical Discoveries
                        </span>
                        {aiInsights.overallConclusions.slice(0, 3).map((c, idx) => (
                          <div key={idx} className="p-3 bg-white border border-[#E5E2D9] rounded-xl space-y-1 shadow-xs">
                            <h5 className="font-medium text-xs text-[#1F1E1D]">{c.title}</h5>
                            <p className="text-xs text-[#73726C] leading-relaxed">{c.description}</p>
                          </div>
                        ))}
                      </div>

                      {/* Model suggestion */}
                      <div className="p-3.5 bg-[#F0F9FF] border border-[#BAE6FD] rounded-xl space-y-1.5 text-xs">
                        <h5 className="font-medium text-[#0284C7]">
                          Recommended Target Predictor
                        </h5>
                        <p className="text-xs text-[#1F1E1D] leading-relaxed">
                          Predict <span className="font-bold text-[#0284C7]">'{aiInsights.modelRecommendations.recommendedTarget}'</span> using a <span className="font-bold text-[#0284C7]">{aiInsights.modelRecommendations.recommendedTask}</span> task.
                        </p>
                        <p className="text-[11px] text-[#73726C] leading-relaxed italic">
                          {aiInsights.modelRecommendations.rationale}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-[#9E9D96] space-y-1.5">
                      <HelpCircle className="h-5 w-5 mx-auto" />
                      <p className="text-xs">No analysis available</p>
                      {analysisError && (
                        <p className="text-xs text-rose-500 bg-rose-50 p-2 rounded-lg border border-rose-200">
                          {analysisError}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </aside>
          )}
        </div>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        userEmail={userSession}
        onLogout={handleLogout}
        onClearWorkspace={handleReset}
      />
    </div>
  );
}
