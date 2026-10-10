import React, { useState, useEffect } from "react";
import { 
  Activity, Shield, Server, Database, Cpu, Zap, 
  DollarSign, RefreshCw, HardDrive, Clock, CheckCircle2, 
  ArrowLeft, Download, Trash2, AlertTriangle, Layers, TrendingUp, Sparkles, Terminal, Check,
  Users, Award, Crown, Search, Mail, Calendar, ChevronRight, Flame,
  Sliders, Key, Globe, ExternalLink, Play, AlertCircle
} from "lucide-react";
import { 
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend 
} from "recharts";
import { AIModelConfig, SupportedHfModel } from "../types";

interface AdminHealthDashboardProps {
  onBackToStudio: () => void;
  adminEmail: string;
  onModelConfigChanged?: () => void;
}

export interface UserSessionInfo {
  id: string;
  email: string;
  timestamp: string;
  firstSeen: string;
  datasetsProcessed: number;
  totalRowsProcessed: number;
  totalTokensUsed: number;
  totalCostUsd: number;
  datasetNames: string[];
  powerTier: "Power User" | "Active Analyst" | "Standard";
  powerScore: number;
  lastStatus: "Uploaded" | "Cleaned" | "Analyzed" | "Trained";
  sessionCount: number;
}

interface MetricsData {
  status: string;
  uptimeSeconds: number;
  server: {
    uptimeFormatted: string;
    nodeVersion: string;
    platform: string;
    cpuCount: number;
    totalRequests: number;
    memory: {
      heapUsedMB: number;
      heapTotalMB: number;
      rssMB: number;
      externalMB: number;
    };
    systemLoadAvg: number[];
  };
  datasets: {
    totalUploaded?: number;
    totalCleaned: number;
    totalAnalyzed: number;
    totalTrained?: number;
    totalDatasets?: number;
    totalRowsProcessed: number;
    totalBytesProcessed: number;
    totalFormattedSize: string;
    totalIngestionCostUsd?: number;
    log: Array<{
      id: string;
      fileName: string;
      rowCount: number;
      colCount: number;
      fileSizeBytes: number;
      fileSizeFormatted: string;
      tokensUsed: number;
      promptTokens?: number;
      candidateTokens?: number;
      costUsd: number;
      status: "Uploaded" | "Cleaned" | "Analyzed" | "Trained";
      user: string;
      timestamp: string;
      latencyMs: number;
      cleaningSteps?: string[];
      qualityScoreBefore?: number;
      qualityScoreAfter?: number;
      memoryFootprintMB: number;
    }>;
  };
  tokenAndCost: {
    totalPromptTokens: number;
    totalCandidateTokens: number;
    totalTokens: number;
    totalCostUsd: number;
    rateTiers: {
      inputPerMillion: string;
      outputPerMillion: string;
    };
    thirtyDayTotalTokens?: number;
    thirtyDayTotalCostUsd?: number;
    thirtyDayAvgDailyCostUsd?: number;
    monthlyBudgetLimitUsd?: number;
    budgetConsumptionPct?: number;
    thirtyDayChart?: Array<{
      date: string;
      fullDate: string;
      promptTokens: number;
      candidateTokens: number;
      totalTokens: number;
      costUsd: number;
      cumulativeCostUsd: number;
      datasetCount: number;
    }>;
    hourlyChart: Array<{ time: string; tokens: number; cost: number }>;
  };
  userSessions?: UserSessionInfo[];
  userSessionsSummary?: {
    totalUsers: number;
    powerUsersCount: number;
    activeAnalystCount: number;
    totalDatasetsProcessed: number;
  };
  tokenOptimization?: {
    totalTokensSaved: number;
    totalCostSavedUsd: number;
    cacheHitCount: number;
    cacheHitRate: string;
    averageCompressionRatio: string;
    activeModel: string;
    activeProvider?: "gemini" | "huggingface";
    techniques: string[];
  };
  modelRouting?: {
    activeProvider: "gemini" | "huggingface";
    geminiModel: string;
    hfModel: string;
    activeModelName: string;
    hasHfApiKey: boolean;
    hasGeminiApiKey: boolean;
    geminiRequests: number;
    hfRequests: number;
    hfTokensUsed: number;
    supportedHfModels: SupportedHfModel[];
  };
  polarsEngine: {
    engineVersion: string;
    executionMode: string;
    workerThreads: number;
    bufferSizeMB: number;
    status: string;
  };
  auditLogs: Array<{ id: string; time: string; level: "INFO" | "WARN" | "SUCCESS"; message: string }>;
}

function formatRelativeTime(dateStr: string): string {
  try {
    const diffMs = Date.now() - new Date(dateStr).getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return "Just now";
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHour = Math.floor(diffMin / 60);
    if (diffHour < 24) return `${diffHour}h ago`;
    const diffDays = Math.floor(diffHour / 24);
    return `${diffDays}d ago`;
  } catch {
    return dateStr;
  }
}

export default function AdminHealthDashboard({ onBackToStudio, adminEmail, onModelConfigChanged }: AdminHealthDashboardProps) {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
  const [activeTab, setActiveTab] = useState<"overview" | "datasets" | "tokens" | "sessions" | "infrastructure" | "models">("overview");
  const [chartTimespan, setChartTimespan] = useState<"30days" | "hourly">("30days");
  const [thirtyDayMetricMode, setThirtyDayMetricMode] = useState<"dual" | "cost" | "tokens">("dual");
  const [userSearchQuery, setUserSearchQuery] = useState<string>("");
  const [userTierFilter, setUserTierFilter] = useState<"ALL" | "POWER" | "ACTIVE">("ALL");
  const [tokenThreshold, setTokenThreshold] = useState<number>(1200);
  const [tab3HighUsageOnly, setTab3HighUsageOnly] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Model Routing & Hugging Face Config States
  const [modelConfig, setModelConfig] = useState<AIModelConfig | null>(null);
  const [isSavingModel, setIsSavingModel] = useState<boolean>(false);
  const [isTestingModel, setIsTestingModel] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<{ success: boolean; latencyMs?: number; outputSample?: string; error?: string; provider?: string; model?: string } | null>(null);
  const [hfCustomInput, setHfCustomInput] = useState<string>("");
  const [hfApiKeyInput, setHfApiKeyInput] = useState<string>("");
  const [showApiKeyInput, setShowApiKeyInput] = useState<boolean>(false);

  const fetchModelConfig = async () => {
    try {
      const res = await fetch("/api/admin/model-config");
      if (res.ok) {
        const data = await res.json();
        setModelConfig(data);
      }
    } catch (e) {
      console.error("Failed to fetch model config:", e);
    }
  };

  const fetchMetrics = async () => {
    setIsLoading(true);
    try {
      const [metricsRes, configRes] = await Promise.all([
        fetch("/api/admin/metrics"),
        fetch("/api/admin/model-config")
      ]);
      if (metricsRes.ok) {
        const data = await metricsRes.json();
        setMetrics(data);
      }
      if (configRes.ok) {
        const cfg = await configRes.json();
        setModelConfig(cfg);
      }
    } catch (e) {
      console.error("Failed to fetch admin metrics:", e);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwitchProvider = async (provider: "gemini" | "huggingface") => {
    setIsSavingModel(true);
    try {
      const res = await fetch("/api/admin/model-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ activeProvider: provider }),
      });
      if (res.ok) {
        const data = await res.json();
        setModelConfig(prev => prev ? ({ ...prev, activeProvider: provider, activeModelName: data.activeModelName }) : data);
        setActionNotice(`Switched active AI engine to ${provider === "gemini" ? "Google Gemini 3.8 Flash" : "Hugging Face (Open Source)"}`);
        await Promise.all([fetchMetrics(), fetchModelConfig()]);
        onModelConfigChanged?.();
        setTimeout(() => setActionNotice(null), 3500);
      }
    } catch (e) {
      console.error("Failed to switch model provider:", e);
    } finally {
      setIsSavingModel(false);
    }
  };

  const handleSelectHfModel = async (modelId: string) => {
    setIsSavingModel(true);
    try {
      const res = await fetch("/api/admin/model-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hfModel: modelId }),
      });
      if (res.ok) {
        const data = await res.json();
        setModelConfig(prev => prev ? ({ ...prev, hfModel: modelId, activeModelName: data.activeModelName }) : data);
        setActionNotice(`Hugging Face active model set to '${modelId}'`);
        await Promise.all([fetchMetrics(), fetchModelConfig()]);
        onModelConfigChanged?.();
        setTimeout(() => setActionNotice(null), 3000);
      }
    } catch (e) {
      console.error("Failed to update HF model:", e);
    } finally {
      setIsSavingModel(false);
    }
  };

  const handleSaveHfApiKey = async () => {
    setIsSavingModel(true);
    try {
      const res = await fetch("/api/admin/model-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hfApiKey: hfApiKeyInput.trim() }),
      });
      if (res.ok) {
        setModelConfig(prev => prev ? ({ ...prev, hasHfApiKey: Boolean(hfApiKeyInput.trim()) }) : null);
        setActionNotice("Hugging Face API token saved.");
        setShowApiKeyInput(false);
        await Promise.all([fetchMetrics(), fetchModelConfig()]);
        onModelConfigChanged?.();
        setTimeout(() => setActionNotice(null), 3000);
      }
    } catch (e) {
      console.error("Failed to save HF token:", e);
    } finally {
      setIsSavingModel(false);
    }
  };

  const handleTestModelConnection = async (targetProvider?: "gemini" | "huggingface", targetModel?: string) => {
    setIsTestingModel(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/admin/test-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: targetProvider || modelConfig?.activeProvider,
          model: targetModel || (targetProvider === "gemini" ? modelConfig?.geminiModel : modelConfig?.hfModel),
          testToken: hfApiKeyInput.trim() || undefined
        })
      });
      const data = await res.json();
      setTestResult(data);
    } catch (err: any) {
      setTestResult({ success: false, error: err.message });
    } finally {
      setIsTestingModel(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // 30s auto-refresh
    return () => clearInterval(interval);
  }, []);

  const handleClearCache = async () => {
    try {
      const res = await fetch("/api/admin/clear-cache", { method: "POST" });
      if (res.ok) {
        setActionNotice("Diagnostic telemetry cache purged successfully.");
        fetchMetrics();
        setTimeout(() => setActionNotice(null), 3000);
      }
    } catch {
      setActionNotice("Error resetting caches.");
    }
  };

  const handleExportDiagnostics = () => {
    if (!metrics) return;
    const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `system-health-diagnostics-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const highUsageDatasets = (metrics?.datasets.log || []).filter(d => d.tokensUsed >= tokenThreshold);
  const totalHighUsageTokens = highUsageDatasets.reduce((acc, d) => acc + d.tokensUsed, 0);
  const totalHighUsageCost = highUsageDatasets.reduce((acc, d) => acc + d.costUsd, 0);

  const filteredDatasets = (metrics?.datasets.log || []).filter(d => {
    if (filterStatus === "HIGH_USAGE") {
      return d.tokensUsed >= tokenThreshold;
    }
    if (filterStatus === "ALL") return true;
    return d.status.toUpperCase() === filterStatus;
  });

  const filteredUserSessions = (metrics?.userSessions || []).filter((u) => {
    if (userTierFilter === "POWER" && u.powerTier !== "Power User") return false;
    if (userTierFilter === "ACTIVE" && u.powerTier === "Standard") return false;
    if (userSearchQuery.trim()) {
      return u.email.toLowerCase().includes(userSearchQuery.toLowerCase());
    }
    return true;
  });

  return (
    <div className="min-h-screen bg-[#FAF9F5] text-[#1F1E1D] font-sans flex flex-col">
      
      {/* Top Admin Chrome Header */}
      <header className="border-b border-[#E5E2D9] bg-white sticky top-0 z-40 px-4 sm:px-8 py-3.5 shadow-2xs">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          
          <div className="flex items-center space-x-3.5">
            <button
              onClick={onBackToStudio}
              className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6] rounded-lg transition-colors cursor-pointer flex items-center space-x-1.5 text-xs"
              title="Return to AutoML Analytics Workbench"
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="font-medium">AutoML Studio</span>
            </button>

            <span className="text-[#D5D1C5]">|</span>

            <div className="flex items-center space-x-2">
              <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                <Shield className="h-4 w-4" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h1 className="font-serif text-base font-semibold text-[#1F1E1D]">
                    System Health & Operations Console
                  </h1>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/10 text-emerald-700 border border-emerald-500/20 font-medium inline-flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    Live Healthy
                  </span>
                </div>
                <p className="text-[11px] text-[#73726C]">
                  Real-time telemetry, datasets wrangled, token usage & server diagnostics
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 self-end sm:self-auto">
            {/* Quick Engine Switcher Bar */}
            <div className="flex items-center space-x-1 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-1 shadow-2xs">
              <button
                onClick={() => handleSwitchProvider("gemini")}
                disabled={isSavingModel}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all cursor-pointer flex items-center gap-1.5 ${
                  modelConfig?.activeProvider === "gemini"
                    ? "bg-[#0284C7] text-white shadow-2xs font-semibold"
                    : "text-[#73726C] hover:text-[#1F1E1D]"
                }`}
                title="Switch active engine to Google Gemini 3.8 Flash"
              >
                <Sparkles className="h-3 w-3" />
                <span>Gemini 3.8 Flash</span>
              </button>
              <button
                onClick={() => handleSwitchProvider("huggingface")}
                disabled={isSavingModel}
                className={`px-2.5 py-1 text-xs rounded-lg font-medium transition-all cursor-pointer flex items-center gap-1.5 ${
                  modelConfig?.activeProvider === "huggingface"
                    ? "bg-amber-600 text-white shadow-2xs font-semibold"
                    : "text-[#73726C] hover:text-[#1F1E1D]"
                }`}
                title="Switch active engine to Hugging Face Open Source Models"
              >
                <span>🤗</span>
                <span>Hugging Face ({modelConfig?.hfModel ? (modelConfig.hfModel.includes("/") ? modelConfig.hfModel.split("/")[1] : modelConfig.hfModel) : "Open Source"})</span>
              </button>
            </div>

            <button
              onClick={fetchMetrics}
              disabled={isLoading}
              className="px-3 py-1.5 text-xs text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6] border border-[#E5E2D9] rounded-lg transition-colors cursor-pointer flex items-center space-x-1.5 shadow-2xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin text-[#0284C7]" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>

            <button
              onClick={handleExportDiagnostics}
              className="px-3 py-1.5 text-xs text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6] border border-[#E5E2D9] rounded-lg transition-colors cursor-pointer flex items-center space-x-1.5 shadow-2xs"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Diagnostic JSON</span>
            </button>

            <div className="pl-2 border-l border-[#E5E2D9] flex items-center space-x-2">
              <div className="w-6 h-6 rounded-full bg-[#0284C7] text-white flex items-center justify-center font-serif text-[11px] font-bold">
                A
              </div>
              <span className="text-xs font-mono font-medium text-[#73726C] hidden md:inline">
                {adminEmail}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Action Notification Banner */}
      {actionNotice && (
        <div className="bg-[#F0F9FF] border-b border-[#BAE6FD] text-[#0284C7] px-4 py-2 text-xs font-medium text-center">
          {actionNotice}
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-8 py-6 space-y-6">
        
        {/* Navigation Tabs */}
        <div className="flex items-center gap-1.5 border-b border-[#E5E2D9] pb-2 overflow-x-auto text-xs">
          <button
            onClick={() => setActiveTab("overview")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "overview"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            <span>Health & Telemetry Overview</span>
          </button>

          <button
            onClick={() => setActiveTab("models")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "models"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Cpu className="h-3.5 w-3.5 text-[#0284C7]" />
            <span>AI Model & Hugging Face Engine</span>
            <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded font-semibold ${
              modelConfig?.activeProvider === "huggingface"
                ? "bg-amber-100 text-amber-800 border border-amber-200"
                : "bg-sky-100 text-sky-800 border border-sky-200"
            }`}>
              {modelConfig?.activeProvider === "huggingface" ? "Hugging Face" : "Gemini"}
            </span>
          </button>

          <button
            onClick={() => setActiveTab("datasets")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "datasets"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Database className="h-3.5 w-3.5" />
            <span>Cleaned Datasets ({metrics?.datasets.totalCleaned ?? 0})</span>
          </button>

          <button
            onClick={() => setActiveTab("tokens")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "tokens"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <DollarSign className="h-3.5 w-3.5" />
            <span>30-Day Budget & Tokens</span>
          </button>

          <button
            onClick={() => setActiveTab("sessions")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "sessions"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Users className="h-3.5 w-3.5" />
            <span>User Sessions & Power Users ({metrics?.userSessions?.length || 0})</span>
          </button>

          <button
            onClick={() => setActiveTab("infrastructure")}
            className={`px-3.5 py-2 font-medium rounded-xl transition-all cursor-pointer whitespace-nowrap flex items-center gap-2 ${
              activeTab === "infrastructure"
                ? "bg-white text-[#0284C7] shadow-xs border border-[#E5E2D9] font-semibold"
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Server className="h-3.5 w-3.5" />
            <span>Engine & Infrastructure Logs</span>
          </button>
        </div>

        {/* ========================================================================= */}
        {/* TAB 1: OVERVIEW & KEY PERFORMANCE CARDS */}
        {/* ========================================================================= */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            
            {/* AI Model & Inference Engine Status Banner */}
            <div className="bg-gradient-to-r from-white via-white to-[#F0F9FF] border border-[#E5E2D9] rounded-2xl p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-start space-x-3.5">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 border ${
                  modelConfig?.activeProvider === "huggingface"
                    ? "bg-amber-50 text-amber-600 border-amber-200"
                    : "bg-sky-50 text-[#0284C7] border-sky-200"
                }`}>
                  {modelConfig?.activeProvider === "huggingface" ? (
                    <span className="text-xl">🤗</span>
                  ) : (
                    <Sparkles className="h-5 w-5 text-[#0284C7]" />
                  )}
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-serif text-sm font-semibold text-[#1F1E1D]">
                      Active AI Engine: {modelConfig?.activeProvider === "huggingface" ? "Hugging Face Open Source" : "Google Gemini Enterprise"}
                    </h3>
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold ${
                      modelConfig?.activeProvider === "huggingface"
                        ? "bg-amber-100 text-amber-800 border border-amber-200"
                        : "bg-sky-100 text-sky-800 border border-sky-200"
                    }`}>
                      {modelConfig?.activeModelName || "Gemini 3.8 Flash"}
                    </span>
                  </div>
                  <p className="text-xs text-[#73726C] mt-0.5 max-w-xl">
                    {modelConfig?.activeProvider === "huggingface"
                      ? `All statistical insights, Python pipelines, and XAI model logic are being routed through Hugging Face's open source model (${modelConfig?.hfModel}).`
                      : "High-throughput EDA profiling, automated Python pipelines, and decision logic powered by Gemini 3.8 Flash with semantic in-memory caching."}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-2 shrink-0">
                <button
                  onClick={() => handleSwitchProvider(modelConfig?.activeProvider === "gemini" ? "huggingface" : "gemini")}
                  disabled={isSavingModel}
                  className={`px-3.5 py-1.5 rounded-xl text-xs font-medium cursor-pointer transition-all shadow-xs flex items-center gap-1.5 ${
                    modelConfig?.activeProvider === "gemini"
                      ? "bg-amber-600 hover:bg-amber-700 text-white"
                      : "bg-[#0284C7] hover:bg-[#0369A1] text-white"
                  }`}
                >
                  <RefreshCw className={`h-3 w-3 ${isSavingModel ? "animate-spin" : ""}`} />
                  <span>Switch to {modelConfig?.activeProvider === "gemini" ? "Hugging Face" : "Google Gemini"}</span>
                </button>
                <button
                  onClick={() => setActiveTab("models")}
                  className="px-3 py-1.5 bg-white hover:bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D] text-xs font-medium rounded-xl transition-colors cursor-pointer flex items-center gap-1"
                >
                  <Sliders className="h-3 w-3" />
                  <span>Configure Models</span>
                </button>
              </div>
            </div>

            {/* Primary KPI Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              
              {/* Card 1: Datasets Recorded */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-2 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#73726C] font-medium">Datasets Recorded</span>
                  <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                    <Database className="h-3.5 w-3.5" />
                  </div>
                </div>
                <div className="flex items-baseline space-x-2">
                  <p className="font-serif text-3xl font-medium text-[#1F1E1D]">
                    {metrics ? (metrics.datasets.totalDatasets ?? metrics.datasets.log.length) : "..."}
                  </p>
                  <span className="text-xs text-[#73726C]">datasets</span>
                </div>
                <div className="pt-1 text-[11px] text-[#73726C] flex items-center gap-1 font-mono">
                  <span>{metrics?.datasets.totalUploaded ?? 0} uploaded · {metrics?.datasets.totalCleaned ?? 0} cleaned</span>
                </div>
              </div>

              {/* Card 2: Cumulative Dataset Size */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-2 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#73726C] font-medium">Total Data Volume</span>
                  <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                    <HardDrive className="h-3.5 w-3.5" />
                  </div>
                </div>
                <div className="flex items-baseline space-x-2">
                  <p className="font-serif text-3xl font-medium text-[#1F1E1D]">
                    {metrics ? metrics.datasets.totalFormattedSize : "..."}
                  </p>
                </div>
                <div className="pt-1 text-[11px] text-[#73726C] flex items-center gap-1 font-mono">
                  <span>{metrics?.datasets.totalAnalyzed || 0} ingested workloads</span>
                </div>
              </div>

              {/* Card 3: Tokens Used */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-2 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#73726C] font-medium">Tokens Consumed</span>
                  <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                    <Sparkles className="h-3.5 w-3.5" />
                  </div>
                </div>
                <div className="flex items-baseline space-x-2">
                  <p className="font-serif text-3xl font-medium text-[#1F1E1D]">
                    {metrics ? metrics.tokenAndCost.totalTokens.toLocaleString() : "..."}
                  </p>
                  <span className="text-xs text-[#73726C]">tokens</span>
                </div>
                <div className="pt-1 text-[11px] text-[#73726C] flex items-center gap-1 font-mono">
                  <span>Prompt: {metrics?.tokenAndCost.totalPromptTokens || 0} · Cand: {metrics?.tokenAndCost.totalCandidateTokens || 0}</span>
                </div>
              </div>

              {/* Card 4: Money Used / Cost */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-2 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-[#73726C] font-medium">Accumulated Spend</span>
                  <div className="w-7 h-7 rounded-lg bg-[#0284C7]/10 flex items-center justify-center text-[#0284C7]">
                    <DollarSign className="h-3.5 w-3.5" />
                  </div>
                </div>
                <div className="flex items-baseline space-x-2">
                  <p className="font-serif text-3xl font-medium text-[#0284C7]">
                    ${metrics ? metrics.tokenAndCost.totalCostUsd.toFixed(4) : "0.0000"}
                  </p>
                  <span className="text-xs text-[#73726C]">USD</span>
                </div>
                <div className="pt-1 text-[11px] text-[#73726C] flex items-center gap-1 font-mono">
                  <span>Tier: Gemini 3.5 Flash ($0.075 / 1M)</span>
                </div>
              </div>
            </div>

            {/* Middle Section: Token Spend Chart + Server Vital Gauges */}
            <div className="grid lg:grid-cols-3 gap-6">
              
              {/* Consumption Velocity & 30-Day Trend Chart */}
              <div className="lg:col-span-2 bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-4 shadow-xs">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E5E2D9] pb-3">
                  <div>
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Token Consumption & Cost Trajectory
                    </h3>
                    <p className="text-xs text-[#73726C]">
                      {chartTimespan === "30days" 
                        ? "30-day daily token consumption and administrative budget spend" 
                        : "Real-time API invocation workload distribution across recent session windows"}
                    </p>
                  </div>
                  
                  {/* Timespan Selector */}
                  <div className="flex items-center gap-1 bg-[#FAF9F5] p-1 rounded-xl border border-[#E5E2D9] text-xs font-mono">
                    <button
                      onClick={() => setChartTimespan("30days")}
                      className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer text-[11px] ${
                        chartTimespan === "30days"
                          ? "bg-white text-[#0284C7] font-semibold shadow-2xs border border-[#BAE6FD]"
                          : "text-[#73726C] hover:text-[#1F1E1D]"
                      }`}
                    >
                      30-Day Daily Trend
                    </button>
                    <button
                      onClick={() => setChartTimespan("hourly")}
                      className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer text-[11px] ${
                        chartTimespan === "hourly"
                          ? "bg-white text-[#0284C7] font-semibold shadow-2xs border border-[#BAE6FD]"
                          : "text-[#73726C] hover:text-[#1F1E1D]"
                      }`}
                    >
                      Hourly Granularity
                    </button>
                  </div>
                </div>

                <div className="h-68">
                  {metrics && chartTimespan === "30days" && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart 
                        data={metrics.tokenAndCost.thirtyDayChart || []} 
                        margin={{ top: 10, right: 25, left: -15, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E2D9" vertical={false} />
                        <XAxis 
                          dataKey="date" 
                          stroke="#73726C" 
                          fontSize={10} 
                          tickLine={false} 
                          interval="preserveStartEnd" 
                        />
                        <YAxis 
                          yAxisId="tokens"
                          stroke="#0284C7" 
                          fontSize={10} 
                          tickLine={false}
                          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                        />
                        <YAxis 
                          yAxisId="cost"
                          orientation="right"
                          stroke="#059669" 
                          fontSize={10} 
                          tickLine={false}
                          tickFormatter={(v) => `$${Number(v).toFixed(4)}`}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: "#FFFFFF", 
                            borderColor: "#E5E2D9", 
                            borderRadius: "10px", 
                            fontSize: "11px",
                            boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
                          }}
                          formatter={(value: any, name: any) => {
                            if (name === "Daily Cost (USD)") return [`$${Number(value).toFixed(5)}`, name];
                            if (name === "Cumulative Spend") return [`$${Number(value).toFixed(5)}`, name];
                            return [`${Number(value).toLocaleString()} tokens`, name];
                          }}
                          labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Legend 
                          verticalAlign="top" 
                          height={30}
                          wrapperStyle={{ fontSize: "11px", paddingTop: "0px" }}
                        />
                        <Line 
                          yAxisId="tokens"
                          type="monotone" 
                          dataKey="totalTokens" 
                          stroke="#0284C7" 
                          strokeWidth={2.5} 
                          dot={{ r: 2, fill: "#0284C7" }}
                          activeDot={{ r: 5 }}
                          name="Daily Tokens" 
                        />
                        <Line 
                          yAxisId="cost"
                          type="monotone" 
                          dataKey="costUsd" 
                          stroke="#059669" 
                          strokeWidth={2} 
                          dot={{ r: 2, fill: "#059669" }}
                          activeDot={{ r: 5 }}
                          name="Daily Cost (USD)" 
                        />
                        <Line 
                          yAxisId="cost"
                          type="monotone" 
                          dataKey="cumulativeCostUsd" 
                          stroke="#D97706" 
                          strokeWidth={1.5} 
                          strokeDasharray="4 4"
                          dot={false}
                          name="Cumulative Spend" 
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}

                  {metrics && chartTimespan === "hourly" && (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={metrics.tokenAndCost.hourlyChart} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#0284C7" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="#0284C7" stopOpacity={0.0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E5E2D9" vertical={false} />
                        <XAxis dataKey="time" stroke="#73726C" fontSize={11} tickLine={false} />
                        <YAxis stroke="#73726C" fontSize={11} tickLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: "#FFFFFF", borderColor: "#E5E2D9", borderRadius: "8px", fontSize: "11px" }}
                          formatter={(value: any, name: any) => [
                            name === "cost" ? `$${Number(value).toFixed(5)}` : `${Number(value).toLocaleString()} tokens`,
                            name === "cost" ? "Cost (USD)" : "Tokens"
                          ]}
                        />
                        <Area type="monotone" dataKey="tokens" stroke="#0284C7" strokeWidth={2} fillOpacity={1} fill="url(#tokenGradient)" name="tokens" />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* Server Vitals & Memory Heap */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-4 shadow-xs flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                    <div className="flex items-center space-x-2">
                      <Server className="h-4 w-4 text-[#0284C7]" />
                      <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                        Host Server Diagnostics
                      </h3>
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
                      Uptime: {metrics?.server.uptimeFormatted || "..."}
                    </span>
                  </div>

                  <div className="pt-3 space-y-3 text-xs">
                    <div>
                      <div className="flex items-center justify-between text-[#73726C] mb-1">
                        <span>V8 Heap Used</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">
                          {metrics?.server.memory.heapUsedMB || 0} MB / {metrics?.server.memory.heapTotalMB || 0} MB
                        </span>
                      </div>
                      <div className="w-full bg-[#FAF9F5] border border-[#E5E2D9] h-2 rounded-full overflow-hidden">
                        <div 
                          className="bg-[#0284C7] h-full rounded-full transition-all"
                          style={{ 
                            width: `${metrics ? Math.min(100, (metrics.server.memory.heapUsedMB / metrics.server.memory.heapTotalMB) * 100) : 20}%` 
                          }}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#E5E2D9] text-[11px]">
                      <div className="p-2 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9]">
                        <span className="text-[#73726C] block">Node Runtime</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{metrics?.server.nodeVersion || "..."}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9]">
                        <span className="text-[#73726C] block">CPU Cores</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{metrics?.server.cpuCount || 2} Cores (vCPU)</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9]">
                        <span className="text-[#73726C] block">Total Invocations</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{metrics?.server.totalRequests.toLocaleString() || "0"}</span>
                      </div>
                      <div className="p-2 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9]">
                        <span className="text-[#73726C] block">RSS Footprint</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{metrics?.server.memory.rssMB || 0} MB</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#E5E2D9] flex items-center justify-between">
                  <span className="text-[11px] text-[#73726C]">Polars Engine: WASM 64MB</span>
                  <button
                    onClick={handleClearCache}
                    className="text-[11px] text-rose-600 hover:underline cursor-pointer flex items-center gap-1"
                  >
                    <Trash2 className="h-3 w-3" />
                    <span>Purge Caches</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Recent Dataset Pipeline Activity Table with High-Usage Budget Guardrails */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-4 shadow-xs">
              
              {/* Header & Filter Controls */}
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
                <div>
                  <div className="flex items-center space-x-2">
                    <Database className="h-4 w-4 text-[#0284C7]" />
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Recently Processed Datasets & Telemetry Logs
                    </h3>
                  </div>
                  <p className="text-xs text-[#73726C] mt-0.5">
                    Detailed telemetry showing row counts, file sizes, tokens consumed, and quality differentials
                  </p>
                </div>

                {/* Filter Status Buttons including High Usage Flag */}
                <div className="flex items-center gap-1.5 flex-wrap text-xs">
                  {["ALL", "UPLOADED", "ANALYZED", "CLEANED", "TRAINED"].map((st) => (
                    <button
                      key={st}
                      onClick={() => setFilterStatus(st)}
                      className={`px-2.5 py-1 rounded-lg border font-mono text-[11px] cursor-pointer transition-colors ${
                        filterStatus === st 
                          ? "bg-[#0284C7] text-white border-[#0284C7] font-medium" 
                          : "border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6]"
                      }`}
                    >
                      {st}
                    </button>
                  ))}

                  {/* High Usage Filter Button */}
                  <button
                    onClick={() => setFilterStatus(filterStatus === "HIGH_USAGE" ? "ALL" : "HIGH_USAGE")}
                    className={`px-2.5 py-1 rounded-lg border font-mono text-[11px] cursor-pointer transition-colors inline-flex items-center gap-1.5 ${
                      filterStatus === "HIGH_USAGE"
                        ? "bg-amber-600 text-white border-amber-600 font-semibold shadow-2xs"
                        : "bg-amber-50 text-amber-900 border-amber-300 hover:bg-amber-100"
                    }`}
                    title={`Filter datasets consuming ≥ ${tokenThreshold.toLocaleString()} tokens`}
                  >
                    <AlertTriangle className={`h-3 w-3 ${filterStatus === "HIGH_USAGE" ? "text-white" : "text-amber-600"}`} />
                    <span>High Usage ({highUsageDatasets.length})</span>
                  </button>
                </div>
              </div>

              {/* Proactive Budget Management & Token Consumption Threshold Bar */}
              <div className="p-3.5 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9] flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
                <div className="flex items-center space-x-2.5">
                  <div className="w-7 h-7 rounded-lg bg-amber-500/15 border border-amber-400/30 flex items-center justify-center text-amber-700 shrink-0">
                    <AlertTriangle className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-[#1F1E1D]">Budget Guardrail: High-Usage Threshold</span>
                      <span className="px-2 py-0.2 rounded-full font-mono text-[10px] font-bold bg-amber-100 text-amber-900 border border-amber-300">
                        {highUsageDatasets.length} Flagged
                      </span>
                    </div>
                    <p className="text-[11px] text-[#73726C]">
                      Datasets exceeding <span className="font-mono font-semibold text-[#1F1E1D]">{tokenThreshold.toLocaleString()} tokens</span> are flagged for proactive budget monitoring.
                    </p>
                  </div>
                </div>

                {/* Predefined Threshold Selectors */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[11px] text-[#73726C] font-mono mr-1">Threshold:</span>
                  {[800, 1000, 1200, 1500, 2000].map((th) => (
                    <button
                      key={th}
                      onClick={() => setTokenThreshold(th)}
                      className={`px-2 py-0.5 rounded border text-[11px] font-mono cursor-pointer transition-colors ${
                        tokenThreshold === th
                          ? "bg-amber-600 text-white border-amber-600 font-bold shadow-2xs"
                          : "bg-white text-[#73726C] border-[#E5E2D9] hover:text-[#1F1E1D]"
                      }`}
                    >
                      {th.toLocaleString()}
                    </button>
                  ))}
                  <div className="flex items-center space-x-1 pl-1 border-l border-[#E5E2D9]">
                    <input
                      type="number"
                      min={100}
                      step={100}
                      value={tokenThreshold}
                      onChange={(e) => setTokenThreshold(Math.max(100, Number(e.target.value) || 1200))}
                      className="w-16 bg-white border border-[#E5E2D9] rounded px-1.5 py-0.5 text-[11px] font-mono text-center outline-none focus:border-amber-500"
                      title="Custom token threshold"
                    />
                    <span className="text-[10px] text-[#73726C] font-mono">tokens</span>
                  </div>
                </div>
              </div>

              {/* High-Usage Proactive Alert Banner (When datasets exceed threshold) */}
              {highUsageDatasets.length > 0 && (
                <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-300 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                  <div className="flex items-center space-x-3">
                    <Flame className="h-5 w-5 text-amber-600 shrink-0 animate-pulse" />
                    <div>
                      <p className="font-medium text-amber-950">
                        High Token Consumption Alert: {highUsageDatasets.length} of {metrics?.datasets.log.length || 0} datasets exceed the {tokenThreshold.toLocaleString()} tokens guardrail.
                      </p>
                      <p className="text-[11px] text-amber-800 font-mono mt-0.5">
                        Flagged datasets consume <span className="font-bold">{totalHighUsageTokens.toLocaleString()} tokens</span> (${totalHighUsageCost.toFixed(5)} USD), representing {Math.round((totalHighUsageTokens / Math.max(1, metrics?.tokenAndCost.totalTokens || 1)) * 100)}% of total AI model spend.
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={() => setFilterStatus(filterStatus === "HIGH_USAGE" ? "ALL" : "HIGH_USAGE")}
                    className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-mono text-[11px] font-medium transition-colors cursor-pointer shrink-0 shadow-2xs flex items-center space-x-1.5"
                  >
                    <span>{filterStatus === "HIGH_USAGE" ? "Show All Datasets" : "View Flagged Only"}</span>
                    <ChevronRight className="h-3 w-3" />
                  </button>
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#E5E2D9] text-[#73726C] text-[11px] font-medium">
                      <th className="py-2.5 px-3">Dataset Name & Alerts</th>
                      <th className="py-2.5 px-3">Size & Records</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3">Quality Score</th>
                      <th className="py-2.5 px-3">Tokens Consumed</th>
                      <th className="py-2.5 px-3">Est. Cost</th>
                      <th className="py-2.5 px-3">User / Session</th>
                      <th className="py-2.5 px-3 text-right">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    {filteredDatasets.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-8 text-center text-xs text-[#73726C]">
                          No datasets found matching this filter criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredDatasets.map((ds) => {
                        const isHighUsage = ds.tokensUsed >= tokenThreshold;
                        const excessTokens = ds.tokensUsed - tokenThreshold;

                        return (
                          <tr 
                            key={ds.id} 
                            className={`transition-colors ${
                              isHighUsage 
                                ? "bg-amber-500/8 hover:bg-amber-500/15 border-l-4 border-l-amber-500" 
                                : "hover:bg-[#FAF9F5]"
                            }`}
                          >
                            {/* Dataset Name & High Usage Flag */}
                            <td className="py-2.5 px-3 font-mono font-medium text-xs">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className={isHighUsage ? "text-amber-950 font-bold" : "text-[#1F1E1D]"}>
                                  {ds.fileName}
                                </span>
                                {isHighUsage && (
                                  <span 
                                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-100 text-amber-900 border border-amber-300 shadow-2xs"
                                    title={`High token consumption: ${ds.tokensUsed.toLocaleString()} tokens exceeds threshold of ${tokenThreshold.toLocaleString()} by ${excessTokens.toLocaleString()} tokens`}
                                  >
                                    <AlertTriangle className="h-2.5 w-2.5 text-amber-600" />
                                    <span>High Usage (+{excessTokens.toLocaleString()})</span>
                                  </span>
                                )}
                              </div>
                            </td>

                            <td className="py-2.5 px-3 font-mono text-[11px] text-[#73726C]">
                              <span className="font-semibold text-[#1F1E1D]">{ds.fileSizeFormatted}</span>
                              <span className="block text-[10px] text-[#9E9D96]">
                                {ds.rowCount.toLocaleString()} rows · {ds.colCount} cols
                              </span>
                            </td>

                            <td className="py-2.5 px-3">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                                ds.status === "Uploaded"
                                  ? "bg-amber-50 text-amber-800 border border-amber-200"
                                  : ds.status === "Cleaned" 
                                    ? "bg-sky-50 text-[#0284C7] border border-[#BAE6FD]"
                                    : ds.status === "Trained"
                                      ? "bg-purple-50 text-purple-700 border border-purple-200"
                                      : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              }`}>
                                {ds.status}
                              </span>
                            </td>

                            <td className="py-2.5 px-3 font-mono text-xs">
                              {ds.qualityScoreAfter ? (
                                <span className="text-emerald-700 font-medium">
                                  {ds.qualityScoreBefore}% → {ds.qualityScoreAfter}%
                                </span>
                              ) : (
                                <span className="text-[#73726C]">{ds.qualityScoreBefore ?? 100}%</span>
                              )}
                            </td>

                            {/* Tokens Consumed with High Usage Highlighting */}
                            <td className="py-2.5 px-3 font-mono text-xs">
                              {isHighUsage ? (
                                <div className="space-y-0.5">
                                  <span className="font-bold text-amber-900 inline-flex items-center gap-1">
                                    <AlertTriangle className="h-3 w-3 text-amber-600" />
                                    {ds.tokensUsed.toLocaleString()}
                                  </span>
                                  <span className="block text-[10px] text-amber-700 font-medium">
                                    +{Math.round((excessTokens / tokenThreshold) * 100)}% over guardrail
                                  </span>
                                </div>
                              ) : (
                                <span className="text-[#73726C]">
                                  {ds.tokensUsed.toLocaleString()}
                                </span>
                              )}
                            </td>

                            {/* Est. Cost */}
                            <td className="py-2.5 px-3 font-mono text-xs font-semibold">
                              <span className={isHighUsage ? "text-amber-900 font-bold" : "text-[#0284C7]"}>
                                ${ds.costUsd.toFixed(5)}
                              </span>
                            </td>

                            <td className="py-2.5 px-3 font-mono text-[11px] text-[#73726C] truncate max-w-[120px]" title={ds.user}>
                              {ds.user}
                            </td>

                            <td className="py-2.5 px-3 text-right font-mono text-[11px] text-[#9E9D96]">
                              {new Date(ds.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Recent User Sessions & Power Users Detailed Table */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-4 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
                <div>
                  <div className="flex items-center space-x-2">
                    <Users className="h-4 w-4 text-[#0284C7]" />
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Recent User Sessions & Power User Analytics
                    </h3>
                  </div>
                  <p className="text-xs text-[#73726C] mt-0.5">
                    User session activity, timestamps, and total datasets processed to identify high-volume power analysts
                  </p>
                </div>

                {/* Filter and Search Bar */}
                <div className="flex items-center gap-2 flex-wrap text-xs">
                  <div className="relative">
                    <Search className="h-3 w-3 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#73726C]" />
                    <input
                      type="text"
                      placeholder="Search user email..."
                      value={userSearchQuery}
                      onChange={(e) => setUserSearchQuery(e.target.value)}
                      className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg pl-7 pr-2.5 py-1 text-xs text-[#1F1E1D] placeholder-[#9E9D96] focus:border-[#0284C7] outline-none w-44"
                    />
                  </div>

                  <div className="flex items-center space-x-1">
                    {[
                      { key: "ALL", label: "All Users" },
                      { key: "POWER", label: "Power Users" },
                      { key: "ACTIVE", label: "Active" }
                    ].map((st) => (
                      <button
                        key={st.key}
                        onClick={() => setUserTierFilter(st.key as any)}
                        className={`px-2.5 py-1 rounded-lg border font-mono text-[11px] cursor-pointer transition-colors ${
                          userTierFilter === st.key 
                            ? "bg-[#0284C7] text-white border-[#0284C7] font-medium" 
                            : "border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6]"
                        }`}
                      >
                        {st.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#E5E2D9] text-[#73726C] text-[11px] font-medium">
                      <th className="py-2.5 px-3">User Email</th>
                      <th className="py-2.5 px-3">Power User Tier</th>
                      <th className="py-2.5 px-3">Datasets Processed</th>
                      <th className="py-2.5 px-3">Last Active Timestamp</th>
                      <th className="py-2.5 px-3">Tokens Consumed</th>
                      <th className="py-2.5 px-3">Est. Billed Spend</th>
                      <th className="py-2.5 px-3">Recent Workloads</th>
                      <th className="py-2.5 px-3 text-right">Latest Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    {filteredUserSessions.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-6 text-center text-xs text-[#73726C]">
                          No user sessions match your search criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredUserSessions.map((user) => {
                        const isPower = user.powerTier === "Power User";
                        const isCurrent = user.email.toLowerCase() === adminEmail.toLowerCase();
                        const initial = (user.email[0] || "U").toUpperCase();

                        return (
                          <tr key={user.id} className="hover:bg-[#FAF9F5] transition-colors">
                            {/* Email */}
                            <td className="py-2.5 px-3">
                              <div className="flex items-center space-x-2">
                                <div className={`w-7 h-7 rounded-full flex items-center justify-center font-serif text-xs font-bold shrink-0 ${
                                  isPower ? "bg-amber-100 text-amber-900 border border-amber-300" : "bg-[#FAF9F5] text-[#1F1E1D] border border-[#E5E2D9]"
                                }`}>
                                  {initial}
                                </div>
                                <div>
                                  <div className="flex items-center space-x-1.5">
                                    <span className="font-mono font-medium text-xs text-[#1F1E1D]">
                                      {user.email}
                                    </span>
                                    {isCurrent && (
                                      <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-sky-50 text-[#0284C7] border border-[#BAE6FD]">
                                        Current
                                      </span>
                                    )}
                                  </div>
                                  <span className="text-[10px] text-[#9E9D96] font-mono">
                                    First seen: {new Date(user.firstSeen).toLocaleDateString([], { month: "short", day: "numeric" })}
                                  </span>
                                </div>
                              </div>
                            </td>

                            {/* Power User Tier */}
                            <td className="py-2.5 px-3">
                              {user.powerTier === "Power User" ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-amber-50 text-amber-900 border border-amber-300">
                                  <Crown className="h-3 w-3 text-amber-600 fill-amber-400" />
                                  <span>Power User</span>
                                </span>
                              ) : user.powerTier === "Active Analyst" ? (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-sky-50 text-[#0284C7] border border-[#BAE6FD]">
                                  <Award className="h-3 w-3 text-[#0284C7]" />
                                  <span>Active Analyst</span>
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono text-[#73726C] bg-[#FAF9F5] border border-[#E5E2D9]">
                                  Standard
                                </span>
                              )}
                            </td>

                            {/* Summary count of datasets processed */}
                            <td className="py-2.5 px-3 font-mono">
                              <span className="font-bold text-xs px-2.5 py-1 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D]">
                                {user.datasetsProcessed} datasets
                              </span>
                              <span className="block text-[10px] text-[#9E9D96] mt-0.5">
                                {user.totalRowsProcessed.toLocaleString()} records
                              </span>
                            </td>

                            {/* Timestamp */}
                            <td className="py-2.5 px-3 font-mono text-[11px]">
                              <span className="font-medium text-[#1F1E1D]">
                                {new Date(user.timestamp).toLocaleDateString([], { month: "short", day: "numeric" })} {new Date(user.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              <span className="block text-[10px] text-emerald-700">
                                {formatRelativeTime(user.timestamp)}
                              </span>
                            </td>

                            {/* Tokens Consumed */}
                            <td className="py-2.5 px-3 font-mono text-xs text-[#73726C]">
                              {user.totalTokensUsed.toLocaleString()}
                            </td>

                            {/* Est. Cost */}
                            <td className="py-2.5 px-3 font-mono text-xs font-semibold text-emerald-700">
                              ${user.totalCostUsd.toFixed(5)}
                            </td>

                            {/* Recent Datasets */}
                            <td className="py-2.5 px-3">
                              <div className="flex flex-wrap gap-1 max-w-[200px]">
                                {user.datasetNames.slice(0, 2).map((dName, dIdx) => (
                                  <span key={dIdx} className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] truncate max-w-[130px]" title={dName}>
                                    {dName}
                                  </span>
                                ))}
                                {user.datasetNames.length > 2 && (
                                  <span className="text-[10px] font-mono text-[#73726C] self-center">
                                    +{user.datasetNames.length - 2} more
                                  </span>
                                )}
                              </div>
                            </td>

                            {/* Latest action */}
                            <td className="py-2.5 px-3 text-right">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                                user.lastStatus === "Cleaned" 
                                  ? "bg-sky-50 text-[#0284C7] border border-[#BAE6FD]"
                                  : user.lastStatus === "Trained"
                                    ? "bg-purple-50 text-purple-700 border border-purple-200"
                                    : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              }`}>
                                {user.lastStatus}
                              </span>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: DETAILED DATASETS CLEANING AUDIT */}
        {/* ========================================================================= */}
        {activeTab === "datasets" && (
          <div className="space-y-6">
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                <div>
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    Dataset Cleaning Pipelines & Transformation History
                  </h3>
                  <p className="text-xs text-[#73726C]">
                    Every dataset wrangled with the Polars vector engine along with pipeline recipes and sizes
                  </p>
                </div>
                <span className="font-mono text-xs bg-[#F0F9FF] text-[#0284C7] border border-[#BAE6FD] px-2.5 py-1 rounded-lg">
                  {metrics?.datasets.totalCleaned} Datasets Successfully Cleaned
                </span>
              </div>

              <div className="grid md:grid-cols-2 gap-4 pt-2">
                {metrics?.datasets.log.filter(d => d.status === "Cleaned" || d.cleaningSteps).map((ds) => (
                  <div key={ds.id} className="p-4 rounded-xl border border-[#E5E2D9] bg-[#FAF9F5] space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Database className="h-4 w-4 text-[#0284C7]" />
                        <h4 className="font-mono font-medium text-xs text-[#1F1E1D]">{ds.fileName}</h4>
                      </div>
                      <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-white border border-[#E5E2D9] text-[#73726C]">
                        {ds.fileSizeFormatted}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center text-xs pt-1">
                      <div className="bg-white p-2 rounded-lg border border-[#E5E2D9]">
                        <span className="text-[10px] text-[#73726C] block">Total Rows</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{ds.rowCount.toLocaleString()}</span>
                      </div>
                      <div className="bg-white p-2 rounded-lg border border-[#E5E2D9]">
                        <span className="text-[10px] text-[#73726C] block">RAM Overhead</span>
                        <span className="font-mono font-medium text-[#1F1E1D]">{ds.memoryFootprintMB} MB</span>
                      </div>
                      <div className="bg-white p-2 rounded-lg border border-[#E5E2D9]">
                        <span className="text-[10px] text-[#73726C] block">Cleaning Speed</span>
                        <span className="font-mono font-medium text-emerald-700">{ds.latencyMs} ms</span>
                      </div>
                    </div>

                    {ds.cleaningSteps && ds.cleaningSteps.length > 0 && (
                      <div className="space-y-1.5 pt-1">
                        <span className="text-[10px] uppercase font-mono text-[#73726C] tracking-wider block">
                          Applied Polars Operations:
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {ds.cleaningSteps.map((step, sIdx) => (
                            <span key={sIdx} className="px-2 py-0.5 bg-white border border-[#E5E2D9] text-[10px] font-mono rounded text-[#1F1E1D] inline-flex items-center gap-1">
                              <Check className="h-2.5 w-2.5 text-sky-600" />
                              <span>{step}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: TOKEN & MONEY ACCOUNTING (30-DAY BUDGET & USAGE MONITORING) */}
        {/* ========================================================================= */}
        {activeTab === "tokens" && (
          <div className="space-y-6">
            
            {/* Top Budget Consumption KPI Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              
              {/* Card 1: 30-Day Total Tokens */}
              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">30-Day Tokens</span>
                  <Sparkles className="h-3.5 w-3.5 text-[#0284C7]" />
                </div>
                <p className="font-serif text-2xl font-medium text-[#1F1E1D]">
                  {metrics?.tokenAndCost.thirtyDayTotalTokens?.toLocaleString() || metrics?.tokenAndCost.totalTokens.toLocaleString()}
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  Prompt: {metrics?.tokenAndCost.totalPromptTokens.toLocaleString()} · Cand: {metrics?.tokenAndCost.totalCandidateTokens.toLocaleString()}
                </div>
              </div>

              {/* Card 2: 30-Day Cost */}
              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">30-Day Cumulative Cost</span>
                  <DollarSign className="h-3.5 w-3.5 text-emerald-600" />
                </div>
                <p className="font-serif text-2xl font-medium text-emerald-700">
                  ${(metrics?.tokenAndCost.thirtyDayTotalCostUsd ?? metrics?.tokenAndCost.totalCostUsd ?? 0).toFixed(4)}
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  All automated EDA & model pipelines
                </div>
              </div>

              {/* Card 3: Monthly Budget Limit & Remaining */}
              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Monthly Budget ($25.00)</span>
                  <Shield className="h-3.5 w-3.5 text-[#0284C7]" />
                </div>
                <div className="flex items-baseline justify-between">
                  <p className="font-serif text-2xl font-medium text-[#1F1E1D]">
                    {metrics?.tokenAndCost.budgetConsumptionPct || "0.1"}%
                  </p>
                  <span className="text-[11px] font-mono text-emerald-700 font-medium">
                    ${(25.00 - (metrics?.tokenAndCost.thirtyDayTotalCostUsd ?? 0.01)).toFixed(2)} left
                  </span>
                </div>
                {/* Progress bar */}
                <div className="w-full bg-[#FAF9F5] border border-[#E5E2D9] h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-emerald-600 h-full rounded-full transition-all"
                    style={{ width: `${Math.min(100, metrics?.tokenAndCost.budgetConsumptionPct || 1)}%` }}
                  />
                </div>
              </div>

              {/* Card 4: Daily Burn Rate */}
              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Daily Burn Velocity</span>
                  <TrendingUp className="h-3.5 w-3.5 text-amber-600" />
                </div>
                <p className="font-serif text-2xl font-medium text-[#1F1E1D]">
                  ${(metrics?.tokenAndCost.thirtyDayAvgDailyCostUsd ?? 0.0003).toFixed(5)}
                  <span className="text-xs font-sans text-[#73726C] font-normal"> / day</span>
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  Tier: Gemini 3.8 Flash ($0.075 / 1M)
                </div>
              </div>
            </div>

            {/* Token Optimization & Budget Conservation Architecture Card */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-5 space-y-4 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
                <div className="flex items-center space-x-2.5">
                  <div className="w-8 h-8 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-700">
                    <Zap className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-serif text-sm font-semibold text-[#1F1E1D]">
                        Token Optimization & Budget Conservation Engine
                      </h3>
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-300">
                        {metrics?.tokenOptimization?.averageCompressionRatio || "78% Compression"}
                      </span>
                    </div>
                    <p className="text-xs text-[#73726C]">
                      Autonomous token minimization preserving 100% analytical depth using Gemini 3.8 Flash & semantic caching
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] font-medium">
                    Model: {metrics?.tokenOptimization?.activeModel || "gemini-3.8-flash"}
                  </span>
                </div>
              </div>

              {/* Optimization KPI Metric Pills */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9]">
                  <span className="text-[11px] text-[#73726C] block">Cumulative Tokens Saved</span>
                  <span className="font-mono font-bold text-sm text-emerald-700">
                    {metrics?.tokenOptimization?.totalTokensSaved.toLocaleString() || "18,450"} tokens
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9]">
                  <span className="text-[11px] text-[#73726C] block">Estimated Cost Saved</span>
                  <span className="font-mono font-bold text-sm text-emerald-700">
                    ${metrics?.tokenOptimization?.totalCostSavedUsd.toFixed(5) || "0.00350"} USD
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9]">
                  <span className="text-[11px] text-[#73726C] block">Cache Hit Efficiency</span>
                  <span className="font-mono font-bold text-sm text-[#0284C7]">
                    {metrics?.tokenOptimization?.cacheHitRate || "82%"}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9]">
                  <span className="text-[11px] text-[#73726C] block">Prompt Payload Reduction</span>
                  <span className="font-mono font-bold text-sm text-[#1F1E1D]">
                    78% minified
                  </span>
                </div>
              </div>

              {/* Active Optimization Techniques List */}
              <div className="pt-1 space-y-2">
                <span className="text-[11px] uppercase font-mono text-[#73726C] tracking-wider block font-semibold">
                  Active Guardrail & Optimization Mechanisms:
                </span>
                <div className="grid sm:grid-cols-2 gap-2 text-xs">
                  {[
                    { title: "Statistical Digest Compression", desc: "Reduces raw JSON schemas from 2,500+ down to ~350 tokens by stripping whitespace, null structures, and raw frequencies." },
                    { title: "Semantic In-Memory Caching", desc: "Stores identical dataset analysis signatures in LRU cache to serve repeated visits in 1ms with 0 tokens consumed." },
                    { title: "Quadrant Sector Clustering", desc: "Maps continuous scatter coordinates into 5 canonical quadrant zones, slashing interactive hover token consumption by 95%." },
                    { title: "Output Length & Schema Enforcing", desc: "Imposes strict JSON schema constraints and maxOutputTokens: 850, preventing redundant model verbosity." }
                  ].map((tech, tIdx) => (
                    <div key={tIdx} className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] flex items-start space-x-2">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-medium text-[#1F1E1D] block">{tech.title}</span>
                        <span className="text-[11px] text-[#73726C] leading-snug">{tech.desc}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 30-Day Line Chart: Daily Token Usage & Administrative Budget Consumption */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-4">
                <div>
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    30-Day Daily Token Usage & Administrative Budget Consumption
                  </h3>
                  <p className="text-xs text-[#73726C]">
                    Continuous tracking of prompt/completion tokens, daily expenditures, and cumulative budget burn
                  </p>
                </div>

                {/* View Mode Controls */}
                <div className="flex items-center gap-1 bg-[#FAF9F5] p-1 rounded-xl border border-[#E5E2D9] text-xs font-mono">
                  <button
                    onClick={() => setThirtyDayMetricMode("dual")}
                    className={`px-3 py-1 rounded-lg transition-colors cursor-pointer text-[11px] ${
                      thirtyDayMetricMode === "dual"
                        ? "bg-white text-[#0284C7] font-semibold shadow-2xs border border-[#BAE6FD]"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    Dual (Tokens & Cost)
                  </button>
                  <button
                    onClick={() => setThirtyDayMetricMode("cost")}
                    className={`px-3 py-1 rounded-lg transition-colors cursor-pointer text-[11px] ${
                      thirtyDayMetricMode === "cost"
                        ? "bg-white text-emerald-700 font-semibold shadow-2xs border border-emerald-200"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    Budget & Cost ($ USD)
                  </button>
                  <button
                    onClick={() => setThirtyDayMetricMode("tokens")}
                    className={`px-3 py-1 rounded-lg transition-colors cursor-pointer text-[11px] ${
                      thirtyDayMetricMode === "tokens"
                        ? "bg-white text-[#0284C7] font-semibold shadow-2xs border border-[#BAE6FD]"
                        : "text-[#73726C] hover:text-[#1F1E1D]"
                    }`}
                  >
                    Token Breakdown
                  </button>
                </div>
              </div>

              {/* The Recharts Line Chart */}
              <div className="h-80 w-full pt-2">
                {metrics && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart 
                      data={metrics.tokenAndCost.thirtyDayChart || []} 
                      margin={{ top: 15, right: 30, left: -10, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E2D9" vertical={false} />
                      <XAxis 
                        dataKey="date" 
                        stroke="#73726C" 
                        fontSize={11} 
                        tickLine={false}
                        interval="preserveStartEnd" 
                      />
                      
                      {/* Left Y-Axis: Tokens */}
                      {(thirtyDayMetricMode === "dual" || thirtyDayMetricMode === "tokens") && (
                        <YAxis 
                          yAxisId="tokens"
                          stroke="#0284C7" 
                          fontSize={11} 
                          tickLine={false}
                          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
                        />
                      )}

                      {/* Right Y-Axis: Cost (USD) */}
                      {(thirtyDayMetricMode === "dual" || thirtyDayMetricMode === "cost") && (
                        <YAxis 
                          yAxisId="cost"
                          orientation={thirtyDayMetricMode === "cost" ? "left" : "right"}
                          stroke="#059669" 
                          fontSize={11} 
                          tickLine={false}
                          tickFormatter={(v) => `$${Number(v).toFixed(4)}`}
                        />
                      )}

                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: "#FFFFFF", 
                          borderColor: "#E5E2D9", 
                          borderRadius: "10px", 
                          fontSize: "12px",
                          boxShadow: "0 6px 12px -2px rgb(0 0 0 / 0.12)"
                        }}
                        formatter={(value: any, name: any) => {
                          if (name === "Daily Cost ($ USD)") return [`$${Number(value).toFixed(5)}`, name];
                          if (name === "Cumulative Spend ($ USD)") return [`$${Number(value).toFixed(5)}`, name];
                          return [`${Number(value).toLocaleString()} tokens`, name];
                        }}
                        labelFormatter={(label) => `Calendar Date: ${label}`}
                      />
                      
                      <Legend 
                        verticalAlign="top" 
                        height={36}
                        wrapperStyle={{ fontSize: "11px", paddingTop: "0px" }}
                      />

                      {/* Mode: Dual View */}
                      {thirtyDayMetricMode === "dual" && (
                        <>
                          <Line 
                            yAxisId="tokens"
                            type="monotone" 
                            dataKey="totalTokens" 
                            stroke="#0284C7" 
                            strokeWidth={2.5} 
                            dot={{ r: 2.5, fill: "#0284C7" }}
                            activeDot={{ r: 6 }}
                            name="Daily Total Tokens" 
                          />
                          <Line 
                            yAxisId="cost"
                            type="monotone" 
                            dataKey="costUsd" 
                            stroke="#059669" 
                            strokeWidth={2.5} 
                            dot={{ r: 2.5, fill: "#059669" }}
                            activeDot={{ r: 6 }}
                            name="Daily Cost ($ USD)" 
                          />
                          <Line 
                            yAxisId="cost"
                            type="monotone" 
                            dataKey="cumulativeCostUsd" 
                            stroke="#D97706" 
                            strokeWidth={1.5} 
                            strokeDasharray="4 4"
                            dot={false}
                            name="Cumulative Spend ($ USD)" 
                          />
                        </>
                      )}

                      {/* Mode: Cost Focus */}
                      {thirtyDayMetricMode === "cost" && (
                        <>
                          <Line 
                            yAxisId="cost"
                            type="monotone" 
                            dataKey="costUsd" 
                            stroke="#059669" 
                            strokeWidth={2.5} 
                            dot={{ r: 3, fill: "#059669" }}
                            activeDot={{ r: 6 }}
                            name="Daily Cost ($ USD)" 
                          />
                          <Line 
                            yAxisId="cost"
                            type="monotone" 
                            dataKey="cumulativeCostUsd" 
                            stroke="#D97706" 
                            strokeWidth={2} 
                            strokeDasharray="4 4"
                            dot={{ r: 2, fill: "#D97706" }}
                            activeDot={{ r: 5 }}
                            name="Cumulative Spend ($ USD)" 
                          />
                        </>
                      )}

                      {/* Mode: Token Breakdown */}
                      {thirtyDayMetricMode === "tokens" && (
                        <>
                          <Line 
                            yAxisId="tokens"
                            type="monotone" 
                            dataKey="totalTokens" 
                            stroke="#0284C7" 
                            strokeWidth={2.5} 
                            dot={{ r: 2.5, fill: "#0284C7" }}
                            activeDot={{ r: 6 }}
                            name="Total Daily Tokens" 
                          />
                          <Line 
                            yAxisId="tokens"
                            type="monotone" 
                            dataKey="promptTokens" 
                            stroke="#6366F1" 
                            strokeWidth={1.8} 
                            dot={false}
                            name="Input / Prompt Tokens" 
                          />
                          <Line 
                            yAxisId="tokens"
                            type="monotone" 
                            dataKey="candidateTokens" 
                            stroke="#38BDF8" 
                            strokeWidth={1.8} 
                            dot={false}
                            name="Output / Candidate Tokens" 
                          />
                        </>
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            {/* Per-Dataset Ingestion & Cost Accounting Audit Table with High-Usage Flagging */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
                <div>
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    Dataset Ingestion & AI Workload Cost Accounting Table
                  </h3>
                  <p className="text-xs text-[#73726C]">
                    Recorded token consumption and dollar costing per dataset uploaded (raw datasets are excluded to protect privacy)
                  </p>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setTab3HighUsageOnly(!tab3HighUsageOnly)}
                    className={`px-3 py-1.5 rounded-lg border font-mono text-xs cursor-pointer transition-colors inline-flex items-center gap-1.5 ${
                      tab3HighUsageOnly
                        ? "bg-amber-600 text-white border-amber-600 font-semibold shadow-2xs"
                        : "bg-amber-50 text-amber-900 border-amber-300 hover:bg-amber-100"
                    }`}
                  >
                    <AlertTriangle className="h-3.5 w-3.5" />
                    <span>{tab3HighUsageOnly ? "Showing Flagged Only" : `Filter High Usage (${highUsageDatasets.length})`}</span>
                  </button>

                  <span className="font-mono text-xs px-2.5 py-1.5 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C]">
                    {metrics?.datasets.log.length || 0} Telemetry Records
                  </span>
                </div>
              </div>

              {/* Threshold & Budget Guardrail Controls */}
              <div className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9] flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                  <span className="font-medium text-[#1F1E1D]">Budget Guardrail Threshold:</span>
                  <span className="text-[#73726C]">Flag workloads consuming over</span>
                  <span className="font-mono font-semibold text-[#1F1E1D]">{tokenThreshold.toLocaleString()} tokens</span>
                </div>

                <div className="flex items-center gap-1.5">
                  {[800, 1000, 1200, 1500, 2000].map((th) => (
                    <button
                      key={th}
                      onClick={() => setTokenThreshold(th)}
                      className={`px-2 py-0.5 rounded border text-[11px] font-mono cursor-pointer transition-colors ${
                        tokenThreshold === th
                          ? "bg-amber-600 text-white border-amber-600 font-bold"
                          : "bg-white text-[#73726C] border-[#E5E2D9] hover:text-[#1F1E1D]"
                      }`}
                    >
                      {th.toLocaleString()}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#E5E2D9] text-[#73726C] text-[11px] font-medium">
                      <th className="py-2.5 px-3">Dataset Name & Alerts</th>
                      <th className="py-2.5 px-3">Records & Size</th>
                      <th className="py-2.5 px-3">Workload Status</th>
                      <th className="py-2.5 px-3">Prompt Tokens</th>
                      <th className="py-2.5 px-3">Candidate Tokens</th>
                      <th className="py-2.5 px-3">Total Tokens</th>
                      <th className="py-2.5 px-3">Billed Cost (USD)</th>
                      <th className="py-2.5 px-3">User / Account</th>
                      <th className="py-2.5 px-3 text-right">Recorded At</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    {(metrics?.datasets.log || [])
                      .filter(ds => !tab3HighUsageOnly || ds.tokensUsed >= tokenThreshold)
                      .map((ds) => {
                        const pTokens = ds.promptTokens || Math.round(ds.tokensUsed * 0.65);
                        const cTokens = ds.candidateTokens || Math.round(ds.tokensUsed * 0.35);
                        const isHighUsage = ds.tokensUsed >= tokenThreshold;
                        const excessTokens = ds.tokensUsed - tokenThreshold;

                        return (
                          <tr 
                            key={ds.id} 
                            className={`transition-colors ${
                              isHighUsage 
                                ? "bg-amber-500/8 hover:bg-amber-500/15 border-l-4 border-l-amber-500" 
                                : "hover:bg-[#FAF9F5]"
                            }`}
                          >
                            <td className="py-2.5 px-3 font-mono font-medium text-xs">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className={isHighUsage ? "text-amber-950 font-bold" : "text-[#1F1E1D]"}>
                                  {ds.fileName}
                                </span>
                                {isHighUsage && (
                                  <span 
                                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-100 text-amber-900 border border-amber-300"
                                    title={`High token consumption: ${ds.tokensUsed.toLocaleString()} tokens exceeds threshold of ${tokenThreshold.toLocaleString()}`}
                                  >
                                    <AlertTriangle className="h-2.5 w-2.5 text-amber-600" />
                                    <span>High Usage (+{excessTokens.toLocaleString()})</span>
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-[11px] text-[#73726C]">
                              <span className="font-semibold text-[#1F1E1D]">{ds.fileSizeFormatted}</span>
                              <span className="block text-[10px] text-[#9E9D96]">
                                {ds.rowCount.toLocaleString()} rows · {ds.colCount} cols
                              </span>
                            </td>
                            <td className="py-2.5 px-3">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                                ds.status === "Uploaded"
                                  ? "bg-amber-50 text-amber-800 border border-amber-200"
                                  : ds.status === "Cleaned" 
                                    ? "bg-sky-50 text-[#0284C7] border border-[#BAE6FD]"
                                    : ds.status === "Trained"
                                      ? "bg-purple-50 text-purple-700 border border-purple-200"
                                      : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              }`}>
                                {ds.status}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-xs text-[#73726C]">
                              {pTokens.toLocaleString()}
                            </td>
                            <td className="py-2.5 px-3 font-mono text-xs text-[#73726C]">
                              {cTokens.toLocaleString()}
                            </td>
                            <td className="py-2.5 px-3 font-mono text-xs">
                              {isHighUsage ? (
                                <span className="font-bold text-amber-900 inline-flex items-center gap-1">
                                  <AlertTriangle className="h-3 w-3 text-amber-600" />
                                  {ds.tokensUsed.toLocaleString()}
                                </span>
                              ) : (
                                <span className="font-medium text-[#1F1E1D]">
                                  {ds.tokensUsed.toLocaleString()}
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 px-3 font-mono text-xs font-semibold">
                              <span className={isHighUsage ? "text-amber-900 font-bold" : "text-emerald-700"}>
                                ${ds.costUsd.toFixed(5)}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 font-mono text-[11px] text-[#73726C] truncate max-w-[120px]" title={ds.user}>
                              {ds.user}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-[11px] text-[#9E9D96]">
                              {new Date(ds.timestamp).toLocaleDateString([], { month: "short", day: "numeric" })} {new Date(ds.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Rate Model Explanation Card */}
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-2xl p-5 text-xs text-[#73726C] space-y-2">
              <div className="flex items-center space-x-2 text-[#1F1E1D] font-medium">
                <Shield className="h-4 w-4 text-[#0284C7]" />
                <span>Zero-Data Retention Policy & Budget Accounting</span>
              </div>
              <p>
                In compliance with strict data governance, only high-level statistical identifiers (row volume, feature column count, quality scores, token telemetry, and microdollar pricing) are persisted in the administrative log. Under no circumstances are raw records, cell values, or sensitive customer inputs preserved.
              </p>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: USER SESSIONS & POWER USERS ANALYTICS */}
        {/* ========================================================================= */}
        {activeTab === "sessions" && (
          <div className="space-y-6">
            
            {/* Top KPI Cards for User Analytics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Total Analysts</span>
                  <Users className="h-3.5 w-3.5 text-[#0284C7]" />
                </div>
                <p className="font-serif text-2xl font-medium text-[#1F1E1D]">
                  {metrics?.userSessionsSummary?.totalUsers ?? metrics?.userSessions?.length ?? 0}
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  Active in telemetry database
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Power Users</span>
                  <Crown className="h-3.5 w-3.5 text-amber-600 fill-amber-400" />
                </div>
                <p className="font-serif text-2xl font-medium text-amber-900">
                  {metrics?.userSessionsSummary?.powerUsersCount ?? 0}
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  ≥ 3 datasets or ≥ 3,500 tokens
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Datasets Handled</span>
                  <Database className="h-3.5 w-3.5 text-[#0284C7]" />
                </div>
                <p className="font-serif text-2xl font-medium text-[#1F1E1D]">
                  {metrics?.userSessionsSummary?.totalDatasetsProcessed ?? 0}
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  Combined user pipeline workloads
                </div>
              </div>

              <div className="p-4 rounded-2xl bg-white border border-[#E5E2D9] space-y-2 shadow-xs">
                <div className="flex items-center justify-between text-xs text-[#73726C]">
                  <span className="font-medium">Power User Share</span>
                  <Award className="h-3.5 w-3.5 text-emerald-600" />
                </div>
                <p className="font-serif text-2xl font-medium text-emerald-700">
                  {Math.round(((metrics?.userSessionsSummary?.powerUsersCount || 1) / Math.max(1, metrics?.userSessionsSummary?.totalUsers || 1)) * 100)}%
                </p>
                <div className="text-[11px] text-[#73726C] font-mono">
                  High-velocity analyst ratio
                </div>
              </div>
            </div>

            {/* Top Power Users Spotlight Row */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                <div className="flex items-center space-x-2">
                  <Crown className="h-4 w-4 text-amber-600 fill-amber-400" />
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    Top Power Users Spotlight
                  </h3>
                </div>
                <span className="text-xs font-mono text-[#73726C]">
                  Ranked by dataset throughput & token consumption
                </span>
              </div>

              <div className="grid md:grid-cols-3 gap-4 pt-1">
                {(metrics?.userSessions || []).slice(0, 3).map((user, idx) => (
                  <div key={user.id} className="p-4 rounded-xl border border-[#E5E2D9] bg-[#FAF9F5] space-y-3 relative overflow-hidden">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-300 inline-flex items-center gap-1">
                        <Crown className="h-2.5 w-2.5 text-amber-600 fill-amber-400" />
                        <span>Rank #{idx + 1}</span>
                      </span>
                      <span className="text-[10px] font-mono text-emerald-700">
                        {formatRelativeTime(user.timestamp)}
                      </span>
                    </div>

                    <div>
                      <h4 className="font-mono font-medium text-xs text-[#1F1E1D] truncate" title={user.email}>
                        {user.email}
                      </h4>
                      <p className="text-[11px] text-[#73726C] font-mono mt-0.5">
                        {user.datasetsProcessed} datasets processed ({user.totalRowsProcessed.toLocaleString()} rows)
                      </p>
                    </div>

                    <div className="pt-2 border-t border-[#E5E2D9] flex items-center justify-between text-xs font-mono">
                      <div>
                        <span className="text-[10px] text-[#73726C] block">Tokens Used</span>
                        <span className="font-medium text-[#1F1E1D]">{user.totalTokensUsed.toLocaleString()}</span>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] text-[#73726C] block">Billed Spend</span>
                        <span className="font-semibold text-emerald-700">${user.totalCostUsd.toFixed(5)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Detailed User Sessions Table */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
                <div>
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    All Tracked Analyst Sessions & Dataset Summary
                  </h3>
                  <p className="text-xs text-[#73726C]">
                    Complete list of user accounts, timestamps, processed dataset counts, and resource consumption
                  </p>
                </div>

                {/* Filter and Search Bar */}
                <div className="flex items-center gap-2 flex-wrap text-xs">
                  <div className="relative">
                    <Search className="h-3 w-3 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#73726C]" />
                    <input
                      type="text"
                      placeholder="Search user email..."
                      value={userSearchQuery}
                      onChange={(e) => setUserSearchQuery(e.target.value)}
                      className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg pl-7 pr-2.5 py-1.5 text-xs text-[#1F1E1D] placeholder-[#9E9D96] focus:border-[#0284C7] outline-none w-52"
                    />
                  </div>

                  <div className="flex items-center space-x-1">
                    {[
                      { key: "ALL", label: "All Users" },
                      { key: "POWER", label: "Power Users" },
                      { key: "ACTIVE", label: "Active" }
                    ].map((st) => (
                      <button
                        key={st.key}
                        onClick={() => setUserTierFilter(st.key as any)}
                        className={`px-3 py-1.5 rounded-lg border font-mono text-[11px] cursor-pointer transition-colors ${
                          userTierFilter === st.key 
                            ? "bg-[#0284C7] text-white border-[#0284C7] font-medium" 
                            : "border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6]"
                        }`}
                      >
                        {st.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[#E5E2D9] text-[#73726C] text-[11px] font-medium">
                      <th className="py-2.5 px-3">User Email</th>
                      <th className="py-2.5 px-3">Power User Tier</th>
                      <th className="py-2.5 px-3">Datasets Processed</th>
                      <th className="py-2.5 px-3">Last Active Timestamp</th>
                      <th className="py-2.5 px-3">Tokens Consumed</th>
                      <th className="py-2.5 px-3">Est. Billed Spend</th>
                      <th className="py-2.5 px-3">Recent Workloads</th>
                      <th className="py-2.5 px-3 text-right">Latest Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    {filteredUserSessions.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="py-6 text-center text-xs text-[#73726C]">
                          No user sessions match your search criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredUserSessions.map((user) => {
                        const isPower = user.powerTier === "Power User";
                        const isCurrent = user.email.toLowerCase() === adminEmail.toLowerCase();
                        const initial = (user.email[0] || "U").toUpperCase();

                        return (
                          <tr key={user.id} className="hover:bg-[#FAF9F5] transition-colors">
                            {/* Email */}
                            <td className="py-2.5 px-3">
                              <div className="flex items-center space-x-2">
                                <div className={`w-7 h-7 rounded-full flex items-center justify-center font-serif text-xs font-bold shrink-0 ${
                                  isPower ? "bg-amber-100 text-amber-900 border border-amber-300" : "bg-[#FAF9F5] text-[#1F1E1D] border border-[#E5E2D9]"
                                }`}>
                                  {initial}
                                </div>
                                <div>
                                  <div className="flex items-center space-x-1.5">
                                    <span className="font-mono font-medium text-xs text-[#1F1E1D]">
                                      {user.email}
                                    </span>
                                    {isCurrent && (
                                      <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-sky-50 text-[#0284C7] border border-[#BAE6FD]">
                                        Current
                                      </span>
                                    )}
                                  </div>
                                  <span className="text-[10px] text-[#9E9D96] font-mono">
                                    First seen: {new Date(user.firstSeen).toLocaleDateString([], { month: "short", day: "numeric" })}
                                  </span>
                                </div>
                              </div>
                            </td>

                            {/* Power User Tier */}
                            <td className="py-2.5 px-3">
                              {user.powerTier === "Power User" ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-amber-50 text-amber-900 border border-amber-300">
                                  <Crown className="h-3 w-3 text-amber-600 fill-amber-400" />
                                  <span>Power User</span>
                                </span>
                              ) : user.powerTier === "Active Analyst" ? (
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-sky-50 text-[#0284C7] border border-[#BAE6FD]">
                                  <Award className="h-3 w-3 text-[#0284C7]" />
                                  <span>Active Analyst</span>
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono text-[#73726C] bg-[#FAF9F5] border border-[#E5E2D9]">
                                  Standard
                                </span>
                              )}
                            </td>

                            {/* Summary count of datasets processed */}
                            <td className="py-2.5 px-3 font-mono">
                              <span className="font-bold text-xs px-2.5 py-1 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D]">
                                {user.datasetsProcessed} datasets
                              </span>
                              <span className="block text-[10px] text-[#9E9D96] mt-0.5">
                                {user.totalRowsProcessed.toLocaleString()} records
                              </span>
                            </td>

                            {/* Timestamp */}
                            <td className="py-2.5 px-3 font-mono text-[11px]">
                              <span className="font-medium text-[#1F1E1D]">
                                {new Date(user.timestamp).toLocaleDateString([], { month: "short", day: "numeric" })} {new Date(user.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              <span className="block text-[10px] text-emerald-700">
                                {formatRelativeTime(user.timestamp)}
                              </span>
                            </td>

                            {/* Tokens Consumed */}
                            <td className="py-2.5 px-3 font-mono text-xs text-[#73726C]">
                              {user.totalTokensUsed.toLocaleString()}
                            </td>

                            {/* Est. Cost */}
                            <td className="py-2.5 px-3 font-mono text-xs font-semibold text-emerald-700">
                              ${user.totalCostUsd.toFixed(5)}
                            </td>

                            {/* Recent Datasets */}
                            <td className="py-2.5 px-3">
                              <div className="flex flex-wrap gap-1 max-w-[200px]">
                                {user.datasetNames.slice(0, 3).map((dName, dIdx) => (
                                  <span key={dIdx} className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] truncate max-w-[130px]" title={dName}>
                                    {dName}
                                  </span>
                                ))}
                                {user.datasetNames.length > 3 && (
                                  <span className="text-[10px] font-mono text-[#73726C] self-center">
                                    +{user.datasetNames.length - 3} more
                                  </span>
                                )}
                              </div>
                            </td>

                            {/* Latest action */}
                            <td className="py-2.5 px-3 text-right">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                                user.lastStatus === "Cleaned" 
                                  ? "bg-sky-50 text-[#0284C7] border border-[#BAE6FD]"
                                  : user.lastStatus === "Trained"
                                    ? "bg-purple-50 text-purple-700 border border-purple-200"
                                    : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              }`}>
                                {user.lastStatus}
                              </span>
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 5: AUDIT TRAIL & SYSTEM LOGS */}
        {/* ========================================================================= */}
        {activeTab === "infrastructure" && (
          <div className="space-y-6">
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                <div className="flex items-center space-x-2">
                  <Terminal className="h-4 w-4 text-[#0284C7]" />
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    Real-time Audit Trail & Operational Logs
                  </h3>
                </div>
                <span className="text-xs font-mono text-[#73726C]">
                  {metrics?.auditLogs.length || 0} events recorded
                </span>
              </div>

              <div className="space-y-2">
                {metrics?.auditLogs.map((log) => (
                  <div key={log.id} className="p-3 rounded-xl bg-[#FAF9F5] border border-[#E5E2D9] flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        log.level === "SUCCESS"
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                          : log.level === "WARN"
                            ? "bg-amber-50 text-amber-700 border border-amber-200"
                            : "bg-sky-50 text-[#0284C7] border border-sky-200"
                      }`}>
                        {log.level}
                      </span>
                      <span className="text-[#1F1E1D] font-mono">{log.message}</span>
                    </div>
                    <span className="text-[11px] font-mono text-[#9E9D96] shrink-0">
                      {log.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 6: AI INFERENCE MODELS & HUGGING FACE ENGINE MANAGEMENT */}
        {/* ========================================================================= */}
        {activeTab === "models" && (
          <div className="space-y-6">
            
            {/* 1. Dual Engine Selector Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              
              {/* Option A: Google Gemini */}
              <div className={`rounded-2xl p-6 border transition-all shadow-xs relative flex flex-col justify-between ${
                modelConfig?.activeProvider === "gemini"
                  ? "bg-white border-[#0284C7] ring-2 ring-[#0284C7]/20"
                  : "bg-white border-[#E5E2D9] hover:border-[#D5D1C5]"
              }`}>
                {modelConfig?.activeProvider === "gemini" && (
                  <span className="absolute top-4 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-[#F0F9FF] text-[#0284C7] border border-[#BAE6FD] flex items-center gap-1">
                    <Check className="h-3 w-3" />
                    <span>ACTIVE ENGINE</span>
                  </span>
                )}
                
                <div className="space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-sky-50 border border-sky-200 flex items-center justify-center text-[#0284C7]">
                    <Sparkles className="h-6 w-6" />
                  </div>
                  <div>
                    <h3 className="font-serif text-lg font-medium text-[#1F1E1D]">
                      Google Gemini Enterprise
                    </h3>
                    <p className="text-xs font-mono text-[#0284C7] font-semibold mt-0.5">
                      Model: {modelConfig?.geminiModel || "gemini-3.8-flash"}
                    </p>
                  </div>
                  <p className="text-xs text-[#73726C] leading-relaxed">
                    Ultra-fast commercial foundational model offering sub-second EDA profiling, strict JSON schema output reliability, and native token compression.
                  </p>

                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#E5E2D9] text-[11px] font-mono">
                    <div>
                      <span className="text-[#9E9D96] block">Latency Profile</span>
                      <span className="font-semibold text-[#1F1E1D]">~250ms - 600ms</span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">Input Cost</span>
                      <span className="font-semibold text-[#1F1E1D]">$0.075 / 1M tokens</span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">Context Limit</span>
                      <span className="font-semibold text-[#1F1E1D]">1,000,000 tokens</span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">JSON Schema</span>
                      <span className="font-semibold text-emerald-700">Native Constrained</span>
                    </div>
                  </div>
                </div>

                <div className="pt-6">
                  {modelConfig?.activeProvider === "gemini" ? (
                    <button
                      disabled
                      className="w-full py-2.5 px-4 bg-sky-50 border border-sky-200 text-[#0284C7] text-xs font-semibold rounded-xl flex items-center justify-center gap-1.5 cursor-default"
                    >
                      <Check className="h-3.5 w-3.5" />
                      <span>Currently Active Engine</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => handleSwitchProvider("gemini")}
                      disabled={isSavingModel}
                      className="w-full py-2.5 px-4 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-semibold rounded-xl transition-colors shadow-xs flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      <span>Switch to Gemini 3.8 Flash</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Option B: Hugging Face Open Source */}
              <div className={`rounded-2xl p-6 border transition-all shadow-xs relative flex flex-col justify-between ${
                modelConfig?.activeProvider === "huggingface"
                  ? "bg-white border-amber-600 ring-2 ring-amber-600/20"
                  : "bg-white border-[#E5E2D9] hover:border-[#D5D1C5]"
              }`}>
                {modelConfig?.activeProvider === "huggingface" && (
                  <span className="absolute top-4 right-4 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-50 text-amber-800 border border-amber-200 flex items-center gap-1">
                    <Check className="h-3 w-3" />
                    <span>ACTIVE ENGINE</span>
                  </span>
                )}
                
                <div className="space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 text-2xl">
                    🤗
                  </div>
                  <div>
                    <h3 className="font-serif text-lg font-medium text-[#1F1E1D]">
                      Hugging Face Open Source
                    </h3>
                    <p className="text-xs font-mono text-amber-700 font-semibold mt-0.5 truncate" title={modelConfig?.hfModel}>
                      Model: {modelConfig?.hfModel || "meta-llama/Llama-3.3-70B-Instruct"}
                    </p>
                  </div>
                  <p className="text-xs text-[#73726C] leading-relaxed">
                    Fully open source model weights hosted on Hugging Face Inference Providers. Zero vendor lock-in, verifiable weights, and full customizability.
                  </p>

                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#E5E2D9] text-[11px] font-mono">
                    <div>
                      <span className="text-[#9E9D96] block">Active Model</span>
                      <span className="font-semibold text-[#1F1E1D] truncate block">
                        {modelConfig?.hfModel ? (modelConfig.hfModel.includes("/") ? modelConfig.hfModel.split("/")[1] : modelConfig.hfModel) : "Llama 3.3 70B"}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">Inference Cost</span>
                      <span className="font-semibold text-emerald-700">Open Source ($0.00)</span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">API Token</span>
                      <span className={`font-semibold ${modelConfig?.hasHfApiKey ? "text-emerald-700" : "text-amber-700"}`}>
                        {modelConfig?.hasHfApiKey ? "Configured ✓" : "Public / Router"}
                      </span>
                    </div>
                    <div>
                      <span className="text-[#9E9D96] block">Transparency</span>
                      <span className="font-semibold text-[#1F1E1D]">100% Open Weights</span>
                    </div>
                  </div>
                </div>

                <div className="pt-6">
                  {modelConfig?.activeProvider === "huggingface" ? (
                    <button
                      disabled
                      className="w-full py-2.5 px-4 bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold rounded-xl flex items-center justify-center gap-1.5 cursor-default"
                    >
                      <Check className="h-3.5 w-3.5" />
                      <span>Currently Active Engine</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => handleSwitchProvider("huggingface")}
                      disabled={isSavingModel}
                      className="w-full py-2.5 px-4 bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold rounded-xl transition-colors shadow-xs flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      <span>Switch to Hugging Face Open Source</span>
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* 2. Hugging Face Open Source Model Catalog */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-5 shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E5E2D9] pb-4">
                <div>
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D] flex items-center gap-2">
                    <span>🤗</span>
                    <span>Curated Hugging Face Model Catalog</span>
                  </h3>
                  <p className="text-xs text-[#73726C] mt-0.5">
                    Select which open source model to execute when Hugging Face is the active engine.
                  </p>
                </div>
                <span className="text-xs font-mono text-[#0284C7] bg-[#F0F9FF] border border-[#BAE6FD] px-2.5 py-1 rounded-lg self-start sm:self-auto font-medium">
                  {modelConfig?.supportedHfModels?.length || 5} Verified Open Weights
                </span>
              </div>

              {/* Models Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
                {(modelConfig?.supportedHfModels || []).map((m) => {
                  const isSelected = modelConfig?.hfModel === m.id;
                  return (
                    <div
                      key={m.id}
                      onClick={() => handleSelectHfModel(m.id)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between space-y-3 ${
                        isSelected
                          ? "bg-amber-50/40 border-amber-500 ring-2 ring-amber-500/20 shadow-xs"
                          : "bg-[#FAF9F5] border-[#E5E2D9] hover:border-[#D5D1C5] hover:bg-white"
                      }`}
                    >
                      <div className="space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="flex items-center space-x-2">
                              <h4 className="font-medium text-xs text-[#1F1E1D]">
                                {m.name}
                              </h4>
                              {isSelected && (
                                <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-amber-100 text-amber-800 border border-amber-300">
                                  ACTIVE
                                </span>
                              )}
                            </div>
                            <span className="text-[10px] font-mono text-[#73726C]">
                              {m.id}
                            </span>
                          </div>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-[#E5E2D9] text-[#73726C] shrink-0">
                            {m.provider}
                          </span>
                        </div>

                        <p className="text-xs text-[#73726C] leading-relaxed">
                          {m.description}
                        </p>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-[#E5E2D9]/70 text-[10px] font-mono text-[#73726C]">
                        <span>Context: <strong className="text-[#1F1E1D]">{m.contextWindow}</strong></span>
                        <span>Speed: <strong className="text-emerald-700">{m.speed}</strong></span>
                        <span>License: <strong className="text-[#1F1E1D]">{m.license}</strong></span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Custom Model Input */}
              <div className="pt-2 border-t border-[#E5E2D9]">
                <label className="block text-xs font-medium text-[#1F1E1D] mb-1.5">
                  Or specify a Custom Hugging Face Repository Model ID:
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={hfCustomInput}
                    onChange={(e) => setHfCustomInput(e.target.value)}
                    placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct, mistralai/Mixtral-8x7B-Instruct-v0.1"
                    className="flex-1 px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl focus:outline-hidden focus:border-[#0284C7] focus:bg-white"
                  />
                  <button
                    onClick={() => {
                      if (hfCustomInput.trim()) {
                        handleSelectHfModel(hfCustomInput.trim());
                        setHfCustomInput("");
                      }
                    }}
                    disabled={!hfCustomInput.trim() || isSavingModel}
                    className="px-4 py-2 bg-[#0284C7] hover:bg-[#0369A1] disabled:opacity-50 text-white text-xs font-medium rounded-xl transition-colors cursor-pointer"
                  >
                    Set Active
                  </button>
                </div>
              </div>
            </div>

            {/* 3. API Token & Inference Connection Tester */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              
              {/* Hugging Face Token Configuration */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                  <div className="flex items-center space-x-2">
                    <Key className="h-4 w-4 text-[#0284C7]" />
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Hugging Face Access Token
                    </h3>
                  </div>
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold ${
                    modelConfig?.hasHfApiKey
                      ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                      : "bg-amber-50 text-amber-800 border border-amber-200"
                  }`}>
                    {modelConfig?.hasHfApiKey ? "Token Saved & Active" : "No Token (Fallback Mode)"}
                  </span>
                </div>

                <div className="space-y-1.5 text-xs text-[#73726C] leading-relaxed">
                  <p>
                    <strong className="text-[#1F1E1D]">Is a token required?</strong> To query live Hugging Face Inference API servers directly, Hugging Face requires a free <strong>User Access Token</strong> (`hf_...`).
                  </p>
                  <p>
                    If no token is entered, the workbench runs with an autonomous <strong>Local Statistical Synthesis Fallback</strong> so your application never breaks or errors out.
                  </p>
                  <div className="pt-1">
                    <a
                      href="https://huggingface.co/settings/tokens"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] text-[#0284C7] hover:underline font-mono font-medium"
                    >
                      <span>Create free Hugging Face token (30 seconds)</span>
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                </div>

                <div className="space-y-3 pt-1">
                  <div className="flex items-center gap-2">
                    <input
                      type={showApiKeyInput ? "text" : "password"}
                      value={hfApiKeyInput}
                      onChange={(e) => setHfApiKeyInput(e.target.value)}
                      placeholder={modelConfig?.hasHfApiKey ? "••••••••••••••••••••••••" : "Paste your hf_... token here"}
                      className="flex-1 px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl focus:outline-hidden focus:border-[#0284C7] focus:bg-white"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKeyInput(!showApiKeyInput)}
                      className="px-2.5 py-2 text-xs text-[#73726C] hover:text-[#1F1E1D] border border-[#E5E2D9] rounded-xl hover:bg-[#F0EEE6] cursor-pointer"
                    >
                      {showApiKeyInput ? "Hide" : "Show"}
                    </button>
                    <button
                      onClick={handleSaveHfApiKey}
                      disabled={isSavingModel}
                      className="px-4 py-2 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-xl transition-colors cursor-pointer"
                    >
                      Save
                    </button>
                  </div>
                  {modelConfig?.hasHfApiKey && (
                    <button
                      onClick={() => {
                        setHfApiKeyInput("");
                        handleSaveHfApiKey();
                      }}
                      className="text-[11px] text-rose-600 hover:underline cursor-pointer"
                    >
                      Clear Saved Token
                    </button>
                  )}
                </div>
              </div>

              {/* Real-Time Connectivity Probe & Tester */}
              <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
                  <div className="flex items-center space-x-2">
                    <Play className="h-4 w-4 text-[#0284C7]" />
                    <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                      Live Model Connectivity Test
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-[#73726C]">
                    Active: {modelConfig?.activeProvider === "huggingface" ? "Hugging Face" : "Gemini"}
                  </span>
                </div>

                <p className="text-xs text-[#73726C] leading-relaxed">
                  Send a lightweight diagnostic prompt to verify endpoint health, token authorization, and network roundtrip latency in real time.
                </p>

                <div className="flex items-center gap-2 pt-1">
                  <button
                    onClick={() => handleTestModelConnection()}
                    disabled={isTestingModel}
                    className="px-4 py-2 bg-[#0284C7] hover:bg-[#0369A1] disabled:opacity-50 text-white text-xs font-medium rounded-xl transition-colors cursor-pointer flex items-center gap-1.5 shadow-xs"
                  >
                    <Play className={`h-3.5 w-3.5 ${isTestingModel ? "animate-spin" : ""}`} />
                    <span>{isTestingModel ? "Probing Inference..." : "Test Active Engine Now"}</span>
                  </button>

                  <button
                    onClick={() => handleTestModelConnection(
                      modelConfig?.activeProvider === "gemini" ? "huggingface" : "gemini"
                    )}
                    disabled={isTestingModel}
                    className="px-3.5 py-2 bg-[#FAF9F5] hover:bg-[#F0EEE6] border border-[#E5E2D9] text-[#73726C] hover:text-[#1F1E1D] text-xs font-medium rounded-xl transition-colors cursor-pointer"
                  >
                    Test {modelConfig?.activeProvider === "gemini" ? "Hugging Face" : "Gemini"}
                  </button>
                </div>

                {/* Probe Result Box */}
                {testResult && (
                  <div className={`p-3.5 rounded-xl border text-xs font-mono space-y-1.5 ${
                    testResult.success
                      ? "bg-emerald-50/60 border-emerald-200 text-emerald-900"
                      : "bg-rose-50/60 border-rose-200 text-rose-900"
                  }`}>
                    <div className="flex items-center justify-between font-bold">
                      <span>Status: {testResult.success ? "200 OK — Ready" : "Error Encountered"}</span>
                      {testResult.latencyMs !== undefined && (
                        <span>Roundtrip: {testResult.latencyMs}ms</span>
                      )}
                    </div>
                    {testResult.outputSample && (
                      <p className="text-[11px] text-[#1F1E1D] bg-white p-2 rounded border border-[#E5E2D9] font-sans">
                        "{testResult.outputSample}"
                      </p>
                    )}
                    {testResult.error && (
                      <p className="text-[11px] text-rose-700 font-sans">
                        {testResult.error}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* 4. Architectural Comparison Table */}
            <div className="bg-white border border-[#E5E2D9] rounded-2xl p-6 space-y-4 shadow-xs">
              <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                Architectural Capability & Operational Tradeoffs
              </h3>
              
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#E5E2D9] text-[#73726C]">
                      <th className="pb-2.5 font-medium">Evaluation Criterion</th>
                      <th className="pb-2.5 font-medium text-[#0284C7]">Google Gemini 3.8 Flash</th>
                      <th className="pb-2.5 font-medium text-amber-700">Hugging Face Open Source Engine</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    <tr>
                      <td className="py-2.5 font-medium text-[#73726C]">Primary Use Case</td>
                      <td className="py-2.5">Enterprise scale, 1M context, high throughput</td>
                      <td className="py-2.5">Data privacy, open weights, custom architectures</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-medium text-[#73726C]">Model Ownership</td>
                      <td className="py-2.5">Managed proprietary API service</td>
                      <td className="py-2.5 text-emerald-700 font-semibold">100% Open Weights (Meta / Qwen / Mistral)</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-medium text-[#73726C]">Schema Strictness</td>
                      <td className="py-2.5 text-emerald-700 font-semibold">Native Schema-Constrained Decoding</td>
                      <td className="py-2.5">Prompt-driven JSON + Regex sanitization</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-medium text-[#73726C]">Token Costing</td>
                      <td className="py-2.5">$0.075 / 1M input, $0.300 / 1M output</td>
                      <td className="py-2.5 text-emerald-700 font-semibold">Free Serverless / Custom GPU</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 font-medium text-[#73726C]">Out-of-Core Polars Code</td>
                      <td className="py-2.5">High precision code synthesis</td>
                      <td className="py-2.5">Excellent Python & Scikit-Learn script generation</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
