import React, { useState, useMemo, useEffect } from "react";
import { 
  DatasetSummary, 
  ModelRecommendationItem, 
  BenchmarkResult, 
  HPOOptimizationResult, 
  HPOTrialResult 
} from "../types";
import { 
  analyzeEdaAndRecommendModels, 
  runModelBenchmark, 
  runHyperparameterOptimization 
} from "../utils/automl";
import { preprocessData, trainTestSplit, standardScale } from "../utils/ml";
import { 
  Sparkles, 
  Trophy, 
  Sliders, 
  Check, 
  Play, 
  RotateCw, 
  ArrowUpRight, 
  ShieldCheck, 
  TrendingUp, 
  Zap, 
  FileText, 
  AlertCircle,
  HelpCircle,
  Cpu
} from "lucide-react";
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell 
} from "recharts";

interface AutoMLPanelProps {
  summary: DatasetSummary;
  rawData?: any[] | null;
  targetCol: string;
  taskType: "Classification" | "Regression";
  activeFeatures: string[];
  currentModelId: string;
  onSelectModel: (modelId: string, suggestedParams?: Record<string, any>) => void;
  onApplyTunedParams: (params: Record<string, any>) => void;
}

export default function AutoMLPanel({
  summary,
  rawData,
  targetCol,
  taskType,
  activeFeatures,
  currentModelId,
  onSelectModel,
  onApplyTunedParams,
}: AutoMLPanelProps) {
  const [activeSubTab, setActiveSubTab] = useState<"recommendations" | "benchmark" | "hpo">("recommendations");

  // User objective configuration
  const [objectiveGoal, setObjectiveGoal] = useState<"balanced" | "accuracy" | "interpretability" | "speed">("balanced");

  // Recommendation items computed from EDA
  const recommendations = useMemo(() => {
    return analyzeEdaAndRecommendModels(summary, targetCol, taskType, objectiveGoal);
  }, [summary, targetCol, taskType, objectiveGoal]);

  // Benchmark State
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState<{ step: number; total: number; currentModel: string }>({ step: 0, total: 4, currentModel: "" });
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[] | null>(null);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  // HPO State
  const [hpoModelId, setHpoModelId] = useState<string>(currentModelId || "decision_tree");
  const [hpoStrategy, setHpoStrategy] = useState<"random" | "grid">("random");
  const [hpoNumTrials, setHpoNumTrials] = useState<number>(10);
  const [isTuning, setIsTuning] = useState(false);
  const [hpoProgressPct, setHpoProgressPct] = useState<number>(0);
  const [currentTrial, setCurrentTrial] = useState<HPOTrialResult | null>(null);
  const [liveTrials, setLiveTrials] = useState<HPOTrialResult[]>([]);
  const [hpoResult, setHpoResult] = useState<HPOOptimizationResult | null>(null);
  const [hpoError, setHpoError] = useState<string | null>(null);
  const [paramsApplied, setParamsApplied] = useState(false);

  // Synchronize HPO selected model with current model if user switches
  useEffect(() => {
    if (currentModelId) {
      setHpoModelId(currentModelId);
    }
  }, [currentModelId]);

  // Execute Benchmark Tournament
  const handleRunBenchmark = async () => {
    if (activeFeatures.length === 0) {
      setBenchmarkError("Select at least one predictor feature to benchmark models.");
      return;
    }

    setIsBenchmarking(true);
    setBenchmarkError(null);
    setBenchmarkResults(null);

    try {
      // Preprocess data
      const dataToUse = rawData && rawData.length > 0
        ? (rawData.length > 2000 ? rawData.slice(0, 2000) : rawData)
        : summary.sampleData;
      const prep = preprocessData(dataToUse, activeFeatures, targetCol, taskType);
      const { XTrain, yTrain, XTest, yTest } = trainTestSplit(prep.X, prep.y, 0.2, 42);
      const { XTrainScaled, XTestScaled } = standardScale(XTrain, XTest);

      const results = await runModelBenchmark(
        XTrain,
        yTrain,
        XTest,
        yTest,
        XTrainScaled,
        XTestScaled,
        prep.classLabels,
        taskType,
        (step, total, currentModel) => {
          setBenchmarkProgress({ step, total, currentModel });
        }
      );

      setBenchmarkResults(results);
    } catch (err: any) {
      console.error(err);
      setBenchmarkError(err.message || "Model benchmark encountered an error.");
    } finally {
      setIsBenchmarking(false);
    }
  };

  // Execute Automated Hyperparameter Optimization
  const handleRunHPO = async () => {
    if (activeFeatures.length === 0) {
      setHpoError("Select at least one predictor feature to tune hyperparameters.");
      return;
    }

    setIsTuning(true);
    setHpoError(null);
    setHpoResult(null);
    setLiveTrials([]);
    setCurrentTrial(null);
    setHpoProgressPct(0);
    setParamsApplied(false);

    try {
      const dataToUse = rawData && rawData.length > 0
        ? (rawData.length > 2000 ? rawData.slice(0, 2000) : rawData)
        : summary.sampleData;
      const prep = preprocessData(dataToUse, activeFeatures, targetCol, taskType);
      const { XTrain, yTrain, XTest, yTest } = trainTestSplit(prep.X, prep.y, 0.2, 42);
      const { XTrainScaled, XTestScaled } = standardScale(XTrain, XTest);

      const modelNameMap: Record<string, string> = {
        decision_tree: taskType === "Classification" ? "Decision Tree Classifier" : "Decision Tree Regressor",
        random_forest: taskType === "Classification" ? "Random Forest Classifier" : "Random Forest Regressor",
        logistic: "Logistic Regression (OLS)",
        regularized_logistic: "Ridge Logistic (L2)",
        linear: "Linear Regression (OLS)",
        ridge: "Ridge Regression (L2)"
      };

      const result = await runHyperparameterOptimization(
        hpoModelId,
        modelNameMap[hpoModelId] || hpoModelId,
        XTrain,
        yTrain,
        XTest,
        yTest,
        XTrainScaled,
        XTestScaled,
        prep.classLabels,
        taskType,
        {
          strategy: hpoStrategy,
          numTrials: hpoNumTrials,
          targetMetric: taskType === "Classification" ? "Accuracy" : "R2 Score (R²)",
          onTrialComplete: (trial, progress) => {
            setCurrentTrial(trial);
            setLiveTrials(prev => [...prev, trial]);
            setHpoProgressPct(progress);
          }
        }
      );

      setHpoResult(result);
    } catch (err: any) {
      console.error(err);
      setHpoError(err.message || "Hyperparameter optimization failed. Verify data dimensions.");
    } finally {
      setIsTuning(false);
    }
  };

  const handleApplyBestParams = () => {
    if (hpoResult?.bestParams) {
      onSelectModel(hpoResult.modelId, hpoResult.bestParams);
      onApplyTunedParams(hpoResult.bestParams);
      setParamsApplied(true);
      setTimeout(() => setParamsApplied(false), 3000);
    }
  };

  return (
    <div id="automl-panel" className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-5 space-y-6">
      {/* Top Banner & Mode Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#E5E2D9] pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="h-4 w-4 text-sky-400" />
            <h3 className="font-mono font-bold text-[#1F1E1D] text-sm tracking-wide uppercase">
              AutoML Intelligence Suite
            </h3>
            <span className="text-[10px] bg-sky-500/10 text-sky-400 border border-sky-500/30 px-2 py-0.5 rounded font-mono font-semibold">
              EDA Grounded
            </span>
          </div>
          <p className="text-xs text-[#73726C] mt-1">
            Automated model suitability recommendations, algorithm benchmarking tournaments, and hyperparameter tuning.
          </p>
        </div>

        {/* Sub-tabs */}
        <div className="flex bg-[#FAF9F5] p-1 rounded border border-[#E5E2D9] self-start sm:self-auto font-mono text-xs">
          <button
            id="tab-btn-recommendations"
            onClick={() => setActiveSubTab("recommendations")}
            className={`px-3 py-1.5 rounded font-medium transition-colors flex items-center space-x-1.5 cursor-pointer ${
              activeSubTab === "recommendations" 
                ? "bg-[#F0EEE6] text-sky-400 shadow-sm" 
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Sparkles className="h-3 w-3" />
            <span>Recommendations</span>
          </button>
          <button
            id="tab-btn-benchmark"
            onClick={() => setActiveSubTab("benchmark")}
            className={`px-3 py-1.5 rounded font-medium transition-colors flex items-center space-x-1.5 cursor-pointer ${
              activeSubTab === "benchmark" 
                ? "bg-[#F0EEE6] text-sky-400 shadow-sm" 
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Trophy className="h-3 w-3" />
            <span>Benchmark Tournament</span>
          </button>
          <button
            id="tab-btn-hpo"
            onClick={() => setActiveSubTab("hpo")}
            className={`px-3 py-1.5 rounded font-medium transition-colors flex items-center space-x-1.5 cursor-pointer ${
              activeSubTab === "hpo" 
                ? "bg-[#F0EEE6] text-sky-400 shadow-sm" 
                : "text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            <Sliders className="h-3 w-3" />
            <span>Hyperparameter Tuning</span>
          </button>
        </div>
      </div>

      {/* ================= VIEW 1: EDA MODEL RECOMMENDATIONS ================= */}
      {activeSubTab === "recommendations" && (
        <div className="space-y-5">
          {/* User Objective Selector */}
          <div className="bg-white p-3.5 rounded border border-[#E5E2D9] flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <span className="text-xs font-mono font-semibold text-[#73726C]">Optimization Goal / Objective:</span>
              <p className="text-[11px] text-[#9E9D96]">Auto-tailor model rankings to your practical deployment objective.</p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {[
                { id: "balanced", label: "Balanced Score" },
                { id: "accuracy", label: "Max Accuracy / R²" },
                { id: "interpretability", label: "Explainable AI (Rules)" },
                { id: "speed", label: "Low Latency / Speed" },
              ].map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setObjectiveGoal(opt.id as any)}
                  className={`px-2.5 py-1 text-[11px] font-mono rounded border transition-colors cursor-pointer ${
                    objectiveGoal === opt.id
                      ? "bg-sky-500/10 border-sky-500/40 text-sky-400 font-bold"
                      : "bg-white border-[#E5E2D9] text-[#73726C] hover:text-[#73726C]"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Model Cards Grid */}
          <div className="grid md:grid-cols-2 gap-4">
            {recommendations.map((rec, idx) => {
              const isSelected = currentModelId === rec.modelId;
              return (
                <div
                  key={rec.modelId}
                  className={`rounded border p-4 flex flex-col justify-between space-y-3.5 transition-all ${
                    isSelected 
                      ? "bg-sky-50 border-sky-500/40 shadow-sm" 
                      : "bg-white border-[#E5E2D9] hover:border-[#D5D1C5]"
                  }`}
                >
                  {/* Card Header */}
                  <div className="space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="space-y-0.5">
                        <div className="flex items-center space-x-2">
                          <span className="text-[10px] font-mono text-[#73726C] font-bold bg-white px-1.5 py-0.5 rounded border border-[#E5E2D9]">
                            #{idx + 1}
                          </span>
                          <h4 className="font-mono font-bold text-[#1F1E1D] text-xs">
                            {rec.name}
                          </h4>
                        </div>
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded inline-block font-semibold ${
                          rec.recommendationLevel === "Highly Recommended"
                            ? "bg-sky-500/15 text-sky-400 border border-sky-500/30"
                            : rec.recommendationLevel === "Recommended"
                            ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                            : rec.recommendationLevel === "Viable Alternative"
                            ? "bg-[#F0EEE6] text-[#73726C]"
                            : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                        }`}>
                          {rec.recommendationLevel}
                        </span>
                      </div>

                      {/* Suitability Score Meter */}
                      <div className="text-right">
                        <span className="text-base font-mono font-bold text-[#1F1E1D]">
                          {rec.suitabilityScore}%
                        </span>
                        <p className="text-[9px] font-mono text-[#9E9D96] uppercase">Suitability</p>
                      </div>
                    </div>

                    {/* Progress Bar for Suitability */}
                    <div className="w-full bg-white h-1.5 rounded overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${
                          rec.suitabilityScore >= 85 ? "bg-sky-400" : rec.suitabilityScore >= 70 ? "bg-blue-400" : "bg-sky-400"
                        }`}
                        style={{ width: `${rec.suitabilityScore}%` }}
                      />
                    </div>
                  </div>

                  {/* EDA Signals (Why recommended based on data profile) */}
                  <div className="space-y-1.5 bg-[#FAF9F5] p-2.5 rounded border border-[#E5E2D9]">
                    <span className="text-[10px] font-mono font-bold text-[#73726C] flex items-center space-x-1">
                      <ShieldCheck className="h-3 w-3 text-sky-400" />
                      <span>EDA Signal Matching:</span>
                    </span>
                    <ul className="space-y-1 text-[11px] text-[#73726C] font-mono">
                      {rec.edaSignals.map((signal, sIdx) => (
                        <li key={sIdx} className="flex items-start space-x-1.5 text-[#73726C] leading-snug">
                          <span className="text-sky-400 flex-shrink-0">›</span>
                          <span>{signal}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Pros & Cons summary */}
                  <div className="space-y-1 text-[11px]">
                    <p className="text-[#73726C]"><strong className="text-sky-400">Pros:</strong> {rec.pros[0]}</p>
                    {rec.cons[0] && (
                      <p className="text-[#9E9D96]"><strong className="text-[#73726C]">Tradeoff:</strong> {rec.cons[0]}</p>
                    )}
                  </div>

                  {/* Action Button */}
                  <button
                    onClick={() => onSelectModel(rec.modelId, rec.suggestedHyperparams)}
                    className={`w-full py-2 px-3 rounded font-mono font-bold text-xs flex items-center justify-center space-x-2 transition cursor-pointer ${
                      isSelected
                        ? "bg-sky-600/20 text-sky-400 border border-sky-500/40 cursor-default"
                        : "bg-[#F0EEE6] hover:bg-[#D5D1C5] text-[#1F1E1D] border border-[#D5D1C5] hover:border-neutral-600"
                    }`}
                  >
                    {isSelected ? (
                      <>
                        <Check className="h-3.5 w-3.5 text-sky-400" />
                        <span>Active Model Selected</span>
                      </>
                    ) : (
                      <>
                        <ArrowUpRight className="h-3.5 w-3.5" />
                        <span>Select Algorithm & Configure</span>
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ================= VIEW 2: AUTOMATED MODEL TOURNAMENT (BENCHMARK) ================= */}
      {activeSubTab === "benchmark" && (
        <div className="space-y-5">
          {/* Tournament Control Bar */}
          <div className="bg-white p-4 rounded border border-[#E5E2D9] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase">
                Cross-Algorithm Tournament
              </h4>
              <p className="text-[11px] text-[#73726C]">
                Trains all 4 candidate algorithms on the same train/test split and ranks performance on {taskType === "Classification" ? "Accuracy" : "R² Score"}.
              </p>
            </div>
            <button
              onClick={handleRunBenchmark}
              disabled={isBenchmarking}
              className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-xs rounded transition flex items-center space-x-2 cursor-pointer disabled:opacity-50 self-start sm:self-auto"
            >
              {isBenchmarking ? (
                <>
                  <RotateCw className="h-3.5 w-3.5 animate-spin" />
                  <span>Evaluating ({benchmarkProgress.step}/{benchmarkProgress.total})...</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" />
                  <span>Run Model Tournament</span>
                </>
              )}
            </button>
          </div>

          {benchmarkError && (
            <div className="p-3 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded flex items-center space-x-2 text-xs font-mono">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{benchmarkError}</span>
            </div>
          )}

          {/* Active Tournament Progress Animation */}
          {isBenchmarking && (
            <div className="p-4 bg-[#FAF9F5] rounded border border-[#E5E2D9] space-y-2">
              <div className="flex justify-between text-xs font-mono text-[#73726C]">
                <span>Fitting: <strong className="text-sky-400">{benchmarkProgress.currentModel}</strong></span>
                <span>{benchmarkProgress.step} of {benchmarkProgress.total}</span>
              </div>
              <div className="w-full bg-white h-2 rounded overflow-hidden">
                <div 
                  className="bg-sky-500 h-full transition-all duration-300"
                  style={{ width: `${(benchmarkProgress.step / benchmarkProgress.total) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Benchmark Results Leaderboard */}
          {benchmarkResults && !isBenchmarking && (
            <div className="space-y-5">
              {/* Leaderboard Table */}
              <div className="overflow-x-auto border border-[#E5E2D9] rounded bg-[#FAF9F5]">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="bg-[#FAF9F5] text-[#73726C] text-[10px] uppercase border-b border-[#E5E2D9]">
                    <tr>
                      <th className="py-2.5 px-3">Rank</th>
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3 text-right">Val {benchmarkResults[0]?.primaryMetricName}</th>
                      <th className="py-2.5 px-3 text-right">Train Metric</th>
                      <th className="py-2.5 px-3 text-right">Fit Latency</th>
                      <th className="py-2.5 px-3 text-center">Interpretability</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800/80">
                    {benchmarkResults.map((b, idx) => {
                      const isChampion = b.isChampion;
                      const isSelected = currentModelId === b.modelId;
                      return (
                        <tr key={b.modelId} className={isChampion ? "bg-sky-50" : "hover:bg-[#FAF9F5]"}>
                          <td className="py-2.5 px-3 whitespace-nowrap">
                            {isChampion ? (
                              <span className="flex items-center space-x-1 text-sky-400 font-bold">
                                <Trophy className="h-3.5 w-3.5 text-sky-400" />
                                <span>#1 Champion</span>
                              </span>
                            ) : (
                              <span className="text-[#9E9D96]">#{idx + 1}</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 whitespace-nowrap font-bold text-[#1F1E1D]">
                            {b.name}
                          </td>
                          <td className="py-2.5 px-3 text-right font-bold text-sky-400">
                            {taskType === "Classification" 
                              ? `${(b.valMetric * 100).toFixed(1)}%` 
                              : b.valMetric.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-[#73726C]">
                            {taskType === "Classification" 
                              ? `${(b.trainMetric * 100).toFixed(1)}%` 
                              : b.trainMetric.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-[#73726C]">
                            {b.fitTimeMs} ms
                          </td>
                          <td className="py-2.5 px-3 text-center">
                            <span className={`px-2 py-0.5 rounded text-[10px] ${
                              b.interpretability === "High" ? "bg-blue-500/10 text-blue-400" : "bg-[#F0EEE6] text-[#73726C]"
                            }`}>
                              {b.interpretability}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => onSelectModel(b.modelId)}
                              className={`px-2.5 py-1 text-[11px] font-mono rounded transition cursor-pointer ${
                                isSelected
                                  ? "bg-sky-500/20 text-sky-400 border border-sky-500/30"
                                  : "bg-[#F0EEE6] hover:bg-[#D5D1C5] text-[#1F1E1D]"
                              }`}
                            >
                              {isSelected ? "Selected" : "Select"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Tournament Visual Comparison Chart */}
              <div className="bg-[#FAF9F5] p-4 rounded border border-[#E5E2D9] space-y-3">
                <h5 className="text-xs font-mono font-bold text-[#73726C] uppercase">
                  Validation Score Comparison
                </h5>
                <div className="h-44 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={benchmarkResults} layout="vertical" margin={{ left: 20, right: 30, top: 10, bottom: 10 }}>
                      <XAxis type="number" stroke="#525252" fontSize={10} domain={[0, taskType === "Classification" ? 1 : 'auto']} />
                      <YAxis type="category" dataKey="name" stroke="#a3a3a3" fontSize={10} width={130} />
                      <Tooltip contentStyle={{ background: "#171717", border: "1px solid #262626", borderRadius: "4px", fontSize: "11px" }} />
                      <Bar dataKey="valMetric" name={`Validation ${benchmarkResults[0]?.primaryMetricName}`} radius={[0, 4, 4, 0]}>
                        {benchmarkResults.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.isChampion ? "#10b981" : "#3b82f6"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ================= VIEW 3: HYPERPARAMETER OPTIMIZATION (HPO) ================= */}
      {activeSubTab === "hpo" && (
        <div className="space-y-5">
          {/* Optimization Controls Bar */}
          <div className="bg-white p-4 rounded border border-[#E5E2D9] grid md:grid-cols-4 gap-4">
            {/* Target Model */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono font-bold text-[#73726C] uppercase">Model To Tune</label>
              <select
                value={hpoModelId}
                onChange={(e) => setHpoModelId(e.target.value)}
                className="w-full bg-white border border-[#E5E2D9] text-[#1F1E1D] font-mono text-xs rounded px-2.5 py-1.5 outline-none focus:ring-1 focus:ring-sky-500 cursor-pointer"
              >
                {taskType === "Classification" ? (
                  <>
                    <option value="decision_tree">Decision Tree Classifier</option>
                    <option value="random_forest">Random Forest Classifier</option>
                    <option value="regularized_logistic">Ridge Logistic (L2)</option>
                    <option value="logistic">Logistic Regression (OLS)</option>
                  </>
                ) : (
                  <>
                    <option value="decision_tree">Decision Tree Regressor</option>
                    <option value="random_forest">Random Forest Regressor</option>
                    <option value="ridge">Ridge Regression (L2)</option>
                    <option value="linear">Linear Regression (OLS)</option>
                  </>
                )}
              </select>
            </div>

            {/* Search Strategy */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono font-bold text-[#73726C] uppercase">Search Strategy</label>
              <select
                value={hpoStrategy}
                onChange={(e) => setHpoStrategy(e.target.value as any)}
                className="w-full bg-white border border-[#E5E2D9] text-[#1F1E1D] font-mono text-xs rounded px-2.5 py-1.5 outline-none focus:ring-1 focus:ring-sky-500 cursor-pointer"
              >
                <option value="random">Randomized Search (Diverse)</option>
                <option value="grid">Systematic Grid Search</option>
              </select>
            </div>

            {/* Trial Budget */}
            <div className="space-y-1">
              <label className="text-[10px] font-mono font-bold text-[#73726C] uppercase">Trial Budget</label>
              <select
                value={hpoNumTrials}
                onChange={(e) => setHpoNumTrials(Number(e.target.value))}
                className="w-full bg-white border border-[#E5E2D9] text-[#1F1E1D] font-mono text-xs rounded px-2.5 py-1.5 outline-none focus:ring-1 focus:ring-sky-500 cursor-pointer"
              >
                <option value="6">Quick Tune (6 trials)</option>
                <option value="10">Standard Tune (10 trials)</option>
                <option value="16">Exhaustive Tune (16 trials)</option>
              </select>
            </div>

            {/* Run Button */}
            <div className="flex items-end">
              <button
                onClick={handleRunHPO}
                disabled={isTuning}
                className="w-full py-1.5 px-3 bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-xs rounded transition flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50"
              >
                {isTuning ? (
                  <>
                    <RotateCw className="h-3.5 w-3.5 animate-spin" />
                    <span>Trial {liveTrials.length + 1}...</span>
                  </>
                ) : (
                  <>
                    <Sliders className="h-3.5 w-3.5" />
                    <span>Optimize Hyperparameters</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {hpoError && (
            <div className="p-3 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded flex items-center space-x-2 text-xs font-mono">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{hpoError}</span>
            </div>
          )}

          {/* Real-time Progress Bar */}
          {isTuning && (
            <div className="p-4 bg-[#FAF9F5] rounded border border-[#E5E2D9] space-y-2">
              <div className="flex justify-between text-xs font-mono text-[#73726C]">
                <span>Evaluating Trial {liveTrials.length} of {hpoNumTrials}</span>
                <span className="text-sky-400 font-bold">{hpoProgressPct}%</span>
              </div>
              <div className="w-full bg-white h-2 rounded overflow-hidden">
                <div 
                  className="bg-sky-500 h-full transition-all duration-200"
                  style={{ width: `${hpoProgressPct}%` }}
                />
              </div>
            </div>
          )}

          {/* Optimization Results Header & Performance Lift */}
          {hpoResult && !isTuning && (
            <div className="bg-sky-50 border border-sky-500/30 rounded p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-sky-400" />
                  <h4 className="font-mono font-bold text-[#1F1E1D] text-sm">
                    Optimization Complete: {hpoResult.modelName}
                  </h4>
                </div>
                <p className="text-xs text-[#73726C] font-mono">
                  Baseline {hpoResult.targetMetric}: <strong className="text-[#73726C]">{hpoResult.baselineScore}</strong> → 
                  Tuned {hpoResult.targetMetric}: <strong className="text-sky-400">{hpoResult.bestScore}</strong>
                  {" "}
                  {hpoResult.improvementPct >= 0 && (
                    <span className="text-sky-400 font-bold">
                      (+{hpoResult.improvementPct}% lift)
                    </span>
                  )}
                </p>
              </div>

              <button
                onClick={handleApplyBestParams}
                className={`px-4 py-2 font-mono font-bold text-xs rounded transition flex items-center space-x-2 cursor-pointer ${
                  paramsApplied
                    ? "bg-sky-600 text-white shadow"
                    : "bg-sky-500 hover:bg-sky-400 text-white font-bold"
                }`}
              >
                {paramsApplied ? (
                  <>
                    <Check className="h-4 w-4" />
                    <span>Applied to Pipeline!</span>
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    <span>Apply Best Hyperparameters</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Optimization History Chart & Trials Log */}
          {liveTrials.length > 0 && (
            <div className="grid lg:grid-cols-2 gap-5">
              {/* Convergence Line Chart */}
              <div className="bg-[#FAF9F5] p-4 rounded border border-[#E5E2D9] space-y-2">
                <h5 className="text-xs font-mono font-bold text-[#73726C] uppercase">
                  Hyperparameter Optimization Sensitivity
                </h5>
                <p className="text-[10px] text-[#9E9D96] font-mono">
                  Validation {taskType === "Classification" ? "Accuracy" : "R² Score"} evolution across parameter trials.
                </p>
                <div className="h-48 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={liveTrials}>
                      <XAxis dataKey="trial" stroke="#525252" fontSize={10} />
                      <YAxis stroke="#525252" fontSize={10} domain={['auto', 'auto']} />
                      <Tooltip contentStyle={{ background: "#171717", border: "1px solid #262626", borderRadius: "4px", fontSize: "11px" }} />
                      <Line type="monotone" dataKey="valScore" name="Validation Score" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="trainScore" name="Train Score" stroke="#3b82f6" strokeWidth={1} strokeDasharray="3 3" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Best Parameters Found Card */}
              <div className="bg-[#FAF9F5] p-4 rounded border border-[#E5E2D9] space-y-3">
                <h5 className="text-xs font-mono font-bold text-[#73726C] uppercase">
                  Optimal Hyperparameter Configuration
                </h5>
                {hpoResult ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                      {Object.entries(hpoResult.bestParams).map(([k, v]) => (
                        <div key={k} className="bg-[#FAF9F5] p-2.5 rounded border border-[#E5E2D9]">
                          <p className="text-[10px] text-[#9E9D96] uppercase">{k}</p>
                          <p className="text-sky-400 font-bold">{String(v)}</p>
                        </div>
                      ))}
                    </div>

                    <div className="pt-2 text-[11px] font-mono text-[#73726C] bg-[#FAF9F5] p-2.5 rounded border border-[#E5E2D9] space-y-1">
                      <p className="flex items-center gap-1.5">
                        <Check className="h-3 w-3 text-sky-600 shrink-0" />
                        <span>Evaluated <strong className="text-[#1F1E1D]">{hpoResult.trials.length}</strong> configurations using <strong className="text-[#1F1E1D]">{hpoResult.strategy} search</strong>.</span>
                      </p>
                      <p className="flex items-center gap-1.5">
                        <Check className="h-3 w-3 text-sky-600 shrink-0" />
                        <span>Found peak validation score: <strong className="text-sky-600">{hpoResult.bestScore}</strong>.</span>
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-[#9E9D96] font-mono py-8 text-center">
                    Run optimization to compute the highest-scoring hyperparameter set.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Trials Detailed Table */}
          {liveTrials.length > 0 && (
            <div className="border border-[#E5E2D9] rounded bg-[#FAF9F5] overflow-hidden">
              <div className="p-2.5 bg-[#FAF9F5] border-b border-[#E5E2D9] font-mono text-xs font-bold text-[#73726C] flex justify-between items-center">
                <span>Optimization Trials Record</span>
                <span className="text-[11px] text-[#9E9D96] font-normal">{liveTrials.length} trials completed</span>
              </div>
              <div className="max-h-52 overflow-y-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="bg-[#FAF9F5] text-[#9E9D96] text-[10px] uppercase border-b border-[#E5E2D9]">
                    <tr>
                      <th className="py-2 px-3">Trial</th>
                      <th className="py-2 px-3">Tested Hyperparameters</th>
                      <th className="py-2 px-3 text-right">Validation Metric</th>
                      <th className="py-2 px-3 text-right">Train Metric</th>
                      <th className="py-2 px-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-850">
                    {liveTrials.map(tr => (
                      <tr key={tr.trial} className={tr.isBest ? "bg-sky-50" : "hover:bg-[#FAF9F5]"}>
                        <td className="py-2 px-3 text-[#73726C]">#{tr.trial}</td>
                        <td className="py-2 px-3 text-[#73726C]">
                          {Object.entries(tr.params).map(([k, v]) => `${k}=${v}`).join(", ")}
                        </td>
                        <td className="py-2 px-3 text-right font-bold text-sky-400">
                          {taskType === "Classification" ? `${(tr.valScore * 100).toFixed(1)}%` : tr.valScore.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-right text-[#73726C]">
                          {taskType === "Classification" ? `${(tr.trainScore * 100).toFixed(1)}%` : tr.trainScore.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-center">
                          {tr.isBest ? (
                            <span className="text-[10px] bg-sky-500/20 text-sky-400 border border-sky-500/30 px-1.5 py-0.5 rounded font-bold">
                              Best
                            </span>
                          ) : (
                            <span className="text-[10px] text-[#73726C]">-</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
