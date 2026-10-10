import React, { useEffect, useState } from "react";
import { 
  Sparkles, X, Trophy, Zap, ShieldCheck, Check, Download, 
  ArrowRight, Activity, Cpu, RotateCw, AlertTriangle, Layers, HelpCircle
} from "lucide-react";
import { ComparedModelRow } from "./MlTab";

interface ModelExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  model: ComparedModelRow | null;
  taskType: "Classification" | "Regression";
  targetCol: string;
  fileName: string;
  features: string[];
  onSelectModel?: (model: ComparedModelRow) => void;
  onDownloadJson?: (model: ComparedModelRow) => void;
}

interface FeatureExplanationItem {
  feature: string;
  importance?: number;
  rank?: number;
  impact?: string;
  explanation: string;
}

interface ModelExplanationData {
  modelSummary: string;
  featureImportanceAnalysis: FeatureExplanationItem[];
  decisionLogic: {
    architectureExplanation: string;
    decisionRules: string[];
    strengthsAndTradeoffs: string;
    businessRecommendation: string;
  };
  telemetry?: {
    source: string;
    tokensUsed: number;
    costUsd?: number;
  };
}

export default function ModelExplanationModal({
  isOpen,
  onClose,
  model,
  taskType,
  targetCol,
  fileName,
  features,
  onSelectModel,
  onDownloadJson,
}: ModelExplanationModalProps) {
  const [data, setData] = useState<ModelExplanationData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !model) {
      setData(null);
      setError(null);
      return;
    }

    let isMounted = true;

    async function fetchExplanation() {
      setIsLoading(true);
      setError(null);

      try {
        const payload = {
          modelId: model.id,
          modelName: model.name,
          category: model.category,
          taskType,
          targetColumn: targetCol,
          features,
          metrics: {
            primaryMetricName: model.primaryMetricName,
            primaryMetricValue: model.primaryMetricVal,
            secondaryMetricName: model.secondaryMetricName,
            secondaryMetricValue: model.secondaryMetricVal,
            trainMetricValue: model.trainMetricVal,
            overfitGap: model.overfitGap,
            durationMs: model.durationMs,
            rank: model.rank,
            isBest: model.isBest,
            isFastest: model.isFastest,
          },
          featureImportances: model.results.featureImportances || [],
          decisionLogicSummary: model.description,
          fileName,
        };

        const res = await fetch("/api/ml/explain-model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`Failed to generate explanation (status ${res.status})`);
        }

        const json = await res.json();
        if (isMounted) {
          setData(json);
        }
      } catch (err: any) {
        console.error("Model explanation error:", err);
        if (isMounted) {
          setError(err.message || "Failed to generate model explanation.");
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    fetchExplanation();

    return () => {
      isMounted = false;
    };
  }, [isOpen, model, taskType, targetCol, fileName, features]);

  if (!isOpen || !model) return null;

  const scoreFormatted = taskType === "Classification"
    ? `${(model.primaryMetricVal * 100).toFixed(1)}%`
    : model.primaryMetricVal.toFixed(4);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-3 sm:p-5 overflow-y-auto">
      <div 
        className="bg-white rounded-lg border border-[#E5E2D9] max-w-3xl w-full shadow-2xl overflow-hidden my-auto flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-explanation-title"
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#E5E2D9] bg-[#FAF9F5] flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded bg-purple-500/10 border border-purple-500/20 text-purple-700 flex items-center justify-center">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 id="model-explanation-title" className="font-mono font-bold text-xs uppercase tracking-wider text-[#1F1E1D]">
                  Model Explanation & Decision Logic
                </h3>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-100 text-purple-800 border border-purple-200 flex items-center space-x-1">
                  <Sparkles className="h-2.5 w-2.5" />
                  <span>{data?.telemetry?.source || "AI Reasoning Engine"}</span>
                </span>
              </div>
              <p className="text-[11px] font-mono text-[#73726C]">
                {model.name} • {model.category} • Target: '{targetCol}'
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#F0EEE6] transition-colors cursor-pointer"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto space-y-5 text-xs font-mono">
          {/* Objective Performance Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-[#FAF9F5] p-3 rounded border border-[#E5E2D9]">
            <div className="space-y-0.5">
              <span className="text-[9px] uppercase text-[#73726C] font-bold">Accuracy/R-squared</span>
              <div className="flex items-center space-x-1.5 font-bold text-sm text-[#1F1E1D]">
                <span>{scoreFormatted}</span>
                {model.isBest && (
                  <span className="text-[9px] px-1 py-0.2 bg-emerald-100 text-emerald-800 rounded font-bold">
                    Best (#1)
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[9px] uppercase text-[#73726C] font-bold">
                {taskType === "Classification" ? "F1-Score" : "RMSE"}
              </span>
              <div className="font-bold text-sm text-[#1F1E1D]">
                {model.secondaryMetricVal.toFixed(4)}
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[9px] uppercase text-[#73726C] font-bold">Train / Fit Gap</span>
              <div className={`font-bold text-sm ${Math.abs(model.overfitGap) > 0.15 ? "text-amber-600" : "text-[#1F1E1D]"}`}>
                {model.overfitGap >= 0 ? `+${model.overfitGap}` : model.overfitGap}
              </div>
            </div>

            <div className="space-y-0.5">
              <span className="text-[9px] uppercase text-[#73726C] font-bold">Training Latency</span>
              <div className="font-bold text-sm text-[#1F1E1D] flex items-center space-x-1">
                <span>{model.durationMs} ms</span>
                {model.isFastest && <Zap className="h-3 w-3 text-sky-500" />}
              </div>
            </div>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="py-12 flex flex-col items-center justify-center space-y-3 bg-[#FAF9F5] rounded border border-[#E5E2D9]">
              <RotateCw className="h-6 w-6 text-purple-600 animate-spin" />
              <div className="text-center space-y-1">
                <p className="font-bold text-xs text-[#1F1E1D]">
                  Synthesizing Model Intelligence & Decision Logic...
                </p>
                <p className="text-[11px] text-[#73726C]">
                  Analyzing decision boundary splits, feature weight contributions, and bias-variance profile
                </p>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && !isLoading && (
            <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded space-y-2 text-rose-700">
              <div className="flex items-center space-x-2 font-bold">
                <AlertTriangle className="h-4 w-4" />
                <span>Explanation Generation Error</span>
              </div>
              <p className="text-[11px]">{error}</p>
            </div>
          )}

          {/* Loaded Explanation Content */}
          {data && !isLoading && (
            <div className="space-y-5">
              {/* 1. Executive Summary */}
              <div className="bg-purple-50/60 border border-purple-200/80 rounded p-4 space-y-1.5">
                <div className="flex items-center space-x-1.5 text-purple-800 font-bold text-[11px] uppercase tracking-wider">
                  <Sparkles className="h-3.5 w-3.5 text-purple-600" />
                  <span>Executive Model Summary</span>
                </div>
                <p className="text-xs text-[#1F1E1D] leading-relaxed">
                  {data.modelSummary}
                </p>
              </div>

              {/* 2. Feature Importance & Influence Analysis */}
              <div className="bg-white rounded border border-[#E5E2D9] p-4 space-y-3">
                <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2">
                  <div className="flex items-center space-x-1.5">
                    <Activity className="h-3.5 w-3.5 text-sky-600" />
                    <h4 className="font-bold text-xs uppercase tracking-wider text-[#1F1E1D]">
                      Feature Importance & Impact Analysis
                    </h4>
                  </div>
                  <span className="text-[10px] text-[#73726C]">
                    Ranked by relative predictive weight
                  </span>
                </div>

                <div className="space-y-2.5">
                  {(data.featureImportanceAnalysis || []).map((item, idx) => {
                    const rawImp = model.results.featureImportances?.find(f => f.feature === item.feature)?.importance ?? (1 - idx * 0.2);
                    const pct = Math.max(5, Math.min(100, Math.round(rawImp * 100)));

                    return (
                      <div key={item.feature || idx} className="p-2.5 rounded bg-[#FAF9F5] border border-[#E5E2D9] space-y-1.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="inline-flex items-center justify-center w-4 h-4 rounded text-[9px] font-bold bg-[#E5E2D9] text-[#1F1E1D]">
                              #{idx + 1}
                            </span>
                            <span className="font-bold text-[#1F1E1D] text-xs">
                              {item.feature}
                            </span>
                            {item.impact && (
                              <span className="text-[9px] px-1.5 py-0.2 rounded bg-purple-100 text-purple-800 border border-purple-200 font-semibold">
                                {item.impact}
                              </span>
                            )}
                          </div>
                          <span className="text-[10px] font-bold text-[#73726C]">
                            {pct}% influence
                          </span>
                        </div>

                        {/* Visual Progress Bar */}
                        <div className="w-full h-1.5 bg-[#E5E2D9] rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${idx === 0 ? "bg-purple-600" : idx === 1 ? "bg-sky-500" : "bg-neutral-500"}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>

                        <p className="text-[11px] text-[#73726C] leading-relaxed pt-0.5">
                          {item.explanation}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 3. Decision Logic & Internal Reasoning */}
              <div className="bg-white rounded border border-[#E5E2D9] p-4 space-y-3">
                <div className="flex items-center space-x-1.5 border-b border-[#E5E2D9] pb-2">
                  <Layers className="h-3.5 w-3.5 text-purple-600" />
                  <h4 className="font-bold text-xs uppercase tracking-wider text-[#1F1E1D]">
                    Decision Logic & Underlying Architecture Mechanics
                  </h4>
                </div>

                <div className="space-y-3">
                  <p className="text-xs text-[#73726C] leading-relaxed">
                    {data.decisionLogic.architectureExplanation}
                  </p>

                  {/* Decision Rules / Heuristics */}
                  {data.decisionLogic.decisionRules && data.decisionLogic.decisionRules.length > 0 && (
                    <div className="space-y-1.5 pt-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-[#1F1E1D]">
                        Extracted Decision Heuristics:
                      </span>
                      <div className="space-y-1">
                        {data.decisionLogic.decisionRules.map((rule, idx) => (
                          <div key={idx} className="flex items-start space-x-2 text-[11px] text-[#73726C] bg-[#FAF9F5] p-2 rounded border border-[#E5E2D9]">
                            <span className="w-4 h-4 rounded-full bg-purple-100 text-purple-800 text-[10px] font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                              {idx + 1}
                            </span>
                            <span className="leading-relaxed">{rule}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* 4. Strengths, Caveats & Production Recommendation */}
              <div className="grid sm:grid-cols-2 gap-3">
                <div className="bg-[#FAF9F5] rounded border border-[#E5E2D9] p-3.5 space-y-1.5">
                  <span className="text-[10px] font-bold uppercase text-[#1F1E1D] flex items-center space-x-1">
                    <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                    <span>Operational Profile & Caveats</span>
                  </span>
                  <p className="text-[11px] text-[#73726C] leading-relaxed">
                    {data.decisionLogic.strengthsAndTradeoffs}
                  </p>
                </div>

                <div className="bg-emerald-50/60 rounded border border-emerald-200 p-3.5 space-y-1.5">
                  <span className="text-[10px] font-bold uppercase text-emerald-900 flex items-center space-x-1">
                    <Trophy className="h-3.5 w-3.5 text-emerald-700" />
                    <span>Deployment Recommendation</span>
                  </span>
                  <p className="text-[11px] text-emerald-950 leading-relaxed">
                    {data.decisionLogic.businessRecommendation}
                  </p>
                </div>
              </div>

              {/* Telemetry Footer info */}
              {data.telemetry && (
                <div className="flex items-center justify-between text-[10px] text-[#9E9D96] pt-1">
                  <span>Provider: {data.telemetry.source}</span>
                  {data.telemetry.tokensUsed > 0 && (
                    <span>Tokens: {data.telemetry.tokensUsed.toLocaleString()}</span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-[#E5E2D9] bg-[#FAF9F5] flex flex-wrap items-center justify-between gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 bg-white border border-[#E5E2D9] hover:bg-[#F0EEE6] text-[#1F1E1D] text-xs font-mono font-bold rounded transition-colors cursor-pointer"
          >
            Close
          </button>

          <div className="flex items-center space-x-2">
            {onDownloadJson && (
              <button
                type="button"
                onClick={() => onDownloadJson(model)}
                className="px-3 py-1.5 bg-white hover:bg-purple-50 text-purple-700 border border-purple-200 hover:border-purple-300 rounded text-xs font-mono font-bold transition-all cursor-pointer flex items-center space-x-1.5 shadow-2xs"
              >
                <Download className="h-3 w-3" />
                <span>Download Model (.json)</span>
              </button>
            )}

            {onSelectModel && (
              <button
                type="button"
                onClick={() => {
                  onSelectModel(model);
                  onClose();
                }}
                className="px-3.5 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded text-xs font-mono font-bold transition-all cursor-pointer flex items-center space-x-1.5 shadow-2xs"
              >
                <Check className="h-3.5 w-3.5" />
                <span>Select & Load This Model</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
