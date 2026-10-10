import React, { useState } from "react";
import { 
  X, Database, Sliders, Shield, 
  Trash2, LogOut, Check, Zap, Cpu, Type
} from "lucide-react";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  userEmail: string | null;
  onLogout: () => void;
  onClearWorkspace: () => void;
}

export default function SettingsModal({
  isOpen,
  onClose,
  userEmail,
  onLogout,
  onClearWorkspace
}: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<"general" | "engine" | "automl" | "account">("general");
  const [cvFolds, setCvFolds] = useState<number>(5);
  const [testSplit, setTestSplit] = useState<number>(20);
  const [iqrThreshold, setIqrThreshold] = useState<number>(1.5);
  const [saveFeedback, setSaveFeedback] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSave = () => {
    setSaveFeedback("Preferences updated");
    setTimeout(() => {
      setSaveFeedback(null);
      onClose();
    }, 600);
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/35 backdrop-blur-xs transition-opacity"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-xl bg-white border border-[#E5E2D9] rounded-2xl shadow-xl overflow-hidden text-[#1F1E1D] flex flex-col max-h-[85vh] animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#E5E2D9]">
          <h3 className="font-serif text-lg font-normal text-[#1F1E1D]">
            Settings & Preferences
          </h3>
          <button
            onClick={onClose}
            className="p-1.5 text-[#73726C] hover:text-[#1F1E1D] rounded-lg hover:bg-[#F0EEE6] transition-colors cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tab navigation */}
        <div className="flex border-b border-[#E5E2D9] px-6 bg-[#FAF9F5]">
          <button
            onClick={() => setActiveTab("general")}
            className={`py-2.5 px-3 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer ${
              activeTab === "general"
                ? "border-[#0284C7] text-[#0284C7]"
                : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            General & Typography
          </button>
          <button
            onClick={() => setActiveTab("engine")}
            className={`py-2.5 px-3 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer ${
              activeTab === "engine"
                ? "border-[#0284C7] text-[#0284C7]"
                : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            Polars Engine
          </button>
          <button
            onClick={() => setActiveTab("automl")}
            className={`py-2.5 px-3 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer ${
              activeTab === "automl"
                ? "border-[#0284C7] text-[#0284C7]"
                : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            AutoML Defaults
          </button>
          <button
            onClick={() => setActiveTab("account")}
            className={`py-2.5 px-3 text-xs font-medium border-b-2 -mb-px transition-colors cursor-pointer ${
              activeTab === "account"
                ? "border-[#0284C7] text-[#0284C7]"
                : "border-transparent text-[#73726C] hover:text-[#1F1E1D]"
            }`}
          >
            Account
          </button>
        </div>

        {/* Tab content body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {activeTab === "general" && (
            <div className="space-y-5">
              <div className="space-y-2">
                <label className="font-medium text-[#1F1E1D] block">
                  Appearance Theme
                </label>
                <div className="p-3 bg-[#F0F9FF] border border-[#BAE6FD] rounded-xl flex items-center justify-between">
                  <div className="space-y-0.5">
                    <p className="font-medium text-[#0284C7]">Clean Light Mode</p>
                    <p className="text-[11px] text-[#73726C]">
                      Default interface is set strictly to light mode with warm parchment backgrounds and light blue accents.
                    </p>
                  </div>
                  <span className="w-2 h-2 rounded-full bg-[#0284C7]" />
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <div className="flex items-center space-x-2">
                  <Type className="h-4 w-4 text-[#0284C7]" />
                  <label className="font-medium text-[#1F1E1D]">
                    Claude.ai Typography Standard
                  </label>
                </div>
                <p className="text-[#73726C] leading-relaxed">
                  The application exclusively employs the Claude.ai typography hierarchy:
                </p>
                <div className="p-3.5 bg-[#FAF9F5] rounded-xl border border-[#E5E2D9] space-y-2.5">
                  <div>
                    <span className="text-[10px] font-mono text-[#73726C] uppercase tracking-wider block">Headings & Titles (Serif)</span>
                    <p className="font-serif text-sm font-medium text-[#1F1E1D]">Copernicus / Tiempos Headline · Editorial Book Serif</p>
                  </div>
                  <div className="border-t border-[#E5E2D9] pt-2">
                    <span className="text-[10px] font-mono text-[#73726C] uppercase tracking-wider block">Body & Interfaces (Sans)</span>
                    <p className="font-sans text-xs text-[#1F1E1D]">Anthropic Sans / System UI Neutral</p>
                  </div>
                  <div className="border-t border-[#E5E2D9] pt-2">
                    <span className="text-[10px] font-mono text-[#73726C] uppercase tracking-wider block">Code & Tabular Numbers (Monospace)</span>
                    <p className="font-mono text-[11px] text-[#73726C]">Fira Code / SFMono · μ = 14.829, p &lt; 0.001</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "engine" && (
            <div className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-[#0284C7]" />
                  <label className="font-medium text-[#1F1E1D]">
                    Polars Vectorized Execution Engine
                  </label>
                </div>
                <p className="text-[#73726C] leading-relaxed">
                  Configures high-throughput out-of-core pipeline operations and memory streaming client-side.
                </p>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <label className="font-medium text-[#1F1E1D] block">
                  IQR Outlier Detection Multiplier
                </label>
                <p className="text-[#73726C]">
                  Multiplier applied to interquartile range (IQR) to flag statistical anomalies.
                </p>
                <div className="flex items-center space-x-3 pt-1">
                  {[1.5, 2.0, 3.0].map((val) => (
                    <button
                      key={val}
                      type="button"
                      onClick={() => setIqrThreshold(val)}
                      className={`px-3 py-1.5 rounded-lg border font-mono cursor-pointer transition-all ${
                        iqrThreshold === val
                          ? "border-[#0284C7] bg-[#F0F9FF] text-[#0284C7] font-semibold"
                          : "border-[#E5E2D9] text-[#73726C]"
                      }`}
                    >
                      {val.toFixed(1)}x {val === 1.5 ? "(Standard)" : val === 3.0 ? "(Extreme)" : ""}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <label className="font-medium text-[#1F1E1D] block">
                  Code Export Format
                </label>
                <p className="text-[#73726C]">
                  Generated pipelines use Python Polars lazyframes (<code className="font-mono bg-[#F0EEE6] px-1 py-0.5 rounded">pl.scan_csv()</code>) for memory efficiency.
                </p>
              </div>
            </div>
          )}

          {activeTab === "automl" && (
            <div className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Cpu className="h-4 w-4 text-[#0284C7]" />
                  <label className="font-medium text-[#1F1E1D]">
                    Cross-Validation & Train/Test Splits
                  </label>
                </div>
                <p className="text-[#73726C] leading-relaxed">
                  Default parameters used when benchmarking candidate estimators in AutoML Tournament.
                </p>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <label className="font-medium text-[#1F1E1D] block">
                  Default K-Fold Validation Folds
                </label>
                <div className="flex items-center space-x-3 pt-1">
                  {[3, 5, 10].map((k) => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setCvFolds(k)}
                      className={`px-3 py-1.5 rounded-lg border font-mono cursor-pointer transition-all ${
                        cvFolds === k
                          ? "border-[#0284C7] bg-[#F0F9FF] text-[#0284C7] font-semibold"
                          : "border-[#E5E2D9] text-[#73726C]"
                      }`}
                    >
                      {k}-Fold {k === 5 ? "(Recommended)" : ""}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <label className="font-medium text-[#1F1E1D] block">
                  Holdout Test Split Ratio
                </label>
                <div className="flex items-center space-x-3 pt-1">
                  {[15, 20, 30].map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setTestSplit(s)}
                      className={`px-3 py-1.5 rounded-lg border font-mono cursor-pointer transition-all ${
                        testSplit === s
                          ? "border-[#0284C7] bg-[#F0F9FF] text-[#0284C7] font-semibold"
                          : "border-[#E5E2D9] text-[#73726C]"
                      }`}
                    >
                      {100 - s}/{s}%
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "account" && (
            <div className="space-y-5">
              <div className="space-y-2">
                <label className="font-medium text-[#1F1E1D] block">
                  Current Active Session
                </label>
                <div className="p-3 bg-[#FAF9F5] rounded-xl border border-[#E5E2D9] flex items-center justify-between">
                  <div className="flex items-center space-x-2.5">
                    <div className="w-8 h-8 rounded-full bg-[#0284C7] text-white flex items-center justify-center font-serif text-xs font-bold">
                      {userEmail ? userEmail.charAt(0).toUpperCase() : "A"}
                    </div>
                    <div>
                      <p className="font-medium text-[#1F1E1D]">{userEmail || "Anonymous Analyst"}</p>
                      <p className="text-[11px] text-[#73726C]">Client Session Authenticated</p>
                    </div>
                  </div>
                  <button
                    onClick={onLogout}
                    className="px-3 py-1.5 text-xs text-rose-600 hover:bg-rose-50 rounded-lg transition-colors cursor-pointer flex items-center space-x-1.5"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-[#E5E2D9]">
                <label className="font-medium text-[#1F1E1D] block">
                  Workspace Cache & State
                </label>
                <p className="text-[#73726C] leading-relaxed">
                  Reset current dataset, clear cached AI insights, and wipe temporary engineered features.
                </p>
                <button
                  type="button"
                  onClick={() => {
                    onClearWorkspace();
                    onClose();
                  }}
                  className="px-3.5 py-2 border border-[#E5E2D9] hover:bg-[#F0EEE6] rounded-xl text-xs font-medium text-[#73726C] hover:text-[#1F1E1D] transition-colors cursor-pointer flex items-center space-x-2"
                >
                  <Trash2 className="h-3.5 w-3.5 text-rose-500" />
                  <span>Reset Current Workspace</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3.5 border-t border-[#E5E2D9] bg-[#FAF9F5]">
          <span className="text-[11px] text-[#73726C]">
            {saveFeedback ? <span className="text-sky-600 font-medium inline-flex items-center gap-1"><Check className="h-3 w-3" /> {saveFeedback}</span> : "AutoML Studio"}
          </span>
          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs text-[#73726C] hover:text-[#1F1E1D] cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 bg-[#0284C7] hover:bg-[#0369A1] text-white text-xs font-medium rounded-lg cursor-pointer transition-colors shadow-xs"
            >
              Save Preferences
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
