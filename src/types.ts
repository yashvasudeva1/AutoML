export type ColumnType = "Numeric" | "Categorical" | "Date" | "Text";

export type SensitivityCategory = 
  | "Email"
  | "Phone"
  | "Name"
  | "Financial"
  | "SSN / Tax ID"
  | "IP Address"
  | "Address / Location"
  | "Credentials / Secret"
  | "Personal ID";

export type SensitivityLevel = "High" | "Medium" | "Low";

export interface ColumnSensitivity {
  isSensitive: boolean;
  category?: SensitivityCategory;
  level?: SensitivityLevel;
  confidence: number;
  reason: string;
}

export interface ColumnStats {
  mean?: number;
  median?: number;
  min?: number;
  max?: number;
  stdDev?: number;
  variance?: number;
  kurtosis?: number;
  q25?: number;
  q75?: number;
  range?: number;
  nonNullCount: number;
  uniquePct?: number;
  missingCount: number;
  uniqueCount: number;
  frequencies?: { [value: string]: number };
  skewness?: number;
}

export interface ColumnInfo {
  name: string;
  type: ColumnType;
  stats: ColumnStats;
  sensitivity?: ColumnSensitivity;
}

export interface StatisticalTest {
  type: "T-Test" | "Chi-Square" | "Pearson Correlation" | "ANOVA";
  variable1: string;
  variable2: string;
  statisticValue: number;
  pValue: number;
  degreesOfFreedom?: number;
  interpretation: string;
}

export interface OutlierResult {
  columnName: string;
  outlierCount: number;
  outlierPercentage: number;
  lowerBound: number;
  upperBound: number;
  isSignificant: boolean;
  method: string;
  extremeValues: number[];
  indices: number[];
}

export interface ChunkInfo {
  chunkIndex: number;
  totalChunks: number;
  startRow: number;
  endRow: number;
  rowCount: number;
  qualityScore: number;
  nullCount: number;
  processingTimeMs: number;
  sampleRows: any[];
}

export interface BatchProcessingReport {
  isBatched: boolean;
  chunkSize: number;
  totalChunks: number;
  totalRows: number;
  chunks: ChunkInfo[];
  processingDurationMs: number;
  estimatedTokensSaved: number;
  tokenReductionPercent: number;
  peakMemoryMB: number;
}

export interface DatasetSummary {
  fileName: string;
  rowCount: number;
  colCount: number;
  columns: ColumnInfo[];
  correlation: { [key: string]: { [key: string]: number } };
  statisticalTests: StatisticalTest[];
  sampleData: Array<{ [key: string]: any }>;
  dataQualityScore?: number;
  outliers?: OutlierResult[];
  batchReport?: BatchProcessingReport;
}

export interface SupportedHfModel {
  id: string;
  name: string;
  provider: string;
  contextWindow: string;
  description: string;
  speed: string;
  license: string;
}

export interface AIModelConfig {
  activeProvider: "gemini" | "huggingface";
  geminiModel: string;
  hfModel: string;
  hasHfApiKey: boolean;
  hasGeminiApiKey: boolean;
  activeModelName: string;
  activeProviderName: string;
  supportedHfModels: SupportedHfModel[];
  stats?: {
    geminiRequests: number;
    hfRequests: number;
    hfTokensUsed: number;
  };
}

export interface AIAnalysisResult {
  columnInsights: Array<{ colName: string; insight: string }>;
  overallConclusions: Array<{ title: string; description: string }>;
  modelRecommendations: {
    recommendedTask: string;
    recommendedTarget: string;
    rationale: string;
    models: Array<{ name: string; reason: string }>;
  };
  executedBy?: {
    provider: "gemini" | "huggingface";
    model: string;
    modelName: string;
  };
}

export interface MLTrainingResult {
  trainMetrics: Record<string, number>;
  testMetrics: Record<string, number>;
  lossHistory?: Array<{ epoch: number; trainLoss: number; valLoss: number }>;
  predictions?: Array<{ actual: any; predicted: any }>;
  featureImportances?: Array<{ feature: string; importance: number; weight?: number }>;
  confusionMatrix?: {
    matrix: number[][];
    labels: string[];
    perClass?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  };
  residualAnalysis?: {
    meanResidual: number;
    maxResidual: number;
    stdResidual: number;
    residuals: Array<{ actual: number; predicted: number; residual: number }>;
  };
  crossValidation?: {
    meanScore: number;
    stdScore: number;
    scores?: number[];
    foldScores?: number[];
    folds?: number;
    metricName?: string;
  };
  modelArtifact?: any;
}

export type ModelTrainingResult = MLTrainingResult;

export interface TrainedModelExportInfo {
  modelId?: string;
  modelName: string;
  taskType: "Classification" | "Regression";
  targetCol?: string;
  targetColumn?: string;
  features: string[];
  hyperparameters: Record<string, any>;
  results: MLTrainingResult;
  trainedAt?: string;
  timestamp?: string;
}

export interface PolarsCleaningStep {
  id: string;
  step?: string;
  type: string;
  column?: string;
  columns?: string[];
  params?: Record<string, any>;
  description?: string;
  polarsExpression?: string;
  isActive: boolean;
  timestamp?: string;
  [key: string]: any;
}

export interface PolarsCleaningReport {
  initialRows: number;
  cleanedRows: number;
  initialColumns: number;
  cleanedColumns: number;
  nullsRemoved: number;
  outliersClipped: number;
  duplicatesRemoved: number;
  initialQualityScore: number;
  finalQualityScore: number;
  executionTimeMs: number;
  estimatedThroughputRowsPerSec: number;
  estimatedPolarsSpeedupVsPandas: number | string;
  [key: string]: any;
}

export interface ModelRecommendationItem {
  modelId: string;
  name: string;
  suitabilityScore: number;
  recommendationLevel: string;
  badgeColor: string;
  edaSignals: string[];
  pros: string[];
  cons: string[];
  suggestedHyperparams: Record<string, any>;
}

export interface BenchmarkResult {
  modelId: string;
  name: string;
  primaryMetricName: string;
  valMetric: number;
  trainMetric: number;
  fitTimeMs: number;
  interpretability: "High" | "Medium" | "Low";
  isChampion: boolean;
  metrics: Record<string, number>;
}

export interface HPOTrialResult {
  trial: number;
  params: Record<string, any>;
  trainScore: number;
  valScore: number;
  isBest: boolean;
}

export interface HPOOptimizationResult {
  modelId: string;
  modelName: string;
  targetMetric: string;
  baselineScore: number;
  bestScore: number;
  improvementPct: number;
  bestParams: Record<string, any>;
  trials: HPOTrialResult[];
  strategy: string;
}
