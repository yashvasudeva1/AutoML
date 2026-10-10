// Automated Model Selection and Hyperparameter Tuning Engine
import { 
  DatasetSummary, 
  ModelRecommendationItem, 
  BenchmarkResult, 
  HPOTrialResult, 
  HPOOptimizationResult 
} from "../types";
import { 
  trainLinearRegression, 
  trainLogisticRegression, 
  trainDecisionTree, 
  trainRandomForest,
  trainGradientBoosting,
  trainKNearestNeighbors,
  trainGaussianNaiveBayes,
  trainElasticNetRegression
} from "./ml";

/**
 * 1. EDA-Driven Model Recommendation Engine
 * Analyzes dataset characteristics (rows, features, outliers, collinearity, skewness, class balance)
 * and generates tailored suitability rankings for candidate algorithms.
 */
export function analyzeEdaAndRecommendModels(
  summary: DatasetSummary,
  targetCol: string,
  taskType: "Classification" | "Regression",
  objectiveGoal: "balanced" | "accuracy" | "interpretability" | "speed" = "balanced"
): ModelRecommendationItem[] {
  const rowCount = summary.rowCount || summary.sampleData?.length || 0;
  const colCount = summary.columns?.length || 0;
  const targetColInfo = summary.columns?.find(c => c.name === targetCol);

  // 1. Analyze Outliers from EDA
  const totalOutliers = summary.outliers?.reduce((acc, o) => acc + (o.outlierCount || 0), 0) || 0;
  const columnsWithOutliers = summary.outliers?.filter(o => o.outlierCount > 0) || [];
  const highOutliersPresent = columnsWithOutliers.some(o => o.outlierPercentage > 5) || totalOutliers > rowCount * 0.05;

  // 2. Analyze Multicollinearity from Correlation Matrix
  let highCorrelationCount = 0;
  const highCorrelatedPairs: Array<[string, string, number]> = [];
  if (summary.correlation) {
    const keys = Object.keys(summary.correlation);
    for (let i = 0; i < keys.length; i++) {
      for (let j = i + 1; j < keys.length; j++) {
        const k1 = keys[i];
        const k2 = keys[j];
        const r = summary.correlation[k1]?.[k2];
        if (r !== undefined && Math.abs(r) >= 0.7 && k1 !== targetCol && k2 !== targetCol) {
          highCorrelationCount++;
          highCorrelatedPairs.push([k1, k2, r]);
        }
      }
    }
  }

  // 3. Analyze Skewness & Non-normality
  const skewedCols = summary.columns?.filter(c => c.type === "Numeric" && c.stats.skewness && Math.abs(c.stats.skewness) > 1.2) || [];
  const hasHighSkewness = skewedCols.length > 0;

  // 4. Analyze Sample Size to Feature Ratio (N/P)
  const npRatio = colCount > 1 ? rowCount / (colCount - 1) : rowCount;
  const isSmallDataset = rowCount < 80;
  const isLargeDataset = rowCount >= 400;

  // 5. Target Class Imbalance (for Classification)
  let isClassImbalanced = false;
  let minorityClassPct = 100;
  if (taskType === "Classification" && targetColInfo?.stats.frequencies) {
    const freqs = Object.values(targetColInfo.stats.frequencies);
    if (freqs.length > 1) {
      const minCount = Math.min(...freqs);
      const totalFreq = freqs.reduce((a, b) => a + b, 0);
      minorityClassPct = (minCount / Math.max(1, totalFreq)) * 100;
      isClassImbalanced = minorityClassPct < 20;
    }
  }

  const recommendations: ModelRecommendationItem[] = [];

  if (taskType === "Classification") {
    // --- Model 1: Random Forest Classifier ---
    let rfScore = 88;
    const rfSignals: string[] = [];
    const rfPros: string[] = [
      "Ensemble of diverse decision trees drastically minimizes variance & overfitting.",
      "Naturally non-parametric: completely invariant to monotonic outliers and feature skewness.",
      "Handles mixed numeric and categorical dummy features without scaling requirements."
    ];
    const rfCons: string[] = [
      "Moderate black-box complexity compared to a single transparent decision tree.",
      "Requires slightly more CPU computations to train multiple estimators."
    ];

    if (highOutliersPresent) {
      rfScore += 5;
      rfSignals.push(`Outliers detected (${totalOutliers} instances): Random Forest is highly resilient to extreme values.`);
    }
    if (hasHighSkewness) {
      rfScore += 4;
      rfSignals.push(`Skewed features detected (${skewedCols.map(c => c.name).slice(0, 2).join(", ")}): Tree ensembling handles non-normal distributions seamlessly.`);
    }
    if (isClassImbalanced) {
      rfScore += 3;
      rfSignals.push(`Class imbalance identified (minority class ${minorityClassPct.toFixed(1)}%): Ensemble voting preserves decision boundaries for rare classes.`);
    }
    if (isSmallDataset) {
      rfScore -= 6;
      rfSignals.push(`Small sample size (N=${rowCount}): Conservative tree depth required to prevent leaf fragmentation.`);
    }
    if (objectiveGoal === "interpretability") rfScore -= 12;
    if (objectiveGoal === "accuracy") rfScore += 6;

    recommendations.push({
      modelId: "random_forest",
      name: "Random Forest Classifier (Ensemble)",
      suitabilityScore: Math.min(99, Math.max(40, rfScore)),
      recommendationLevel: rfScore >= 88 ? "Highly Recommended" : "Recommended",
      badgeColor: "sky",
      edaSignals: rfSignals.length > 0 ? rfSignals : [`Dataset dimensions (${rowCount} rows, ${colCount} cols) provide ideal sample-to-feature ratio.`],
      pros: rfPros,
      cons: rfCons,
      suggestedHyperparams: {
        nEstimators: isSmallDataset ? 6 : 12,
        maxDepth: isSmallDataset ? 3 : 5,
        minSamplesSplit: isSmallDataset ? 4 : 2,
        maxFeaturesRatio: 0.8
      }
    });

    // --- Model 2: Decision Tree Classifier ---
    let dtScore = 82;
    const dtSignals: string[] = [];
    const dtPros: string[] = [
      "100% white-box interpretable: generates explicit human-readable if/then logic rules.",
      "Fast training and inference time with zero feature scaling required.",
      "Captures non-linear thresholds and complex categorical splits."
    ];
    const dtCons: string[] = [
      "Single trees are prone to high variance if allowed to grow deeply without pruning.",
      "Orthogonal split boundaries can struggle with smooth continuous linear frontiers."
    ];

    if (objectiveGoal === "interpretability") {
      dtScore += 15;
      dtSignals.push("Top pick for interpretability: Decision trees provide immediate visual decision paths.");
    }
    if (highOutliersPresent) {
      dtScore += 4;
      dtSignals.push("Robust against feature outliers due to rank-based threshold partitioning.");
    }
    if (isSmallDataset) {
      dtScore += 4;
      dtSignals.push(`Small dataset (N=${rowCount}): A shallow decision tree (depth 2-3) avoids heavy parameter overhead.`);
    }
    if (highCorrelationCount > 0) {
      dtScore -= 3;
      dtSignals.push(`High correlation pairs found (${highCorrelationCount}): Correlated features can compete arbitrarily at tree splits.`);
    }

    recommendations.push({
      modelId: "decision_tree",
      name: "Decision Tree Classifier (CART)",
      suitabilityScore: Math.min(98, Math.max(40, dtScore)),
      recommendationLevel: dtScore >= 85 ? "Highly Recommended" : "Recommended",
      badgeColor: "blue",
      edaSignals: dtSignals.length > 0 ? dtSignals : ["Standard balanced fit across tabular features."],
      pros: dtPros,
      cons: dtCons,
      suggestedHyperparams: {
        maxDepth: isSmallDataset ? 3 : 5,
        minSamplesSplit: isSmallDataset ? 4 : 2,
        criterion: "gini"
      }
    });

    // --- Model 3: Regularized Logistic Regression (L2 / Ridge) ---
    let regLogScore = 80;
    const regLogSignals: string[] = [];
    const regLogPros: string[] = [
      "L2 regularization penalizes large feature weights, preventing overfitting and weight explosion.",
      "Stabilizes coefficients when predictor features exhibit multicollinearity.",
      "Provides probabilistic confidence outputs and calibrated log-odds."
    ];
    const regLogCons: string[] = [
      "Assumes linear decision boundaries in log-odds space.",
      "Sensitive to unscaled features (features must undergo Z-score normalization)."
    ];

    if (highCorrelationCount > 0) {
      regLogScore += 12;
      regLogSignals.push(`Multicollinearity detected: ${highCorrelatedPairs.slice(0, 2).map(([a, b]) => `${a} & ${b}`).join(", ")} (|r| ≥ 0.70). L2 penalty is critical to condition the inversion matrix.`);
    }
    if (npRatio < 10) {
      regLogScore += 6;
      regLogSignals.push(`Low sample-to-feature ratio (${npRatio.toFixed(1)}): Regularization prevents overfitting.`);
    }
    if (highOutliersPresent) {
      regLogScore -= 5;
      regLogSignals.push("Outliers present: Least-squares / gradient log-loss can be sensitive to extreme leverage points.");
    }

    recommendations.push({
      modelId: "regularized_logistic",
      name: "Ridge Logistic Regression (L2 Regularized)",
      suitabilityScore: Math.min(96, Math.max(35, regLogScore)),
      recommendationLevel: regLogScore >= 85 ? "Highly Recommended" : "Recommended",
      badgeColor: "purple",
      edaSignals: regLogSignals.length > 0 ? regLogSignals : ["Solid parametric baseline for linearly separable features."],
      pros: regLogPros,
      cons: regLogCons,
      suggestedHyperparams: {
        learningRate: 0.1,
        epochs: isSmallDataset ? 150 : 250,
        l2Penalty: highCorrelationCount > 0 ? 0.05 : 0.01
      }
    });

    // --- Model 4: Standard Logistic Regression ---
    let logScore = 74;
    const logSignals: string[] = [];
    const logPros: string[] = [
      "Classic statistical baseline with direct odds-ratio interpretation.",
      "Extremely fast gradient descent convergence on normalized data."
    ];
    const logCons: string[] = [
      "Can suffer from numerical instability in the presence of strong collinearity.",
      "Requires monotonic linear separability."
    ];

    if (highCorrelationCount > 0) {
      logScore -= 14;
      logSignals.push(`Warning: Strong collinearity between predictors may destabilize unregularized weights.`);
    }
    if (highOutliersPresent) {
      logScore -= 6;
      logSignals.push("Warning: Outliers can distort the optimal hyperplane slope.");
    }

    recommendations.push({
      modelId: "logistic",
      name: "Logistic Regression (Standard OLS)",
      suitabilityScore: Math.min(90, Math.max(25, logScore)),
      recommendationLevel: logScore >= 75 ? "Recommended" : (logScore >= 60 ? "Viable Alternative" : "Caution Advised"),
      badgeColor: logScore >= 70 ? "sky" : "rose",
      edaSignals: logSignals.length > 0 ? logSignals : ["Simple baseline for unconstrained linear boundaries."],
      pros: logPros,
      cons: logCons,
      suggestedHyperparams: {
        learningRate: 0.08,
        epochs: 200,
        l2Penalty: 0
      }
    });

    // --- Model 5: Gradient Boosted Trees (GBDT) ---
    let gbScore = 92;
    const gbSignals: string[] = [];
    if (objectiveGoal === "accuracy") gbScore += 6;
    if (highOutliersPresent) {
      gbScore += 3;
      gbSignals.push("Gradient boosting corrects iterative pseudo-residuals effectively.");
    }
    if (isSmallDataset) {
      gbScore -= 8;
      gbSignals.push("Requires careful regularization (low learning rate, shallow depth) on smaller datasets.");
    }
    recommendations.push({
      modelId: "gradient_boosting",
      name: "Gradient Boosted Trees (GBDT)",
      suitabilityScore: Math.min(99, Math.max(45, gbScore)),
      recommendationLevel: gbScore >= 88 ? "Highly Recommended" : "Recommended",
      badgeColor: "sky",
      edaSignals: gbSignals.length > 0 ? gbSignals : ["State-of-the-art competitive performance on structured tabular records."],
      pros: [
        "Iteratively minimizes log-loss residuals for superior discrimination.",
        "Built-in feature interaction modeling with shrinkage regularization."
      ],
      cons: [
        "Sequential tree construction cannot be fully parallelized.",
        "Requires tuning of learning rate and tree depth to avoid over-specialization."
      ],
      suggestedHyperparams: {
        nEstimators: isSmallDataset ? 10 : 20,
        learningRate: 0.1,
        maxDepth: 3
      }
    });

    // --- Model 6: K-Nearest Neighbors (KNN) ---
    let knnScore = 78;
    const knnSignals: string[] = [];
    if (npRatio > 15) {
      knnScore += 4;
      knnSignals.push("Dense geometric neighborhood available for local metric distance voting.");
    }
    if (colCount > 15) {
      knnScore -= 10;
      knnSignals.push("Curse of dimensionality: high feature counts dilute Euclidean distance metrics.");
    }
    recommendations.push({
      modelId: "knn",
      name: "K-Nearest Neighbors (KNN)",
      suitabilityScore: Math.min(92, Math.max(30, knnScore)),
      recommendationLevel: knnScore >= 75 ? "Recommended" : "Viable Alternative",
      badgeColor: "blue",
      edaSignals: knnSignals.length > 0 ? knnSignals : ["Instance-based non-parametric classifier based on local feature proximity."],
      pros: [
        "Completely non-parametric: makes zero assumptions about underlying probability distributions.",
        "Simple and transparent distance-weighted nearest neighbor mechanism."
      ],
      cons: [
        "Inference cost scales linearly with the number of training samples.",
        "Sensitive to irrelevant features and unscaled numerical variables."
      ],
      suggestedHyperparams: {
        kNeighbors: 5,
        weights: "distance"
      }
    });

    // --- Model 7: Gaussian Naive Bayes ---
    let nbScore = 72;
    const nbSignals: string[] = [];
    if (isSmallDataset) {
      nbScore += 8;
      nbSignals.push("Low sample count: Naive Bayes' strong independence assumption resists severe overfitting.");
    }
    if (highCorrelationCount > 0) {
      nbScore -= 12;
      nbSignals.push("Collinear features violate the conditional feature independence assumption.");
    }
    recommendations.push({
      modelId: "naive_bayes",
      name: "Gaussian Naive Bayes (GNB)",
      suitabilityScore: Math.min(88, Math.max(25, nbScore)),
      recommendationLevel: nbScore >= 75 ? "Recommended" : "Viable Alternative",
      badgeColor: "purple",
      edaSignals: nbSignals.length > 0 ? nbSignals : ["Probabilistic Bayesian classifier using normal distribution likelihoods."],
      pros: [
        "Blazing fast closed-form analytical training with no iterative optimization required.",
        "Excellent baseline for small sample sets and high variance regimes."
      ],
      cons: [
        "Assumes feature independence conditional on the class label.",
        "Assumes continuous features follow Gaussian normal distributions."
      ],
      suggestedHyperparams: {}
    });

  } else {
    // === REGRESSION TASK ===

    // --- Model 1: Random Forest Regressor ---
    let rfScore = 89;
    const rfSignals: string[] = [];
    const rfPros: string[] = [
      "Superb non-linear modeling without requiring feature transformations or basis expansions.",
      "Resilient to outliers in predictor features; builds piecewise constant approximations.",
      "Reduces variance via tree bagging across random feature subspaces."
    ];
    const rfCons: string[] = [
      "Cannot extrapolate beyond the minimum and maximum target ranges in the training set.",
      "Larger memory footprint than compact parametric linear regression."
    ];

    if (highOutliersPresent) {
      rfScore += 6;
      rfSignals.push(`Outliers detected in ${columnsWithOutliers.length} columns: Ensemble splits are insensitive to extreme values.`);
    }
    if (hasHighSkewness) {
      rfScore += 5;
      rfSignals.push(`High numeric skewness in predictors: CART trees isolate skewed regimes without distortion.`);
    }
    if (objectiveGoal === "accuracy") rfScore += 6;
    if (objectiveGoal === "interpretability") rfScore -= 10;

    recommendations.push({
      modelId: "random_forest",
      name: "Random Forest Regressor (Ensemble)",
      suitabilityScore: Math.min(99, Math.max(40, rfScore)),
      recommendationLevel: rfScore >= 88 ? "Highly Recommended" : "Recommended",
      badgeColor: "sky",
      edaSignals: rfSignals.length > 0 ? rfSignals : [`Clean fit for continuous target with ${colCount} predictors.`],
      pros: rfPros,
      cons: rfCons,
      suggestedHyperparams: {
        nEstimators: isSmallDataset ? 6 : 12,
        maxDepth: isSmallDataset ? 3 : 5,
        minSamplesSplit: isSmallDataset ? 4 : 2,
        maxFeaturesRatio: 0.8
      }
    });

    // --- Model 2: Ridge Regression (L2 Regularized) ---
    let ridgeScore = 84;
    const ridgeSignals: string[] = [];
    const ridgePros: string[] = [
      "Contracts coefficients towards zero, drastically mitigating multicollinearity variance.",
      "Continuous smooth extrapolation with well-defined gradients.",
      "High computational speed and exact parameter interpretability (standardized slopes)."
    ];
    const ridgeCons: string[] = [
      "Limited to planar linear hyperplanes unless polynomial interactions are manually engineered.",
      "Sensitive to extreme target outliers (squared loss penalty)."
    ];

    if (highCorrelationCount > 0) {
      ridgeScore += 12;
      ridgeSignals.push(`Multicollinearity found (|r| ≥ 0.70): L2 Ridge shrinkage resolves unstable singular matrices.`);
    }
    if (highOutliersPresent) {
      ridgeScore -= 4;
      ridgeSignals.push("Outliers detected: Squared error loss penalizes large residuals aggressively.");
    }
    if (objectiveGoal === "speed" || objectiveGoal === "interpretability") ridgeScore += 6;

    recommendations.push({
      modelId: "ridge",
      name: "Ridge Regression (L2 Regularized)",
      suitabilityScore: Math.min(98, Math.max(40, ridgeScore)),
      recommendationLevel: ridgeScore >= 85 ? "Highly Recommended" : "Recommended",
      badgeColor: "purple",
      edaSignals: ridgeSignals.length > 0 ? ridgeSignals : ["Standard regularized linear baseline."],
      pros: ridgePros,
      cons: ridgeCons,
      suggestedHyperparams: {
        learningRate: 0.1,
        epochs: isSmallDataset ? 150 : 250,
        l2Penalty: highCorrelationCount > 0 ? 0.05 : 0.01
      }
    });

    // --- Model 3: Decision Tree Regressor ---
    let dtScore = 78;
    const dtSignals: string[] = [];
    const dtPros: string[] = [
      "Captures non-linear thresholds, step functions, and conditional segmentations.",
      "Transparent decision path for operational business rules."
    ];
    const dtCons: string[] = [
      "Produces piecewise constant predictions with non-smooth steps.",
      "Prone to overfitting on small continuous targets if depth is excessive."
    ];

    if (objectiveGoal === "interpretability") {
      dtScore += 12;
      dtSignals.push("High interpretability: Generates human-auditable numeric partition rules.");
    }
    if (highOutliersPresent) {
      dtScore += 4;
      dtSignals.push("Robust against predictor outliers.");
    }

    recommendations.push({
      modelId: "decision_tree",
      name: "Decision Tree Regressor (CART)",
      suitabilityScore: Math.min(94, Math.max(35, dtScore)),
      recommendationLevel: dtScore >= 80 ? "Recommended" : "Viable Alternative",
      badgeColor: "blue",
      edaSignals: dtSignals.length > 0 ? dtSignals : ["Useful for segmented or tiered target dynamics."],
      pros: dtPros,
      cons: dtCons,
      suggestedHyperparams: {
        maxDepth: isSmallDataset ? 3 : 5,
        minSamplesSplit: isSmallDataset ? 4 : 2,
        criterion: "mse"
      }
    });

    // --- Model 4: Standard Linear Regression (OLS) ---
    let linScore = 70;
    const linSignals: string[] = [];
    const linPros: string[] = [
      "Classic Ordinary Least Squares baseline with simple weight coefficients.",
      "Zero hyperparameter complexity."
    ];
    const linCons: string[] = [
      "Suffers large variance when features are correlated.",
      "Sensitive to leverage outliers."
    ];

    if (highCorrelationCount > 0) {
      linScore -= 15;
      linSignals.push("Collinear features detected: Unpenalized weights can blow up in magnitude.");
    }
    if (highOutliersPresent) {
      linScore -= 8;
      linSignals.push("Outliers will heavily skew the least-squares best-fit plane.");
    }

    recommendations.push({
      modelId: "linear",
      name: "Linear Regression (Ordinary Least Squares)",
      suitabilityScore: Math.min(88, Math.max(20, linScore)),
      recommendationLevel: linScore >= 75 ? "Recommended" : (linScore >= 60 ? "Viable Alternative" : "Caution Advised"),
      badgeColor: linScore >= 70 ? "sky" : "rose",
      edaSignals: linSignals.length > 0 ? linSignals : ["Simple benchmark baseline."],
      pros: linPros,
      cons: linCons,
      suggestedHyperparams: {
        learningRate: 0.05,
        epochs: 200,
        l2Penalty: 0
      }
    });

    // --- Model 5: Gradient Boosted Regressor (GBDT) ---
    let gbRegScore = 93;
    const gbRegSignals: string[] = [];
    if (objectiveGoal === "accuracy") gbRegScore += 5;
    if (highOutliersPresent) {
      gbRegScore += 4;
      gbRegSignals.push("Sequential residual trees model non-linear target manifolds with high precision.");
    }
    recommendations.push({
      modelId: "gradient_boosting",
      name: "Gradient Boosted Regressor (GBDT)",
      suitabilityScore: Math.min(99, Math.max(45, gbRegScore)),
      recommendationLevel: gbRegScore >= 88 ? "Highly Recommended" : "Recommended",
      badgeColor: "sky",
      edaSignals: gbRegSignals.length > 0 ? gbRegSignals : ["Industry gold standard for tabular regression with non-linear relationships."],
      pros: [
        "Iteratively fits shallow trees to minimize squared loss residuals.",
        "Captures complex high-order feature interactions without feature engineering."
      ],
      cons: [
        "Slightly longer training time due to sequential tree construction.",
        "Requires learning rate and tree depth tuning to guard against overfitting."
      ],
      suggestedHyperparams: {
        nEstimators: isSmallDataset ? 10 : 20,
        learningRate: 0.1,
        maxDepth: 3
      }
    });

    // --- Model 6: ElasticNet Regression (L1 + L2) ---
    let enScore = 86;
    const enSignals: string[] = [];
    if (highCorrelationCount > 0) {
      enScore += 8;
      enSignals.push("ElasticNet groups correlated features together via L2 ridge while selecting sparse active subsets via L1 lasso.");
    }
    if (colCount > 8) {
      enScore += 4;
      enSignals.push("Multi-feature space: L1 sparsity penalty zeroes out redundant predictors.");
    }
    recommendations.push({
      modelId: "elastic_net",
      name: "ElasticNet Regression (L1 + L2 Regularized)",
      suitabilityScore: Math.min(97, Math.max(35, enScore)),
      recommendationLevel: enScore >= 85 ? "Highly Recommended" : "Recommended",
      badgeColor: "purple",
      edaSignals: enSignals.length > 0 ? enSignals : ["Balances sparsity feature selection with numerical coefficient stability."],
      pros: [
        "Overcomes Lasso's limitation when features are strongly correlated by retaining feature groups.",
        "Generates sparse, easily explainable linear coefficient weights."
      ],
      cons: [
        "Requires tuning both the total alpha penalty and the L1 ratio.",
        "Assumes continuous linear response surface."
      ],
      suggestedHyperparams: {
        learningRate: 0.08,
        epochs: 200,
        l1Ratio: 0.5,
        l2Penalty: 0.05
      }
    });

    // --- Model 7: K-Nearest Neighbors Regressor ---
    let knnRegScore = 77;
    const knnRegSignals: string[] = [];
    if (npRatio > 15) {
      knnRegScore += 5;
      knnRegSignals.push("Sufficient sample density enables smooth local distance interpolation.");
    }
    recommendations.push({
      modelId: "knn",
      name: "K-Nearest Neighbors Regressor (KNN)",
      suitabilityScore: Math.min(91, Math.max(30, knnRegScore)),
      recommendationLevel: knnRegScore >= 75 ? "Recommended" : "Viable Alternative",
      badgeColor: "blue",
      edaSignals: knnRegSignals.length > 0 ? knnRegSignals : ["Locally weighted geometric interpolation across nearest feature records."],
      pros: [
        "Zero training phase assumptions: predictions adapt locally to arbitrary geometry.",
        "Intuitive distance-weighted prediction averaging."
      ],
      cons: [
        "Cannot extrapolate beyond observed training extremes.",
        "Performance degrades as feature dimensionality increases."
      ],
      suggestedHyperparams: {
        kNeighbors: 5,
        weights: "distance"
      }
    });
  }

  // Sort descending by suitabilityScore
  return recommendations.sort((a, b) => b.suitabilityScore - a.suitabilityScore);
}

/**
 * 2. Automated Model Tournament / Benchmark
 * Runs all candidate models head-to-head on the current feature/target data
 * and produces a comparative leaderboard with a crowned champion.
 */
export async function runModelBenchmark(
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  XTrainScaled: number[][],
  XTestScaled: number[][],
  classLabels: string[],
  taskType: "Classification" | "Regression",
  onProgress?: (step: number, total: number, currentModel: string) => void
): Promise<BenchmarkResult[]> {
  const candidateModels = taskType === "Classification" 
    ? [
        { id: "gradient_boosting", name: "Gradient Boosted Trees (GBDT)", isScaled: false },
        { id: "random_forest", name: "Random Forest (Ensemble)", isScaled: false },
        { id: "decision_tree", name: "Decision Tree (CART)", isScaled: false },
        { id: "knn", name: "K-Nearest Neighbors (KNN)", isScaled: true },
        { id: "naive_bayes", name: "Gaussian Naive Bayes", isScaled: false },
        { id: "regularized_logistic", name: "Ridge Logistic (L2)", isScaled: true },
        { id: "logistic", name: "Logistic Regression (OLS)", isScaled: true },
      ]
    : [
        { id: "gradient_boosting", name: "Gradient Boosted Regressor (GBDT)", isScaled: false },
        { id: "random_forest", name: "Random Forest (Ensemble)", isScaled: false },
        { id: "decision_tree", name: "Decision Tree (CART)", isScaled: false },
        { id: "knn", name: "K-Nearest Neighbors Regressor (KNN)", isScaled: true },
        { id: "elastic_net", name: "ElasticNet (L1 + L2)", isScaled: true },
        { id: "ridge", name: "Ridge Regression (L2)", isScaled: true },
        { id: "linear", name: "Linear Regression (OLS)", isScaled: true },
      ];

  const results: BenchmarkResult[] = [];
  const total = candidateModels.length;

  for (let i = 0; i < candidateModels.length; i++) {
    const cm = candidateModels[i];
    if (onProgress) onProgress(i + 1, total, cm.name);

    const t0 = performance.now();
    let trainRes;

    if (taskType === "Classification") {
      if (cm.id === "gradient_boosting") {
        trainRes = await trainGradientBoosting(XTrain, yTrain, XTest, yTest, classLabels, "Classification", { nEstimators: 12, maxDepth: 3 });
      } else if (cm.id === "knn") {
        trainRes = await trainKNearestNeighbors(XTrain, yTrain, XTest, yTest, classLabels, "Classification", { kNeighbors: 5 });
      } else if (cm.id === "naive_bayes") {
        trainRes = await trainGaussianNaiveBayes(XTrain, yTrain, XTest, yTest, classLabels);
      } else if (cm.id === "logistic") {
        trainRes = await trainLogisticRegression(XTrainScaled, yTrain, XTestScaled, yTest, classLabels, 200, 0.1, undefined, 0, true);
      } else if (cm.id === "regularized_logistic") {
        trainRes = await trainLogisticRegression(XTrainScaled, yTrain, XTestScaled, yTest, classLabels, 200, 0.1, undefined, 0.05, true);
      } else if (cm.id === "decision_tree") {
        trainRes = await trainDecisionTree(XTrain, yTrain, XTest, yTest, classLabels, 5, "Classification", 2, "gini", true);
      } else {
        trainRes = await trainRandomForest(XTrain, yTrain, XTest, yTest, classLabels, "Classification", { nEstimators: 10, maxDepth: 4 });
      }
    } else {
      if (cm.id === "gradient_boosting") {
        trainRes = await trainGradientBoosting(XTrain, yTrain, XTest, yTest, classLabels, "Regression", { nEstimators: 12, maxDepth: 3 });
      } else if (cm.id === "knn") {
        trainRes = await trainKNearestNeighbors(XTrain, yTrain, XTest, yTest, classLabels, "Regression", { kNeighbors: 5 });
      } else if (cm.id === "elastic_net") {
        trainRes = await trainElasticNetRegression(XTrainScaled, yTrain, XTestScaled, yTest, 180, 0.08, 0.5, 0.05);
      } else if (cm.id === "linear") {
        trainRes = await trainLinearRegression(XTrainScaled, yTrain, XTestScaled, yTest, 200, 0.08, undefined, 0, true);
      } else if (cm.id === "ridge") {
        trainRes = await trainLinearRegression(XTrainScaled, yTrain, XTestScaled, yTest, 200, 0.08, undefined, 0.05, true);
      } else if (cm.id === "decision_tree") {
        trainRes = await trainDecisionTree(XTrain, yTrain, XTest, yTest, classLabels, 5, "Regression", 2, "mse", true);
      } else {
        trainRes = await trainRandomForest(XTrain, yTrain, XTest, yTest, classLabels, "Regression", { nEstimators: 10, maxDepth: 4 });
      }
    }

    const fitTimeMs = Math.round(performance.now() - t0);
    const primaryMetricName = taskType === "Classification" ? "Accuracy" : "R2 Score (R²)";
    const valMetric = trainRes.testMetrics[primaryMetricName] ?? (taskType === "Classification" ? 0 : -1);
    const trainMetric = trainRes.trainMetrics[primaryMetricName] ?? 0;

    results.push({
      modelId: cm.id,
      name: cm.name,
      primaryMetricName,
      valMetric,
      trainMetric,
      fitTimeMs,
      interpretability: cm.id.includes("decision_tree") || cm.id.includes("linear") || cm.id.includes("logistic") || cm.id.includes("naive") ? "High" : (cm.id.includes("random") || cm.id.includes("knn") ? "Medium" : "Low"),
      isChampion: false,
      metrics: trainRes.testMetrics
    });

    // Brief pause for UI rendering
    await new Promise(r => setTimeout(r, 40));
  }

  // Determine champion (highest validation primary metric)
  results.sort((a, b) => b.valMetric - a.valMetric);
  if (results.length > 0) {
    results[0].isChampion = true;
  }

  return results;
}

/**
 * 3. Automated Hyperparameter Optimization (HPO Engine)
 * Runs systematic Grid or Random Search over candidate parameters to maximize validation score.
 */
export async function runHyperparameterOptimization(
  modelId: string,
  modelName: string,
  XTrain: number[][],
  yTrain: number[],
  XTest: number[][],
  yTest: number[],
  XTrainScaled: number[][],
  XTestScaled: number[][],
  classLabels: string[],
  taskType: "Classification" | "Regression",
  options: {
    strategy?: "grid" | "random";
    numTrials?: number;
    targetMetric?: string;
    onTrialComplete?: (trial: HPOTrialResult, progress: number) => void;
  } = {}
): Promise<HPOOptimizationResult> {
  const strategy = options.strategy || "random";
  const numTrials = Math.max(4, Math.min(24, options.numTrials || 10));
  const targetMetric = options.targetMetric || (taskType === "Classification" ? "Accuracy" : "R2 Score (R²)");

  // 1. Generate search space according to model type
  const paramCombinations: Array<Record<string, any>> = [];

  if (modelId === "decision_tree") {
    const depths = [2, 3, 4, 5, 6, 8, 10];
    const splits = [2, 4, 6, 8];
    const criteria = taskType === "Classification" ? ["gini", "entropy"] : ["mse", "mae"];

    // Seed with standard baseline
    paramCombinations.push({ maxDepth: 5, minSamplesSplit: 2, criterion: criteria[0] });

    if (strategy === "grid") {
      for (const d of [2, 4, 6, 8]) {
        for (const s of [2, 4]) {
          for (const c of criteria) {
            paramCombinations.push({ maxDepth: d, minSamplesSplit: s, criterion: c });
          }
        }
      }
    } else {
      while (paramCombinations.length < numTrials) {
        const d = depths[Math.floor(Math.random() * depths.length)];
        const s = splits[Math.floor(Math.random() * splits.length)];
        const c = criteria[Math.floor(Math.random() * criteria.length)];
        if (!paramCombinations.some(p => p.maxDepth === d && p.minSamplesSplit === s && p.criterion === c)) {
          paramCombinations.push({ maxDepth: d, minSamplesSplit: s, criterion: c });
        }
      }
    }
  } else if (modelId === "gradient_boosting") {
    const estimatorsList = [8, 12, 16, 20, 25];
    const depths = [2, 3, 4, 5];
    const lrs = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2];

    paramCombinations.push({ nEstimators: 15, maxDepth: 3, learningRate: 0.1 });

    while (paramCombinations.length < numTrials) {
      const e = estimatorsList[Math.floor(Math.random() * estimatorsList.length)];
      const d = depths[Math.floor(Math.random() * depths.length)];
      const lr = lrs[Math.floor(Math.random() * lrs.length)];
      if (!paramCombinations.some(p => p.nEstimators === e && p.maxDepth === d && p.learningRate === lr)) {
        paramCombinations.push({ nEstimators: e, maxDepth: d, learningRate: lr });
      }
    }
  } else if (modelId === "knn") {
    const kList = [2, 3, 5, 7, 9, 11];
    const weightList: Array<"distance" | "uniform"> = ["distance", "uniform"];

    paramCombinations.push({ kNeighbors: 5, weights: "distance" });

    for (const k of kList) {
      for (const w of weightList) {
        if (!paramCombinations.some(p => p.kNeighbors === k && p.weights === w)) {
          paramCombinations.push({ kNeighbors: k, weights: w });
        }
      }
    }
  } else if (modelId === "elastic_net") {
    const l1List = [0.1, 0.3, 0.5, 0.7, 0.9];
    const l2List = [0.01, 0.03, 0.05, 0.1, 0.2];
    const lrs = [0.05, 0.08, 0.1];

    paramCombinations.push({ l1Ratio: 0.5, l2Penalty: 0.05, learningRate: 0.08, epochs: 200 });

    while (paramCombinations.length < numTrials) {
      const l1 = l1List[Math.floor(Math.random() * l1List.length)];
      const l2 = l2List[Math.floor(Math.random() * l2List.length)];
      const lr = lrs[Math.floor(Math.random() * lrs.length)];
      if (!paramCombinations.some(p => p.l1Ratio === l1 && p.l2Penalty === l2 && p.learningRate === lr)) {
        paramCombinations.push({ l1Ratio: l1, l2Penalty: l2, learningRate: lr, epochs: 200 });
      }
    }
  } else if (modelId === "random_forest") {
    const estimatorsList = [5, 8, 10, 12, 16];
    const depths = [3, 4, 5, 7];
    const splits = [2, 4, 6];
    const featureRatios = [0.6, 0.8, 1.0];

    paramCombinations.push({ nEstimators: 8, maxDepth: 4, minSamplesSplit: 2, maxFeaturesRatio: 0.8 });

    while (paramCombinations.length < numTrials) {
      const e = estimatorsList[Math.floor(Math.random() * estimatorsList.length)];
      const d = depths[Math.floor(Math.random() * depths.length)];
      const s = splits[Math.floor(Math.random() * splits.length)];
      const fr = featureRatios[Math.floor(Math.random() * featureRatios.length)];
      paramCombinations.push({ nEstimators: e, maxDepth: d, minSamplesSplit: s, maxFeaturesRatio: fr });
    }
  } else {
    // Linear or Logistic Regression (OLS or Ridge)
    const lrs = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2];
    const epochList = [100, 150, 200, 250, 300, 400];
    const l2List = modelId.includes("ridge") || modelId.includes("regularized") 
      ? [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
      : [0, 0.001, 0.01, 0.05];

    paramCombinations.push({ learningRate: 0.1, epochs: 200, l2Penalty: modelId.includes("ridge") ? 0.05 : 0 });

    while (paramCombinations.length < numTrials) {
      const lr = lrs[Math.floor(Math.random() * lrs.length)];
      const ep = epochList[Math.floor(Math.random() * epochList.length)];
      const l2 = l2List[Math.floor(Math.random() * l2List.length)];
      if (!paramCombinations.some(p => p.learningRate === lr && p.epochs === ep && p.l2Penalty === l2)) {
        paramCombinations.push({ learningRate: lr, epochs: ep, l2Penalty: l2 });
      }
    }
  }

  const trialsToRun = paramCombinations.slice(0, numTrials);
  const trials: HPOTrialResult[] = [];
  let baselineScore = 0;
  let bestScore = -Infinity;
  let bestParams = trialsToRun[0];

  for (let t = 0; t < trialsToRun.length; t++) {
    const params = trialsToRun[t];
    let fitResult;

    if (modelId === "decision_tree") {
      fitResult = await trainDecisionTree(
        XTrain,
        yTrain,
        XTest,
        yTest,
        classLabels,
        params.maxDepth,
        taskType,
        params.minSamplesSplit,
        params.criterion,
        true
      );
    } else if (modelId === "random_forest") {
      fitResult = await trainRandomForest(
        XTrain,
        yTrain,
        XTest,
        yTest,
        classLabels,
        taskType,
        params
      );
    } else if (modelId === "gradient_boosting") {
      fitResult = await trainGradientBoosting(
        XTrain,
        yTrain,
        XTest,
        yTest,
        classLabels,
        taskType,
        params
      );
    } else if (modelId === "knn") {
      fitResult = await trainKNearestNeighbors(
        XTrain,
        yTrain,
        XTest,
        yTest,
        classLabels,
        taskType,
        params
      );
    } else if (modelId === "elastic_net") {
      fitResult = await trainElasticNetRegression(
        XTrainScaled,
        yTrain,
        XTestScaled,
        yTest,
        params.epochs || 200,
        params.learningRate || 0.08,
        params.l1Ratio || 0.5,
        params.l2Penalty || 0.05
      );
    } else if (taskType === "Classification") {
      fitResult = await trainLogisticRegression(
        XTrainScaled,
        yTrain,
        XTestScaled,
        yTest,
        classLabels,
        params.epochs,
        params.learningRate,
        undefined,
        params.l2Penalty || 0,
        true
      );
    } else {
      fitResult = await trainLinearRegression(
        XTrainScaled,
        yTrain,
        XTestScaled,
        yTest,
        params.epochs,
        params.learningRate,
        undefined,
        params.l2Penalty || 0,
        true
      );
    }

    const valScore = fitResult.testMetrics[targetMetric] ?? (taskType === "Classification" ? 0 : -1);
    const trainScore = fitResult.trainMetrics[targetMetric] ?? 0;

    if (t === 0) {
      baselineScore = valScore;
      bestScore = valScore;
      bestParams = params;
    } else if (valScore > bestScore) {
      bestScore = valScore;
      bestParams = params;
    }

    const trialRes: HPOTrialResult = {
      trial: t + 1,
      params,
      trainScore,
      valScore,
      isBest: valScore >= bestScore
    };

    trials.push(trialRes);

    if (options.onTrialComplete) {
      options.onTrialComplete(trialRes, Math.round(((t + 1) / trialsToRun.length) * 100));
    }

    // Small delay so users see trial progression in the UI
    await new Promise(r => setTimeout(r, 60));
  }

  // Update isBest marks relative to the final best
  trials.forEach(tr => {
    tr.isBest = tr.valScore === bestScore;
  });

  // Calculate lift percentage
  let improvementPct = 0;
  if (baselineScore !== 0) {
    improvementPct = Number((((bestScore - baselineScore) / Math.abs(baselineScore)) * 100).toFixed(2));
  } else if (bestScore > 0) {
    improvementPct = Number((bestScore * 100).toFixed(2));
  }

  return {
    modelId,
    modelName,
    targetMetric,
    baselineScore: Number(baselineScore.toFixed(4)),
    bestScore: Number(bestScore.toFixed(4)),
    improvementPct,
    bestParams,
    trials,
    strategy
  };
}
