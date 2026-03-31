import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from column_identification import (
    classify_numeric_columns,
    detect_datetime_columns,
    detect_categorical_columns,
    is_id_like_numeric
)
from utils import (
    get_data_quality_metrics_df,
    clean_dataset,
    get_distribution_insights_df
)
from data_analysis import (
    get_categorical_descriptive_df,
    get_numerical_descriptive_df,
    get_numeric_correlation_diagnostics,
    get_pearson_corr_matrix,
    get_kendall_correlation_df,
    get_spearman_correlation_df,
    get_correlation_pairs_df,
    numeric_prescriptive_df,
    categorical_prescriptive_df,
    correlation_prescriptive_df,
    dataset_prescriptive_summary
)
from models import (
    infer_target_type,  
    get_model_ready_features_df,
    get_model_recommendations_df,
    get_training_plan
)
from baseline_model import train_baseline_model
st.markdown(
    """
    <h1 style="
        font-style:Italic;
        font-size:100px;
        font-weight:700;
        margin-bottom:10px;
    ">
        AutoML
    </h1>
    """,
    unsafe_allow_html=True
)
st.divider()
# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Automated Dataset Cleaner & Analyzer",
    layout="wide"
)


sns.set_theme(style="darkgrid")

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------
st.sidebar.title("🧠 Dataset Intelligence System")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Analysis",
        "ML Models",
        "AI Assistant"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

df = pd.read_csv(uploaded_file) if uploaded_file else None

# ==========================================================
# GLOBAL COLUMN INTELLIGENCE (RUN ONCE)
# ==========================================================
if df is not None:
    object_cols = df.select_dtypes(include="object").columns.tolist()
    native_numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    converted_numeric, discrete_numeric, continuous_numeric = classify_numeric_columns(
        df, object_cols
    )

    numeric_cols = list(set(native_numeric + converted_numeric))
    datetime_cols = detect_datetime_columns(df)
    categorical_cols = detect_categorical_columns(df)

    categorical_cols = list(
        set(categorical_cols)
        - set(datetime_cols)
        - set(numeric_cols)
    )
    column_type_map = {}

    for col in df.columns:
        if col in continuous_numeric:
            column_type_map[col] = "Continuous Numeric"
        elif col in discrete_numeric:
            column_type_map[col] = "Discrete Numeric"
        elif col in datetime_cols:
            column_type_map[col] = "Date / Time"
        elif col in categorical_cols:
            column_type_map[col] = "Categorical"
        else:
            column_type_map[col] = "Unknown"
else:
    discrete_numeric = []
    continuous_numeric = []
    datetime_cols = []
    categorical_cols = []
    numeric_cols = []

# ==========================================================
# PAGE 1: OVERVIEW & CLEANING
# ==========================================================
if page == "Overview":


    st.subheader("Dataset Overview")

    if df is None:
        st.info("Upload a dataset to begin")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "Dataset Structure",
        "Cleaning Report",
        "Distribution",
    ])
    with tab1:
        # --------- TOP METRICS ---------
        rows, cols = df.shape
        missing_pct = round(df.isna().mean().mean() * 100, 2)
        memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", rows)
        m2.metric("Columns", cols)
        m3.metric("Missing Data", f"{missing_pct}%")
        m4.metric("Memory", f"{memory_mb} MB")

        # --------- CLEANING NOTE ---------
        st.markdown(
            """
            <div style="
                background-color:#4b5320;
                padding:14px;
                border-radius:10px;
                margin-top:12px;
                margin-bottom:16px;
                color:#f5f5dc;
                font-weight:500;
            ">
            <b>Note:</b> Some columns may have been automatically removed during cleaning
            (ID columns, constants, or high-missing data).
            See <b>Ingestion Report</b> for details.
            </div>
            """,
            unsafe_allow_html=True
        )

        # --------- COLUMN TYPE SUMMARY ---------
        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            f"""
            <div style="background:#102a43;padding:15px;border-radius:10px;color:#7cc4ff;">
            <b>Continuous Numeric</b><br>
            <span style="font-size:26px;">{len(continuous_numeric)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c2.markdown(
            f"""
            <div style="background:#123d2f;padding:15px;border-radius:10px;color:#6ef3b4;">
            <b>Discrete Numeric</b><br>
            <span style="font-size:26px;">{len(discrete_numeric)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c3.markdown(
            f"""
            <div style="background:#3b3a14;padding:15px;border-radius:10px;color:#f1e05a;">
            <b>Categorical</b><br>
            <span style="font-size:26px;">{len(categorical_cols)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c4.markdown(
            f"""
            <div style="background:#3b1f25;padding:15px;border-radius:10px;color:#ff6b6b;">
            <b>Date / Time</b><br>
            <span style="font-size:26px;">{len(datetime_cols)}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        st.subheader("Column Classification Summary")
        if df is None:
            st.info("Upload a dataset to begin")
            st.stop()

        rows = []
        for col in df.columns:
            unique_count = df[col].nunique()

            if col in datetime_cols:
                col_type = "Datetime"
                reasoning = "Valid datetime values detected"
            elif col in discrete_numeric:
                col_type = "Discrete Numeric"
                reasoning = f"Integer with limited values ({unique_count})"
            elif col in continuous_numeric:
                col_type = "Continuous Numeric"
                reasoning = f"High cardinality numeric ({unique_count})"
            elif col in categorical_cols:
                col_type = "Categorical"
                reasoning = f"Low cardinality ({unique_count})"
            else:
                col_type = "Unclassified"
                reasoning = "Ambiguous behavior"

            rows.append({
                "Column": col,
                "Type": col_type,
                "Reasoning": reasoning,
                "Unique Values": unique_count,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with tab2:
        # --------- CLEANING REPORT ---------
        st.subheader("Data Quality Metrics")

        metrics_df = get_data_quality_metrics_df(df)

        st.dataframe(metrics_df)
        # --------- CLEANED DATA PREVIEW ---------
        st.divider()
        st.subheader("Cleaned Dataset Preview")
        cleaned_df = clean_dataset(df)
        st.dataframe(cleaned_df.head(100), use_container_width=True)

    with tab3:
        st.subheader("Column Distribution Overview")
        
        # Get all columns to display
        all_columns = df.columns.tolist()
        
        # Display graphs in rows of 3
        for i in range(0, len(all_columns), 3):
            cols = st.columns(3)
            
            for j in range(3):
                if i + j < len(all_columns):
                    selected_col = all_columns[i + j]
                    column_type = column_type_map[selected_col]
                    
                    with cols[j]:
                        # Create the plot
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        if selected_col in continuous_numeric:
                            sns.histplot(df[selected_col], kde=True, bins=30, ax=ax)
                            ax.set_title(f"Distribution of {selected_col}")
                        
                        elif selected_col in discrete_numeric:
                            if is_id_like_numeric(df[selected_col]):
                                vc = df[selected_col].astype(str).value_counts()
                                sns.barplot(
                                    x=vc.index,
                                    y=vc.values,
                                    ax=ax
                                )
                                ax.set_title(f"Frequency of {selected_col}")
                                ax.set_xlabel(selected_col)
                                ax.set_ylabel("Count")
                                ax.tick_params(axis="x", rotation=45)
                            else:
                                sns.histplot(df[selected_col], discrete=True, ax=ax)
                                ax.set_title(f"Discrete Distribution of {selected_col}")
                        
                        elif selected_col in categorical_cols:
                            vc = df[selected_col].value_counts().reset_index()
                            vc.columns = [selected_col, "count"]
                            sns.barplot(data=vc, x=selected_col, y="count", ax=ax)
                            ax.set_title(f"Category Counts of {selected_col}")
                            ax.tick_params(axis="x", rotation=45)
                        
                        elif selected_col in datetime_cols:
                            parsed = pd.to_datetime(df[selected_col], errors="coerce").dropna()
                            month_counts = (
                                parsed.dt.month_name()
                                .value_counts()
                                .reindex([
                                    "January", "February", "March", "April", "May", "June",
                                    "July", "August", "September", "October", "November", "December"
                                ])
                                .dropna()
                            )
                            sns.barplot(
                                x=month_counts.index,
                                y=month_counts.values,
                                ax=ax
                            )
                            ax.set_title(f"Monthly Distribution of {selected_col}")
                            ax.set_xlabel("Month")
                            ax.set_ylabel("Count")
                            ax.tick_params(axis="x", rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display metrics below the graph
                        insights_df = get_distribution_insights_df(
                            df,
                            selected_col,
                            column_type
                        )
                        
                        st.dataframe(insights_df, use_container_width=True, hide_index=True)


# ==========================================================
# Data Analysis
# ==========================================================
elif page == "Data Analysis":
    st.subheader("Data Analysis")
    if df is None:
            st.info("Upload a dataset to begin")
            st.stop()
    tab4, tab5, tab6 = st.tabs([
        "Descriptive",
        "Diagnostic",
        "Prescriptive",
    ]) 
    with tab4:
        st.subheader("Numerical Statistics")

        numerical_df = get_numerical_descriptive_df(
            df,
            continuous_numeric + discrete_numeric
        )

        st.dataframe(
            numerical_df.style.background_gradient(cmap="Blues"),
            use_container_width=True
        )

        with st.expander("What these metrics mean"):
            st.markdown("""
        - **Mean / Median**: Central tendency  
        - **Std**: Spread of values  
        - **Skewness**: Distribution asymmetry  
        - **Kurtosis**: Tail heaviness  
        - **CV**: Relative variability (std / mean)
        """)

        st.divider()

        # ---------- CATEGORICAL ----------
        st.subheader("Categorical Statistics")

        categorical_df = get_categorical_descriptive_df(
            df,
            categorical_cols
        )

        st.dataframe(categorical_df, use_container_width=True)

        with st.expander("What entropy means"):
            st.markdown("""
        - **Low entropy** → Dominated categories  
        - **High entropy** → Balanced distribution  
        - Useful for encoding decisions
        """)
    with tab5:
        st.subheader("Diagnostic Analysis")
        col1, col2 = st.columns([1.5, 1])

        # ---------- LEFT: HEATMAP ----------
        with col1:

            corr_matrix = get_pearson_corr_matrix(
                df,
                continuous_numeric
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr_matrix,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                ax=ax
            )
            st.subheader("Pearson Correlation Heatmap")
            ax.set_title("Pearson Correlation Matrix")
            st.pyplot(fig)
            st.dataframe(corr_matrix, use_container_width=True)
        # ---------- RIGHT: DIAGNOSTICS TABLE ----------
        with col2:
            st.subheader("Spearman Correlation \n (Monotonic Relationships)")

            spearman_df = get_spearman_correlation_df(
                df,
                continuous_numeric,
                threshold=0.6
            )

            if spearman_df.empty:
                st.success("No strong Spearman correlations detected.")
            else:
                st.dataframe(spearman_df, use_container_width=True)

            st.divider()

            # -------- Kendall --------
            st.subheader("Kendall Correlation \n (Ordinal / Robust)")

            kendall_df = get_kendall_correlation_df(
                df,
                continuous_numeric,
                threshold=0.5
            )

            if kendall_df.empty:
                st.success("No strong Kendall correlations detected.")
            else:
                st.dataframe(kendall_df, use_container_width=True)
        with tab6:
            corr_diag_df = get_numeric_correlation_diagnostics(
                df,
                continuous_numeric,
                threshold=0.8
            )

            st.subheader("Numeric Feature Recommendations")
            num_pres_df = numeric_prescriptive_df(
                df,
                continuous_numeric,
                discrete_numeric
            )
            st.dataframe(num_pres_df, use_container_width=True)

            st.subheader("Categorical Feature Recommendations")
            cat_pres_df = categorical_prescriptive_df(
                df,
                categorical_cols
            )
            st.dataframe(cat_pres_df, use_container_width=True)

            st.subheader("Correlation-Based Feature Removal")
            corr_pres_df = correlation_prescriptive_df(corr_diag_df)
            st.dataframe(corr_pres_df, use_container_width=True)

            st.subheader("Dataset-Level Actions")
            dataset_pres_df = dataset_prescriptive_summary(df)
            st.dataframe(dataset_pres_df, use_container_width=True)
        
    

# ==========================================================
# PAGE 3: ML MODELS
# ==========================================================
elif page == "ML Models":
    if df is None:
        st.info("Upload a dataset to begin")
        st.stop()
    st.subheader("ML Models")

    tab7, tab8, tab9 = st.tabs([
        "Recommendations",
        "Models",
        "Export "
    ])
    with tab7:
        # ------------------------------------
        # TARGET SELECTION
        # ------------------------------------
        target_col = st.selectbox(
            "Select Target Variable",
            options=df.columns
        )

        # ------------------------------------
        # COMPUTE TRAINING PLAN
        # ------------------------------------
        target_type, feature_df, model_df = get_training_plan(
            df=df,
            target_col=target_col,
            continuous_numeric=continuous_numeric,
            discrete_numeric=discrete_numeric,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            dropped_corr_features=(
                corr_pres_df["Feature to Drop"].tolist()
                if "corr_pres_df" in globals() and not corr_pres_df.empty
                else []
            )
        )

        # ------------------------------------
        # TARGET TYPE SUMMARY
        # ------------------------------------
        st.write("Selected Target: ", target_col)

        st.divider()

        # ------------------------------------
        # MODEL-READY FEATURES
        # ------------------------------------
        st.subheader("Model-Ready Features")

        if feature_df.empty:
            st.warning("No suitable features found for training.")
        else:
            st.dataframe(
                feature_df,
                use_container_width=True
            )

        with st.expander("How features were selected"):
            st.markdown("""
        - Target column excluded
        - Datetime columns excluded (unless engineered)
        - Constant columns removed
        - Highly correlated features removed
        - Only ML-safe columns retained
        """)

        st.divider()

        # ------------------------------------
        # MODEL RECOMMENDATIONS
        # ------------------------------------
        st.subheader("Recommended Models")

        if model_df.empty:
            st.warning("No model recommendations available.")
        else:
            st.dataframe(
                model_df,
                use_container_width=True
            )

        with st.expander("Why these models?"):
            st.markdown("""
        Model recommendations are based on:
        - Target variable type
        - Dataset size assumptions
        - Robustness to noise
        - Industry best practices
        """)
    with tab8:
        st.subheader("Model Training & Evaluation")
        
        if st.button("🚀 Train Baseline Model"):
            result = train_baseline_model(
                df=df,
                target_col=target_col,
                feature_df=feature_df,
                target_type=target_type
            )

            st.subheader("📊 Model Metrics")
            st.dataframe(result["metrics_df"], use_container_width=True)

            st.subheader("📈 Train vs Test Performance")
            st.pyplot(result["performance_fig"])

            if result["confusion_matrix_fig"] is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Confusion Matrix")
                    st.pyplot(result["confusion_matrix_fig"])

                with col2:
                    st.subheader("Classification Report")
                    st.dataframe(
                        result["classification_report_df"],
                        use_container_width=True
                    )
        
        # Hyperparameter Configuration Section
        st.divider()
        st.subheader("Hyperparameter Configuration")
        
        hyperparams = {}
        
        if target_type == "Regression":
            st.markdown("**Linear Regression Hyperparameters**")
            col1, col2, col3 = st.columns(3)
            with col1:
                fit_intercept = st.checkbox("Fit Intercept", value=True, key="fit_intercept")
                learning_rate = st.number_input("Learning Rate (α)", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f", key="learning_rate")
                max_iter = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000, step=100, key="reg_max_iter")
            with col2:
                regularization = st.selectbox("Regularization Type", options=["None", "Ridge (L2)", "Lasso (L1)", "ElasticNet"], index=0, key="regularization")
                alpha = st.number_input("Regularization Strength (λ)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="alpha")
                l1_ratio = st.number_input("L1 Ratio (ElasticNet)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="l1_ratio")
            with col3:
                polynomial_degree = st.number_input("Polynomial Degree", min_value=1, max_value=5, value=1, step=1, key="poly_degree")
                batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=32, step=16, key="batch_size")
                feature_scaling = st.selectbox("Feature Scaling", options=["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0, key="feature_scaling")
            
            hyperparams = {
                "fit_intercept": fit_intercept,
                "learning_rate": learning_rate,
                "max_iter": max_iter,
                "regularization": regularization,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "polynomial_degree": polynomial_degree,
                "batch_size": batch_size,
                "feature_scaling": feature_scaling
            }
            
        elif target_type == "Binary Classification":
            st.markdown("**Logistic Regression Hyperparameters**")
            col1, col2, col3 = st.columns(3)
            with col1:
                max_iter = st.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100, key="max_iter")
                C = st.number_input("Regularization Strength (C)", min_value=0.001, max_value=100.0, value=1.0, step=0.1, key="C")
                learning_rate_param = st.selectbox("Learning Rate", options=["constant", "optimal", "invscaling", "adaptive"], index=1, key="lr_param")
            with col2:
                solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0, key="solver")
                penalty = st.selectbox("Penalty", options=["l2", "l1", "elasticnet", "none"], index=0, key="penalty")
                l1_ratio = st.number_input("L1 Ratio (ElasticNet)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="log_l1_ratio")
            with col3:
                fit_intercept = st.checkbox("Fit Intercept", value=True, key="log_fit_intercept")
                batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=32, step=16, key="log_batch_size")
                feature_scaling = st.selectbox("Feature Scaling", options=["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0, key="log_feature_scaling")
            
            hyperparams = {
                "max_iter": max_iter,
                "C": C,
                "solver": solver,
                "penalty": penalty,
                "l1_ratio": l1_ratio,
                "fit_intercept": fit_intercept,
                "learning_rate": learning_rate_param,
                "batch_size": batch_size,
                "feature_scaling": feature_scaling
            }
            
        else:  # Multiclass Classification
            st.markdown("**Random Forest Classifier Hyperparameters**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.number_input("Number of Trees", min_value=10, max_value=500, value=100, step=10, key="n_estimators")
                max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10, step=1, key="max_depth")
                min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1, key="min_samples_split")
            with col2:
                min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1, key="min_samples_leaf")
                max_features = st.selectbox("Max Features", options=["sqrt", "log2", None], index=0, key="max_features")
                bootstrap = st.checkbox("Bootstrap", value=True, key="bootstrap")
            with col3:
                criterion = st.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0, key="criterion")
                fit_intercept = st.checkbox("Fit Intercept", value=True, key="rf_fit_intercept")
                feature_scaling = st.selectbox("Feature Scaling", options=["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0, key="rf_feature_scaling")
            
            hyperparams = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "bootstrap": bootstrap,
                "criterion": criterion,
                "fit_intercept": fit_intercept,
                "feature_scaling": feature_scaling
            }
        
        if st.button("⚙️ Train with Custom Hyperparameters"):
            result_custom = train_baseline_model(
                df=df,
                target_col=target_col,
                feature_df=feature_df,
                target_type=target_type,
                hyperparams=hyperparams
            )

            st.subheader("📊 Custom Model Metrics")
            st.dataframe(result_custom["metrics_df"], use_container_width=True)

            st.subheader("📈 Train vs Test Performance (Custom)")
            st.pyplot(result_custom["performance_fig"])

            if result_custom["confusion_matrix_fig"] is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Confusion Matrix")
                    st.pyplot(result_custom["confusion_matrix_fig"])

                with col2:
                    st.subheader("Classification Report")
                    st.dataframe(
                        result_custom["classification_report_df"],
                        use_container_width=True
                    )
# ==========================================================
# PAGE 4: CHATBOT
# ==========================================================
elif page == "AI Assistant":
    st.title("🤖 Dataset AI Assistant")
    
    # if df is None:
    #     st.info("📁 Please upload a dataset to start chatting!")
    #     st.stop()
    
    # # Import the chatbot
    # from chatbot import DatasetChatbot
    
    # # Initialize chatbot in session state
    # if 'chatbot' not in st.session_state or st.session_state.get('chatbot_df_hash') != hash(df.to_string()):
    #     st.session_state.chatbot = DatasetChatbot(df)
    #     st.session_state.chatbot_df_hash = hash(df.to_string())
    #     st.session_state.chat_history = []
    
    # chatbot = st.session_state.chatbot
    
    # # Sidebar info
    # with st.sidebar:
    #     st.divider()
    #     st.subheader("📊 Dataset Info")
    #     st.write(f"**Rows:** {df.shape[0]:,}")
    #     st.write(f"**Columns:** {df.shape[1]}")
    #     st.write(f"**Numeric:** {len(chatbot.numeric_cols)}")
    #     st.write(f"**Categorical:** {len(chatbot.categorical_cols)}")
    
    # # Main chat interface
    # st.markdown("""
    # <div style="
    #     background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    #     padding: 20px;
    #     border-radius: 10px;
    #     margin-bottom: 20px;
    # ">
    #     <h4 style="color: #e94560; margin: 0;">💬 Ask me anything about your dataset!</h4>
    #     <p style="color: #a0a0a0; margin: 5px 0 0 0;">
    #         I can answer questions about statistics, missing values, correlations, distributions, and more.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # Suggested questions
    # st.subheader("💡 Suggested Questions")
    # suggestions = chatbot.get_suggested_questions()
    
    # # Display suggestions in columns
    # cols = st.columns(3)
    # for i, suggestion in enumerate(suggestions[:9]):
    #     with cols[i % 3]:
    #         if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
    #             # Process immediately without rerun
    #             st.session_state.chat_history.append({"role": "user", "content": suggestion})
    #             response = chatbot.chat(suggestion)
    #             st.session_state.chat_history.append({"role": "assistant", "content": response})
    #             st.rerun()
    
    # st.divider()
    
    # # Chat history display
    # st.subheader("💬 Conversation")
    
    # # Create a container for chat messages
    # chat_container = st.container()
    
    # with chat_container:
    #     for message in st.session_state.chat_history:
    #         if message["role"] == "user":
    #             st.markdown(f"""
    #             <div style="
    #                 background-color: #2d3748;
    #                 padding: 10px 15px;
    #                 border-radius: 10px;
    #                 margin: 5px 0;
    #                 border-left: 4px solid #4299e1;
    #             ">
    #                 <strong>🧑 You:</strong> {message["content"]}
    #             </div>
    #             """, unsafe_allow_html=True)
    #         else:
    #             st.markdown(f"""
    #             <div style="
    #                 background-color: #1a202c;
    #                 padding: 10px 15px;
    #                 border-radius: 10px;
    #                 margin: 5px 0;
    #                 border-left: 4px solid #48bb78;
    #             ">
    #                 <strong>🤖 Assistant:</strong>
    #             </div>
    #             """, unsafe_allow_html=True)
    #             st.markdown(message["content"])
    
    # # Chat input using a form to prevent infinite loop
    # st.divider()
    
    # with st.form(key="chat_form", clear_on_submit=True):
    #     user_input = st.text_input(
    #         "Ask a question about your dataset:",
    #         placeholder="e.g., What is the average value of Sales column?",
    #         key="chat_input",
    #         label_visibility="collapsed"
    #     )
    #     submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        
    #     if submit_button and user_input:
    #         # Add user message to history
    #         st.session_state.chat_history.append({"role": "user", "content": user_input})
            
    #         # Get chatbot response
    #         response = chatbot.chat(user_input)
            
    #         # Add assistant response to history
    #         st.session_state.chat_history.append({"role": "assistant", "content": response})
            
    #         # Rerun to update the chat display
    #         st.rerun()
    
    # # Clear chat button
    # st.divider()
    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col2:
    #     if st.button("🗑️ Clear Chat History", use_container_width=True):
    #         st.session_state.chat_history = []
    #         st.session_state.chatbot = DatasetChatbot(df)
    #         st.rerun()
    
    # # Quick stats at the bottom
    # with st.expander("📊 Quick Dataset Overview"):
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         st.metric("Total Rows", f"{df.shape[0]:,}")
    #     with col2:
    #         st.metric("Total Columns", df.shape[1])
    #     with col3:
    #         missing_pct = round(df.isna().mean().mean() * 100, 2)
    #         st.metric("Missing Data", f"{missing_pct}%")
    #     with col4:
    #         duplicates = df.duplicated().sum()
    #         st.metric("Duplicates", f"{duplicates:,}")
