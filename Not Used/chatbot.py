"""
Dataset-Aware Chatbot Module
============================
A local chatbot that can answer questions about a pandas DataFrame
without using any external API. Uses rule-based NLP and pattern matching.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DatasetChatbot:
    """
    A rule-based chatbot that understands and answers questions about a DataFrame.
    No external API required - all processing is done locally.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the chatbot with a DataFrame.
        
        Args:
            df: The pandas DataFrame to analyze
        """
        self.df = df
        self.column_names = df.columns.tolist()
        self.column_names_lower = [col.lower() for col in self.column_names]
        self.column_map = {col.lower(): col for col in self.column_names}
        
        # Precompute column metadata
        self._analyze_columns()
        
        # Define intent patterns
        self._setup_intent_patterns()
        
        # Conversation history
        self.conversation_history = []
    
    def _analyze_columns(self):
        """Analyze and categorize all columns in the DataFrame."""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = []
        
        # Detect datetime columns
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_cols.append(col)
            elif self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col].dropna().head(100))
                    self.datetime_cols.append(col)
                except:
                    pass
        
        # Column statistics cache
        self.column_stats = {}
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                self.column_stats[col] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'sum': series.sum(),
                    'count': len(series),
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75)
                }
    
    def _setup_intent_patterns(self):
        """Define regex patterns for intent detection."""
        self.intent_patterns = {
            # Dataset overview
            'dataset_shape': r'\b(how many|number of|count|total)\s*(rows?|records?|entries|columns?|features?|observations?)\b',
            'dataset_info': r'\b(tell me about|describe|overview|summary of|info about)\s*(the\s*)?(dataset|data|dataframe)\b',
            'column_list': r'\b(what|which|list|show|display)\s*(are\s*)?(the\s*)?(columns?|features?|variables?|fields?)\b',
            'column_types': r'\b(what|which)\s*(are\s*)?(the\s*)?(data\s*)?types?\s*(of|for)?\s*(columns?|features?)?\b',
            
            # Statistical queries
            'mean': r'\b(mean|average|avg)\s*(of|for|value)?\s*',
            'median': r'\b(median)\s*(of|for|value)?\s*',
            'sum': r'\b(sum|total)\s*(of|for|value)?\s*',
            'min': r'\b(min|minimum|smallest|lowest)\s*(of|for|value)?\s*',
            'max': r'\b(max|maximum|largest|highest|biggest)\s*(of|for|value)?\s*',
            'std': r'\b(std|standard\s*deviation|deviation)\s*(of|for)?\s*',
            'range': r'\b(range)\s*(of|for)?\s*',
            'statistics': r'\b(statistics?|stats?|describe)\s*(of|for|about)?\s*',
            
            # Missing data
            'missing_values': r'\b(missing|null|nan|empty|na)\s*(values?|data|count|percentage)?\b',
            'null_columns': r'\b(which|what)\s*(columns?|features?)\s*(have|has|contain)\s*(missing|null|nan)\b',
            
            # Unique values
            'unique_values': r'\b(unique|distinct|different)\s*(values?|count|entries)?\s*(of|for|in)?\s*',
            'value_counts': r'\b(value\s*counts?|frequency|distribution|count\s*of\s*each|breakdown)\s*(of|for|in)?\s*',
            'categories': r'\b(what|which|list|show)\s*(are\s*)?(the\s*)?(categories|classes|groups|types)\s*(of|for|in)?\s*',
            
            # Correlation
            'correlation': r'\b(correlat|relationship|related|connection)\w*\s*(between|of|for|with)?\s*',
            
            # Filtering/Querying
            'filter': r'\b(filter|where|find|show|get|select)\s*(rows?|records?|entries|data)?\s*(where|with|having|when|if)?\s*',
            'top_n': r'\b(top|first|highest|largest)\s*(\d+)\s*',
            'bottom_n': r'\b(bottom|last|lowest|smallest)\s*(\d+)\s*',
            
            # Specific column queries
            'column_info': r'\b(tell me about|describe|info about|what is|explain)\s*(the\s*)?(column|feature|variable)?\s*',
            
            # Comparisons
            'compare': r'\b(compare|comparison|difference|vs|versus)\s*(between)?\s*',
            
            # Outliers
            'outliers': r'\b(outliers?|anomal\w*|unusual|extreme)\s*(in|of|for|values?)?\s*',
            
            # Distribution
            'distribution': r'\b(distribution|spread|dispersion)\s*(of|for|in)?\s*',
            
            # Data quality
            'duplicates': r'\b(duplicate|duplicated|repeated|duplicates?)\b',
            'quality': r'\b(data\s*)?quality\s*(of|for|report|check)?\b',
            
            # Greeting
            'greeting': r'\b(hi|hello|hey|greetings|good\s*(morning|afternoon|evening))\b',
            'thanks': r'\b(thanks?|thank\s*you|thx|cheers)\b',
            'help': r'\b(help|assist|what\s*can\s*you\s*do|capabilities|commands?)\b',
            
            # Sample data
            'sample': r'\b(sample|example|preview|show\s*me|head|tail|first\s*few|last\s*few)\s*(of\s*)?(the\s*)?(data|rows?|records?)?\b',
        }
    
    def _find_column_in_query(self, query: str) -> List[str]:
        """
        Extract column names mentioned in the query.
        Uses fuzzy matching to handle variations.
        """
        query_lower = query.lower()
        found_columns = []
        
        # Direct match
        for col_lower, col_original in self.column_map.items():
            # Check for exact match or word boundary match
            patterns = [
                rf'\b{re.escape(col_lower)}\b',
                rf"'{re.escape(col_lower)}'",
                rf'"{re.escape(col_lower)}"',
                rf'`{re.escape(col_lower)}`',
            ]
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    found_columns.append(col_original)
                    break
        
        # If no direct match, try partial matching
        if not found_columns:
            for col_lower, col_original in self.column_map.items():
                # Split column name into words
                col_words = re.split(r'[_\s-]', col_lower)
                for word in col_words:
                    if len(word) > 3 and word in query_lower:
                        found_columns.append(col_original)
                        break
        
        return list(set(found_columns))
    
    def _detect_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the intent of the user's query.
        
        Returns:
            Tuple of (intent_name, extracted_entities)
        """
        query_lower = query.lower()
        
        # Define priority order - more specific intents should come first
        priority_intents = [
            'greeting', 'help', 'thanks',
            'outliers', 'duplicates', 'quality',
            'correlation', 'missing_values', 'null_columns',
            'dataset_shape', 'dataset_info', 'column_list', 'column_types',
            'mean', 'median', 'sum', 'min', 'max', 'std', 'range', 'statistics',
            'unique_values', 'value_counts', 'categories',
            'distribution',
            'top_n', 'bottom_n',
            'sample',
            'filter', 'compare', 'column_info'
        ]
        
        # First pass: check priority intents in order
        matched_intents = []
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                matched_intents.append(intent)
        
        # Return the highest priority matched intent
        for intent in priority_intents:
            if intent in matched_intents:
                entities = {
                    'columns': self._find_column_in_query(query),
                    'numbers': re.findall(r'\d+(?:\.\d+)?', query),
                }
                return intent, entities
        
        # If any intent matched but wasn't in priority list
        if matched_intents:
            entities = {
                'columns': self._find_column_in_query(query),
                'numbers': re.findall(r'\d+(?:\.\d+)?', query),
            }
            return matched_intents[0], entities
        
        # Check if it's a column-specific question
        columns_found = self._find_column_in_query(query)
        if columns_found:
            return 'column_info', {'columns': columns_found, 'numbers': []}
        
        return 'unknown', {'columns': [], 'numbers': []}
    
    def _format_number(self, num: float, decimals: int = 2) -> str:
        """Format a number for display."""
        if pd.isna(num):
            return "N/A"
        if abs(num) >= 1e6:
            return f"{num:,.0f}"
        elif abs(num) >= 1000:
            return f"{num:,.2f}"
        else:
            return f"{round(num, decimals)}"
    
    def _get_dataset_shape(self) -> str:
        """Get dataset shape information."""
        rows, cols = self.df.shape
        return f"📊 **Dataset Shape:**\n- **Rows:** {rows:,}\n- **Columns:** {cols}"
    
    def _get_dataset_info(self) -> str:
        """Get comprehensive dataset overview."""
        rows, cols = self.df.shape
        memory = self.df.memory_usage(deep=True).sum() / 1024**2
        missing_pct = (self.df.isna().sum().sum() / (rows * cols)) * 100
        duplicates = self.df.duplicated().sum()
        
        response = f"""📊 **Dataset Overview**

**Basic Info:**
- Total Rows: {rows:,}
- Total Columns: {cols}
- Memory Usage: {memory:.2f} MB

**Column Types:**
- Numeric Columns: {len(self.numeric_cols)}
- Categorical Columns: {len(self.categorical_cols)}
- DateTime Columns: {len(self.datetime_cols)}

**Data Quality:**
- Missing Values: {missing_pct:.2f}%
- Duplicate Rows: {duplicates:,}
"""
        return response
    
    def _get_column_list(self) -> str:
        """List all columns with their types."""
        lines = ["📋 **Columns in Dataset:**\n"]
        for i, col in enumerate(self.column_names, 1):
            dtype = str(self.df[col].dtype)
            if col in self.numeric_cols:
                type_icon = "🔢"
            elif col in self.datetime_cols:
                type_icon = "📅"
            else:
                type_icon = "📝"
            lines.append(f"{i}. {type_icon} **{col}** ({dtype})")
        return "\n".join(lines)
    
    def _get_column_types(self) -> str:
        """Get column types summary."""
        lines = ["📊 **Column Data Types:**\n"]
        
        type_counts = self.df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            lines.append(f"- **{dtype}**: {count} column(s)")
        
        lines.append("\n**Detailed Breakdown:**")
        for col in self.column_names:
            lines.append(f"- {col}: `{self.df[col].dtype}`")
        
        return "\n".join(lines)
    
    def _get_statistics(self, columns: List[str] = None) -> str:
        """Get statistics for specified columns or all numeric columns."""
        if not columns:
            columns = self.numeric_cols[:5]  # Limit to first 5 for readability
        
        if not columns:
            return "❌ No numeric columns found in the dataset."
        
        lines = ["📈 **Statistical Summary:**\n"]
        
        for col in columns:
            if col in self.numeric_cols:
                stats = self.column_stats.get(col, {})
                if stats:
                    lines.append(f"**{col}:**")
                    lines.append(f"  - Mean: {self._format_number(stats['mean'])}")
                    lines.append(f"  - Median: {self._format_number(stats['median'])}")
                    lines.append(f"  - Std Dev: {self._format_number(stats['std'])}")
                    lines.append(f"  - Min: {self._format_number(stats['min'])}")
                    lines.append(f"  - Max: {self._format_number(stats['max'])}")
                    lines.append("")
        
        return "\n".join(lines) if len(lines) > 1 else f"No statistics available for: {columns}"
    
    def _get_mean(self, columns: List[str]) -> str:
        """Get mean of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Mean Values:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                mean_val = self.df[col].mean()
                lines.append(f"- **{col}**: {self._format_number(mean_val)}")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No numeric columns specified."
    
    def _get_median(self, columns: List[str]) -> str:
        """Get median of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Median Values:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                median_val = self.df[col].median()
                lines.append(f"- **{col}**: {self._format_number(median_val)}")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No numeric columns specified."
    
    def _get_sum(self, columns: List[str]) -> str:
        """Get sum of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Sum Values:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                sum_val = self.df[col].sum()
                lines.append(f"- **{col}**: {self._format_number(sum_val)}")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No numeric columns specified."
    
    def _get_min(self, columns: List[str]) -> str:
        """Get minimum of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Minimum Values:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                min_val = self.df[col].min()
                lines.append(f"- **{col}**: {self._format_number(min_val)}")
            elif col in self.categorical_cols:
                lines.append(f"- **{col}**: '{self.df[col].min()}'")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No columns specified."
    
    def _get_max(self, columns: List[str]) -> str:
        """Get maximum of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Maximum Values:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                max_val = self.df[col].max()
                lines.append(f"- **{col}**: {self._format_number(max_val)}")
            elif col in self.categorical_cols:
                lines.append(f"- **{col}**: '{self.df[col].max()}'")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No columns specified."
    
    def _get_std(self, columns: List[str]) -> str:
        """Get standard deviation of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Standard Deviation:**\n"]
        for col in columns:
            if col in self.numeric_cols:
                std_val = self.df[col].std()
                lines.append(f"- **{col}**: {self._format_number(std_val)}")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No numeric columns specified."
    
    def _get_range(self, columns: List[str]) -> str:
        """Get range (max - min) of specified columns."""
        if not columns:
            columns = self.numeric_cols
        
        lines = ["📊 **Range Values (Max - Min):**\n"]
        for col in columns:
            if col in self.numeric_cols:
                range_val = self.df[col].max() - self.df[col].min()
                lines.append(f"- **{col}**: {self._format_number(range_val)} "
                           f"(from {self._format_number(self.df[col].min())} to "
                           f"{self._format_number(self.df[col].max())})")
        
        return "\n".join(lines) if len(lines) > 1 else "❌ No numeric columns specified."
    
    def _get_missing_values(self, columns: List[str] = None) -> str:
        """Get missing value information."""
        if columns:
            subset = columns
        else:
            subset = self.column_names
        
        lines = ["🔍 **Missing Values Analysis:**\n"]
        total_missing = 0
        
        for col in subset:
            if col in self.column_names:
                missing = self.df[col].isna().sum()
                total = len(self.df)
                pct = (missing / total) * 100
                total_missing += missing
                if missing > 0:
                    lines.append(f"- **{col}**: {missing:,} ({pct:.2f}%)")
        
        if len(lines) == 1:
            lines.append("✅ No missing values found!")
        else:
            lines.insert(1, f"**Total Missing Values:** {total_missing:,}\n")
        
        return "\n".join(lines)
    
    def _get_unique_values(self, columns: List[str]) -> str:
        """Get unique value counts."""
        if not columns:
            columns = self.column_names[:10]  # Limit
        
        lines = ["🔢 **Unique Values Count:**\n"]
        for col in columns:
            if col in self.column_names:
                unique_count = self.df[col].nunique()
                total = len(self.df)
                lines.append(f"- **{col}**: {unique_count:,} unique values "
                           f"({(unique_count/total)*100:.1f}% of total)")
        
        return "\n".join(lines)
    
    def _get_value_counts(self, columns: List[str]) -> str:
        """Get value counts for specified columns."""
        if not columns:
            columns = self.categorical_cols[:1] if self.categorical_cols else self.column_names[:1]
        
        lines = ["📊 **Value Counts:**\n"]
        
        for col in columns:
            if col in self.column_names:
                vc = self.df[col].value_counts().head(10)
                lines.append(f"**{col}:**")
                for value, count in vc.items():
                    pct = (count / len(self.df)) * 100
                    lines.append(f"  - {value}: {count:,} ({pct:.1f}%)")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_correlation(self, columns: List[str]) -> str:
        """Get correlation between columns."""
        if len(columns) < 2:
            # Show top correlations
            if len(self.numeric_cols) >= 2:
                corr_matrix = self.df[self.numeric_cols].corr()
                
                # Get top correlations
                corr_pairs = []
                for i in range(len(self.numeric_cols)):
                    for j in range(i+1, len(self.numeric_cols)):
                        col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val):
                            corr_pairs.append((col1, col2, corr_val))
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                lines = ["🔗 **Top Correlations:**\n"]
                for col1, col2, corr in corr_pairs[:10]:
                    strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                    direction = "positive" if corr > 0 else "negative"
                    lines.append(f"- **{col1}** ↔ **{col2}**: {corr:.3f} ({strength} {direction})")
                
                return "\n".join(lines)
            else:
                return "❌ Need at least 2 numeric columns for correlation analysis."
        
        # Check if both columns are numeric
        valid_cols = [c for c in columns if c in self.numeric_cols]
        if len(valid_cols) < 2:
            return "❌ Need at least 2 numeric columns for correlation."
        
        col1, col2 = valid_cols[0], valid_cols[1]
        corr = self.df[col1].corr(self.df[col2])
        
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        
        return f"""🔗 **Correlation Analysis:**

**{col1}** ↔ **{col2}**
- Correlation Coefficient: {corr:.4f}
- Strength: {strength}
- Direction: {direction}

**Interpretation:** {'These columns are highly related.' if abs(corr) > 0.7 else 'These columns have some relationship.' if abs(corr) > 0.4 else 'These columns are weakly related.'}"""
    
    def _get_column_info(self, columns: List[str]) -> str:
        """Get detailed information about specific columns."""
        if not columns:
            return "❓ Please specify which column you want to know about."
        
        lines = []
        for col in columns:
            if col not in self.column_names:
                lines.append(f"❌ Column '{col}' not found in dataset.")
                continue
            
            series = self.df[col]
            dtype = str(series.dtype)
            unique = series.nunique()
            missing = series.isna().sum()
            missing_pct = (missing / len(self.df)) * 100
            
            lines.append(f"📋 **Column: {col}**\n")
            lines.append(f"- **Data Type:** {dtype}")
            lines.append(f"- **Unique Values:** {unique:,}")
            lines.append(f"- **Missing Values:** {missing:,} ({missing_pct:.1f}%)")
            
            if col in self.numeric_cols:
                stats = self.column_stats.get(col, {})
                if stats:
                    lines.append(f"\n**Statistics:**")
                    lines.append(f"- Mean: {self._format_number(stats['mean'])}")
                    lines.append(f"- Median: {self._format_number(stats['median'])}")
                    lines.append(f"- Std Dev: {self._format_number(stats['std'])}")
                    lines.append(f"- Min: {self._format_number(stats['min'])}")
                    lines.append(f"- Max: {self._format_number(stats['max'])}")
                    lines.append(f"- 25th Percentile: {self._format_number(stats['q1'])}")
                    lines.append(f"- 75th Percentile: {self._format_number(stats['q3'])}")
            else:
                # Categorical column
                vc = series.value_counts().head(5)
                lines.append(f"\n**Top Values:**")
                for val, count in vc.items():
                    lines.append(f"- {val}: {count:,}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_outliers(self, columns: List[str]) -> str:
        """Detect outliers using IQR method."""
        if not columns:
            columns = self.numeric_cols[:5]
        
        lines = ["🔍 **Outlier Analysis (IQR Method):**\n"]
        
        for col in columns:
            if col in self.numeric_cols:
                series = self.df[col].dropna()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                outliers = series[(series < lower) | (series > upper)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(series)) * 100
                
                lines.append(f"**{col}:**")
                lines.append(f"  - Outliers: {outlier_count:,} ({outlier_pct:.1f}%)")
                lines.append(f"  - Lower Bound: {self._format_number(lower)}")
                lines.append(f"  - Upper Bound: {self._format_number(upper)}")
                if outlier_count > 0:
                    lines.append(f"  - Range of Outliers: {self._format_number(outliers.min())} to {self._format_number(outliers.max())}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_distribution(self, columns: List[str]) -> str:
        """Get distribution information for columns."""
        if not columns:
            columns = self.numeric_cols[:3]
        
        lines = ["📊 **Distribution Analysis:**\n"]
        
        for col in columns:
            if col in self.numeric_cols:
                series = self.df[col].dropna()
                skew = series.skew()
                kurt = series.kurtosis()
                
                # Interpret skewness
                if skew > 1:
                    skew_desc = "Highly right-skewed (positive)"
                elif skew > 0.5:
                    skew_desc = "Moderately right-skewed"
                elif skew < -1:
                    skew_desc = "Highly left-skewed (negative)"
                elif skew < -0.5:
                    skew_desc = "Moderately left-skewed"
                else:
                    skew_desc = "Approximately symmetric"
                
                lines.append(f"**{col}:**")
                lines.append(f"  - Skewness: {skew:.3f} ({skew_desc})")
                lines.append(f"  - Kurtosis: {kurt:.3f}")
                lines.append("")
            elif col in self.categorical_cols:
                vc = self.df[col].value_counts()
                lines.append(f"**{col}:** {len(vc)} categories")
                lines.append(f"  - Most common: {vc.index[0]} ({vc.iloc[0]:,})")
                if len(vc) > 1:
                    lines.append(f"  - Least common: {vc.index[-1]} ({vc.iloc[-1]:,})")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_duplicates(self) -> str:
        """Get duplicate row information."""
        dup_count = self.df.duplicated().sum()
        total = len(self.df)
        dup_pct = (dup_count / total) * 100
        
        response = f"""📋 **Duplicate Analysis:**

- **Duplicate Rows:** {dup_count:,}
- **Percentage:** {dup_pct:.2f}%
- **Unique Rows:** {total - dup_count:,}
"""
        
        if dup_count > 0:
            # Find which columns have duplicates
            response += "\n**Columns with Most Duplicates:**\n"
            for col in self.column_names[:10]:
                col_dups = self.df[col].duplicated().sum()
                if col_dups > 0:
                    response += f"- {col}: {col_dups:,} duplicates\n"
        
        return response
    
    def _get_quality(self) -> str:
        """Get data quality report."""
        total_cells = self.df.size
        missing_cells = self.df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        duplicate_rows = self.df.duplicated().sum()
        
        # Calculate overall quality score
        quality_score = 100 - missing_pct - (duplicate_rows / len(self.df) * 100)
        quality_score = max(0, min(100, quality_score))
        
        grade = "Excellent" if quality_score >= 90 else "Good" if quality_score >= 70 else "Fair" if quality_score >= 50 else "Poor"
        
        return f"""📊 **Data Quality Report**

**Overall Quality Score:** {quality_score:.1f}% ({grade})

**Details:**
- Total Cells: {total_cells:,}
- Missing Values: {missing_cells:,} ({missing_pct:.2f}%)
- Duplicate Rows: {duplicate_rows:,}
- Complete Rows: {len(self.df.dropna()):,}

**Column Completeness:**
{self._get_column_completeness()}
"""
    
    def _get_column_completeness(self) -> str:
        """Get column completeness report."""
        lines = []
        for col in self.column_names:
            completeness = (1 - self.df[col].isna().mean()) * 100
            bar = "█" * int(completeness // 10) + "░" * (10 - int(completeness // 10))
            lines.append(f"  {col}: [{bar}] {completeness:.0f}%")
        return "\n".join(lines[:10])  # Limit to 10 columns
    
    def _get_sample(self, n: int = 5, position: str = "head") -> str:
        """Get sample rows from the dataset."""
        if position == "tail":
            sample = self.df.tail(n)
        else:
            sample = self.df.head(n)
        
        return f"📋 **Sample Data ({position.title()} {n} rows):**\n\n```\n{sample.to_string()}\n```"
    
    def _get_top_n(self, n: int, columns: List[str]) -> str:
        """Get top N values for a column."""
        if not columns:
            return "❓ Please specify which column to get top values for."
        
        col = columns[0]
        if col not in self.column_names:
            return f"❌ Column '{col}' not found."
        
        if col in self.numeric_cols:
            top_vals = self.df.nlargest(n, col)[[col]]
            return f"📊 **Top {n} values in {col}:**\n\n```\n{top_vals.to_string()}\n```"
        else:
            vc = self.df[col].value_counts().head(n)
            lines = [f"📊 **Top {n} values in {col}:**\n"]
            for val, count in vc.items():
                lines.append(f"- {val}: {count:,}")
            return "\n".join(lines)
    
    def _get_bottom_n(self, n: int, columns: List[str]) -> str:
        """Get bottom N values for a column."""
        if not columns:
            return "❓ Please specify which column to get bottom values for."
        
        col = columns[0]
        if col not in self.column_names:
            return f"❌ Column '{col}' not found."
        
        if col in self.numeric_cols:
            bottom_vals = self.df.nsmallest(n, col)[[col]]
            return f"📊 **Bottom {n} values in {col}:**\n\n```\n{bottom_vals.to_string()}\n```"
        else:
            vc = self.df[col].value_counts().tail(n)
            lines = [f"📊 **Bottom {n} values in {col}:**\n"]
            for val, count in vc.items():
                lines.append(f"- {val}: {count:,}")
            return "\n".join(lines)
    
    def _filter_data(self, query: str) -> str:
        """Parse and execute filter queries."""
        # Try to extract filter conditions
        columns = self._find_column_in_query(query)
        
        # Common filter patterns
        patterns = [
            (r'(\w+)\s*[>=]\s*(\d+(?:\.\d+)?)', 'gte'),
            (r'(\w+)\s*[<=]\s*(\d+(?:\.\d+)?)', 'lte'),
            (r'(\w+)\s*[>]\s*(\d+(?:\.\d+)?)', 'gt'),
            (r'(\w+)\s*[<]\s*(\d+(?:\.\d+)?)', 'lt'),
            (r'(\w+)\s*[=]\s*(\d+(?:\.\d+)?)', 'eq'),
            (r'(\w+)\s*equals?\s*["\']?([^"\']+)["\']?', 'eq_str'),
        ]
        
        for pattern, op in patterns:
            match = re.search(pattern, query)
            if match:
                col_name = match.group(1)
                value = match.group(2)
                
                # Find matching column
                actual_col = None
                for c in self.column_names:
                    if col_name.lower() in c.lower():
                        actual_col = c
                        break
                
                if actual_col:
                    try:
                        if op in ['gte', 'lte', 'gt', 'lt', 'eq']:
                            value = float(value)
                            if op == 'gte':
                                result = self.df[self.df[actual_col] >= value]
                            elif op == 'lte':
                                result = self.df[self.df[actual_col] <= value]
                            elif op == 'gt':
                                result = self.df[self.df[actual_col] > value]
                            elif op == 'lt':
                                result = self.df[self.df[actual_col] < value]
                            else:
                                result = self.df[self.df[actual_col] == value]
                        else:
                            result = self.df[self.df[actual_col].astype(str).str.lower() == value.lower()]
                        
                        n_results = len(result)
                        if n_results == 0:
                            return f"❌ No rows match the filter condition."
                        
                        preview = result.head(10)
                        return f"""📋 **Filter Results:**

Found **{n_results:,}** rows matching the condition.

**Preview (first 10 rows):**
```
{preview.to_string()}
```"""
                    except Exception as e:
                        return f"❌ Error applying filter: {str(e)}"
        
        return """❓ **Filter Query Help**

You can filter data using patterns like:
- "Show rows where **column** > 100"
- "Find records where **column** = 'value'"
- "Get data where **column** >= 50"

Please specify the column name and condition."""
    
    def _get_greeting(self) -> str:
        """Return a greeting response."""
        return """👋 **Hello!** I'm your Dataset Assistant.

I can help you explore and understand your data. Here are some things you can ask me:

📊 **Dataset Overview:**
- "Tell me about the dataset"
- "How many rows and columns?"
- "What are the column types?"

📈 **Statistics:**
- "What's the mean of [column]?"
- "Show me statistics for [column]"
- "What's the correlation between [column1] and [column2]?"

🔍 **Data Quality:**
- "Are there missing values?"
- "Check for duplicates"
- "Find outliers in [column]"

📋 **Explore Data:**
- "Show me the top 10 values in [column]"
- "What are the unique values in [column]?"
- "Give me a sample of the data"

Just ask away! 🚀"""
    
    def _get_help(self) -> str:
        """Return help information."""
        return """🤖 **Dataset Chatbot Capabilities**

I can answer questions about your uploaded dataset. Here's what I can do:

**📊 Dataset Information:**
- Dataset shape, size, and memory usage
- List all columns and their data types
- Data quality report

**📈 Statistical Analysis:**
- Mean, median, mode, sum, min, max
- Standard deviation and range
- Correlation between columns
- Distribution analysis
- Outlier detection

**🔍 Data Exploration:**
- Missing value analysis
- Unique value counts
- Value frequency (value_counts)
- Sample data preview
- Top/Bottom N values

**🎯 Data Filtering:**
- Filter rows by conditions
- Compare values

**Example Questions:**
1. "What is the average of Sales?"
2. "How many unique values are in the Category column?"
3. "Show me missing values"
4. "What's the correlation between Price and Quantity?"
5. "Find outliers in the Revenue column"
6. "Show me the top 5 products by sales"

Just type your question naturally! 💬"""
    
    def _get_thanks(self) -> str:
        """Return a thanks response."""
        responses = [
            "You're welcome! 😊 Let me know if you have more questions.",
            "Happy to help! 🎉 Feel free to ask anything else about your data.",
            "Glad I could assist! 💪 What else would you like to know?",
            "No problem! 🌟 I'm here whenever you need help with your dataset."
        ]
        import random
        return random.choice(responses)
    
    def _get_unknown_response(self, query: str) -> str:
        """Handle unknown queries."""
        columns = self._find_column_in_query(query)
        
        response = f"""❓ I'm not sure what you're asking. 

**Your question:** "{query}"
"""
        
        if columns:
            response += f"""
I noticed you mentioned: **{', '.join(columns)}**

Here's what I can tell you about {'these columns' if len(columns) > 1 else 'this column'}:

"""
            response += self._get_column_info(columns)
        else:
            response += """
**Here are some things you can ask:**
- "Tell me about the dataset"
- "What is the mean of [column_name]?"
- "Show me missing values"
- "How many unique values in [column_name]?"

Type **help** to see all my capabilities!"""
        
        return response
    
    def chat(self, query: str) -> str:
        """
        Main chat function - processes user query and returns response.
        
        Args:
            query: User's natural language question
            
        Returns:
            Response string
        """
        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Detect intent
        intent, entities = self._detect_intent(query)
        columns = entities.get('columns', [])
        numbers = entities.get('numbers', [])
        
        # Route to appropriate handler
        response_map = {
            'dataset_shape': lambda: self._get_dataset_shape(),
            'dataset_info': lambda: self._get_dataset_info(),
            'column_list': lambda: self._get_column_list(),
            'column_types': lambda: self._get_column_types(),
            'mean': lambda: self._get_mean(columns),
            'median': lambda: self._get_median(columns),
            'sum': lambda: self._get_sum(columns),
            'min': lambda: self._get_min(columns),
            'max': lambda: self._get_max(columns),
            'std': lambda: self._get_std(columns),
            'range': lambda: self._get_range(columns),
            'statistics': lambda: self._get_statistics(columns),
            'missing_values': lambda: self._get_missing_values(columns),
            'null_columns': lambda: self._get_missing_values(columns),
            'unique_values': lambda: self._get_unique_values(columns),
            'value_counts': lambda: self._get_value_counts(columns),
            'categories': lambda: self._get_value_counts(columns),
            'correlation': lambda: self._get_correlation(columns),
            'column_info': lambda: self._get_column_info(columns),
            'outliers': lambda: self._get_outliers(columns),
            'distribution': lambda: self._get_distribution(columns),
            'duplicates': lambda: self._get_duplicates(),
            'quality': lambda: self._get_quality(),
            'sample': lambda: self._get_sample(),
            'filter': lambda: self._filter_data(query),
            'top_n': lambda: self._get_top_n(int(numbers[0]) if numbers else 5, columns),
            'bottom_n': lambda: self._get_bottom_n(int(numbers[0]) if numbers else 5, columns),
            'greeting': lambda: self._get_greeting(),
            'thanks': lambda: self._get_thanks(),
            'help': lambda: self._get_help(),
        }
        
        if intent in response_map:
            response = response_map[intent]()
        else:
            response = self._get_unknown_response(query)
        
        # Store response in history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_suggested_questions(self) -> List[str]:
        """
        Generate context-aware suggested questions based on the dataset.
        """
        suggestions = [
            "Tell me about this dataset",
            "What are all the columns?",
            "Show me missing values",
        ]
        
        # Add column-specific suggestions
        if self.numeric_cols:
            col = self.numeric_cols[0]
            suggestions.append(f"What is the average {col}?")
            suggestions.append(f"Find outliers in {col}")
        
        if self.categorical_cols:
            col = self.categorical_cols[0]
            suggestions.append(f"What are the unique values in {col}?")
            suggestions.append(f"Show value counts for {col}")
        
        if len(self.numeric_cols) >= 2:
            suggestions.append(f"What's the correlation between {self.numeric_cols[0]} and {self.numeric_cols[1]}?")
        
        suggestions.extend([
            "Check for duplicate rows",
            "Show me a data quality report",
            "Give me a sample of the data"
        ])
        
        return suggestions[:10]  # Return top 10 suggestions


# Example usage and testing
if __name__ == "__main__":
    # Create a sample DataFrame for testing
    import numpy as np
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'] * 20,
        'Age': np.random.randint(20, 60, 100),
        'Salary': np.random.uniform(30000, 100000, 100),
        'Department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], 100),
        'Experience': np.random.randint(0, 20, 100),
        'Score': np.random.uniform(0, 100, 100)
    })
    
    # Add some missing values
    test_df.loc[5:10, 'Salary'] = np.nan
    test_df.loc[15:18, 'Age'] = np.nan
    
    # Initialize chatbot
    chatbot = DatasetChatbot(test_df)
    
    # Test queries
    test_queries = [
        "Hello",
        "Tell me about the dataset",
        "What are the columns?",
        "What is the mean of Salary?",
        "Show me missing values",
        "What's the correlation between Age and Salary?",
        "Find outliers in Salary",
        "Show value counts for Department",
        "Check for duplicates",
        "help"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")
        print(chatbot.chat(query))
