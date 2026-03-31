"""
Advanced Dataset Chatbot with Enhanced NLP
==========================================
Extended version with more sophisticated query understanding,
aggregation queries, and natural language to pandas translation.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict, Any, Optional, Callable
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class AdvancedDatasetChatbot:
    """
    Advanced chatbot with sophisticated NLP for dataset Q&A.
    Supports complex queries, aggregations, and data exploration.
    """
    
    def __init__(self, df: pd.DataFrame, column_metadata: Dict = None):
        """
        Initialize the advanced chatbot.
        
        Args:
            df: The pandas DataFrame
            column_metadata: Optional dict with column type information
        """
        self.df = df
        self.original_df = df.copy()
        self.column_names = df.columns.tolist()
        self.column_metadata = column_metadata or {}
        
        # Build column lookup
        self._build_column_index()
        
        # Analyze data types
        self._analyze_data_types()
        
        # Build keyword synonyms
        self._build_synonyms()
        
        # Intent classifiers
        self._build_intent_classifiers()
        
        # Aggregation functions
        self._build_aggregation_functions()
        
        # Cache for expensive computations
        self._cache = {}
        
        # Conversation context
        self.context = {
            'last_column': None,
            'last_result': None,
            'last_query_type': None
        }
    
    def _build_column_index(self):
        """Build various indexes for column name matching."""
        self.column_lower_map = {col.lower(): col for col in self.column_names}
        
        # Word-based index
        self.column_word_index = defaultdict(list)
        for col in self.column_names:
            # Split by common separators
            words = re.split(r'[_\s\-\.]+', col.lower())
            for word in words:
                if len(word) > 2:
                    self.column_word_index[word].append(col)
        
        # Abbreviation index
        self.column_abbrev_index = {}
        for col in self.column_names:
            # Create abbreviation from first letters
            words = re.split(r'[_\s\-\.]+', col)
            abbrev = ''.join(w[0].lower() for w in words if w)
            self.column_abbrev_index[abbrev] = col
    
    def _analyze_data_types(self):
        """Analyze and categorize columns by data type."""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        self.datetime_cols = []
        
        # Detect datetime columns
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_cols.append(col)
        
        # Identify high cardinality categorical
        self.high_cardinality_cols = []
        for col in self.categorical_cols:
            if self.df[col].nunique() > 50:
                self.high_cardinality_cols.append(col)
        
        # Precompute statistics for numeric columns
        self.numeric_stats = {}
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                self.numeric_stats[col] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'sum': series.sum(),
                    'count': len(series),
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75),
                    'skew': series.skew(),
                    'kurtosis': series.kurtosis()
                }
    
    def _build_synonyms(self):
        """Build synonym mappings for common terms."""
        self.aggregation_synonyms = {
            'average': 'mean', 'avg': 'mean', 'mean': 'mean',
            'median': 'median', 'middle': 'median',
            'total': 'sum', 'sum': 'sum', 'add': 'sum',
            'minimum': 'min', 'min': 'min', 'smallest': 'min', 'lowest': 'min',
            'maximum': 'max', 'max': 'max', 'largest': 'max', 'highest': 'max', 'biggest': 'max',
            'count': 'count', 'number': 'count', 'how many': 'count',
            'standard deviation': 'std', 'std': 'std', 'deviation': 'std',
            'variance': 'var', 'var': 'var',
            'first': 'first', 'last': 'last'
        }
        
        self.comparison_synonyms = {
            'greater than': '>', 'more than': '>', 'above': '>', 'over': '>',
            'greater than or equal': '>=', 'at least': '>=',
            'less than': '<', 'below': '<', 'under': '<', 'fewer than': '<',
            'less than or equal': '<=', 'at most': '<=',
            'equal to': '==', 'equals': '==', 'is': '==', 'exactly': '=='
        }
        
        self.groupby_keywords = [
            'by', 'per', 'for each', 'grouped by', 'group by',
            'broken down by', 'segmented by', 'split by'
        ]
    
    def _build_intent_classifiers(self):
        """Build intent classification patterns."""
        self.intent_patterns = {
            # Basic dataset info
            'shape': r'\b(shape|dimensions?|size|how\s+(big|large))\b',
            'info': r'\b(info|information|about|describe|overview|summary)\s+(the\s+)?(dataset|data|dataframe)\b',
            'columns': r'\b(columns?|features?|variables?|fields?|attributes?)\b',
            'types': r'\b(data\s*)?types?\b',
            'head': r'\b(head|first|top|beginning|preview|sample)\s*(\d+)?\s*(rows?)?\b',
            'tail': r'\b(tail|last|bottom|end)\s*(\d+)?\s*(rows?)?\b',
            
            # Statistics
            'describe': r'\b(describe|statistics?|stats?|summary)\s+(of|for)?\s*',
            'mean': r'\b(mean|average|avg)\b',
            'median': r'\b(median|middle)\b',
            'mode': r'\b(mode|most\s+(common|frequent))\b',
            'sum': r'\b(sum|total)\b',
            'min': r'\b(min|minimum|smallest|lowest)\b',
            'max': r'\b(max|maximum|largest|highest|biggest)\b',
            'std': r'\b(std|standard\s*deviation|deviation)\b',
            'variance': r'\b(variance|var)\b',
            'range': r'\b(range)\b',
            'percentile': r'\b(\d+)(th|st|nd|rd)?\s*percentile\b',
            
            # Aggregation with groupby
            'groupby': r'\b(by|per|for\s+each|group\s*by|grouped\s*by)\b',
            
            # Data quality
            'missing': r'\b(missing|null|nan|empty|na|blank)\b',
            'duplicates': r'\b(duplicate|duplicated|repeated|redundant)\b',
            'outliers': r'\b(outliers?|anomal\w*|unusual|extreme)\b',
            'quality': r'\b(quality|health|completeness)\b',
            
            # Unique/distinct values
            'unique': r'\b(unique|distinct|different)\b',
            'value_counts': r'\b(value\s*counts?|frequency|distribution|breakdown|count\s+of\s+each)\b',
            
            # Correlation
            'correlation': r'\b(correlat\w*|relationship|related|association)\b',
            
            # Filtering
            'filter': r'\b(filter|where|find|show|get|select|rows?\s+where|records?\s+where)\b',
            'between': r'\b(between)\s+\d+\s+and\s+\d+\b',
            
            # Comparison operators
            'comparison': r'[<>=!]+\s*\d+|greater\s+than|less\s+than|equal\s+to|more\s+than|fewer\s+than',
            
            # Top/Bottom N
            'top_n': r'\b(top|highest|largest|biggest|best)\s*\d+\b',
            'bottom_n': r'\b(bottom|lowest|smallest|worst)\s*\d+\b',
            
            # Sorting
            'sort': r'\b(sort|order|rank|arrange)\b',
            
            # Date-related
            'date': r'\b(year|month|day|date|week|quarter)\b',
            
            # Help/greeting
            'help': r'\b(help|what\s+can\s+you\s+do|capabilities|commands?)\b',
            'greeting': r'\b(hi|hello|hey|greetings)\b',
        }
    
    def _build_aggregation_functions(self):
        """Build mapping of aggregation functions."""
        self.agg_functions = {
            'mean': lambda x: x.mean(),
            'median': lambda x: x.median(),
            'sum': lambda x: x.sum(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'count': lambda x: x.count(),
            'std': lambda x: x.std(),
            'var': lambda x: x.var(),
            'first': lambda x: x.iloc[0] if len(x) > 0 else None,
            'last': lambda x: x.iloc[-1] if len(x) > 0 else None,
            'nunique': lambda x: x.nunique(),
        }
    
    def _find_columns(self, query: str) -> List[str]:
        """
        Find column names mentioned in the query using multiple strategies.
        """
        query_lower = query.lower()
        found = []
        
        # Strategy 1: Exact match
        for col_lower, col in self.column_lower_map.items():
            if col_lower in query_lower:
                found.append(col)
        
        # Strategy 2: Word-based match
        if not found:
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            for word in query_words:
                if word in self.column_word_index:
                    found.extend(self.column_word_index[word])
        
        # Strategy 3: Abbreviation match
        if not found:
            for abbrev, col in self.column_abbrev_index.items():
                if abbrev in query_lower.replace(' ', ''):
                    found.append(col)
        
        # Strategy 4: Fuzzy match (simple)
        if not found:
            for col in self.column_names:
                col_clean = re.sub(r'[_\-\.]', '', col.lower())
                query_clean = re.sub(r'[_\-\.]', '', query_lower)
                if col_clean in query_clean or query_clean in col_clean:
                    found.append(col)
        
        return list(set(found))
    
    def _extract_number(self, query: str) -> Optional[float]:
        """Extract a number from the query."""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        return float(numbers[0]) if numbers else None
    
    def _extract_all_numbers(self, query: str) -> List[float]:
        """Extract all numbers from the query."""
        return [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', query)]
    
    def _detect_aggregation(self, query: str) -> Optional[str]:
        """Detect which aggregation function is requested."""
        query_lower = query.lower()
        for keyword, agg in self.aggregation_synonyms.items():
            if keyword in query_lower:
                return agg
        return None
    
    def _detect_groupby_column(self, query: str, target_cols: List[str]) -> Optional[str]:
        """Detect if there's a groupby column specified."""
        query_lower = query.lower()
        
        # Check for groupby keywords
        has_groupby = any(kw in query_lower for kw in self.groupby_keywords)
        
        if not has_groupby:
            return None
        
        # Find the groupby column (usually categorical)
        for col in self.categorical_cols:
            if col.lower() in query_lower and col not in target_cols:
                return col
        
        return None
    
    def _detect_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the primary intent of the query.
        Returns (intent, metadata)
        """
        query_lower = query.lower()
        
        matched_intents = []
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                matched_intents.append(intent)
        
        # Prioritize intents
        priority_order = [
            'groupby', 'filter', 'comparison', 'correlation',
            'top_n', 'bottom_n', 'sort',
            'mean', 'median', 'sum', 'min', 'max', 'std',
            'describe', 'unique', 'value_counts',
            'missing', 'duplicates', 'outliers', 'quality',
            'shape', 'info', 'columns', 'types',
            'head', 'tail', 'help', 'greeting'
        ]
        
        primary_intent = 'unknown'
        for intent in priority_order:
            if intent in matched_intents:
                primary_intent = intent
                break
        
        # Extract metadata
        metadata = {
            'columns': self._find_columns(query),
            'numbers': self._extract_all_numbers(query),
            'aggregation': self._detect_aggregation(query),
            'all_intents': matched_intents
        }
        
        # Check for groupby
        if primary_intent in ['mean', 'median', 'sum', 'min', 'max', 'std', 'count']:
            groupby_col = self._detect_groupby_column(query, metadata['columns'])
            if groupby_col:
                metadata['groupby'] = groupby_col
                primary_intent = 'groupby_agg'
        
        return primary_intent, metadata
    
    def _format_number(self, num: float, decimals: int = 2) -> str:
        """Format number for display."""
        if pd.isna(num):
            return "N/A"
        if abs(num) >= 1e9:
            return f"{num/1e9:.2f}B"
        if abs(num) >= 1e6:
            return f"{num/1e6:.2f}M"
        if abs(num) >= 1e3:
            return f"{num:,.2f}"
        return f"{round(num, decimals)}"
    
    # ==================== Response Handlers ====================
    
    def _handle_shape(self, metadata: Dict) -> str:
        rows, cols = self.df.shape
        memory = self.df.memory_usage(deep=True).sum() / 1024**2
        return f"""📊 **Dataset Dimensions**
- Rows: **{rows:,}**
- Columns: **{cols}**
- Memory: **{memory:.2f} MB**"""
    
    def _handle_info(self, metadata: Dict) -> str:
        rows, cols = self.df.shape
        memory = self.df.memory_usage(deep=True).sum() / 1024**2
        missing = self.df.isna().sum().sum()
        missing_pct = (missing / self.df.size) * 100
        dups = self.df.duplicated().sum()
        
        return f"""📊 **Dataset Overview**

| Metric | Value |
|--------|-------|
| Rows | {rows:,} |
| Columns | {cols} |
| Memory | {memory:.2f} MB |
| Missing Values | {missing:,} ({missing_pct:.2f}%) |
| Duplicate Rows | {dups:,} |

**Column Types:**
- Numeric: {len(self.numeric_cols)}
- Categorical: {len(self.categorical_cols)}
- DateTime: {len(self.datetime_cols)}
- Boolean: {len(self.bool_cols)}"""
    
    def _handle_columns(self, metadata: Dict) -> str:
        lines = ["📋 **Columns in Dataset:**\n"]
        lines.append("| # | Column | Type | Non-Null | Unique |")
        lines.append("|---|--------|------|----------|--------|")
        
        for i, col in enumerate(self.column_names, 1):
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].notna().sum()
            unique = self.df[col].nunique()
            lines.append(f"| {i} | {col} | {dtype} | {non_null:,} | {unique:,} |")
        
        return "\n".join(lines)
    
    def _handle_describe(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        
        if cols:
            target_cols = [c for c in cols if c in self.numeric_cols]
        else:
            target_cols = self.numeric_cols[:5]
        
        if not target_cols:
            return "❌ No numeric columns found for statistical description."
        
        desc = self.df[target_cols].describe()
        return f"📈 **Statistical Summary:**\n\n{desc.to_markdown()}"
    
    def _handle_mean(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Mean Values:**\n"]
        for col in target_cols:
            mean_val = self.df[col].mean()
            lines.append(f"- **{col}**: {self._format_number(mean_val)}")
        
        return "\n".join(lines)
    
    def _handle_median(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Median Values:**\n"]
        for col in target_cols:
            median_val = self.df[col].median()
            lines.append(f"- **{col}**: {self._format_number(median_val)}")
        
        return "\n".join(lines)
    
    def _handle_sum(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Sum Values:**\n"]
        for col in target_cols:
            sum_val = self.df[col].sum()
            lines.append(f"- **{col}**: {self._format_number(sum_val)}")
        
        return "\n".join(lines)
    
    def _handle_min(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Minimum Values:**\n"]
        for col in target_cols:
            min_val = self.df[col].min()
            lines.append(f"- **{col}**: {self._format_number(min_val)}")
        
        return "\n".join(lines)
    
    def _handle_max(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Maximum Values:**\n"]
        for col in target_cols:
            max_val = self.df[col].max()
            lines.append(f"- **{col}**: {self._format_number(max_val)}")
        
        return "\n".join(lines)
    
    def _handle_std(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        target_cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols
        
        lines = ["📊 **Standard Deviation:**\n"]
        for col in target_cols:
            std_val = self.df[col].std()
            lines.append(f"- **{col}**: {self._format_number(std_val)}")
        
        return "\n".join(lines)
    
    def _handle_groupby_agg(self, metadata: Dict) -> str:
        """Handle aggregation with groupby."""
        groupby_col = metadata.get('groupby')
        agg_func = metadata.get('aggregation', 'mean')
        cols = metadata.get('columns', [])
        
        value_cols = [c for c in cols if c in self.numeric_cols]
        if not value_cols:
            value_cols = self.numeric_cols[:3]
        
        if not groupby_col or not value_cols:
            return "❌ Could not determine columns for groupby aggregation."
        
        result = self.df.groupby(groupby_col)[value_cols].agg(agg_func)
        
        return f"""📊 **{agg_func.title()} of {', '.join(value_cols)} by {groupby_col}:**

{result.to_markdown()}"""
    
    def _handle_missing(self, metadata: Dict) -> str:
        cols = metadata.get('columns', []) or self.column_names
        
        lines = ["🔍 **Missing Values Analysis:**\n"]
        lines.append("| Column | Missing | Percentage |")
        lines.append("|--------|---------|------------|")
        
        total_missing = 0
        for col in cols:
            if col in self.column_names:
                missing = self.df[col].isna().sum()
                pct = (missing / len(self.df)) * 100
                total_missing += missing
                if missing > 0:
                    lines.append(f"| {col} | {missing:,} | {pct:.2f}% |")
        
        if total_missing == 0:
            return "✅ **No missing values found in the dataset!**"
        
        lines.insert(2, f"\n**Total Missing:** {total_missing:,}\n")
        return "\n".join(lines)
    
    def _handle_duplicates(self, metadata: Dict) -> str:
        dup_count = self.df.duplicated().sum()
        dup_pct = (dup_count / len(self.df)) * 100
        
        if dup_count == 0:
            return "✅ **No duplicate rows found!**"
        
        return f"""📋 **Duplicate Analysis:**

| Metric | Value |
|--------|-------|
| Duplicate Rows | {dup_count:,} |
| Percentage | {dup_pct:.2f}% |
| Unique Rows | {len(self.df) - dup_count:,} |"""
    
    def _handle_unique(self, metadata: Dict) -> str:
        cols = metadata.get('columns', []) or self.column_names[:10]
        
        lines = ["🔢 **Unique Values Count:**\n"]
        lines.append("| Column | Unique | Total | % Unique |")
        lines.append("|--------|--------|-------|----------|")
        
        for col in cols:
            if col in self.column_names:
                unique = self.df[col].nunique()
                total = len(self.df)
                pct = (unique / total) * 100
                lines.append(f"| {col} | {unique:,} | {total:,} | {pct:.1f}% |")
        
        return "\n".join(lines)
    
    def _handle_value_counts(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        if not cols:
            cols = self.categorical_cols[:1] if self.categorical_cols else self.column_names[:1]
        
        lines = []
        for col in cols[:2]:  # Limit to 2 columns
            if col in self.column_names:
                vc = self.df[col].value_counts().head(10)
                lines.append(f"\n📊 **Value Counts for {col}:**\n")
                lines.append("| Value | Count | Percentage |")
                lines.append("|-------|-------|------------|")
                for val, count in vc.items():
                    pct = (count / len(self.df)) * 100
                    lines.append(f"| {val} | {count:,} | {pct:.1f}% |")
        
        return "\n".join(lines)
    
    def _handle_correlation(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        cols = [c for c in cols if c in self.numeric_cols]
        
        if len(cols) >= 2:
            # Specific correlation between two columns
            corr = self.df[cols[0]].corr(self.df[cols[1]])
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            
            return f"""🔗 **Correlation Analysis:**

**{cols[0]}** ↔ **{cols[1]}**
- Coefficient: **{corr:.4f}**
- Strength: {strength} {direction}"""
        
        # Top correlations
        if len(self.numeric_cols) < 2:
            return "❌ Need at least 2 numeric columns for correlation."
        
        corr_matrix = self.df[self.numeric_cols].corr()
        pairs = []
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                pairs.append((
                    self.numeric_cols[i],
                    self.numeric_cols[j],
                    corr_matrix.iloc[i, j]
                ))
        
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        lines = ["🔗 **Top Correlations:**\n"]
        lines.append("| Column 1 | Column 2 | Correlation |")
        lines.append("|----------|----------|-------------|")
        
        for c1, c2, corr in pairs[:10]:
            if not pd.isna(corr):
                lines.append(f"| {c1} | {c2} | {corr:.4f} |")
        
        return "\n".join(lines)
    
    def _handle_outliers(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        cols = [c for c in cols if c in self.numeric_cols] or self.numeric_cols[:5]
        
        lines = ["🔍 **Outlier Analysis (IQR Method):**\n"]
        lines.append("| Column | Outliers | % | Lower | Upper |")
        lines.append("|--------|----------|---|-------|-------|")
        
        for col in cols:
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).sum()
            pct = (outliers / len(series)) * 100
            lines.append(f"| {col} | {outliers:,} | {pct:.1f}% | {self._format_number(lower)} | {self._format_number(upper)} |")
        
        return "\n".join(lines)
    
    def _handle_top_n(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        n = int(metadata.get('numbers', [5])[0]) if metadata.get('numbers') else 5
        
        if not cols:
            cols = self.numeric_cols[:1]
        
        col = cols[0]
        if col in self.numeric_cols:
            result = self.df.nlargest(n, col)[[col]]
            return f"📊 **Top {n} values in {col}:**\n\n```\n{result.to_string()}\n```"
        else:
            vc = self.df[col].value_counts().head(n)
            lines = [f"📊 **Top {n} values in {col}:**\n"]
            for val, count in vc.items():
                lines.append(f"- {val}: {count:,}")
            return "\n".join(lines)
    
    def _handle_bottom_n(self, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        n = int(metadata.get('numbers', [5])[0]) if metadata.get('numbers') else 5
        
        if not cols:
            cols = self.numeric_cols[:1]
        
        col = cols[0]
        if col in self.numeric_cols:
            result = self.df.nsmallest(n, col)[[col]]
            return f"📊 **Bottom {n} values in {col}:**\n\n```\n{result.to_string()}\n```"
        else:
            vc = self.df[col].value_counts().tail(n)
            lines = [f"📊 **Bottom {n} values in {col}:**\n"]
            for val, count in vc.items():
                lines.append(f"- {val}: {count:,}")
            return "\n".join(lines)
    
    def _handle_head(self, metadata: Dict) -> str:
        n = int(metadata.get('numbers', [5])[0]) if metadata.get('numbers') else 5
        return f"📋 **First {n} rows:**\n\n```\n{self.df.head(n).to_string()}\n```"
    
    def _handle_tail(self, metadata: Dict) -> str:
        n = int(metadata.get('numbers', [5])[0]) if metadata.get('numbers') else 5
        return f"📋 **Last {n} rows:**\n\n```\n{self.df.tail(n).to_string()}\n```"
    
    def _handle_filter(self, query: str, metadata: Dict) -> str:
        """Handle filtering queries."""
        cols = metadata.get('columns', [])
        numbers = metadata.get('numbers', [])
        
        # Try to parse filter condition
        patterns = [
            (r'(\w+)\s*>=\s*(\d+(?:\.\d+)?)', '>='),
            (r'(\w+)\s*<=\s*(\d+(?:\.\d+)?)', '<='),
            (r'(\w+)\s*>\s*(\d+(?:\.\d+)?)', '>'),
            (r'(\w+)\s*<\s*(\d+(?:\.\d+)?)', '<'),
            (r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', '=='),
        ]
        
        for pattern, op in patterns:
            match = re.search(pattern, query)
            if match:
                col_hint = match.group(1).lower()
                value = float(match.group(2))
                
                # Find matching column
                col = None
                for c in self.column_names:
                    if col_hint in c.lower():
                        col = c
                        break
                
                if col and col in self.numeric_cols:
                    if op == '>=':
                        result = self.df[self.df[col] >= value]
                    elif op == '<=':
                        result = self.df[self.df[col] <= value]
                    elif op == '>':
                        result = self.df[self.df[col] > value]
                    elif op == '<':
                        result = self.df[self.df[col] < value]
                    else:
                        result = self.df[self.df[col] == value]
                    
                    if len(result) == 0:
                        return "❌ No rows match the filter."
                    
                    preview = result.head(10)
                    return f"""📋 **Filter Results:** {len(result):,} rows match

**Preview (first 10):**
```
{preview.to_string()}
```"""
        
        return "❓ Could not parse filter condition. Try: 'filter column > 100'"
    
    def _handle_help(self, metadata: Dict) -> str:
        return """🤖 **Dataset Chatbot Help**

I can answer questions about your data in natural language!

**📊 Statistics:**
- "What's the average of [column]?"
- "Show me the sum of [column] by [category]"
- "What's the median salary?"

**🔍 Data Quality:**
- "Are there missing values?"
- "Find duplicates"
- "Check for outliers in [column]"

**📋 Exploration:**
- "Show me the columns"
- "Value counts for [column]"
- "Top 10 values in [column]"
- "Correlation between [col1] and [col2]"

**🎯 Filtering:**
- "Filter where [column] > 100"
- "Show rows where [column] = 'value'"

**💡 Tips:**
- I can understand column names even with typos
- Use natural language - I'll figure out what you mean!
"""
    
    def _handle_greeting(self, metadata: Dict) -> str:
        return "👋 Hello! I'm your dataset assistant. Ask me anything about your data!"
    
    def _handle_unknown(self, query: str, metadata: Dict) -> str:
        cols = metadata.get('columns', [])
        
        if cols:
            # Try to give info about mentioned columns
            lines = [f"I noticed you mentioned: **{', '.join(cols)}**\n"]
            for col in cols[:2]:
                if col in self.numeric_cols:
                    stats = self.numeric_stats.get(col, {})
                    if stats:
                        lines.append(f"\n**{col}:**")
                        lines.append(f"- Mean: {self._format_number(stats['mean'])}")
                        lines.append(f"- Min: {self._format_number(stats['min'])}")
                        lines.append(f"- Max: {self._format_number(stats['max'])}")
                else:
                    unique = self.df[col].nunique()
                    top_val = self.df[col].value_counts().index[0]
                    lines.append(f"\n**{col}:** {unique} unique values, most common: {top_val}")
            
            return "\n".join(lines)
        
        return f"""❓ I'm not sure what you're asking.

**Your question:** "{query}"

Try asking things like:
- "What's the average of [column]?"
- "Show me missing values"
- "Describe the dataset"

Type **help** to see all my capabilities!"""
    
    def chat(self, query: str) -> str:
        """
        Main chat interface - process query and return response.
        """
        if not query.strip():
            return "Please enter a question about your dataset."
        
        # Detect intent
        intent, metadata = self._detect_intent(query)
        
        # Update context
        if metadata.get('columns'):
            self.context['last_column'] = metadata['columns'][0]
        self.context['last_query_type'] = intent
        
        # Route to handler
        handlers = {
            'shape': self._handle_shape,
            'info': self._handle_info,
            'columns': self._handle_columns,
            'describe': self._handle_describe,
            'mean': self._handle_mean,
            'median': self._handle_median,
            'sum': self._handle_sum,
            'min': self._handle_min,
            'max': self._handle_max,
            'std': self._handle_std,
            'groupby_agg': self._handle_groupby_agg,
            'missing': self._handle_missing,
            'duplicates': self._handle_duplicates,
            'unique': self._handle_unique,
            'value_counts': self._handle_value_counts,
            'correlation': self._handle_correlation,
            'outliers': self._handle_outliers,
            'top_n': self._handle_top_n,
            'bottom_n': self._handle_bottom_n,
            'head': self._handle_head,
            'tail': self._handle_tail,
            'filter': lambda m: self._handle_filter(query, m),
            'comparison': lambda m: self._handle_filter(query, m),
            'help': self._handle_help,
            'greeting': self._handle_greeting,
        }
        
        handler = handlers.get(intent)
        if handler:
            try:
                return handler(metadata)
            except Exception as e:
                return f"❌ Error processing query: {str(e)}"
        
        return self._handle_unknown(query, metadata)
    
    def get_suggestions(self) -> List[str]:
        """Generate context-aware suggestions."""
        suggestions = [
            "Tell me about this dataset",
            "What columns are available?",
            "Show me missing values",
        ]
        
        if self.numeric_cols:
            col = self.numeric_cols[0]
            suggestions.append(f"What's the average {col}?")
            
        if len(self.numeric_cols) >= 2:
            suggestions.append(f"Correlation between {self.numeric_cols[0]} and {self.numeric_cols[1]}")
        
        if self.categorical_cols:
            col = self.categorical_cols[0]
            suggestions.append(f"Value counts for {col}")
            
            if self.numeric_cols:
                suggestions.append(f"Average {self.numeric_cols[0]} by {col}")
        
        suggestions.extend([
            "Find outliers",
            "Check for duplicates",
            "Show first 10 rows"
        ])
        
        return suggestions[:10]


# Test the advanced chatbot
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'Product': np.random.choice(['Widget', 'Gadget', 'Gizmo', 'Device'], 200),
        'Category': np.random.choice(['Electronics', 'Home', 'Office', 'Outdoor'], 200),
        'Sales': np.random.uniform(100, 10000, 200),
        'Quantity': np.random.randint(1, 100, 200),
        'Price': np.random.uniform(10, 500, 200),
        'Rating': np.random.uniform(1, 5, 200),
        'Discount': np.random.uniform(0, 0.5, 200)
    })
    
    # Add some nulls
    test_df.loc[10:15, 'Sales'] = np.nan
    test_df.loc[25:30, 'Rating'] = np.nan
    
    chatbot = AdvancedDatasetChatbot(test_df)
    
    # Test queries
    queries = [
        "hi",
        "tell me about the dataset",
        "what columns are there",
        "average sales",
        "average sales by category",
        "missing values",
        "correlation between sales and quantity",
        "top 5 sales",
        "outliers in price",
        "filter sales > 5000",
        "help"
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"User: {q}")
        print('='*60)
        print(chatbot.chat(q))
