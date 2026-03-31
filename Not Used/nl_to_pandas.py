"""
Natural Language to Pandas Query Translator
============================================
Converts natural language questions about data into executable pandas code.
This is a key component for the dataset chatbot.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class NLToPandasTranslator:
    """
    Translates natural language questions into pandas operations.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_names = df.columns.tolist()
        self.column_lower_map = {col.lower().replace('_', ' '): col for col in self.column_names}
        
        # Analyze column types
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def find_column(self, text: str) -> Optional[str]:
        """Find a column name in the given text."""
        text_lower = text.lower()
        
        # Direct match
        for col in self.column_names:
            if col.lower() in text_lower:
                return col
        
        # Word-by-word match
        for col_lower, col in self.column_lower_map.items():
            if col_lower in text_lower:
                return col
        
        # Partial match
        words = text_lower.split()
        for word in words:
            for col in self.column_names:
                if word in col.lower() and len(word) > 3:
                    return col
        
        return None
    
    def find_all_columns(self, text: str) -> List[str]:
        """Find all column names mentioned in text."""
        text_lower = text.lower()
        found = []
        
        for col in self.column_names:
            if col.lower() in text_lower:
                found.append(col)
        
        return found
    
    def extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        return float(numbers[0]) if numbers else None
    
    def extract_comparison(self, text: str) -> Optional[Tuple[str, float]]:
        """Extract comparison operator and value."""
        patterns = [
            (r'greater\s+than\s+(\d+(?:\.\d+)?)', '>'),
            (r'more\s+than\s+(\d+(?:\.\d+)?)', '>'),
            (r'above\s+(\d+(?:\.\d+)?)', '>'),
            (r'over\s+(\d+(?:\.\d+)?)', '>'),
            (r'>\s*(\d+(?:\.\d+)?)', '>'),
            (r'less\s+than\s+(\d+(?:\.\d+)?)', '<'),
            (r'fewer\s+than\s+(\d+(?:\.\d+)?)', '<'),
            (r'below\s+(\d+(?:\.\d+)?)', '<'),
            (r'under\s+(\d+(?:\.\d+)?)', '<'),
            (r'<\s*(\d+(?:\.\d+)?)', '<'),
            (r'at\s+least\s+(\d+(?:\.\d+)?)', '>='),
            (r'>=\s*(\d+(?:\.\d+)?)', '>='),
            (r'at\s+most\s+(\d+(?:\.\d+)?)', '<='),
            (r'<=\s*(\d+(?:\.\d+)?)', '<='),
            (r'equal\s+to\s+(\d+(?:\.\d+)?)', '=='),
            (r'equals?\s+(\d+(?:\.\d+)?)', '=='),
            (r'=\s*(\d+(?:\.\d+)?)', '=='),
        ]
        
        for pattern, op in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return (op, float(match.group(1)))
        
        return None
    
    def extract_aggregation(self, text: str) -> Optional[str]:
        """Extract aggregation function from text."""
        text_lower = text.lower()
        
        agg_map = {
            'average': 'mean', 'avg': 'mean', 'mean': 'mean',
            'sum': 'sum', 'total': 'sum',
            'count': 'count', 'number of': 'count', 'how many': 'count',
            'maximum': 'max', 'max': 'max', 'highest': 'max', 'largest': 'max',
            'minimum': 'min', 'min': 'min', 'lowest': 'min', 'smallest': 'min',
            'median': 'median', 'middle': 'median',
            'standard deviation': 'std', 'std': 'std',
        }
        
        for keyword, agg in agg_map.items():
            if keyword in text_lower:
                return agg
        
        return None
    
    def extract_groupby(self, text: str) -> Optional[str]:
        """Extract groupby column from text."""
        text_lower = text.lower()
        
        # Check for groupby keywords
        groupby_patterns = [
            r'by\s+(\w+)',
            r'per\s+(\w+)',
            r'for\s+each\s+(\w+)',
            r'grouped\s+by\s+(\w+)',
        ]
        
        for pattern in groupby_patterns:
            match = re.search(pattern, text_lower)
            if match:
                word = match.group(1)
                # Find matching column
                for col in self.categorical_cols:
                    if word in col.lower():
                        return col
        
        return None
    
    def translate(self, query: str) -> Tuple[str, Any]:
        """
        Translate natural language to pandas operation.
        
        Returns:
            Tuple of (operation_description, result)
        """
        query_lower = query.lower()
        
        # Shape/size queries
        if any(kw in query_lower for kw in ['how many rows', 'number of rows', 'shape', 'size']):
            rows, cols = self.df.shape
            return f"df.shape", f"({rows}, {cols})"
        
        # Column list
        if any(kw in query_lower for kw in ['columns', 'features', 'variables']):
            return "df.columns.tolist()", self.column_names
        
        # Missing values
        if any(kw in query_lower for kw in ['missing', 'null', 'nan']):
            col = self.find_column(query)
            if col:
                missing = self.df[col].isna().sum()
                return f"df['{col}'].isna().sum()", missing
            else:
                missing = self.df.isna().sum()
                return "df.isna().sum()", missing
        
        # Aggregation queries
        agg = self.extract_aggregation(query)
        if agg:
            col = self.find_column(query)
            groupby_col = self.extract_groupby(query)
            
            if groupby_col and col:
                result = self.df.groupby(groupby_col)[col].agg(agg)
                return f"df.groupby('{groupby_col}')['{col}'].{agg}()", result
            elif col:
                result = getattr(self.df[col], agg)()
                return f"df['{col}'].{agg}()", result
            else:
                # Apply to all numeric columns
                result = getattr(self.df[self.numeric_cols], agg)()
                return f"df[numeric_cols].{agg}()", result
        
        # Filter queries
        comparison = self.extract_comparison(query)
        if comparison:
            op, value = comparison
            col = self.find_column(query)
            if col:
                if op == '>':
                    result = self.df[self.df[col] > value]
                elif op == '<':
                    result = self.df[self.df[col] < value]
                elif op == '>=':
                    result = self.df[self.df[col] >= value]
                elif op == '<=':
                    result = self.df[self.df[col] <= value]
                else:
                    result = self.df[self.df[col] == value]
                
                return f"df[df['{col}'] {op} {value}]", result
        
        # Unique values
        if 'unique' in query_lower:
            col = self.find_column(query)
            if col:
                unique = self.df[col].nunique()
                return f"df['{col}'].nunique()", unique
        
        # Value counts
        if any(kw in query_lower for kw in ['value counts', 'frequency', 'distribution']):
            col = self.find_column(query)
            if col:
                vc = self.df[col].value_counts()
                return f"df['{col}'].value_counts()", vc
        
        # Describe
        if 'describe' in query_lower or 'statistics' in query_lower:
            col = self.find_column(query)
            if col:
                desc = self.df[col].describe()
                return f"df['{col}'].describe()", desc
            else:
                desc = self.df.describe()
                return "df.describe()", desc
        
        # Correlation
        if 'correlation' in query_lower or 'correlated' in query_lower:
            cols = self.find_all_columns(query)
            if len(cols) >= 2:
                corr = self.df[cols[0]].corr(self.df[cols[1]])
                return f"df['{cols[0]}'].corr(df['{cols[1]}'])", corr
            else:
                corr = self.df[self.numeric_cols].corr()
                return "df[numeric_cols].corr()", corr
        
        # Head/tail
        if 'first' in query_lower or 'head' in query_lower:
            n = self.extract_number(query) or 5
            return f"df.head({int(n)})", self.df.head(int(n))
        
        if 'last' in query_lower or 'tail' in query_lower:
            n = self.extract_number(query) or 5
            return f"df.tail({int(n)})", self.df.tail(int(n))
        
        # Top N
        if 'top' in query_lower or 'highest' in query_lower or 'largest' in query_lower:
            n = self.extract_number(query) or 5
            col = self.find_column(query)
            if col and col in self.numeric_cols:
                result = self.df.nlargest(int(n), col)
                return f"df.nlargest({int(n)}, '{col}')", result
        
        # Bottom N
        if 'bottom' in query_lower or 'lowest' in query_lower or 'smallest' in query_lower:
            n = self.extract_number(query) or 5
            col = self.find_column(query)
            if col and col in self.numeric_cols:
                result = self.df.nsmallest(int(n), col)
                return f"df.nsmallest({int(n)}, '{col}')", result
        
        # Duplicates
        if 'duplicate' in query_lower:
            dups = self.df.duplicated().sum()
            return "df.duplicated().sum()", dups
        
        return "Could not translate query", None
    
    def get_executable_code(self, query: str) -> str:
        """
        Generate executable Python/pandas code from natural language query.
        """
        code, _ = self.translate(query)
        return code


class QueryExecutor:
    """
    Safely executes pandas queries on a DataFrame.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.translator = NLToPandasTranslator(df)
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a natural language query and return results.
        
        Returns:
            Dict with 'code', 'result', 'result_type', 'error'
        """
        try:
            code, result = self.translator.translate(query)
            
            if result is None:
                return {
                    'code': code,
                    'result': None,
                    'result_type': 'error',
                    'error': 'Could not understand the query'
                }
            
            # Determine result type
            if isinstance(result, pd.DataFrame):
                result_type = 'dataframe'
            elif isinstance(result, pd.Series):
                result_type = 'series'
            elif isinstance(result, (int, float, np.number)):
                result_type = 'scalar'
            elif isinstance(result, str):
                result_type = 'string'
            elif isinstance(result, (list, tuple)):
                result_type = 'list'
            else:
                result_type = 'other'
            
            return {
                'code': code,
                'result': result,
                'result_type': result_type,
                'error': None
            }
            
        except Exception as e:
            return {
                'code': None,
                'result': None,
                'result_type': 'error',
                'error': str(e)
            }
    
    def format_result(self, result: Dict[str, Any], max_rows: int = 20) -> str:
        """
        Format the query result for display.
        """
        if result['error']:
            return f"❌ Error: {result['error']}"
        
        output_lines = []
        
        # Show the generated code
        if result['code']:
            output_lines.append(f"**Code:** `{result['code']}`\n")
        
        # Format based on result type
        res = result['result']
        res_type = result['result_type']
        
        if res_type == 'dataframe':
            if len(res) > max_rows:
                output_lines.append(f"**Result:** ({len(res):,} rows, showing first {max_rows})\n")
                output_lines.append(res.head(max_rows).to_markdown())
            else:
                output_lines.append(f"**Result:** ({len(res):,} rows)\n")
                output_lines.append(res.to_markdown())
                
        elif res_type == 'series':
            if len(res) > max_rows:
                output_lines.append(f"**Result:** ({len(res):,} items, showing first {max_rows})\n")
                output_lines.append(res.head(max_rows).to_markdown())
            else:
                output_lines.append(f"**Result:**\n")
                output_lines.append(res.to_markdown())
                
        elif res_type == 'scalar':
            if isinstance(res, float):
                output_lines.append(f"**Result:** {res:,.4f}")
            else:
                output_lines.append(f"**Result:** {res:,}")
                
        elif res_type == 'list':
            output_lines.append(f"**Result:** {res}")
            
        elif res_type == 'string':
            output_lines.append(f"**Result:** {res}")
            
        else:
            output_lines.append(f"**Result:** {res}")
        
        return "\n".join(output_lines)


# Example usage
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'Product': np.random.choice(['Widget', 'Gadget', 'Gizmo'], 100),
        'Category': np.random.choice(['Electronics', 'Home', 'Office'], 100),
        'Sales': np.random.uniform(100, 1000, 100),
        'Quantity': np.random.randint(1, 50, 100),
        'Price': np.random.uniform(10, 100, 100)
    })
    
    translator = NLToPandasTranslator(test_df)
    executor = QueryExecutor(test_df)
    
    test_queries = [
        "how many rows",
        "what columns are there",
        "average sales",
        "sum of sales by category",
        "show me rows where sales greater than 500",
        "top 5 by sales",
        "unique products",
        "describe sales",
        "correlation between sales and quantity"
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        result = executor.execute(q)
        print(executor.format_result(result))
