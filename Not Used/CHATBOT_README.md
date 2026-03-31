# Dataset AI Chatbot - Documentation

## Overview

The Dataset AI Chatbot is a **fully local, API-free** natural language interface for interacting with pandas DataFrames. It allows users to ask questions about their data in plain English and receive meaningful answers without any external API calls.

## Features

### 🔢 Statistical Analysis
- **Mean/Average**: "What is the average of Sales?"
- **Median**: "What's the median salary?"
- **Sum/Total**: "Total revenue"
- **Min/Max**: "What's the maximum price?"
- **Standard Deviation**: "Show me the standard deviation of Age"
- **Range**: "What's the range of Temperature?"

### 📊 Data Exploration
- **Dataset Shape**: "How many rows and columns?"
- **Column List**: "What columns are available?"
- **Column Types**: "Show me the data types"
- **Sample Data**: "Show me the first 10 rows"
- **Unique Values**: "How many unique values in Category?"
- **Value Counts**: "Show value counts for Product"

### 🔍 Data Quality
- **Missing Values**: "Are there any missing values?"
- **Duplicates**: "Check for duplicate rows"
- **Outliers**: "Find outliers in Price"
- **Data Quality Report**: "Give me a data quality report"

### 🔗 Correlation Analysis
- **Pairwise**: "What's the correlation between Sales and Quantity?"
- **Top Correlations**: "Show me the strongest correlations"

### 🎯 Filtering & Querying
- **Greater Than**: "Show rows where Sales > 1000"
- **Less Than**: "Filter Price < 50"
- **Equal To**: "Find records where Category = 'Electronics'"
- **Top N**: "Top 10 products by Revenue"
- **Bottom N**: "Bottom 5 by rating"

### 📈 Aggregation with Groupby
- **Group By Category**: "Average sales by category"
- **Per Group**: "Total quantity per product"
- **For Each**: "Count for each region"

## Architecture

### Components

1. **chatbot.py** - Main chatbot class with rule-based NLP
   - Intent detection using regex patterns
   - Column name extraction with fuzzy matching
   - Response generation for each intent type

2. **advanced_chatbot.py** - Enhanced version with:
   - More sophisticated query understanding
   - Support for groupby aggregations
   - Better column name matching
   - Context awareness

3. **nl_to_pandas.py** - Natural Language to Pandas translator
   - Converts questions to pandas code
   - Provides executable code snippets
   - Safe query execution

## Usage

### Basic Usage

```python
from chatbot import DatasetChatbot
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Initialize chatbot
chatbot = DatasetChatbot(df)

# Ask questions
response = chatbot.chat("What is the average of Sales?")
print(response)

response = chatbot.chat("Show me missing values")
print(response)
```

### With Streamlit

The chatbot is integrated into the Streamlit app. Navigate to "AI Assistant" in the sidebar.

### Advanced Usage

```python
from advanced_chatbot import AdvancedDatasetChatbot

chatbot = AdvancedDatasetChatbot(df)

# Groupby aggregations
response = chatbot.chat("average sales by category")

# Complex queries
response = chatbot.chat("top 10 products by revenue")

# Get suggestions
suggestions = chatbot.get_suggestions()
```

## Example Questions

### Dataset Information
- "Tell me about this dataset"
- "How many rows are there?"
- "What are all the columns?"
- "What are the column data types?"

### Statistics
- "What's the mean of Age?"
- "Show me the median salary"
- "Total sales"
- "Maximum price"
- "Minimum quantity"
- "Describe the Revenue column"

### Data Quality
- "Are there missing values?"
- "Which columns have nulls?"
- "Check for duplicates"
- "Find outliers in Income"
- "Data quality report"

### Exploration
- "Show me first 5 rows"
- "Sample of the data"
- "Unique values in Category"
- "Value counts for Product"
- "Distribution of Sales"

### Correlation
- "Correlation between Price and Quantity"
- "Are Sales and Revenue related?"
- "Show me top correlations"

### Filtering
- "Show rows where Age > 30"
- "Filter Sales greater than 1000"
- "Find records where Status = 'Active'"
- "Top 10 by Revenue"

### Groupby
- "Average salary by department"
- "Total sales per region"
- "Count by category"
- "Sum of revenue for each product"

## How It Works

### Intent Detection
The chatbot uses a multi-stage intent detection system:

1. **Pattern Matching**: Regex patterns identify the type of question
2. **Keyword Detection**: Statistical keywords like "mean", "sum", "average"
3. **Column Extraction**: Fuzzy matching finds column names in the query
4. **Aggregation Detection**: Identifies if a groupby operation is needed

### Column Matching
Multiple strategies for finding columns:
- Exact match (case-insensitive)
- Word-based matching (splits on underscores/spaces)
- Abbreviation matching
- Fuzzy/partial matching

### Response Generation
Each intent has a dedicated handler that:
1. Extracts relevant columns and parameters
2. Performs the pandas operation
3. Formats the result with markdown
4. Returns a user-friendly response

## Extending the Chatbot

### Adding New Intents

```python
# In chatbot.py, add to intent_patterns:
self.intent_patterns['new_intent'] = r'\b(pattern|to|match)\b'

# Add handler method:
def _handle_new_intent(self, columns: List[str]) -> str:
    # Your logic here
    return "Response"

# Add to response_map in chat():
'new_intent': lambda: self._handle_new_intent(columns),
```

### Adding Synonyms

```python
# In advanced_chatbot.py, extend aggregation_synonyms:
self.aggregation_synonyms['new_word'] = 'existing_function'
```

## Limitations

1. **No ML/AI Understanding**: Uses rule-based NLP, not a language model
2. **English Only**: Designed for English queries
3. **Simple Filter Logic**: Complex multi-condition filters not supported
4. **No Memory Across Sessions**: Each session starts fresh

## Comparison with API-based Chatbots

| Feature | Local Chatbot | API-based |
|---------|---------------|-----------|
| Privacy | ✅ Data never leaves | ❌ Data sent to cloud |
| Cost | ✅ Free | ❌ Per-query cost |
| Speed | ✅ Instant | ⚠️ Network latency |
| Offline | ✅ Works offline | ❌ Requires internet |
| Accuracy | ⚠️ Rule-based | ✅ ML-powered |
| Flexibility | ⚠️ Fixed patterns | ✅ Natural language |

## File Structure

```
Dataset Cleaner and Analyzer/
├── app.py                 # Main Streamlit application
├── chatbot.py             # Core chatbot implementation
├── advanced_chatbot.py    # Enhanced chatbot with more features
├── nl_to_pandas.py        # NL to pandas translator
└── CHATBOT_README.md      # This documentation
```

## Future Improvements

1. **Sentence Transformers**: Add semantic similarity for better intent matching
2. **Context Memory**: Remember previous queries in conversation
3. **Visualization**: Auto-generate charts based on queries
4. **Export**: Save query results to CSV/Excel
5. **Custom Intents**: Let users define their own patterns

## Requirements

- Python 3.8+
- pandas
- numpy
- streamlit (for web interface)

No additional NLP libraries required - the chatbot is self-contained!
