# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:
1. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Analysis Tools

The workflow currently runs five deterministic analysis tools:

1. **`profile_dataset`**
   Builds a structural profile (shape, dtypes, numeric summaries, top category frequencies) so downstream steps have a reliable schema baseline.
2. **`analyze_missingness`**
   Quantifies null counts/percentages and complete-row coverage to quickly surface data completeness risks.
3. **`compute_aggregates`**
   Adds first-pass business context with:
   - overall numeric rollups (sum/mean/median/std/min/max)
   - grouped aggregates on low-cardinality categorical columns
   - temporal aggregates (monthly and day-of-week) when a date-like column is detected
4. **`analyze_distributions`**
   Adds univariate diagnostics to surface shape and quality risks with:
   - per-column quantiles, skewness, and IQR-based outlier rates
   - zero-inflation and near-constant numeric feature flags
   - rare-category detection for low-frequency categorical values
5. **`analyze_relationships`**
   Adds interaction and quality checks with:
   - numeric correlation matrix and strongest correlation pairs
   - actionable correlation filtering to deprioritize obvious derived relationships (for example quantity vs total)
   - excluded-correlation reasons for traceability of what was filtered out
   - duplicate-row rate
   - high-cardinality categorical column detection (identifier-like fields)
   - suspicious categorical token detection (e.g., `unknown`, `error`, `missing`)
   - arithmetic consistency checks for quantity x unit price vs total when compatible columns exist

### Why These Additions

These three added tools (`compute_aggregates`, `analyze_distributions`, `analyze_relationships`) were chosen because they cover the highest-value questions in a first-pass EDA with minimal complexity:

- **Where is volume/value concentrated?** (grouped + temporal aggregates)
- **Which columns have outliers or heavy skew?** (distribution diagnostics)
- **Which variables move together?** (correlations)
- **Are there obvious integrity issues?** (duplicates, suspicious tokens, formula mismatches)

This keeps the workflow production-friendly: deterministic, fast, and interpretable before deeper modeling or hypothesis testing.
Observation extraction is also constrained to avoid generic tautologies and prioritize decision-relevant findings.

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.
If OpenAI or Mermaid network calls are unavailable, the script falls back to deterministic analysis without LLM synthesis.

## Project Structure

```
eda-agent/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py             # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
