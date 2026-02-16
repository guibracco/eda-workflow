import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    
    def compute_aggregates_node(state: EDAState):
        """Compute group-by aggregates on key columns.
        """
        logger.info("Computing aggregates")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        overall_numeric = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            overall_numeric[col] = {
                "sum": round(float(series.sum()), 4),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }
        
        grouped_aggregates = {}
        max_unique_for_groupby = min(25, max(2, int(len(df) * 0.1))) if len(df) > 0 else 2
        candidate_group_cols = []
        for col in categorical_cols:
            unique_count = df[col].nunique(dropna=True)
            if 2 <= unique_count <= max_unique_for_groupby:
                candidate_group_cols.append(col)
        
        for col in candidate_group_cols[:3]:
            grouped = df.groupby(col, dropna=False).size().to_frame("row_count")
            for metric_col in numeric_cols[:3]:
                grouped[f"{metric_col}__sum"] = (
                    df.groupby(col, dropna=False)[metric_col].sum()
                )
                grouped[f"{metric_col}__mean"] = (
                    df.groupby(col, dropna=False)[metric_col].mean()
                )
            grouped = grouped.sort_values("row_count", ascending=False).head(10)
            grouped = grouped.reset_index()
            grouped = grouped.where(pd.notnull(grouped), None)
            grouped_aggregates[col] = grouped.round(4).to_dict(orient="records")
        
        temporal_aggregates = {}
        parsed_dates = {}
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors="coerce")
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                lowered = col.lower()
                likely_date_name = any(
                    token in lowered for token in ["date", "time", "timestamp", "day", "month", "year"]
                )
                if not likely_date_name:
                    continue
                parsed = pd.to_datetime(df[col], errors="coerce")
            else:
                continue
            if parsed.notna().mean() >= 0.8:
                parsed_dates[col] = parsed
        
        if parsed_dates:
            date_col = max(parsed_dates, key=lambda c: parsed_dates[c].notna().mean())
            date_series = parsed_dates[date_col]
            date_df = df.copy()
            date_df["_date"] = date_series
            date_df = date_df.dropna(subset=["_date"])
            
            preferred_metric = None
            preferred_keywords = ["total", "amount", "revenue", "sales", "spent"]
            for keyword in preferred_keywords:
                for col in numeric_cols:
                    if keyword in col.lower():
                        preferred_metric = col
                        break
                if preferred_metric:
                    break
            if preferred_metric is None and numeric_cols:
                preferred_metric = numeric_cols[0]
            
            monthly_period = date_df["_date"].dt.to_period("M").astype(str)
            monthly = date_df.groupby(monthly_period).size().to_frame("row_count")
            day_of_week = date_df.groupby(date_df["_date"].dt.day_name()).size().to_frame("row_count")
            
            if preferred_metric:
                monthly[f"{preferred_metric}__sum"] = (
                    date_df.groupby(monthly_period)[preferred_metric].sum()
                )
                monthly[f"{preferred_metric}__mean"] = (
                    date_df.groupby(monthly_period)[preferred_metric].mean()
                )
                day_of_week[f"{preferred_metric}__sum"] = (
                    date_df.groupby(date_df["_date"].dt.day_name())[preferred_metric].sum()
                )
            
            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            day_of_week = day_of_week.reindex(day_order).fillna(0)
            
            temporal_aggregates = {
                "date_column": date_col,
                "metric_column": preferred_metric,
                "date_coverage": {
                    "min_date": str(date_df["_date"].min().date()),
                    "max_date": str(date_df["_date"].max().date()),
                    "rows_with_valid_dates": int(len(date_df)),
                },
                "monthly": monthly.sort_index().reset_index(
                    names="period"
                ).round(4).to_dict(orient="records"),
                "day_of_week": day_of_week.reset_index(
                    names="day"
                ).round(4).to_dict(orient="records"),
            }
        
        results["compute_aggregates"] = {
            "numeric_overview": overall_numeric,
            "grouped_aggregates": grouped_aggregates,
            "temporal_aggregates": temporal_aggregates,
        }
        
        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze relationships between variables.
        """
        logger.info("Analyzing relationships")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        def find_column(keywords: list[str]):
            for col in numeric_cols:
                lowered = col.lower()
                if any(keyword in lowered for keyword in keywords):
                    return col
            return None
        
        correlations = {}
        strongest_correlations = []
        actionable_correlations = []
        excluded_correlations = []
        
        quantity_col = find_column(["quantity", "qty", "units", "count"])
        unit_price_col = find_column(["price", "unit_price", "cost", "rate"])
        total_col = find_column(["total", "amount", "spent", "revenue", "sales"])
        
        structural_pairs = set()
        if quantity_col and total_col:
            structural_pairs.add(tuple(sorted((quantity_col, total_col))))
        if unit_price_col and total_col:
            structural_pairs.add(tuple(sorted((unit_price_col, total_col))))
        
        min_abs_correlation_for_actionable = 0.3
        
        def is_obvious_pair(col_a: str, col_b: str):
            pair = tuple(sorted((col_a, col_b)))
            if pair in structural_pairs:
                return "structural relationship with total-like field"
            
            explicit_tokens = {
                "id", "quantity", "qty", "unit", "price", "cost", "rate",
                "total", "amount", "spent", "revenue", "sales",
            }
            lower_tokens_a = set(col_a.lower().replace("-", " ").replace("_", " ").split())
            lower_tokens_b = set(col_b.lower().replace("-", " ").replace("_", " ").split())
            overlap = lower_tokens_a & lower_tokens_b
            if overlap & explicit_tokens:
                return "column names indicate overlapping/derived business metric semantics"
            return None
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().round(4)
            correlations = corr_matrix.to_dict()
            
            all_pairs = []
            for i, col_a in enumerate(corr_matrix.columns):
                for j, col_b in enumerate(corr_matrix.columns):
                    if j <= i:
                        continue
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.isna(corr_value):
                        continue
                    all_pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": round(float(corr_value), 4),
                        "abs_correlation": round(abs(float(corr_value)), 4),
                    })
            all_pairs = sorted(
                all_pairs,
                key=lambda x: x["abs_correlation"],
                reverse=True,
            )
            strongest_correlations = all_pairs[:5]
            
            for pair in all_pairs:
                obvious_reason = is_obvious_pair(pair["column_a"], pair["column_b"])
                if obvious_reason:
                    excluded_correlations.append({
                        **pair,
                        "reason": obvious_reason,
                    })
                    continue
                
                if pair["abs_correlation"] < min_abs_correlation_for_actionable:
                    excluded_correlations.append({
                        **pair,
                        "reason": (
                            f"below minimum abs(correlation) threshold "
                            f"({min_abs_correlation_for_actionable}) for actionable signal"
                        ),
                    })
                    continue
                
                actionable_correlations.append(pair)
                if len(actionable_correlations) >= 5:
                    break
        
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / len(df) * 100), 2) if len(df) > 0 else 0.0
        
        high_cardinality_columns = {}
        for col in categorical_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio > 0.9:
                high_cardinality_columns[col] = {
                    "unique_count": int(non_null.nunique()),
                    "non_null_count": int(len(non_null)),
                    "unique_ratio": round(float(unique_ratio), 4),
                }
        
        suspicious_tokens = {
            "unknown", "error", "n/a", "na", "none", "null", "missing", "other", "-"
        }
        suspicious_values = {}
        for col in categorical_cols:
            non_null = df[col].dropna().astype(str).str.strip()
            if non_null.empty:
                continue
            normalized = non_null.str.lower()
            flagged = normalized.isin(suspicious_tokens)
            if flagged.any():
                suspicious_counts = non_null[flagged].value_counts().to_dict()
                suspicious_values[col] = {
                    "count": int(flagged.sum()),
                    "percentage": round(float(flagged.sum() / len(non_null) * 100), 2),
                    "value_counts": {str(k): int(v) for k, v in suspicious_counts.items()},
                }
        
        formula_check = {
            "status": "not_applicable",
            "reason": "No compatible quantity, unit price, and total numeric columns found.",
        }
        if quantity_col and unit_price_col and total_col:
            subset_cols = [quantity_col, unit_price_col, total_col]
            id_col = next((col for col in df.columns if "id" in col.lower()), None)
            if id_col:
                subset_cols = [id_col] + subset_cols
            check_df = df[subset_cols].dropna(subset=[quantity_col, unit_price_col, total_col])
            if len(check_df) > 0:
                expected_total = check_df[quantity_col] * check_df[unit_price_col]
                absolute_error = (expected_total - check_df[total_col]).abs()
                mismatch_mask = absolute_error > 0.01
                mismatch_rows = check_df[mismatch_mask]
                formula_check = {
                    "status": "evaluated",
                    "quantity_column": quantity_col,
                    "unit_price_column": unit_price_col,
                    "total_column": total_col,
                    "rows_evaluated": int(len(check_df)),
                    "mismatch_count": int(mismatch_mask.sum()),
                    "mismatch_percentage": round(float(mismatch_mask.mean() * 100), 2),
                    "mean_absolute_error": round(float(absolute_error.mean()), 4),
                    "max_absolute_error": round(float(absolute_error.max()), 4),
                    "sample_mismatches": mismatch_rows.head(5).to_dict(orient="records"),
                }
        
        results["analyze_relationships"] = {
            "numeric_correlations": correlations,
            "strongest_correlations": strongest_correlations,
            "actionable_correlations": actionable_correlations,
            "excluded_correlations": excluded_correlations[:10],
            "duplicate_rows": {
                "count": duplicate_rows,
                "percentage": duplicate_pct,
            },
            "high_cardinality_columns": high_cardinality_columns,
            "suspicious_categorical_values": suspicious_values,
            "formula_check": formula_check,
        }
        
        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations}
        
        step_results = results.get(current_step, {})
        
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations
        
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {})
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")
        
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
        
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
