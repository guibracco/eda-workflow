"""
Example usage of the EDA Workflow with OpenAI.

This demonstrates how to:
1. Initialize the OpenAI model
2. Create an EDA workflow
3. Run analysis on a dataset
4. Retrieve results

Requires: OPENAI_API_KEY in .env file or environment variable
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

# Load environment variables from .env file
load_dotenv()

# Path to sample dataset
data_path = os.path.join("data", "cafe_sales.csv")

# Initialize OpenAI model when available
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
else:
    llm = None
    print("OPENAI_API_KEY not found. Running workflow without LLM.\n")

# Create EDA workflow with the model
workflow = EDAWorkflow(model=llm)

# Save a visual diagram of the graph (best effort, requires network access by default)
try:
    workflow._compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Graph diagram saved to graph.png\n")
except Exception as exc:
    print(f"Skipping graph diagram export: {exc}\n")

# Run analysis on the dataset
print("Running EDA analysis...\n")
try:
    workflow.invoke_workflow(data_path)
except Exception as exc:
    if llm is not None:
        print(f"LLM-backed run failed, retrying without LLM: {exc}\n")
        workflow = EDAWorkflow(model=None)
        workflow.invoke_workflow(data_path)
    else:
        raise

# Retrieve results
summary = workflow.get_summary()
recommendations = workflow.get_recommendations()
observations = workflow.get_observations()
results = workflow.get_results()

# Display results sequentially: tool result → observations → next tool
analysis_steps = [
    ("profile_dataset", "Dataset Profile"),
    ("analyze_missingness", "Missingness Analysis"),
    ("compute_aggregates", "Aggregates Analysis"),
    ("analyze_distributions", "Distribution Analysis"),
    ("analyze_relationships", "Relationships Analysis"),
]

for step_key, step_title in analysis_steps:
    print("=" * 60)
    print(f"{step_title.upper()}")
    print("=" * 60)
    
    # Show results
    if step_key in results:
        step_results = results[step_key]
        for key, value in step_results.items():
            if isinstance(value, dict) and len(str(value)) > 200:
                print(f"{key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"{key}: {value}")
    
    # Show observations
    print(f"\nObservations:")
    if step_key in observations and observations[step_key]:
        for obs in observations[step_key]:
            print(f"  • {obs}")
    else:
        print("  (No observations)")
    print()

# Final synthesis
print("=" * 60)
print("FINAL SYNTHESIS")
print("=" * 60)
print(f"\nSummary:\n{summary if summary else '(Not implemented yet)'}")
print("\nRecommendations:")
if recommendations:
    for rec in recommendations:
        print(f"  • {rec}")
else:
    print("  (Not implemented yet)")
