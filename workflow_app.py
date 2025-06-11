from __future__ import annotations
from typing import List, Dict, TypedDict, Any
import pandas as pd
import plotly.express as px
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END, RetryPolicy
class Step(BaseModel):
    op: str
    expr: str | None = None
    new: str | None = None
    by: List[str] | None = None
    aggs: Dict[str, str] | None = None
class TransformPlan(BaseModel):
    source: str
    steps: List[Step]
class AgentState(TypedDict):
    input: str
    df_raw: pd.DataFrame
    dsl: dict | None
    df_processed: pd.DataFrame | None
    chart_code: str | None
    chart_json: str | None
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = PydanticOutputParser(pydantic_object=TransformPlan)
def llm_transform_planner(state: AgentState) -> AgentState:
    cols = ", ".join(state["df_raw"].columns)
    prompt = (
        "You are a data engineer. Given the columns "
        f"{cols} and the request '{state['input']}', "
        "return only JSON for TransformPlan."
    )
    response = llm.invoke(prompt)
    plan = parser.parse(response.content)
    state["dsl"] = plan.model_dump()
    return state
def transform_exec(state: AgentState) -> AgentState:
    plan = TransformPlan.model_validate(state["dsl"])
    df = state["df_raw"]
    for step in plan.steps:
        if step.op == "filter" and step.expr:
            df = df.query(step.expr)
        elif step.op == "mutate" and step.expr and step.new:
            df[step.new] = df.eval(step.expr)
        elif step.op == "groupby" and step.by and step.aggs:
            df = df.groupby(step.by).agg(step.aggs).reset_index()
        elif step.op == "sort" and step.by:
            df = df.sort_values(step.by)
    state["df_processed"] = df
    return state
def llm_chart_code(state: AgentState) -> AgentState:
    cols = ", ".join(state["df_processed"].columns)
    prompt = (
        "You are a plotting assistant. Using plotly.express (px) or graph_objects, "
        "write Python code that assigns a Figure to variable 'fig'. "
        f"Columns: {cols}. Request: {state['input']}."
    )
    response = llm.invoke(prompt)
    state["chart_code"] = response.content
    return state
class PythonExecTool:
    def _run(self, code: str, state: AgentState) -> str:
        g: Dict[str, Any] = {"pd": pd, "px": px, "fig": None, "df": state["df_processed"]}
        l: Dict[str, Any] = {}
        exec(code, g, l)
        fig = l.get("fig") or g.get("fig")
        return fig.to_json() if fig is not None else ""
def python_exec(state: AgentState) -> AgentState:
    tool = PythonExecTool()
    state["chart_json"] = tool._run(state["chart_code"], state)
    return state
def build_workflow() -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("llm_transform_planner", llm_transform_planner)
    builder.add_node("transform_exec", transform_exec, retry=RetryPolicy(max_attempts=2))
    builder.add_node("llm_chart_code", llm_chart_code)
    builder.add_node("python_exec", python_exec, retry=RetryPolicy(max_attempts=2))
    builder.add_edge("llm_transform_planner", "transform_exec")
    builder.add_edge("transform_exec", "llm_chart_code")
    builder.add_edge("llm_chart_code", "python_exec")
    builder.add_edge("python_exec", END)
    builder.set_entry_point("llm_transform_planner")
    return builder.compile()
if __name__ == "__main__":
    df_raw = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "region": ["関東", "関西", "関東"],
        "sales": [100, 150, 200],
    })
    init_state = {"input": "関東の日別売上グラフ", "df_raw": df_raw}
    workflow = build_workflow()
    result = workflow.invoke(init_state)
    print(result["chart_json"][:200])
