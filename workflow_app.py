from __future__ import annotations
import json
import subprocess
import base64
from typing import List, Dict, Any, TypedDict

import pandas as pd
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import RetryPolicy
from langchain_openai import ChatOpenAI


class Step(BaseModel):
    op: str
    expr: str | None = None
    new: str | None = None
    by: List[str] | None = None
    aggs: Dict[str, str] | None = None


class TransformPlan(BaseModel):
    source: str
    steps: List[Step]


class AgentState(TypedDict, total=False):
    input: str
    df_raw: pd.DataFrame
    dsl: dict
    df_processed: pd.DataFrame
    option_json: str
    chart_png: bytes


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def llm_transform_planner(state: AgentState) -> Dict[str, Any]:
    cols = ", ".join(state["df_raw"].columns)
    prompt = f"入力:{state['input']}\n列:{cols}\nTransformPlan JSONのみ"
    resp = llm.invoke(prompt)
    state["dsl"] = json.loads(resp.content)
    return {"dsl": state["dsl"]}


def transform_exec(state: AgentState) -> Dict[str, Any]:
    plan = TransformPlan.model_validate(state["dsl"])
    df = state["df_raw"].copy()
    for s in plan.steps:
        if s.op == "filter":
            df = df.query(s.expr)
        elif s.op == "mutate":
            df[s.new] = df.eval(s.expr)
        elif s.op == "groupby":
            df = df.groupby(s.by).agg(s.aggs).reset_index()
        elif s.op == "sort":
            df = df.sort_values(s.by)
    state["df_processed"] = df
    return {"df_processed": df}


def llm_echarts_option(state: AgentState) -> Dict[str, Any]:
    cols = ", ".join(state["df_processed"].columns)
    prompt = (
        f"入力:{state['input']}\n列:{cols}\n"
        "dataset/transformを使いseries[0].datasetId='transformed'にしたECharts option JSONを返答"
    )
    resp = llm.invoke(prompt)
    state["option_json"] = resp.content.strip()
    return {"option_json": state["option_json"]}


def echarts_exec_tool(state: AgentState) -> Dict[str, Any]:
    opt = state["option_json"]
    b64 = subprocess.check_output(["node", "run_echarts.js", opt], text=True)
    state["chart_png"] = base64.b64decode(b64)
    return {"chart_png": state["chart_png"]}


graph = StateGraph(AgentState)

graph.add_node("llm_plan", llm_transform_planner)

graph.add_node(
    "transform_exec", transform_exec, retry_policy=RetryPolicy(max_attempts=2)
)

graph.add_node("llm_echarts_option", llm_echarts_option)

graph.add_node(
    "echarts_exec", echarts_exec_tool, retry_policy=RetryPolicy(max_attempts=2)
)

graph.set_entry_point("llm_plan")

graph.connect("llm_plan", "transform_exec")

graph.connect("transform_exec", "llm_echarts_option")

graph.connect("llm_echarts_option", "echarts_exec")

graph.connect("echarts_exec", END)

workflow = graph.compile()


if __name__ == "__main__":
    df_raw = pd.DataFrame(
        {
            "date": pd.date_range("2025-06-01", periods=5, freq="D"),
            "region": ["関東", "関西", "関東", "関東", "関西"],
            "sales": [100, 80, 120, 90, 110],
        }
    )
    init_state = {"input": "関東の日別売上を棒グラフ", "df_raw": df_raw}
    result = workflow.invoke(init_state)
    with open("chart.png", "wb") as f:
        f.write(result["chart_png"])
