from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import json
import pandas as pd
import os
import uuid


class DispatchDecision(BaseModel):
    """専門家エージェントにタスクを割り振る際の意思決定スキーマ"""
    next_agent: str = Field(description="次に指名すべき専門家（ノード）の名前。利用可能な専門家がいない場合は 'FINISH'。")
    task_description: str = Field(description="指名した専門家に与える、具体的で明確な指示内容。")
    rationale: str = Field(description="なぜその判断を下したのかの簡潔な理由。")

# 引数スキーマ ― code 文字列だけ渡す
class ExecArgs(BaseModel):
    code: str = Field(description="Python source code to execute")

class PythonExecTool(BaseTool):
    """
    ・コンストラクタで exec_scope（locals）を受け取って保持
    ・_run では code だけを受け取り scope で実行
    """
    name: str = "python_exec"
    description: str = ""
    args_schema: type[BaseModel] = ExecArgs 
    locals: dict  = {}


    def __init__(self, *, locals: dict, **kwargs):
        super().__init__(**kwargs)
        self._scope = locals          # ← ここに実行スコープを保持

    def _run(self, code: str) -> dict:
        banned = ("import os", "subprocess", "__")
        if any(b in code for b in banned) or len(code) > 8000:
            raise ValueError("unsafe code")

        exec(compile(code, "<llm_code>", "exec"), self._scope, self._scope)

        final_df = self._scope.get("final_df")
        chart = self._scope.get("chart")
        altair_chart_json = chart.to_json() if chart is not None else None

        payload = {
            "altair_chart_json": altair_chart_json,
            "final_df_json": (
                final_df.to_json(orient="records", force_ascii=False)
                if isinstance(final_df, pd.DataFrame) else None
            )
        }

        return json.dumps(payload, ensure_ascii=False)

class NodeStatus(str, Enum):
    success = "success"
    error = "error"

class NodeType(str, Enum):
    sql_node= "sql_node"
    processing_node= "processing_node"
    interpret_node= "interpret_node"
    planning_node= "planning_node"
    ask_user_node= "ask_user_node"
    metadata_retrieval_node= "metadata_retrieval_node"
    

class NodeResponse(BaseModel):
    status: NodeStatus # 成功 / 失敗
    node: NodeType # nodeの種類
    summary: str = Field(min_length=1) # 1 行サマリ
    result_payload: Dict[str, Any] | None = None # 任意の自由領域
