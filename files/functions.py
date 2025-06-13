import re
import sqlite3
import pandas as pd
import logging
from typing import List, Optional
import ast
import streamlit as st
import datetime
from langchain_core.messages import ToolMessage, AIMessage
import yaml

#SQL関連の関数
#SQLのコードブロックがあった際にそれを削除
def extract_sql(sql_text):
    match = re.search(r"```sql\s*(.*?)```", sql_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", sql_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return sql_text.strip()

#SQLのアウトプットを必ずList[str]として後続処理に流す。
def parse_llm_list_output(llm_output_str: str) -> List[str]:
    try:
        cleaned_str = re.sub(r'^\s*[-*]\s*', '', llm_output_str, flags=re.MULTILINE)
        if not cleaned_str.strip().startswith('['):
            lines = [line.strip().replace("'", "\\'") for line in cleaned_str.split('\n') if line.strip()]
            if all(not (line.startswith("'") and line.endswith("'")) and \
                   not (line.startswith('"') and line.endswith('"')) for line in lines):
                 cleaned_str = "[" + ", ".join([f"'{line}'" for line in lines]) + "]"
            else:
                 cleaned_str = "[" + ", ".join(lines) + "]"
        parsed_list = ast.literal_eval(cleaned_str)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        return []
    except (ValueError, SyntaxError) as e:
        logging.warning(f"ast.literal_evalによるLLMリスト出力の解析に失敗しました: {e}。元の文字列: '{llm_output_str}'。改行による分割にフォールバックします。")
        str_for_split = llm_output_str.strip()
        if str_for_split.startswith('[') and str_for_split.endswith(']'):
            str_for_split = str_for_split[1:-1]
        suggestions = []
        for line in str_for_split.split('\n'):
            item = re.sub(r'^\s*[-*]\s*', '', line).strip()
            item = item.strip('\'",')
            if item:
                 suggestions.append(item)
        return suggestions

#SQLを実行
def try_sql_execute(sql_text):
    try:
        conn = sqlite3.connect("my_data.db")
        df = pd.read_sql(sql_text, conn)
        return df, None
    except Exception as e:
        logging.error(f"SQLの実行に失敗しました。エラー: {e}\nSQL: {sql_text}", exc_info=True)
        return None, str(e)

#SQLのエラーがあった際にエラー内容を日本語にする
def transform_sql_error(sql_error: str) -> str:
    if "no such table" in sql_error.lower():
        return f"指定されたテーブルが見つからなかったため、クエリを実行できませんでした。テーブル名を確認してください。(詳細: {sql_error})"
    elif "no such column" in sql_error.lower():
        return f"指定された列が見つからなかったため、クエリを実行できませんでした。列名を確認してください。(詳細: {sql_error})"
    elif "syntax error" in sql_error.lower():
        return f"SQL構文にエラーがあります。クエリを確認してください。(詳細: {sql_error})"
    else:
        return f"SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: {sql_error})"

#エラー内容に沿ってSQLを修正
def fix_sql_with_llm(llm, original_sql, error_message, rag_tables, rag_queries, task_description, context):
    prompt = f"""
    以下はユーザーの質問・関連情報・AIが生成したSQLとその実行時のエラー内容です。
    エラー内容を踏まえて、SQLを修正してください。
    SQLのみ出力し、前後のコメントや説明文は不要です。

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【テーブル定義に関する情報】
    {rag_tables}

    【類似する問い合わせ例とそのSQL】
    {rag_queries}

    【生成SQL】
    {original_sql}

    【エラー内容】
    {error_message}
    """
    response = llm.invoke(prompt)
    fixed_sql = extract_sql(response.content)
    return fixed_sql

def get_table_name_from_formatted_doc(doc_content: str) -> Optional[str]:
    for line in doc_content.split('\n'):
        line = line.strip()
        if line.startswith("<table_name>") and line.endswith("</table_name>"):
            return line[len("<table_name>"):-len("</table_name>")].strip()
    return None  # for全部回して見つからなかったらNone

# helpers.py
import json, logging
from langchain.schema.messages import AIMessage, ToolMessage

# superVisorのAIMessegeからTool_callを取り出して、エラーがないかを確認したうえでtask_descriptionとcontextを抽出する。
def fetch_tool_args(
        state: dict,
        required_keys: list[str],
        history_back_number: int = 6
    ) -> tuple[dict, str] | tuple[None, dict]:
    """
    supervisor の AIMessage → tool_call → args を取り出す。
    正常なら (args, tool_call_id) を返す。
    異常なら (None, error_dict) を返す。
    """
    try:
        supervisor_msg = state["messages"][-1]
        if not isinstance(supervisor_msg, AIMessage) or not supervisor_msg.tool_calls:
            raise ValueError("スーパーバイザーからの AIMessage(tool_calls 付き) が見つかりません。")

        call = supervisor_msg.tool_calls[0]
        tool_call_id = call["id"]
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("arguments の形式が dict ではありません。")

        # 必須キーチェック
        for k in required_keys:
            if k not in args:
                raise ValueError(f"{k} が arguments に含まれていません。")

        conversation = state.get("messages", [])
        context = "\n".join([
            f"{m.type}: {m.content}"
            for m in [msg for msg in conversation if msg.type != "system"][-history_back_number:]
        ])
        args["_history_context"] = context
        args["_tool_call_id"] = tool_call_id

        return args, None

    except Exception as e:
        return None, {"error_message":str(e), "tool_call_id":locals().get("tool_call_id", None)}
    
def plan_list_conv(plan, plan_cursor):
    """
    # 計画リストを整形し、現在のステップを視覚的に強調します
    """
    plan_list_str = []
    for i, step in enumerate(plan):
        if i < plan_cursor:
            prefix = f"  - [完了済] ステップ{i+1}"
        elif i == plan_cursor:
            prefix = f"> - [現在地] ステップ{i+1}" # 強調表示
        else:
            prefix = f"  - [着手予定] ステップ{i+1}"
        plan_list_str.append(f"{prefix} ({step['agent']}): {step['task']}")
    plan_str = "\n".join(plan_list_str)
    # 現在実行すべきタスクを明記します
    current_task = plan[plan_cursor]
    current_task_str = f"「{current_task['task']}」({current_task['agent']})"
    plan_now = f"""

    実行中の分析計画は以下のとおりです。これに従ってタスクを進めてください。
    == 実行中の計画 ==\n
    {plan_str}\n\n
    現在のタスクは {current_task_str} です。\n

    """
    return plan_now

def render_plan_sidebar():
    if not st.session_state.plan_steps:
        return

    steps = st.session_state.plan_steps
    cursor = st.session_state.plan_cursor
    done_ratio = max(0, min(cursor, len(steps))) # 進捗 0-1

    with st.sidebar:
        st.subheader("🗺️ プラン進捗")
        st.progress(done_ratio)

        for idx, step in enumerate(steps):
            if idx < cursor:
                icon = "✔️"     # 完了
            elif idx == cursor:
                icon = "🟢"     # 実行中
            else:
                icon = "□"      # 未着手
            st.markdown(f"{icon} **{step["task"]}**")

def extract_alerts(chunk):
    alerts = []   # list に変更

    for node_key, node_val in chunk.items():
        msgs = node_val.get("messages", [])
        for msg in msgs:
            try:
                payload = json.loads(msg.content) \
                    if isinstance(msg, (ToolMessage, AIMessage)) else None
            except json.JSONDecodeError:
                continue

            if payload and payload.get("status") in ("error", "warning"):
                alerts.append({
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "node": payload.get("node", node_key),
                    "status": payload["status"],
                    "summary": payload.get("summary", payload.get("error_message", "")),
                })
    return alerts   # 0 件なら []
def load_prompts(file_path):
    """YAMLファイルからプロンプトを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"プロンプトファイルが見つかりません: {file_path}")
        return {}
    except Exception as e:
        logging.error(f"プロンプトファイルの読み込み中にエラーが発生しました: {e}")
        return {}
