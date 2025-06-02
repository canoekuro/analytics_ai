# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

from langchain_community.vectorstores import FAISS
import difflib # Make sure this import is added at the top of the file
import sqlite3
import pandas as pd
import logging
from langgraph.graph import StateGraph, END
from langchain.agents import Tool, initialize_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import os
import base64
import re
from typing import TypedDict, List, Optional, Any
import ast # For literal_eval
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
import japanize_matplotlib
import seaborn as sns
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import json # Moved import json to the top

# Configure basic logging
logging.basicConfig(level=logging.INFO)

google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) # Or other suitable Gemini embedding model
SIMILARITY_THRESHOLD = 0.8

# ベクトルストア
vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)

# Get LLM model name from environment variable, with a default
llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-lite")

llm = ChatGoogleGenerativeAI(
    model=llm_model_name,
    temperature=0,
    google_api_key=google_api_key
) # Or "gemini-1.5-flash", "gemini-1.5-pro" for higher capabilities

DATA_CLEARED_MESSAGE = "データが正常にクリアされました。"


import collections
class MyState(TypedDict, total=False):
    input: str                       # ユーザーの問い合わせ
    condition: Optional[str]         # 各ノードの実行状態
    intent_list: List[str]           # 分類結果（データ取得/グラフ作成/データ解釈）
    latest_df: Optional[collections.OrderedDict[str, list]] # Changed: Dict mapping requirement to its dataframe (list of records)
    df_history: Optional[List[dict]] # SQL実行結果のDataFrameの履歴 {"id": str, "query": str, "timestamp": str, "dataframe_dict": list, "SQL": Optional[str]}
    SQL: Optional[str]               # 生成されたSQL
    interpretation: Optional[str]    # データ解釈（分析コメント）
    chart_result: Optional[str]      # グラフ画像（base64など）
    metadata_answer: Optional[str]   # メタデータ検索結果の回答
    error: Optional[str]             # SQL等でエラーがあれば
    query_history: Optional[List[str]] # ユーザーの問い合わせ履歴
    data_requirements: List[str]     # データ要件
    missing_data_requirements: Optional[List[str]] # New: List of data requirements not found in history
    clarification_question: Optional[str] = None
    analysis_options: Optional[List[str]] = None
    user_clarification: Optional[str] = None
    analysis_plan: Optional[List[dict]] = None # Stores the list of analysis steps, e.g., [{"action": "sql", "details": "Overall monthly sales"}, {"action": "interpret", "details": "Interpret monthly sales data"}]
    current_plan_step_index: Optional[int] = None # Index of the current step in analysis_plan
    awaiting_step_confirmation: Optional[bool] = None # True if waiting for user to proceed to next step
    complex_analysis_original_query: Optional[str] = None # Stores the original multi-step query
    user_action: Optional[str] = None # New field for frontend actions like "proceed_analysis_step" or "cancel_analysis_plan"

#ユーザーの入力に応じて意図を分類
def classify_intent_node(state):
    user_input = state["input"]
    if user_input == "SYSTEM_CLEAR_HISTORY":
        current_history = state.get("query_history", [])
        return {
            **state,
            "intent_list": ["clear_data_intent"],
            "condition": "分類完了",
            "query_history": current_history
        }

    current_history = state.get("query_history", [])
    if not current_history or current_history[-1] != user_input:
         current_history.append(user_input)

    prompt = f"""
    ユーザーの質問の意図を判定してください。
    次の4つのうち、該当するものを「,」区切りで全て列挙してください：
    - データ取得
    - グラフ作成
    - データ解釈
    - メタデータ検索

    質問: 「{user_input}」
    例: 
    input:「カテゴリの合計販売金額を出して」 output:「データ取得」
    input:「＊＊＊のデータを出して」 output:「データ取得」
    input:「＊＊＊のデータを取得してグラフ化して」 output:「データ取得,グラフ作成」
    input:「＊＊＊のデータを取得して解釈して」 output:「データ取得,データ解釈」
    input:「＊＊＊のデータのグラフ化と解釈して」 output:「データ取得,グラフ作成,データ解釈」
    input:「sales_dataテーブルにはどんなカラムがありますか？」 output:「メタデータ検索」
    input:「categoryカラムの情報を教えてください」 output:「メタデータ検索」

    他の情報は不要なので、outputの部分（「データ取得,グラフ作成,データ解釈,メタデータ検索」）だけを必ず返すようにしてください。
    """
    result = llm.invoke(prompt).content.strip()
    steps = [x.strip() for x in result.split(",") if x.strip()]

    return {**state, "intent_list": steps, "condition": "分類完了", "query_history": current_history}

def metadata_retrieval_node(state):
    user_query = state["input"]
    retrieved_docs = vectorstore_tables.similarity_search(user_query)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_template = """
    以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。
    ユーザーが理解しやすいように、テーブルやカラムの役割、データ型、関連性などを説明してください。

    【テーブル定義情報】
    {retrieved_table_info}

    【ユーザー質問】
    {user_query}

    【回答】
    """
    llm_prompt = prompt_template.format(retrieved_table_info=retrieved_table_info, user_query=user_query)
    response = llm.invoke(llm_prompt)
    answer = response.content.strip()
    return {**state, "metadata_answer": answer, "condition": "メタデータ検索完了"}

def clear_data_node(state):
    return {
        "input": state.get("input"),
        "intent_list": state.get("intent_list"),
        "latest_df": collections.OrderedDict(),
        "df_history": [],
        "SQL": None,
        "interpretation": DATA_CLEARED_MESSAGE,
        "chart_result": None,
        "metadata_answer": state.get("metadata_answer"),
        "condition": "データクリア完了",
        "error": None,
        "query_history": []
    }
    
# New Analysis Planning Node (create_analysis_plan_node)
def create_analysis_plan_node(state: MyState) -> MyState:
    user_query = state.get("input", "")
    user_clarification = state.get("user_clarification")

    # Mocking Strategy
    mock_plan = []
    plan_json_str = ""

    if user_clarification:
        original_query_for_llm = state.get("complex_analysis_original_query", user_query)
        mock_plan = [
            {"action": "check_history", "details": ["clarified sales data"]},
            {"action": "sql", "details": "clarified sales data"},
            {"action": "interpret", "details": "interpret clarified sales data"}
        ]
    elif "ambiguous" in user_query.lower() or "vague" in user_query.lower():
        mock_plan = [{"action": "clarify", "details": "Could you please specify the time period for the sales data?"}]
    elif "show sales then chart it" in user_query.lower():
        mock_plan = [
            {"action": "check_history", "details": ["sales data"]},
            {"action": "sql", "details": "sales data"},
            {"action": "chart", "details": "chart sales data"}
        ]
    elif "show sales" in user_query.lower() or "interpret data" in user_query.lower():
        data_req = "sales data" if "sales" in user_query.lower() else "general data"
        mock_plan = [
            {"action": "check_history", "details": [data_req]},
            {"action": "sql", "details": data_req},
            {"action": "interpret", "details": f"interpret {data_req}"}
        ]
    elif "hello" in user_query.lower() or "hi" in user_query.lower():
        mock_plan = []
    else:
        mock_plan = [
            {"action": "check_history", "details": ["general data from query"]},
            {"action": "sql", "details": "general data from query"},
            {"action": "interpret", "details": "interpret general data from query"}
        ]

    try:
        plan_json_str = json.dumps(mock_plan)
    except Exception as e:
        logging.error(f"Error serializing mock_plan to JSON: {e}")
        return {
            **state,
            "analysis_plan": None,
            "current_plan_step_index": None,
            "user_clarification": None,
            "condition": "plan_generation_failed",
            "error": f"分析計画の保存中にエラーが発生しました。: {e}"
        }

    parsed_plan = None
    try:
        parsed_plan = json.loads(plan_json_str)
        if not isinstance(parsed_plan, list):
            raise ValueError("分析プランがリスト型になっていません。")
        for step in parsed_plan:
            if not isinstance(step, dict) or "action" not in step or "details" not in step:
                raise ValueError("分析プラン内の各ステップが正しい形式ではありません（'action'と'details'必須）。")

    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"分析計画のパース・検証に失敗: {e}. Plan string: '{plan_json_str}'")
        return {
            **state,
            "analysis_plan": None,
            "current_plan_step_index": None,
            "user_clarification": None,
            "condition": "plan_generation_failed",
            "error": f"分析計画の形式や内容に不備があります: {e}"
        }

    current_input = state.get("input", "")
    original_query_for_complex = state.get("complex_analysis_original_query", current_input if parsed_plan else None)
    if user_clarification and not state.get("complex_analysis_original_query"):
        original_query_for_complex = current_input

    return {
        **state,
        "complex_analysis_original_query": original_query_for_complex,
        "analysis_plan": parsed_plan,
        "current_plan_step_index": 0 if parsed_plan else None,
        "awaiting_step_confirmation": False,
        "user_clarification": None,
        "condition": "plan_generated" if parsed_plan else "empty_plan_generated",
        "error": None
    }

#分析計画に基づいて、ユーザーに質問
def clarify_node(state: MyState) -> MyState:
    current_plan_step_index = state.get("current_plan_step_index")
    analysis_plan = state.get("analysis_plan", [])
    if current_plan_step_index is None or not analysis_plan or not (0 <= current_plan_step_index < len(analysis_plan)):
        logging.error("clarify_node:分析プランの状態が不正です。")
        return {**state, "error": "分析プランの状態が不正です。", "condition": "clarify_error_invalid_plan"}

    current_step = analysis_plan[current_plan_step_index]
    action = current_step.get("action")

    if action == "clarify":
        clarification_question_from_plan = current_step.get("details")
        if not isinstance(clarification_question_from_plan, str) or not clarification_question_from_plan.strip():
            logging.warning("clarify_node: プランに具体的な質問文がありません。")
            clarification_question_from_plan = "追加で確認したい内容がありますが、質問文がプランに見当たりません。"

        return {
            **state,
            "clarification_question": clarification_question_from_plan,
            "user_clarification": None,
            "condition": "awaiting_user_clarification"
        }
    else:
        logging.error(f"clarify_node: 想定外のアクションが渡されました。アクションの内容はこちらです。'{action}'")
        return {
            **state,
            "error": f"想定外のアクションが渡されました。アクションの内容はこちらです。 {action}",
            "condition": "clarify_error_unexpected_action"
        }

# 追加質問をした際に、実行をキャンセルした際のノード
def cancel_analysis_plan_node(state: MyState) -> MyState:
    original_query = state.get("complex_analysis_original_query", "The analysis plan was cancelled.")
    return {
        **state,
        "analysis_plan": None,
        "current_plan_step_index": None,
        "awaiting_step_confirmation": False,
        "complex_analysis_original_query": None,
        "input": original_query,
        "interpretation": "The multi-step analysis plan has been cancelled by the user.",
        "condition": "plan_cancelled",
        "user_action": None
    }

# 過去取得したデータを参照するノード
def check_history_node(state: MyState) -> MyState:
    analysis_plan = state.get("analysis_plan")
    current_plan_step_index = state.get("current_plan_step_index")

    if not analysis_plan or not isinstance(analysis_plan, list) or not analysis_plan:
        # プランがNone、リスト型でない、または空リストの場合はエラーとして返す
        return {
            **state,
            "latest_df": collections.OrderedDict(),
            "missing_data_requirements": [],
            "condition": "history_checked_invalid_plan_or_index",
            "error": "過去のデータをチェックしようとしましたが分析プランがありませんでした。"
        }

    if current_plan_step_index is None or \
       not isinstance(current_plan_step_index, int) or \
       not (0 <= current_plan_step_index < len(analysis_plan)):
         # インデックスがNone、int型でない、範囲外の場合はエラーとして返す
        safe_details_for_error_msg = []
        if analysis_plan and \
           isinstance(analysis_plan, list) and \
           current_plan_step_index is not None and \
           isinstance(current_plan_step_index, int) and \
           0 <= current_plan_step_index < len(analysis_plan) and \
           isinstance(analysis_plan[current_plan_step_index], dict):
            safe_details_for_error_msg = analysis_plan[current_plan_step_index].get("details", [])

        return {
            **state,
            "latest_df": collections.OrderedDict(),
            "missing_data_requirements": safe_details_for_error_msg,
            "condition": "history_checked_invalid_plan_or_index",
            "error": f"Invalid current_plan_step_index ({current_plan_step_index}) for analysis_plan length ({len(analysis_plan) if analysis_plan else 0}) in check_history_node."
        }

    # df_historyを取得する
    df_history = state.get("df_history", [])
    data_requirements_for_step = []
    current_step_details = state.get("input") # This is set by execute_plan_router to current_step["details"]

    # inputが文字列の場合はリスト化、リストならそのまま
    if isinstance(current_step_details, str):
        data_requirements_for_step = [current_step_details]
    elif isinstance(current_step_details, list):
        data_requirements_for_step = [str(item) for item in current_step_details]
    
    # 以下のブロックは、inputに値が入っていない場合にプランからdetailsを取得していますが、
    # execute_plan_routerが常にstate.inputにplan[index].detailsを正しくセットする場合は冗長かもしれません。
    # ただし、堅牢性確保や、このノードが通常のルーターフロー以外から呼ばれた場合も考慮して残しています。
    if not data_requirements_for_step:
        # inputが正しくセットされていない場合、プランのdetailsから救済する（想定外呼び出し時の保険）
        current_step = analysis_plan[current_plan_step_index]
        raw_details = current_step.get("details")
        logging.warning(f"過去のデータ履歴チェック中に不正なステップ番号（{current_plan_step_index}）が指定されました: {raw_details}")
        if isinstance(raw_details, str):
            data_requirements_for_step = [raw_details]
        elif isinstance(raw_details, list):
            data_requirements_for_step = [str(item) for item in raw_details]

    if not data_requirements_for_step:
        # それでも要件が取得できない場合はmissing_requirementsは空で返却
        return {
            **state,
            "latest_df": collections.OrderedDict(),
            "missing_data_requirements": [],
            "condition": "history_checked_no_requirements",
            "SQL": None, "interpretation": None, "chart_result": None
        }

    found_data_map = collections.OrderedDict()
    missing_requirements = []
    used_history_ids = set()

    # 各要件について、過去履歴から最も類似するデータを1件だけ再利用する
    for req in data_requirements_for_step:
        found_for_req = False
        best_similarity_for_req = 0.0
        best_match_entry = None
        for entry in df_history:
            if entry.get("id") in used_history_ids:
                continue
            similarity = difflib.SequenceMatcher(None, req, entry.get("query", "")).ratio()
            if similarity >= SIMILARITY_THRESHOLD and similarity > best_similarity_for_req:
                best_similarity_for_req = similarity
                best_match_entry = entry
        if best_match_entry:
            found_data_map[req] = best_match_entry["dataframe_dict"]
            used_history_ids.add(best_match_entry.get("id"))
        else:
            missing_requirements.append(req)

    return {
        **state,
        "latest_df": found_data_map,
        "missing_data_requirements": missing_requirements,
        "condition": "history_checked",
        "SQL": None, "interpretation": None, "chart_result": None
    }

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
        logging.warning(f"Failed to parse LLM list output with ast.literal_eval: {e}. Original string: '{llm_output_str}'. Falling back to newline split.")
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
        logging.error(f"SQL execution failed. Error: {e}\nSQL: {sql_text}", exc_info=True)
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
def fix_sql_with_llm(original_sql, error_message, rag_tables, rag_queries, user_query):
    prompt = f"""
    以下はユーザーの質問・関連情報・AIが生成したSQLとその実行時のエラー内容です。
    エラー内容を踏まえて、SQLを修正してください。
    SQLのみ出力し、前後のコメントや説明文は不要です。

    【ユーザー質問】
    {user_query}

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

#SQLの生成と実行を行うノード
def sql_node(state: MyState) -> MyState:
    # check_historyやルータでセットされるstate.missing_data_requirementsをデータ取得要件とする。
    requirements_to_fetch = state.get("missing_data_requirements", [])

    if not requirements_to_fetch:
        logging.info("sql_node: 取得すべき要件が見つからなかったためSQLの作成・実行をスキップします。")
        return {
            **state,
            "condition": "sql_execution_skipped_no_reqs",
            "SQL": None,
            "error": "No requirements provided for SQL execution."
        }

    # latest_dfは必ずOrderedDictにして、すでにデータがあれば引き継ぐ
    current_latest_df = state.get("latest_df", collections.OrderedDict())
    if not isinstance(current_latest_df, collections.OrderedDict):
        current_latest_df = collections.OrderedDict(current_latest_df or {})

    # 全体の質問文脈（Clarify含む）があればそちら、なければ直近入力
    current_df_history = state.get("df_history", [])
    overall_user_input_context = state.get("complex_analysis_original_query", state.get("input", ""))
    
    last_sql_generated = None
    any_error_occurred = False
    successfully_fetched_reqs = [] # 今回取得できた要件を記録
    accumulated_errors = []

    system_prompt_sql_generation = """
    あなたはSQL生成AIです。この後の質問に対し、SQLiteの標準的なSQL文のみを出力してください。
    - 使用できるSQL構文は「SELECT」「WHERE」「GROUP BY」「ORDER BY」「LIMIT」のみです。
    - 日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    - 必ず1つのテーブルだけを使い、簡単な集計・フィルタ・並べ替えまでにしてください。
    - SQLの前後にコメントや説明文は出力しないでください。出力された内容をそのまま実行するため、SQL文のみをそのまま出力してください。
    """

    for req_string in requirements_to_fetch:
        if not req_string or not isinstance(req_string, str):
            logging.warning(f"sql_node: 無効な要件のためスキップ: {req_string}")
            continue

        logging.info(f"sql_node: 要件を処理中: '{req_string}'")
        retrieved_tables_docs = vectorstore_tables.similarity_search(req_string, k=3)
        rag_tables = "\n".join([doc.page_content for doc in retrieved_tables_docs])
        retrieved_queries_docs = vectorstore_queries.similarity_search(req_string, k=3)
        rag_queries = "\n".join([doc.page_content for doc in retrieved_queries_docs])

        user_prompt_for_req = f"""
        【ユーザーの全体的な質問の文脈】
        {overall_user_input_context}
        【テーブル定義に関する情報】
        {rag_tables}
        【類似する問い合わせ例とそのSQL】
        {rag_queries}
        【現在の具体的なデータ取得要件】
        「{req_string}」
        この要件を満たすためのSQLを生成してください。
        """
        response = llm.invoke([
            {"role": "system", "content": system_prompt_sql_generation},
            {"role": "user", "content": user_prompt_for_req}
        ])
        sql_generated_clean = extract_sql(response.content.strip())
        original_sql_for_logging = sql_generated_clean
        last_sql_generated = sql_generated_clean # Store the latest SQL attempt for this requirement
        result_df, sql_error = try_sql_execute(sql_generated_clean)

        if sql_error:
            logging.warning(f"Initial SQL for '{req_string}' failed: {sql_error}. Retrying with fix.")
            fixed_sql = fix_sql_with_llm(sql_generated_clean, sql_error, rag_tables, rag_queries, req_string)
            last_sql_generated = fixed_sql
            result_df, sql_error = try_sql_execute(fixed_sql)
            if sql_error:
                logging.error(f"Fixed SQL for '{req_string}' also failed: {sql_error}.")
                any_error_occurred = True
                user_friendly_error = transform_sql_error(sql_error)
                accumulated_errors.append(f"For '{req_string}': {user_friendly_error}")

        if not sql_error and result_df is not None:
            result_df_dict = result_df.to_dict(orient="records")
            current_latest_df[req_string] = result_df_dict # 取得したデータをlatest_dfに保存
            successfully_fetched_reqs.append(req_string)
            new_history_entry = {
                "id": uuid.uuid4().hex[:8], "query": req_string,
                "timestamp": datetime.now().isoformat(),
                "dataframe_dict": result_df_dict, "SQL": last_sql_generated
            }
            current_df_history.append(new_history_entry)
        elif not sql_error and result_df is None: # エラーはないがデータが0件
             current_latest_df[req_string] = []
             successfully_fetched_reqs.append(req_string)
             logging.info(f"SQL for '{req_string}' は正常にSQL実行されましたが、データがありませんでした。")

    # 最終的なconditionを判定
    if not requirements_to_fetch: 
        final_condition = "sql_execution_skipped_no_reqs"
    elif not successfully_fetched_reqs and any_error_occurred: # SQLが全部失敗
        final_condition = "sql_execution_failed"
    elif any_error_occurred: # 一部成功・一部失敗
        final_condition = "sql_execution_partial_success"
    else: #全取得成功
        final_condition = "sql_execution_done"

    # 取得できなかったものだけmissing_data_requirementsとして再セット
    current_missing_reqs_in_state = state.get("missing_data_requirements", [])
    updated_missing_requirements = [req for req in current_missing_reqs_in_state if req not in successfully_fetched_reqs]
    
    return {
        **state,
        "latest_df": current_latest_df,
        "df_history": current_df_history,
        "SQL": last_sql_generated, # Stores the very last SQL executed in the loop
        "condition": final_condition,
        "error": "; ".join(accumulated_errors) if accumulated_errors else None,
        "missing_data_requirements": updated_missing_requirements
    }

def interpret_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")
    # 解釈用の文脈は state.input（execute_plan_router が plan step details からセット）を利用
    plan_details_context = state.get("input", "全般的な傾向") # Default if no specific detail

    if latest_df_data is None:
        logging.info("interpret_node: latest_dfがNoneのため、解釈できるデータがありません。")
        return {**state, "interpretation": "解釈するデータがありません。", "condition": "interpretation_failed_no_data"}

    if not isinstance(latest_df_data, collections.OrderedDict):
        logging.error(f"interpret_node: latest_dfの型がOrderedDictではありません (type: {type(latest_df_data)}).")
        return {**state, "interpretation": "データの形式が不正です (OrderedDictではありません)。", "condition": "interpretation_failed_bad_format"}

    if not latest_df_data: # Check if the OrderedDict is empty
        logging.info("interpret_node: latest_dfが空（データ無し）です。")
        return {**state, "interpretation": "解釈するデータが空です。", "condition": "interpretation_failed_empty_data"}

    full_data_string = ""
    
    # 各要件ごとにDataFrameを生成し、テキスト化
    for req_string, df_data_list in latest_df_data.items():
        if df_data_list:
            try:
                df = pd.DataFrame(df_data_list)
                if not df.empty:
                    full_data_string += f"■「{req_string}」に関するデータ:\n{df.to_string(index=False)}\n\n"
                else:
                    full_data_string += f"■「{req_string}」に関するデータ:\n(この要件に対するデータは空でした)\n\n"
            except Exception as e:
                logging.error(f"interpret_node: Error converting data for '{req_string}' to DataFrame: {e}")
                full_data_string += f"■「{req_string}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n"
        else:
            full_data_string += f"■「{req_string}」に関するデータ:\n(この要件に対するデータはありませんでした)\n\n"

    # 一応OrderedDictとしてデータ構造はあるが、中身が全て空リストや形式エラーで“実質的に有効なデータが1件も無い”」場合にキャッチ
    # 例{"2022年売上": [], "2023年売上": []}
    processed_parts = [part for part in full_data_string.split("■")[1:] if part] # Get non-empty parts after splitting by "■"
    all_parts_indicate_no_data = all(
        "(この要件に対するデータはありませんでした)" in part or \
        "(データ形式エラーのため表示できません)" in part or \
        "(この要件に対するデータは空でした)" in part \
        for part in processed_parts
    )

    if not full_data_string.strip() or (processed_parts and all_parts_indicate_no_data):
        logging.info("interpret_node: データ内容が全て空またはエラーのため、解釈対象がありません。")
        return {**state, "interpretation": "解釈対象の有効なデータがありませんでした。", "condition": "interpretation_failed_empty_data"}

    system_prompt = "あなたは優秀なデータ分析の専門家です。"
    user_prompt = f"""
    以下のデータセット群について、データから読み取れる特徴や傾向を簡潔な日本語で解説してください。
    特に「{plan_details_context if plan_details_context else "全般的な傾向"}」という観点に注目して分析し、必要であればデータ間の関連性や組み合わせから洞察できる示唆も述べてください。

    {full_data_string}
    """
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        interpretation_text = response.content.strip() if response.content else ""

        if not interpretation_text:
            logging.warning("interpret_node: LLM returned empty interpretation.")
            final_condition = "interpretation_failed"
            interpretation_text = "解釈の生成に失敗しました。"
        else:
            final_condition = "interpretation_done"
    except Exception as e:
        logging.error(f"interpret_node: Exception during LLM call: {e}")
        interpretation_text = "解釈中にエラーが発生しました。"
        final_condition = "interpretation_failed"
    return {**state, "interpretation": interpretation_text, "condition": final_condition}


def chart_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")
    # Charting instructions from plan step details, now in state.input
    chart_instructions = state.get("input", "Generate a suitable chart for the available data.")

    if not latest_df_data or not isinstance(latest_df_data, collections.OrderedDict) or not any(v for v in latest_df_data.values() if v): # ensure at least one df has data
        return {**state, "chart_result": None, "condition": "chart_generation_failed_no_data"}

    df_to_plot = None
    selected_key_for_chart = None

    # Attempt to select DataFrame based on chart_instructions referencing a key
    if isinstance(chart_instructions, str):
        for key in latest_df_data.keys():
            # Ensure data for the key is not empty before considering it a match
            if latest_df_data.get(key) and isinstance(latest_df_data[key], list) and len(latest_df_data[key]) > 0 and key.lower() in chart_instructions.lower():
                selected_key_for_chart = key
                break

    if selected_key_for_chart and latest_df_data.get(selected_key_for_chart):
        try:
            df_candidate = pd.DataFrame(latest_df_data[selected_key_for_chart])
            if not df_candidate.empty:
                df_to_plot = df_candidate
                chart_context_message = f"以下の「{selected_key_for_chart}」に関するデータと、ユーザーからの「{chart_instructions}」という指示に基づいて"
            else:
                logging.warning(f"chart_node: DataFrame for selected key '{selected_key_for_chart}' is empty.")
                selected_key_for_chart = None # Nullify to trigger fallback
        except Exception as e:
            logging.error(f"chart_node: Error creating DataFrame for selected key '{selected_key_for_chart}': {e}")
            selected_key_for_chart = None # Nullify to trigger fallback

    if df_to_plot is None: # Fallback: use the first non-empty DataFrame
        for key, data_list in latest_df_data.items():
            if data_list and isinstance(data_list, list) and len(data_list) > 0:
                try:
                    df_candidate = pd.DataFrame(data_list)
                    if not df_candidate.empty:
                        df_to_plot = df_candidate
                        selected_key_for_chart = key
                        chart_context_message = f"以下の「{selected_key_for_chart}」に関するデータ（複数あるデータセットの最初で、有効なデータを含むもの）と、ユーザーからの「{chart_instructions}」という指示に基づいて"
                        if len(latest_df_data) > 1 and (not isinstance(chart_instructions, str) or selected_key_for_chart.lower() not in chart_instructions.lower()): # Check if chart_instructions is not str OR selected_key not in chart_instructions
                            logging.warning(f"chart_node: Multiple DataFrames available. Defaulting to first non-empty one ('{selected_key_for_chart}') due to generic or non-matching chart instructions: '{chart_instructions}'.")
                        break
                except Exception as e:
                    logging.error(f"chart_node: Error creating DataFrame from latest_df key '{key}' during fallback: {e}")
                    continue

    if df_to_plot is None or df_to_plot.empty:
        return {**state, "chart_result": None, "condition": "chart_generation_failed_empty_selected_data", "error": "チャート作成に適したデータが見つかりませんでした。"}

    python_tool = PythonAstREPLTool(
        locals={"df": df_to_plot, "sns": sns, "pd": pd, "japanize_matplotlib": japanize_matplotlib},
        description=(
            "Pythonコードを実行してデータを分析・グラフ化できます。dfとしてプロット対象のDataFrameが定義済みです。"
            "グラフは 'output.png' として保存してください。日本語対応のため japanize_matplotlib.japanize() を実行し、sns.set(font='IPAexGothic') も試してください。"
        )
    )
    tools = [python_tool]
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True,
        agent_kwargs={"handle_parsing_errors": True} # Added for more robust error handling if LLM output is bad for agent
    )

    chart_prompt = f"""
    あなたはPythonプログラミングとデータ可視化の専門家です。
    {chart_context_message}最適なグラフを生成し、'output.png'というファイル名で保存してください。
    日本語フォントを利用するために、コードの先頭で `import japanize_matplotlib; japanize_matplotlib.japanize()` を実行してください。
    Seabornを使用する場合は `sns.set(font='IPAexGothic')` も試してください。
    dfの列情報: {df_to_plot.info()}
    dfの最初の5行:
    {df_to_plot.head().to_string(index=False)}
    指示: {chart_instructions}
    """
    try:
        if os.path.exists("output.png"):
            os.remove("output.png")

        agent_response = agent.invoke(chart_prompt)
        logging.info(f"chart_node: Agent response: {agent_response}")

        if os.path.exists("output.png"):
            fig = base64.b64encode(open("output.png", "rb").read()).decode('utf-8')
            return {**state, "chart_result": fig, "condition": "chart_generation_done"}
        else:
            logging.error("chart_node: Agent executed but output.png was not found. Agent output might indicate why.")
            error_message = "グラフファイルの生成に失敗しました。"
            if isinstance(agent_response, dict) and "output" in agent_response:
                error_message += f" (Agent: {agent_response['output']})"
            return {**state, "chart_result": None, "condition": "chart_generation_failed_no_output_file", "error": error_message}
    except Exception as e:
        logging.error(f"chart_node: Error during agent execution: {e}", exc_info=True)
        return {**state, "chart_result": None, "condition": "chart_generation_failed_agent_error", "error": str(e)}

def classify_next(state: MyState):
    user_action = state.get("user_action")
    intents = state.get("intent_list", [])

    if state.get("user_clarification"):
        logging.info("classify_next: User clarification detected. Routing to dispatch_plan_step.")
        return "dispatch_plan_step"

    if user_action == "proceed_analysis_step":
        logging.info("classify_next: 'proceed_analysis_step' action. Routing to dispatch_plan_step.")
        return "dispatch_plan_step"
    elif user_action == "cancel_analysis_plan":
        logging.info("classify_next: 'cancel_analysis_plan' action. Routing to cancel_analysis_plan node.")
        return "cancel_analysis_plan"

    if "clear_data_intent" in intents:
        return "clear_data"
    elif "メタデータ検索" in intents:
        return "metadata_retrieval"

    if state.get("analysis_plan") and state.get("current_plan_step_index") is not None:
        logging.info("classify_next: Existing plan detected. Routing to dispatch_plan_step to continue.")
        return "dispatch_plan_step"
    else:
        logging.info("classify_next: No existing plan or overriding action. Routing to create_analysis_plan.")
        return "create_analysis_plan"

def clarify_next(state: MyState):
    condition = state.get("condition")
    if condition == "awaiting_user_clarification":
        return END
    elif condition == "clarify_error_invalid_plan" or condition == "clarify_error_unexpected_action":
        return END
    else:
        logging.warning(f"clarify_next: Unexpected condition '{condition}' from clarify_node. Defaulting to END.")
        return END

def create_analysis_plan_next(state: MyState):
    condition = state.get("condition")
    if condition == "plan_generated":
        logging.info("create_analysis_plan_next: Plan generated. Routing to dispatch_plan_step.")
        return "dispatch_plan_step"
    elif condition == "empty_plan_generated":
        logging.info("create_analysis_plan_next: Empty plan generated. Routing to END")
        return END
    elif condition == "plan_generation_failed":
        logging.error("create_analysis_plan_next: Plan generation failed. Routing to END.")
        return END
    else:
        logging.warning(f"create_analysis_plan_next: Unknown condition '{condition}'. Routing to END.")
        return END

def check_history_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
    state["awaiting_step_confirmation"] = False
    return "dispatch_plan_step"

def sql_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("sql_next: SQL node executed outside of a planned context.")
    intents = state.get("intent_list", [])
    if "グラフ作成" in intents: return "chart"
    if "データ解釈" in intents: return "interpret"
    return END

def chart_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("chart_next: Chart node executed outside of a planned context.")
    if "データ解釈" in state.get("intent_list", []): return "interpret"

def interpret_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("interpret_next: Interpret node executed outside of a planned context.")

# 分析計画（analysis_plan）」の現在ステップに応じて、次にどのノード（処理）に進むかを決定
def execute_plan_router(state: MyState) -> str:
    # Clarify（追加質問）へのユーザー回答があれば、分析プランを再生成する
    if state.get("user_clarification"):
        logging.info("execute_plan_router: ユーザーの追加情報が入力されたため、分析プランを再作成します。")
        return "create_analysis_plan"
    
    analysis_plan = state.get("analysis_plan")
    current_plan_step_index = state.get("current_plan_step_index")

    # プランが存在しない・インデックスが不正・プラン完了時は初期化して提案ノードに遷移
    if not analysis_plan or current_plan_step_index is None or not (0 <= current_plan_step_index < len(analysis_plan)):
        logging.info("execute_plan_router: プランが完了、または不正な状態です。")
        state["analysis_plan"] = None
        state["current_plan_step_index"] = None
        state["awaiting_step_confirmation"] = False
        return END
        
    current_step = analysis_plan[current_plan_step_index]
    action = current_step.get("action")
    logging.info(f"execute_plan_router: Dispatching to action '{action}' for step {current_plan_step_index}.")
    details = current_step.get("details", "")
    state["input"] = details
    
    if action == "sql":
        # SQL取得用の要件をセットし、各状態を初期化
        state["missing_data_requirements"] = [details] if isinstance(details, str) else details if isinstance(details, list) else []
        if not isinstance(state.get("latest_df"), collections.OrderedDict):
            state["latest_df"] = collections.OrderedDict()
        state["SQL"] = None
        state["error"] = None
        return "sql"
    elif action == "check_history":
        return "check_history"
    elif action == "clarify":
        return "clarify"
    elif action == "chart":
        return "chart"
    elif action == "interpret":
        return "interpret"
    else:
        logging.error(f"プランに未対応のアクション '{action}' が含まれています。")
        state["error"] = f"プランに未対応のアクションが指定されています: {action}"
        return END

def build_workflow():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MyState)

    # Add all nodes
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("create_analysis_plan", create_analysis_plan_node)
    workflow.add_node("cancel_analysis_plan", cancel_analysis_plan_node)
    workflow.add_node("dispatch_plan_step", lambda state: state) # New dummy node for central routing logic
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("check_history", check_history_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_node("metadata_retrieval", metadata_retrieval_node)
    workflow.add_node("clear_data", clear_data_node)
    
    # Define edges
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", classify_next)
    workflow.add_edge("clear_data", END)
    workflow.add_edge("metadata_retrieval", END)
    workflow.add_edge("cancel_analysis_plan", END)
    workflow.add_conditional_edges("create_analysis_plan", create_analysis_plan_next)

    workflow.add_conditional_edges(
        "dispatch_plan_step",
        execute_plan_router,
        {
            "create_analysis_plan": "create_analysis_plan",
            "check_history": "check_history",
            "clarify": "clarify",
            "sql": "sql",
            "chart": "chart",
            "interpret": "interpret",
            "END": END
        }
    )

    workflow.add_conditional_edges(
        "sql",
        sql_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END  # Explicitly map END if it's a possible return
        }
    )
    workflow.add_conditional_edges(
        "chart",
        chart_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END # Explicitly map END if it's a possible return (though not in current chart_next)
        }
    )
    workflow.add_conditional_edges(
        "interpret",
        interpret_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END # Explicitly map END if it's a possible return (though not in current interpret_next)
        }
    )
    workflow.add_conditional_edges(
        "check_history",
        check_history_next,
        {
            "dispatch_plan_step": "dispatch_plan_step"
            # If check_history_next could return other values, map them here.
            # For now, it only returns "dispatch_plan_step".
        }
    )
    workflow.add_conditional_edges(
        "clarify",
        clarify_next,
        {
            END: END
            # clarify_next always returns END or conditions that should lead to END.
        }
    )


    return workflow.compile(checkpointer=memory)

user_query = "カテゴリの合計販売金額を出して"
workflow = build_workflow()
config = {"configurable": {"thread_id": "2"}}
res = workflow.invoke({"input": user_query}, config=config)
# print(res)
# sample_history_entry = {
#     "id": "hist_001",
#     "query": "A商品の売上集計",
#     "timestamp": datetime.now().isoformat(),
#     "dataframe_dict": [{"product": "A", "sales": 100}, {"product": "A", "sales": 150}],
#     "SQL": "SELECT product, sales FROM sales_table WHERE product = 'A'"
# }
# current_state = workflow.get_state(config)
# current_df_history = current_state.values.get('df_history', [])
# current_df_history.append(sample_history_entry)
# workflow.update_state(config, {"df_history": current_df_history})
