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
    intent_list: List[str]           # 分類結果（データ取得/グラフ作成/データ解釈）
    latest_df: Optional[collections.OrderedDict[str, list]] # Changed: Dict mapping requirement to its dataframe (list of records)
    df_history: Optional[List[dict]] # SQL実行結果のDataFrameの履歴 {"id": str, "query": str, "timestamp": str, "dataframe_dict": list, "SQL": Optional[str]}
    SQL: Optional[str]               # 生成されたSQL
    interpretation: Optional[str]    # データ解釈（分析コメント）
    chart_result: Optional[str]      # グラフ画像（base64など）
    metadata_answer: Optional[str]   # メタデータ検索結果の回答
    condition: Optional[str]         # 各ノードの実行状態
    error: Optional[str]             # SQL等でエラーがあれば
    query_history: Optional[List[str]] # ユーザーの問い合わせ履歴
    data_requirements: List[str]     # データ要件
    missing_data_requirements: Optional[List[str]] # New: List of data requirements not found in history
    clarification_question: Optional[str] = None
    analysis_options: Optional[List[str]] = None
    user_clarification: Optional[str] = None
    # For multi-stage analysis planning
    analysis_plan: Optional[List[dict]] = None # Stores the list of analysis steps, e.g., [{"action": "sql", "details": "Overall monthly sales"}, {"action": "interpret", "details": "Interpret monthly sales data"}]
    current_plan_step_index: Optional[int] = None # Index of the current step in analysis_plan
    awaiting_step_confirmation: Optional[bool] = None # True if waiting for user to proceed to next step
    complex_analysis_original_query: Optional[str] = None # Stores the original multi-step query
    user_action: Optional[str] = None # New field for frontend actions like "proceed_analysis_step" or "cancel_analysis_plan"

# Analysis Planning Node
def analysis_planning_node(state: MyState) -> MyState:
    user_query = state.get("input", "")

    # LLM Prompt to identify multi-step queries and generate a plan
    # For this subtask, the actual LLM call is mocked below.
    # prompt_for_plan = f"""
    # Analyze the user's query: "{user_query}"
    # Is this a multi-step analysis request?
    # If YES, decompose it into a series of logical steps. Each step should map to an action: "sql", "interpret", "chart".
    # For each step, specify the action and "details" (e.g., for "sql", what data to fetch based on the query portion for that step).
    # Output the plan as a JSON string: '[{{"action": "action_type", "details": "description_for_step"}}, ...]'
    # Example: User query "Show monthly sales, interpret it, then chart the sales trend."
    # Output: '[{{"action": "sql", "details": "monthly sales"}}, {{"action": "interpret", "details": "interpret monthly sales"}}, {{"action": "chart", "details": "chart sales trend"}}]'
    # If NO (it's a single-step request or not an analysis task), output the exact phrase: "NOT_MULTI_STEP"
    # """
    # llm_response_str = llm.invoke(prompt_for_plan).content.strip()

    # Mocked response for subtask:
    if "plan" in user_query.lower() or "まず" in user_query or "次に" in user_query or "then" in user_query.lower(): # Simple keyword check
        # Example: "まず月次売上を表示し、次にそれを解釈し、最後に売上トレンドをグラフ化してplan"
        # This mock plan assumes the LLM can break this down.
        # A real LLM would need to extract "月次売上" for the first sql, etc.
        plan_parts = []
        if "月次売上" in user_query or "monthly sales" in user_query:
            plan_parts.append({"action": "sql", "details": "Overall monthly sales"})
            if "解釈" in user_query or "interpret" in user_query:
                plan_parts.append({"action": "interpret", "details": "Interpret monthly sales"})
            if "グラフ" in user_query or "chart" in user_query or "トレンド" in user_query or "trend" in user_query:
                 plan_parts.append({"action": "chart", "details": "Sales trend chart"})

        if not plan_parts: # Fallback mock plan if keywords aren't specific enough for the above
            plan_parts = [{"action": "sql", "details": "General data related to query"}, {"action": "interpret", "details": "Interpret general data"}]

        plan_json_str = json.dumps(plan_parts) # Convert list of dicts to JSON string
    else:
        plan_json_str = "NOT_MULTI_STEP"

    if plan_json_str == "NOT_MULTI_STEP":
        return {
            **state,
            "analysis_plan": None,
            "condition": "single_step_request"
        }
    else:
        try:
            parsed_plan = json.loads(plan_json_str)
            if not isinstance(parsed_plan, list) or not all(isinstance(step, dict) and "action" in step and "details" in step for step in parsed_plan):
                raise ValueError("Plan is not a list of dicts with action and details")

            if not parsed_plan: # Check if the list is empty after parsing
                # This handles cases where the LLM might return an empty list '[]' for the plan
                logging.warning("LLM generated an empty plan. Treating as single_step_request.")
                return {
                    **state,
                    "analysis_plan": None, # Ensure plan is None
                    "input": state.get("complex_analysis_original_query", user_query), # Restore original input
                    "condition": "single_step_request"
                }

            # Set input for the first step.
            # Subsequent steps will have their input set by execute_planned_step_node or plan_step_transition_node
            first_step_input = parsed_plan[0].get("details", "") # Now safe to access parsed_plan[0]
            first_step_action = parsed_plan[0].get("action")

            return {
                **state,
                "complex_analysis_original_query": user_query, # Store original multi-step query
                "analysis_plan": parsed_plan,
                "current_plan_step_index": 0,
                "awaiting_step_confirmation": False, # Initially false, set to true after a step completes
                "input": first_step_input, # Modify input to be for the first step
                "data_requirements": [first_step_input] if first_step_action == "sql" and first_step_input else [], # Pre-populate data_requirements for first SQL step
                "condition": "plan_generated"
            }
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse analysis plan: {e}. Plan string: '{plan_json_str}'")
            return {
                **state,
                "analysis_plan": None,
                "condition": "single_step_request", # Treat as single step if plan parsing fails
                "error": f"Error parsing analysis plan: {e}"
            }

# Execute Planned Step Node
def execute_planned_step_node(state: MyState) -> MyState:
    plan = state.get("analysis_plan")
    current_index = state.get("current_plan_step_index")

    if not plan or current_index is None or not (0 <= current_index < len(plan)):
        logging.error("Invalid plan or step index in execute_planned_step_node.")
        return {**state, "error": "Invalid plan state.", "condition": "plan_error"}

    current_step = plan[current_index]
    action = current_step.get("action")
    details = current_step.get("details", "") # Details for the current step

    # Prepare state for the target node
    # Set 'input' to current step's details.
    # If action is 'sql', also populate 'data_requirements'.
    # Clear other potentially interfering state fields from previous steps if necessary.

    # Reset specific fields that should be populated by the upcoming node
    # Preserve latest_df if current action is interpret or chart, otherwise clear it (e.g., for SQL action)
    new_latest_df = state.get("latest_df")
    if action == "sql": # If the current planned step is SQL, then latest_df should be cleared to receive new data.
        new_latest_df = collections.OrderedDict()
    # For 'interpret' or 'chart', new_latest_df will retain the existing state.latest_df,
    # which should contain the output of the previous SQL step.

    current_state_for_step = {
        **state,
        "input": details, # Target node will use this as its primary input for the step
        "data_requirements": [details] if action == "sql" and details else [],
        "SQL": None,
        "interpretation": None,
        "chart_result": None,
        "error": None,
        "latest_df": new_latest_df, # Use the adjusted new_latest_df
        "missing_data_requirements": [],
        "clarification_question": None,
        "user_clarification": None,
        "user_action": None, # Clear user_action as it has been processed by routing to this node
        "awaiting_step_confirmation": False, # Ensure this is false as we are executing a step
        "condition": f"executing_plan_step_{action}"
    }

    return current_state_for_step

# Plan Step Transition Node
def plan_step_transition_node(state: MyState) -> MyState:
    current_index = state.get("current_plan_step_index")
    plan_length = len(state.get("analysis_plan", []))

    if current_index is None:
        logging.error("current_plan_step_index is None in plan_step_transition_node")
        return {**state, "error": "Plan execution error: step index missing.", "condition": "plan_error_no_index"}

    next_index = current_index + 1

    if next_index < plan_length:
        # Prepare for the next step
        next_step_details = state["analysis_plan"][next_index].get("details", "")
        next_step_action = state["analysis_plan"][next_index].get("action")

        return {
            **state,
            "current_plan_step_index": next_index,
            "awaiting_step_confirmation": True, # Pause for frontend confirmation
            "input": next_step_details, # Set input for the upcoming step (will be used by execute_planned_step_node)
            "data_requirements": [next_step_details] if next_step_action == "sql" and next_step_details else [],
            "condition": "awaiting_next_step_confirmation"
        }
    else:
        # Plan complete
        original_query = state.get("complex_analysis_original_query", "")
        return {
            **state,
            "analysis_plan": None,
            "current_plan_step_index": None,
            "awaiting_step_confirmation": None,
            "complex_analysis_original_query": None,
            "input": original_query, # Restore original query or set to a summary
            "data_requirements": [], # Clear for post-plan phase
            "condition": "plan_completed"
        }

# Node to handle cancellation of an analysis plan
def cancel_analysis_plan_node(state: MyState) -> MyState:
    original_query = state.get("complex_analysis_original_query", "The analysis plan was cancelled.")
    return {
        **state,
        "analysis_plan": None,
        "current_plan_step_index": None,
        "awaiting_step_confirmation": False, # Ensure this is false
        "complex_analysis_original_query": None,
        "input": original_query, # Restore original query or use a cancellation message
        "interpretation": "The multi-step analysis plan has been cancelled by the user.",
        "condition": "plan_cancelled",
        "user_action": None # Clear the action
    }

#意図の判別

def extract_sql(sql_text):
    """AIが生成したSQLから```やコードブロックを除去"""
    match = re.search(r"```sql\s*(.*?)```", sql_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", sql_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return sql_text.strip()

def parse_llm_list_output(llm_output_str: str) -> List[str]:
    try:
        # Handle common LLM list-like outputs, e.g. newlines, hyphens
        # Remove potential markdown list characters like '-' or '*'
        cleaned_str = re.sub(r'^\s*[-*]\s*', '', llm_output_str, flags=re.MULTILINE)

        # If it's not already in list format, try to make it so,
        # especially if it's a simple newline-separated list of suggestions
        if not cleaned_str.strip().startswith('['):
            lines = [line.strip().replace("'", "\\'") for line in cleaned_str.split('\n') if line.strip()]
            # Ensure each item is quoted if it's a simple list of strings
            # This helps ast.literal_eval to parse it correctly.
            if all(not (line.startswith("'") and line.endswith("'")) and \
                   not (line.startswith('"') and line.endswith('"')) for line in lines):
                 cleaned_str = "[" + ", ".join([f"'{line}'" for line in lines]) + "]"
            else: # It might already be a list of quoted strings, just needs brackets
                 cleaned_str = "[" + ", ".join(lines) + "]"


        parsed_list = ast.literal_eval(cleaned_str)
        if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
            return parsed_list
        return [] # Return empty if not a list of strings
    except (ValueError, SyntaxError) as e:
        logging.warning(f"Failed to parse LLM list output with ast.literal_eval: {e}. Original string: '{llm_output_str}'. Falling back to newline split.")
        # Fallback: Try to extract lines if it looks like a multi-line string output
        # that ast.literal_eval couldn't handle (e.g. no quotes, just items on new lines)
        # Remove any surrounding brackets or list-like artifacts before splitting
        str_for_split = llm_output_str.strip()
        if str_for_split.startswith('[') and str_for_split.endswith(']'):
            str_for_split = str_for_split[1:-1]

        # Split by newline, strip quotes/commas from each item, filter empty
        suggestions = []
        for line in str_for_split.split('\n'):
            # Remove list item markers like '-' or '*' and surrounding whitespace/quotes
            item = re.sub(r'^\s*[-*]\s*', '', line).strip()
            item = item.strip('\'",') # Remove surrounding quotes
            if item:
                 suggestions.append(item)
        return suggestions


def try_sql_execute(sql_text):
    """SQLを実行し、エラーがあれば内容を返す"""
    try:
        conn = sqlite3.connect("my_data.db")
        df = pd.read_sql(sql_text, conn)
        return df, None
    except Exception as e:
        logging.error(f"SQL execution failed. Error: {e}\nSQL: {sql_text}", exc_info=True)
        return None, str(e)
    
def transform_sql_error(sql_error: str) -> str:
    """Transforms raw SQL error messages into user-friendly messages."""
    if "no such table" in sql_error.lower():
        return f"指定されたテーブルが見つからなかったため、クエリを実行できませんでした。テーブル名を確認してください。(詳細: {sql_error})"
    elif "no such column" in sql_error.lower():
        return f"指定された列が見つからなかったため、クエリを実行できませんでした。列名を確認してください。(詳細: {sql_error})"
    elif "syntax error" in sql_error.lower():
        return f"SQL構文にエラーがあります。クエリを確認してください。(詳細: {sql_error})"
    # Add more specific common SQL error translations if possible
    else:
        return f"SQLクエリの処理中に予期せぬエラーが発生しました。管理者に連絡してください。(詳細: {sql_error})"

def fix_sql_with_llm(original_sql, error_message, rag_tables, rag_queries, user_query):
    """エラー内容からSQL修正をAIに依頼"""
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

def find_similar_query_node(state: MyState) -> MyState:
    data_requirements = state.get("data_requirements", [])
    df_history = state.get("df_history", [])

    found_data_map = {}
    missing_requirements = []

    # To avoid reusing the same history entry for multiple requirements,
    # we can keep track of used history entry IDs.
    used_history_ids = set()

    if not data_requirements:
        # No data requirements specified, so nothing to find.
        # This might happen if "データ取得" was not an intent but the node is still reached.
        return {
            **state,
            "latest_df": collections.OrderedDict(), # Consistent initialization
            "missing_data_requirements": [],
            "condition": "no_requirements_specified" # New condition
        }

    found_data_map = collections.OrderedDict() # Initialize as OrderedDict

    for req in data_requirements:
        found_for_req = False
        best_similarity_for_req = 0.0
        best_match_entry = None

        for i, entry in enumerate(df_history):
            if entry.get("id") in used_history_ids:
                continue # Skip already used history entries

            # Compare the current data requirement (req) with the query that generated the history entry.
            # This is an approximation. Ideally, history entries would be tagged with the requirements they satisfy.
            similarity = difflib.SequenceMatcher(None, req, entry.get("query", "")).ratio()
            # print(f"Comparing requirement '{req}' with history query '{entry.get('query', '')}': Similarity: {similarity:.2f}") # DEBUG

            if similarity >= SIMILARITY_THRESHOLD and similarity > best_similarity_for_req:
                best_similarity_for_req = similarity
                best_match_entry = entry

        if best_match_entry:
            # print(f"Found similar data for requirement '{req}' in history (ID: {best_match_entry.get('id', 'N/A')}, Similarity: {best_similarity_for_req:.2f}).") # DEBUG
            found_data_map[req] = best_match_entry["dataframe_dict"]
            used_history_ids.add(best_match_entry.get("id"))
            found_for_req = True

        if not found_for_req:
            # print(f"No similar data found in history for requirement: '{req}'") # DEBUG
            missing_requirements.append(req)

    final_condition = ""
    if not missing_requirements:
        final_condition = "all_data_found"
        # print("All data requirements found in history.") # DEBUG
    else:
        final_condition = "missing_data"
        # print(f"Missing data requirements: {missing_requirements}") # DEBUG

    return {
        **state,
        "latest_df": found_data_map,
        "missing_data_requirements": missing_requirements,
        "condition": final_condition,
        "SQL": None,  # Clear SQL from previous full query if any
        "interpretation": None, # Clear previous interpretation
        "chart_result": None # Clear previous chart
    }

# SQL生成＋実行
def sql_node(state: MyState) -> MyState:
    missing_requirements = state.get("missing_data_requirements", [])
    overall_user_input = state.get("input", "")
    current_latest_df = state.get("latest_df", collections.OrderedDict())
    if not isinstance(current_latest_df, collections.OrderedDict): # Ensure it's an OrderedDict
        current_latest_df = collections.OrderedDict(current_latest_df or {})

    current_df_history = state.get("df_history", [])
    
    last_sql_generated = None
    any_error_occurred = False
    accumulated_errors = []

    system_prompt_sql_generation = """
    あなたはSQL生成AIです。この後の質問に対し、SQLiteの標準的なSQL文のみを出力してください。
    - 使用できるSQL構文は「SELECT」「WHERE」「GROUP BY」「ORDER BY」「LIMIT」のみです。
    - 日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    - 必ず1つのテーブルだけを使い、簡単な集計・フィルタ・並べ替えまでにしてください。
    - SQLの前後にコメントや説明文は出力しないでください。出力された内容をそのまま実行するため、SQL文のみをそのまま出力してください。
    """

    if missing_requirements:
        # print(f"Processing missing data requirements: {missing_requirements}") # DEBUG
        for req_string in missing_requirements:
            # print(f"Attempting to fetch data for requirement: '{req_string}'") # DEBUG

            # 1. RAG specific to the requirement
            retrieved_tables_docs = vectorstore_tables.similarity_search(req_string, k=3)
            rag_tables = "\n".join([doc.page_content for doc in retrieved_tables_docs])
            retrieved_queries_docs = vectorstore_queries.similarity_search(req_string, k=3)
            rag_queries = "\n".join([doc.page_content for doc in retrieved_queries_docs])

            # 2. Prompt for LLM, focused on the requirement
            # The overall_user_input is included for broader context if useful.
            user_prompt_for_req = f"""
            【ユーザーの全体的な質問】
            {overall_user_input}

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
            # print(f"SQL generated for '{req_string}': {sql_generated_clean}") # DEBUG
            original_sql_for_logging = sql_generated_clean # Store for logging if fix fails
            last_sql_generated = sql_generated_clean
            result_df, sql_error = try_sql_execute(sql_generated_clean)

            if sql_error:
                logging.warning(
                    f"Initial SQL execution failed for requirement. "
                    f"User requirement: '{req_string}', "
                    f"Generated SQL: '{sql_generated_clean}', "
                    f"Error: {sql_error}. Attempting to fix..."
                )
                # Pass req_string as user_query context to fix_sql_with_llm
                fixed_sql = fix_sql_with_llm(sql_generated_clean, sql_error, rag_tables, rag_queries, req_string)
                # print(f"Fixed SQL for '{req_string}': {fixed_sql}") # DEBUG
                last_sql_generated = fixed_sql
                result_df, sql_error = try_sql_execute(fixed_sql)

                if sql_error: # Error persisted after fix attempt
                    logging.error(
                        f"SQL execution failed definitively for requirement. "
                        f"User requirement: '{req_string}', "
                        f"Original SQL: '{original_sql_for_logging}', "
                        f"Fixed SQL: '{fixed_sql}', "
                        f"Error: {sql_error}"
                    )
                    any_error_occurred = True
                    user_friendly_error = transform_sql_error(sql_error)
                    accumulated_errors.append(f"For '{req_string}': {user_friendly_error}")
            elif result_df is not None:
                # print(f"Successfully fetched data for requirement: '{req_string}'") # DEBUG
                result_df_dict = result_df.to_dict(orient="records")
                current_latest_df[req_string] = result_df_dict

                new_history_entry = {
                    "id": uuid.uuid4().hex[:8],
                    "query": req_string, # Query is the specific requirement string
                    "timestamp": datetime.now().isoformat(),
                    "dataframe_dict": result_df_dict,
                    "SQL": last_sql_generated
                }
                current_df_history.append(new_history_entry)
            else: # No error, but no data (e.g. SELECT query that returns 0 rows)
                # print(f"No data returned for requirement '{req_string}', but no SQL error.") # DEBUG
                # Storing empty list to signify the query ran but had no results for this req.
                current_latest_df[req_string] = []


        final_condition = "SQL部分的失敗" if any_error_occurred else "SQL実行完了"
        final_error_message = "; ".join(accumulated_errors) if accumulated_errors else None

        return {
            **state,
            "latest_df": current_latest_df,
            "df_history": current_df_history,
            "SQL": last_sql_generated, # Store the last SQL executed in this batch
            "condition": final_condition,
            "error": final_error_message,
            # missing_data_requirements might be cleared or updated based on success/failure.
            # For now, let's assume it's processed, and subsequent nodes decide if they need it.
            # Or, we can remove successfully fetched reqs from missing_data_requirements.
            # Let's clear it, assuming this node tried to fetch all of them.
            "missing_data_requirements": [req for req in missing_requirements if req not in current_latest_df], # Update missing list
        }

    else:
        # Fallback to existing behavior: process state["input"] as a single query
        # This part largely mirrors the original single-query logic but adapts state update.
        retrieved_tables_docs = vectorstore_tables.similarity_search(overall_user_input, k=3)
        rag_tables = "\n".join([doc.page_content for doc in retrieved_tables_docs])
        retrieved_queries_docs = vectorstore_queries.similarity_search(overall_user_input, k=3)
        rag_queries = "\n".join([doc.page_content for doc in retrieved_queries_docs])

        user_prompt_for_main_input = f"""
        【テーブル定義に関する情報】
        {rag_tables}
        【類似する問い合わせ例とそのSQL】
        {rag_queries}
        【ユーザー質問】
        {overall_user_input}
        """

        response = llm.invoke([
            {"role": "system", "content": system_prompt_sql_generation},
            {"role": "user", "content": user_prompt_for_main_input}
        ])

        sql_generated_clean = extract_sql(response.content.strip())
        # print(f"SQL generated for full input '{overall_user_input}': {sql_generated_clean}") # DEBUG
        original_sql_for_logging = sql_generated_clean # Store for logging if fix fails
        result_df, sql_error = try_sql_execute(sql_generated_clean)
        current_error_message = None # Initialize

        if sql_error:
            logging.warning(
                f"Initial SQL execution failed for full input. "
                f"User input: '{overall_user_input}', "
                f"Generated SQL: '{sql_generated_clean}', "
                f"Error: {sql_error}. Attempting to fix..."
            )
            fixed_sql = fix_sql_with_llm(sql_generated_clean, sql_error, rag_tables, rag_queries, overall_user_input)
            # print(f"Fixed SQL for full input: {fixed_sql}") # DEBUG
            # sql_generated_clean should store the fixed_sql if a fix was attempted, for logging purposes.
            # However, if the fixed_sql also fails, sql_generated_clean (which becomes original_sql_for_logging in the next error log)
            # should be the original one, and fixed_sql is the one that failed.
            # Let's adjust: use a different variable for the SQL that is currently being executed.
            current_executed_sql = fixed_sql
            result_df, sql_error = try_sql_execute(current_executed_sql)

            if sql_error: # Error persisted after fix attempt
                logging.error(
                    f"SQL execution failed definitively for full input. "
                    f"User input: '{overall_user_input}', "
                    f"Original SQL: '{original_sql_for_logging}', "
                    f"Fixed SQL: '{current_executed_sql}', "
                    f"Error: {sql_error}"
                )
                current_error_message = transform_sql_error(sql_error)
                # sql_generated_clean for the state should reflect the last SQL that was tried and failed.
                sql_generated_clean = current_executed_sql
            else: # Fix was successful
                current_error_message = None # Clear error if fix was successful
                sql_generated_clean = current_executed_sql # Update to the successfully fixed SQL
        else: # No initial error
             current_error_message = None
             # sql_generated_clean is already the initially generated SQL, which was successful.

        if result_df is not None:
            result_df_dict = result_df.to_dict(orient="records")
            # Store it under a generic key like the input query itself, or a predefined key
            # This maintains the Dict[str, list] structure for latest_df
            current_latest_df[overall_user_input] = result_df_dict

            new_history_entry = {
                "id": uuid.uuid4().hex[:8],
                "query": overall_user_input, # Query is the full input
                "timestamp": datetime.now().isoformat(),
                "dataframe_dict": result_df_dict,
                "SQL": sql_generated_clean
            }
            current_df_history.append(new_history_entry)

            return {
                **state,
                "latest_df": current_latest_df,
                "df_history": current_df_history,
                "SQL": sql_generated_clean,
                "condition": "SQL実行完了",
                "error": None
            }
        else:
            return {
                **state,
                "latest_df": current_latest_df, # Potentially empty if it was before
                "SQL": sql_generated_clean,
                "condition": "SQL実行失敗",
                "error": current_error_message # This will be the user-friendly message or None
            }

# 解釈
def interpret_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")

    if not latest_df_data: # Handles None or empty OrderedDict
        return {**state, "interpretation": "まだデータがありません。先にSQL質問をするか、メタデータ検索を試してください。", "condition": "解釈失敗"}

    full_data_string = ""
    if isinstance(latest_df_data, collections.OrderedDict):
        if not latest_df_data: # Empty OrderedDict
             return {**state, "interpretation": "データが取得されましたが、内容は空です。", "condition": "解釈失敗"}
        for req_string, df_data_list in latest_df_data.items():
            if df_data_list: # If list is not empty
                df = pd.DataFrame(df_data_list)
                full_data_string += f"■データ要件名: 「{req_string}」の実行結果:\n{df.to_string(index=False)}\n\n"
            else:
                full_data_string += f"■データ要件名: 「{req_string}」の実行結果:\n(この要件に対するデータはありません)\n\n"
    elif isinstance(latest_df_data, list): # Fallback for old format (should ideally not happen)
        if not latest_df_data:
            return {**state, "interpretation": "データが取得されましたが、内容は空です。", "condition": "解釈失敗"}
        df = pd.DataFrame(latest_df_data)
        full_data_string = df.to_string(index=False)
    else: # Should not happen with proper state management
        return {**state, "interpretation": "データの形式が不正です。", "condition": "解釈失敗"}

    if not full_data_string.strip(): # If all datasets were empty or not processable
        return {**state, "interpretation": "データが取得されましたが、内容は空でした。", "condition": "解釈失敗"}

    system_prompt = "あなたは優秀なデータ分析の専門家です。"
    user_prompt = f"""
    以下のSQLクエリ実行結果群（複数の場合あり）について、それぞれのデータから読み取れる特徴や傾向、
    またはデータ間の関連性や組み合わせから洞察できる示唆があれば、簡潔な日本語で解説してください。

    {full_data_string}
    """
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    interpretation_text = ""
    if response.content is not None:
        interpretation_text = response.content.strip()
    return {**state, "interpretation": interpretation_text, "condition": "解釈完了"}


def chart_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")
    user_input_for_chart = state.get("input", "") # The original user query

    if not latest_df_data: # Handles None or empty OrderedDict
        return {**state, "chart_result": None, "condition": "グラフ化失敗"}

    df_to_plot = None
    chart_context_message = ""

    if isinstance(latest_df_data, collections.OrderedDict):
        if not latest_df_data: # Empty OrderedDict
            return {**state, "chart_result": None, "condition": "グラフ化失敗"}

        # Select the first DataFrame for charting as per requirement
        first_req_string = next(iter(latest_df_data.keys()))
        first_df_data_list = next(iter(latest_df_data.values()))

        if first_df_data_list:
            df_to_plot = pd.DataFrame(first_df_data_list)
            chart_context_message = f"以下のデータ（「{first_req_string}」に関するもの）とユーザーからの依頼に答える"
            # print(f"Chart node: Using data for requirement '{first_req_string}' for charting.") # DEBUG
        else:
            # print(f"Chart node: Data for the first requirement '{first_req_string}' is empty.") # DEBUG
            return {**state, "chart_result": None, "condition": "グラフ化失敗"}

    elif isinstance(latest_df_data, list): # Fallback for old format
        if not latest_df_data:
             return {**state, "chart_result": None, "condition": "グラフ化失敗"}
        df_to_plot = pd.DataFrame(latest_df_data)
        chart_context_message = "以下のdf（pandas DataFrame）とユーザーからの依頼に答える"

    if df_to_plot is None or df_to_plot.empty:
        # print("Chart node: No valid DataFrame to plot.") # DEBUG
        return {**state, "chart_result": None, "condition": "グラフ化失敗"}

    python_tool = PythonAstREPLTool(
        locals={"df": df_to_plot, "sns": sns},
        description=(
            "Pythonコードを実行してデータを分析できます。"
            "あらかじめ dfが定義されています。"
            "import seaborn as snsを使ってグラフ生成も可能です。"
            "グラフ化するときは、日本語に対応するためにsns.set(font='IPAexGothic')をしてください。"
            "作成したグラフはoutput.pngという名前で保存してください。"
        )
    )
    tools = [python_tool]
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )

    chart_prompt = f"""
    あなたはPythonプログラミングが得意なデータ分析の専門家です。
    {chart_context_message}
    最適なグラフをsns(seaborn)で作成して、output.pngという名前で保存してください。

    ユーザーからの全体的な依頼:
    {user_input_for_chart}

    dfの最初の10行:
    {df_to_plot.head(10).to_string(index=False)}
    """
    agent.invoke(chart_prompt)

    if os.path.exists("output.png"):
        fig = base64.b64encode(open("output.png", "rb").read()).decode('utf-8')
        return {**state, "chart_result": fig, "condition": "グラフ化完了"}
    else:
        return {**state, "chart_result": None, "condition": "グラフ化失敗"} # ensure chart_result is None

def classify_intent_node(state):
    user_input = state["input"]
    if user_input == "SYSTEM_CLEAR_HISTORY":
        # Preserve existing query_history if any, or initialize
        current_history = state.get("query_history", [])
        return {
            **state,
            "intent_list": ["clear_data_intent"],
            "condition": "分類完了", # Match standard condition output
            "query_history": current_history # Preserve history up to this point if needed by clear_data_node
        }

    # Initialize or append to query_history (moved down)
    current_history = state.get("query_history", [])
    # Avoid duplicating system messages or if history already has this exact user input as last item
    if not current_history or current_history[-1] != user_input:
         current_history.append(user_input)

    # ここをAzure OpenAIで複数分類に拡張
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

# データ要件抽出ノード
def extract_data_requirements_node(state):
    user_input = state["input"]
    prompt = f"""
    ユーザーの質問から、必要となる具体的なデータの要件を抽出してください。
    各データ要件は簡潔に記述し、複数ある場合はカンマ区切りで列挙してください。

    例:
    質問: 「A商品の売上集計とお客様属性のクロス集計グラフを出して」
    出力: A商品の売上集計,顧客属性データ

    質問: 「製品Xの月次販売数と、それに対応する主要な競合製品Yの販売数を比較したい。」
    出力: 製品Xの月次販売数,競合製品Yの月次販売数

    質問: 「東京23区内の平均家賃と、各区の人口密度、平均所得を地図上に可視化して。」
    出力: 東京23区内の平均家賃,東京23区の人口密度,東京23区の平均所得

    質問: 「{user_input}」
    出力:
    """
    response = llm.invoke(prompt)
    extracted_requirements_str = response.content.strip()
    # LLMの出力が空文字列の場合、空のリストを返す
    if not extracted_requirements_str:
        extracted_requirements = []
    else:
        extracted_requirements = [req.strip() for req in extracted_requirements_str.split(",") if req.strip()]

    # print(f"Extracted data requirements: {extracted_requirements}") # DEBUG

    return {
        **state,
        "data_requirements": extracted_requirements,
        "condition": "データ要件抽出完了"
    }

# Information Gathering Node
def information_gathering_node(state: MyState) -> MyState:
    user_input = state.get("input", "")
    data_requirements = state.get("data_requirements", [])
    df_history = state.get("df_history", [])
    user_clarification = state.get("user_clarification")

    # 1. If user_clarification is present, incorporate it
    if user_clarification:
        # Simple incorporation: append to original input.
        # More sophisticated logic might be needed to update data_requirements directly.
        updated_input = f"{user_input} [User Clarification: {user_clarification}]"
        # Potentially re-run data requirements extraction or directly use it.
        # For now, we assume this clarification is enough to proceed.
        return {
            **state,
            "input": updated_input, # Augment input
            "user_clarification": None, # Clear after processing
            "clarification_question": None, # Clear previous question
            "condition": "clarification_not_needed"
        }

    # 2. If no user_clarification, proceed to check if clarification is needed
    # Create a summary of df_history (e.g., list of queries)
    df_history_summary_list = []
    if df_history:
        for entry in df_history:
            query = entry.get("query", "N/A")
            # Truncate long queries if necessary
            summary = query if len(query) < 50 else query[:47] + "..."
            df_history_summary_list.append(summary)
    df_history_summary = ", ".join(df_history_summary_list) if df_history_summary_list else "No relevant data history."

    prompt_for_clarification = f"""
You are an intelligent assistant helping to clarify a user's data analysis request.
Your goal is to identify if essential information for data retrieval or analysis is missing.
If crucial information is missing, formulate a single, clear question to the user to obtain it.
If no clarification is needed, respond with the exact phrase: NO_CLARIFICATION_NEEDED

Here is the current context:
User's overall request: "{user_input}"
Identified data requirements by a previous step: {data_requirements if data_requirements else "None yet."}
Summary of previously retrieved data relevant to this conversation: {df_history_summary}

Examples of missing information and good clarification questions:
- If the user asks for "sales data" without a period: "For what period would you like to see the sales data (e.g., last month, specific year, specific dates)?"
- If the user asks to "analyze performance" without specifying metrics or subjects: "What specific aspects or metrics of performance are you interested in analyzing, and for which subjects (e.g., products, employees)?"
- If data requirements mention "customer segments" but no segmentation criteria are clear: "How would you like to segment the customers? (e.g., by demographics, purchase history, location)"
- If the user asks for a comparison but one of the items to compare is unclear: "You mentioned comparing X with Y, but Y is unclear. What is Y?"

Based on the user's request and identified data requirements, is there any missing information?
If yes, what question should be asked to the user?
If no, reply with NO_CLARIFICATION_NEEDED.
"""
    # response = llm.invoke(prompt_for_clarification) # This would be the actual LLM call
    # llm_response_text = response.content.strip()

    # For now, retain the mock logic but with the new prompt structure in mind for future integration.
    # The mock logic below should be replaced by a real LLM call using the prompt above.
    llm_response_text = "" # Placeholder for actual LLM response
    if not data_requirements and ("show me data" in user_input.lower() or "データを見せて" in user_input.lower()):
        llm_response_text = "To proceed, I need a bit more information: What specific data would you like to see and for what purpose?"
    elif "analyze" in user_input.lower() and not any("specific" in req.lower() for req in data_requirements):
        llm_response_text = "To proceed, I need a bit more information: What specific aspects are you interested in analyzing?"
    elif ("sales data" in user_input.lower() or "売上データ" in user_input.lower()) and not any(period_keyword in req.lower() for period_keyword in ["month", "year", "period", "date", "月", "年", "期間", "日付"] for req in data_requirements):
        # Check if user_clarification already provided this
        if not (user_clarification and any(pk in user_clarification.lower() for pk in ["month", "year", "period", "date"])):
             llm_response_text = "For what period would you like to see the sales data (e.g., last month, specific dates)?"
        else:
             llm_response_text = "NO_CLARIFICATION_NEEDED" # Assume clarification fixed it
    else:
        llm_response_text = "NO_CLARIFICATION_NEEDED"

    if llm_response_text != "NO_CLARIFICATION_NEEDED":
        return {
            **state,
            "clarification_question": llm_response_text,
            "condition": "awaiting_user_clarification"
        }
    else:
        return {
            **state,
            "clarification_question": None,
            "condition": "clarification_not_needed"
        }

# classifyノードの分岐（次に何へ進むか？）
def classify_next(state: MyState):
    user_action = state.get("user_action")
    intents = state.get("intent_list", []) # Populated by classify_intent_node

    # Priority 1: User actions from frontend buttons
    if user_action == "proceed_analysis_step":
        # Frontend payload for "proceed_analysis_step" includes "awaiting_step_confirmation": False.
        # user_action will be cleared in execute_planned_step_node.
        return "execute_planned_step"
    elif user_action == "cancel_analysis_plan":
        # user_action will be cleared in cancel_analysis_plan_node.
        return "cancel_analysis_plan"

    # Priority 2: System intents (like clear history)
    if "clear_data_intent" in intents:
        return "clear_data"

    # Priority 3: Metadata search intent
    elif "メタデータ検索" in intents: # "metadata_search"
        return "metadata_retrieval"

    # Priority 4: Default to analysis planning for any other user query/intent
    else:
        # This path is for new user queries or existing queries where no specific user_action was triggered.
        # classify_intent_node would have run, populating intents. If intents are e.g. ["データ取得"],
        # it still goes to analysis_planning_node which will then decide if it's single_step_request
        # or if a plan needs to be generated.
        return "analysis_planning"

# Conditional logic for information_gathering_node
def information_gathering_next(state: MyState):
    condition = state.get("condition")
    # This node is part of the single-step execution path.
    # It should not be aware of multi-step plans.
    if condition == "awaiting_user_clarification":
        return END # Go to END, frontend will handle clarification
    elif condition == "clarification_not_needed":
        return "find_similar_query" # Original single-step flow continues
    else: # Should not happen
        logging.warning(f"Unexpected condition in information_gathering_next: {condition}")
        return END

# Conditional logic for analysis_planning_node
def analysis_planning_next(state: MyState):
    condition = state.get("condition")
    if condition == "plan_generated":
        # If a plan is generated, and user was confirming, this confirmation is for the *first* step.
        # If awaiting_step_confirmation is true here, it means user clicked "Proceed" on the overall plan.
        if state.get("awaiting_step_confirmation") is True:
             return "execute_planned_step" # Directly execute first step
        else:
            # This is the first time plan is generated, ask for confirmation to start.
            # Or, could proceed directly if no initial overall confirmation is desired.
            # For now, let's assume we always ask for confirmation for the first step too.
            # The frontend would show the plan and a "Start Plan" button.
            # To simplify for backend, we might assume first step confirmation is implicit or handled by frontend sending a specific input.
            # Let's proceed to execute_planned_step directly for now.
            # Frontend can control this by sending a specific "user_confirms_plan_start" input if needed.
            return "execute_planned_step"
    elif condition == "single_step_request":
        return "extract_data_requirements" # Fallback to original single-step flow
    else: # plan_error or other
        return END

# Conditional logic for execute_planned_step_node
def execute_planned_step_next(state: MyState):
    # This node prepares the state for the actual action node (sql, interpret, chart)
    # The condition was set to "executing_plan_step_{action}"
    condition = state.get("condition", "")
    if "executing_plan_step_sql" in condition:
        return "sql"
    elif "executing_plan_step_interpret" in condition:
        return "interpret"
    elif "executing_plan_step_chart" in condition:
        return "chart"
    else: # plan_error etc.
        logging.error(f"Unknown condition after execute_planned_step: {condition}")
        return END

# Conditional logic for plan_step_transition_node
def plan_step_transition_next(state: MyState):
    condition = state.get("condition")
    if condition == "awaiting_next_step_confirmation":
        # State now has awaiting_step_confirmation = True.
        # Frontend should get this, display results, and offer "Proceed".
        # When user clicks "Proceed", workflow is invoked again.
        # classify_next will see awaiting_step_confirmation = True and route to execute_planned_step.
        return END
    elif condition == "plan_completed":
        return "suggest_analysis_paths" # Or END, depending on desired final action
    elif condition == "plan_error_no_index": # Error case from plan_step_transition
        return END
    else: # Should not happen
        return END

def find_similar_next(state: MyState):
    condition = state.get("condition")
    intents = state.get("intent_list", [])

    if condition == "all_data_found":
        # print("Transitioning from find_similar_query: All data found.") # DEBUG
        # All data requirements are met from history. Proceed to next steps based on intent.
        if "グラフ作成" in intents:
            return "chart"
        elif "データ解釈" in intents:
            return "interpret"
        else:
            return END # All data found, but no further processing requested.
    elif condition == "missing_data":
        # print("Transitioning from find_similar_query: Missing data. Proceeding to SQL node.") # DEBUG
        # Some data is missing, so go to sql_node to fetch it.
        # sql_node will (eventually) need to know what's in "missing_data_requirements".
        return "sql"
    elif condition == "no_requirements_specified":
        # This case implies "データ取得" was not an intent, or requirements extraction failed.
        # If there are other intents like chart/interpret, they might operate on whatever is in latest_df.
        # Or, it could be an end state if no other actions are specified.
        # print("Transitioning from find_similar_query: No data requirements were specified.") # DEBUG
        if "グラフ作成" in intents:
            return "chart"
        elif "データ解釈" in intents:
            return "interpret"
        else:
            return END # No specific data needed, and no further actions.
    else: # Default fallback, should ideally not be reached if conditions are comprehensive
        # print(f"Transitioning from find_similar_query: Unknown condition ('{condition}'). Defaulting to SQL node.") # DEBUG
        return "sql"


# sqlノード後の遷移
def sql_next(state: MyState):
    # If part of a plan, after SQL execution, go to plan transition.
    if state.get("analysis_plan") and state.get("current_plan_step_index") is not None:
        return "plan_step_transition"

    # Original single-step logic
    intents = state.get("intent_list", []) # intent_list might be empty/irrelevant in planned mode
    if not intents and state.get("complex_analysis_original_query"): # Try to infer from original query if in single_step_request mode after planning failed
        # This part is tricky, ideally single_step_request would re-run intent classification or have its own logic
        pass # For now, rely on explicit intents or direct plan execution

    if "グラフ作成" in intents:
        return "chart"
    elif "データ解釈" in intents:
        return "interpret"
    else:
        return END # Or suggest_analysis_paths if appropriate for single SQL query

# chartノード後の遷移
def chart_next(state: MyState):
    # If part of a plan, after chart execution, go to plan transition.
    if state.get("analysis_plan") and state.get("current_plan_step_index") is not None:
        return "plan_step_transition"

    # Original single-step logic
    intents = state.get("intent_list", [])
    if "データ解釈" in intents:
        return "interpret"
    else:
        return "suggest_analysis_paths"

# interpretノード後の遷移 (New, as interpret_node was a terminal before routing to suggest_analysis_paths)
def interpret_next(state: MyState):
    # If part of a plan, after interpretation, go to plan transition.
    if state.get("analysis_plan") and state.get("current_plan_step_index") is not None:
        return "plan_step_transition"

    # Original single-step logic: after interpretation, suggest next paths
    return "suggest_analysis_paths"


# Conditional logic for suggest_analysis_paths_node
def suggest_analysis_paths_next(state: MyState):
    # Currently, always ends after suggesting paths, frontend handles selection.
    return END

# 新しいメタデータ検索ノード
def metadata_retrieval_node(state):
    user_query = state["input"]
    
    # 1. テーブル定義DBから検索 (vectorstore_tablesを使用)
    retrieved_docs = vectorstore_tables.similarity_search(user_query, k=3)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. プロンプト組み立て
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
    
    # 3. LLMに問い合わせ
    response = llm.invoke(llm_prompt)
    answer = response.content.strip()
    
    return {**state, "metadata_answer": answer, "condition": "メタデータ検索完了"}

def clear_data_node(state):
    # Clear DataFrame, SQL, interpretation, chart, errors, and query history
    # Keep 'input' and 'intent_list' as they are from the current query that triggered the clear.
    # Also keep metadata_answer as per instructions.
    return {
        "input": state.get("input"), # Preserve current input
        "intent_list": state.get("intent_list"), # Preserve current intent list
        "latest_df": collections.OrderedDict(), # Consistent initialization
        "df_history": [],  # New field, cleared
        "SQL": None,
        "interpretation": DATA_CLEARED_MESSAGE, # Confirmation message
        "chart_result": None,
        "metadata_answer": state.get("metadata_answer"), # Keep this
        "condition": "データクリア完了",
        "error": None,
        "query_history": [] # Reset query history
    }

# Suggest Analysis Paths Node
# Ensure DATA_CLEARED_MESSAGE is defined globally or passed appropriately if used in this node.
# For the subtask, we'll assume it's accessible.

def suggest_analysis_paths_node(state: MyState) -> MyState:
    original_user_input = state.get("input", "")
    latest_df_ordered_dict = state.get("latest_df")
    current_interpretation = state.get("interpretation", "")
    query_history = state.get("query_history", [])
    # DATA_CLEARED_MESSAGE should be accessible here if used in conditions

    no_data_messages = [
        "まだデータがありません。",
        "データが取得されましたが、内容は空です。",
        "データが取得されましたが、内容は空でした。",
        DATA_CLEARED_MESSAGE # Make sure this constant is available
    ]

    # Initial check for meaningful data or interpretation
    if not latest_df_ordered_dict and (not current_interpretation or any(msg in current_interpretation for msg in no_data_messages)):
        return {**state, "analysis_options": [], "condition": "no_paths_suggested"}

    data_summary_parts = []
    if latest_df_ordered_dict:
        for req, data_list in latest_df_ordered_dict.items():
            if data_list:
                try:
                    df = pd.DataFrame(data_list)
                    if not df.empty:
                        data_summary_parts.append(f"Data for '{req}': {df.shape[0]} rows, {df.shape[1]} columns (Columns: {', '.join(df.columns)})")
                    else:
                        data_summary_parts.append(f"Data for '{req}': Empty")
                except Exception:
                    data_summary_parts.append(f"Data for '{req}': Could not load as DataFrame")
            else:
                data_summary_parts.append(f"Data for '{req}': Empty or not available")

    data_summary = "\n".join(data_summary_parts) if data_summary_parts else "No data retrieved or all data was empty."
    recent_query_history_summary = "\n".join(query_history[-3:]) if query_history else "No query history."

    prompt_for_suggestions = f"""
You are an intelligent data analysis assistant. Based on the user's recent activity and the data obtained, suggest 2-3 relevant follow-up questions or analysis actions the user might be interested in.
Format your response as a Python-style list of strings. For example: ["Show sales by region", "Analyze customer churn"]
If no specific suggestions come to mind or if the data is insufficient for further meaningful analysis, return an empty list: [].

User's original request: "{original_user_input}"
Summary of current data:
{data_summary}
Current interpretation/summary of findings: "{current_interpretation if current_interpretation else 'Not yet available.'}"
Recent query history:
{recent_query_history_summary}

Based on this, what are some logical next steps or questions?
Response (as a Python list of strings):
"""
    # In a real scenario, an LLM call would be made here:
    # response = llm.invoke(prompt_for_suggestions)
    # llm_response_str = response.content.strip()

    # Mock LLM response for now, using logic similar to the original but ensuring it's integrated with the new prompt structure
    llm_response_str = "[]" # Default to no suggestions
    if data_summary_parts:
        if "sales" in original_user_input.lower() or "売上" in original_user_input:
            llm_response_str = '["月別の売上トレンドを表示しますか？", "最も売上が高い商品カテゴリは何ですか？", "前期と比較した売上はどうですか？"]'
        elif "customer" in original_user_input.lower() or "顧客" in original_user_input:
            llm_response_str = '["顧客をデモグラフィック情報でセグメント分けしますか？", "新規顧客獲得数のトレンドを表示しますか？", "顧客の購入頻度を分析しますか？"]'
        else:
            llm_response_str = '["このデータを異なる種類のグラフで表示しますか？", "主要な発見を要約してください。", "業界ベンチマークと比較してどうですか？"]'
    elif current_interpretation and len(current_interpretation) > 50 and not any(msg in current_interpretation for msg in no_data_messages):
        llm_response_str = '["この分析の限界は何ですか？", "これをさらに詳細に分析できますか？"]'

    parsed_options = parse_llm_list_output(llm_response_str)

    if not parsed_options:
        return {**state, "analysis_options": [], "condition": "no_paths_suggested"}
    else:
        return {**state, "analysis_options": parsed_options[:3], "condition": "paths_suggested"}

def build_workflow():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MyState)

    # Add all nodes
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("analysis_planning", analysis_planning_node) # New
    workflow.add_node("execute_planned_step", execute_planned_step_node) # New
    workflow.add_node("plan_step_transition", plan_step_transition_node) # New
    workflow.add_node("cancel_analysis_plan", cancel_analysis_plan_node) # New

    workflow.add_node("extract_data_requirements", extract_data_requirements_node)
    workflow.add_node("information_gathering", information_gathering_node)
    workflow.add_node("find_similar_query", find_similar_query_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_node("suggest_analysis_paths", suggest_analysis_paths_node)
    workflow.add_node("metadata_retrieval", metadata_retrieval_node)
    workflow.add_node("clear_data", clear_data_node)
    
    # Define edges
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", classify_next)

    workflow.add_edge("clear_data", END)
    workflow.add_edge("metadata_retrieval", END)
    workflow.add_edge("cancel_analysis_plan", "suggest_analysis_paths") # Or END if preferred after cancellation

    # Analysis Planning Path
    workflow.add_conditional_edges("analysis_planning", analysis_planning_next)
    workflow.add_conditional_edges("execute_planned_step", execute_planned_step_next)

    # Edges from action nodes (sql, interpret, chart) now go to their respective "_next" router
    # which will decide if they go to plan_step_transition or old path.
    workflow.add_conditional_edges("sql", sql_next)
    workflow.add_conditional_edges("chart", chart_next)
    # workflow.add_edge("interpret", "suggest_analysis_paths") # Old edge
    workflow.add_conditional_edges("interpret", interpret_next) # New conditional edge for interpret

    workflow.add_conditional_edges("plan_step_transition", plan_step_transition_next)

    # Original Single-Step Path (after analysis_planning decides it's single_step_request)
    workflow.add_edge("extract_data_requirements", "information_gathering")
    workflow.add_conditional_edges("information_gathering", information_gathering_next)
    workflow.add_conditional_edges("find_similar_query", find_similar_next)

    # Common end path after single or planned analysis (if not ending sooner)
    workflow.add_conditional_edges("suggest_analysis_paths", suggest_analysis_paths_next)
    return workflow.compile(checkpointer=memory)

user_query = "カテゴリの合計販売金額を出して" # This will not trigger metadata_retrieval
# To test metadata_retrieval, use a query like:
# user_query = "sales_dataテーブルにはどんなカラムがありますか？" 
workflow = build_workflow()
config = {"configurable": {"thread_id": "2"}} # Changed thread_id for potentially fresh state
# Example queries for testing:
# 1. Simple data fetch (likely misses history at first)
# user_query = "A商品の売上データをください"
# 2. Query that might have partial history match later
# user_query = "A商品の売上データと顧客属性データをください" (assume A商品売上は履歴にあるが顧客属性はない)
# 3. Query for charting that might use found data
# user_query = "A商品の売上データをグラフにして"

res = workflow.invoke({"input": user_query}, config=config) # Use the original user_query for now
# print(res) # DEBUG - Main workflow output

# Example of how to populate df_history for testing
# You'd need to run this separately or ensure your DB has these from previous runs.
# sample_history_entry = {
#     "id": "hist_001",
#     "query": "A商品の売上集計", # This query ideally matches a data requirement
#     "timestamp": datetime.now().isoformat(),
#     "dataframe_dict": [{"product": "A", "sales": 100}, {"product": "A", "sales": 150}],
#     "SQL": "SELECT product, sales FROM sales_table WHERE product = 'A'"
# }
# Accessing and updating state for a thread (example, not for direct execution here):
# current_state = workflow.get_state(config)
# current_df_history = current_state.values.get('df_history', [])
# current_df_history.append(sample_history_entry)
# workflow.update_state(config, {"df_history": current_df_history})
