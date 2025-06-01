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
    return {**state, "interpretation": response.content.strip(), "condition": "解釈完了"}


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

# classifyノードの分岐（次に何へ進むか？）
def classify_next(state):
    intents = state.get("intent_list", [])
    if "clear_data_intent" in intents: # Prioritize clear intent
        return "clear_data"
    elif "メタデータ検索" in intents: # Then metadata search
        return "metadata_retrieval"
    elif "データ取得" in intents:
        return "extract_data_requirements" # Changed from "find_similar_query"
    # These checks are likely redundant if metadata or data acquisition is primary
    elif "グラフ作成" in intents: # This case might be hit if "データ取得" is not in intents
        # If only chart is requested, and no data acquisition, where does the data come from?
        # This path might need to go to find_similar_query to see if *any* relevant data exists,
        # or it implies the user expects to use previously loaded data.
        # For now, if "データ取得" is not an intent, chart/interpret might rely on existing latest_df.
        return "chart"
    elif "データ解釈" in intents:
        return "interpret"
    else:
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
def sql_next(state):
    intents = state.get("intent_list", [])
    if "グラフ作成" in intents:
        return "chart"
    elif "データ解釈" in intents:
        return "interpret"
    else:
        return END

# chartノード後の遷移
def chart_next(state):
    intents = state.get("intent_list", [])
    if "データ解釈" in intents:
        return "interpret"
    else:
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


def build_workflow():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MyState)
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("extract_data_requirements", extract_data_requirements_node) # New node
    workflow.add_node("find_similar_query", find_similar_query_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_node("metadata_retrieval", metadata_retrieval_node) # 新しいノードを追加
    workflow.add_node("clear_data", clear_data_node) # Add clear_data node
    
    workflow.add_conditional_edges("classify", classify_next)
    workflow.add_edge("extract_data_requirements", "find_similar_query") # Edge from new node
    workflow.add_conditional_edges("find_similar_query", find_similar_next)
    workflow.add_conditional_edges("sql", sql_next)
    workflow.add_conditional_edges("chart", chart_next)
    workflow.add_edge("interpret", END)
    workflow.add_edge("metadata_retrieval", END) # メタデータ検索後は終了
    workflow.add_edge("clear_data", END) # Add edge for clear_data node

    workflow.set_entry_point("classify")
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
