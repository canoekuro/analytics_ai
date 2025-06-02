# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

from langchain_community.vectorstores import FAISS
import difflib # このインポートがファイルの先頭に追加されていることを確認してください
import sqlite3
import pandas as pd
import logging
from langgraph.graph import StateGraph, END
from langchain.agents import Tool, initialize_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import os
import base64
import io
import re
from typing import TypedDict, List, Optional, Any
import ast # literal_evalのため
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
import japanize_matplotlib
import seaborn as sns
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import json # import jsonを先頭に移動しました
import google.generativeai as genai # Gemini API のインポート
from google.api_core import exceptions as google_exceptions # Google APIエラー処理用

# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

# 環境変数からLLMモデル名を取得します（デフォルト値あり）
llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro") # デフォルトをgemini-1.5-proに変更
google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) # または他の適切なGemini埋め込みモデル
SIMILARITY_THRESHOLD = 0.8

# ベクトルストア
vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(
    model=llm_model_name,
    temperature=0,
    google_api_key=google_api_key
) # または、より高性能な "gemini-1.5-flash"、"gemini-1.5-pro"

DATA_CLEARED_MESSAGE = "データが正常にクリアされました。"


import collections
class MyState(TypedDict, total=False):
    input: str                       # ユーザーの問い合わせ
    condition: Optional[str]         # 各ノードの実行状態
    intent_list: List[str]           # 分類結果（データ取得/グラフ作成/データ解釈）
    latest_df: Optional[collections.OrderedDict[str, list]] # 変更：要件をそのデータフレーム（レコードのリスト）にマッピングする辞書
    df_history: Optional[List[dict]] # SQL実行結果のDataFrameの履歴 {"id": str, "query": str, "timestamp": str, "dataframe_dict": list, "SQL": Optional[str]}
    SQL: Optional[str]               # 生成されたSQL
    interpretation: Optional[str]    # データ解釈（分析コメント）
    chart_result: Optional[str]      # グラフ画像（base64など）
    metadata_answer: Optional[str]   # メタデータ検索結果の回答
    error: Optional[str]             # SQL等でエラーがあれば
    query_history: Optional[List[str]] # ユーザーの問い合わせ履歴
    data_requirements: List[str]     # データ要件
    missing_data_requirements: Optional[List[str]] # 新規：履歴に見つからなかったデータ要件のリスト
    clarification_question: Optional[str] = None
    analysis_options: Optional[List[str]] = None
    user_clarification: Optional[str] = None
    analysis_plan: Optional[List[dict]] = None # 分析ステップのリストを格納します。例：[{"action": "sql", "details": "月次総売上"},{"action": "interpret", "details": "月次売上データを解釈"}]
    current_plan_step_index: Optional[int] = None # analysis_plan内の現在のステップのインデックス
    awaiting_step_confirmation: Optional[bool] = None # ユーザーが次のステップに進むのを待機している場合はTrue
    original_query: Optional[str] = None # clarifyが発生したときに元のクエリを格納します
    user_action: Optional[str] = None # "proceed_analysis_step"や"cancel_analysis_plan"のようなフロントエンドアクション用の新しいフィールド

#ユーザーの入力に応じて意図を分類
def classify_intent_node(state: MyState):
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
    次の5つのうち、該当するものを「,」区切りで全て列挙してください：
    - データ取得
    - グラフ作成
    - データ解釈
    - メタデータ検索
    - データ加工

    質問: 「{user_input}」
    例: 
    input:「カテゴリの合計販売金額を出して」 output:「データ取得」
    input:「＊＊＊のデータを出して」 output:「データ取得」
    input:「＊＊＊のデータを取得してグラフ化して」 output:「データ取得,グラフ作成」
    input:「＊＊＊のデータを取得して解釈して」 output:「データ取得,データ解釈」
    input:「＊＊＊のデータのグラフ化と解釈して」 output:「データ取得,グラフ作成,データ解釈」
    input:「sales_dataテーブルにはどんなカラムがありますか？」 output:「メタデータ検索」
    input:「categoryカラムの情報を教えてください」 output:「メタデータ検索」
    input:「AとBを結合して」 output:「データ加工」

    他の情報は不要なので、outputの部分（「データ取得,グラフ作成,データ解釈,メタデータ検索,データ加工」）だけを必ず返すようにしてください。
    """
    result = llm.invoke(prompt).content.strip()
    steps = [x.strip() for x in result.split(",") if x.strip()]

    return {**state, "intent_list": steps, "condition": "分類完了", "query_history": current_history}

def metadata_retrieval_node(state: MyState):
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

def clear_data_node(state: MyState):
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
    
# 新しい分析計画ノード (create_analysis_plan_node)
def create_analysis_plan_node(state: MyState) -> MyState:
    user_query = state.get("input", "")
    user_clarification = state.get("user_clarification")
    intent_list = state.get("intent_list", [])
    parsed_plan = None # 初期化

    try:
        # ツールのスキーマ定義
        plan_generation_tool = genai.types.Tool(
        function_declarations=[
            genai.types.FunctionDeclaration(
                name="create_plan",
                description="ユーザーの要求に基づいて分析計画を作成します。",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "plan_steps": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            items=genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                properties={
                                    "action": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                        description="実行するアクション（clarify, check_history, sql, chart, interpret, data_processing）"
                                    ),
                                    "details": genai.types.Schema(
                                        type=genai.types.Type.ANY, # string or array of strings
                                        description="アクションの詳細。clarifyとsqlは文字列、check_historyは文字列の配列。"
                                    )
                                },
                                required=["action", "details"]
                            )
                        )
                    },
                    required=["plan_steps"]
                )
            )
        ]
        )

        SYSTEM_PROMPT = """あなたはAIデータアナリストです。ユーザーの要求を分析し、ステップバイステップの分析計画を作成してください。
        # 指示
        - KPI・ブランド・期間など、分析に必要な情報が不足していれば `clarify` を必ずプランの先頭に置いてください。
        - 具体的な情報が揃っている場合は `clarify` は不要です。
        - 利用可能なアクションは`clarify`, `check_history`, `sql`, `chart`, `interpret` です。各アクションの具体的な内容は以下のとおりです。
            - clarify: ユーザーに詳細を質問します。 (例: "期間の指定を求める")
              - "details": (string) ユーザーへの質問文。
            - check_history: 過去に同様のデータが取得されているか確認します。 (例: "'月次売上データ'を確認")
              - "details": (array of strings) 確認するデータ要件のリスト。
            - sql: データベースからデータを取得します。 (例: "'特定の製品の売上データを取得'")
              - "details": (string) SQLで取得する具体的なデータの内容。
            - chart: 取得したデータからグラフを作成します。 (例: "'売上データを棒グラフで表示'")
              - "details": (string) グラフ作成の指示 (例: "売上データを棒グラフで表示", "DF_KEY['売上データ'] を使用して時系列グラフを作成")。
            - interpret: データやグラフを解釈し、洞察を提供します。 (例: "'売上傾向を分析する'")
              - "details": (string) 解釈の焦点や内容。
            - data_processing: 既存のデータを加工・変換します。 (例: "'売上データに利益率列を追加する'")
              - "details": (string) データ加工の具体的な指示。処理対象のDataFrameを指定するには `DF_KEY['your_dataframe_key_in_latest_df']` を含めてください。
            
        以下は計画作成の例です:
        
        例1: 曖昧なリクエスト
        ユーザーのクエリ: "売上を見せて"
        生成されるプラン:
        ```json
        [
          {"action": "clarify", "details": "Could you please specify the time period for the sales data (e.g., last month, last quarter)?"}
        ]
        ```
        
        例2: 複数ステップのリクエスト
        ユーザーのクエリ: "先月の製品Xの総売上を取得し、棒グラフで可視化して、最後にそのグラフを解釈してください。"
        生成されるプラン:
        ```json
        [
          {"action": "check_history", "details": ["先月の製品Xの総売上"]},
          {"action": "sql", "details": "先月の製品Xの総売上"},
          {"action": "chart", "details": "先月の製品Xの総売上を棒グラフで表示"},
          {"action": "interpret", "details": "先月の製品Xの総売上の棒グラフを解釈"}
        ]
        ```
        
        例3: データ加工を含むリクエスト
        ユーザーのクエリ: "顧客データセットを読み込み、'age'列を基に新しい列'age_group'を追加し（例: 18歳未満: Teen, 18-65歳: Adult, 65歳超: Senior）、処理後のデータの最初の5行を見せてください。"
        生成されるプラン:
        ```json
        [
          {"action": "check_history", "details": ["顧客データセット"]},
          {"action": "sql", "details": "顧客データセット"},
          {"action": "data_processing", "details": "DF_KEY['顧客データセット']の'age'列を基に、新しい列'age_group'を追加する（例：18歳未満: Teen, 18-65歳: Adult, 65歳超: Senior）"},
          {"action": "interpret", "details": "新しい'age_group'列を含む、処理後の顧客データセット（DF_KEY['顧客データセット']）の最初の5行を説明する"}
        ]
        ```
        
        必ずJSON形式で plan_steps を含む応答を生成してください。
        """
        prompt_parts = [SYSTEM_PROMPT]

        # Add query history to prompt
        query_history = state.get("query_history", [])
        MAX_HISTORY_LEN = 5 # 最大履歴数
        if query_history:
            history_start_index = max(0, len(query_history) - MAX_HISTORY_LEN -1) # 現在のクエリは除くため-1
            relevant_history = query_history[history_start_index:-1] if len(query_history) > 1 else []
            if relevant_history:
                formatted_history = "\n\n過去の関連する問い合わせ履歴 (新しいものが最後):\n"
                for hist_item in relevant_history:
                    formatted_history += f"- {hist_item}\n"
                prompt_parts.append(formatted_history)

        prompt_parts.append("\nユーザーの現在のクエリ: ")
        prompt_parts.append(user_query)

        if user_clarification:
            prompt_parts.append(f"\nユーザーからの追加情報: {user_clarification}")
            original_query = state.get("original_query", "")
            if original_query and original_query != user_query :
                prompt_parts.append(f"\n(この追加情報は、以前のクエリ「{original_query}」に関連しています)")

        final_prompt_string = "".join(prompt_parts)
        logging.info(f"Geminiへの最終プロンプト:\n{final_prompt_string}")

        # APIコールとレスポンス処理
        try:
            response = llm.generate_content(
                final_prompt_string,
                tools=[plan_generation_tool],
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            )
            logging.info(f"Geminiからのレスポンス: {response}")

            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Geminiからのレスポンスに有効な候補またはパートがありません。")

            function_call_part = next((part for part in response.candidates[0].content.parts if part.function_call), None)

            if not function_call_part:
                raw_text_response = response.text if hasattr(response, 'text') else "利用可能なテキストレスポンスがありません。"
                logging.warning(f"Geminiレスポンスにfunction_callが含まれていません。テキストレスポンスを試みます: {raw_text_response}")
                try:
                    parsed_plan_from_text = json.loads(raw_text_response)
                    if not (isinstance(parsed_plan_from_text, list) and \
                       all(isinstance(step, dict) and "action" in step and "details" in step for step in parsed_plan_from_text)):
                        raise ValueError("テキストレスポンスは有効なプラン形式ではありません。")
                    logging.info("Geminiがテキスト形式で有効なプランを返しました。これを採用します。")
                    plan_steps = parsed_plan_from_text
                except (json.JSONDecodeError, ValueError) as e_text_parse:
                    logging.error(f"Geminiのテキストレスポンスのパースに失敗: {e_text_parse}")
                    raise ValueError(f"モデル応答は期待されるfunction_call形式ではなく、テキスト応答の解析にも失敗しました。詳細: {e_text_parse}") from e_text_parse
            else:
                function_args = function_call_part.function_call.args
                if not function_args or "plan_steps" not in function_args:
                    raise ValueError("Geminiレスポンスのfunction_call.argsにplan_stepsが含まれていません。")
                plan_steps = function_args["plan_steps"]

            if isinstance(plan_steps, str): # plan_stepsが文字列でJSON配列としてエンコードされている場合
                try:
                    plan_steps = json.loads(plan_steps)
                except json.JSONDecodeError as e_json_str:
                    raise ValueError(f"plan_stepsのJSON文字列のデコードに失敗: {e_json_str}. 文字列: '{plan_steps}'")

            if not isinstance(plan_steps, list):
                raise ValueError(f"plan_stepsがリスト形式ではありません。型: {type(plan_steps)}")

            for step in plan_steps: # 各ステップの検証
                if not (isinstance(step, dict) and "action" in step and "details" in step):
                    raise ValueError("分析プラン内の各ステップが正しい形式ではありません（'action'と'details'必須）。")
                allowed_actions = {"clarify", "check_history", "sql", "chart", "interpret", "data_processing"}
                if step["action"] not in allowed_actions:
                    logging.warning(f"プランに未知のアクション '{step['action']}' が含まれています。")

            parsed_plan = plan_steps

        except google_exceptions.GoogleAPICallError as e_api:
            logging.error(f"Gemini API呼び出し中にエラーが発生しました: {e_api}", exc_info=True)
            return {**state, "analysis_plan": None, "current_plan_step_index": None, "user_clarification": None,
                    "condition": "plan_generation_failed", "error": f"分析計画の作成中にモデル呼び出しでエラーが発生しました。管理者にご連絡ください。(詳細: {e_api.message})"}
        except (ValueError, AttributeError, json.JSONDecodeError) as e_parse:
            logging.error(f"Geminiレスポンスの処理または検証中にエラーが発生しました: {e_parse}", exc_info=True)
            return {**state, "analysis_plan": None, "current_plan_step_index": None, "user_clarification": None,
                    "condition": "plan_generation_failed", "error": f"分析計画の生成中にモデル応答の解析に失敗しました。形式が正しくない可能性があります。(詳細: {str(e_parse)})"}

    except Exception as e_outer: # モデル初期化やプロンプト構築など、APIコール以外の部分での予期せぬエラー
        logging.error(f"分析計画作成ノードで予期せぬエラーが発生しました: {e_outer}", exc_info=True)
        return {
            **state, "analysis_plan": None, "current_plan_step_index": None, "user_clarification": None,
            "condition": "plan_generation_failed", "error": f"分析計画の作成中に予期せぬシステムエラーが発生しました。管理者にご連絡ください。(詳細: {str(e_outer)})"
        }

    # current_inputの取得とoriginal_queryのロジックは変更なし
    current_input = state.get("input", "")
    original_query = state.get("original_query")
    if parsed_plan and not original_query and \
       ("clarify" in [step.get("action") for step in parsed_plan if isinstance(step, dict)] or user_clarification) :
        original_query = user_query

    # user_clarification はここで消費されるのでNoneにする
    return {
        **state,
        "original_query": original_query,
        "analysis_plan": parsed_plan, # parsed_planがNoneの場合もある（エラー時など）
        "current_plan_step_index": 0 if parsed_plan else None,
        "awaiting_step_confirmation": False,
        "user_clarification": None,
        "condition": "plan_generated" if parsed_plan else "empty_plan_generated", # エラー時はplan_generation_failedがセットされているはず
        "error": state.get("error") # エラー時はここでセットされたエラーメッセージが維持される
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
    original_query = state.get("original_query", "分析計画はキャンセルされました。")
    return {
        **state,
        "analysis_plan": None,
        "current_plan_step_index": None,
        "awaiting_step_confirmation": False,
        "original_query": None,
        "input": original_query,
        "interpretation": "複数ステップの分析計画はユーザーによってキャンセルされました。",
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
            "error": f"check_history_nodeにおいて、analysis_planの長さ（{len(analysis_plan) if analysis_plan else 0}）に対して無効なcurrent_plan_step_index（{current_plan_step_index}）です。"
        }

    # df_historyを取得する
    df_history = state.get("df_history", [])
    data_requirements_for_step = []
    current_step_details = state.get("input") # これはexecute_plan_routerによってcurrent_step["details"]に設定されます

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
            "error": "SQL実行のための要件が提供されていません。"
        }

    # latest_dfは必ずOrderedDictにして、すでにデータがあれば引き継ぐ
    current_latest_df = state.get("latest_df", collections.OrderedDict())
    if not isinstance(current_latest_df, collections.OrderedDict):
        current_latest_df = collections.OrderedDict(current_latest_df or {})

    # 全体の質問文脈（Clarify含む）があればそちら、なければ直近入力
    current_df_history = state.get("df_history", [])
    overall_user_input_context = state.get("original_query", state.get("input", "")) # これはexecute_plan_routerによってcurrent_step["details"]に設定されます
    
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
            logging.warning(f"'{req_string}'に対する最初のSQLが失敗しました: {sql_error}。修正して再試行します。")
            fixed_sql = fix_sql_with_llm(sql_generated_clean, sql_error, rag_tables, rag_queries, req_string)
            last_sql_generated = fixed_sql # この要件に対する最新のSQL試行を保存
            result_df, sql_error = try_sql_execute(fixed_sql)
            if sql_error:
                logging.error(f"'{req_string}'に対する修正SQLも失敗しました: {sql_error}。")
                user_friendly_error = transform_sql_error(sql_error)
                # Return immediately if SQL execution fails after attempting to fix it
                return {
                    **state,
                    "latest_df": current_latest_df,
                    "df_history": current_df_history,
                    "SQL": last_sql_generated,
                    "condition": "sql_execution_failed",
                    "error": f"SQLの実行に失敗しました (要件: '{req_string}'): {user_friendly_error}",
                    "missing_data_requirements": requirements_to_fetch # Keep all requirements as missing if one fails
                }

        if not sql_error and result_df is not None:
            if result_df.empty:
                logging.info(f"SQL for '{req_string}' は正常にSQL実行されましたが、データがありませんでした。")
                # Return immediately if SQL execution returns no data
                return {
                    **state,
                    "latest_df": current_latest_df,
                    "df_history": current_df_history,
                    "SQL": last_sql_generated,
                    "condition": "sql_execution_empty_result",
                    "error": f"SQLの実行結果が空でした (要件: '{req_string}')。",
                    "missing_data_requirements": requirements_to_fetch # Keep all requirements as missing if one returns empty
                }
            result_df_dict = result_df.to_dict(orient="records")
            current_latest_df[req_string] = result_df_dict # 取得したデータをlatest_dfに保存
            successfully_fetched_reqs.append(req_string)
            new_history_entry = {
                "id": uuid.uuid4().hex[:8], "query": req_string,
                "timestamp": datetime.now().isoformat(),
                "dataframe_dict": result_df_dict, "SQL": last_sql_generated
            }
            current_df_history.append(new_history_entry)
            
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
    plan_details_context = state.get("input", "全般的な傾向") # 特定の詳細がない場合のデフォルト

    if latest_df_data is None:
        logging.info("interpret_node: latest_dfがNoneのため、解釈できるデータがありません。")
        return {**state, "interpretation": "解釈するデータがありません。", "condition": "interpretation_failed_no_data"}

    if not isinstance(latest_df_data, collections.OrderedDict):
        logging.error(f"interpret_node: latest_dfの型がOrderedDictではありません (type: {type(latest_df_data)}).")
        return {**state, "interpretation": "データの形式が不正です (OrderedDictではありません)。", "condition": "interpretation_failed_bad_format"}

    if not latest_df_data: # OrderedDictが空かどうかを確認
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
                logging.error(f"interpret_node: '{req_string}'のデータをDataFrameに変換中にエラーが発生しました: {e}")
                full_data_string += f"■「{req_string}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n"
        else:
            full_data_string += f"■「{req_string}」に関するデータ:\n(この要件に対するデータはありませんでした)\n\n"

    # 一応OrderedDictとしてデータ構造はあるが、中身が全て空リストや形式エラーで“実質的に有効なデータが1件も無い”」場合にキャッチ
    # 例{"2022年売上": [], "2023年売上": []}
    processed_parts = [part for part in full_data_string.split("■")[1:] if part] # "■"で分割した後、空でない部分を取得
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
            logging.warning("interpret_node: LLMが空の解釈を返しました。")
            final_condition = "interpretation_failed"
            interpretation_text = "解釈の生成に失敗しました。"
        else:
            final_condition = "interpretation_done"
    except Exception as e:
        logging.error(f"interpret_node: LLM呼び出し中に例外が発生しました: {e}")
        interpretation_text = "解釈中にエラーが発生しました。"
        final_condition = "interpretation_failed"
    return {**state, "interpretation": interpretation_text, "condition": final_condition}


def chart_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")
    # 計画ステップ詳細からのグラフ作成指示（現在はstate.input内）
    chart_instructions = state.get("input", "Generate a suitable chart for the available data.")

    if not latest_df_data or not isinstance(latest_df_data, collections.OrderedDict) or not any(v for v in latest_df_data.values() if v): # 少なくとも1つのdfにデータがあることを確認
        return {**state, "chart_result": None, "condition": "chart_generation_failed_no_data"}

    df_to_plot = None
    selected_key_for_chart = None

    # chart_instructionsがキーを参照している場合、それに基づいてDataFrameを選択しようと試みます
    if isinstance(chart_instructions, str):
        for key in latest_df_data.keys():
            # それを一致と見なす前に、キーのデータが空でないことを確認してください
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
                logging.warning(f"chart_node: 選択されたキー '{selected_key_for_chart}' のDataFrameが空です。")
                selected_key_for_chart = None # フォールバックをトリガーするために無効化
        except Exception as e:
            logging.error(f"chart_node: 選択されたキー '{selected_key_for_chart}' のDataFrame作成中にエラーが発生しました: {e}")
            selected_key_for_chart = None # フォールバックをトリガーするために無効化

    if df_to_plot is None: # フォールバック：最初の空でないDataFrameを使用
        for key, data_list in latest_df_data.items():
            if data_list and isinstance(data_list, list) and len(data_list) > 0:
                try:
                    df_candidate = pd.DataFrame(data_list)
                    if not df_candidate.empty:
                        df_to_plot = df_candidate
                        selected_key_for_chart = key
                        chart_context_message = f"以下の「{selected_key_for_chart}」に関するデータ（複数あるデータセットの最初で、有効なデータを含むもの）と、ユーザーからの「{chart_instructions}」という指示に基づいて"
                        if len(latest_df_data) > 1 and (not isinstance(chart_instructions, str) or selected_key_for_chart.lower() not in chart_instructions.lower()): # chart_instructionsが文字列でないか、またはselected_keyがchart_instructionsに含まれていないかを確認
                            logging.warning(f"chart_node: 複数のDataFrameが利用可能です。一般的または不一致のグラフ作成指示（'{chart_instructions}'）のため、最初の空でないDataFrame（'{selected_key_for_chart}'）をデフォルトとします。")
                        break
                except Exception as e:
                    logging.error(f"chart_node: フォールバック中にlatest_dfキー '{key}' からDataFrameを作成中にエラーが発生しました: {e}")
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
        agent_kwargs={"handle_parsing_errors": True} # LLMの出力がエージェントにとって不適切だった場合の、より堅牢なエラー処理のために追加
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
            logging.error("chart_node: エージェントは実行されましたが、output.pngが見つかりませんでした。エージェントの出力に理由が示されている可能性があります。")
            error_message = "グラフファイルの生成に失敗しました。"
            if isinstance(agent_response, dict) and "output" in agent_response:
                error_message += f" (エージェント: {agent_response['output']})"
            return {**state, "chart_result": None, "condition": "chart_generation_failed_no_output_file", "error": error_message}
    except Exception as e:
        logging.error(f"chart_node: エージェント実行中にエラーが発生しました: {e}", exc_info=True)
        return {**state, "chart_result": None, "condition": "chart_generation_failed_agent_error", "error": str(e)}

def data_processing_node(state: MyState) -> MyState:
    latest_df_data = state.get("latest_df")
    processing_instructions = state.get("input", "データフレームに対して指示された処理を実行してください。") # Plan step details

    if not latest_df_data or not isinstance(latest_df_data, collections.OrderedDict) or not any(v for v in latest_df_data.values() if v):
        return {
            **state,
            "latest_df": latest_df_data, # Keep existing data even if empty
            "condition": "data_processing_failed_no_data",
            "error": "データ処理のための有効なデータがありません。"
        }

    df_to_process = None
    selected_key = None # This will store the key of the df that is chosen for processing
    processing_context_message = ""

    # New logic: Check for DF_KEY['key_name'] pattern in instructions
    df_key_match = re.search(r"DF_KEY\['(.*?)'\]", processing_instructions)

    if df_key_match:
        explicit_key = df_key_match.group(1)
        if explicit_key in latest_df_data and latest_df_data[explicit_key] and isinstance(latest_df_data[explicit_key], list):
            try:
                df_candidate = pd.DataFrame(latest_df_data[explicit_key])
                if not df_candidate.empty:
                    df_to_process = df_candidate
                    selected_key = explicit_key
                    processing_context_message = f"DF_KEY['{explicit_key}']で指定されたデータと、ユーザーからの「{processing_instructions}」という指示に基づいて"
                    logging.info(f"data_processing_node: DF_KEY syntax found. Processing DataFrame with key '{selected_key}'.")
                else:
                    logging.warning(f"data_processing_node: DF_KEY['{explicit_key}'] found, but the data is empty. Proceeding to fallback selection.")
            except Exception as e:
                logging.error(f"data_processing_node: DF_KEY['{explicit_key}'] found, but failed to create DataFrame: {e}. Proceeding to fallback selection.")
        else:
            logging.warning(f"data_processing_node: DF_KEY['{explicit_key}'] not found in latest_df_data or data is invalid/empty. Proceeding to fallback selection.")

    # Fallback logic: If DF_KEY was not used, or was invalid.
    if df_to_process is None:
        # Try to find a DataFrame based on instruction key reference (original fallback)
        if isinstance(processing_instructions, str):
            for key_option in latest_df_data.keys(): # iterate through available DFs
                # Check if key_option is mentioned in instructions (and not part of a DF_KEY pattern already handled)
                # This check is simplified; more robust would be to ensure it's not within DF_KEY[...]
                if key_option.lower() in processing_instructions.lower() and (not df_key_match or key_option != df_key_match.group(1)):
                    if latest_df_data.get(key_option) and isinstance(latest_df_data[key_option], list) and len(latest_df_data[key_option]) > 0:
                        try:
                            df_candidate = pd.DataFrame(latest_df_data[key_option])
                            if not df_candidate.empty:
                                df_to_process = df_candidate
                                selected_key = key_option # Set selected_key here
                                processing_context_message = f"指示内容で言及されたキー「{selected_key}」のデータと、ユーザーからの「{processing_instructions}」という指示に基づいて"
                                logging.info(f"data_processing_node: Fallback - Found matching key '{selected_key}' in instructions.")
                                break # Found a suitable DF by instruction mention
                        except Exception as e:
                            logging.error(f"data_processing_node: Fallback - Error creating DataFrame for key '{key_option}': {e}")
                            continue # Try next key

        if df_to_process is None: # If still no DF, use the first non-empty one
            for key_option, data_list in latest_df_data.items():
                if data_list and isinstance(data_list, list) and len(data_list) > 0:
                    try:
                        df_candidate = pd.DataFrame(data_list)
                        if not df_candidate.empty:
                            df_to_process = df_candidate
                            selected_key = key_option # Set selected_key here
                            processing_context_message = f"利用可能な最初のデータセット「{selected_key}」と、ユーザーからの「{processing_instructions}」という指示に基づいて"
                            logging.info(f"data_processing_node: Fallback - No specific key matched or DF_KEY invalid. Using first available non-empty DataFrame with key '{selected_key}'.")
                            if len(latest_df_data) > 1:
                                 logging.warning(f"data_processing_node: 複数のDataFrameが存在し、特定のDFが指示されなかったため、最初の空でないDataFrame（'{selected_key}'）を使用します。特定のDFを対象にするには DF_KEY['キー名'] を使用してください。")
                            break
                    except Exception as e:
                        logging.error(f"data_processing_node: Fallback - Error creating first available DataFrame for key '{key_option}': {e}")
                        continue

    if df_to_process is None or df_to_process.empty:
        return {
            **state,
            "latest_df": latest_df_data,
            "condition": "data_processing_failed_empty_selected_data",
            "error": "データ処理に適したデータが見つかりませんでした。"
        }

    python_tool = PythonAstREPLTool(
        locals={"df": df_to_process.copy(), "pd": pd}, # dfのコピーを渡すことで、元のREPL環境への影響を限定
        description="Pythonコードを実行してDataFrameを処理します。処理対象のDataFrameは 'df' として利用可能です。"
    )
    tools = [python_tool]
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True,
        agent_kwargs={"handle_parsing_errors": "True"} # より堅牢なエラー処理
    )

    # df.info()の出力を文字列としてキャプチャ
    buffer = io.StringIO()
    df_to_process.info(buf=buffer)
    df_info_str = buffer.getvalue()

    available_dataframes_info = "\n".join([f"- '{key}': {len(value)}行" for key, value in latest_df_data.items() if isinstance(value, list)])

    prompt_guidance_for_df_selection = ""
    if len(latest_df_data) > 1:
        prompt_guidance_for_df_selection = (
            f"複数のDataFrameが利用可能です。特定のDataFrameを処理対象とする場合、指示内に `DF_KEY['キー名']` という形式で指定してください。\n"
            f"例: `DF_KEY['sales_data_2023'] 列'A'の値を2倍にする`\n"
            f"利用可能なDataFrame (キー: 行数):\n{available_dataframes_info}\n"
            f"DF_KEY構文を使用しない場合、最も適切と思われるDataFrameを自動選択します。\n"
        )

    processing_prompt = f"""
    あなたはPythonプログラミングとデータ処理の専門家です。
    {prompt_guidance_for_df_selection}
    {processing_context_message}pandas DataFrameに必要な処理を実行してください。
    処理後のDataFrameは、必ず 'df' という名前の変数に再代入してください。
    dfの列情報:
{df_info_str}
    dfの最初の5行:
    {df_to_process.head().to_string(index=False)}
    処理指示: {processing_instructions}

    実行すべきPythonコードのみを考えてください。最終的なDataFrameは 'df' に格納する必要があります。
    """

    try:
        agent_response = agent.invoke(processing_prompt)
        logging.info(f"data_processing_node: Agent response: {agent_response}")

        processed_df = python_tool.globals.get("df")

        if not isinstance(processed_df, pd.DataFrame):
            logging.error(f"data_processing_node: エージェント実行後、'df' はDataFrameではありません。Type: {type(processed_df)}")
            # エージェントの出力（思考プロセスや最終的なテキスト応答）をエラーメッセージに含める
            agent_output_for_error = agent_response.get('output', 'エージェントからの具体的な出力なし') if isinstance(agent_response, dict) else str(agent_response)
            return {
                **state,
                "latest_df": latest_df_data, # Keep original data
                "condition": "data_processing_failed_bad_output",
                "error": f"データ処理後、期待されるDataFrame形式ではありませんでした。エージェントの出力: {agent_output_for_error}"
            }

        if processed_df.equals(df_to_process):
            logging.warning("data_processing_node: DataFrameはエージェントによって変更されませんでした。")
            # ユーザーに通知するか、何もしないかは要件による。ここでは成功として扱うが、interpretationに含めることを検討。
            # interpretation = state.get("interpretation", "") + "\n注意: データ処理の指示がありましたが、結果のデータに変更はありませんでした。"
            # state["interpretation"] = interpretation

        # Update latest_df with the processed DataFrame
        updated_latest_df = latest_df_data.copy() # Make a copy to modify
        updated_latest_df[selected_key] = processed_df.to_dict(orient="records")

        return {
            **state,
            "latest_df": updated_latest_df,
            "condition": "data_processing_done",
            "error": None
        }

    except Exception as e:
        logging.error(f"data_processing_node: エージェント実行または結果処理中にエラー: {e}", exc_info=True)
        return {
            **state,
            "latest_df": latest_df_data, # Keep original data on error
            "condition": "data_processing_failed_agent_error",
            "error": f"データ処理エージェントの実行中にエラーが発生しました: {str(e)}"
        }

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
        logging.warning(f"clarify_next: clarify_nodeから予期しない状態 '{condition}' が返されました。ENDにデフォルト設定します。")
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
        logging.error("create_analysis_plan_next: 分析計画の生成に失敗しました。ENDにルーティングします。")
        return END
    else:
        logging.warning(f"create_analysis_plan_next: 不明な状態 '{condition}' です。ENDにルーティングします。")
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
    logging.warning("sql_next: SQLノードが計画外のコンテキストで実行されました。")
    return END

def chart_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("chart_next: チャートノードが計画外のコンテキストで実行されました。")
    return END

def interpret_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("interpret_next: 解釈ノードが計画外のコンテキストで実行されました。")
    return END

def data_processing_next(state: MyState):
    current_index = state.get("current_plan_step_index")
    if current_index is not None:
        state["current_plan_step_index"] = current_index + 1
        state["awaiting_step_confirmation"] = False
        return "dispatch_plan_step"
    logging.warning("data_processing_next: データ処理ノードが計画外のコンテキストで実行されました。")
    return END

# 分析計画（analysis_plan）」の現在ステップに応じて、次にどのノード（処理）に進むかを決定
def execute_plan_dispatch_step(state: MyState) -> str:
    # Clarify（追加質問）へのユーザー回答があれば、分析プランを再生成する
    if state.get("user_clarification"):
        logging.info("execute_plan_dispatch_step: ユーザーの追加情報が入力されたため、分析プランを再作成します。")
        return "create_analysis_plan"
    
    analysis_plan = state.get("analysis_plan")
    current_plan_step_index = state.get("current_plan_step_index")

    # プランが存在しない・インデックスが不正・プラン完了時は初期化して提案ノードに遷移
    if not analysis_plan or current_plan_step_index is None or not (0 <= current_plan_step_index < len(analysis_plan)):
        logging.info("execute_plan_dispatch_step: プランが完了、または不正な状態です。")
        state["analysis_plan"] = None
        state["current_plan_step_index"] = None
        state["awaiting_step_confirmation"] = False
        return END
        
    current_step = analysis_plan[current_plan_step_index]
    action = current_step.get("action")
    logging.info(f"execute_plan_dispatch_step: Dispatching to action '{action}' for step {current_plan_step_index}.")
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
    elif action == "data_processing":
        return "data_processing"
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
    workflow.add_node("dispatch_plan_step", lambda state: state) # 中央ルーティングロジック用の新しいダミーノード
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("check_history", check_history_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_node("data_processing", data_processing_node) # 新しいノードを追加
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
        execute_plan_dispatch_step,
        {
            "create_analysis_plan": "create_analysis_plan",
            "check_history": "check_history",
            "clarify": "clarify",
            "sql": "sql",
            "chart": "chart",
            "interpret": "interpret",
            "data_processing": "data_processing", # 新しいノードへのルーティングを追加
            "END": END
        }
    )

    workflow.add_conditional_edges(
        "sql",
        sql_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END  # ENDが返される可能性がある場合は明示的にマッピング
        }
    )
    workflow.add_conditional_edges(
        "data_processing",
        data_processing_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "chart",
        chart_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END # ENDが返される可能性がある場合は明示的にマッピング（現在のchart_nextにはありませんが）
        }
    )
    workflow.add_conditional_edges(
        "interpret",
        interpret_next,
        {
            "dispatch_plan_step": "dispatch_plan_step",
            END: END # ENDが返される可能性がある場合は明示的にマッピング（現在のinterpret_nextにはありませんが）
        }
    )
    workflow.add_conditional_edges(
        "check_history",
        check_history_next,
        {
            "dispatch_plan_step": "dispatch_plan_step"
            # check_history_nextが他の値を返す可能性がある場合は、ここでマッピングします。
            # 現時点では "dispatch_plan_step" のみを返します。
        }
    )
    workflow.add_conditional_edges(
        "clarify",
        clarify_next,
        {
            END: END
            # clarify_next は常に END または END につながる条件を返します。
        }
    )


    return workflow.compile(checkpointer=memory)

# user_query = "カテゴリの合計販売金額を出して"
# workflow = build_workflow()
# config = {"configurable": {"thread_id": "2"}}
# res = workflow.invoke({"input": user_query}, config=config)
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
