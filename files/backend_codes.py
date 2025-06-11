# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

import os
import pandas as pd
import logging
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import ast # literal_evalのため
import plotly.express as px
import json # import jsonを先頭に移動しました

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langchain.callbacks.base import BaseCallbackHandler
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.graph import StateGraph, END
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from files.functions import extract_sql, try_sql_execute, fix_sql_with_llm, get_table_name_from_formatted_doc

# # 環境変数からLLMモデル名を取得します（デフォルト値あり）
# llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash") # デフォルトをgemini-1.5-proに変更
# google_api_key = os.getenv("GOOGLE_API_KEY")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
# llm = ChatGoogleGenerativeAI(
#     model=llm_model_name,
#     temperature=0,
#     google_api_key=google_api_key
# ) # または、より高性能な "gemini-1.5-flash"、"gemini-1.5-pro"

#Chatモデル（SQL生成用）
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
version4emb = os.getenv("AZURE_OPENAI_API_VERSION4EMB")
deployment4emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4EMB")
version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

llm = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version=version,
    deployment_name=deployment,
    temperature=0,
    streaming=False
)

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version=version4emb,
    azure_deployment=deployment4emb
)
# ベクトルストア
vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)

class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str  # 次に実行すべきノード名
    df_history: List[dict]
    plan: List[Dict[str, str]]  # planning_nodeが生成したプラン
    plan_step: int  # 現在のステップ番号

history_back_number = 5
RERANK_CANDIDATE_COUNT = 5

@tool
def metadata_retrieval_node(task_description: str, conversation_history: list[str]):
    """
    自然言語のタスク記述と直近の会話履歴を受け取り、文脈を理解した上でテーブル定義を示します。
    ユーザーからデータやテーブル、カラムについて質問があった時に使用します。
    """
    logging.info(f"metadata_retrieval_node: 開始")
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    # Rag情報
    retrieved_docs = vectorstore_tables.similarity_search(task_description)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_template = """
    以下のテーブル定義情報を参照して、ユーザーの質問に対して簡潔に、わかりやすく答えてください。

    【テーブル定義情報】
    {retrieved_table_info}

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【回答】
    """
    llm_prompt = prompt_template.format(retrieved_table_info=retrieved_table_info, task_description=task_description, context=context)
    response = llm.invoke(llm_prompt)
    metadata_answer = response.content.strip()
    logging.info(f"metadata_retrieval_node: 回答完了{metadata_answer}")

    return metadata_answer

@tool
def analysis_plan_node(task_description: str, conversation_history: list[str]):
    """
    自然言語のタスク記述と直近の会話履歴を受け取り、文脈を理解した上で必要な分析ステップを考えます。
    ユーザーから分析依頼があり、分析要件を具体化したい際に使います。
    """
    logging.info(f"analysis_plan_node: 開始")

    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    # Rag情報
    logging.info(f"analysis_plan_node: RAG情報を読み込み中 '{task_description}'")
    retrieved_docs = vectorstore_tables.similarity_search(task_description)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt_template = """
    以下の情報を参照して、ユーザーの質問に答えるための分析計画を立ててください。

    【テーブル定義情報】
    {retrieved_table_info}

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【回答】
    """
    logging.info(f"analysis_plan_node: 生成AIが考えています・・・ '{task_description}'")
    llm_prompt = prompt_template.format(retrieved_table_info=retrieved_table_info, task_description=task_description, context=context)
    response = llm.invoke(llm_prompt)
    step_answer = response.content.strip()
    logging.info(f"analysis_plan_node: 回答完了{step_answer}")

    return step_answer


def planning_node(state: AgentState):
    """ユーザーの最初の要望から分析プランをJSON形式で生成する"""
    logging.info("planning_node: 開始")

    conversation_history = state.get("messages", [])
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    user_request = ""
    for msg in reversed(conversation_history):
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            break

    prompt_template = """
    あなたはデータ分析のプロジェクトマネージャーです。ユーザーの要望を実現するための
    具体的な分析手順を考え、次のJSON形式で出力してください。

    フォーマット例:
    {
      "plan": [
        {"agent": "sql_node", "task": "必要なデータをSQLで取得する"},
        {"agent": "processing_node", "task": "取得したデータを加工・可視化する"},
        {"agent": "interpret_node", "task": "結果を解釈して結論をまとめる"}
      ]
    }

    【ユーザー要望】
    {user_request}

    【文脈】
    {context}
    """

    llm_prompt = prompt_template.format(user_request=user_request, context=context)
    response = llm.invoke(llm_prompt)
    plan_text = response.content.strip()

    try:
        plan_dict = json.loads(plan_text)
        plan = plan_dict.get("plan", [])
    except Exception as e:
        logging.error(f"planning_node: JSON解析に失敗しました: {e}")
        plan = []

    ai_message = AIMessage(content=plan_text)

    logging.info(f"planning_node: プラン生成完了 {plan}")

    return {
        "messages": [ai_message],
        "plan": plan,
        "plan_step": 0,
        "next": "supervisor",
    }

def sql_node(state: AgentState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、ルールを強く意識した上でSQLを生成・実行します。
    （二段階選抜とルール遵守CoTを実装）
    """
    try:
        supervisor_ai_message = state["messages"][-1]
        if not isinstance(supervisor_ai_message, AIMessage) or not supervisor_ai_message.tool_calls:
            error_message_payload = {"status": "error", "error_message": "不正な呼び出し形式です。スーパーバイザーからのAIMessage(tool_calls付き)が見つかりません。"}
            logging.error(f"sql_node: {error_message_payload['error_message']}")
            return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=None)]}

        first_tool_call = supervisor_ai_message.tool_calls[0]
        tool_call_id = first_tool_call['id']

        arguments = first_tool_call.get('args')
        if not isinstance(arguments, dict):
            error_message_payload = {"status": "error", "error_message": "argumentsの形式がdictではありません。"}
            logging.error(f"sql_node: {error_message_payload['error_message']}")
            return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=tool_call_id)]}
        
        task_description = arguments.get('task_description')
        if task_description is None:
            error_message_payload = {"status": "error", "error_message": "task_descriptionがスーパーバイザーの指示に含まれていません。"}
            logging.error(f"sql_node: {error_message_payload['error_message']}")
            return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=tool_call_id)]}
        # --- 修正ここまで ---

        conversation_history = state.get("messages", [])
        non_system_history = [msg for msg in conversation_history if msg.type != "system"]
        recent_history = non_system_history[-history_back_number:]
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    except (IndexError, KeyError, json.JSONDecodeError, AttributeError, ValueError) as e:
        error_message_detail = f"ノード初期化エラー: {e}. Messages: {state.get('messages')}"
        logging.error(f"sql_node: {error_message_detail}")
        current_tool_call_id = locals().get("tool_call_id", None) # tool_call_idが取得できていればそれを使う
        error_message_payload = {"status": "error", "error_message": error_message_detail}
        return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=current_tool_call_id)]}

    # ★ --- 1. 候補となるテーブルを多めに取得（Re-rankingステップ） ---
    logging.info(f"sql_node(Re-rank): 候補テーブルを広く検索中 (k=RERANK_CANDIDATE_COUNT)...")
    candidate_docs = vectorstore_tables.similarity_search(task_description, k=RERANK_CANDIDATE_COUNT)
    candidate_table_names = [get_table_name_from_formatted_doc(doc.page_content) for doc in candidate_docs]
    candidate_tables_info_for_prompt = "\n\n".join(
        [doc.page_content for name, doc in zip(candidate_table_names, candidate_docs) if name]
    )
    # ★ --- 2. LLMにテーブルを選別させる（1回目の問いかけ） ---
    if candidate_tables_info_for_prompt:
        logging.info(f"sql_node(Re-rank): LLMにテーブル選別を依頼中...")
        rerank_prompt = f"""
        ユーザーの最終的な要求は「{task_description}」です。
        この要求に答えるために、以下のテーブル定義リストの中から、SQLクエリの生成に必要だと思われるテーブル名を1つだけ選び出して出力してください。
        余計な説明は不要です。テーブル名のみを出力してください。

        【テーブル定義リスト】
        {candidate_tables_info_for_prompt}

        【必要なテーブル名リスト（1つだけ）】
        """
        rerank_response = llm.invoke(rerank_prompt)
        required_table_names_str = rerank_response.content.strip()
        if required_table_names_str:
             required_table_names = [name.strip() for name in required_table_names_str.split(',')]
        else:
            logging.warning("LLMが選別するテーブル名を返さなかったため、検索候補を全て使用します。")
            required_table_names = [name for name in candidate_table_names if name]

        logging.info(f"sql_node(Re-rank): LLMが選別したテーブル: {required_table_names}")
    else:
        logging.warning("RAG検索でテーブル候補が見つかりませんでした。")
        required_table_names = []

    # ★ --- 3. LLMの選別結果を基に、使用するテーブル定義を絞り込む ---
    final_docs = [doc for doc in candidate_docs if get_table_name_from_formatted_doc(doc.page_content) in required_table_names]
    final_rag_tables_text = "\n\n".join([doc.page_content for doc in final_docs])

    if not final_rag_tables_text:
        logging.error("sql_node: 最終的に使用するテーブル情報が見つかりませんでした。")
        result_payload = {"status": "error", "error_message": "関連するテーブル情報が見つかりませんでした。"}
        return {"messages": [ToolMessage(content=json.dumps(result_payload), tool_call_id=tool_call_id)]}

    # ★ --- 4. ルール遵守CoTを組み込んだシステムプロンプトを作成 ---
    system_prompt = """
    あなたは非常に優秀なSQL生成AIです。ユーザーの要求、提供されたテーブル定義、
    そして最も重要な【テーブル全体に関するルール】（<table_explanation>タグ内）と
    各【カラム定義と個別ルール】（<column>タグ内の<explanation>タグ）を基に、SQLiteのSQLを生成します。

    SQLを生成する前に、必ず以下の思考プロセスに従って、ステップバイステップで考察を行ってください。
    あなたの最終的な出力は、思考プロセスを含まない、実行可能なSQLite SQLクエリ文のみにしてください。
    ---
    ### 思考プロセス (Chain-of-Thought)

    1.  **ユーザー要求の理解**: ユーザーが最終的に何を知りたいのかを明確にする。

    2.  **使用テーブルの確認とルール遵守**:
        * 今回使用するテーブルは、提供された【使用するテーブル定義】の中から最も適切なものを1つだけ選択する。
        * 選択したテーブルの【テーブル全体に関するルール】（<table_explanation>の内容）を特定し、このルールをSQL生成時にどのように考慮するか記述する。

    3.  **使用カラムの特定とルールチェック**:
        テーブル内のカラムについて【カラム定義と個別ルール】（<column>タグ内の<explanation>の内容）を確認します。以下のチェックリスト形式で思考を整理してください。ルールがない場合は「特になし」と記載する。

        | 使用するカラム名 | カラムの個別ルール                                     | ルールをSQLにどう反映させるか（SELECT句、WHERE句など） |
        | :--------------- | :----------------------------------------------------- | :----------------------------------------------------- |
        | (例) カテゴリ    | 必ず出力に含めてください。指定がない場合はWhere句で’All’ | SELECT句に「カテゴリ」を含める。WHERE句の条件を確認。 |
        | ...              | ...                                                    | ...                                                    |

    4.  **SQL組み立て方針の決定**:
        * `SELECT`句: 上記のルールチェックに基づき、どのカラムを含めるか。
        * `FROM`句: ステップ2で確認したテーブル名。
        * `WHERE`句: 上記のルールチェックとユーザー要求に基づき、どのような条件が必要か。
        * その他（`GROUP BY`, `ORDER BY`, `LIMIT`）: 必要に応じて。ただし、テーブル全体のルール（例：集計関数禁止など）やSQLiteの制約（JOIN禁止、単一テーブル使用）を厳守する。

    5.  **最終SQLの生成**: 上記の方針に基づき、最終的なSQLite SQLクエリを生成する。ルール違反がないか最終確認する。
    ---

    上記の思考プロセスを頭の中で実行し、完成したSQLクエリのみを出力してください。
    SQLの前後にコメントや説明文は出力しないでください。
    使用できるSQL構文は「SELECT」「WHERE」「GROUP BY」「ORDER BY」「LIMIT」のみです。
    日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    必ず1つのテーブルだけを使い、簡単な集計・フィルタ・並べ替えまでにしてください。
    """

    # --- 5. 本番のSQL生成用ユーザープロンプトを組み立てる ---
    retrieved_queries_docs = vectorstore_queries.similarity_search(task_description, k=3)
    rag_queries = "\n\n".join([doc.page_content for doc in retrieved_queries_docs])

    user_prompt = f"""
    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【使用するテーブル定義】
    {final_rag_tables_text}

    【類似する問い合わせ例とそのSQL】
    {rag_queries}

    この要件を満たし、かつ提示された全てのルールを厳守するSQLite SQLクエリを生成してください。
    """
    new_history_item = {} # tryブロックの外で参照できるように初期化
    # --- 6. SQL生成の実行と後続処理（2回目の問いかけ & 実行） ---
    try:
        logging.info(f"sql_node(CoT): 生成AIが考えています・・・ '{task_description}'")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        sql_generated_clean = extract_sql(response.content.strip())
        last_sql_generated = sql_generated_clean
        logging.info(f"sql_node: 生成SQL{last_sql_generated }")
        result_df, sql_error = try_sql_execute(sql_generated_clean)

        if sql_error:
            logging.warning(f"最初のSQLが失敗しました: {sql_error}。修正して再試行します。")
            fixed_sql = fix_sql_with_llm(llm, sql_generated_clean, sql_error, final_rag_tables_text, rag_queries, task_description, context)
            last_sql_generated = fixed_sql
            logging.info(f"sql_node: 修正SQL{last_sql_generated }")
            result_df, sql_error = try_sql_execute(fixed_sql)

            if sql_error:
                logging.error(f"修正SQLも失敗: {sql_error}")
                result_payload = {"status": "error", "error_message": str(sql_error), "last_attempted_sql": last_sql_generated}
            else:
                logging.info(f"sql_node: データの取得が完了しました。'")
                result_payload = {"status": "success", "executed_sql": last_sql_generated, "dataframe_as_json": result_df.to_json(orient="records")}
                if result_df is not None and not result_df.empty:
                     new_history_item = {task_description: result_df.to_dict(orient="records")}
        else:
            logging.info(f"sql_node: データの取得が完了しました。'")
            result_payload = {"status": "success", "executed_sql": last_sql_generated, "dataframe_as_json": result_df.to_json(orient="records")}
            if result_df is not None and not result_df.empty:
                new_history_item = {task_description: result_df.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"sql_nodeで予期せぬエラー: {e}")
        result_payload = {"status": "error", "error_message": f"SQLノードの実行中に予期せぬエラーが発生しました: {e}", "last_attempted_sql": locals().get("last_sql_generated", "N/A")}

    updated_df_history = state.get("df_history", [])
    if new_history_item: # 成功した場合のみ履歴に追加 (空の辞書でないことを確認)
        updated_df_history = updated_df_history + [new_history_item]

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(result_payload, ensure_ascii=False),
                tool_call_id=tool_call_id
            )
        ],
        "df_history": updated_df_history
    }

# 解釈ノード
def interpret_node(state: AgentState):
    """
    自然言語のタスク記述と直近の会話、そしてデータを受け取り、文脈を理解した上でデータを解釈します。
    分析のためにデータを解釈する必要がある際に使用します。
    """
    try:
        logging.info(f"interpret_node: 開始")
        # --- 修正: スーパーバイザーからの指示の受け取り方 ---
        supervisor_ai_message = state["messages"][-1]
        if not isinstance(supervisor_ai_message, AIMessage) or not supervisor_ai_message.tool_calls:
            error_message = "不正な呼び出し形式です。スーパーバイザーからのAIMessage(tool_calls付き)が見つかりません。"
            logging.error(f"interpret_node: {error_message}")
            return {"messages": [ToolMessage(content=error_message, tool_call_id=None)]}

        first_tool_call = supervisor_ai_message.tool_calls[0]
        tool_call_id = first_tool_call['id']

        arguments = first_tool_call.get('args')
        if not isinstance(arguments, dict):
            error_message_payload = {"status": "error", "error_message": "argumentsの形式がdictではありません。"}
            logging.error(f"interpret_node: {error_message_payload['error_message']}")
            return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=tool_call_id)]}

        task_description = arguments.get('task_description')
        if task_description is None:
            error_message = "task_descriptionがスーパーバイザーの指示に含まれていません。"
            logging.error(f"interpret_node: {error_message}")
            return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}
        # --- 修正ここまで ---

    except (IndexError, KeyError, json.JSONDecodeError, AttributeError, ValueError) as e:
        error_message_detail = f"ノード初期化エラー: {e}. Messages: {state.get('messages')}"
        logging.error(f"interpret_node: {error_message_detail}")
        current_tool_call_id = locals().get("tool_call_id", None)
        return {"messages": [ToolMessage(content=error_message_detail, tool_call_id=current_tool_call_id)]}

    logging.info(f"interpret_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", []) # getのデフォルトを空リストに
    if not df_history: # df_historyがNoneまたは空の場合
        # RuntimeErrorを発生させる代わりに、ToolMessageでエラーを返す
        error_message = "interpret_node: df_historyが空です。利用可能なデータがありません。"
        logging.warning(error_message) # Warningレベルに変更
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    full_data_list = []
    for entry in df_history:
        for question, data in entry.items():
            try:
                df = pd.DataFrame(data)
                if not df.empty:
                    full_data_list.append(f"■「{question}」に関するデータ:\n{df.to_string(index=False)}\n\n")
                else:
                    full_data_list.append(f"■「{question}」に関するデータ:\n(この要件に対するデータは空でした)\n\n")
            except Exception as e:
                logging.error(f"interpret_node: 'df_historyをDataFrameに変換中にエラーが発生しました: {e}")
                full_data_list.append(f"■「{question}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n")
    full_data_string = "".join(full_data_list)

    processed_parts = [part for part in full_data_string.split("■")[1:] if part] # ""を除外
    all_parts_indicate_no_data = False # 初期化
    if not processed_parts: # 分割後が空（つまりfull_data_stringが空か「■」を含まない）
        all_parts_indicate_no_data = True
    else:
        all_parts_indicate_no_data = all(
            "(この要件に対するデータはありませんでした)" in part or \
            "(データ形式エラーのため表示できません)" in part or \
            "(この要件に対するデータは空でした)" in part
            for part in processed_parts
        )

    if all_parts_indicate_no_data:
        # RuntimeErrorを発生させる代わりに、ToolMessageでエラーを返す
        error_message = "interpret_node: 利用可能なデータがありませんでした（空またはエラー）。"
        logging.warning(error_message) # Warningレベルに変更
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    conversation_history = state.get("messages", [])
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    system_prompt = "あなたは優秀なデータ分析の専門家です。"
    user_prompt = f"""
    現在のタスクと文脈を踏まえて、データから読み取れる特徴や傾向を簡潔な日本語で解説してください。

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【データの内容】
    {full_data_string}
    """
    try:
        logging.info(f"interpret_node: 生成AIが考えています・・・ '{task_description}'")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        interpretation_text = response.content.strip() if response.content else ""

        if interpretation_text:
            logging.info(f"interpret_node: 解釈作成完了{interpretation_text}")
            return {
                "messages": [
                    ToolMessage(
                        content=interpretation_text,
                        tool_call_id=tool_call_id
                    )
                ]
            }
        else:
            # RuntimeErrorを発生させる代わりに、ToolMessageでエラーを返す
            error_message = "interpret_node: LLMが空の解釈を返しました。"
            logging.warning(error_message) # Warningレベルに変更
            return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    except Exception as e:
        error_message = f"データの解釈中にエラーが発生しました: {e}"
        logging.error(error_message)
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}



# python用のCallback
class CodeCollectorCallback(BaseCallbackHandler):
    def __init__(self):
        self.codes = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> Any:
        if serialized.get("name") == "python_ast_repl":
            self.codes.append(input_str)

def processing_node(state: AgentState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータの加工やグラフの作成を行います。
    分析のためにデータの加工やグラフ作成をする必要がある際に使用します。
    """
    try:
        logging.info(f"processing_node: 開始")
        # --- 修正: スーパーバイザーからの指示の受け取り方 ---
        supervisor_ai_message = state["messages"][-1]
        if not isinstance(supervisor_ai_message, AIMessage) or not supervisor_ai_message.tool_calls:
            error_message = "不正な呼び出し形式です。スーパーバイザーからのAIMessage(tool_calls付き)が見つかりません。"
            logging.error(f"processing_node: {error_message}")
            return {"messages": [ToolMessage(content=error_message, tool_call_id=None)]}

        first_tool_call = supervisor_ai_message.tool_calls[0]
        tool_call_id = first_tool_call['id']

        arguments = first_tool_call.get('args')
        if not isinstance(arguments, dict):
            error_message_payload = {"status": "error", "error_message": "argumentsの形式がdictではありません。"}
            logging.error(f"processing_node: {error_message_payload['error_message']}")
            return {"messages": [ToolMessage(content=json.dumps(error_message_payload), tool_call_id=tool_call_id)]}

        task_description = arguments.get('task_description')
        if task_description is None:
            error_message = "task_descriptionがスーパーバイザーの指示に含まれていません。"
            logging.error(f"processing_node: {error_message}")
            return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    except (IndexError, KeyError, json.JSONDecodeError, AttributeError, ValueError) as e:
        error_message_detail = f"ノード初期化エラー: {e}. Messages: {state.get('messages')}"
        logging.error(f"processing_node: {error_message_detail}")
        current_tool_call_id = locals().get("tool_call_id", None)
        return {"messages": [ToolMessage(content=error_message_detail, tool_call_id=current_tool_call_id)]}

    logging.info(f"processing_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", []) # getのデフォルトを空リストに
    if not df_history: # df_historyがNoneまたは空の場合
        error_message = "processing_node: df_historyが空です。利用可能なデータがありません。"
        logging.warning(error_message)
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    recent_items = df_history[-history_back_number:]
    df_explain_list = []
    df_locals = {}
    for idx, entry in enumerate(recent_items):
        for question, data in entry.items():
            try:
                df = pd.DataFrame(data)
                df_name = f"df{idx}"
                if not df.empty:
                    df_locals[df_name] = df
                    explain = f"""
                    # {df_name}
                    ## 内容:【{question}】に関するデータ
                    ## dfの列情報: {list(df.columns)}
                    ## dfの最初の5行:\n{df.head().to_string(index=False)}
                    """
                    df_explain_list.append(explain)
                else:
                    df_explain_list.append(f"■「{question}」に関するデータ:\n(この要件に対するデータは空でした)\n\n")
            except Exception as e:
                logging.error(f"processing_node: df_historyをDataFrameに変換中にエラーが発生しました: {e}")
                df_explain_list.append(f"■「{question}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n")

    if not df_locals:
        error_message = "processing_node: 利用可能なDataFrameが0でした。"
        logging.warning(error_message)
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    full_explain_string = "\n".join(df_explain_list)
    locals_for_tool = {**df_locals, "px": px, "pd": pd} # `locals`は組み込み関数なので変数名を変更
    python_tool = PythonAstREPLTool(
        name="python_ast_repl",
        locals=locals_for_tool,
        description=(
            f"""Pythonコードを実行してデータの加工とグラフ化ができます。対象のDataFrameとして、{",".join(df_locals.keys())}が使用可能です。
            グラフを作成するときにはplotly.express (px) を使用してインタラクティブなグラフを生成し、最終的なグラフオブジェクトは必ず `fig` という変数に格納してください。
            データフレームを成果物とする場合は、データフレームは `final_df` という変数に格納してください。"""
        )
    )
    tools_for_agent = [python_tool] # `tools`も上書きしないように変数名を変更

    system_prompt = """
    あなたはデータ分析のエキスパートです。ユーザーの指示に従い、ツールを使って分析し、結果を返してください。

    - 渡された`df`というDataFrameを操作・分析してください。
    - 可視化を行う場合、最適なインタラクティブグラフを`plotly.express` (例: `px`) を使用して生成してください。最終的なグラフオブジェクトは必ず `fig` という変数に格納してください。
    - データフレームを最終成果物とする場合、そのデータフレームは `final_df` という変数に格納してください。

    思考の過程:
    {agent_scratchpad}
    """
    conversation_history = state.get("messages", [])
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

    user_prompt = f"""
    あなたはPythonプログラミングとデータ可視化の専門家です。
    現在のタスクと文脈を踏まえて、最適なデータを選択して、最適なPythonコードを書いてください。

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【利用可能なデータの内容】
    {full_explain_string}

     - 可視化を行う場合、最適なインタラクティブグラフを`plotly.express` (例: `px`) を使用して生成してください。最終的なグラフオブジェクトは必ず `fig` という変数に格納してください。
    実行例:
    import plotly.express as px
    fig = px.line(df, x='your_x_column', y='your_y_column')

    - 最終成果物のデータフレームは `final_df` という変数に格納してください。
    実行例:
    import pandas as pd
    final_df = df.groupby('date')['sales'].sum().reset_index()
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    logging.info(f"processing_node: データ加工開始・・・ '{task_description}'")
    llm_with_tool = llm.bind_tools(tools_for_agent) # 修正: tools_for_agentを使用
    agent = create_openai_tools_agent(llm_with_tool, tools_for_agent, prompt) # 修正: tools_for_agentを使用
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent, # 修正: tools_for_agentを使用
        verbose=True,
        handle_parsing_errors=True
    )

    code_collector = CodeCollectorCallback()
    logging.info(f"processing_node: 生成AIが考えています・・・ '{task_description}'")
    agent_response = agent_executor.invoke(
        {"input": user_prompt},
        {"callbacks": [code_collector]}
    )
    logging.info(f"processing_node: Agent response: {agent_response}")
    logging.info(f"processing_node: Agent codes: {code_collector.codes}") # "Agent response"から"Agent codes"に修正

    fig_json = None
    result_df_json = None
    new_history_item = None # 初期化

    try:
        logging.info(f"processing_node: コード検証中・・・ '{task_description}'")
        execution_scope = locals_for_tool.copy() # PythonAstREPLToolのlocalsをコピーして使用
        for code_block_str in code_collector.codes:
            # LangChainのPythonAstREPLToolは、入力が直接コード文字列であることを期待している場合がある
            # もしcode_block_strが {'query': '...'} のような形式ならパースが必要
            code_to_execute = code_block_str
            if isinstance(code_block_str, str):
                try:
                    # 文字列が辞書形式（例: "{'query': 'print(1)'}"）であるか試す
                    code_dict = ast.literal_eval(code_block_str)
                    if isinstance(code_dict, dict) and "query" in code_dict:
                        code_to_execute = code_dict["query"]
                except (SyntaxError, ValueError):
                    # ast.literal_evalが失敗した場合は、そのままコード文字列として扱う
                    pass
            
            exec(code_to_execute, execution_scope)

        fig = execution_scope.get('fig')
        result_df = execution_scope.get('final_df')

        if fig is not None:
            fig_json = fig.to_json()
        if result_df is not None:
            result_df_json = result_df.to_json(orient="records", force_ascii=False)
            if not result_df.empty: # 空のDataFrameは履歴に追加しないようにする
                new_history_item = {task_description: result_df.to_dict(orient="records")}


        if fig_json is None and result_df_json is None:
            content = "データも、グラフも生成されませんでした"
        else:
            content = json.dumps({
                "result_df_json": result_df_json,
                "fig_json": fig_json
            }, ensure_ascii=False)

    except Exception as e:
        error_message = f"生成AIがデータ加工・グラフ作成に失敗しました。: {e}. 実行しようとしたコード: {code_collector.codes}"
        logging.error(f"processing_node: {error_message}")
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    updated_df_history = state.get("df_history", [])
    if new_history_item: # new_history_itemがNoneでない（つまり有効なDataFrameが生成された）場合のみ追加
        updated_df_history = updated_df_history + [new_history_item]
    
    logging.info(f"processing_node:実行完了")
    return {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id
            )
        ],
        "df_history": updated_df_history,
    }

# --- 2. スーパーバイザーの出力スキーマ定義 (Pydantic) ---
class DispatchDecision(BaseModel):
    """専門家エージェントにタスクを割り振る際の意思決定スキーマ"""
    next_agent: str = Field(description="次に指名すべき専門家（ノード）の名前。利用可能な専門家がいない場合は 'FINISH'。")
    task_description: str = Field(description="指名した専門家に与える、具体的で明確な指示内容。")
    rationale: str = Field(description="なぜその判断を下したのかの簡潔な理由。")

members = ["sql_node", "processing_node", "interpret_node"] # interpret_node を専門家リストに追加
supervisor_tools = [metadata_retrieval_node, analysis_plan_node] # supervisorが直接使うツール名変更

system_prompt_supervisor = (
    "あなたはAIチームを率いる熟練のプロジェクトマネージャーです。\n"
    "あなたの仕事は、ユーザーの要求に基づき、以下のいずれかのアクションを選択することです。\n"
    "1. 自分でツールを実行する: 簡単な情報照会などは、あなたが直接ツールを実行して回答してください。\n"
    "2. 専門家に仕事を割り振る: 複雑な作業は、タスクを分解し、以下の専門家メンバーに具体的で明確な指示を出してください。\n\n"
    "== 利用可能なツール ==\n"
    "- metadata_retrieval_node: データベースのテーブル定義やカラムについて答える。\n" #ツール名を修正
    "- analysis_plan_node: 分析のプランを立てる\n\n"

    "== 専門家メンバーリスト ==\n"
    "- sql_node: 依頼に従ってSQLを生成し、データを取得する\n"
    "- processing_node: 依頼に従ってデータの加工やグラフ作成をする\n"
    "- interpret_node: 依頼に従ってデータの解釈を行う\n\n" # interpret_node を専門家リストに追加
    "== 判断ルール ==\n"
    "まずツールで解決できるか検討し、できなければ専門家への割り振りを考えてください。\n"
    "専門家へ割り振る際は、次に指名すべきメンバーと、そのメンバーへの具体的な指示内容を決定してください。\n"
    "全てのタスクが完了したと判断した場合は、next_agentに 'FINISH' を指定してください。"
)

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_supervisor),
    MessagesPlaceholder(variable_name="messages"),
])
llm_with_supervisor_tools = llm.bind_tools(supervisor_tools + [DispatchDecision])
supervisor_chain = supervisor_prompt | llm_with_supervisor_tools

def supervisor_node(state: AgentState):
    """planning_nodeで作成したプランに従って次のノードを決定する"""
    logging.info("--- スーパーバイザー ノード実行中 ---")

    plan = state.get("plan")
    step = state.get("plan_step", 0)

    if not plan:
        logging.info("supervisor_node: プランが無いので planning_node を実行します")
        return {"messages": [], "next": "planning_node"}

    if step >= len(plan):
        logging.info("supervisor_node: 全てのステップが完了しました")
        finish_message = AIMessage(content="分析が完了しました")
        return {"messages": [finish_message], "next": "FINISH"}

    current = plan[step]
    next_agent = current.get("agent")
    task_description = current.get("task", "")
    tool_call_id = f"plan_step_{step}"
    ai_message = AIMessage(
        content="", 
        tool_calls=[{"name": next_agent, "args": {"task_description": task_description}, "id": tool_call_id}]
    )

    logging.info(f"supervisor_node: 次に {next_agent} を実行します。タスク: {task_description}")

    return {
        "messages": [ai_message],
        "next": next_agent,
        "plan_step": step + 1
    }

def build_workflow():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("planning_node", planning_node)
    graph_builder.add_node("sql_node", sql_node)
    graph_builder.add_node("processing_node", processing_node)
    graph_builder.add_node("interpret_node", interpret_node) # interpret_node を追加
    tool_node = ToolNode(supervisor_tools) # 修正: supervisor_tools を使用
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("supervisor")

    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "planning_node": "planning_node",
            "sql_node": "sql_node",
            "processing_node": "processing_node",
            "interpret_node": "interpret_node", # interpret_node へのエッジを追加
            "tools": "tools",
            "FINISH": END
        }
    )

    graph_builder.add_edge("planning_node", "supervisor")
    graph_builder.add_edge("sql_node", "supervisor")
    graph_builder.add_edge("processing_node", "supervisor")
    graph_builder.add_edge("interpret_node", "supervisor") # interpret_node から supervisor へのエッジを追加
    graph_builder.add_edge("tools", "supervisor")
    graph_builder.add_edge("supervisor", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)
