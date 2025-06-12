# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

import os
import pandas as pd
import logging
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import plotly.express as px
import json
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from files.functions import extract_sql, try_sql_execute, get_table_name_from_formatted_doc, fetch_tool_args, plan_list_conv
from files.classes import PythonExecTool, DispatchDecision

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

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str
    df_history: List[dict]
    fig_history: List
    plan: Optional[List[Dict[str, str]]] = None  # planning_nodeが生成したプラン
    plan_step: Optional[int] = None  # 現在のステップ番号
    retry_counts: Dict[str, int]


history_back_number = 5
RERANK_CANDIDATE_COUNT = 5
RETRY_LIMITS: dict[str, int] = {
    "planning_node": 1,
    "sql_node": 2,
    "processing_node": 2,
    "metadata_retrieval_node": 3,
    "interpret_node": 1,
    # 必要に応じて追加
}


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
    result = json.dumps({
        "status": "success",
        "node": "metadata_retrieval_node",
        "summary": metadata_answer,
        "result_payload": None
    })

    return result

def planning_node(state: AgentState):
    """
    自然言語のタスク記述と直近の会話履歴を受け取り、文脈を理解した上で必要な分析ステップをJSON形式で出力します。
    ユーザーから分析依頼があり、分析要件を具体化したい際に使います。
    """
    logging.info(f"planning_node: 開始")
    args, error = fetch_tool_args(state, ["task_description"])
    if args is None:
        error_message = error["error_message"]
        tool_call_id = error["tool_call_id"]
        logging.info(f"planning_node: {error_message}")
        result = json.dumps({
            "status": "error",
            "node": "planning_node",
            "summary": f"AIMessegeにエラーがあります。: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    task_description = args["task_description"]
    context = args["_history_context"]
    tool_call_id = args["_tool_call_id"]

    # Rag情報
    logging.info(f"planning_node: RAG情報を読み込み中 '{task_description}'")
    retrieved_docs = vectorstore_tables.similarity_search(task_description)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])
    

    prompt_template = """
    以下の情報を参照して、ユーザーの質問に答えるための分析計画を立ててください。
    回答は、【回答例】にあるようなフォーマットの形式で出力してください。

    【テーブル定義情報】
    {retrieved_table_info}

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【回答例】
    {
      "plan": [
        {"agent": "sql_node", "task": "必要なデータをSQLで取得する"},
        {"agent": "processing_node", "task": "取得したデータを加工・可視化する"},
        {"agent": "interpret_node", "task": "結果を解釈して結論をまとめる"}
      ]
    }
    """

    logging.info(f"planning_node: 生成AIが考えています・・・ '{task_description}'")
    llm_prompt = prompt_template.format(retrieved_table_info=retrieved_table_info, task_description=task_description, context=context)
    response = llm.invoke(llm_prompt)
    step_answer = response.content.strip()
    logging.info(f"planning_node: 回答完了{step_answer}")

    try:
        #成功した場合
        parsed = json.loads(step_answer)
        plan_list = parsed.get("plan", [])
        result = json.dumps({
            "status": "success",
            "node": "planning_node",
            "summary": "プランの生成に成功しました",
            "result_payload": plan_list
        })
        return {
            "messages": [ToolMessage(content=result, tool_call_id=tool_call_id)],
            "plan": plan_list,
            "plan_cursor": 0
            }
        
    except Exception as e:
        result = json.dumps({
            "status": "error",
            "node": "planning_node",
            "summary": "プラン生成中にエラーが発生しました: {e}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}


def sql_node(state: AgentState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、ルールを強く意識した上でSQLを生成・実行します。
    （二段階選抜とルール遵守CoTを実装）
    """
    args, error = fetch_tool_args(state, ["task_description"])
    if args is None:
        error_message = error["error_message"]
        tool_call_id = error["tool_call_id"]
        logging.info(f"sql_node: {error_message}")
        result = json.dumps({
            "status": "error",
            "node": "sql_node",
            "summary": f"AIMessegeにエラーがあります。: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    task_description = args["task_description"]
    context = args["_history_context"]
    tool_call_id = args["_tool_call_id"]

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
        logging.error("sql_node: 関連するテーブル情報が見つかりませんでした。")
        result = json.dumps({
            "status": "error",
            "node": "sql_node",
            "summary": "関連するテーブル情報が見つかりませんでした。",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

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
    それ以外の関数、例えば日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    必ず1つのテーブルだけを使い、フィルタ・並べ替えまでにしてください。
    （※「ローデータ」と示された集計が許可されているテーブルは簡単な集計も可）
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

        
        if result_df is not None and not result_df.empty:
            logging.info(f"sql_node: データの取得が完了しました。'")
            new_history_item = {task_description: result_df.to_dict(orient="records")}
            updated_df_history = state.get("df_history", [])
            updated_df_history = updated_df_history + [new_history_item]
            result = json.dumps({
                "status": "success",
                "node": "sql_node",
                "summary": f"{result_df}のデータを取得しました",
                "result_payload": {"result_df_json":result_df.to_json(orient="records"), "executed_sql": last_sql_generated}
            })
            return {
                "messages": [ToolMessage(content=result, tool_call_id=tool_call_id)], 
                "df_history": updated_df_history
                }

    except Exception as e:
        logging.error(f"sql_nodeで予期せぬエラー: {e}")
        result = json.dumps({
            "status": "error",
            "node": "sql_node",
            "summary": f"sql_nodeで予期せぬエラー: {e}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

# 解釈ノード
def interpret_node(state: AgentState):
    """
    自然言語のタスク記述と直近の会話、そしてデータを受け取り、文脈を理解した上でデータを解釈します。
    分析のためにデータを解釈する必要がある際に使用します。
    """

    args, error = fetch_tool_args(state, ["task_description"])
    if args is None:
        error_message = error["error_message"]
        tool_call_id = error["tool_call_id"]
        logging.info(f"interpret_node: {error_message}")
        result = json.dumps({
            "status": "error",
            "node": "interpret_node",
            "summary": f"AIMessegeにエラーがあります。: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}


    task_description = args["task_description"]
    context = args["_history_context"]
    tool_call_id = args["_tool_call_id"]

    logging.info(f"interpret_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", []) # getのデフォルトを空リストに
    if not df_history: # df_historyがNoneまたは空の場合
        # RuntimeErrorを発生させる代わりに、ToolMessageでエラーを返す
        error_message = "interpret_node: df_historyが空です。利用可能なデータがありません。"
        logging.warning(error_message) # Warningレベルに変更
        result = json.dumps({
            "status": "error",
            "node": "interpret_node",
            "summary": "プラン生成中にエラーが発生しました: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

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
        result = json.dumps({
            "status": "error",
            "node": "interpret_node",
            "summary": "プラン生成中にエラーが発生しました: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}


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
            result = json.dumps({
                "status": "success",
                "node": "interpret_node",
                "summary": interpretation_text,
                "result_payload": {"interpretation_text":interpretation_text}
            })
            return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}
        else:
            # RuntimeErrorを発生させる代わりに、ToolMessageでエラーを返す
            error_message = "interpret_node: LLMが空の解釈を返しました。"
            logging.warning(error_message) # Warningレベルに変更
            result = json.dumps({
                    "status": "error",
                    "node": "interpret_node",
                    "summary": error_message,
                    "result_payload": None
                })
            return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    except Exception as e:
            error_message = "interpret_node: LLMが空の解釈を返しました。"
            logging.warning(error_message) # Warningレベルに変更
            result = json.dumps({
                    "status": "error",
                    "node": "interpret_node",
                    "summary": error_message,
                    "result_payload": None
                })
            return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}


def processing_node(state: AgentState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータの加工やグラフの作成を行います。
    AgentExecutorを使わずに、LLMによるコード生成と実行を直接制御することでループを防ぎます。
    """
    args, error = fetch_tool_args(state, ["task_description"])
    if args is None:
        error_message = error["error_message"]
        tool_call_id = error["tool_call_id"]
        logging.info(f"processing_node: {error_message}")
        result = json.dumps({
            "status": "error",
            "node": "processing_node",
            "summary": f"AIMessegeにエラーがあります。: {error_message}",
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}
        
    task_description = args["task_description"]
    context = args["_history_context"]
    tool_call_id = args["_tool_call_id"]

    logging.info(f"processing_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", [])
    if not df_history:
        error_message = "processing_node: df_historyが空です。利用可能なデータがありません。"
        logging.warning(error_message)
        result = json.dumps({
            "status": "error",
            "node": "processing_node",
            "summary": error_message,
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    recent_items = df_history[-history_back_number:]
    df_explain_list = []
    df_locals = {}
    for idx, entry in enumerate(recent_items):
        for question, data in entry.items():
            try:
                # 履歴のデータが様々な形式（リスト、辞書）で保存されている可能性を考慮
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame.from_dict(data)
                else:
                    logging.warning(f"processing_node: サポートされていないデータ形式 ({type(data)}) のためスキップします。")
                    continue
                
                df_name = f"df{idx}"
                if not df.empty:
                    df_locals[df_name] = df
                    explain = f"""
                    # DataFrame `{df_name}`
                    ## 内容: 「{question}」に関するデータ
                    ## カラム情報: {list(df.columns)}
                    ## データサンプル (最初の5行):
                    {df.head().to_string(index=False)}
                    """
                    df_explain_list.append(explain)
                else:
                    df_explain_list.append(f"# DataFrame `{df_name}` は「{question}」に関するデータですが、空でした。")
            except Exception as e:
                logging.error(f"processing_node: df_historyをDataFrameに変換中にエラーが発生しました: {e}")
                df_explain_list.append(f"# 「{question}」に関するデータの処理中にエラーが発生しました: {e}")

    if not df_locals:
        error_message = "processing_node: 利用可能なDataFrameがありませんでした。"
        logging.warning(error_message)
        result = json.dumps({
            "status": "error",
            "node": "processing_node",
            "summary": error_message,
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    full_explain_string = "\n".join(df_explain_list)

    # AgentExecutorを廃止し、LLMに直接コードを生成させるためのプロンプト
    code_generation_prompt = f"""
    あなたはPythonプログラミングとデータ可視化の専門家です。
    現在のタスクと文脈、そして提供されたデータ情報を踏まえて、以下のルールに従ってPythonコードを生成してください。
    生成するコードは```python タグで囲んでください。それ以外の説明やコメントは一切不要です。

    【現在のタスク】
    {task_description}

    【ユーザーの全体的な質問の文脈】
    {context}

    【利用可能なデータとその説明】
    {full_explain_string}

    【コード生成ルール】
    - 利用可能なDataFrame ({", ".join(df_locals.keys())}) のみを使用してください。
    - データの加工や集計には `pandas` (別名 `pd`) を使用できます。
    - 可視化を行う場合、`plotly.express` (別名 `px`) を使用してインタラクティブなグラフを生成してください。
    - 生成したグラフオブジェクトは、必ず `fig` という名前の変数に格納してください。
    - 最終的な成果物としてデータフレームを返す場合は、そのデータフレームを `final_df` という名前の変数に格納してください。
    - `fig` と `final_df` の両方を同時に生成・格納することも可能です。
    - コードは1つのブロックにまとめてください。許可されていないライブラリ（os, subprocess等）のimportは含めないでください。

    【生成するPythonコード】
    """

    logging.info(f"processing_node: 生成AIにコード生成を依頼中・・・ '{task_description}'")
    
    # LLMを直接呼び出してコードを生成
    response = llm.invoke(code_generation_prompt)
    generated_code_raw = response.content.strip()

    # 生成された応答からPythonコード部分のみを抽出
    if "```python" in generated_code_raw:
        generated_code = generated_code_raw.split("```python")[1].split("```")[0].strip()
    else:
        generated_code = generated_code_raw # タグがない場合はそのままコードとみなす

    if not generated_code:
        error_message = "processing_node: LLMがコードを生成しませんでした。"
        logging.error(error_message)
        result = json.dumps({
            "status": "error",
            "node": "processing_node",
            "summary": error_message,
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

    logging.info(f"processing_node: 生成されたコード:\n---\n{generated_code}\n---")

    # Python実行ツールの準備
    locals_for_tool = {**df_locals, "px": px, "pd": pd}
    python_tool = PythonExecTool(locals=locals_for_tool)

    try:
        # ツールを直接実行
        tool_output_str = python_tool._run(code=generated_code)
        content_payload = json.loads(tool_output_str) # ツール出力はJSON文字列なのでパース
        
        final_df_json = content_payload.get("final_df_json")
        fig_json = content_payload.get("fig_json")

        logging.info(f"processing_node: コード実行成功。fig_json: {'あり' if fig_json else 'なし'}, final_df_json: {'あり' if final_df_json else 'なし'}")

        # stateの更新
        updated_df_history = state.get("df_history", [])
        if final_df_json:
            # 新しいdfを履歴に追加。JSON文字列をPythonのリスト（of dicts）に戻す。
            new_df_records = json.loads(final_df_json)
            # 新しいタスク名（加工後）と共に履歴に追加
            new_task_description = f"{task_description} (加工・分析後のデータ)"
            updated_df_history = updated_df_history + [{new_task_description: new_df_records}]

        updated_fig_history = state.get("fig_history", [])
        if fig_json:
            updated_fig_history = updated_fig_history + [fig_json]

        # Supervisorに返すToolMessageのコンテンツを作成
        result_summary = []
        if fig_json:
            result_summary.append("グラフを生成しました。")
        if final_df_json:
            result_summary.append("加工されたデータフレームを生成しました。")
        
        # supervisorが判断しやすいように、実行結果の要約を返す
        result_summary = " ".join(result_summary)
        result = json.dumps({
            "status": "success",
            "node": "processing_node",
            "summary": result_summary,
            "result_payload": {"fig_json":fig_json, "result_df_json":final_df_json}
        })
        return {
            "messages": [ToolMessage(content=result, tool_call_id=tool_call_id)],
            "df_history": updated_df_history,
            "fig_history": updated_fig_history
            }

    except Exception as e:
        error_message = f"processing_node: 生成されたコードの実行中にエラーが発生しました: {e}\n\n試行したコード:\n{generated_code}"
        logging.error(error_message)
        result = json.dumps({
            "status": "error",
            "node": "processing_node",
            "summary": error_message,
            "result_payload": None
        })
        return {"messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]}

def ask_user_node(state: AgentState):
    """
    ユーザーに質問するためにワークフローを中断させるためのプレースホルダーノード。
    このノードに到達する前にグラフが中断され、ユーザーの入力を待ちます。
    ユーザーからの返信後、グラフが再開されるとこのノードが実行され、
    次のエッジ（supervisor）に処理を引き継ぎます。
    """
    logging.info("ask_user_node: ユーザーへの質問のため、処理を中断します。")
    # このノード自体は、状態を変更せず、次のノードに処理を渡すだけです。
    return

supervisor_tools = [metadata_retrieval_node] 
def supervisor_node(state: AgentState):
    """スーパーバイザーとして次にどのアクションをとるべきか判断する"""
    logging.info("--- スーパーバイザー ノード実行中 ---")

    # supervisorが直接使うツール名
    llm_with_supervisor_tools = llm.bind_tools(supervisor_tools + [DispatchDecision]) 
    messages = state['messages']
    plan = state.get("plan")
    plan_cursor = state.get("plan_cursor") 
    counts = state.get("retry_counts", {})

    # 直前のメッセージが専門家からのToolMessageである場合
    if isinstance(messages[-1], ToolMessage):
        last_tool_message = messages[-1]
        # contentをJSONとしてパース
        payload = json.loads(last_tool_message.content)
        status = payload.get("status")
        pre_node = payload.get("node")
        summary = payload.get("summary")
        
        # statusキーが"success"かどうかで成否を判断
        if isinstance(payload, dict) and  status == "success":
            logging.info("supervisor_node: タスク成功を検出 (status: success)")
            summary_parts = f"""

                == 直前のタスク結果 ==\n
                {pre_node}による直前のタスクは成功しました。結果は{summary}です。

                """
            counts[pre_node] = 0

            # plan実行中の場合
            if plan and plan_cursor is not None and len(messages) > 1:
                # タスクが成功していた場合、カーソルを更新
                plan_cursor = plan_cursor + 1
                plan_now = plan_list_conv(plan, plan_cursor)
                logging.info(f"supervisor_node: plan_cursorを {plan_cursor} に更新します。")

            else:
                # plan実行中ではない場合
                plan_now = ""            
        else:
            # statusキーがerrorの場合
            logging.warning(f"supervisor_node: タスク失敗 {summary}")
            summary_parts = f"""

                == 直前のタスク結果 ==\n
                {pre_node}による直前のタスクは失敗しました。エラー内容は{summary}です。
                
            """
            failed_agent = pre_node
            counts[failed_agent] = counts.get(failed_agent, 0) + 1
            limit = RETRY_LIMITS.get(failed_agent, 0)
            if counts[failed_agent] > limit:
                # エラー回数が上限オーバー時はAsk Userで止める。
                logging.warning(f"{failed_agent} が最大試行回数を超過しました。")
                ask_msg = AIMessage(content=(f"{failed_agent} でエラーが続いたため処理が止まりました。次にどうするか教えてください。"))
                return {"messages": [ask_msg], "next": "ask_user_node"}
            
            # plan実行中の場合
            if plan and plan_cursor is not None and len(messages) > 1:
                # タスクが失敗していた場合はカーソルを更新しない
                plan_cursor = plan_cursor
                plan_now = plan_list_conv(plan, plan_cursor)
                logging.info(f"supervisor_node: plan_cursorは {plan_cursor} のままです。")
            else:
                # plan実行中ではない場合
                plan_now = ""            
    else:
        #直前のメッセージがtoolではない場合
        summary_parts = ""
        # plan実行中の場合
        if plan and plan_cursor is not None and len(messages) > 1:
                plan_now = plan_list_conv(plan, plan_cursor)
                logging.info(f"supervisor_node: plan_cursorは {plan_cursor} のままです。")
        else:
            plan_now = ""

    # 上記の情報をすべて盛り込んだシステムプロンプトを生成します
    prompt = f"""
            あなたはAIチームを率いる熟練のプロジェクトマネージャーです。\n"
            あなたの仕事は、ユーザーの要求と分析計画、直前のタスク結果に基づき、[現在地]と示されたタスクに対して以下のいずれかのアクションを選択することです。\n\n
            1. 自分でツールを実行する: 簡単な情報照会のタスクなどは、あなたが直接ツールを実行して回答してください。\n
            2. 専門家に仕事を割り振る: 複雑な作業はタスクを分解し、以下の専門家メンバーに具体的で明確な指示を出してください。\n\n
            {summary_parts}
            {plan_now}
            == 利用可能なツール ==\n
            - metadata_retrieval_node: データベースのテーブル定義やカラムについて答える。\n
            
            == 専門家メンバーリスト ==\n
            - sql_node: 依頼に従ってSQLを生成し、データを取得する\n
            - processing_node: 依頼に従ってデータの加工やグラフ作成をする\n
            - interpret_node: 依頼に従ってデータの解釈を行う\n
            - ask_user_node: ユーザーに直接質問して、不明な点や必要な情報を確認する
            - planning_node: 複雑な分析に関して、分析プランを作成する
            - ask_user_node: 不明確な分析依頼に対して、ユーザーに追加で質問をする

            == 判断ルール ==\n
            計画の次のステップを実行するために、まずツールで解決できるか検討し、できなければ専門家への割り振りを考えてください。\n
            専門家へ割り振る際は、次に指名すべきメンバーと、そのメンバーへの具体的な指示内容を決定してください。\n
            計画の最後のステップが完了したと判断した場合は、next_agentに 'FINISH' を指定してください。\n\n
        """
 
    # 動的に生成したプロンプトでChainを組み立てます
    supervisor_prompt  = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    supervisor_chain_plan = supervisor_prompt | llm_with_supervisor_tools


    response = supervisor_chain_plan.invoke({"messages": state["messages"]})
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                logging.info(f"Supervisor messages: {call.get('args')}")

    # AIMessageの場合、tool_callsは response.tool_calls でアクセス
    # responseがAIMessageインスタンスであることを確認
    if not isinstance(response, AIMessage):
        # 通常はAIMessageが返されるはずですが、万が一異なる場合のエラーハンドリング
        logging.error("supervisor_node: responseがAIMessageではありません。")
        error_response = AIMessage(content="エラーが発生しました。SupervisorがAIMessageを受け取れませんでした。")
        return {"messages": [error_response], "next": "FINISH"}

    tool_calls_list = response.tool_calls or []
    if not tool_calls_list:
        logging.info("supervisor_node: 判断: 応答完了または直接応答")
        return {"messages": [response], "next": "FINISH", "plan":[], "plan_cursor":0, "retry_counts":{}}
    else:
        #ツールコールが存在する場合
        first_call = tool_calls_list[0] # response.tool_calls の要素は {'name': ..., 'args': ..., 'id': ...} の形式
        called_tool_name = first_call.get('name')
        called_tool_arguments = first_call.get('args')

        if not called_tool_name or not isinstance(called_tool_arguments, dict):
            logging.error(f"supervisor_node: 不正なtool_call構造 (nameまたはargsが期待通りでない): {first_call}")
            error_response = AIMessage(content="不正なtool_call構造です。 (nameまたはargsが期待通りでない)")
            return {"messages": [error_response], "next": "FINISH"}

        if "next_agent" not in called_tool_arguments:
            # 'next_agent' が引数にない場合、スーパーバイザーが直接ツールを実行する(例: metadata_retrieval_node や planning_node)
            logging.info(f"supervisor_node: 判断: ツール '{called_tool_name}' を直接実行します。")
            # この場合、response (AIMessage) にはツール実行に必要な情報(tool_call)が含まれている
            # ToolNode はこの response.tool_calls を見てツールを実行する
            return {"messages": [response], "next": "tools", "retry_counts": counts}
        else:
            # 'next_agent' が引数にある場合、専門家へのディスパッチ (DispatchDecision が呼ばれた)
            # called_tool_arguments は DispatchDecision の引数 (next_agent, task_description, rationale)
            try:
                dispatch_decision = DispatchDecision(**called_tool_arguments)
                logging.info(f"supervisor_node: 判断: 専門家 '{dispatch_decision.next_agent}' にタスクを割り振ります。")
                logging.info(f"supervisor_node: 判断理由: {dispatch_decision.rationale}")
                logging.info(f"supervisor_node: 具体的な指示: {dispatch_decision.task_description}")

                # AIMessage (response) をそのまま履歴に追加し、専門家ノードがこのAIMessageのtool_callsから指示を読み取る
                # response.tool_calls[0]['args'] の中に task_description が含まれている想定
                return {
                    "messages": [response],
                    "next": dispatch_decision.next_agent,
                    "plan":plan,
                    "plan_cursor":plan_cursor,
                    "retry_counts": counts
                }
            except Exception as e: # Pydanticのバリデーションエラーなど
                logging.error(f"supervisor_node: DispatchDecisionのパースに失敗: {e}, args: {called_tool_arguments}")
                error_response = AIMessage(content=f"Failed to parse dispatch decision: {e}")
                return {"messages": [error_response], "next": "FINISH"}

def build_workflow():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("sql_node", sql_node)
    graph_builder.add_node("processing_node", processing_node)
    graph_builder.add_node("interpret_node", interpret_node)
    graph_builder.add_node("planning_node", planning_node)
    graph_builder.add_node("ask_user_node", ask_user_node)
    tool_node = ToolNode(supervisor_tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("supervisor")

    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "sql_node": "sql_node",
            "processing_node": "processing_node",
            "interpret_node": "interpret_node", 
            "planning_node": "planning_node",
            "ask_user_node": "ask_user_node",
            "tools": "tools",
            "FINISH": END
        }
    )

    graph_builder.add_edge("sql_node", "supervisor")
    graph_builder.add_edge("processing_node", "supervisor")
    graph_builder.add_edge("interpret_node", "supervisor") 
    graph_builder.add_edge("planning_node", "supervisor")
    graph_builder.add_edge("ask_user_node", "supervisor")
    graph_builder.add_edge("tools", "supervisor")
    graph_builder.add_edge("supervisor", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory, interrupt_before=["ask_user_node"])
