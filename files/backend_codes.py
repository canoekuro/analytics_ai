# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
import pandas as pd
import logging
from langgraph.graph import StateGraph, END
from langchain.agents import initialize_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import io
from typing import TypedDict, List, Optional, Dict, Any
import ast # literal_evalのため
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
import plotly.express as px
import json # import jsonを先頭に移動しました
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from functions import extract_sql, try_sql_execute, fix_sql_with_llm
import os

# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
version4emb = os.getenv("AZURE_OPENAI_API_VERSION4EMB")
deployment4emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4EMB")
version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# Chatモデル（SQL生成用）
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

class MyState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add] # 基本的な会話履歴
    task_description_history: Optional[List[str]]             # 各エージェントへの指示
    metadata_answer: Optional[List[dict[str, str]]]    # メタデータ検索結果の回答
    df_history: Optional[List[dict[str, Any]]]   # SQL実行結果のDataFrameの履歴
    sql_history: Optional[List[dict[str, Any]]]   # SQLの履歴
    interpretation_history: Optional[List[dict[str, str]]]     # データ解釈（分析コメント）
    chart_history: Optional[List[dict[str, str]]]      # Plotlyによって生成されたグラフのJSON文字列
    analyze_step: Optional[List[dict[str, str]]] 
    error: Optional[str]


history_back_number = 5

@tool
def metadata_retrieval_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、文脈を理解した上でテーブル定義を示します。
    ユーザーからデータやテーブル、カラムについて質問があった時に使用します。
    """
    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])

    # Rag情報
    retrieved_docs = vectorstore_tables.similarity_search(task_description)
    retrieved_table_info = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_template = """
    以下のテーブル定義情報を参照して、ユーザーの質問に答えてください。
    ユーザーが理解しやすいように、テーブルやカラムの役割、データ型、関連性などを説明してください。

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
    state.setdefault("task_description_history", []).append(task_description)
    state.setdefault("metadata_answer", []).append({task_description: metadata_answer})
    return state

@tool
def analyze_step_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、文脈を理解した上で必要な分析ステップを考えます。
    ユーザーから分析依頼があり、分析要件を具体化したい際に使います。
    """
    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])

    # Rag情報
    logging.info(f"analyze_step_node: RAG情報を読み込み中 '{task_description}'")
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
    logging.info(f"analyze_step_node: 生成AIが考えています・・・ '{task_description}'")
    llm_prompt = prompt_template.format(retrieved_table_info=retrieved_table_info, task_description=task_description, context=context)
    response = llm.invoke(llm_prompt)
    step_answer = response.content.strip()
    state.setdefault("task_description_history", []).append(task_description)
    state.setdefault("analyze_step", []).append({task_description:step_answer})
    return state

@tool
def sql_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、、文脈を理解した上でSQLを生成・実行します。
    分析のためにデータを取得する必要がある際に使用します。
    """
    #システムプロンプト
    system_prompt_sql_generation = """
    あなたはSQL生成AIです。この後の質問に対し、SQLiteの標準的なSQL文のみを出力してください。
    - 使用できるSQL構文は「SELECT」「WHERE」「GROUP BY」「ORDER BY」「LIMIT」のみです。
    - 日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    - 必ず1つのテーブルだけを使い、簡単な集計・フィルタ・並べ替えまでにしてください。
    - SQLの前後にコメントや説明文は出力しないでください。出力された内容をそのまま実行するため、SQL文のみをそのまま出力してください。
    """
    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])
    
    # Rag情報
    logging.info(f"sql_node: 要件を処理中: '{task_description}'")
    retrieved_tables_docs = vectorstore_tables.similarity_search(task_description, k=3)
    rag_tables = "\n".join([doc.page_content for doc in retrieved_tables_docs])
    retrieved_queries_docs = vectorstore_queries.similarity_search(task_description, k=3)
    rag_queries = "\n".join([doc.page_content for doc in retrieved_queries_docs])

    #ユーザープロンプト
    user_prompt_for_req = f"""
    【現在のタスク】
    {task_description}
    
    【ユーザーの全体的な質問の文脈】
    {context}
    
    【テーブル定義に関する情報】
    {rag_tables}
    
    【類似する問い合わせ例とそのSQL】
    {rag_queries}
    
    この要件を満たすためのSQLを生成してください。
    """

    logging.info(f"sql_node: 生成AIが考えています・・・ '{task_description}'")
    response = llm.invoke([
        {"role": "system", "content": system_prompt_sql_generation},
        {"role": "user", "content": user_prompt_for_req}
    ])
    sql_generated_clean = extract_sql(response.content.strip())
    last_sql_generated = sql_generated_clean # Store the latest SQL attempt for this requirement
    result_df, sql_error = try_sql_execute(sql_generated_clean)

    if sql_error:
        logging.warning(f"'最初のSQLが失敗しました: {sql_error}。修正して再試行します。")
        fixed_sql = fix_sql_with_llm(llm, sql_generated_clean, sql_error, rag_tables, rag_queries, task_description, context)
        last_sql_generated = fixed_sql # この要件に対する最新のSQL試行を保存
        result_df, sql_error = try_sql_execute(fixed_sql)
        if sql_error:
            logging.error(f"'修正SQLも失敗しました: {sql_error}。")
            raise RuntimeError(f"SQLの実行に失敗しました (要件: '{task_description}'): {sql_error}")

    state.setdefault("task_description_history", []).append(task_description)
    result_df_dict = result_df.to_dict(orient="records") 
    result_dict = {task_description: result_df_dict}
    state.setdefault("df_history", []).append(result_dict)
    sql_dict = {task_description: last_sql_generated}
    state.setdefault("sql_history", []).append(sql_dict)

    return state

@tool
def interpret_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータを解釈します。
    分析のためにデータを解釈する必要がある際に使用します。
    """

    logging.info(f"interpret_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", None)
    if df_history is None:
        raise RuntimeError("interpret_node: df_historyが空です。利用可能なデータがありません。")
    full_data_list = []
    for entry in df_history:
        for question, data in entry.items():
            try:
                # DataFrame化
                df = pd.DataFrame(data)
                if not df.empty:
                    full_data_list.append(f"■「{question}」に関するデータ:\n{df.to_string(index=False)}\n\n")
                else:
                    full_data_list.append(f"■「{question}」に関するデータ:\n(この要件に対するデータは空でした)\n\n")
        
            except Exception as e:
                logging.error(f"interpret_node: 'df_historyをDataFrameに変換中にエラーが発生しました: {e}")
                full_data_list.append(f"■「{question}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n")
    full_data_string = "".join(full_data_list)

    # 全て空・エラーだった場合に備えたチェック
    processed_parts = [part for part in full_data_string.split("■")[1:] if part]
    all_parts_indicate_no_data = all(
        "(この要件に対するデータはありませんでした)" in part or \
        "(データ形式エラーのため表示できません)" in part or \
        "(この要件に対するデータは空でした)" in part \
        for part in processed_parts
    )
    if all_parts_indicate_no_data:
        raise RuntimeError("interpret_node: 利用可能なデータがありませんでした。")
    
    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])


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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        interpretation_text = response.content.strip() if response.content else ""

        if not interpretation_text:
            raise RuntimeError("interpret_node: LLMが空の解釈を返しました。")

    except Exception as e:
        raise

    state.setdefault("task_description_history", []).append(task_description)
    interpretation_dict = {task_description: interpretation_text}
    state.setdefault("interpretation_history", []).append(interpretation_dict)
    return state

@tool
def chart_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でグラフを作成します。
    分析のためにグラフを作成する必要がある際に使用します。
    """
    
    logging.info(f"chart_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", None)
    if df_history is None:
        raise RuntimeError("chart_node: df_historyが空です。利用可能なデータがありません。")


    # 直近N件だけ取り出し
    recent_items = df_history[-history_back_number:]
    df_explain_list = []
    df_locals = {}
    for idx, entry in enumerate(recent_items):
        for question, data in entry.items():
            try:
                df = pd.DataFrame(data)
                df_name = f"df{idx}"  # 例: df0, df1, ...
                if not df.empty:
                    df_locals[df_name] = df  # python_tool用
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
                logging.error(f"chart_node: df_historyをDataFrameに変換中にエラーが発生しました: {e}")
                df_explain_list.append(f"■「{question}」に関するデータ:\n(データ形式エラーのため表示できません)\n\n")

    # 有効なDataFrameが1つもなければ終了
    if not df_locals:
        raise RuntimeError("chart_node: 利用可能なデータが0でした。")
    
    full_explain_string = "\n".join(df_explain_list)

    python_tool = PythonAstREPLTool(
        locals={**df_locals, "px": px, "pd": pd}, 
        description=(
            f"""Pythonコードを実行してデータを分析・グラフ化できます。プロット対象のDataFrameとして、{",".join(df_locals.keys())}が使用可能です。
            plotly.express (px) を使用してインタラクティブなグラフを生成し、fig.to_json() でJSON文字列として出力してください。"""
        )
    )
    tools = [python_tool]
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True,
        agent_kwargs={"handle_parsing_errors": True}
    )

    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])

    user_prompt = f"""
    あなたはPythonプログラミングとデータ可視化の専門家です。
    現在のタスクと文脈を踏まえて、最適なデータを選択し、最適なグラフを作成してください。

    【現在のタスク】
    {task_description}
    
    【ユーザーの全体的な質問の文脈】
    {context}

    【利用可能なデータの内容】
    {full_explain_string}

    最適なインタラクティブグラフを `plotly.express` (例: `px`) を使用して生成してください。
    生成したFigureオブジェクトを `fig` という変数に格納し、その後 `fig.to_json()` を呼び出してJSON文字列に変換し、そのJSON文字列を `print` してください。


    実行例:
    import plotly.express as px
    fig = px.line(df, x='your_x_column', y='your_y_column')
    print(fig.to_json())
    """
    try:
        logging.info(f"chart_node: 生成AIが考えています・・・ '{task_description}'")
        agent_response = agent.invoke(
            {"input":user_prompt}
            )
        logging.info(f"chart_node: Agent response: {agent_response}")

        # The agent's output should be the Plotly JSON string
        plotly_json_string = agent_response['output']
        try:
            json.loads(plotly_json_string)
            state.setdefault("task_description_history", []).append(task_description)
            chart_dict = {task_description: plotly_json_string}
            state.setdefault("chart_history", []).append(chart_dict)
            return state
        
        except (json.JSONDecodeError, TypeError) as e:
            raise RuntimeError(f"生成されたグラフのJSONが無効です: {e}")
    except Exception as e:
        raise

@tool
def processing_node(task_description: str, state: MyState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータを加工します。
    分析のためにデータを加工する必要がある際に使用します。
    """
    
    df_history = state.get("df_history", None)
    if df_history is None:
        raise RuntimeError("processing_node: df_historyが空です。利用可能なデータがありません。")

    # 直近N件だけ取り出し
    recent_items = df_history[-history_back_number:]
    df_explain_list = []
    df_locals = {}
    for idx, entry in enumerate(recent_items):
        for question, data in entry.items():
            try:
                df = pd.DataFrame(data)
                df_name = f"df{idx}"  # 例: df0, df1, ...
                if not df.empty:
                    df_locals[df_name] = df  # python_tool用
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

    # 有効なDataFrameが1つもなければ終了
    if not df_locals:
        raise RuntimeError("processing_node: 利用可能なデータが0でした。")
    
    full_explain_string = "\n".join(df_explain_list)
    python_tool = PythonAstREPLTool(
        locals={**df_locals, "pd": pd}, 
        description=(
            f"""Pythonコードを実行してデータを加工できます。DataFrameとして、{",".join(df_locals.keys())}が使用可能です。"""
        )
    )
    tools = [python_tool]
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True,
        agent_kwargs={"handle_parsing_errors": True}
    )

    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.role != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_history])

    user_prompt = f"""
    あなたはPythonプログラミングの専門家です。
    現在のタスクと文脈を踏まえて、最適なデータを選択し、求められたデータ加工を行ってください。
    データ加工後の結果は必ず DataFrameとし、'result'という名前の変数に代入して返却してください。

    【現在のタスク】
    {task_description}
    
    【ユーザーの全体的な質問の文脈】
    {context}

    【利用可能なデータの内容】
    {full_explain_string}

    実行すべきPythonコードのみを考えてください。
    最終的な結果は'result'という名称のDataFrameに格納する必要があります。
    これは絶対に守ってください。
    """
    try:
        logging.info(f"dataprocess_node: 生成AIが考えています・・・ '{task_description}'")
        agent_response = agent.invoke(
            {"input":user_prompt}
            )
        logging.info(f"processed_node: Agent response: {agent_response}")

        state.setdefault("task_description_history", []).append(task_description)
        processed_df = python_tool.globals.get("result")

        if not isinstance(processed_df, pd.DataFrame):
        # エラー処理または警告をログに出力
            logging.error("Processing node did not return a DataFrame in 'result'.")
            raise RuntimeError("Data processing agentは有効なデータフレームを返しませんでした。")
        state.setdefault("df_history", []).append({task_description: processed_df.to_dict(orient="records") if isinstance(processed_df, pd.DataFrame) else [] })

        return state

    except Exception as e:
        raise

tools = [metadata_retrieval_node, sql_node, interpret_node, chart_node, processing_node, analyze_step_node]
def supervisor_node(state: MyState):
    try:
        supervisor_llm = llm.bind_tools(tools)
        logging.info("supervisor: Thinking...")
        response = supervisor_llm.invoke(state['messages'])
        return {"messages": [response]}
    except Exception as e:
        state["error"] = f"処理中にエラーが発生しました: {e}"
        return state
    

def build_workflow():
    graph_builder = StateGraph(MyState)

    # 1. スーパーバイザーノードを追加
    graph_builder.add_node("supervisor", supervisor_node)

    # 2. ツール実行ノードを追加
    # ToolNodeは、スーパーバイザーが呼び出しを決めたツールを自動で実行してくれる便利なノード
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # 3. グラフのエッジ（処理の流れ）を定義
    # まずスーパーバイザーから開始
    graph_builder.set_entry_point("supervisor")

    # スーパーバイザーの判断に応じて処理を分岐
    # - ツールを呼び出すべきか？ (tools_conditionがTrue)
    # - それともユーザーに回答して終了すべきか？ (tools_conditionがFalse)
    graph_builder.add_conditional_edges(
        "supervisor",
        tools_condition, # LLMの出力にtool_callsが含まれているか判定する組み込み関数
        {
            "tools": "tools",  # Trueならツール実行ノードへ
            END: END           # Falseなら終了
        }
    )

    # ツールを実行し終わったら、その結果を持って再びスーパーバイザーに戻り、次の指示を仰ぐ
    graph_builder.add_edge("tools", "supervisor")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)
