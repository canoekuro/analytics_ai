# 【重要！！】コメントアウトやエラーメッセージはできる限り日本語で残すこと。

import os
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
import pandas as pd
import logging
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent
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
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from files.functions import extract_sql, try_sql_execute, fix_sql_with_llm


# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# endpoint = os.getenv("AZURE_OPENAI_API_BASE")
# version4emb = os.getenv("AZURE_OPENAI_API_VERSION4EMB")
# deployment4emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4EMB")
# version = os.getenv("AZURE_OPENAI_API_VERSION")
# deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
# # 環境変数からLLMモデル名を取得します（デフォルト値あり）
llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash") # デフォルトをgemini-1.5-proに変更
google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) 
llm = ChatGoogleGenerativeAI(
    model=llm_model_name,
    temperature=0,
    google_api_key=google_api_key
) # または、より高性能な "gemini-1.5-flash"、"gemini-1.5-pro"

# Chatモデル（SQL生成用）
# llm = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=endpoint,
#     openai_api_version=version,
#     deployment_name=deployment,
#     temperature=0,
#     streaming=False
# )

# embeddings = AzureOpenAIEmbeddings(
#     openai_api_key=api_key,
#     azure_endpoint=endpoint,
#     openai_api_version=version4emb,
#     azure_deployment=deployment4emb
# )
# ベクトルストア
vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next: str  # 次に実行すべきノード名
    df_history: List[dict] 

history_back_number = 5

@tool
def metadata_retrieval_node(task_description: str, conversation_history: list[str]):
    """
    自然言語のタスク記述と直近の会話履歴を受け取り、文脈を理解した上でテーブル定義を示します。
    ユーザーからデータやテーブル、カラムについて質問があった時に使用します。
    """

    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

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

    return metadata_answer

@tool
def analysis_plan_node(task_description: str, conversation_history: list[str]):
    """
    自然言語のタスク記述と直近の会話履歴を受け取り、文脈を理解した上で必要な分析ステップを考えます。
    ユーザーから分析依頼があり、分析要件を具体化したい際に使います。
    """
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

    return step_answer

def interpret_node(state: AgentState):
    """
    自然言語のタスク記述と直近の会話、そしてデータを受け取り、文脈を理解した上でデータを解釈します。
    分析のためにデータを解釈する必要がある際に使用します。
    """
    try:
        task_description = state["messages"][-1].content
        tool_call_id = state["messages"][-2].tool_calls[0]['id']
    except (IndexError, KeyError):
        error_message = "不正な呼び出し形式です。スーパーバイザーからの指示が正しくありません。"
        # tool_call_idが不明なため、Noneを指定
        return {"messages": [ToolMessage(content=error_message, tool_call_id=None)]}

    logging.info(f"interpret_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", [])
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
    
    # システムメッセージ以外を抽出
    conversation_history = state.get("messages", [])
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        interpretation_text = response.content.strip() if response.content else ""

        if not interpretation_text:
            raise RuntimeError("interpret_node: LLMが空の解釈を返しました。")

    except Exception as e:
        error_message = f"データの解釈中にエラーが発生しました: {e}"
        logging.error(error_message)
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    # --- 3. 出口: 結果をToolMessageで報告する ---
    return {
        "messages": [
            ToolMessage(
                content=interpretation_text,
                tool_call_id=tool_call_id
            )
        ]
    }


def sql_node(state: AgentState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、、文脈を理解した上でSQLを生成・実行します。
    分析のためにデータを取得する必要がある際に使用します。
    """
    try:
        task_description = state["messages"][-1].content
        tool_call_id = state["messages"][-2].tool_calls[0]['id']
    except (IndexError, KeyError):
        # スーパーバイザーからの呼び出し形式が正しくない場合のエラーハンドリング
        error_message = {"status": "error", "error_message": "不正な呼び出し形式です。"}
        # tool_call_idが不明なため、Noneを指定
        return {"messages": [ToolMessage(content=json.dumps(error_message), tool_call_id=None)]}


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
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])
    
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
    try:
        logging.info(f"sql_node: 生成AIが考えています・・・ '{task_description}'")
        response = llm.invoke([
            {"role": "system", "content": system_prompt_sql_generation},
            {"role": "user", "content": user_prompt_for_req}
        ])
        sql_generated_clean = extract_sql(response.content.strip())
        last_sql_generated = sql_generated_clean # Store the latest SQL attempt for this requirement
        result_df, sql_error = try_sql_execute(sql_generated_clean)

        #1回目は失敗した場合
        if sql_error:
            logging.warning(f"'最初のSQLが失敗しました: {sql_error}。修正して再試行します。")
            fixed_sql = fix_sql_with_llm(llm, sql_generated_clean, sql_error, rag_tables, rag_queries, task_description, context)
            last_sql_generated = fixed_sql # この要件に対する最新のSQL試行を保存
            result_df, sql_error = try_sql_execute(fixed_sql)

            #2回目も失敗した場合
            if sql_error:
                logging.error(f"修正SQLも失敗: {sql_error}")
                # エラーの場合も、その情報をメッセージとして返す
                result_payload = {
                    "status": "error",
                    "error_message": str(sql_error),
                    "last_attempted_sql": last_sql_generated
                    }
            else:
                # 2回目は成功した場合
                result_payload = {
                    "status": "success",
                    "executed_sql": last_sql_generated,
                    "dataframe_as_json": result_df.to_json(orient="records")
                }
                new_history_item = {task_description: result_df.to_dict(orient="records")}
        else:
            # 1回で成功した場合
            result_payload = {
                "status": "success",
                "executed_sql": last_sql_generated,
                "dataframe_as_json": result_df.to_json(orient="records")
            }
            new_history_item = {task_description: result_df.to_dict(orient="records")}
    except Exception as e:
        # このノード全体で予期せぬエラーが起きた場合
        logging.error(f"sql_nodeで予期せぬエラー: {e}")
        result_payload = {
            "status": "error",
            "error_message": f"SQLノードの実行中に予期せぬエラーが発生しました: {e}",
            "last_attempted_sql": locals().get("last_sql_generated", "N/A")
        }
    # 状態を直接いじる代わりに、結果をメッセージとして返す
    return {
        "messages": [
            ToolMessage(
                content=json.dumps(result_payload, ensure_ascii=False),
                tool_call_id=tool_call_id
            )
        ],
        "df_history": state.get("df_history", []) + [new_history_item]
    }

def chart_node(state: AgentState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でグラフを作成します。
    分析のためにグラフを作成する必要がある際に使用します。
    """
    # messagesの最後にあるのが、スーパーバイザーからのツールコール指示 or 専門家へのディスパッチ指示
    # その指示内容(content)を現在のタスクとして扱う
    task_description = state["messages"][-1].content
    # どの指示に対する応答なのかを紐付けるために、tool_call_idを取得しておく
    tool_call_id = state["messages"][-2].tool_calls[0]['id']
    
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
    class ChartOutput(BaseModel):
        """グラフのJSONを格納するためのデータスキーマ"""
        plotly_json: str = Field(description="Plotlyで生成されたグラフのJSON文字列")

    tools = [python_tool, ChartOutput]
    system_prompt = """
    最適なインタラクティブグラフを `plotly.express` (例: `px`) を使用して生成してください。
    生成したFigureオブジェクトを `fig` という変数に格納し、その後 `fig.to_json()` を呼び出してJSON文字列に変換し、そのJSON文字列を `print` してください。
    余計な説明は不要です。最終的な出力はJSON文字列だけにしてください
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm_with_chart_tool = llm.bind_tools(tools)
    chart_agent = create_openai_tools_agent(llm_with_chart_tool, tools, prompt)
    agent_executor = AgentExecutor(
        agent=chart_agent, 
        tools=[python_tool], # ChartOutputクラスは実行ツールではないので含めない
        verbose=True,
        handle_parsing_errors=True # 解析エラー時の処理を追加するとより堅牢
    )

    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

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

    logging.info(f"chart_node: 生成AIが考えています・・・ '{task_description}'")
    agent_response = agent_executor.invoke(
        {"input":user_prompt}
        )
    logging.info(f"chart_node: Agent response: {agent_response}")

    # The agent's output should be the Plotly JSON string
    plotly_json_string = agent_response['tool_calls'][0]['args']['plotly_json']
    try:
        json.loads(plotly_json_string)

    except (json.JSONDecodeError, TypeError) as e:
        error_message = f"生成されたグラフのJSONが無効です: {e}"
        logging.info(f"chart_node: {error_message}")
        # エラーが発生した場合も、ToolMessageで結果を返す
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    # stateを直接いじるのではなく、結果をメッセージとして返す
    return {
        "messages": [
            ToolMessage(
                content=plotly_json_string,
                tool_call_id=tool_call_id
            )
        ]
    }


def processing_node(state: AgentState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータを加工します。
    分析のためにデータを加工する必要がある際に使用します。
    """
    task_description = state["messages"][-1].content
    tool_call_id = state["messages"][-2].tool_calls[0]['id']

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
    # 加工後のDataFrame(JSON文字列)を受け取るためのダミーのツール
    class ProcessedDataFrame(BaseModel):
        """データ加工後のDataFrameを格納するためのデータスキーマ"""
        dataframe_as_json_string: str = Field(
            description="データ加工後のPandas DataFrameを to_json(orient='records') で変換したJSON文字列。"
        )
    tools = [python_tool, ProcessedDataFrame]
    # LLMにツールをバインド
    processing_llm = llm.bind_tools(tools)
    system_prompt = """
    あなたはPythonとPandasを使ったデータ加工の専門家です。
    与えられたデータと指示に基づき、データ加工を行うPythonコードを作成・実行してください。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    processing_agent = create_openai_tools_agent(processing_llm, tools, prompt)
    processing_agent_executor = AgentExecutor(
        agent=processing_agent,
        tools=[python_tool], # ProcessedDataFrameクラスは実行ツールではないので含めない
        verbose=True,
        handle_parsing_errors=True # 解析エラー時の処理を追加するとより堅牢
    )


    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
    recent_history = non_system_history[-history_back_number:]
    context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])

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
        # 内部エージェントを実行
        response = processing_agent_executor.invoke({"input": user_prompt})
        
        # エージェントが ProcessedDataFrame ツールを呼び出したことを確認
        processed_df_json = None
        for tool_call in response.get('tool_calls', []):
            if tool_call['name'] == ProcessedDataFrame.__name__:
                processed_df_json = tool_call['args']['dataframe_as_json_string']
                break
        
        if processed_df_json is None:
            raise RuntimeError("エージェントは加工後のDataFrameを返しませんでした。")
        
        processed_df = pd.read_json(io.StringIO(processed_df_json), orient="records")
        new_history_item = {task_description: processed_df.to_dict(orient="records")}

    except Exception as e:
        error_message = f"データ加工中にエラーが発生しました: {e}"
        print(error_message)
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    # --- 6. 出口の変更: 結果をToolMessageで報告する ---
    return {
        "messages": [
            ToolMessage(
                content=processed_df_json,
                tool_call_id=tool_call_id
            )
        ],
        "df_history": state.get("df_history", []) + [new_history_item]
    }

# --- 2. スーパーバイザーの出力スキーマ定義 (Pydantic) ---
# --- 2. スーパーバイザーのディスパッチ指示用スキーマ ---
# LLMにこの形式で出力するように強制する
class DispatchDecision(BaseModel):
    """専門家エージェントにタスクを割り振る際の意思決定スキーマ"""
    next_agent: str = Field(description="次に指名すべき専門家（ノード）の名前。利用可能な専門家がいない場合は 'FINISH'。")
    task_description: str = Field(description="指名した専門家に与える、具体的で明確な指示内容。")
    rationale: str = Field(description="なぜその判断を下したのかの簡潔な理由。")
                           
# スーパーバイザーのLLMチェーン定義 ---
# 利用可能な専門家（ノード）のリスト
members = ["sql_node", "chart_node", "processing_node", "interpret_node"]
tools = [metadata_retrieval_node, analysis_plan_node]

# スーパーバイザーに与えるシステムプロンプト
system_prompt_supervisor = (
    "あなたはAIチームを率いる熟練のプロジェクトマネージャーです。\n"
    "あなたの仕事は、ユーザーの要求に基づき、以下のいずれかのアクションを選択することです。\n"
    "1. 自分でツールを実行する: 簡単な情報照会などは、あなたが直接ツールを実行して回答してください。\n"
    "2. 専門家に仕事を割り振る: 複雑な作業は、タスクを分解し、以下の専門家メンバーに具体的で明確な指示を出してください。\n\n"
    "== 利用可能なツール ==\n"
    "- metadata_retrieval_tool: データベースのテーブル定義やカラムについて答える。\n\n"
    "- analysis_plan_node: 分析のプランを立てる\n\n"

    "== 専門家メンバーリスト ==\n"
    "- sql_node: 依頼に従ってSQLを生成し、データを取得する\n\n"
    "- chart_node: 依頼に従ってデータからグラフを作成する\n\n"
    "- processing_node: 依頼に従ってデータを加工する\n\n"
    "- interpret_node: 依頼に従ってデータの解釈を行う\n\n"
    "== 判断ルール ==\n"
    "まずツールで解決できるか検討し、できなければ専門家への割り振りを考えてください。\n"
    "専門家へ割り振る際は、次に指名すべきメンバーと、そのメンバーへの具体的な指示内容を決定してください。\n"
    "全てのタスクが完了したと判断した場合は、next_agentに 'FINISH' を指定してください。"
)

# プロンプトテンプレートの作成
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_supervisor),
    MessagesPlaceholder(variable_name="messages"),
])
# LLMに「ディスパッチ指示」という名の「ツール」と、実際のツール群の両方を教える
# これにより、LLMはディスパッチするか、ツールを使うかを選べるようになる
llm_with_tools = llm.bind_tools(tools + [DispatchDecision])

# LLMと出力スキーマを結合して、構造化出力を強制するチェーンを作成
# これがスーパーバイザーの頭脳となる
supervisor_chain = supervisor_prompt | llm_with_tools

# --- 4. スーパーバイザーノード本体 ---
def supervisor_node(state: AgentState):
    """スーパーバイザーとして次にどのアクションをとるべきか判断する"""
    print("--- スーパーバイザー ノード実行 ---")
    
    response = supervisor_chain.invoke({"messages": state["messages"]})
    
    # CASE 1: LLMがツールを使うと判断した場合
    if response.tool_calls:
        print("判断: ツールを直接実行します。")
        # 新しいメッセージとしてツールコールを追加し、'next'を'tools'に設定
        return {"messages": [response], "next": "tools"}
        
    # CASE 2: LLMが専門家に割り振ると判断した場合
    # response.tool_calls[0]['args']にDispatchDecisionの引数が入ってくる
    else:
        # Pydanticモデルにパースして扱いやすくする
        dispatch_decision = DispatchDecision(**response.tool_calls[0]['args'])
        print(f"判断: 専門家 '{dispatch_decision.next_agent}' にタスクを割り振ります。")
        print(f"判断理由: {dispatch_decision.rationale}")
        print(f"具体的な指示: {dispatch_decision.task_description}")
        
        # 次のノード（専門家）に渡すためのメッセージを作成
        # これにより、専門家は「スーパーバイザーからこの指示が来た」と認識できる
        dispatch_message = ToolMessage(
            content=f"以下の指示を実行してください：\n\n{dispatch_decision.task_description}",
            tool_call_id=response.tool_calls[0]['id'] # どの判断に対応するかを紐付け
        )
        return {
            "messages": [response, dispatch_message],
            "next": dispatch_decision.next_agent
        }

def build_workflow():
    graph_builder = StateGraph(AgentState)

    # 1. スーパーバイザーノードを追加
    graph_builder.add_node("supervisor", supervisor_node)

    # 2. ツール実行ノードを追加
    # ToolNodeは、スーパーバイザーが呼び出しを決めたツールを自動で実行してくれる便利なノード
    # スーパーバイザーと各専門家をノードとして追加
    graph_builder.add_node("sql", sql_node) 
    graph_builder.add_node("chart", chart_node)
    graph_builder.add_node("processing", processing_node)
    graph_builder.add_node("interpret", interpret_node)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # エントリーポイントはスーパーバイザー
    graph_builder.set_entry_point("supervisor")

    # スーパーバイザーの決定 (`state['next']`) に基づいて、次に進むノードを決める
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "sql": "sql",
            "chart": "chart",
            "processing": "processing",
            "tools":"tools",
            "FINISH": END
        }
    )


    # 各専門家の作業が終わったら、必ずスーパーバイザーに報告に戻る
    graph_builder.add_edge("sql", "supervisor")
    graph_builder.add_edge("chart", "supervisor")
    graph_builder.add_edge("processing", "supervisor")
    graph_builder.add_edge("tools", "supervisor")
    
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)
