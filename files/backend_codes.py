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
from functions import extract_sql, try_sql_execute, fix_sql_with_llm



# 基本的なロギングを設定
logging.basicConfig(level=logging.INFO)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
version4emb = os.getenv("AZURE_OPENAI_API_VERSION4EMB")
deployment4emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4EMB")
version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
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
    next: str  # 次に実行すべきノード名
    df_history: List[dict] 

history_back_number = 5
RERANK_CANDIDATE_COUNT = 5

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
    
def sql_node(state: AgentState):
    """
    自然言語のタスク記述とstate内の会話履歴を受け取り、ルールを強く意識した上でSQLを生成・実行します。
    （二段階選抜とルール遵守CoTを実装）
    """
    try:
        task_description = state["messages"][-1].content
        tool_call_id = state["messages"][-2].tool_calls[0]['id']
        conversation_history = state.get("messages", [])
        non_system_history = [msg for msg in conversation_history if msg.type != "system"]
        recent_history = non_system_history[-history_back_number:]
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])
    except (IndexError, KeyError):
        error_message = {"status": "error", "error_message": "不正な呼び出し形式です。"}
        return {"messages": [ToolMessage(content=json.dumps(error_message), tool_call_id=None)]}

    # ★ --- 1. 候補となるテーブルを多めに取得（Re-rankingステップ） ---
    logging.info(f"sql_node(Re-rank): 候補テーブルを広く検索中 (k=RERANK_CANDIDATE_COUNT)...")
    # vectorstore_tablesには、format_table_schema_for_llmで変換済みのテキストが格納されている想定
    candidate_docs = vectorstore_tables.similarity_search(task_description, k=5)
    
    def get_table_name_from_formatted_doc(doc_content: str) -> Optional[str]:
        # format_table_schema_for_llm で <table_name> タグが使われることを想定
        for line in doc_content.split('\n'):
            line = line.strip()
            if line.startswith("<table_name>") and line.endswith("</table_name>"):
                            return line[len("<table_name>"):-len("</table_name>")].strip()
        return None
        
    candidate_table_names = [get_table_name_from_formatted_doc(doc.page_content) for doc in candidate_docs]
    candidate_tables_info_for_prompt = "\n\n".join(
        # 選別用プロンプトでは、ルールも見えるようにそのままのテキストを渡す
        [doc.page_content for name, doc in zip(candidate_table_names, candidate_docs) if name] 
    )

    # ★ --- 2. LLMにテーブルを選別させる（1回目の問いかけ） ---
    if candidate_tables_info_for_prompt: # 候補がある場合のみ選別
        logging.info(f"sql_node(Re-rank): LLMにテーブル選別を依頼中...")
        rerank_prompt = f"""
        ユーザーの最終的な要求は「{task_description}」です。
        この要求に答えるために、以下のテーブル定義リストの中から、SQLクエリの生成に本当に必要だと思われるテーブル名をすべて選び出し、カンマ区切りで出力してください。
        余計な説明は不要です。テーブル名のみをカンマ区切りで出力してください。

        【テーブル定義リスト】
        {candidate_tables_info_for_prompt}

        【必要なテーブル名リスト（カンマ区切り）】
        """
        rerank_response = llm.invoke(rerank_prompt)
        required_table_names_str = rerank_response.content.strip()
        if required_table_names_str: # LLMが何かしらテーブル名を返した場合
             required_table_names = [name.strip() for name in required_table_names_str.split(',')]
        else: # LLMがテーブル名を選べなかった場合、元の候補を全て使うか、エラーにするか検討。ここでは全て使う。
            logging.warning("LLMが選別するテーブル名を返さなかったため、検索候補を全て使用します。")
            required_table_names = [name for name in candidate_table_names if name] # Noneを除外

        logging.info(f"sql_node(Re-rank): LLMが選別したテーブル: {required_table_names}")
    else: # RAG検索で候補が一つもなかった場合
        logging.warning("RAG検索でテーブル候補が見つかりませんでした。")
        required_table_names = []


    # ★ --- 3. LLMの選別結果を基に、使用するテーブル定義を絞り込む ---
    final_docs = [doc for doc in candidate_docs if get_table_name_from_formatted_doc(doc.page_content) in required_table_names]
    final_rag_tables_text = "\n\n".join([doc.page_content for doc in final_docs])

    if not final_rag_tables_text: # 最終的に使用するテーブル情報がない場合はエラー
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

    # --- 6. SQL生成の実行と後続処理（2回目の問いかけ & 実行） ---
    try:
        logging.info(f"sql_node(CoT): 生成AIが考えています・・・ '{task_description}'")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        sql_generated_clean = extract_sql(response.content.strip())
        last_sql_generated = sql_generated_clean
        result_df, sql_error = try_sql_execute(sql_generated_clean)

        if sql_error:
            logging.warning(f"最初のSQLが失敗しました: {sql_error}。修正して再試行します。")
            fixed_sql = fix_sql_with_llm(llm, sql_generated_clean, sql_error, final_rag_tables_text, rag_queries, task_description, context)
            last_sql_generated = fixed_sql
            result_df, sql_error = try_sql_execute(fixed_sql)

            if sql_error:
                logging.error(f"修正SQLも失敗: {sql_error}")
                result_payload = {"status": "error", "error_message": str(sql_error), "last_attempted_sql": last_sql_generated}
            else:
                result_payload = {"status": "success", "executed_sql": last_sql_generated, "dataframe_as_json": result_df.to_json(orient="records")}
                new_history_item = {task_description: result_df.to_dict(orient="records")} if not result_df.empty else {}
        else:
            result_payload = {"status": "success", "executed_sql": last_sql_generated, "dataframe_as_json": result_df.to_json(orient="records")}
            new_history_item = {task_description: result_df.to_dict(orient="records")} if not result_df.empty else {}
    except Exception as e:
        logging.error(f"sql_nodeで予期せぬエラー: {e}")
        result_payload = {"status": "error", "error_message": f"SQLノードの実行中に予期せぬエラーが発生しました: {e}", "last_attempted_sql": locals().get("last_sql_generated", "N/A")}
    
    current_df_history = state.get("df_history", [])
    if new_history_item: # 成功した場合のみ履歴に追加
        current_df_history.append(new_history_item)
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

# 解釈ノード
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
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
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

# python用のCallback
class CodeCollectorCallback(BaseCallbackHandler):
    def __init__(self):
        self.codes = []
    
    # ★★★ on_agent_action から on_tool_start に変更 ★★★
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> Any:
        """ツールが実行を開始する直前に呼び出される"""
        # 実行されるツールの名前が "python_ast_repl" かどうかをチェック
        if serialized.get("name") == "python_ast_repl":
            # ツールの入力（Pythonコード）をリストに追加
            self.codes.append(input_str)

def processing_node(state: AgentState):
    """
    自然言語のタスク記述とstate内のデータを受け取り、文脈を理解した上でデータの加工やグラフの作成を行います。
    分析のためにデータの加工やグラフ作成をする必要がある際に使用します。
    """
    # messagesの最後にあるのが、スーパーバイザーからのツールコール指示 or 専門家へのディスパッチ指示
    # その指示内容(content)を現在のタスクとして扱う
    task_description = state["messages"][-1].content
    # どの指示に対する応答なのかを紐付けるために、tool_call_idを取得しておく
    tool_call_id = state["messages"][-2].tool_calls[0]['id']
    
    logging.info(f"processing_node: df_historyの読み込み開始・・・ '{task_description}'")
    df_history = state.get("df_history", None)
    if df_history is None:
        raise RuntimeError("processing_node: df_historyが空です。利用可能なデータがありません。")

    # 会話履歴の直近N件だけ取り出し
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

    #pythonツールの定義
    locals = {**df_locals, "px": px, "pd": pd}
    python_tool = PythonAstREPLTool(
        name="python_ast_repl",
        locals=locals, 
        description=(
            f"""Pythonコードを実行してデータの加工とグラフ化ができます。対象のDataFrameとして、{",".join(df_locals.keys())}が使用可能です。
            グラフを作成するときにはplotly.express (px) を使用してインタラクティブなグラフを生成し、最終的なグラフオブジェクトは必ず `fig` という変数に格納してください。
            データフレームを成果物とする場合は、データフレームは `final_df` という変数に格納してください。"""
        )
    )
    tools = [python_tool]

    #system_prompt
    system_prompt = """
    あなたはデータ分析のエキスパートです。ユーザーの指示に従い、ツールを使って分析し、結果を返してください。

    - 渡された`df`というDataFrameを操作・分析してください。
    - 可視化を行う場合、最適なインタラクティブグラフを`plotly.express` (例: `px`) を使用して生成してください。最終的なグラフオブジェクトは必ず `fig` という変数に格納してください。
    - データフレームを最終成果物とする場合、そのデータフレームは `final_df` という変数に格納してください。

    思考の過程:
    {agent_scratchpad}
    """

    # user_prompt準備
    # 直近の会話履歴
    conversation_history = state.get("messages", [])
    # システムメッセージ以外を抽出
    non_system_history = [msg for msg in conversation_history if msg.type != "system"]
    # 直近N件だけ取り出し
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

    #promptをまとめてAgentを準備
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    logging.info(f"processing_node: 分析開始・・・ '{task_description}'")
    llm_with_tool = llm.bind_tools(tools)
    agent = create_openai_tools_agent(llm_with_tool, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=[python_tool],
        verbose=True,
        handle_parsing_errors=True # 解析エラー時の処理を追加するとより堅牢
    )

    #agentの実行開始
    code_collector = CodeCollectorCallback()
    logging.info(f"processing_node: 生成AIが考えています・・・ '{task_description}'")
    agent_response = agent_executor.invoke(
        {"input":user_prompt},
        {"callbacks": [code_collector]} 
        )
    logging.info(f"processing_node: Agent response: {agent_response}")
    logging.info(f"processing_node: Agent response: {code_collector.codes}")

    try:
        logging.info(f"processing_node: コード検証中・・・ '{task_description}'")
        # 実行環境のスコープはPythonAstREPLを使いまわす
        execution_scope = locals
        for i in code_collector.codes:
            code_dict = ast.literal_eval(i)
            code_str = code_dict["query"]   
            exec(code_str, execution_scope)
        
        # 最終的なオブジェクトをスコープから取得して利用
        fig = execution_scope.get('fig')
        result_df = execution_scope.get('final_df')
    
        if fig is not None:
            fig_json = fig.to_json()
        else:
            fig_json = None

        if result_df is not None:
            result_df_json = result_df.to_json(orient="records", force_ascii=False)
            new_history_item = {task_description: result_df.to_dict(orient="records")}
        else:
            result_df_json = None
            new_history_item = None

        if (fig_json is None) and (result_df_json is None):
            content = "データも、グラフも生成されませんでした"
        else:
            content = json.dumps({
                "result_df_json": result_df_json,
                "fig_json": fig_json
            }, ensure_ascii=False)

    except Exception as e:
        error_message = f"生成AIがデータ加工・グラフ作成に失敗しました。: {e}"
        logging.info(f"processing_node: {error_message}")
        # エラーが発生した場合も、ToolMessageで結果を返す
        return {"messages": [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}

    # stateを直接いじるのではなく、結果をメッセージとして返す
    return {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id
            )
        ],
        "df_history": state.get("df_history", []) + [new_history_item],
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
members = ["sql_node", "processing_node", "interpret_node"]
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
    "- processing_node: 依頼に従ってデータの加工やグラフ作成をする\n\n"
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
    logging.info("--- スーパーバイザー ノード実行 ---")
    
    response = supervisor_chain.invoke({"messages": state["messages"]})
    logging.info(response)

    tool_calls = response.additional_kwargs.get("tool_calls", [])
    # 2. tool_callsがあるかどうかで分岐
    if not tool_calls:
        logging.info("supervisor_node: 判断:応答完了")
        return{
            "messages":[*state["messages"], response],
            "next":"FINISH"
            }
    else:
        first_call = tool_calls[0]
        args_json = first_call["function"]["arguments"]
        try:
            args = json.loads(args_json)
        except Exception as e:
            raise (f"argumentsのJSONデコードに失敗:{e}")

        if "next_agent" not in args:            
            logging.info("判断: ツールを直接実行します。")
            # 新しいメッセージとしてツールコールを追加し、'next'を'tools'に設定
            return {"messages": [*state["messages"], response], "next": "tools"}
            
        # CASE 2: LLMが専門家に割り振ると判断した場合
        # response.tool_calls[0]['args']にDispatchDecisionの引数が入ってくる
        else:
            # Pydanticモデルにパースして扱いやすくする
            dispatch_decision = DispatchDecision(**response.tool_calls[0]['args'])
            logging.info(f"判断: 専門家 '{dispatch_decision.next_agent}' にタスクを割り振ります。")
            logging.info(f"判断理由: {dispatch_decision.rationale}")
            logging.info(f"具体的な指示: {dispatch_decision.task_description}")
            
            # 次のノード（専門家）に渡すためのメッセージを作成
            # これにより、専門家は「スーパーバイザーからこの指示が来た」と認識できる
            dispatch_message = ToolMessage(
                content=f"以下の指示を実行してください：\n\n{dispatch_decision.task_description}",
                tool_call_id=response.tool_calls[0]['id'] # どの判断に対応するかを紐付け
            )
            return {
                "messages": [*state["messages"], response, dispatch_message], # 必要に応じてHumanMessage/AIMessage
                "next": dispatch_decision.next_agent
            }


def build_workflow():
    graph_builder = StateGraph(AgentState)

    # 1. スーパーバイザーノードを追加
    graph_builder.add_node("supervisor", supervisor_node)

    # 2. ツール実行ノードを追加
    # ToolNodeは、スーパーバイザーが呼び出しを決めたツールを自動で実行してくれる便利なノード
    # スーパーバイザーと各専門家をノードとして追加
    graph_builder.add_node("sql_node", sql_node) 
    graph_builder.add_node("processing_node", processing_node)
    graph_builder.add_node("interpret_node", interpret_node)
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # エントリーポイントはスーパーバイザー
    graph_builder.set_entry_point("supervisor")

    # スーパーバイザーの決定 (`state['next']`) に基づいて、次に進むノードを決める
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "sql_node": "sql_node",
            "processing_node": "processing_node",
            "tools":"tools",
            "FINISH": END
        }
    )


    # 各専門家の作業が終わったら、必ずスーパーバイザーに報告に戻る
    graph_builder.add_edge("sql_node", "supervisor")
    graph_builder.add_edge("processing_node", "supervisor")
    graph_builder.add_edge("tools", "supervisor")
    
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)
