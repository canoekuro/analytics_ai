from langchain_community.vectorstores import FAISS
import sqlite3
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain.agents import Tool, initialize_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import os
import base64
import re
from typing import TypedDict, List, Optional, Any
from langgraph.checkpoint.memory import MemorySaver
import japanize_matplotlib
import seaborn as sns
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key) # Or other suitable Gemini embedding model

# ベクトルストア
vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    temperature=0,
    google_api_key=google_api_key
) # Or "gemini-1.5-flash", "gemini-1.5-pro" for higher capabilities

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
        return None, str(e)
    
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

# SQL生成＋実行
def sql_node(state):
    state["df"] = None
    state["interpretation"] = None
    state["chart_result"] = None
    # 1. テーブル定義DBから検索
    retrieved_tables = vectorstore_tables.similarity_search(state["input"], k=3)
    rag_tables = "\n".join([doc.page_content for doc in retrieved_tables])

    # 2. クエリ例DBから検索
    retrieved_queries = vectorstore_queries.similarity_search(state["input"], k=3)
    rag_queries = "\n".join([doc.page_content for doc in retrieved_queries])

     # 3. プロンプト組み立て
    user_prompt = f"""
    
    【テーブル定義に関する情報】
    {rag_tables}

    【類似する問い合わせ例とそのSQL】
    {rag_queries}

    【ユーザー質問】
    {state['input']}
    """

    system_prompt = """
    あなたはSQL生成AIです。この後の質問に対し、SQLiteの標準的なSQL文のみを出力してください。
    - 使用できるSQL構文は「SELECT」「WHERE」「GROUP BY」「ORDER BY」「LIMIT」のみです。
    - 日付関数や高度な型変換、サブクエリやウィンドウ関数、JOINは使わないでください。
    - 必ず1つのテーブルだけを使い、簡単な集計・フィルタ・並べ替えまでにしてください。
    - SQLの前後にコメントや説明文は出力しないでください。出力された内容をそのまま実行するため、SQL文のみをそのまま出力してください。
    """

    # メッセージリストで渡す
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    sql_generated = response.content.strip()
    sql_generated_clean = extract_sql(sql_generated)
    print(sql_generated_clean)
    result_df, sql_error = try_sql_execute(sql_generated_clean)
    if sql_error:
        # LLMで修正を依頼
        sql_generated_clean = fix_sql_with_llm(sql_generated_clean, sql_error, rag_tables, rag_queries, state["input"])
        print(sql_generated_clean)
        # 再実行
        result_df, sql_error2 = try_sql_execute(sql_generated_clean)
        if sql_error2:
            print("再実行でもエラー:", sql_error2)
        else:
            print("=== 修正版SQLでの結果 ===")
            print(result_df)

    result_df_dict = result_df.to_dict(orient="records")
    return {**state, "df": result_df_dict, "SQL":sql_generated_clean, "condition": "SQL実行完了"}

# 解釈
def interpret_node(state):
    result_df_dict = state.get("df")
    if result_df_dict is None:
        return {"interpretation": "まだデータがありません。先にSQL質問をしてください。"}
    
    result_df = pd.DataFrame(result_df_dict)
    df_text = result_df.to_string(index=False)
    
    system_prompt = "あなたは優秀なデータ分析の専門家です。"
    user_prompt = f"""
    以下はSQLクエリの実行結果です。
    このデータから読み取れる特徴や傾向、示唆を簡潔な日本語で解説してください。

    {df_text}
    """

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return {**state, "interpretation": response.content.strip(), "condition": "解釈完了"}


def chart_node(state):
    result_df_dict = state.get("df")
    if result_df_dict is None:
        return {**state, "chart_result": None, "condition": "グラフ化失敗"}

    result_df = pd.DataFrame(result_df_dict)
    python_tool = PythonAstREPLTool(
        locals={
            "df": result_df,
            "sns": sns
        },
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
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    chart_prompt = f"""
    あなたはPythonプログラミングが得意なデータ分析の専門家です。
    以下のdf（pandas DataFrame）とユーザーからの依頼に答える
    最適なグラフをsns(seaborn)で作成して、output.pngという名前で保存してください。

    ユーザーからの依頼:
    {state['input']}
    dfの内容:
    {result_df.head(10).to_string(index=False)}
    """
    agent.invoke(chart_prompt)

    if os.path.exists("output.png"):
        fig = base64.b64encode(open("output.png", "rb").read()).decode('utf-8')
        return {**state, "chart_result": fig, "condition": "グラフ化完了"}
    else:
        return {**state, "condition": "グラフ化失敗"}

def classify_intent_node(state):
    user_input = state["input"]
    # ここをAzure OpenAIで複数分類に拡張
    prompt = f"""
    ユーザーの質問の意図を判定してください。
    次の3つのうち、該当するものを「,」区切りで全て列挙してください：
    - データ取得
    - グラフ作成
    - データ解釈

    質問: 「{user_input}」
    例: 
    input:「カテゴリの合計販売金額を出して」 output:「データ取得」
    input:「＊＊＊のデータを出して」 output:「データ取得」
    input:「＊＊＊のデータを取得してグラフ化して」 output:「データ取得,グラフ作成」
    input:「＊＊＊のデータを取得して解釈して」 output:「データ取得,データ解釈」
    input:「＊＊＊のデータのグラフ化と解釈して」 output:「データ取得,グラフ作成,データ解釈」 

    他の情報は不要なので、outputの部分（「データ取得,グラフ作成,データ解釈」）だけを必ず返すようにしてください。
    """
    result = llm.invoke(prompt).content.strip()
    steps = [x.strip() for x in result.split(",") if x.strip()]
    return {**state, "intent_list": steps, "condition": "分類完了"}

# classifyノードの分岐（次に何へ進むか？）
def classify_next(state):
    intents = state.get("intent_list", [])
    if "データ取得" in intents:
        return "sql"
    elif "グラフ作成" in intents:
        return "chart"
    elif "データ解釈" in intents:
        return "interpret"
    else:
        return END

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

class MyState(TypedDict, total=False):
    input: str                       # ユーザーの問い合わせ
    intent_list: List[str]           # 分類結果（データ取得/グラフ作成/データ解釈）
    df: Optional[list]               # SQL実行後のpandas.DataFrameのdict
    SQL: Optional[str]               # 生成されたSQL
    interpretation: Optional[str]    # データ解釈（分析コメント）
    chart_result: Optional[str]      # グラフ画像（base64など）
    condition: Optional[str]         # 各ノードの実行状態
    error: Optional[str]             # SQL等でエラーがあれば

def build_workflow():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MyState)
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("chart", chart_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_conditional_edges("classify", classify_next)
    workflow.add_conditional_edges("sql", sql_next)
    workflow.add_conditional_edges("chart", chart_next)
    workflow.add_edge("interpret", END)
    workflow.set_entry_point("classify")
    return workflow.compile(checkpointer=memory)

user_query = "カテゴリの合計販売金額を出して"
workflow = build_workflow()
config = {"configurable": {"thread_id": "1"}}
res = workflow.invoke({"input": user_query}, config=config)
print(res)
