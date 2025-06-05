import os
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv() # .env ファイルから環境変数を読み込む

# --- 1. 環境変数の設定 (APIキーを安全に管理するために推奨) ---

# 環境変数からAPIキーを読み込む
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

# --- 2. 埋め込みモデルの初期化 (Gemini Embeddings) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# --- 3. JSONファイルの読み込み ---
def load_json_data(file_path):
    """JSONファイルを読み込むヘルパー関数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None
    except json.JSONDecodeError:
        print(f"エラー: ファイル '{file_path}' のJSON形式が不正です。")
        return None

table_definitions_data = load_json_data("table_definitions.json")
query_examples_data = load_json_data("query_examples.json")

if table_definitions_data is None or query_examples_data is None:
    print("データファイルの読み込みに失敗したため、処理を終了します。")
    exit()

def format_table_schema_for_llm(table_json_string: str) -> str:
    """
    テーブル定義のJSON文字列を、LLMがルールを認識しやすい
    構造化されたテキスト形式に変換します。
    """
    try:
        data = json.loads(table_json_string)
    except json.JSONDecodeError:
        # JSONとしてパースできない場合は、元の文字列をそのまま返すかエラー処理
        return table_json_string 

    text_parts = []
    text_parts.append(f"<schema>")
    text_parts.append(f"  <table_name>{data['table_name']}</table_name>")
    
    if data.get("table_explanation"):
        text_parts.append(f"  <table_rules_header>【テーブル全体に関するルール】</table_rules_header>")
        text_parts.append(f"  <table_explanation>{data['table_explanation']}</table_explanation>")
    
    if data.get("columns"):
        text_parts.append(f"  <columns_header>【カラム定義と個別ルール】</columns_header>")
        text_parts.append(f"  <columns>")
        for col in data["columns"]:
            text_parts.append(f"    <column>")
            text_parts.append(f"      <name>{col['name']}</name>")
            if col.get("explanation"):
                text_parts.append(f"      <explanation>{col['explanation']}</explanation>")
            # 必要であれば型情報などもここに含める
            # text_parts.append(f"      <type>{col.get('type', '不明')}</type>")
            text_parts.append(f"    </column>")
        text_parts.append(f"  </columns>")
    text_parts.append(f"</schema>")
    
    return "\n".join(text_parts)

# --- 4. ドキュメントの準備 ---

# テーブル定義のドキュメントを作成
table_docs = []
for table in table_definitions_data:
    table_name = table["table_name"]
    table_description = table["table_description"]
    columns_info = []
    for col in table["columns"]:
        columns_info.append(f"- {col['name']} ({col['type']}): {col['description']}")
    
    # 結合してコンテンツを作成
    content = f"テーブル名: {table_name}\n概要: {table_description}\nカラム:\n" + "\n".join(columns_info)
    
    # メタデータも追加可能
    metadata = {"source": "table_definition", "table_name": table_name}
    table_docs.append(Document(page_content=content, metadata=metadata))
    print(f"テーブル定義ドキュメント作成: {table_name}")

# クエリ例のドキュメントを作成
query_docs = []
for example in query_examples_data:
    query = example["query_example"]
    sql = example["sql_example"]
    
    # ユーザーの質問とそれに対応するSQLをセットでドキュメント化
    content = f"ユーザー質問例: {query}\n対応SQL例: {sql}"
    
    metadata = {"source": "query_example", "query": query, "sql": sql}
    query_docs.append(Document(page_content=content, metadata=metadata))
    print(f"クエリ例ドキュメント作成: {query}")

# --- 5. FAISS ベクトルストアの構築と保存 ---

# スクリプトがあるディレクトリのパスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))

# FAISSの保存パスを結合
faiss_tables_path = os.path.join(script_dir, "faiss_tables")
faiss_queries_path = os.path.join(script_dir, "faiss_queries")

# テーブル定義のベクトルストア
print("\nテーブル定義のFAISSベクトルストアを構築中...")
vectorstore_tables = FAISS.from_documents(table_docs, embeddings)
vectorstore_tables.save_local(faiss_tables_path) # ここを変更
print(f"テーブル定義のFAISSベクトルストアを '{faiss_tables_path}' に保存しました。")

# クエリ例のベクトルストア
print("\nクエリ例のFAISSベクトルストアを構築中...")
vectorstore_queries = FAISS.from_documents(query_docs, embeddings)
vectorstore_queries.save_local(faiss_queries_path) # ここを変更
print(f"クエリ例のFAISSベクトルストアを '{faiss_queries_path}' に保存しました。")

# (オプション) 構築したベクトルストアをロードして確認する例
# print("\n--- 構築したベクトルストアのロードと確認 ---")
# loaded_vectorstore_tables = FAISS.load_local("faiss_tables", embeddings, allow_dangerous_deserialization=True)
# print(f"ロードされたテーブルストアのドキュメント数: {len(loaded_vectorstore_tables.index_to_docstore_id)}")

# loaded_vectorstore_queries = FAISS.load_local("faiss_queries", embeddings, allow_dangerous_deserialization=True)
# print(f"ロードされたクエリストアのドキュメント数: {len(loaded_vectorstore_queries.index_to_docstore_id)}")

# # 検索例
# user_query = "商品カテゴリごとの売上を知りたい"
# retrieved_docs = loaded_vectorstore_tables.similarity_search(user_query, k=2)
# print(f"\nユーザーの質問 '{user_query}' に関連するテーブル定義:")
# for doc in retrieved_docs:
#     print(f"- {doc.page_content[:100]}...")

# retrieved_query_examples = loaded_vectorstore_queries.similarity_search(user_query, k=2)
# print(f"\nユーザーの質問 '{user_query}' に関連するクエリ例:")
# for doc in retrieved_query_examples:
#     print(f"- {doc.page_content[:100]}...")
