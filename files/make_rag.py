import os
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Assuming you intend to use this
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv() # .env ファイルから環境変数を読み込む

# --- 1. 環境変数の設定 ---
# google_api_key = os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

# # --- 2. 埋め込みモデルの初期化 ---
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

#Chatモデル（SQL生成用）
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
version4emb = os.getenv("AZURE_OPENAI_API_VERSION4EMB")
deployment4emb = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4EMB")

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=api_key,
    azure_endpoint=endpoint,
    openai_api_version=version4emb,
    azure_deployment=deployment4emb
)

# --- 3. JSONデータの読み込み ---
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

# --- 4. LLM向けテキスト表現への変換関数 ---
def format_table_schema_for_llm(table_dict: dict) -> str: # ★ 入力を辞書型に変更
    """
    テーブル定義のPython辞書を、LLMがルールを認識しやすい
    構造化されたテキスト形式に変換します。
    """
    data = table_dict # ★ json.loadsは不要、既に関数呼び出し元で辞書になっている想定

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
            text_parts.append(f"    </column>")
        text_parts.append(f"  </columns>")
    text_parts.append(f"</schema>")
    
    return "\n".join(text_parts)

# --- 6. データのロード ---
# table_definitions_dataは、複数のテーブル定義を含む「リスト」としてロードされる
table_definitions_list = load_json_data("table_definitions.json")
query_examples_data = load_json_data("query_examples.json")

if table_definitions_list is None or query_examples_data is None:
    print("データファイルの読み込みに失敗したため、処理を終了します。")
    exit()

# --- 7. ドキュメントの準備 ---
table_docs = []
if isinstance(table_definitions_list, list): # ★ リストであることを確認
    for table_definition_dict in table_definitions_list: # ★ 各テーブル定義(辞書)に対して処理
        # ★ LLM向けのテキスト表現に変換
        formatted_content = format_table_schema_for_llm(table_definition_dict)
        
        table_name = table_definition_dict.get("table_name", "不明なテーブル")
        
        # ★ メタデータも更新
        metadata = {"source": "table_definition", "table_name": table_name}
        table_docs.append(Document(page_content=formatted_content, metadata=metadata))
        print(f"テーブル定義ドキュメント作成 (LLM向け整形済): {table_name}")
else:
    print("エラー: table_definitions.json の内容がリスト形式ではありません。")
    exit()


# クエリ例のドキュメントを作成 (ここは変更なしで動くはず)
query_docs = []
for example in query_examples_data:
    query = example.get("query_example", "不明な質問例")
    sql = example.get("sql_example", "不明なSQL例")
    
    content = f"ユーザー質問例: {query}\n対応SQL例: {sql}"
    
    metadata = {"source": "query_example", "query": query, "sql": sql}
    query_docs.append(Document(page_content=content, metadata=metadata))
    print(f"クエリ例ドキュメント作成: {query}")

# --- 8. FAISS ベクトルストアの構築と保存 ---
faiss_tables_path = os.path.join("faiss_tables")
faiss_queries_path = os.path.join("faiss_queries")

if table_docs: # ★ table_docsが空でないことを確認
    print("\nテーブル定義のFAISSベクトルストアを構築中...")
    vectorstore_tables = FAISS.from_documents(table_docs, embeddings)
    vectorstore_tables.save_local(faiss_tables_path)
    print(f"テーブル定義のFAISSベクトルストアを保存しました。")
else:
    print("テーブル定義ドキュメントが作成されなかったため、FAISSベクトルストア（テーブル用）は構築されませんでした。")


if query_docs: # ★ query_docsが空でないことを確認
    print("\nクエリ例のFAISSベクトルストアを構築中...")
    vectorstore_queries = FAISS.from_documents(query_docs, embeddings)
    vectorstore_queries.save_local(faiss_queries_path)
    print(f"クエリ例のFAISSベクトルストアを保存しました。")
else:
    print("クエリ例ドキュメントが作成されなかったため、FAISSベクトルストア（クエリ用）は構築されませんでした。")

print("\n処理が完了しました。")
