import sqlite3
import pandas as pd
import json

# ====== 1. ダミーデータ定義 ======

# sales_data
sales_data = [
    (1, "飲料", "コーラ", 12000, "2024-05-01"),
    (2, "飲料", "お茶", 15000, "2024-05-01"),
    (3, "食品", "パン", 9000, "2024-05-01"),
    (4, "飲料", "コーラ", 11000, "2024-05-02"),
    (5, "食品", "サンドイッチ", 13000, "2024-05-02"),
    (6, "食品", "パン", 9500, "2024-05-02"),
    (7, "飲料", "お茶", 17000, "2024-05-02"),
]
sales_columns = ["id", "category", "product_name", "sales", "date"]

# users
users = [
    (1, "山田太郎", 30, "東京都"),
    (2, "佐藤花子", 25, "大阪府"),
    (3, "鈴木次郎", 40, "福岡県"),
    (4, "高橋美咲", 35, "北海道"),
]
users_columns = ["user_id", "user_name", "age", "prefecture"]

# products
products = [
    (1, "コーラ", "飲料", 120),
    (2, "お茶", "飲料", 100),
    (3, "パン", "食品", 150),
    (4, "サンドイッチ", "食品", 300),
]
products_columns = ["product_id", "product_name", "category", "price"]

# reviews
reviews = [
    (1, 1, 5, "美味しかったです"),
    (2, 3, 3, "普通でした"),
    (3, 2, 4, "スッキリして飲みやすい"),
    (4, 4, 2, "イマイチでした"),
]
reviews_columns = ["review_id", "product_id", "rating", "comment"]

# ====== 2. SQLite DB作成 ======
conn = sqlite3.connect("dummy_db.db")
pd.DataFrame(sales_data, columns=sales_columns).to_sql("sales_data", conn, index=False, if_exists="replace")
pd.DataFrame(users, columns=users_columns).to_sql("users", conn, index=False, if_exists="replace")
pd.DataFrame(products, columns=products_columns).to_sql("products", conn, index=False, if_exists="replace")
pd.DataFrame(reviews, columns=reviews_columns).to_sql("reviews", conn, index=False, if_exists="replace")
conn.close()
print("dummy_db.db作成完了！")

# ====== 3. RAG用 テーブル定義 ======
table_definitions = [
    {
        "table_name": "sales_data",
        "table_description": "売上データを格納するテーブル。商品カテゴリ、商品名、販売金額、日付を持つ。",
        "columns": [
            {"name": "id", "type": "INTEGER", "description": "レコードID（主キー）"},
            {"name": "category", "type": "TEXT", "description": "商品カテゴリ名"},
            {"name": "product_name", "type": "TEXT", "description": "商品名"},
            {"name": "sales", "type": "INTEGER", "description": "販売金額"},
            {"name": "date", "type": "TEXT", "description": "販売日（YYYY-MM-DD）"}
        ]
    },
    {
        "table_name": "users",
        "table_description": "ユーザー情報を格納するテーブル。ユーザー名、年齢、都道府県が格納される。",
        "columns": [
            {"name": "user_id", "type": "INTEGER", "description": "ユーザーID（主キー）"},
            {"name": "user_name", "type": "TEXT", "description": "ユーザー名"},
            {"name": "age", "type": "INTEGER", "description": "年齢"},
            {"name": "prefecture", "type": "TEXT", "description": "都道府県"}
        ]
    },
    {
        "table_name": "products",
        "table_description": "商品マスタテーブル。商品名、カテゴリ、価格が格納される。",
        "columns": [
            {"name": "product_id", "type": "INTEGER", "description": "商品ID（主キー）"},
            {"name": "product_name", "type": "TEXT", "description": "商品名"},
            {"name": "category", "type": "TEXT", "description": "商品カテゴリ"},
            {"name": "price", "type": "INTEGER", "description": "価格"}
        ]
    },
    {
        "table_name": "reviews",
        "table_description": "商品レビューを格納するテーブル。商品ID、評価（rating）、コメントが格納される。",
        "columns": [
            {"name": "review_id", "type": "INTEGER", "description": "レビューID（主キー）"},
            {"name": "product_id", "type": "INTEGER", "description": "商品ID"},
            {"name": "rating", "type": "INTEGER", "description": "5段階評価（1-5）"},
            {"name": "comment", "type": "TEXT", "description": "レビューコメント"}
        ]
    }
]
with open("table_definitions.json", "w", encoding="utf-8") as f:
    json.dump(table_definitions, f, ensure_ascii=False, indent=2)

# ====== 4. RAG用 クエリ例 ======
query_examples = [
    # sales_data
    {
        "query_example": "各カテゴリごとの合計売上を出してください",
        "sql_example": "SELECT category, SUM(sales) AS total_sales FROM sales_data GROUP BY category;"
    },
    {
        "query_example": "2024年5月1日の商品ごとの売上を教えて",
        "sql_example": "SELECT product_name, sales FROM sales_data WHERE date = '2024-05-01';"
    },
    {
        "query_example": "パンの売上合計を出して",
        "sql_example": "SELECT SUM(sales) FROM sales_data WHERE product_name = 'パン';"
    },
    # users
    {
        "query_example": "全ユーザーの名前と都道府県を一覧表示してください",
        "sql_example": "SELECT user_name, prefecture FROM users;"
    },
    {
        "query_example": "30歳以上のユーザー名を抽出して",
        "sql_example": "SELECT user_name FROM users WHERE age >= 30;"
    },
    # products
    {
        "query_example": "商品の一覧を表示してください",
        "sql_example": "SELECT * FROM products;"
    },
    {
        "query_example": "飲料カテゴリの商品を抽出して",
        "sql_example": "SELECT product_name FROM products WHERE category = '飲料';"
    },
    # reviews
    {
        "query_example": "全商品の平均評価（rating）を計算して",
        "sql_example": "SELECT product_id, AVG(rating) AS avg_rating FROM reviews GROUP BY product_id;"
    },
    {
        "query_example": "評価が5のレビューのコメントを抽出して",
        "sql_example": "SELECT comment FROM reviews WHERE rating = 5;"
    }
]
with open("query_examples.json", "w", encoding="utf-8") as f:
    json.dump(query_examples, f, ensure_ascii=False, indent=2)

print("テーブル定義・クエリ例も保存完了！")
