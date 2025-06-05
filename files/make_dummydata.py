import sqlite3
import pandas as pd
from io import StringIO

# データベースファイル名
DB_FILE = "dummy_store.db"

def create_tables(conn):
    cursor = conn.cursor()

    # productsテーブル
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        category TEXT,
        unit_price INTEGER,
        stock_quantity INTEGER
    )
    """)

    # salesテーブル
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sale_date TEXT,
        product_id INTEGER,
        customer_id INTEGER,
        quantity INTEGER,
        sales_amount INTEGER,
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

    # customersテーブル
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        prefecture TEXT,
        email TEXT,
        status TEXT,
        registration_date TEXT
    )
    """)
    conn.commit()
    print("テーブルが作成されました（または既に存在します）。")

def insert_dummy_data(conn):
    cursor = conn.cursor()

    # productsデータ
    products_data = [
        (1, '高性能ラップトップ', 'エレクトロニクス', 150000, 50),
        (2, 'オーガニックコーヒー豆', '食品', 2500, 200),
        (3, 'デザインTシャツ', 'アパレル', 4000, 150),
        (4, 'AI技術解説書', '書籍', 3200, 80),
        (5, 'ワイヤレスイヤホン', 'エレクトロニクス', 12000, 120),
        (6, 'クラフトビールセット', '食品', 4500, 70),
        (7, 'スニーカーXモデル', 'アパレル', 18000, 90)
    ]
    cursor.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)", products_data)

    # customersデータ
    customers_data = [
        (101, '山田 太郎', '東京都', 'yamada@example.com', 'active', '2023-01-15'),
        (102, '佐藤 花子', '大阪府', 'sato@example.com', 'active', '2023-03-22'),
        (103, '田中 一郎', '福岡県', 'tanaka@example.com', 'inactive', '2022-11-01'),
        (104, '鈴木 美紀', '東京都', 'suzuki@example.com', 'active', '2024-02-10'),
        (105, '高橋 大輔', '北海道', 'takahashi@example.com', 'pending', '2024-05-01')
    ]
    cursor.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?)", customers_data)

    # salesデータ (現在の日付を考慮して調整)
    # 今日が2025-06-05と仮定して、過去1ヶ月強のデータを作成
    sales_data = [
        (1001, '2025-05-01', 1, 101, 1, 150000),
        (1002, '2025-05-03', 2, 102, 2, 5000),
        (1003, '2025-05-05', 3, 101, 1, 4000),
        (1004, '2025-05-10', 5, 104, 1, 12000),
        (1005, '2025-05-15', 1, 102, 1, 150000),
        (1006, '2025-05-20', 4, 101, 1, 3200),
        (1007, '2025-05-25', 6, 103, 1, 4500), # inactive customer
        (1008, '2025-06-01', 2, 104, 3, 7500),
        (1009, '2025-06-03', 7, 101, 1, 18000)
    ]
    cursor.executemany("INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?,?)", sales_data)

    conn.commit()
    print("ダミーデータが挿入されました（または既に存在します）。")

def verify_data(conn):
    print("\n--- productsテーブル (最初の3件) ---")
    print(pd.read_sql_query("SELECT * FROM products LIMIT 3", conn))
    print("\n--- salesテーブル (最初の3件) ---")
    print(pd.read_sql_query("SELECT * FROM sales LIMIT 3", conn))
    print("\n--- customersテーブル (最初の3件) ---")
    print(pd.read_sql_query("SELECT * FROM customers LIMIT 3", conn))

if __name__ == "__main__":
    conn = sqlite3.connect(DB_FILE)
    create_tables(conn)
    insert_dummy_data(conn)
    verify_data(conn)
    conn.close()
    print(f"\nデータベース '{DB_FILE}' の準備が完了しました。")