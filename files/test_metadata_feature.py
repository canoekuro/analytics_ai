"""
Test cases and Python snippets for verifying metadata retrieval functionality
in files/backend_codes.py.
"""

# --- Test Cases ---

test_cases = [
    # Intent Classification Tests
    {
        "test_description": "Intent Classification: Question about table columns",
        "type": "intent_classification",
        "user_input": "sales_dataテーブルにはどんなカラムがありますか？",
        "expected_intent": ["メタデータ検索"],
        "expected_output_keywords": None
    },
    {
        "test_description": "Intent Classification: Question about a specific column's details",
        "type": "intent_classification",
        "user_input": "categoryカラムの情報を教えてください",
        "expected_intent": ["メタデータ検索"],
        "expected_output_keywords": None
    },
    {
        "test_description": "Intent Classification: Regular data query question",
        "type": "intent_classification",
        "user_input": "カテゴリごとの売上を知りたい",
        "expected_intent": ["データ取得"],
        "expected_output_keywords": None
    },
    {
        "test_description": "Intent Classification: General greeting",
        "type": "intent_classification",
        "user_input": "こんにちは",
        "expected_intent": [], # Or based on how LLM handles greetings, might be empty or a specific intent
        "expected_output_keywords": None
    },

    # Metadata Retrieval Node Tests
    # Note: The exact keywords will depend on your LLM's output and the content of your FAISS index.
    # These keywords are illustrative. You'll need to adjust them after observing actual outputs.
    {
        "test_description": "Metadata Retrieval: General question about 'users' table",
        "type": "metadata_retrieval",
        "user_input": "usersテーブルについて教えて",
        "expected_intent": None,
        "expected_output_keywords": ["users", "ユーザー", "ID", "名前", "email", "作成日"] # Example keywords
    },
    {
        "test_description": "Metadata Retrieval: Specific question about 'products' table's 'price' column",
        "type": "metadata_retrieval",
        "user_input": "productsテーブルのpriceカラムは何ですか",
        "expected_intent": None,
        "expected_output_keywords": ["products", "price", "価格", "数値", "金額"] # Example keywords
    },
    {
        "test_description": "Metadata Retrieval: Question about a non-existent table",
        "type": "metadata_retrieval",
        "user_input": "存在しないテーブルについて教えて",
        "expected_intent": None,
        "expected_output_keywords": ["情報が見つかりませんでした", "存在しないようです", "確認してください"] # Example keywords for graceful failure
    },
    {
        "test_description": "Metadata Retrieval: Vague question that might hit some table info",
        "type": "metadata_retrieval",
        "user_input": "顧客データについて", # Assuming 'users' or a similar table might be relevant
        "expected_intent": None,
        "expected_output_keywords": ["顧客", "ユーザー", "情報"] # Example keywords
    }
]

# --- Python Snippets for Manual Verification ---

# IMPORTANT:
# To run these snippets, you need to ensure that:
# 1. The `files.backend_codes` module can be imported. This might mean
#    running the script from the parent directory of `files` or adjusting PYTHONPATH.
# 2. The FAISS indexes ("faiss_tables", "faiss_queries") and the database ("my_data.db")
#    are present in the location expected by `files/backend_codes.py`.
# 3. The GOOGLE_API_KEY environment variable is set.

def run_classification_test(user_input_str):
    """Helper to test classify_intent_node."""
    try:
        from files.backend_codes import classify_intent_node, MyState, llm
        # Initialize llm if it's not already (it's global in backend_codes)
        if llm is None:
            print("LLM not initialized")
            return

        print(f"\n--- Testing classify_intent_node ---")
        initial_state_classify = MyState(input=user_input_str, intent_list=[])
        # Note: classify_intent_node modifies state in place and returns it
        result_state_classify = classify_intent_node(initial_state_classify)
        
        print(f"User Input: \"{user_input_str}\"")
        print(f"Detected Intent(s): {result_state_classify.get('intent_list')}")
        print(f"Condition: {result_state_classify.get('condition')}")
        return result_state_classify.get('intent_list')
    except ImportError:
        print("Could not import from files.backend_codes. Ensure PYTHONPATH is set correctly.")
    except Exception as e:
        print(f"An error occurred: {e}")

def run_metadata_retrieval_test(user_input_str):
    """Helper to test metadata_retrieval_node."""
    try:
        from files.backend_codes import metadata_retrieval_node, MyState, llm, vectorstore_tables
        # Initialize llm and vectorstore_tables if not already
        if llm is None or vectorstore_tables is None:
            print("LLM or vectorstore_tables not initialized")
            return

        print(f"\n--- Testing metadata_retrieval_node ---")
        # metadata_retrieval_node expects 'input' in the state
        initial_state_metadata = MyState(input=user_input_str)
        result_state_metadata = metadata_retrieval_node(initial_state_metadata)

        print(f"User Input: \"{user_input_str}\"")
        print(f"Metadata Answer: {result_state_metadata.get('metadata_answer')}")
        print(f"Condition: {result_state_metadata.get('condition')}")
        return result_state_metadata.get('metadata_answer')
    except ImportError:
        print("Could not import from files.backend_codes. Ensure PYTHONPATH is set correctly.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("--- Running Test Verification Snippets ---")

    # Example Usage:
    
    # Test Case 1: Intent Classification for table columns
    test_case_1_input = "sales_dataテーブルにはどんなカラムがありますか？"
    run_classification_test(test_case_1_input)

    # Test Case 2: Intent Classification for column details
    test_case_2_input = "categoryカラムの情報を教えてください"
    run_classification_test(test_case_2_input)

    # Test Case 3: Intent Classification for regular data query
    test_case_3_input = "カテゴリごとの売上を知りたい"
    run_classification_test(test_case_3_input)
    
    print("\n---------------------------------------\n")
    
    # Test Case 4: Metadata Retrieval for a general table question
    test_case_4_input = "usersテーブルについて教えて" # Assuming 'users' table schema is in your FAISS index
    run_metadata_retrieval_test(test_case_4_input)

    # Test Case 5: Metadata Retrieval for a specific column question
    test_case_5_input = "productsテーブルのpriceカラムは何ですか" # Assuming 'products.price' is in your FAISS index
    run_metadata_retrieval_test(test_case_5_input)

    # You can iterate through the test_cases list for more structured testing:
    print("\n--- Iterating through defined test cases ---")
    for case in test_cases:
        print(f"\nRunning Test: {case['test_description']}")
        if case["type"] == "intent_classification":
            intents = run_classification_test(case["user_input"])
            if intents is not None:
                print(f"Expected Intent: {case['expected_intent']}, Got: {intents}")
                # Add more sophisticated assertion/comparison logic here if needed
                if sorted(intents) == sorted(case['expected_intent']):
                    print("Intent MATCHED expected.")
                else:
                    print("Intent MISMATCHED expected.")
        elif case["type"] == "metadata_retrieval":
            answer = run_metadata_retrieval_test(case["user_input"])
            if answer and case["expected_output_keywords"]:
                found_keywords = [kw for kw in case["expected_output_keywords"] if kw.lower() in answer.lower()]
                print(f"Expected Keywords: {case['expected_output_keywords']}")
                print(f"Found Keywords in Answer: {found_keywords}")
                if len(found_keywords) >= len(case["expected_output_keywords"]) / 2: # Example: at least half the keywords
                     print("Output seems RELEVANT based on keywords.")
                else:
                     print("Output might NOT BE RELEVANT based on keywords.")

    print("\nNote: For metadata retrieval tests, keyword matching is a basic check. "
          "Actual relevance and quality of the LLM's response require manual review.")
    print("Ensure FAISS indexes and GOOGLE_API_KEY are correctly set up before running.")

"""
Instructions for running these snippets:

1.  Save this content as `files/test_metadata_feature.py`.
2.  Ensure your environment is set up to run `files/backend_codes.py` (FAISS files, DB, API keys).
3.  Navigate to the directory containing the `files` directory (e.g., your project root).
4.  Run the script using `python -m files.test_metadata_feature` or `python files/test_metadata_feature.py`
    (the former is often better for handling relative imports within a package-like structure).

You may need to adjust imports in the snippets if your project structure or how you run scripts differs.
The current snippets assume `files/backend_codes.py` can be imported as `files.backend_codes`.
"""
