import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score, classification_report
from data_processor import BookIndexer
from reasoner import StoryValidator

# --- CONFIGURATION ---
DATASET_DIR = "./data"
BOOKS_DIR = os.path.join(DATASET_DIR, "Books")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
OUTPUT_FILE = "validation_results.csv"

# Set this to None to run on all rows, or an integer (e.g., 5) for a quick test
SAMPLE_LIMIT = 5 

def main():
    print("--- Starting Validation Run ---")

    # 1. Initialize Pathway Indexer
    if not os.path.exists(BOOKS_DIR):
        print(f"Error: Books folder not found at {BOOKS_DIR}")
        return

    print("Building Index from Books...")
    indexer = BookIndexer(BOOKS_DIR)
    indexer.build_index()
    
    # 2. Setup Logic Engine
    validator = StoryValidator(retrieval_func=indexer.search)
    
    # 3. Read Training Data
    if not os.path.exists(TRAIN_CSV):
        print("Error: train.csv not found.")
        return
        
    df_train = pd.read_csv(TRAIN_CSV)
    
    # Optional: Run on a small sample first to save API costs/time
    if SAMPLE_LIMIT:
        print(f"Running on first {SAMPLE_LIMIT} rows only for testing...")
        df_train = df_train.head(SAMPLE_LIMIT)
    
    print(f"Validating on {len(df_train)} examples.")

    results = []
    y_true = []
    y_pred = []

    # 4. Validation Loop
    for idx, row in df_train.iterrows():
        story_id = row['id']
        book_name = row['book_name']
        backstory = str(row['content']) # Using 'content' as the backstory text
        ground_truth = int(row['label']) # Ensure label is integer (0 or 1)
        
        print(f"Processing ID {story_id} (Book: {book_name})...")
        
        # Run the Pipeline
        # prediction is 0 or 1, rationale is text
        prediction, rationale = validator.process_row(backstory, book_name)
        
        # Store for Metrics
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        # Check if correct
        is_correct = (prediction == ground_truth)
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"  -> Truth: {ground_truth} | Pred: {prediction} | {status}")

        results.append({
            "Story ID": story_id,
            "Book": book_name,
            "Ground Truth": ground_truth,
            "Prediction": prediction,
            "Correct": is_correct,
            "Rationale": rationale
        })
        
        # Rate limit safety for API
        time.sleep(1) 

    # 5. Calculate & Print Metrics
    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print("="*40)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Inconsistent (0)", "Consistent (1)"]))

    # 6. Save Detailed Results
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()