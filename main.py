import os
import sys
from dotenv import load_dotenv
from groq import Groq
from data_ingestion import NovelIngestionPipeline
from validator import run_validation

# Load environment variables from .env file
load_dotenv()

def main():
    print("🚀 Starting Backstory Consistency Checker...")

    # 1. Setup Client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables.")
        print("   Please create a .env file with GROQ_API_KEY=your_key_here")
        sys.exit(1)

    client = Groq(api_key=api_key)
    
    # Configuration
    MODEL_NAME = "llama3-70b-8192" 
    DATA_DIR = os.path.join("Dataset", "Books")
    
    # 2. Ingest Data
    print(f"📚 Indexing books from: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
         print(f"❌ Error: Directory {DATA_DIR} does not exist.")
         sys.exit(1)

    pipeline = NovelIngestionPipeline(DATA_DIR)
    index_df = pipeline.run_indexing()
    
    if index_df.empty:
        print("⚠️ Warning: Index is empty. Check your dataset.")
        sys.exit(1)

    # 3. Run Validation
    print("🔍 Running validation pipeline...")
    # You can adjust the limit here (e.g., limit=50 or limit=None for all)
    results = run_validation(client, MODEL_NAME, index_df, limit=10)
    
    # 4. Save/Show Results
    if not results.empty:
        print("\n✅ Validation Complete!")
        print(results.head())
        output_file = "results.csv"
        results.to_csv(output_file, index=False)
        print(f"💾 Results saved to {output_file}")
    else:
        print("⚠️ No results generated.")

if __name__ == "__main__":
    main()
