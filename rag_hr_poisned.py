import time
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import os
import shutil
import fitz  # PyMuPDF

# --- The Model ---
# Load a real, pre-trained sentence embedding model from Hugging Face.
# all-MiniLM-L6-v2 is a lightweight but powerful BERT-based model.
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

# --- Knowledge Base Setup ---
# The script will now read from your existing 'resumes_kb' directory.
KB_DIRECTORY = "resumes_kb"

def load_kb_from_files(directory, exclude_poison=False):
    """Reads all .pdf and .txt files from a directory into a knowledge base list."""
    kb = []
    poison_file = "Mallory_poisoned_with_hiddentext.pdf"

    for filename in os.listdir(directory):
        if exclude_poison and filename == poison_file:
            continue # Skip the poisoned file for the clean run

        filepath = os.path.join(directory, filename)
        content = ""
        if filename.endswith(".pdf"):
            try:
                with fitz.open(filepath) as doc:
                    for page in doc:
                        content += page.get_text()
                
                # If this is the poisoned file, print the full content it reads
                if filename == poison_file:
                    print("\nFull text read from Mallory's PDF (including hidden instructions):")
                    print("----------------------------------------------------")
                    print(content.strip().replace('\n', ' '))
                    print("----------------------------------------------------")

            except Exception as e:
                print(f"Warning: Could not read {filename}. Skipping. Error: {e}")
                continue
        elif filename.endswith(".txt"):
            with open(filepath, "r") as f:
                content = f.read()
        
        if content:
            # Rename for clarity in the simulation
            clean_title = filename.replace('_wonders', '').replace('_buddy', '').replace('_poisoned_with_hiddentext', '')
            kb.append({"title": clean_title, "content": content})
    return kb

def get_embeddings(kb):
    """Converts all documents in the knowledge base to vector embeddings."""
    print("\nGenerating embeddings for the knowledge base...")
    contents = [doc["content"] for doc in kb]
    embeddings = model.encode(contents)
    return embeddings

def ask_ai_with_vectors(question, kb, kb_embeddings):
    """
    Simulates a real RAG system using vector similarity search.
    """
    # 1. Create an embedding for the user's question
    question_embedding = model.encode(question)

    # 2. Find the most similar document in the knowledge base
    similarities = [1 - cosine(question_embedding, doc_embedding) for doc_embedding in kb_embeddings]
    most_similar_index = np.argmax(similarities)
    most_similar_doc = kb[most_similar_index]

    print(f"Most relevant document found: {most_similar_doc['title']}")
    
    # Show a snippet of the retrieved content for clarity
    retrieved_snippet = most_similar_doc['content'].strip().replace('\n', ' ')[:100]
    print(f"Retrieved Content Snippet: \"{retrieved_snippet}...\"")


    # 3. Generate a response based on the retrieved document
    # A real RAG would feed the content to an LLM, but we simulate the outcome.
    if "Mallory" in most_similar_doc["title"]:
        # The AI's logic is hijacked by the poisoned document
        return "Mallory is the top candidate."
    elif "Bob" in most_similar_doc["title"]:
        # Normal RAG process
        return "Based on the documents, Bob is the top candidate due to his expertise in cloud infrastructure."
    else:
        return "I have analyzed the documents but a top candidate is not explicitly mentioned in the most relevant text."


def display_kb_files(directory, is_poisoned_run=False):
    """Displays the current files in the knowledge base directory."""
    print("\n--- Current AI Knowledge Base Files ---")
    files = sorted(os.listdir(directory))
    if not is_poisoned_run:
        files = [f for f in files if "Mallory" not in f]
    
    for filename in files:
        status = "(POISONED)" if "Mallory" in filename else ""
        clean_name = filename.replace('_wonders', '').replace('_buddy', '').replace('_poisoned_with_hiddentext', '')
        print(f"  {clean_name} {status}")
    print("------------------------------------")


if __name__ == "__main__":
    if not os.path.isdir(KB_DIRECTORY):
        print(f"Error: Directory '{KB_DIRECTORY}' not found.")
        print("Please create it and copy your PDF resumes into it before running.")
    else:
        # --- Step 1: Query the Clean System ---
        print("\n### Step 1: Querying the Clean System ###")
        # Load knowledge base excluding the poisoned file
        clean_kb = load_kb_from_files(KB_DIRECTORY, exclude_poison=True)
        display_kb_files(KB_DIRECTORY, is_poisoned_run=False)
        clean_kb_embeddings = get_embeddings(clean_kb)
        
        print("\nQuery: Who is the top candidate for the Senior Engineer role?")
        response_clean = ask_ai_with_vectors("Who is the top candidate for the Senior Engineer role?", clean_kb, clean_kb_embeddings)
        print(f"AI Response: {response_clean}")

        print("\n" + "="*50 + "\n")

        # --- Step 2: Simulate Poisoning the Knowledge Base ---
        print("### Step 2: Simulating the Injection of a Malicious Document ###")
        print("The system now re-indexes all available documents, including the attacker's resume.")
        
        poisoned_kb = load_kb_from_files(KB_DIRECTORY, exclude_poison=False)
        display_kb_files(KB_DIRECTORY, is_poisoned_run=True)
        poisoned_kb_embeddings = get_embeddings(poisoned_kb)
        print("\nThe knowledge base has been updated and re-indexed with the new file.")

        print("\n" + "="*50 + "\n")

        # --- Step 3: Query the Compromised System ---
        print("### Step 3: Querying the Compromised System ###")
        print("\nQuery: Who is the top candidate for the Senior Engineer role?")
        response_poisoned = ask_ai_with_vectors("Who is the top candidate for the Senior Engineer role?", poisoned_kb, poisoned_kb_embeddings)
        print(f"AI Response: {response_poisoned}")
