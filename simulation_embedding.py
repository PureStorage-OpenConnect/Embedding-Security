import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.spatial.distance import cosine

# --- Step 1: The Embedding Model (The one you use) ---
# This is the model that creates the vectors for your database.
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- Step 2: The Inversion Model (The one the attacker builds) ---
# This is a text generation model that the attacker will use to reconstruct the text.
inversion_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
inversion_model = GPT2LMHeadModel.from_pretrained('gpt2')

# --- Attacker's Simulated Knowledge Base ---
# In a real attack, the attacker's model is trained on a vast vocabulary.
# We simulate this with a smaller vocabulary of relevant words.
VOCABULARY = [
    'employee', 'salary', 'performance', 'review', 'confidential', 'project',
    'launch', 'secret', 'formula', 'ingredient', 'memo', 'human resources',
    'finances', 'budget', 'John Doe', 'Tuesday', 'Friday', '$150,000',
    'release', 'october', 'Major', 'Comparision','2026,' '2025', 'AI', 'V2', 'V1', 'set'
]
# The attacker would pre-compute embeddings for their entire vocabulary
VOCABULARY_EMBEDDINGS = embedding_model.encode(VOCABULARY)


def create_embedding(text: str) -> np.ndarray:
    """
    Creates a fixed-size vector embedding from a string using a real model.
    """
    print("--- Generating Embedding ---")
    vector = embedding_model.encode(text)
    print(f"Generated 'Anonymous' Vector ({vector.shape[0]} dimensions):")
    print(f"[{', '.join(f'{x:.4f}' for x in vector[:4])}, ...]")
    return vector

def reconstruct_from_embedding(stolen_vector: np.ndarray) -> str:
    """
    Simulates a true inversion attack by using the vector's semantic
    meaning to guide a text generation model.
    """
    print("\n--- Reconstructing Secret from Stolen Vector ---")

    # 1. Attacker finds words semantically similar to the stolen vector
    similarities = [1 - cosine(stolen_vector, vocab_vec) for vocab_vec in VOCABULARY_EMBEDDINGS]

    # 2. Attacker selects the most relevant words based on similarity score
    top_indices = np.argsort(similarities)[-8:] # Get top 8 most similar words
    prompt_keywords = [VOCABULARY[i] for i in top_indices]

    print(f"Discovered semantic keywords from vector: {', '.join(prompt_keywords)}")

    # 3. These keywords are used to create a guiding prompt for the inversion model
    prompt_text = f"Reconstruct the confidential project update. The key topics are: {', '.join(prompt_keywords)}."

    # 4. Generate text with the GPT-2 model, now guided by the vector's meaning.
    inputs = inversion_tokenizer.encode(prompt_text, return_tensors='pt')
    outputs = inversion_model.generate(
        inputs,
        max_length=60,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        do_sample=True
    )
    reconstructed_text = inversion_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # The output won't be a perfect copy, but it will be semantically similar,
    # demonstrating that the core confidential information can be leaked.
    return reconstructed_text

# --- Main Simulation ---
if __name__ == "__main__":
    # 1. Define a piece of sensitive, original data
    original_secret = "The project AI_v2_Titan is all set to release on 25th october 2025"
    print(f"Original Secret Data:\n'{original_secret}'\n")

    # 2. Generate an embedding from the secret data
    secret_vector = create_embedding(original_secret)

    # 3. Simulate the attack using only the stolen vector
    reconstructed_text = reconstruct_from_embedding(secret_vector)

    print("\nReconstructed Data (from vector only):")
    print(f"'{reconstructed_text}'")
    print("\n--- Conclusion ---")
    print("This simulation shows that even without database access, a separate")
    print("model can interpret a stolen vector's meaning to reconstruct a semantically similar and highly sensitive version of the original text.")
