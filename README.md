# Embedding-Security


Embedding Inversion Simulation
This script demonstrates how a vector embedding, which may seem anonymous, can be reverse-engineered to reconstruct the original sensitive data it represents. It uses a real sentence-transformer model to generate embeddings and a text generation model (GPT-2) to simulate the reconstruction attack.

Requirements
Python 3.7+

pip (Python package installer)

Installation
Clone or download the repository/script.

Navigate to the script's directory in your terminal.

Install the required Python libraries using the provided requirements.txt file. Run the following command:

pip install -r requirements.txt

This will install all necessary packages, including numpy, torch, sentence-transformers, transformers, and scipy.

Running the Simulation
Once the installation is complete, you can run the simulation script directly from your terminal:

python3 simulation_embedding.py

The script will then execute the simulation, printing the original secret, the generated "anonymous" vector, the discovered semantic keywords, and the final reconstructed text to your console.

###########################################################





Second script demonstrate  Data Poisoning the AI's knowledge base to skew its output or inject bias.	Uploading malicious documents or compromising data feeds before they are vectorized.


python3  rag_poisoning.py 
