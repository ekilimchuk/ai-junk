# Install ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install nomic-embed-text
ollama pull nomic-embed-text

# Create python env
mkdir test
cd test
python3 -m venv ./

# Activate env
. bin/activate

# Instal libs from pip
pip install pymilvus ollama

# Run an example
python example.py
