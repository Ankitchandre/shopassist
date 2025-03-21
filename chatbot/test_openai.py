import openai

# Read API key
with open("OpenAI_API_Key.txt", 'r') as f:
    api_key = f.read().strip()

# Set API key
openai.api_key = api_key

# Test a simple API call
try:
    models = openai.models.list()
    print("API connection successful!")
    print(f"Available models: {[model.id for model in models.data[:5]]}")  # Print first 5 models
except Exception as e:
    print(f"Error: {e}") 