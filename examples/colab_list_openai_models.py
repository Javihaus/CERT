"""
Google Colab Cell: List Available OpenAI Models

Copy-paste this entire cell into Google Colab to see what models
are available with your API key.
"""

# Install OpenAI library if needed
try:
    import openai
except ImportError:
    print("Installing OpenAI library...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "openai"])
    import openai

from openai import OpenAI
from getpass import getpass

# Get API key
api_key = getpass("Enter your OpenAI API key: ")

# Initialize client
client = OpenAI(api_key=api_key)

print("=" * 70)
print("OpenAI Models Available with Your API Key")
print("=" * 70)

# Fetch all models
try:
    models = client.models.list()

    # Categorize models
    chat_models = []
    embedding_models = []

    for model in models.data:
        model_id = model.id
        if any(model_id.startswith(p) for p in ["gpt-", "o1-", "chatgpt"]):
            chat_models.append(model_id)
        elif "embedding" in model_id:
            embedding_models.append(model_id)

    # Display chat models
    print("\n🤖 Chat/Generation Models (use these for CERT):")
    print("-" * 70)
    for model_id in sorted(chat_models):
        print(f"  • {model_id}")

    print(f"\nTotal: {len(chat_models)} chat models")

    # Check for specific models
    print("\n" + "=" * 70)
    print("Model Availability Check")
    print("=" * 70)

    models_to_check = [
        ("gpt-5", "Latest GPT-5 model (if released)"),
        ("gpt-4o", "GPT-4 Optimized"),
        ("gpt-4o-mini", "Smaller GPT-4o variant"),
        ("o1-preview", "O1 reasoning model preview"),
        ("o1-mini", "Smaller O1 variant"),
        ("chatgpt-4o-latest", "Latest ChatGPT-4o"),
    ]

    all_model_ids = {m.id for m in models.data}

    print()
    available_models = []
    for model_name, description in models_to_check:
        if model_name in all_model_ids:
            print(f"  ✓ {model_name:<25} {description}")
            available_models.append(model_name)
        else:
            # Check for similar
            matches = [m for m in all_model_ids if m.startswith(model_name)]
            if matches:
                print(f"  ~ {model_name:<25} Similar: {matches[0]}")
                available_models.append(matches[0])
            else:
                print(f"  ✗ {model_name:<25} NOT AVAILABLE")

    # Recommendation
    print("\n" + "=" * 70)
    print("Recommended Model for CERT Measurement")
    print("=" * 70)

    if available_models:
        recommended = available_models[0]  # First available from priority list
        print(f"\n✓ Use: {recommended}")
        print(f"\nUpdate your notebook:")
        print(f'  MODEL_NAME = "{recommended}"')
        print(f'  MODEL_FAMILY = "{recommended.upper()}"')
    else:
        print("\n⚠ No priority models found")
        print(f"Use any of the {len(chat_models)} chat models listed above")

    print("\n" + "=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPossible issues:")
    print("  • Invalid API key")
    print("  • Network connectivity")
    print("  • OpenAI API service unavailable")
