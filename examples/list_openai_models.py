#!/usr/bin/env python3
"""
List Available OpenAI Models

This script connects to OpenAI API and lists all models available
with your API key, including new models like GPT-5, o1, etc.

Usage:
    python examples/list_openai_models.py
"""

from openai import OpenAI
from getpass import getpass
import sys


def list_models(api_key: str):
    """List all OpenAI models available with the API key."""

    try:
        client = OpenAI(api_key=api_key)

        print("=" * 70)
        print("OpenAI Models Available with Your API Key")
        print("=" * 70)

        # Get all models
        models = client.models.list()

        # Filter for chat/generation models (gpt, o1, etc.)
        chat_models = []
        embedding_models = []
        other_models = []

        for model in models.data:
            model_id = model.id

            if any(model_id.startswith(p) for p in ["gpt-", "o1-", "chatgpt"]):
                chat_models.append(model)
            elif "embedding" in model_id:
                embedding_models.append(model)
            else:
                other_models.append(model)

        # Display chat models
        print("\n🤖 Chat/Generation Models:")
        print("-" * 70)
        if chat_models:
            for model in sorted(chat_models, key=lambda x: x.id):
                print(f"  • {model.id}")
        else:
            print("  (none found)")

        print(f"\nTotal chat models: {len(chat_models)}")

        # Display embedding models
        print("\n📊 Embedding Models:")
        print("-" * 70)
        if embedding_models:
            for model in sorted(embedding_models, key=lambda x: x.id):
                print(f"  • {model.id}")
        else:
            print("  (none found)")

        # Check for specific models
        print("\n" + "=" * 70)
        print("Model Availability Check")
        print("=" * 70)

        models_to_check = [
            "gpt-5",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1-preview",
            "o1-mini",
            "chatgpt-4o-latest",
        ]

        all_model_ids = {m.id for m in models.data}

        print()
        for model_name in models_to_check:
            # Check exact match
            if model_name in all_model_ids:
                print(f"  ✓ {model_name:<25} AVAILABLE")
            else:
                # Check if any model starts with this name
                matches = [m for m in all_model_ids if m.startswith(model_name)]
                if matches:
                    print(f"  ~ {model_name:<25} Similar: {', '.join(matches[:2])}")
                else:
                    print(f"  ✗ {model_name:<25} NOT AVAILABLE")

        # Recommendations
        print("\n" + "=" * 70)
        print("Recommendations for CERT Baseline Measurement")
        print("=" * 70)
        print()

        # Find the best available chat model
        priority_models = [
            "gpt-5",
            "o1-preview",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4o-mini",
        ]

        recommended = None
        for model in priority_models:
            if model in all_model_ids:
                recommended = model
                break
            # Check for variants
            matches = [m for m in all_model_ids if m.startswith(model)]
            if matches:
                recommended = matches[0]
                break

        if recommended:
            print(f"✓ Recommended model for measurement: {recommended}")
            print(f"\nUse this in the notebook:")
            print(f'  MODEL_NAME = "{recommended}"')
        else:
            print("⚠ No recommended models found. Using any gpt-4 variant should work.")

        print("\n" + "=" * 70)

        return all_model_ids

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. Network connectivity")
        print("  3. OpenAI API service unavailable")
        return None


def main():
    """Main entry point."""
    print("=" * 70)
    print("OpenAI Model Lister")
    print("=" * 70)
    print("\nThis script will list all models available with your API key.")
    print("Including new models like GPT-5, o1-preview, etc.\n")

    # Get API key
    api_key = getpass("Enter your OpenAI API key: ")

    if not api_key:
        print("❌ Error: API key is required")
        sys.exit(1)

    print("\n🔍 Fetching available models...")

    models = list_models(api_key)

    if models:
        print(f"\n✓ Found {len(models)} total models")
        print("\nYou can now use any of the chat models listed above")
        print("in your CERT baseline measurement script.")
    else:
        print("\n❌ Failed to fetch models")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
