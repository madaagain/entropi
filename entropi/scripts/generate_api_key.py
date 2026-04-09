import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from entropi.db.api_keys import create_api_key


def main():
    parser = argparse.ArgumentParser(description="Generate a new Entropi API key")
    parser.add_argument("--name", required=True, help="Name for this API key")
    args = parser.parse_args()

    api_key = create_api_key(args.name)
    print(f"\nAPI key generated for '{args.name}':")
    print(f"  {api_key}")
    print("\nSave this key — it won't be shown again.\n")


if __name__ == "__main__":
    main()
