"""
Entry point for running TeilEngine modules.
"""
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m flashtensors <module>")
        print("Available modules:")
        print("  storage_server - Run the gRPC storage server")
        sys.exit(1)
    
    module = sys.argv[1]
    
    if module == "storage_server":
        # Remove the module argument and run storage server
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .storage_server import main as storage_main
        storage_main()
    else:
        print(f"Unknown module: {module}")
        sys.exit(1)

if __name__ == "__main__":
    main()