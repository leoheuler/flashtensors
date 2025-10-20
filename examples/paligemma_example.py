import sys
import os
from vllm import SamplingParams
from PIL import Image


# Add the flashtensors directory to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    print("Make sure you've built and installed the C++ extensions:")
    print("1. cd flashtensors")
    print("2. python setup.py build_ext --inplace")
    print("3. python setup.py bdist_wheel")
    print("4. pip install dist/*.whl")
    sys.exit(1)

def main():
    flash.shutdown_server()  # Ensure any existing server is shut down

    flash.configure(
        storage_path="/tmp/models",   # Where models will be stored
        mem_pool_size=1024**3*30,                # 30GB memory pool (GPU Size)
        chunk_size=1024**2*32,                   # 32MB chunks
        num_threads=4,                           # Number of threads
        gpu_memory_utilization=0.8,             # Use 80% of GPU memory
        server_host="0.0.0.0",                # gRPC server host
        server_port=8073                        # gRPC server port
    )

    model_id = "google/paligemma-3b-mix-224"
    hf_token = ""

    flash.activate_vllm_integration()

    print(f"\n🔄 Transforming model {model_id}...")
    result = flash.register_model(
        model_id=model_id,
        backend="vllm", # We should have an "auto" backend option
        torch_dtype="float16",
        force=False,  # Don't overwrite if already exists
        hf_token=hf_token  
    )

    print(f"✅ Model transformation completed:")
    print(f"   Status: {result['status']}")
    print(f"   Path: {result.get('path', 'N/A')}")

    if "metrics" in result:
        metrics = result["metrics"]
        print(f"   Download time: {metrics['download_time']:.2f}s")
        print(f"   Transform time: {metrics['transform_time']:.2f}s") 
        print(f"   Total time: {metrics['total_time']:.2f}s")
        print(f"   Model size: {metrics['model_size'] / (1024**3):.2f} GB")

    print(f"\n⚡ Loading model {model_id} with fast loading...")

    import time
    load_start_time = time.time()

    llm = flash.load_model(
        model_id=model_id,
        backend="vllm",
        dtype="float16",
        gpu_memory_utilization=0.8
    )

    load_time = time.time() - load_start_time
    print(f"✅ Model loaded successfully with fast loading in {load_time:.2f}s")

    print("\n🤖 Running inference...")
    prompt = "Is the person speaking?"

    img_url = "https://i.imgur.com/NYFN475.png"
    image = Image.open(img_url)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    })

    print("📝 Generated outputs:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    print("🧹 Cleaning up GPU memory...")
    flash.cleanup_gpu()
    print("✅ GPU memory cleaned")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        flash.shutdown_server()
