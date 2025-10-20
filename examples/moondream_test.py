import sys
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

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
    flash.configure(
        storage_path="/workspace",
        mem_pool_size=1024**3 * 70,
        chunk_size=1024**2 * 32,
        num_threads=8,
        gpu_memory_utilization=0.8,
        server_host="0.0.0.0",
        server_port=8073,
    )

    model_id = "vikhyatk/moondream2"
    hf_token = ""
    register_model(model_id, hf_token)

    load_start_time = time.time()
    model, tokenizer = flash.load_model(
        model_id=model_id,
        backend="transformers",
        torch_dtype=torch.float16,
        device_map="auto",
        hf_model_class="AutoModelForCausalLM",
    )
    load_time = time.time() - load_start_time

    image = Image.open("/root/flashtensors/examples/image.jpg")

    print("Short caption:")
    print(model.caption(image, length="short")["caption"])

    print("\nNormal caption:")
    for t in model.caption(image, length="normal", stream=True)["caption"]:
        # Streaming generation example, supported for caption() and detect()
        print(t, end="", flush=True)
    print(model.caption(image, length="normal"))

    # Visual Querying
    print("\nVisual query: 'How many people are in the image?'")
    print(model.query(image, "How many people are in the image?")["answer"])

    # Object Detection
    print("\nObject detection: 'face'")
    objects = model.detect(image, "face")["objects"]
    print(f"Found {len(objects)} face(s)")

    # Pointing
    print("\nPointing: 'person'")
    points = model.point(image, "person")["points"]
    print(f"Found {len(points)} person(s)")


def register_model(model_id, hf_token):
    result = flash.register_model(
        model_id=model_id,
        backend="transformers",
        torch_dtype="float16",
        force=False,
        hf_token=hf_token,
    )

    print(f"✅ Transformation completed: {model_id}")
    print(f"   Status: {result['status']}")
    print(f"   Path: {result.get('path', 'N/A')}")

    if "metrics" in result:
        metrics = result["metrics"]
        print(f"   Download time: {metrics['download_time']:.2f}s")
        print(f"   Transform time: {metrics['transform_time']:.2f}s")
        print(f"   Total time: {metrics['total_time']:.2f}s")
        print(f"   Model size: {metrics['model_size'] / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
