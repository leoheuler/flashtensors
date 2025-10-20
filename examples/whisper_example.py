import sys
import time
import torch
import librosa

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
    flash.shutdown_server()
    flash.configure(
        storage_path="/workspace",
        mem_pool_size=1024**3 * 30,
        chunk_size=1024**2 * 32,
        num_threads=4,
        gpu_memory_utilization=0.8,
        server_host="0.0.0.0",
        server_port=8073,
    )

    hf_token = ""
    whisper_model_id = "openai/whisper-small.en"

    print(f"\n=== Registering and transforming whisper ===")
    register_model(whisper_model_id, hf_token)

    print(f"\n=== Loading whisper ===")
    load_start_time = time.time()

    whisper_model, processor = flash.load_model(
        model_id=whisper_model_id,
        backend="whisper",
        torch_dtype=torch.float32,
        device_map="auto",
        hf_model_class="WhisperForConditionalGeneration",
    )
    load_time = time.time() - load_start_time

    print(f"✅ Model loaded: {whisper_model_id}")
    print(f"   Load time: {load_time:.2f}s")

    audio, sr = librosa.load("/root/flashtensors/examples/recorded_audio.mp3", sr=16000)
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda")

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"✅ Transcription completed")
    print(f"   Transcription: {transcription}")

    print("🧹 Cleaning up GPU memory & shutting down server...")
    flash.cleanup_gpu()
    flash.shutdown_server()
    print("✅ Server shut down")


def register_model(model_id, hf_token):
    result = flash.register_model(
        model_id=model_id,
        backend="whisper",
        torch_dtype="float32",
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
