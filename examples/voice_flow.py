import sys
import os
import torch
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    print("Make sure you've built and installed the C++ extensions:")
    print("1. cd flashtensors_grpc_lib")
    print("2. python setup.py build_ext --inplace")
    print("3. python setup.py bdist_wheel")
    print("4. pip install dist/*.whl")
    sys.exit(1)


def main():
    flash.configure(
        storage_path="/workspace",   
        mem_pool_size=1024**3*70,               
        chunk_size=1024**2*32,                   
        num_threads=8,                          
        gpu_memory_utilization=0.8,             
        server_host="0.0.0.0",                
        server_port=8073                        
    )

    whisper_model_id = "openai/whisper-small.en"
    llm_model_id = "Qwen/Qwen3-8B"
    hf_token = ""

    print(f"\n=== Registering and transforming models ===")

    register_model(whisper_model_id, hf_token)
    register_model(llm_model_id, hf_token)

    print(f"\n=== Loading whisper ===")
    load_start_time = time.time()
    processor = WhisperProcessor.from_pretrained(whisper_model_id, token=hf_token)

    # load_start_time = time.time()
    # whisper_model, _ = flash.load_model(
    #    model_id=whisper_model_id,
    #    backend="transformers",
    #    torch_dtype=torch.float16,
    #    device_map="auto",
    #    hf_model_class="WhisperForConditionalGeneration"
    # )
    # load_time = time.time() - load_start_time

    # print(f"✅ Model loaded: {whisper_model_id}")
    # print(f"   Load time: {load_time:.2f}s")

    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
    whisper_model.to("cuda")
    load_time = time.time() - load_start_time

    print(f"✅ Model loaded: {whisper_model_id}")
    print(f"   Load time: {load_time:.2f}s")

    audio, sr = librosa.load("/root/flashtensors/examples/recorded_audio.mp3", sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            input_features
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    load_start_time = time.time()
    llm_model, tokenizer = flash.load_model(
        model_id=llm_model_id,
        backend="transformers",
        torch_dtype=torch.float16,
        device_map="auto",
        hf_model_class="AutoModelForCausalLM"
    )
    load_time = time.time() - load_start_time

    print(f"✅ Model loaded: {llm_model_id}")
    print(f"   Load time: {load_time:.2f}s")

    inputs = tokenizer(transcription, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 50,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")


def register_model(model_id, hf_token):
    result = flash.register_model(
        model_id=model_id,
        backend="transformers", 
        torch_dtype="float16",
        force=False,  
        hf_token=hf_token  
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
