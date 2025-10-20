import time
import scipy.io.wavfile as wavfile
import numpy as np
import torch
from transformers import BarkModel, AutoProcessor

def main():
    bark_model_id = "suno/bark-small"
    hf_token = ""

    print(f"\n=== Loading Bark Model ===")
    load_start_time = time.time()

    print(f"Loading {bark_model_id}...")
    bark_model = BarkModel.from_pretrained(
        bark_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        token=hf_token
    )
    
    processor = AutoProcessor.from_pretrained(
        bark_model_id,
        token=hf_token
    )

    load_time = time.time() - load_start_time
    print(f"✅ Model loaded: {bark_model_id}")
    print(f"   Load time: {load_time:.2f}s")

    text_prompt = "Hello! This is a test of the Bark text to speech system. How does this sound?"
    inputs = processor(text_prompt, return_tensors="pt").to("cuda")

    print(f"🗣️  Generating speech for: '{text_prompt}'")

    with torch.no_grad():
        audio_array = bark_model.generate(
            **inputs, 
            do_sample=True,
            temperature=0.1
        )

    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = bark_model.generation_config.sample_rate
    
    output_file = "bark_test_output.wav"
    wavfile.write(output_file, sample_rate, (audio_array * 32767).astype(np.int16))

    print(f"✅ Audio saved as: {output_file}")
    print(f"🎧 Sample rate: {sample_rate} Hz")
    print(f"⏱️  Audio duration: {len(audio_array) / sample_rate:.2f} seconds")

    print("🧹 Cleaning up GPU memory...")
    del bark_model, processor
    torch.cuda.empty_cache()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
