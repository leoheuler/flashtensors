import sys
import os
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
import numpy as np
import sounddevice as sd
import torch
import librosa
from collections import deque
from transformers import BarkModel, AutoProcessor
import gc

# Add flashtensors path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    sys.exit(1)


class VoiceAgent:
    def __init__(self):
        self.hf_token = ""

        # Models
        self.whisper_model_id = "openai/whisper-small.en"
        self.llm_model_id = "Qwen/Qwen3-1.7B"
        self.bark_model_id = "suno/bark-small"

        # State management
        self.state = "IDLE"
        self.current_model = None
        self.models_registered = False
        self.interrupt_requested = False

        # OPTIMIZATION: More aggressive audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.2  # 200ms chunks (faster response)
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = 0.008  # More sensitive
        self.silence_duration = 0.6  # Faster cutoff

        # OPTIMIZATION: Smaller buffers for lower latency
        self.audio_buffer = deque(maxlen=int(3 * self.sample_rate))  # 3 second buffer
        self.silence_counter = 0

        # OPTIMIZATION: Streaming text generation
        self.text_chunk_size = 20  # Smaller chunks for faster TTS
        self.max_response_length = 120  # Shorter responses

        # Threading and queues for parallel processing
        self.audio_queue = queue.Queue(maxsize=10)
        self.tts_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.tts_thread = None
        self.preload_thread = None

        # Model preloading
        self.next_model_hint = None

        print("🚀 Optimized Voice Agent initialized")

    def setup_flashtensors(self):
        """Configure flashtensors with optimized settings for Jetson Nano"""
        print("⚡ Configuring flashtensors for maximum performance...")

        flash.configure(
            storage_path="/workspace",
            mem_pool_size=1024**3 * 3.5,  # 3.5GB
            chunk_size=1024**2 * 8,  # 8MB chunks (smaller for faster loading)
            num_threads=2,  # Optimize for Jetson's 4 cores
            gpu_memory_utilization=0.65,  # More conservative to avoid OOM
            server_host="0.0.0.0",
            server_port=8073,
        )
        print("✅ flashtensors optimized")

    def register_models(self):
        """Register models with parallel downloads where possible"""
        if self.models_registered:
            return

        print("📦 Parallel model registration...")

        # Start Bark download in background thread
        bark_thread = threading.Thread(target=self.register_bark_model)
        bark_thread.start()

        # Register flashtensors models in parallel
        flash_models = [
            (self.whisper_model_id, "whisper", torch.float32),
            (self.llm_model_id, "transformers", torch.float16),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for model_id, backend, float_type in flash_models:
                future = executor.submit(
                    self._register_single_model,
                    model_id,
                    backend,
                    float_type=float_type,
                )
                futures.append(future)

            # Wait for all registrations
            for future in futures:
                future.result()

        # Wait for Bark download to complete
        bark_thread.join()

        self.models_registered = True
        print("✅ All models registered in parallel")

    def _register_single_model(self, model_id, backend, float_type):
        """Register a single model with flashtensors"""
        print(f"  Registering {model_id}...")
        result = flash.register_model(
            model_id=model_id,
            backend=backend,
            torch_dtype=float_type,
            force=False,
            hf_token=self.hf_token,
        )
        print(f"    {model_id}: {result['status']}")
        return result

    def register_bark_model(self):
        """Download and cache Bark with retry logic"""
        print(f"  Downloading {self.bark_model_id}...")

        max_retries = 3
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                model_path = snapshot_download(
                    self.bark_model_id,
                    cache_dir=os.path.expanduser("~/.cache/huggingface"),
                    token=self.hf_token,
                    max_workers=2,
                    resume_download=True,
                    use_auth_token=self.hf_token,
                    etag_timeout=60,
                )

                # Quick validation load
                time.sleep(1)  # Small delay
                bark_model = BarkModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    token=self.hf_token,
                )
                processor = AutoProcessor.from_pretrained(
                    model_path, token=self.hf_token
                )

                # Cleanup immediately
                del bark_model, processor
                flash.cleanup_gpu()
                torch.cuda.empty_cache()
                gc.collect()

                print(f"    ✅ Bark cached successfully")
                return

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(f"    Rate limited, waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    ⚠️ Bark registration failed: {e}")
                    return

    def load_model(self, model_type, preload_next=None):
        """Optimized model loading with preloading hints"""
        if self.current_model == model_type:
            return

        # Aggressive cleanup of current model
        if self.current_model:
            flash.cleanup_gpu()
            torch.cuda.empty_cache()
            gc.collect()

        print(f"⚡ Loading {model_type}...")
        start_time = time.time()

        if model_type == "whisper":
            self.whisper_model, self.whisper_processor = flash.load_model(
                model_id=self.whisper_model_id,
                backend="whisper",
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif model_type == "llm":
            self.llm_model, self.llm_tokenizer = flash.load_model(
                model_id=self.llm_model_id,
                backend="transformers",
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif model_type == "bark":
            # Load from cache with optimizations
            self.bark_model = BarkModel.from_pretrained(
                self.bark_model_id,
                torch_dtype=torch.float32,
                device_map="auto",
                token=self.hf_token,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.bark_processor = AutoProcessor.from_pretrained(
                self.bark_model_id, token=self.hf_token, local_files_only=True
            )

        self.current_model = model_type
        load_time = time.time() - start_time
        print(f"✅ {model_type} loaded in {load_time:.2f}s")

        # Start preloading next model if hinted
        if preload_next:
            self.start_preload(preload_next)

    def start_preload(self, next_model):
        """Start preloading next model in background"""
        if self.preload_thread and self.preload_thread.is_alive():
            return

        self.next_model_hint = next_model
        # Note: Actual preloading is tricky with limited GPU memory
        # For now, just set the hint for faster switching

    def enhanced_vad(self, audio_chunk):
        """Enhanced Voice Activity Detection with spectral features"""
        # RMS energy
        rms = np.sqrt(np.mean(audio_chunk**2))

        # Spectral centroid (simple pitch detection)
        if len(audio_chunk) > 512:
            fft = np.fft.rfft(audio_chunk)
            magnitude = np.abs(fft)
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / (
                np.sum(magnitude) + 1e-12
            )

            # Voice typically has spectral centroid between certain frequencies
            voice_freq_range = (200, 4000)  # Hz
            freq_per_bin = self.sample_rate / (2 * len(magnitude))
            sc_freq = spectral_centroid * freq_per_bin

            is_voice_freq = voice_freq_range[0] < sc_freq < voice_freq_range[1]
            is_loud_enough = rms > self.silence_threshold

            return is_voice_freq and is_loud_enough

        return rms > self.silence_threshold

    def audio_callback(self, indata, frames, time, status):
        """Optimized real-time audio callback"""
        if self.state == "LISTENING" and not self.interrupt_requested:
            audio_chunk = indata[:, 0]
            self.audio_buffer.extend(audio_chunk)

            # Enhanced voice activity detection
            if self.enhanced_vad(audio_chunk):
                self.silence_counter = 0
            else:
                self.silence_counter += 1

            # Faster silence detection
            silence_chunks = self.silence_duration / self.chunk_duration
            if self.silence_counter > silence_chunks:
                threading.Thread(target=self.stop_listening).start()

    def start_listening(self):
        """Start listening with interrupt capability"""
        print("👂 Listening...")
        self.state = "LISTENING"
        self.interrupt_requested = False
        self.audio_buffer.clear()
        self.silence_counter = 0

        self.audio_stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            latency="low",  # Request low latency
        )
        self.audio_stream.start()

    def stop_listening(self):
        """Stop listening and process audio"""
        if self.state != "LISTENING":
            return

        print("🔇 Processing...")
        self.state = "PROCESSING"
        self.audio_stream.stop()
        self.audio_stream.close()

        if len(self.audio_buffer) > self.sample_rate * 0.5:  # At least 0.5s of audio
            audio_data = np.array(list(self.audio_buffer))
            # Process audio in background thread for responsiveness
            threading.Thread(target=self.process_audio, args=(audio_data,)).start()
        else:
            print("Insufficient audio")
            self.state = "IDLE"

    def process_audio(self, audio_data):
        """Optimized audio processing"""
        print("🎤 Transcribing...")

        # Preload next model while transcribing
        self.load_model("whisper", preload_next="llm")

        # Faster audio preprocessing
        audio_data = audio_data.astype(np.float32)
        if len(audio_data) > self.sample_rate * 5:  # Truncate if too long
            audio_data = audio_data[-self.sample_rate * 5 :]

        input_features = self.whisper_processor(
            audio_data, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to("cuda")

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                input_features,
                max_length=128,  # Limit length for speed
                num_beams=1,  # Faster than beam search
                temperature=0.0,  # Deterministic for speed
            )
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

        print(f"📝 User: '{transcription}'")

        if transcription and len(transcription) > 3:
            threading.Thread(
                target=self.generate_response, args=(transcription,)
            ).start()
        else:
            self.state = "IDLE"

    def generate_response(self, user_input):
        """Optimized streaming response generation"""
        print("🧠 Generating...")

        self.load_model("llm", preload_next="bark")

        # Optimized prompt
        prompt = f"User: {user_input}\nBot:"

        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        self.state = "SPEAKING"
        chunk_buffer = ""

        # Start TTS thread
        self.start_tts_thread()

        with torch.no_grad():
            for i in range(self.max_response_length):
                if self.interrupt_requested:
                    break

                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=0.6,  # Slightly lower for consistency
                    do_sample=True,
                    top_p=0.85,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

                new_token_id = outputs[0, -1:]
                new_token = self.llm_tokenizer.decode(
                    new_token_id, skip_special_tokens=True
                )

                if new_token_id.item() == self.llm_tokenizer.eos_token_id:
                    break

                chunk_buffer += new_token

                # Queue TTS chunks
                if len(chunk_buffer) >= self.text_chunk_size or new_token in ".!?\n":
                    if chunk_buffer.strip():
                        self.tts_queue.put(chunk_buffer.strip())
                        chunk_buffer = ""

                # Update inputs
                inputs["input_ids"] = outputs
                inputs["attention_mask"] = torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones((1, 1), device=inputs["attention_mask"].device),
                    ],
                    dim=1,
                )

        # Queue remaining text
        if chunk_buffer.strip():
            self.tts_queue.put(chunk_buffer.strip())

        # Signal end of text generation
        self.tts_queue.put(None)

        # Wait for TTS to complete
        if self.tts_thread:
            self.tts_thread.join()

        self.state = "IDLE"

    def start_tts_thread(self):
        """Start TTS processing in background thread"""
        self.tts_thread = threading.Thread(target=self.process_tts_queue)
        self.tts_thread.start()

    def process_tts_queue(self):
        """Process TTS queue in background"""
        self.load_model("bark")

        while True:
            try:
                text = self.tts_queue.get(timeout=10)
                if text is None:  # End signal
                    break

                if self.interrupt_requested:
                    break

                self.generate_speech_chunk(text)

            except queue.Empty:
                break

    def generate_speech_chunk(self, text):
        """Optimized speech generation"""
        if not text.strip() or self.interrupt_requested:
            return

        inputs = self.bark_processor(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            audio_array = self.bark_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.05,  # Very low for speed and consistency
                max_length=64,  # Shorter for faster generation
            )

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.bark_model.generation_config.sample_rate

        # Non-blocking audio playback
        sd.play(audio_array, sample_rate)

    def request_interrupt(self):
        """Request interruption of current operation"""
        self.interrupt_requested = True
        if self.state == "SPEAKING":
            sd.stop()  # Stop current audio
            # Clear TTS queue
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except queue.Empty:
                    break

    def run(self):
        """Optimized main loop"""
        self.setup_flashtensors()
        self.register_models()

        print("\n🎙️ Ultra-Fast Voice Agent Ready!")
        print("Optimized for Jetson Nano - Say something to start")
        print("Press Ctrl+C to exit")

        try:
            while not self.stop_event.is_set():
                if self.state == "IDLE":
                    self.start_listening()

                time.sleep(0.05)  # Very responsive loop

        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.stop_event.set()
            self.request_interrupt()

            if hasattr(self, "audio_stream"):
                self.audio_stream.stop()
                self.audio_stream.close()

            flash.cleanup_gpu()
            flash.shutdown_server()
            print("✅ Stopped")


def main():
    agent = VoiceAgent()
    agent.run()


if __name__ == "__main__":
    main()


# Jetson Nano Optimization
# Maximize Jetson performance

# sudo nvpmodel -m 0  # Max performance mode
# sudo jetson_clocks   # Max clocks

# Increase swap for large model loading
# sudo fallocate -l 8G /swapfile
