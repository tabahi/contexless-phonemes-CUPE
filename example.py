"""
CUPE: Easy usage with automatic downloading from Hugging Face Hub
"""

import torch
import torchaudio
from huggingface_hub import hf_hub_download
import importlib.util
import sys
import os

def load_cupe_model(model_name="english", device="auto"):
    """
    Load CUPE model with automatic downloading from Hugging Face Hub
    
    Args:
        model_name: "english", "multilingual-mls", or "multilingual-mswc"
        device: "auto", "cpu", or "cuda"
    
    Returns:
        Tuple of (extractor, windowing_module)
    """
    
    # Model checkpoint mapping
    model_files = {
        "english": "en_libri1000_uj01d_e199_val_GER=0.2307.ckpt",
        "multilingual-mls": "multi_MLS8_uh02_e36_val_GER=0.2334.ckpt",
        "multilingual-mswc": "multi_mswc38_ug20_e59_val_GER=0.5611.ckpt"
    }
    
    if model_name not in model_files:
        raise ValueError(f"Model {model_name} not available. Choose from: {list(model_files.keys())}")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CUPE {model_name} model...")
    
    # Download model files from Hugging Face Hub
    repo_id = "Tabahi/CUPE-2i"
    
    model_file = hf_hub_download(repo_id=repo_id, filename="model2i.py")
    windowing_file = hf_hub_download(repo_id=repo_id, filename="windowing.py")
    mapper_file = hf_hub_download(repo_id=repo_id, filename="mapper.py")
    model_utils_file = hf_hub_download(repo_id=repo_id, filename="model_utils.py")
    checkpoint_file = hf_hub_download(repo_id=repo_id, filename=f"ckpt/{model_files[model_name]}")
    
    # Dynamically import the modules
    def import_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    _ = import_module_from_file("model_utils", model_utils_file)
    model2i = import_module_from_file("model2i", model_file)
    windowing = import_module_from_file("windowing", windowing_file)
    mapper = import_module_from_file("mapper", mapper_file)

    phoneme_to_token = mapper.phoneme_mapped_index
    token_to_phoneme = {v: k for k, v in phoneme_to_token.items()}
    group_to_token = mapper.phoneme_groups_index
    token_to_group = {v: k for k, v in group_to_token.items()}

    # Initialize the model
    extractor = model2i.CUPEEmbeddingsExtractor(checkpoint_file, device=device)
    
    print(f"Model loaded on {device}")
    return extractor, windowing, token_to_phoneme, token_to_group

def predict_phonemes(audio_path, model_name="english", device="auto"):
    """
    Predict phonemes from audio file
    
    Args:
        audio_path: Path to audio file
        model_name: CUPE model variant to use
        device: Device to run inference on
    
    Returns:
        Dictionary with predictions and metadata
    """
    
    # Load model
    extractor, windowing, token_to_phoneme, token_to_group = load_cupe_model(model_name, device)
    
    # Audio processing parameters
    sample_rate = 16000
    window_size_ms = 120
    stride_ms = 80
    
    # Load and preprocess audio
    audio, orig_sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    resampler = torchaudio.transforms.Resample(
            sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    
    # Always use resampler for consistency
    audio = resampler(audio)
    
    # Move to device and add batch dimension
    audio = audio.to(device)
    audio_batch = audio.unsqueeze(0)
    
    print(f"Processing audio: {audio.shape[1]/sample_rate:.2f}s duration")
    
    # Window the audio
    windowed_audio = windowing.slice_windows(
        audio_batch,
        sample_rate,
        window_size_ms,
        stride_ms
    )
    
    batch_size, num_windows, window_size = windowed_audio.shape
    windows_flat = windowed_audio.reshape(-1, window_size)
    
    # Get model predictions
    logits_phonemes, logits_groups = extractor.predict(
        windows_flat, 
        return_embeddings=False, 
        groups_only=False
    )
    
    # Reshape and stitch predictions
    frames_per_window = logits_phonemes.shape[1]
    
    logits_phonemes = logits_phonemes.reshape(batch_size, num_windows, frames_per_window, -1)
    logits_groups = logits_groups.reshape(batch_size, num_windows, frames_per_window, -1)
    
    phoneme_logits = windowing.stich_window_predictions(
        logits_phonemes,
        original_audio_length=audio_batch.size(2),
        cnn_output_size=frames_per_window,
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        stride_ms=stride_ms
    )
    
    group_logits = windowing.stich_window_predictions(
        logits_groups,
        original_audio_length=audio_batch.size(2),
        cnn_output_size=frames_per_window,
        sample_rate=sample_rate,
        window_size_ms=window_size_ms,
        stride_ms=stride_ms
    )
    
    # Convert to probabilities and predictions
    phoneme_probs = torch.softmax(phoneme_logits.squeeze(0), dim=-1)
    group_probs = torch.softmax(group_logits.squeeze(0), dim=-1)
    
    phoneme_preds = torch.argmax(phoneme_probs, dim=-1)
    group_preds = torch.argmax(group_probs, dim=-1)
    
    phonemes_sequence = [token_to_phoneme[int(p)] for p in phoneme_preds.cpu().numpy()]
    groups_sequence = [token_to_group[int(g)] for g in group_preds.cpu().numpy()]
    # remove noise
    phonemes_sequence = [p for p in phonemes_sequence if p != 'noise']
    groups_sequence = [g for g in groups_sequence if g != 'noise']
    
    
    # Calculate timestamps (approximately 16ms per frame)
    num_frames = phoneme_probs.shape[0]
    
    print(f"Processed {num_frames} frames ({num_frames*16}ms total)")
    
    return {
        'phoneme_probabilities': phoneme_probs.cpu().numpy(),
        'phoneme_predictions': phoneme_preds.cpu().numpy(),
        'group_probabilities': group_probs.cpu().numpy(), 
        'group_predictions': group_preds.cpu().numpy(),
        'phonemes_sequence': phonemes_sequence,
        'groups_sequence': groups_sequence,
        'model_info': {
            'model_name': model_name,
            'sample_rate': sample_rate,
            'frames_per_second': 1000/16,  # ~62.5 fps
            'num_phoneme_classes': phoneme_probs.shape[-1],
            'num_group_classes': group_probs.shape[-1]
        }
    }

# Example usage
if __name__ == "__main__":
    
    # Simple example
    audio_file = "samples/109867__timkahn__butterfly.wav.wav"  # Replace with your audio file
    
    
    if not os.path.exists(audio_file):
        print(f"Audio file {audio_file} does not exist. Please provide a valid path.")
        sys.exit(1)
    
    torch.manual_seed(42)  # For reproducibility
    # Predict with English model
    results = predict_phonemes(
        audio_path=audio_file,
        model_name="english",  # or "multilingual-mls" or "multilingual-mswc"
        device="cpu"
    )
    
    print(f"\nResults:")
    print(f"Phoneme predictions shape: {results['phoneme_predictions'].shape}")
    print(f"Group predictions shape: {results['group_predictions'].shape}")
    print(f"Model info: {results['model_info']}")
    
    # Show first 10 predictions with timestamps
    print(f"\nFirst 10 frame predictions:")
    for i in range(min(10, len(results['phoneme_predictions']))):
        print(f"Frame {i}: phoneme={results['phoneme_predictions'][i]}, "
              f"group={results['group_predictions'][i]}")

    print(f"\nPhonemes sequence: {results['phonemes_sequence'][:10]}...")  # Show first 10 phonemes
    print(f"Groups sequence: {results['groups_sequence'][:10]}...")

''' output:
Loading CUPE english model...
Model loaded on cpu
Processing audio: 1.26s duration
Processed 75 frames (1200ms total)

Results:
Phoneme predictions shape: (75,)
Group predictions shape: (75,)
Model info: {'model_name': 'english', 'sample_rate': 16000, 'frames_per_second': 62.5, 'num_phoneme_classes': 67, 'num_group_classes': 17}

First 10 frame predictions:
Frame 0: phoneme=66, group=16
Frame 1: phoneme=66, group=16
Frame 2: phoneme=29, group=7
Frame 3: phoneme=66, group=16
Frame 4: phoneme=66, group=16
Frame 5: phoneme=66, group=16
Frame 6: phoneme=10, group=2
Frame 7: phoneme=66, group=16
Frame 8: phoneme=66, group=16
Frame 9: phoneme=66, group=16

Phonemes sequence: ['b', 'ʌ', 't', 'h', 'ʌ', 'f', 'l', 'æ']...
Groups sequence: ['voiced_stops', 'central_vowels', 'voiceless_stops', 'voiceless_fricatives', 'central_vowels', 'voiceless_fricatives', 'laterals', 'low_vowels']...
'''
