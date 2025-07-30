"""
Phonemes probabilities extraction
Audio wavs -> CUPE model -> phoneme probabilities 
"""

import torch
import torchaudio
import os
from tqdm import tqdm



from model2i import CUPEEmbeddingsExtractor  # main CUPE model's feature extractor
import windowing as windowing  # import slice_windows, stich_window_predictions

class EmbeddingsExtractionPipeline:
    """
    Pipeline for extracting allophone probabilities from audio using CUPE model
    """
    def __init__(self, cupe_ckpt_path, max_duration=10, verbose=True, device="cpu"):
        """
        Initialize the pipeline
        
        Args:
            cupe_ckpt_path: Path to CUPE model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.verbose = verbose
        self.extractor = CUPEEmbeddingsExtractor(cupe_ckpt_path, device=self.device)

        

        

        self.config(max_duration=max_duration)
        if self.verbose:
            print("max_frames_per_clip:", self.max_frames_per_clip.item())

        dummy_wav = torch.zeros(1, self.max_wav_len, dtype=torch.float32, device='cpu')  # dummy waveform for config
        dummy_wav = dummy_wav.unsqueeze(0)  # add batch dimension
        dummy_logits, dummy_spectral_lens = self._process_audio_batch(audio_batch=dummy_wav, wav_lens=torch.tensor([dummy_wav.shape[2]], dtype=torch.long) )

        if self.verbose:
            print (f"Dummy logits shape: {dummy_logits.shape}, Dummy spectral lengths: {dummy_spectral_lens}")

        assert dummy_logits.shape[1] == self.max_frames_per_clip, f"Dummy logits shape mismatch: {dummy_logits.shape[1]} vs {self.max_frames_per_clip}"
        assert dummy_logits.shape[2] == self.output_dim, f"Dummy logits output dimension mismatch: {dummy_logits.shape[2]} vs {self.output_dim}"

        # resampler for audio preprocessing - recommended for all audio inputs even if they are already 16kHz for consistency
        self.resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

    def config(self, max_duration=10, window_size_ms=120, stride_ms=80, ):
        """
        Configure pipeline parameters
        
        Args:
            max_duration: Maximum duration of audio in seconds
            window_size_ms: Window size in milliseconds
            stride_ms: Stride size in milliseconds
        """
        self.sample_rate = 16000
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms
        self.phoneme_classes = 66
        self.phoneme_groups = 16
        self.extract_phoneme_groups = True  # whether to extract phoneme groups (16 phoneme groups)
        self.extract_phoneme_individuals = True  # whether to extract phoneme individuals (66 phoneme classes)
        # if both of the above are True, then the output dimension is 66 + 16 = 82 (logits from both tasks concatenated)
        self.output_dim = 0
        if self.extract_phoneme_individuals: self.output_dim += self.phoneme_classes
        if self.extract_phoneme_groups: self.output_dim += self.phoneme_groups
        
        
        self.window_size_wav = int(window_size_ms * self.sample_rate / 1000)
        self.stride_size_wav = int(stride_ms * self.sample_rate / 1000)
        self.max_wav_len = max_duration * self.sample_rate  # max duration in seconds to samples
        # Get frames_per_window from model
        self.frames_per_window = self.extractor.model.update_frames_per_window(self.window_size_wav)[0]
        self.max_frames_per_clip = windowing.calc_spec_len_ext(torch.tensor([self.max_wav_len], dtype=torch.long),self.window_size_ms, self.stride_ms, self.sample_rate, frames_per_window=self.frames_per_window,disable_windowing=False,wav_len_max=self.max_wav_len)[0]
        self.frames_per_second = windowing.calc_spec_len_ext(torch.tensor([100000*self.sample_rate], dtype=torch.long),self.window_size_ms, self.stride_ms, self.sample_rate, frames_per_window=self.frames_per_window,disable_windowing=False,wav_len_max=self.max_wav_len)[0].float()/100000
        
        self.ms_per_frame = int(1000 / self.frames_per_second.item())
        if self.verbose:
            print(f"frames_per_window: {self.frames_per_window.item()}, Max frames per clip: {self.max_frames_per_clip.item()}")  # INFO: 10 frames per 120ms window, 620 frames for 10s clip
            print(f"Frames per second: {self.frames_per_second.item()}")  # INFO: 62.49995040893555 frames per second
        
            print(f"milliseconds per frame: {self.ms_per_frame}")  # INFO: 16 milliseconds per frame

        

        

    def _process_audio_batch(self, audio_batch, wav_lens):
        """
        Process a batch of audio to extract logits
        
        Args:
            audio_batch: Batch of audio waveforms
            wav_lens: Lengths of each audio in the batch
            
        Returns:
            logits_class: Combined windows predictions
            spectral_lens: Lengths of spectral features
        """
        # Window the audio
        windowed_audio = windowing.slice_windows(
            audio_batch.to(self.device), 
            self.sample_rate, 
            self.window_size_ms, 
            self.stride_ms
        )
        batch_size, num_windows, window_size = windowed_audio.shape
        windows_flat = windowed_audio.reshape(-1, window_size)
        
        # Get predictions
        _logits_class, _logits_group = self.extractor.predict(windows_flat, return_embeddings=False, groups_only=False)
        
        
        # concatenate logits_class and logits_group
        logits = torch.cat([_logits_class[:, :, :self.phoneme_classes], _logits_group[:, :, :self.phoneme_groups]], dim=2)
        assert(logits.shape[2]==self.output_dim)
        
        frames_per_window = logits.shape[1] # INFO: 10 frames per window
        assert frames_per_window == self.frames_per_window, f"Expected {self.frames_per_window} frames per window, got {frames_per_window}"
        


        # Reshape and stitch window predictions
        logits = logits.reshape(batch_size, num_windows, frames_per_window, -1)
        logits = windowing.stich_window_predictions(
            logits, 
            original_audio_length=audio_batch.size(2), 
            cnn_output_size=frames_per_window, 
            sample_rate=self.sample_rate, 
            window_size_ms=self.window_size_ms, 
            stride_ms=self.stride_ms
        )
        # batch_size, seq_len, num_classes = logits.shape

        assert logits.shape[1] == self.max_frames_per_clip, f"Phoneme logits shape mismatch: {logits.shape[1]} vs {self.max_frames_per_clip}"

            
        

        # Calculate spectral lengths
        spectral_lens = windowing.calc_spec_len_ext(
            wav_lens, 
            self.window_size_ms, 
            self.stride_ms,
            self.sample_rate, 
            frames_per_window=self.frames_per_window,
            disable_windowing=False,
            wav_len_max=self.max_wav_len
        )
        #frames_per_clip = logits_class.shape[1] # INFO: 620 frames for 10s clip
        

        assert max(spectral_lens) <= self.max_frames_per_clip, f"Max spectral length {max(spectral_lens)} exceeds {self.max_frames_per_clip}"
        assert min(spectral_lens) > 0, f"Min spectral length {min(spectral_lens)} is not valid"

        return logits, spectral_lens
    
    def extract_embeddings_dataloader(self, dataloader): # example code, dataloader not implemented
        
        
        print("Starting phoneme embeddings extraction process...")
        features_collation = None
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Extracting phonemes")):
        
            audio_batch, wav_lens, clip_id = batch_data
            
                
            # Process audio and get predictions
            class_probs, spectral_lens = self._process_audio_batch(audio_batch, wav_lens) # returns shape (batch_size, max_frames_per_clip, phoneme_classes)
            
            # Process each sequence in the batch manually 
            batch_size = spectral_lens.shape[0]
            

            # concat class_probs, frames_confidence, formants
        

            for i in range(batch_size):

                # Get sequence data
                if spectral_lens is not None:
                    class_probs_i = class_probs[i][:spectral_lens[i]]
                else:
                    class_probs_i = class_probs[i]
                
                
                features_i = class_probs_i.detach()
                self.output_handler(clip_id[i].item(), features_i)

        print(f"Extracted {len(features_collation)} allophone embeddings")
        if len(features_collation) == 0:
            raise ValueError("No valid phoneme features were extracted.")
        
        return features_collation
    

    def output_handler(self, clip_id, features): # callback function to handle the output of the pipeline - Not implemented
        """
        Handle the output of the pipeline
        
        Args:
            features_collation: Collated features from the extraction process
        """
        output_length = features.shape[0] # already de-padded
        if self.verbose:
            print(f"Output handler received {len(features)}-dim features of length {output_length} for clip {clip_id}")
        


def process_single_clip(path_to_audio, pipeline, unpad_output=True):
    """
    Process a single audio clip to extract phoneme embeddings
    
    Args:
        path_to_audio: Path to the audio file
        pipeline: EmbeddingsExtractionPipeline instance
    """
    audio_clip, sr = torchaudio.load(path_to_audio)
    
    if sr != pipeline.sample_rate:
        raise ValueError(f"Sample rate mismatch: {sr} vs {pipeline.sample_rate}")

    if audio_clip.shape[0] > 1:
        audio_clip = audio_clip.mean(dim=0, keepdim=True)  # Convert to mono if stereo

    audio_clip = audio_clip.to(pipeline.device)
    audio_clip = pipeline.resampler(audio_clip)  # Resample to 16kHz if needed
    audio_clip = audio_clip.unsqueeze(0)  # Add batch dimension

    if audio_clip.shape[2] > pipeline.max_wav_len:
        print(f"Audio clip {path_to_audio} exceeds max length {pipeline.max_wav_len}, trimming to max length.")
        audio_clip = audio_clip[:, :pipeline.max_wav_len]  # Trim to max length

    original_length = audio_clip.shape[2]
    if audio_clip.shape[2] < pipeline.max_wav_len:
        audio_clip = torch.nn.functional.pad(audio_clip, (0, pipeline.max_wav_len - audio_clip.shape[2]))

    features, output_length = pipeline._process_audio_batch(audio_batch=audio_clip, wav_lens=torch.tensor([original_length], dtype=torch.long))
    
    features = features.squeeze(0)  # Remove batch dimension

    if unpad_output: features = features[:output_length, :]

    print(f"Output shape: {features.shape} for audio clip {path_to_audio}")
    return features


if __name__ == "__main__":

    torch.manual_seed(42)

    cupe_ckpt_path = "ckpt/en_libri1000_uj01d_e199_val_GER=0.2307.ckpt"
    pipeline = EmbeddingsExtractionPipeline(cupe_ckpt_path, max_duration=10, device="cpu", verbose=False)
    
    audio_clip1_path = "samples/109867__timkahn__butterfly.wav.wav"
    audio_clip2_path = "samples/Schwa-What.mp3.wav"

    features1 = process_single_clip(audio_clip1_path, pipeline)
    features2 = process_single_clip(audio_clip2_path, pipeline)

    print("Done!")