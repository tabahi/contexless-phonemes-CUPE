
import torch
import torchaudio
import torch.nn as nn

import modules.p1cupe.model_utils as model_utils



class FinetuneXLSR(nn.Module):
    def __init__(self, hp, input_wav_length, freeze_feature_encoder=False):
        super().__init__()

        self.hp = hp
        self.noise_level = hp.noise_level
        self.xls_dim = 1024
        self.output_dim = hp.phoneme_classes + 1  # 66 phonemes + blank token for CTC
        
        # Load pre-trained XLSR model
        #bundle = torchaudio.pipelines.WAV2VEC2_XLSR53  # referenced as 'xa' in yaml
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M    # referenced as 'xb' in yaml
        
        self.XLSR = bundle.get_model()

        # reset paramters for XLSR:

        # reset parameters for XLSR:
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                #print("reset_parameters for ", module)

        if hasattr(self.hp , 'xlrs_reset'):
            if (self.hp.xlrs_reset):
                print("reset_parameters for XLSR")
                self.XLSR.apply(reset_parameters)
        
        self.freeze_feature_encoder = freeze_feature_encoder
        # Optionally freeze only the feature encoder
        # It's common practice to keep the feature encoder frozen while fine-tuning the rest
        if self.freeze_feature_encoder:
            # Freeze the feature extractor part
            for param in self.XLSR.model.feature_extractor.parameters():
                param.requires_grad = False
            
            # Optionally, you might also want to freeze the feature projection
            # for param in self.XLSR.model.encoder.feature_projection.parameters():
            #    param.requires_grad = False
            
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.xls_dim, self.xls_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(self.xls_dim, self.output_dim)
        )

        
        #self.layer_dims = self.make_layer_sizer()
        self.layer_dims = model_utils.ModelUtils.extract_layer_dims(self)
        self.frames_per_window = model_utils.ModelUtils.calculate_layer_sizes(self.layer_dims, torch.tensor([input_wav_length]), -1)[0].int()
        self.frames_per_window = torch.ceil((self.frames_per_window-1)).int()
        self.model_utils = model_utils.ModelUtils(self.layer_dims, input_wav_length, self.frames_per_window)
        #self.model_utils.print_model_info()
        

    
    def update_frames_per_window(self, input_wav_length):
        self.frames_per_window = self.model_utils.calculate_layer_sizes(self.layer_dims, torch.tensor([input_wav_length]), -1)[0].int()
        self.frames_per_window = torch.ceil((self.frames_per_window-1)).int()
        print("frames_per_window (frames per clip if disable_windowing):", self.frames_per_window.item())
        return self.frames_per_window

    def forward(self, x):
        if self.training and self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level
        
        #x = x.unsqueeze(1)
        
        # Remove torch.no_grad() to allow gradients to flow through XLSR
        #if self.freeze_feature_encoder:
        #    with torch.no_grad():
        #        features, _ = self.XLSR.extract_features(x)
        #else:
        features, _ = self.XLSR.extract_features(x)
        features = features[-1]
        
        logits = self.classifier(features)
        return logits




def test_model():
    

    #bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    #model = bundle.get_model().to(device)

    device = torch.device("cuda:0")
    
    print(torch.__version__)
    print(torchaudio.__version__)
    torch.random.manual_seed(0)
    print(device)

    # Initialize model without freezing any layers
    model = FinetuneXLSR(noise_level=0.01, freeze_feature_encoder=False).to(device)

    print(model.__class__)

    # Resample audio to the expected sampling rate
    sample_path = "tmp/data/audio_samples/9860_8338_000010.flac.wav"

    sample_waveform, sample_samplerate = torchaudio.load(sample_path)
    sample_waveform = sample_waveform.to(device)
    waveform = torchaudio.functional.resample(sample_waveform, sample_samplerate, 16000)
    
    batch = waveform.to(device)
    print(waveform.shape)
    
    # Extract acoustic features
    with torch.inference_mode():
        logits = model(batch)

    print(len(logits), logits[0].shape)
    for x, element in enumerate(logits):
        print(x, element)

def main():
    test_model()

if __name__ == "__main__":
    main()