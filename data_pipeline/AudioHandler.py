"""
This class will handle the conversions between audio and spectrogram.
"""

import torchaudio


class AudioHandler:
    def __init__(self, params, device='cuda'):
        self.params = params

        self.device = device

        self.wav_to_spec = torchaudio.transforms.Spectrogram(
            n_fft = params['n_fft']
        ).to(device)

        self.spec_to_mel = torchaudio.transforms.MelScale(
            n_mels = params['num_frequencies'],
            sample_rate = params['sr'],
            n_stft = params['n_fft'] // 2 + 1
        ).to(device)

        self.mel_to_spec = torchaudio.transforms.InverseMelScale(
            n_mels = 64,
            sample_rate = params['sr'],
            n_stft = params['n_fft'] // 2 + 1
        ).to(device)


        self.spec_to_wav = torchaudio.transforms.GriffinLim(
            n_fft = params['n_fft']
        ).to(device)

    def convert_wav_to_spectrogram(self, wav_tensor):
        """
        Converts a wav tensor to a MelSpectrogram.

        Args:
            wav_tensor: The tensor object of the loaded waveform.
        """
        # Convert to real valued tensor
        spec_real_valued = self.wav_to_spec(wav_tensor)

        # Convert to melspectrogram
        melspectrogram = self.spec_to_mel(spec_real_valued)

        return melspectrogram
    
    def convert_spec_to_wav(self, mel_spec):
        """
        Converts a MelSpectrogram to a wavform using the Griffin Lim algorithm.
        Note that this is lossy.

        Args:
            mel_spec: The MelSpectrogram that we will convert to waveform
        """
        # Convert MelSpectrogram to Power Spectrogram using InverseMel
        spec_real_valued = self.mel_to_spec(mel_spec)

        # Convert Power Spectrogram to Waveform using Griffin Lim
        wav_approx = self.spec_to_wav(spec_real_valued)

        return wav_approx
