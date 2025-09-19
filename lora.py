import pandas as pd
import torch, torchaudio
from dataclasses import dataclass
from pathlib import Path

MAX_AUDIO_FILES = 4

@dataclass
class AudioTextPair:
    audio_path: str
    text: str
    speaker_id: int

    def load_audio(self, sample_rate=24000) -> torch.Tensor:
        waveform, sr = torchaudio.load(self.audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        processed_audio = waveform.squeeze(0)
        return processed_audio

def get_speaker_name(path):
    p = Path(path)
    parts = p.parts  # tuple of all path components
    # find the date folder (pattern: DD.MM.YYYY)
    for i, part in enumerate(parts):
        if len(part.split(".")) == 3:  # crude date detection
            if i + 1 < len(parts):
                return parts[i + 1]  # the folder right after the date
    return None

def transcribe_audio_files(metafile_paths: str = None):
    audio_text_pairs = []

    # Metafile mode
    for metafile_path in metafile_paths:
        meta_df = pd.read_csv(metafile_path, sep="|", header=None)

        # Iterate over rows
        for _, row in meta_df.iterrows():
            local_path = row[0]
            transcription = row[1]

            # Get parent directory
            speaker_name = get_speaker_name(local_path)  # Output: Айганыш

            if "Тимур" == speaker_name:
                speaker_id = 0
            elif "Айганыш" == speaker_name or "Айганыш" == speaker_name:
                speaker_id = 1
            else:
                print(speaker_name)
                print(local_path)
                raise ValueError()

            if "neutral".lower() in metafile_path.lower():
                tone = "<neutral>"
            elif "strict".lower() in metafile_path.lower():
                tone = "<strict>"
            else:
                raise ValueError()

            audio_text_pairs.append(AudioTextPair(audio_path=local_path,
                                                  text=tone + " " + transcription,
                                                  speaker_id=speaker_id))

            if MAX_AUDIO_FILES > 0 and len(audio_text_pairs) >= MAX_AUDIO_FILES:
                print(f"Reached MAX_AUDIO_FILES limit ({MAX_AUDIO_FILES}) while reading metafile.")
                break

    return audio_text_pairs