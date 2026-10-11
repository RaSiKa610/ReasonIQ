"""
ReasonIQ - Voice Activity Detection

Silero VAD with automatic:
- Stereo -> Mono conversion
- Any sample rate -> 16 kHz resampling
"""

import wave
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from scipy.signal import resample_poly


class ReasonIQVAD:

    def __init__(self):
        print("Loading Silero VAD...")

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils

        print("Silero VAD loaded successfully.")

    def load_wav(self, audio_path: str):
        """
        Load WAV file and convert it to:
        - mono
        - 16 kHz
        - float32
        """

        with wave.open(audio_path, "rb") as wav:

            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()

            frames = wav.readframes(
                wav.getnframes()
            )

        # --------------------------------------------------
        # Validate WAV format
        # --------------------------------------------------

        if sample_width != 2:
            raise ValueError(
                "WAV must be 16-bit PCM."
            )

        if channels not in (1, 2):
            raise ValueError(
                f"Unsupported number of channels: {channels}"
            )

        # --------------------------------------------------
        # PCM bytes -> NumPy int16
        #
        # .copy() removes the non-writable buffer warning.
        # --------------------------------------------------

        audio_np = np.frombuffer(
            frames,
            dtype=np.int16
        ).copy()

        # --------------------------------------------------
        # int16 -> float32
        # --------------------------------------------------

        audio = torch.from_numpy(
            audio_np
        ).float() / 32768.0

        # --------------------------------------------------
        # Stereo -> Mono
        # --------------------------------------------------

        if channels == 2:

            audio = audio.reshape(
                -1,
                2
            ).mean(dim=1)

        # --------------------------------------------------
        # Convert to NumPy for resampling
        # --------------------------------------------------

        audio_np = audio.numpy()

        target_rate = 16000

        # --------------------------------------------------
        # Resample to 16 kHz
        # --------------------------------------------------

        if sample_rate != target_rate:

            print(
                f"Resampling audio: "
                f"{sample_rate} Hz -> "
                f"{target_rate} Hz"
            )

            audio_np = resample_poly(
                audio_np,
                target_rate,
                sample_rate
            ).astype(np.float32)

        # --------------------------------------------------
        # NumPy -> PyTorch tensor
        # --------------------------------------------------

        audio = torch.from_numpy(
            audio_np
        )

        return audio, target_rate

    def get_speech_segments(
        self,
        wav_tensor: torch.Tensor,
        sample_rate: int = 16000,
        threshold: float = 0.35,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 250
    ) -> List[Dict]:
        """
        Extract speech timestamps with conservative thresholds and adequate padding
        to prevent cutting off beginning consonants or trailing word endings.
        Merges adjacent speech segments to prevent fragmented transcriptions.
        """
        raw_segments = self.get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )

        if not raw_segments:
            return []

        # Merge segments separated by less than 0.5s of silence
        merged = []
        for seg in raw_segments:
            if not merged:
                merged.append(dict(seg))
            else:
                prev = merged[-1]
                gap_samples = seg["start"] - prev["end"]
                if gap_samples <= int(0.50 * sample_rate):
                    prev["end"] = max(prev["end"], seg["end"])
                else:
                    merged.append(dict(seg))

        return merged

    def detect_speech(
        self,
        audio_path: str
    ) -> List[Dict]:
        """
        Detect speech regions in an audio file.

        Returns:
            [
                {
                    "start": float,
                    "end": float,
                    "duration": float
                }
            ]
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        wav, sample_rate = self.load_wav(
            str(path)
        )

        speech_timestamps = self.get_speech_segments(
            wav,
            sample_rate=sample_rate
        )

        results = []
        for segment in speech_timestamps:
            start = segment["start"] / sample_rate
            end = segment["end"] / sample_rate
            duration = end - start

            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2)
            })

        return results


# Backward compatibility alias
BiasLessVAD = ReasonIQVAD


def main():

    print("=" * 60)
    print("ReasonIQ Voice Activity Detection Test")
    print("=" * 60)

    audio_file = (
        r"C:\Users\Rasika\BiasLess"
        r"\tests\sample_audio.wav"
    )

    try:

        # --------------------------------------------------
        # Initialize VAD
        # --------------------------------------------------

        vad = ReasonIQVAD()

        # --------------------------------------------------
        # Detect speech
        # --------------------------------------------------

        results = vad.detect_speech(
            audio_file
        )

        print("\n" + "=" * 60)
        print("SPEECH DETECTION RESULT")
        print("=" * 60)

        if not results:

            print(
                "No speech detected."
            )

        else:

            print(
                f"Speech segments detected: "
                f"{len(results)}"
            )

            for i, segment in enumerate(
                results,
                start=1
            ):

                print(
                    f"Segment {i}: "
                    f"{segment['start']:.2f}s - "
                    f"{segment['end']:.2f}s "
                    f"({segment['duration']:.2f}s)"
                )

        print("=" * 60)

    except Exception as error:

        print("\nVAD failed.")
        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    main()