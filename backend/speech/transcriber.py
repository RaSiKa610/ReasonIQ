"""
ReasonIQ - Local Speech-to-Text Transcriber

Uses faster-whisper for local speech recognition.

Pipeline:

Audio
  ↓
faster-whisper
  ↓
Transcribed Segments
  ↓
Clean Text
  ↓
ReasonIQ NLP Engine
"""

from pathlib import Path
from typing import List, Dict, Optional

from faster_whisper import WhisperModel


class ReasonIQTranscriber:
    """
    Local speech-to-text component for ReasonIQ.

    Uses faster-whisper without requiring a paid API.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize the Whisper model.

        Parameters
        ----------
        model_size:
            Whisper model size.

            Available examples:
            - tiny
            - base
            - small
            - medium
            - large-v3

        device:
            "cpu" or "cuda"

        compute_type:
            CPU:
                "int8"
                "float32"

            CUDA:
                "float16"
                "int8_float16"
        """

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        print("Loading faster-whisper model...")
        print(f"Model: {model_size}")
        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        print("faster-whisper model loaded successfully.")

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> Dict:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        language:
            Language code.
            Example:
                "en" for English
                "hi" for Hindi

            If None, Whisper automatically detects the language.

        Returns
        -------
        Dictionary containing:

        {
            "text": "...",
            "language": "en",
            "language_probability": 0.99,
            "segments": [...]
        }
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

        print(f"\nTranscribing: {path.name}")

        segments, info = self.model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True
        )

        segment_results = []
        complete_text = []

        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            segment_data = {
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text
            }

            segment_results.append(segment_data)
            complete_text.append(text)

        final_text = " ".join(complete_text).strip()

        return {
            "text": final_text,
            "language": info.language,
            "language_probability": round(
                float(info.language_probability),
                4
            ),
            "segments": segment_results
        }

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> str:
        """
        Simple transcription interface.

        Returns only the final text.
        """

        result = self.transcribe_file(
            audio_path=audio_path,
            language=language
        )

        return result["text"]

    def transcribe_segments(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> List[Dict]:
        """
        Return timestamped transcription segments.
        """

        result = self.transcribe_file(
            audio_path=audio_path,
            language=language
        )

        return result["segments"]


def print_transcription(result: Dict):
    """
    Print a formatted transcription result.
    """

    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)

    print(f"Language: {result['language']}")
    print(
        f"Language probability: "
        f"{result['language_probability']}"
    )

    print("\nFull Text:")
    print(result["text"])

    print("\nSegments:")

    for segment in result["segments"]:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        print(
            f"[{start:.2f}s - {end:.2f}s] "
            f"{text}"
        )

    print("=" * 60)


# Backward compatibility alias
BiasLessTranscriber = ReasonIQTranscriber


def main():

    print("=" * 60)
    print("ReasonIQ Speech-to-Text Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # Change this path to your test audio file.
    # ---------------------------------------------------------

    audio_file = (
        r"C:\Users\Rasika\BiasLess\tests\sample_audio.wav"
    )

    # ---------------------------------------------------------
    # Initialize transcriber.
    #
    # "base" gives a good balance between speed and accuracy.
    #
    # CPU + int8 is selected because the speech pipeline
    # should work reliably without requiring CUDA.
    # ---------------------------------------------------------

    transcriber = ReasonIQTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8"
    )

    # ---------------------------------------------------------
    # Transcribe audio.
    # ---------------------------------------------------------

    try:

        result = transcriber.transcribe_file(
            audio_path=audio_file,
            language="en"
        )

        print_transcription(result)

    except FileNotFoundError:

        print("\nAudio file was not found.")

        print(
            "\nCreate or copy a test audio file here:"
        )

        print(
            r"C:\Users\Rasika\BiasLess\tests\sample_audio.wav"
        )

        print(
            "\nThen run this program again."
        )

    except Exception as error:

        print("\nTranscription failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()