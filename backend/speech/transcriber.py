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
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize the Whisper model.
        Model size can be configured via argument or WHISPER_MODEL env var (defaults to 'base').
        """
        import os
        selected_model = model_size or os.getenv("WHISPER_MODEL", "base.en")
        self.model_size = selected_model
        self.device = device
        self.compute_type = compute_type

        print("Loading faster-whisper model...")
        print(f"Model: {selected_model}")
        print(f"Device: {device}")
        print(f"Compute type: {compute_type}")

        self.model = WhisperModel(
            selected_model,
            device=device,
            compute_type=compute_type
        )

        print("faster-whisper model loaded successfully.")

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        initial_prompt: Optional[str] = "A formal debate speech discussing artificial intelligence, ethics, technology, education, culture, logic, and philosophy.",
        vad_filter: bool = False
    ) -> Dict:
        """
        Transcribe an audio file using faster-whisper with debate context prior,
        hallucination suppression, and transcription quality assessment.
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
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4
        )

        segment_results = []
        complete_text = []
        avg_logprob_sum = 0.0
        seg_count = 0

        # Known Whisper hallucination phrases on trailing noise or chopped speech
        phantom_patterns = [
            r"\bwas done that she was\b",
            r"\bthank you for watching\b",
            r"\bsubtitles by\b",
            r"\bbye[- ]bye\.?\b",
            r"\bstint!\b"
        ]

        import re

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            # Strip obvious trailing hallucinations
            for phantom in phantom_patterns:
                text = re.sub(phantom, "", text, flags=re.IGNORECASE).strip()

            if not text:
                continue

            avg_logprob = getattr(segment, "avg_logprob", 0.0)
            avg_logprob_sum += avg_logprob
            seg_count += 1

            segment_data = {
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": text,
                "avg_logprob": round(float(avg_logprob), 3) if avg_logprob else 0.0
            }

            segment_results.append(segment_data)
            complete_text.append(text)

        final_text = " ".join(complete_text).strip()
        mean_logprob = (avg_logprob_sum / seg_count) if seg_count > 0 else 0.0

        transcription_warning = None
        if seg_count > 0 and mean_logprob < -1.1:
            transcription_warning = "Audio quality or pronunciation was unclear; transcription may be approximate."

        return {
            "text": final_text,
            "language": info.language,
            "language_probability": round(
                float(info.language_probability),
                4
            ),
            "segments": segment_results,
            "mean_logprob": round(mean_logprob, 3),
            "transcription_warning": transcription_warning
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