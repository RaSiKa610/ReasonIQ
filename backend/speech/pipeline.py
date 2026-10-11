"""
ReasonIQ - Speech + NLP Analysis Pipeline

Audio
  ↓
Silero VAD
  ↓
Speech Segments
  ↓
Extract Speech Audio
  ↓
faster-whisper
  ↓
Clean Transcription
  ↓
ReasonIQ NLP Engine
  ↓
Structured Analysis
"""

import sys
import wave
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# ------------------------------------------------------------
# Make backend available for imports
# ------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from speech.vad import BiasLessVAD, ReasonIQVAD
from speech.transcriber import BiasLessTranscriber, ReasonIQTranscriber
from nlp.engine import BiasLessNLPEngine, ReasonIQNLPEngine


class ReasonIQSpeechPipeline:

    def __init__(self, whisper_model: str = "base"):

        print("Initializing ReasonIQ Speech Pipeline...")

        # ----------------------------------------------------
        # Load Voice Activity Detection
        # ----------------------------------------------------

        print("\nLoading Voice Activity Detection...")

        self.vad = ReasonIQVAD()

        # ----------------------------------------------------
        # Load Whisper
        # ----------------------------------------------------

        print("\nLoading Speech-to-Text...")

        self.transcriber = ReasonIQTranscriber(
            model_size=whisper_model,
            device="cpu",
            compute_type="int8"
        )

        # ----------------------------------------------------
        # Load complete NLP engine
        # ----------------------------------------------------

        print("\nLoading ReasonIQ NLP Engine...")

        self.nlp_engine = ReasonIQNLPEngine()

        print("\nSpeech + NLP pipeline ready.")

    # --------------------------------------------------------
    # Save a speech segment as a temporary WAV file
    # --------------------------------------------------------

    def save_segment(
        self,
        audio: torch.Tensor,
        start: float,
        end: float,
        sample_rate: int
    ) -> str:

        start_sample = int(
            start * sample_rate
        )

        end_sample = int(
            end * sample_rate
        )

        segment = audio[
            start_sample:end_sample
        ]

        # ----------------------------------------------------
        # Convert float32 -> int16 PCM
        # ----------------------------------------------------

        segment_np = (
            segment
            .clamp(-1.0, 1.0)
            .numpy()
        )

        pcm = (
            segment_np * 32767
        ).astype(
            np.int16
        )

        # ----------------------------------------------------
        # Create temporary WAV
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp_file.name

        temp_file.close()

        with wave.open(
            temp_path,
            "wb"
        ) as wav:

            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)

            wav.writeframes(
                pcm.tobytes()
            )

        return temp_path

    # --------------------------------------------------------
    # Process complete audio
    # --------------------------------------------------------

    def process(
        self,
        audio_path: str
    ) -> Dict:

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        # ====================================================
        # STEP 1 - VOICE ACTIVITY DETECTION
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 1 - VOICE ACTIVITY DETECTION")
        print("=" * 60)

        # Load and normalize audio once
        audio, sample_rate = (
            self.vad.load_wav(
                str(path)
            )
        )

        # Detect speech
        speech_segments = (
            self.vad.get_speech_timestamps(
                audio,
                self.vad.model,
                sampling_rate=sample_rate
            )
        )

        print(
            f"Detected "
            f"{len(speech_segments)} "
            f"possible speech segment(s)."
        )

        results = []

        # ====================================================
        # PROCESS EACH VAD SEGMENT
        # ====================================================

        for index, segment in enumerate(
            speech_segments,
            start=1
        ):

            start = (
                segment["start"]
                / sample_rate
            )

            end = (
                segment["end"]
                / sample_rate
            )

            duration = end - start

            print("\n" + "-" * 60)

            print(
                f"Segment {index}: "
                f"{start:.2f}s - "
                f"{end:.2f}s "
                f"({duration:.2f}s)"
            )

            # ------------------------------------------------
            # Ignore extremely short segments
            # ------------------------------------------------

            if duration < 0.30:

                print(
                    "Skipped: segment too short."
                )

                continue

            # ------------------------------------------------
            # STEP 2 - EXTRACT SPEECH AUDIO
            # ------------------------------------------------

            print(
                "Extracting speech segment..."
            )

            temp_path = self.save_segment(
                audio,
                start,
                end,
                sample_rate
            )

            try:

                # ============================================
                # STEP 3 - SPEECH TO TEXT
                # ============================================

                print(
                    "Running Whisper..."
                )

                transcription = (
                    self.transcriber
                    .transcribe_file(
                        temp_path,
                        language="en"
                    )
                )

                text = (
                    transcription["text"]
                    .strip()
                )

                # ------------------------------------------------
                # Keep only segments containing recognizable text
                # ------------------------------------------------

                if not text:

                    print(
                        "No recognizable speech. "
                        "Segment discarded."
                    )

                    continue

                print(
                    f"Transcription: {text}"
                )

                # ============================================
                # STEP 4 - NLP ANALYSIS
                # ============================================

                print(
                    "Running ReasonIQ NLP analysis..."
                )

                nlp_result = (
                    self.nlp_engine.analyze(
                        text
                    )
                )

                print(
                    "NLP analysis completed."
                )

                # ============================================
                # STORE COMPLETE RESULT
                # ============================================

                results.append({

                    "start": round(
                        start,
                        2
                    ),

                    "end": round(
                        end,
                        2
                    ),

                    "duration": round(
                        duration,
                        2
                    ),

                    "text": text,

                    "nlp_analysis": nlp_result
                })

            finally:

                # ------------------------------------------------
                # Delete temporary WAV
                # ------------------------------------------------

                temp_file = Path(
                    temp_path
                )

                if temp_file.exists():

                    temp_file.unlink()

        # ====================================================
        # COMBINE RECOGNIZED SPEECH
        # ====================================================

        full_text = " ".join(
            segment["text"]
            for segment in results
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            "audio": str(path),

            "language": "en",

            "segments": results,

            "text": full_text
        }


# Backward compatibility alias
BiasLessSpeechPipeline = ReasonIQSpeechPipeline


# ============================================================
# TEST FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("BiasLess Speech + NLP Pipeline Test")
    print("=" * 60)

    audio_file = (
        r"C:\Users\Rasika\BiasLess"
        r"\tests\sample_audio.wav"
    )

    try:

        # ----------------------------------------------------
        # Initialize pipeline
        # ----------------------------------------------------

        pipeline = BiasLessSpeechPipeline(
            whisper_model="base"
        )

        # ----------------------------------------------------
        # Process audio
        # ----------------------------------------------------

        result = pipeline.process(
            audio_file
        )

        # ====================================================
        # DISPLAY FINAL RESULT
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL PIPELINE RESULT")
        print("=" * 60)

        # ----------------------------------------------------
        # Final transcript
        # ----------------------------------------------------

        print(
            f"\nFinal Transcript:\n"
            f"{result['text']}"
        )

        # ----------------------------------------------------
        # Recognized segments
        # ----------------------------------------------------

        print("\nRecognized Segments:")

        if not result["segments"]:

            print(
                "No recognizable speech found."
            )

        else:

            for i, segment in enumerate(
                result["segments"],
                start=1
            ):

                print(
                    f"\n{i}. "
                    f"[{segment['start']:.2f}s - "
                    f"{segment['end']:.2f}s] "
                    f"{segment['text']}"
                )

                # ============================================
                # DISPLAY NLP RESULT
                # ============================================

                analysis = (
                    segment.get(
                        "nlp_analysis",
                        {}
                    )
                )

                print(
                    "\n   NLP Analysis:"
                )

                # ------------------------------------------------
                # Argument component
                # ------------------------------------------------

                argument = analysis.get(
                    "argument_component",
                    {}
                )

                if argument:

                    print(
                        f"   Argument Component: "
                        f"{argument}"
                    )

                # ------------------------------------------------
                # Fallacy
                # ------------------------------------------------

                fallacy = analysis.get(
                    "fallacy",
                    {}
                )

                if fallacy:

                    print(
                        f"   Fallacy: "
                        f"{fallacy}"
                    )

                # ------------------------------------------------
                # Bias
                # ------------------------------------------------

                bias = analysis.get(
                    "bias",
                    {}
                )

                if bias:

                    print(
                        f"   Bias Signal: "
                        f"{bias}"
                    )

                # ------------------------------------------------
                # Sentiment
                # ------------------------------------------------

                sentiment = analysis.get(
                    "sentiment",
                    {}
                )

                if sentiment:

                    print(
                        f"   Sentiment: "
                        f"{sentiment}"
                    )

                # ------------------------------------------------
                # Evidence
                # ------------------------------------------------

                evidence = analysis.get(
                    "evidence",
                    {}
                )

                if evidence:

                    print(
                        f"   Evidence: "
                        f"{evidence}"
                    )

                # ------------------------------------------------
                # Quality
                # ------------------------------------------------

                quality = analysis.get(
                    "quality",
                    {}
                )

                if quality:

                    print(
                        f"   Argument Quality: "
                        f"{quality}"
                    )

                # ------------------------------------------------
                # Counterargument
                # ------------------------------------------------

                counterargument = analysis.get(
                    "counterargument",
                    {}
                )

                if counterargument:

                    print(
                        f"   Counterargument: "
                        f"{counterargument}"
                    )

        print("\n" + "=" * 60)

    except Exception as error:

        print("\nSpeech + NLP pipeline failed.")

        print(
            f"Error: {error}"
        )

        # ----------------------------------------------------
        # Useful for debugging during development
        # ----------------------------------------------------

        import traceback

        traceback.print_exc()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()