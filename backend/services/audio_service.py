"""
ReasonIQ - Audio Conversion and Speech-to-NLP Processing Service

Acts purely as a robust adapter between incoming browser audio streams
(WebM/Opus, MP4, Ogg, WAV, etc.) and the existing ReasonIQSpeechPipeline.
Uses PyAV (av) to decode any container format into standard 16-bit 16kHz mono PCM WAV.

Provides detailed stage-aware error reporting:
- AUDIO_ERROR: Failed to decode audio bytes
- NO_SPEECH: No voice activity or empty transcript
- TRANSCRIPTION_ERROR: Speech-to-text failure
- NLP_ERROR: NLP engine failure (preserves transcript)
- COMPLETED: Full successful analysis
"""

import io
import os
import wave
import uuid
import tempfile
import traceback
import numpy as np
import av
from typing import Dict, Any, Optional, Union

from speech.pipeline import ReasonIQSpeechPipeline, BiasLessSpeechPipeline


class AudioService:
    def __init__(self, speech_pipeline: Optional[Union[ReasonIQSpeechPipeline, BiasLessSpeechPipeline]] = None):
        self.speech_pipeline = speech_pipeline

    def set_pipeline(self, pipeline: Union[ReasonIQSpeechPipeline, BiasLessSpeechPipeline]):
        self.speech_pipeline = pipeline

    def convert_to_wav(
        self,
        audio_bytes: bytes,
        target_sample_rate: int = 16000,
        utterance_id: Optional[str] = None
    ) -> str:
        """
        Decodes arbitrary browser audio bytes (WebM, Opus, MP4, WAV, OGG, etc.)
        using PyAV and writes a 16-bit mono 16kHz PCM WAV file to a unique temporary location.
        Returns the path to the temporary WAV file.
        """
        uid = utterance_id or str(uuid.uuid4())
        input_buffer = io.BytesIO(audio_bytes)
        temp_wav_path = os.path.join(tempfile.gettempdir(), f"reasoniq_utterance_{uid}.wav")

        # -----------------------------------------------------
        # Strategy A: PyAV Container Decoding
        # -----------------------------------------------------
        pcm_frames = []
        try:
            container = av.open(input_buffer)
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            
            if audio_stream:
                resampler = av.AudioResampler(
                    format='s16',
                    layout='mono',
                    rate=target_sample_rate
                )

                for packet in container.demux(audio_stream):
                    try:
                        for frame in packet.decode():
                            resampled_frames = resampler.resample(frame)
                            if resampled_frames:
                                for rf in resampled_frames:
                                    pcm_frames.append(rf.to_ndarray().tobytes())
                    except Exception as packet_err:
                        # Continue processing previous frames if trailing packet is truncated
                        print(f"[AudioService] Warning decoding packet: {packet_err}")
                        continue

                # Flush resampler buffer
                try:
                    flushed_frames = resampler.resample(None)
                    if flushed_frames:
                        for rf in flushed_frames:
                            pcm_frames.append(rf.to_ndarray().tobytes())
                except Exception as flush_err:
                    print(f"[AudioService] Warning flushing resampler: {flush_err}")

            container.close()

        except Exception as pyav_err:
            print(f"[AudioService] PyAV container opening error: {pyav_err}")

        # If PyAV extracted frames successfully:
        if pcm_frames:
            try:
                raw_bytes = b"".join(pcm_frames)
                # Check amplitude and apply safe gain normalization if speech is quiet
                if len(raw_bytes) >= 2:
                    audio_arr = np.frombuffer(raw_bytes, dtype=np.int16)
                    peak = int(np.max(np.abs(audio_arr)))
                    if 150 < peak < 10000:
                        scale = min(3.5, 18000.0 / max(1, peak))
                        audio_arr = np.clip(audio_arr * scale, -32767, 32767).astype(np.int16)
                        raw_bytes = audio_arr.tobytes()

                with wave.open(temp_wav_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                    wav_file.setframerate(target_sample_rate)
                    wav_file.writeframes(raw_bytes)
                return temp_wav_path
            except Exception as write_err:
                print(f"[AudioService] Error writing decoded WAV: {write_err}")

        # -----------------------------------------------------
        # Strategy B: Fallback for Raw WAV Files
        # -----------------------------------------------------
        try:
            input_buffer.seek(0)
            if audio_bytes[:4] == b'RIFF':
                with wave.open(io.BytesIO(audio_bytes), "rb") as raw_wav:
                    n_channels = raw_wav.getnchannels()
                    sampwidth = raw_wav.getsampwidth()
                    framerate = raw_wav.getframerate()
                    raw_frames = raw_wav.readframes(raw_wav.getnframes())

                    audio_data = np.frombuffer(raw_frames, dtype=np.int16)
                    if n_channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

                    # Resample if not 16kHz
                    if framerate != target_sample_rate and len(audio_data) > 0:
                        from scipy.signal import resample_poly
                        audio_data = resample_poly(audio_data, target_sample_rate, framerate).astype(np.int16)

                    peak = int(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0
                    if 150 < peak < 10000:
                        scale = min(3.5, 18000.0 / max(1, peak))
                        audio_data = np.clip(audio_data * scale, -32767, 32767).astype(np.int16)

                    with wave.open(temp_wav_path, "wb") as out_wav:
                        out_wav.setnchannels(1)
                        out_wav.setsampwidth(2)
                        out_wav.setframerate(target_sample_rate)
                        out_wav.writeframes(audio_data.tobytes())
                    return temp_wav_path
        except Exception as wav_fallback_err:
            print(f"[AudioService] WAV fallback error: {wav_fallback_err}")

        raise ValueError(f"Could not decode audio stream (input size: {len(audio_bytes)} bytes).")

    def process_audio(
        self,
        audio_bytes: bytes,
        utterance_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        speaker_label: Optional[str] = None,
        room_code: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes audio bytes through the speech pipeline with stage-aware isolation:
        1. PyAV decoding -> 16kHz mono WAV
        2. Silero VAD -> voice activity check
        3. faster-whisper -> transcription
        4. ReasonIQ NLP Engine -> analytical breakdown
        """
        import time
        t_start = time.time()
        uid = utterance_id or str(uuid.uuid4())
        audio_size = len(audio_bytes) if audio_bytes else 0

        if not self.speech_pipeline:
            try:
                print("[AUDIO_SERVICE] Initializing ReasonIQSpeechPipeline...")
                self.speech_pipeline = ReasonIQSpeechPipeline()
            except Exception as init_err:
                return {
                    "status": "AUDIO_ERROR",
                    "stage": "initialization",
                    "transcript": "",
                    "nlp_analysis": None,
                    "segments": [],
                    "error_message": f"Speech pipeline initialization failed: {str(init_err)}",
                    "utterance_id": uid
                }

        if not audio_bytes or len(audio_bytes) == 0:
            print(f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} mime={mime_type} size={audio_size} error=NO_SPEECH_EMPTY")
            return {
                "status": "NO_SPEECH",
                "stage": "vad",
                "transcript": "",
                "nlp_analysis": None,
                "segments": [],
                "error_message": "No recognizable speech detected.",
                "error": "No recognizable speech detected.",
                "utterance_id": uid
            }

        temp_wav_path = None

        # -----------------------------------------------------
        # STAGE 1: AUDIO CONVERSION
        # -----------------------------------------------------
        try:
            temp_wav_path = self.convert_to_wav(audio_bytes, utterance_id=uid)
        except Exception as conv_err:
            print(f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} mime={mime_type} size={audio_size} error=AUDIO_CONVERSION_ERROR: {conv_err}")
            return {
                "status": "AUDIO_ERROR",
                "stage": "audio_conversion",
                "transcript": "",
                "nlp_analysis": None,
                "segments": [],
                "error_message": f"Audio conversion failed: {str(conv_err)}",
                "error": f"Audio conversion failed: {str(conv_err)}",
                "utterance_id": uid
            }

        # -----------------------------------------------------
        # STAGE 2: VAD & TRANSCRIPTION
        # -----------------------------------------------------
        try:
            pipeline_result = self.speech_pipeline.process(temp_wav_path)
            full_text = pipeline_result.get("text", "").strip()
            segments = pipeline_result.get("segments", [])
            transcription_warning = pipeline_result.get("transcription_warning")
            speech_dur = pipeline_result.get("speech_duration", 0.0)
            total_dur = pipeline_result.get("total_duration", 0.0)
            model_name = getattr(self.speech_pipeline.transcriber, "model_size", "base")

            if not full_text or len(segments) == 0:
                print(f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} mime={mime_type} size={audio_size} audio_duration={total_dur}s speech_duration=0s error=NO_SPEECH_DETECTED")
                return {
                    "status": "NO_SPEECH",
                    "stage": "vad",
                    "transcript": "",
                    "nlp_analysis": None,
                    "segments": [],
                    "error_message": "No recognizable speech detected.",
                    "error": "No recognizable speech detected.",
                    "utterance_id": uid
                }

        except Exception as speech_err:
            print(f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} mime={mime_type} size={audio_size} error=TRANSCRIPTION_ERROR: {speech_err}")
            return {
                "status": "TRANSCRIPTION_ERROR",
                "stage": "transcription",
                "transcript": "",
                "nlp_analysis": None,
                "segments": [],
                "error_message": f"Speech transcription failed: {str(speech_err)}",
                "error": f"Speech transcription failed: {str(speech_err)}",
                "utterance_id": uid
            }

        # -----------------------------------------------------
        # STAGE 3: NLP ANALYSIS (Preserves transcript on error)
        # -----------------------------------------------------
        nlp_analysis = None
        try:
            if pipeline_result.get("nlp_analysis"):
                nlp_analysis = pipeline_result["nlp_analysis"]
            elif len(segments) == 1 and segments[0].get("nlp_analysis"):
                nlp_analysis = segments[0]["nlp_analysis"]
            else:
                nlp_analysis = self.speech_pipeline.nlp_engine.analyze(full_text, transcription_warning=transcription_warning)

            # Attach transcription warning if present
            if transcription_warning and nlp_analysis:
                nlp_analysis["transcription_warning"] = transcription_warning

            proc_time = round(time.time() - t_start, 2)
            print(
                f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} "
                f"mime={mime_type} size={audio_size} audio_duration={total_dur}s "
                f"vad_segments={len(segments)} speech_duration={speech_dur}s "
                f"model={model_name} proc_time={proc_time}s transcript='{full_text}'"
            )

            return {
                "status": "COMPLETED",
                "stage": "completed",
                "transcript": full_text,
                "transcription_warning": transcription_warning,
                "nlp_analysis": nlp_analysis,
                "segments": segments,
                "error_message": None,
                "error": None,
                "utterance_id": uid
            }

        except Exception as nlp_err:
            proc_time = round(time.time() - t_start, 2)
            print(f"[DIAGNOSTICS] room={room_code} utterance={uid} speaker={speaker_label} transcript='{full_text}' error=NLP_ERROR: {nlp_err}")
            return {
                "status": "NLP_ERROR",
                "stage": "nlp",
                "transcript": full_text,  # Transcript preserved!
                "transcription_warning": transcription_warning,
                "nlp_analysis": None,
                "segments": segments,
                "error_message": f"NLP analysis failed: {str(nlp_err)}",
                "error": f"NLP analysis failed: {str(nlp_err)}",
                "utterance_id": uid
            }

        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass


# Singleton instance
audio_service = AudioService()


def convert_to_wav(
    audio_bytes: bytes,
    target_sample_rate: int = 16000,
    utterance_id: Optional[str] = None
) -> bytes:
    """
    Convenience function that converts audio bytes to 16kHz mono WAV bytes.
    """
    path = audio_service.convert_to_wav(audio_bytes, target_sample_rate, utterance_id)
    try:
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass


def process_audio_file(
    audio_bytes: bytes,
    utterance_id: Optional[str] = None,
    participant_id: Optional[str] = None,
    speaker_label: Optional[str] = None,
    room_code: Optional[str] = None,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function that processes audio bytes through the speech and NLP pipeline.
    """
    return audio_service.process_audio(
        audio_bytes=audio_bytes,
        utterance_id=utterance_id,
        participant_id=participant_id,
        speaker_label=speaker_label,
        room_code=room_code
    )

