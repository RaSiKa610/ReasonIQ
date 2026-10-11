"""
ReasonIQ - In-Memory Runtime Room Manager

Manages live debate rooms, participants, dynamic speaker label assignment,
room state transitions, and utterance history in memory.

Strictly runtime in-memory state for Phase 1.
"""

import time
import uuid
import random
import string
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Pydantic Models for State Representation
# ------------------------------------------------------------

class ParticipantState(BaseModel):
    participant_id: str
    room_code: str
    name: str
    role: str = "participant"  # "host" or "participant"
    speaker_label: Optional[str] = None  # "Speaker A", "Speaker B", etc.
    status: str = "WAITING"  # "WAITING", "ADMITTED", "REJECTED"
    connected: bool = True
    mic_active: bool = False
    joined_at: float = Field(default_factory=time.time)


class UtteranceState(BaseModel):
    utterance_id: str
    room_code: str
    participant_id: str
    speaker_label: str
    speaker_name: str
    transcript: str = ""
    status: str = "PROCESSING"  # "PROCESSING", "COMPLETED", "NO_SPEECH", "ERROR"
    nlp_analysis: Optional[Dict] = None
    created_at: float = Field(default_factory=time.time)
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class RoomState(BaseModel):
    room_id: str
    room_code: str
    topic: str
    host_name: str
    host_token: str  # Secret authorization token for host operations
    status: str = "WAITING"  # "WAITING", "READY", "LIVE", "ENDED"
    allow_interruptions: bool = False
    participants: Dict[str, ParticipantState] = Field(default_factory=dict)
    speaker_letters: List[str] = Field(default_factory=lambda: list(string.ascii_uppercase))
    next_speaker_idx: int = 0
    utterances: List[UtteranceState] = Field(default_factory=list)
    final_reports: Dict[str, Dict] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


# ------------------------------------------------------------
# Room Manager Class
# ------------------------------------------------------------

class RoomManager:
    """
    In-memory manager for live debate rooms.
    """

    def __init__(self):
        # Keyed by uppercase room_code
        self.rooms: Dict[str, RoomState] = {}

    def _generate_room_code(self, length: int = 6) -> str:
        """
        Generate a unique, uppercase alphanumeric room code.
        E.g. RQ7K2Q
        """
        chars = string.ascii_uppercase + string.digits
        # Avoid visually ambiguous characters like O, 0, I, 1
        safe_chars = [c for c in chars if c not in ('O', '0', 'I', '1')]
        
        while True:
            # Prefix with 'BL' or generate 6 random characters
            code = "BL" + "".join(random.choices(safe_chars, k=length - 2))
            if code not in self.rooms:
                return code

    def create_room(self, host_name: str, topic: str) -> RoomState:
        """
        Dynamically create a new room in memory.
        """
        if not host_name or not host_name.strip():
            raise ValueError("Host name cannot be empty.")
        if not topic or not topic.strip():
            raise ValueError("Debate topic cannot be empty.")

        room_code = self._generate_room_code()
        room_id = str(uuid.uuid4())
        host_token = str(uuid.uuid4())

        room = RoomState(
            room_id=room_id,
            room_code=room_code,
            topic=topic.strip(),
            host_name=host_name.strip(),
            host_token=host_token,
            status="WAITING"
        )

        self.rooms[room_code] = room
        return room

    def get_room(self, room_code: str) -> Optional[RoomState]:
        """
        Get room by room_code (case-insensitive).
        """
        return self.rooms.get(room_code.strip().upper())

    def verify_host(self, room_code: str, host_token: str) -> bool:
        """
        Verify that the given host_token matches the room's host_token.
        """
        room = self.get_room(room_code)
        if not room:
            return False
        return room.host_token == host_token

    def join_request(self, room_code: str, participant_name: str) -> ParticipantState:
        """
        A participant requests to join a room.
        Creates a participant in 'WAITING' status.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError(f"Room '{room_code}' does not exist.")

        if room.status == "ENDED":
            raise ValueError("Cannot join a debate that has already ended.")

        if not participant_name or not participant_name.strip():
            raise ValueError("Participant name cannot be empty.")

        # Check for duplicate names in the room
        clean_name = participant_name.strip()
        for p in room.participants.values():
            if p.name.lower() == clean_name.lower() and p.status != "REJECTED":
                raise ValueError(f"A participant named '{clean_name}' is already in this room.")

        participant_id = str(uuid.uuid4())
        participant = ParticipantState(
            participant_id=participant_id,
            room_code=room.room_code,
            name=clean_name,
            role="participant",
            status="WAITING"
        )

        room.participants[participant_id] = participant
        return participant

    def admit_participant(self, room_code: str, host_token: str, participant_id: str) -> ParticipantState:
        """
        Host admits a participant.
        Dynamically assigns the next available speaker label (Speaker A, Speaker B, etc.).
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if not self.verify_host(room_code, host_token):
            raise PermissionError("Unauthorized: Invalid host token.")

        if participant_id not in room.participants:
            raise ValueError(f"Participant '{participant_id}' not found in room.")

        participant = room.participants[participant_id]

        if participant.status == "ADMITTED":
            return participant

        # Assign speaker label
        idx = room.next_speaker_idx
        if idx < len(room.speaker_letters):
            label_char = room.speaker_letters[idx]
            speaker_label = f"Speaker {label_char}"
        else:
            # Fallback if more than 26 speakers
            speaker_label = f"Speaker {idx + 1}"

        room.next_speaker_idx += 1
        participant.speaker_label = speaker_label
        participant.status = "ADMITTED"

        # If room was WAITING and we now have admitted participants, room is READY
        if room.status == "WAITING":
            room.status = "READY"

        return participant

    def reject_participant(self, room_code: str, host_token: str, participant_id: str) -> ParticipantState:
        """
        Host rejects a participant.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if not self.verify_host(room_code, host_token):
            raise PermissionError("Unauthorized: Invalid host token.")

        if participant_id not in room.participants:
            raise ValueError(f"Participant '{participant_id}' not found in room.")

        participant = room.participants[participant_id]
        participant.status = "REJECTED"
        return participant

    def start_debate(self, room_code: str, host_token: str) -> RoomState:
        """
        Transition room from READY (or WAITING) to LIVE.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if not self.verify_host(room_code, host_token):
            raise PermissionError("Unauthorized: Invalid host token.")

        if room.status == "ENDED":
            raise ValueError("Cannot start a debate that has already ended.")

        if room.status == "LIVE":
            return room

        room.status = "LIVE"
        return room

    def update_settings(
        self,
        room_code: str,
        host_token: str,
        allow_interruptions: bool
    ) -> RoomState:
        """
        Host updates debate settings atomically without resetting room or participants.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if not self.verify_host(room_code, host_token):
            raise PermissionError("Unauthorized: Invalid host token.")

        if room.status == "ENDED":
            raise ValueError("Cannot modify debate settings because the debate has ended.")

        if not isinstance(allow_interruptions, bool) or isinstance(allow_interruptions, int) and allow_interruptions not in (True, False):
            raise ValueError("allow_interruptions must be a boolean.")

        # Atomic mutation of independent setting
        room.allow_interruptions = bool(allow_interruptions)
        return room

    def end_debate(self, room_code: str, host_token: str) -> RoomState:
        """
        Transition room from LIVE to ENDED and generate authoritative speaker reports.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if not self.verify_host(room_code, host_token):
            raise PermissionError("Unauthorized: Invalid host token.")

        room.status = "ENDED"

        # Generate reports for all admitted participants
        from services.report_service import report_service
        for p in room.participants.values():
            if p.status == "ADMITTED":
                report = report_service.generate_speaker_report(room, p)
                room.final_reports[p.participant_id] = report

        return room

    def create_utterance(
        self,
        room_code: str,
        participant_id: str,
        utterance_id: Optional[str] = None
    ) -> UtteranceState:
        """
        Create a new utterance entry before processing audio.
        Validates that the room is LIVE and participant is ADMITTED.
        """
        room = self.get_room(room_code)
        if not room:
            raise ValueError("Room not found.")

        if room.status != "LIVE":
            raise ValueError(f"Cannot submit audio: Room is not LIVE (current status: {room.status}).")

        if participant_id not in room.participants:
            raise ValueError(f"Participant '{participant_id}' does not belong to this room.")

        participant = room.participants[participant_id]
        if participant.status != "ADMITTED":
            raise ValueError(f"Participant '{participant.name}' is not admitted.")

        uid = utterance_id or str(uuid.uuid4())
        utterance = UtteranceState(
            utterance_id=uid,
            room_code=room.room_code,
            participant_id=participant_id,
            speaker_label=participant.speaker_label or "Unknown Speaker",
            speaker_name=participant.name,
            status="PROCESSING"
        )

        room.utterances.append(utterance)
        return utterance

    def complete_utterance(
        self,
        room_code: str,
        utterance_id: str,
        transcript: str,
        nlp_analysis: Optional[Dict],
        duration_seconds: Optional[float] = None
    ) -> Optional[UtteranceState]:
        """
        Update an utterance with its completed transcript and NLP analysis.
        """
        room = self.get_room(room_code)
        if not room:
            return None

        for u in room.utterances:
            if u.utterance_id == utterance_id:
                u.transcript = transcript
                u.nlp_analysis = nlp_analysis
                u.status = "COMPLETED"
                u.duration_seconds = duration_seconds
                return u

        return None

    def mark_utterance_no_speech(
        self,
        room_code: str,
        utterance_id: str
    ) -> Optional[UtteranceState]:
        """
        Mark utterance as NO_SPEECH when no recognizable speech was detected.
        """
        room = self.get_room(room_code)
        if not room:
            return None

        for u in room.utterances:
            if u.utterance_id == utterance_id:
                u.status = "NO_SPEECH"
                u.transcript = ""
                u.nlp_analysis = None
                return u

        return None

    def mark_utterance_error(
        self,
        room_code: str,
        utterance_id: str,
        error_message: str
    ) -> Optional[UtteranceState]:
        """
        Mark utterance as ERROR if transcription or analysis fails.
        """
        room = self.get_room(room_code)
        if not room:
            return None

        for u in room.utterances:
            if u.utterance_id == utterance_id:
                u.status = "ERROR"
                u.error_message = error_message
                return u

        return None


# Global singleton instance for in-memory room management
room_manager = RoomManager()
