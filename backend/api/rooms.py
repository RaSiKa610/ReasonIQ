"""
ReasonIQ - Room API Endpoints

Handles room creation, joining, host admission/rejection, debate settings,
debate lifecycle, audio utterance submission with server-enforced counterargument privacy,
and final performance reports.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header, Query, UploadFile, File, Form, Depends, Request
from pydantic import BaseModel

from services.room_manager import room_manager
from services.audio_service import audio_service
from services.report_service import report_service
from api.websocket import ws_manager

router = APIRouter(prefix="/api/rooms", tags=["rooms"])


# ------------------------------------------------------------
# Request Models
# ------------------------------------------------------------

class CreateRoomRequest(BaseModel):
    host_name: str
    topic: str
    allow_interruptions: bool = False


class JoinRoomRequest(BaseModel):
    participant_name: Optional[str] = None
    name: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.participant_name or self.name or "Participant"



class UpdateSettingsRequest(BaseModel):
    allow_interruptions: bool


# ------------------------------------------------------------
# Helper Dependency for Host Token
# ------------------------------------------------------------

def get_host_token(
    x_host_token: Optional[str] = Header(None, alias="X-Host-Token"),
    token: Optional[str] = Query(None)
) -> str:
    effective_token = x_host_token or token
    if not effective_token:
        raise HTTPException(
            status_code=401,
            detail="Missing host authorization token (provide 'X-Host-Token' header or 'token' query parameter)."
        )
    return effective_token


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@router.post("")
@router.post("/create")
async def create_room(req: CreateRoomRequest):
    """
    Host creates a new debate room dynamically in runtime memory.
    """
    try:
        room = room_manager.create_room(
            host_name=req.host_name,
            topic=req.topic
        )
        room.allow_interruptions = req.allow_interruptions
        return {
            "success": True,
            "room_code": room.room_code,
            "room_id": room.room_id,
            "topic": room.topic,
            "host_name": room.host_name,
            "host_token": room.host_token,
            "status": room.status,
            "allow_interruptions": room.allow_interruptions,
            "created_at": room.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{room_code}")
async def get_room(
    room_code: str,
    x_host_token: Optional[str] = Header(None, alias="X-Host-Token"),
    x_participant_id: Optional[str] = Header(None, alias="X-Participant-Id")
):
    """
    Get current public room details, participants, and status.
    Sanitizes private counterarguments unless requested by the host or utterance author.
    """
    room = room_manager.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    is_host = room_manager.verify_host(room_code, x_host_token) if x_host_token else False

    sanitized_utterances = []
    for u in room.utterances:
        u_dict = u.model_dump()
        # If not host and not the author of this utterance, strip private counterargument
        if not is_host and (not x_participant_id or u.participant_id != x_participant_id):
            if u_dict.get("nlp_analysis") and isinstance(u_dict["nlp_analysis"], dict):
                u_dict["nlp_analysis"] = {
                    k: v for k, v in u_dict["nlp_analysis"].items()
                    if k != "counterargument"
                }
        sanitized_utterances.append(u_dict)

    return {
        "success": True,
        "room_code": room.room_code,
        "topic": room.topic,
        "host_name": room.host_name,
        "status": room.status,
        "allow_interruptions": room.allow_interruptions,
        "participants": [p.model_dump() for p in room.participants.values()],
        "utterances": sanitized_utterances,
        "has_final_reports": len(room.final_reports) > 0,
        "created_at": room.created_at
    }


@router.post("/{room_code}/settings")
async def update_settings(
    room_code: str,
    req: Request,
    host_token: str = Depends(get_host_token)
):
    """
    Host updates debate room settings (e.g. allow_interruptions rule).
    Validates atomic update and broadcasts room settings update event.
    """
    code = room_code.upper()
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ROOM_SETTING",
                "message": "Invalid JSON body."
            }
        )

    if not isinstance(body, dict) or "allow_interruptions" not in body:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ROOM_SETTING",
                "message": "allow_interruptions field is required."
            }
        )

    allow_int = body.get("allow_interruptions")
    # Strict boolean validation
    if not isinstance(allow_int, bool) or (isinstance(allow_int, int) and allow_int not in (True, False)):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_ROOM_SETTING",
                "message": "allow_interruptions must be a boolean."
            }
        )

    old_room = room_manager.get_room(code)
    old_setting = old_room.allow_interruptions if old_room else None
    print(f"SETTINGS_UPDATE_REQUEST room_code={code} host_token={host_token} old_allow_interruptions={old_setting} new_allow_interruptions={allow_int}")

    try:
        room = room_manager.update_settings(
            room_code=code,
            host_token=host_token,
            allow_interruptions=allow_int
        )

        print(f"SETTINGS_UPDATE_SUCCESS room_code={code} allow_interruptions={room.allow_interruptions}")

        await ws_manager.broadcast(
            code,
            "ROOM_SETTINGS_UPDATED",
            {
                "allow_interruptions": room.allow_interruptions
            }
        )
        # Also broadcast SETTINGS_UPDATED for backward compatibility
        await ws_manager.broadcast(
            code,
            "SETTINGS_UPDATED",
            {
                "allow_interruptions": room.allow_interruptions
            }
        )

        return {
            "success": True,
            "allow_interruptions": room.allow_interruptions
        }
    except PermissionError as e:
        print(f"SETTINGS_UPDATE_FAILED room_code={code} error_type=PermissionError safe_message={str(e)}")
        raise HTTPException(
            status_code=403,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Unauthorized: Invalid host token."
            }
        )
    except ValueError as e:
        err_msg = str(e)
        print(f"SETTINGS_UPDATE_FAILED room_code={code} error_type=ValueError safe_message={err_msg}")
        is_ended = "ended" in err_msg.lower()
        is_not_found = "not found" in err_msg.lower()
        err_code = "DEBATE_ENDED" if is_ended else ("ROOM_NOT_FOUND" if is_not_found else "INVALID_ROOM_SETTING")
        status_code = 404 if is_not_found else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "code": err_code,
                "message": err_msg
            }
        )
    except Exception as e:
        print(f"SETTINGS_UPDATE_FAILED room_code={code} error_type={type(e).__name__} safe_message=Unable to update debate settings.")
        await ws_manager.broadcast(
            code,
            "ROOM_ERROR",
            {
                "code": "SETTINGS_UPDATE_FAILED",
                "message": "Unable to update the interruption setting.",
                "recoverable": True
            }
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SETTINGS_UPDATE_FAILED",
                "message": "Unable to update debate settings."
            }
        )


@router.post("/{room_code}/join")
async def join_room(room_code: str, req: JoinRoomRequest):
    """
    A participant submits a request to join a room.
    """
    try:
        participant = room_manager.join_request(
            room_code=room_code,
            participant_name=req.effective_name
        )

        # Notify host and room clients via WebSocket
        await ws_manager.broadcast(
            room_code,
            "NEW_JOIN_REQUEST",
            {
                "participant_id": participant.participant_id,
                "name": participant.name,
                "status": participant.status
            }
        )

        return {
            "success": True,
            "participant_id": participant.participant_id,
            "room_code": participant.room_code,
            "name": participant.name,
            "status": participant.status
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{room_code}/participants")
async def list_participants(room_code: str):
    """
    List all participants in the room.
    """
    room = room_manager.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    return {
        "success": True,
        "participants": [p.model_dump() for p in room.participants.values()]
    }


@router.post("/{room_code}/participants/{participant_id}/admit")
@router.post("/{room_code}/admit/{participant_id}")
async def admit_participant(
    room_code: str,
    participant_id: str,
    host_token: str = Depends(get_host_token)
):
    """
    Host admits a participant and dynamically assigns the next speaker label.
    """
    try:
        participant = room_manager.admit_participant(
            room_code=room_code,
            host_token=host_token,
            participant_id=participant_id
        )

        room = room_manager.get_room(room_code)

        # Broadcast admission to everyone in the room
        await ws_manager.broadcast(
            room_code,
            "PARTICIPANT_ADMITTED",
            {
                "participant_id": participant.participant_id,
                "name": participant.name,
                "speaker_label": participant.speaker_label,
                "status": participant.status,
                "room_status": room.status if room else "READY"
            }
        )

        # Send targeted notification directly to the admitted participant
        await ws_manager.send_to_participant(
            room_code,
            participant_id,
            "YOU_ARE_ADMITTED",
            {
                "participant_id": participant.participant_id,
                "speaker_label": participant.speaker_label,
                "room_code": room_code
            }
        )

        return {
            "success": True,
            "participant": participant.model_dump()
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{room_code}/participants/{participant_id}/reject")
@router.post("/{room_code}/reject/{participant_id}")
async def reject_participant(
    room_code: str,
    participant_id: str,
    host_token: str = Depends(get_host_token)
):
    """
    Host rejects a participant's join request.
    """
    try:
        participant = room_manager.reject_participant(
            room_code=room_code,
            host_token=host_token,
            participant_id=participant_id
        )

        await ws_manager.broadcast(
            room_code,
            "PARTICIPANT_REJECTED",
            {
                "participant_id": participant.participant_id,
                "name": participant.name
            }
        )

        await ws_manager.send_to_participant(
            room_code,
            participant_id,
            "YOU_ARE_REJECTED",
            {
                "participant_id": participant.participant_id
            }
        )

        return {
            "success": True,
            "participant": participant.model_dump()
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{room_code}/start")
async def start_debate(
    room_code: str,
    host_token: str = Depends(get_host_token)
):
    """
    Host starts the debate, transitioning the room to LIVE.
    """
    try:
        room = room_manager.start_debate(
            room_code=room_code,
            host_token=host_token
        )

        await ws_manager.broadcast(
            room_code,
            "DEBATE_STARTED",
            {
                "room_code": room.room_code,
                "status": room.status
            }
        )

        return {
            "success": True,
            "status": room.status
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{room_code}/end")
async def end_debate(
    room_code: str,
    host_token: str = Depends(get_host_token)
):
    """
    Host ends the debate, transitioning the room to ENDED and generating final reports.
    """
    try:
        room = room_manager.end_debate(
            room_code=room_code,
            host_token=host_token
        )

        # Broadcast debate ended globally
        await ws_manager.broadcast(
            room_code,
            "DEBATE_ENDED",
            {
                "room_code": room.room_code,
                "status": room.status
            }
        )

        # Notify each admitted participant that their individual report is available
        for p_id, rep in room.final_reports.items():
            await ws_manager.send_to_participant(
                room_code,
                p_id,
                "FINAL_REPORT_AVAILABLE",
                {
                    "participant_id": p_id,
                    "speaker_label": rep.get("speaker_label"),
                    "final_score": rep.get("final_score"),
                    "performance_tier": rep.get("performance_tier")
                }
            )

        # Notify host with all reports summary
        await ws_manager.send_to_role(
            room_code,
            "host",
            "FINAL_REPORT_AVAILABLE",
            {
                "room_code": room.room_code,
                "reports": {
                    pid: {
                        "speaker_label": r.get("speaker_label"),
                        "speaker_name": r.get("speaker_name"),
                        "final_score": r.get("final_score"),
                        "performance_tier": r.get("performance_tier")
                    } for pid, r in room.final_reports.items()
                }
            }
        )

        return {
            "success": True,
            "status": room.status,
            "reports_count": len(room.final_reports),
            "reports": room.final_reports
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------
# Performance Reports Endpoints
# ------------------------------------------------------------

@router.get("/{room_code}/reports/me")
async def get_my_report(
    room_code: str,
    x_participant_id: Optional[str] = Header(None, alias="X-Participant-Id"),
    participant_id: Optional[str] = Query(None)
):
    """
    Retrieve author's private final performance report (including private counterargument coaching).
    """
    p_id = x_participant_id or participant_id
    if not p_id:
        raise HTTPException(status_code=400, detail="Missing participant identity.")

    code = room_code.upper()
    room = room_manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    if p_id not in room.final_reports:
        # If debate hasn't ended or reports not yet generated, attempt on-the-fly generation if admitted
        if p_id in room.participants and room.participants[p_id].status == "ADMITTED":
            rep = report_service.generate_speaker_report(room, room.participants[p_id])
            return {"success": True, "report": rep}
        raise HTTPException(status_code=404, detail="Report not available for this participant.")

    return {
        "success": True,
        "report": room.final_reports[p_id]
    }


@router.get("/{room_code}/reports/{target_participant_id}")
async def get_participant_report(
    room_code: str,
    target_participant_id: str,
    x_host_token: Optional[str] = Header(None, alias="X-Host-Token"),
    x_participant_id: Optional[str] = Header(None, alias="X-Participant-Id")
):
    """
    Retrieve a specific speaker's report.
    Returns full private report if requested by host or author; otherwise returns sanitized public report.
    """
    code = room_code.upper()
    room = room_manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    if target_participant_id not in room.final_reports:
        if target_participant_id in room.participants and room.participants[target_participant_id].status == "ADMITTED":
            rep = report_service.generate_speaker_report(room, room.participants[target_participant_id])
        else:
            raise HTTPException(status_code=404, detail="Report not found.")
    else:
        rep = room.final_reports[target_participant_id]

    is_host = room_manager.verify_host(code, x_host_token) if x_host_token else False
    is_owner = (x_participant_id == target_participant_id) if x_participant_id else False

    if is_host or is_owner:
        return {"success": True, "report": rep}
    else:
        sanitized = report_service.sanitize_report_for_public(rep)
        return {"success": True, "report": sanitized}


@router.get("/{room_code}/reports")
async def list_all_reports(
    room_code: str,
    host_token: str = Depends(get_host_token)
):
    """
    Host retrieves all participant reports for the debate.
    """
    code = room_code.upper()
    room = room_manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    if not room_manager.verify_host(code, host_token):
        raise HTTPException(status_code=403, detail="Unauthorized host access.")

    return {
        "success": True,
        "reports": room.final_reports
    }


# ------------------------------------------------------------
# Utterance Submission
# ------------------------------------------------------------

@router.post("/{room_code}/utterance")
async def submit_utterance(
    room_code: str,
    participant_id: str = Form(...),
    utterance_id: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Participant submits a completed audio recording of their speaking turn.
    Enforces server-side counterargument privacy:
    - Author receives full private analysis with counterargument coaching.
    - Other room participants receive sanitized public analysis without counterargument.
    """
    upload_file = audio_file or audio
    if upload_file is None:
        raise HTTPException(status_code=400, detail="Missing audio upload file in form data.")

    code = room_code.upper()
    room = room_manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    if room.status != "LIVE":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit audio: Debate is not LIVE (current status: {room.status})."
        )

    if participant_id not in room.participants:
        raise HTTPException(status_code=403, detail="Participant does not belong to this room.")

    participant = room.participants[participant_id]
    if participant.status != "ADMITTED":
        raise HTTPException(status_code=403, detail="Participant is not admitted to this debate.")

    # Interruption check: If interruptions are disabled, check if someone else is actively recording
    if not room.allow_interruptions:
        for other_p in room.participants.values():
            if other_p.participant_id != participant_id and other_p.mic_active:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "SPEAKER_TURN_BLOCKED",
                        "message": f"{other_p.speaker_label or other_p.name} is currently speaking. Please wait for your turn."
                    }
                )

    # Create runtime utterance record
    utterance = room_manager.create_utterance(
        room_code=code,
        participant_id=participant_id,
        utterance_id=utterance_id
    )

    # Broadcast processing started
    await ws_manager.broadcast(
        code,
        "UTTERANCE_STARTED",
        {
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "status": "PROCESSING"
        }
    )

    # Read audio bytes
    audio_bytes = await upload_file.read()
    if not audio_bytes or len(audio_bytes) < 100:
        room_manager.mark_utterance_no_speech(code, utterance.utterance_id)
        await ws_manager.broadcast(
            code,
            "UTTERANCE_NO_SPEECH",
            {
                "utterance_id": utterance.utterance_id,
                "participant_id": participant_id,
                "speaker_label": utterance.speaker_label,
                "speaker_name": utterance.speaker_name,
                "message": "Audio file was empty or too small."
            }
        )
        return {
            "success": True,
            "status": "NO_SPEECH",
            "utterance_id": utterance.utterance_id,
            "message": "Audio file was empty or too small."
        }

    # Look up preceding turn text for conversational context (e.g. straw man distortion checks)
    prev_turn_text = None
    if room and room.utterances:
        for prev_u in reversed(room.utterances):
            if (
                prev_u.utterance_id != utterance.utterance_id
                and prev_u.status == "COMPLETED"
                and prev_u.transcript
            ):
                prev_turn_text = prev_u.transcript
                break

    # Process through PyAV adapter and existing Speech+NLP pipeline with full traceability
    proc_result = audio_service.process_audio(
        audio_bytes=audio_bytes,
        utterance_id=utterance.utterance_id,
        participant_id=participant_id,
        speaker_label=utterance.speaker_label,
        room_code=code,
        mime_type=upload_file.content_type,
        context_prev_text=prev_turn_text
    )

    status = proc_result.get("status")

    if status == "NO_SPEECH":
        room_manager.mark_utterance_no_speech(code, utterance.utterance_id)
        await ws_manager.broadcast(
            code,
            "UTTERANCE_NO_SPEECH",
            {
                "utterance_id": utterance.utterance_id,
                "participant_id": participant_id,
                "speaker_label": utterance.speaker_label,
                "speaker_name": utterance.speaker_name,
                "message": "No recognizable speech detected."
            }
        )
        return {
            "success": True,
            "status": "NO_SPEECH",
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "message": "No recognizable speech detected."
        }

    elif status == "AUDIO_ERROR":
        error_msg = proc_result.get("error_message", "Audio could not be processed.")
        room_manager.mark_utterance_error(code, utterance.utterance_id, error_msg)
        await ws_manager.broadcast(
            code,
            "UTTERANCE_ERROR",
            {
                "utterance_id": utterance.utterance_id,
                "participant_id": participant_id,
                "speaker_label": utterance.speaker_label,
                "speaker_name": utterance.speaker_name,
                "error_type": "AUDIO_ERROR",
                "message": "Audio could not be processed. Please try again."
            }
        )
        return {
            "success": False,
            "status": "AUDIO_ERROR",
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "message": "Audio could not be processed. Please try again."
        }

    elif status == "TRANSCRIPTION_ERROR":
        error_msg = proc_result.get("error_message", "Speech transcription failed.")
        room_manager.mark_utterance_error(code, utterance.utterance_id, error_msg)
        await ws_manager.broadcast(
            code,
            "UTTERANCE_ERROR",
            {
                "utterance_id": utterance.utterance_id,
                "participant_id": participant_id,
                "speaker_label": utterance.speaker_label,
                "speaker_name": utterance.speaker_name,
                "error_type": "TRANSCRIPTION_ERROR",
                "message": "Speech transcription failed. Please try again."
            }
        )
        return {
            "success": False,
            "status": "TRANSCRIPTION_ERROR",
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "message": "Speech transcription failed. Please try again."
        }

    elif status == "NLP_ERROR":
        # Transcript was preserved!
        transcript = proc_result.get("transcript", "")
        error_msg = proc_result.get("error_message", "NLP analysis failed.")
        room_manager.complete_utterance(
            room_code=code,
            utterance_id=utterance.utterance_id,
            transcript=transcript,
            nlp_analysis=None
        )
        await ws_manager.broadcast(
            code,
            "UTTERANCE_ERROR",
            {
                "utterance_id": utterance.utterance_id,
                "participant_id": participant_id,
                "speaker_label": utterance.speaker_label,
                "speaker_name": utterance.speaker_name,
                "transcript": transcript,
                "error_type": "NLP_ERROR",
                "message": "Speech was transcribed, but analysis is temporarily unavailable."
            }
        )
        return {
            "success": True,
            "status": "NLP_ERROR",
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "transcript": transcript,
            "message": "Speech was transcribed, but analysis is temporarily unavailable."
        }

    else:
        # COMPLETED
        transcript = proc_result["transcript"]
        nlp_analysis = proc_result["nlp_analysis"]
        transcription_warning = proc_result.get("transcription_warning")

        updated_utterance = room_manager.complete_utterance(
            room_code=code,
            utterance_id=utterance.utterance_id,
            transcript=transcript,
            nlp_analysis=nlp_analysis,
            transcription_warning=transcription_warning
        )

        # ----------------------------------------------------
        # SERVER-SIDE COUNTERARGUMENT PRIVACY ENFORCEMENT
        # ----------------------------------------------------
        # Public analysis (without private counterargument)
        public_nlp = {
            k: v for k, v in (nlp_analysis or {}).items()
            if k != "counterargument"
        }

        public_payload = {
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "transcript": transcript,
            "transcription_warning": transcription_warning,
            "nlp_analysis": public_nlp,
            "status": "COMPLETED",
            "created_at": updated_utterance.created_at if updated_utterance else None
        }

        # Private payload (WITH counterargument coaching)
        private_payload = {
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "transcript": transcript,
            "transcription_warning": transcription_warning,
            "nlp_analysis": nlp_analysis,
            "status": "COMPLETED",
            "created_at": updated_utterance.created_at if updated_utterance else None
        }

        # Broadcast public analysis to other participants in the room
        await ws_manager.broadcast_to_others(
            code,
            exclude_participant_id=participant_id,
            event_type="UTTERANCE_COMPLETED",
            data=public_payload
        )

        # Send full private analysis exclusively to the author participant
        await ws_manager.send_to_participant(
            code,
            participant_id,
            "UTTERANCE_COMPLETED",
            private_payload
        )

        # Send full analysis to host
        await ws_manager.send_to_role(
            code,
            "host",
            "UTTERANCE_COMPLETED",
            private_payload
        )

        # Return full private payload in REST response to author
        return {
            "success": True,
            "status": "COMPLETED",
            "utterance_id": utterance.utterance_id,
            "participant_id": participant_id,
            "speaker_label": utterance.speaker_label,
            "speaker_name": utterance.speaker_name,
            "transcript": transcript,
            "nlp_analysis": nlp_analysis
        }
