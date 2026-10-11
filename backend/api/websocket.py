"""
BiasLess - WebSocket Connection Manager and Handlers

Manages real-time WebSocket connections for live debate rooms.
Handles broadcasts and participant-targeted events.
"""

import json
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.room_manager import room_manager

router = APIRouter()


class ConnectionManager:
    """
    Tracks and broadcasts to active WebSocket connections per room.
    """

    def __init__(self):
        # room_code -> list of (WebSocket, participant_id, role)
        self.active_connections: Dict[str, List[Dict]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        room_code: str,
        participant_id: Optional[str] = None,
        role: str = "participant"
    ):
        await websocket.accept()
        code = room_code.upper()
        if code not in self.active_connections:
            self.active_connections[code] = []

        self.active_connections[code].append({
            "websocket": websocket,
            "participant_id": participant_id,
            "role": role
        })

    def disconnect(
        self,
        websocket: WebSocket,
        room_code: str
    ):
        code = room_code.upper()
        if code in self.active_connections:
            self.active_connections[code] = [
                conn for conn in self.active_connections[code]
                if conn["websocket"] != websocket
            ]
            if not self.active_connections[code]:
                del self.active_connections[code]

    async def broadcast(self, room_code: str, event_type: str, data: Dict):
        """
        Broadcast a JSON event to all connected clients in a room.
        """
        code = room_code.upper()
        if code not in self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data
        }
        json_str = json.dumps(message)

        dead_connections = []
        for conn in self.active_connections[code]:
            ws = conn["websocket"]
            try:
                await ws.send_text(json_str)
            except Exception:
                dead_connections.append(ws)

        # Cleanup any broken connections
        for ws in dead_connections:
            self.disconnect(ws, code)

    async def send_to_participant(
        self,
        room_code: str,
        participant_id: str,
        event_type: str,
        data: Dict
    ):
        """
        Send a targeted event to a specific participant.
        """
        code = room_code.upper()
        if code not in self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data
        }
        json_str = json.dumps(message)

        dead_connections = []
        for conn in self.active_connections[code]:
            if conn["participant_id"] == participant_id:
                ws = conn["websocket"]
                try:
                    await ws.send_text(json_str)
                except Exception:
                    dead_connections.append(ws)

        for ws in dead_connections:
            self.disconnect(ws, code)


    async def send_to_role(
        self,
        room_code: str,
        role: str,
        event_type: str,
        data: Dict
    ):
        """
        Send an event to all connections matching a specific role (e.g. 'host').
        """
        code = room_code.upper()
        if code not in self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data
        }
        json_str = json.dumps(message)

        dead_connections = []
        for conn in self.active_connections[code]:
            if conn.get("role") == role:
                ws = conn["websocket"]
                try:
                    await ws.send_text(json_str)
                except Exception:
                    dead_connections.append(ws)

        for ws in dead_connections:
            self.disconnect(ws, code)

    async def broadcast_to_others(
        self,
        room_code: str,
        exclude_participant_id: str,
        event_type: str,
        data: Dict
    ):
        """
        Broadcast to all clients in the room except the specified participant and host.
        """
        code = room_code.upper()
        if code not in self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data
        }
        json_str = json.dumps(message)

        dead_connections = []
        for conn in self.active_connections[code]:
            if conn.get("participant_id") != exclude_participant_id and conn.get("role") != "host":
                ws = conn["websocket"]
                try:
                    await ws.send_text(json_str)
                except Exception:
                    dead_connections.append(ws)

        for ws in dead_connections:
            self.disconnect(ws, code)


ws_manager = ConnectionManager()


@router.websocket("/ws/rooms/{room_code}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_code: str,
    participant_id: Optional[str] = None,
    role: str = "participant"
):
    code = room_code.upper()
    room = room_manager.get_room(code)
    if not room:
        await websocket.close(code=4004, reason="Room not found.")
        return

    await ws_manager.connect(
        websocket=websocket,
        room_code=code,
        participant_id=participant_id,
        role=role
    )

    try:
        # Prepare privacy-aware utterances list for initial ROOM_SYNC
        synced_utterances = []
        for u in room.utterances:
            u_dict = u.model_dump()
            # If not host and not author, strip private counterargument
            if role != "host" and u.participant_id != participant_id:
                if u_dict.get("nlp_analysis") and isinstance(u_dict["nlp_analysis"], dict):
                    u_dict["nlp_analysis"] = {
                        k: v for k, v in u_dict["nlp_analysis"].items()
                        if k != "counterargument"
                    }
            synced_utterances.append(u_dict)

        # Send initial room sync upon connection
        await websocket.send_text(json.dumps({
            "type": "ROOM_SYNC",
            "data": {
                "room_code": room.room_code,
                "topic": room.topic,
                "host_name": room.host_name,
                "status": room.status,
                "allow_interruptions": room.allow_interruptions,
                "participants": [p.model_dump() for p in room.participants.values()],
                "utterances": synced_utterances
            }
        }))

        while True:
            # Keep connection open and handle any client ping/events
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
                event_type = data.get("type")

                if event_type == "PING":
                    await websocket.send_text(json.dumps({"type": "PONG"}))
                elif event_type == "MIC_STATUS":
                    # Broadcast participant mic status (e.g. recording indicator)
                    mic_on = bool(data.get("mic_active", False))
                    p_id = data.get("participant_id")
                    if p_id and p_id in room.participants:
                        # Server-authoritative strict turn check
                        if mic_on and not room.allow_interruptions:
                            active_speaker = next(
                                (other_p for other_p in room.participants.values() if other_p.participant_id != p_id and other_p.mic_active),
                                None
                            )
                            if active_speaker:
                                await ws_manager.send_to_participant(
                                    code,
                                    p_id,
                                    "SPEAKER_TURN_BLOCKED",
                                    {
                                        "code": "SPEAKER_TURN_BLOCKED",
                                        "message": f"{active_speaker.speaker_label or active_speaker.name} is currently speaking. Please wait for your turn."
                                    }
                                )
                                continue

                        room.participants[p_id].mic_active = mic_on
                        await ws_manager.broadcast(
                            code,
                            "PARTICIPANT_MIC_UPDATE",
                            {
                                "participant_id": p_id,
                                "speaker_label": room.participants[p_id].speaker_label,
                                "mic_active": mic_on
                            }
                        )
            except Exception:
                pass

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, code)
    except Exception:
        ws_manager.disconnect(websocket, code)
