import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import HomeView from './components/HomeView';
import HostView from './components/HostView';
import ParticipantView from './components/ParticipantView';
import { RoomSocket } from './services/websocket';
import { getRoom } from './services/api';

export default function App() {
  const [view, setView] = useState('home'); // 'home' | 'host' | 'participant'
  const [room, setRoom] = useState(null);
  const [hostToken, setHostToken] = useState(null);
  const [participant, setParticipant] = useState(null);
  const [roomError, setRoomError] = useState(null);
  const [turnBlockedNotice, setTurnBlockedNotice] = useState(null);

  const socketRef = useRef(null);

  // Clean up socket on unmount or view leave
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const initWebSocket = (roomCode, participantId = null, role = 'participant') => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }

    const socket = new RoomSocket(
      roomCode,
      participantId,
      role,
      (type, data) => handleSocketMessage(type, data),
      (status) => console.log('Socket status:', status)
    );

    socket.connect();
    socketRef.current = socket;
  };

  const handleSocketMessage = (type, data) => {
    console.log('WS Event:', type, data);

    switch (type) {
      case 'ROOM_SYNC':
        setRoom(prev => ({
          ...prev,
          room_code: data.room_code,
          topic: data.topic,
          host_name: data.host_name,
          status: data.status,
          allow_interruptions: data.allow_interruptions,
          participants: data.participants,
          utterances: data.utterances
        }));
        setParticipant(prev => {
          if (!prev) return prev;
          const match = data.participants?.find(p => p.participant_id === prev.participant_id);
          if (match) {
            return {
              ...prev,
              status: match.status,
              speaker_label: match.speaker_label
            };
          }
          return prev;
        });
        break;

      case 'NEW_JOIN_REQUEST':
        setRoom(prev => {
          if (!prev) return prev;
          const exists = prev.participants?.some(p => p.participant_id === data.participant_id);
          const updated = exists
            ? prev.participants.map(p => p.participant_id === data.participant_id ? { ...p, ...data } : p)
            : [...(prev.participants || []), data];
          return { ...prev, participants: updated };
        });
        break;

      case 'PARTICIPANT_ADMITTED':
        setRoom(prev => {
          if (!prev) return prev;
          const updated = (prev.participants || []).map(p =>
            p.participant_id === data.participant_id
              ? { ...p, status: 'ADMITTED', speaker_label: data.speaker_label }
              : p
          );
          return {
            ...prev,
            status: data.room_status || prev.status,
            participants: updated
          };
        });
        setParticipant(prev => {
          if (!prev || prev.participant_id !== data.participant_id) return prev;
          return {
            ...prev,
            status: 'ADMITTED',
            speaker_label: data.speaker_label
          };
        });
        break;

      case 'PARTICIPANT_REJECTED':
        setRoom(prev => {
          if (!prev) return prev;
          const updated = (prev.participants || []).map(p =>
            p.participant_id === data.participant_id
              ? { ...p, status: 'REJECTED' }
              : p
          );
          return { ...prev, participants: updated };
        });
        setParticipant(prev => {
          if (!prev || prev.participant_id !== data.participant_id) return prev;
          return {
            ...prev,
            status: 'REJECTED'
          };
        });
        break;

      case 'YOU_ARE_ADMITTED':
        setParticipant(prev => {
          if (!prev || prev.participant_id !== data.participant_id) return prev;
          return {
            ...prev,
            status: 'ADMITTED',
            speaker_label: data.speaker_label
          };
        });
        break;

      case 'YOU_ARE_REJECTED':
        setParticipant(prev => {
          if (!prev || prev.participant_id !== data.participant_id) return prev;
          return {
            ...prev,
            status: 'REJECTED'
          };
        });
        break;

      case 'ROOM_SETTINGS_UPDATED':
      case 'SETTINGS_UPDATED':
        setRoom(prev => prev ? { ...prev, allow_interruptions: data.allow_interruptions } : prev);
        break;

      case 'PARTICIPANT_MIC_UPDATE':
        setRoom(prev => {
          if (!prev) return prev;
          const updated = (prev.participants || []).map(p =>
            p.participant_id === data.participant_id
              ? { ...p, mic_active: data.mic_active }
              : p
          );
          return { ...prev, participants: updated };
        });
        break;

      case 'ROOM_ERROR':
        setRoomError(data);
        break;

      case 'SPEAKER_TURN_BLOCKED':
        setTurnBlockedNotice(data.message || 'Another speaker is currently speaking. Please wait for your turn.');
        break;

      case 'FINAL_REPORT_AVAILABLE':
        setRoom(prev => prev ? { ...prev, status: 'ENDED', has_reports: true } : prev);
        break;

      case 'DEBATE_STARTED':
        setRoom(prev => prev ? { ...prev, status: 'LIVE' } : prev);
        break;

      case 'DEBATE_ENDED':
        setRoom(prev => prev ? { ...prev, status: 'ENDED' } : prev);
        break;

      case 'UTTERANCE_STARTED':
        setRoom(prev => {
          if (!prev) return prev;
          const existing = prev.utterances || [];
          const exists = existing.some(u => u.utterance_id === data.utterance_id);
          if (exists) return prev;
          return {
            ...prev,
            utterances: [
              ...existing,
              {
                utterance_id: data.utterance_id,
                speaker_label: data.speaker_label,
                speaker_name: data.speaker_name,
                status: 'PROCESSING',
                transcript: '',
                nlp_analysis: null,
                created_at: Date.now() / 1000
              }
            ]
          };
        });
        break;

      case 'UTTERANCE_COMPLETED':
        setRoom(prev => {
          if (!prev) return prev;
          const existing = prev.utterances || [];
          const exists = existing.some(u => u.utterance_id === data.utterance_id);
          const updated = exists
            ? existing.map(u => u.utterance_id === data.utterance_id ? { ...u, ...data, status: 'COMPLETED' } : u)
            : [...existing, { ...data, status: 'COMPLETED' }];
          return { ...prev, utterances: updated };
        });
        break;

      case 'UTTERANCE_NO_SPEECH':
        setRoom(prev => {
          if (!prev) return prev;
          const updated = (prev.utterances || []).map(u =>
            u.utterance_id === data.utterance_id
              ? { ...u, status: 'NO_SPEECH' }
              : u
          );
          return { ...prev, utterances: updated };
        });
        break;

      case 'UTTERANCE_ERROR':
        setRoom(prev => {
          if (!prev) return prev;
          const updated = (prev.utterances || []).map(u =>
            u.utterance_id === data.utterance_id
              ? {
                  ...u,
                  status: data.error_type || 'ERROR',
                  error_type: data.error_type,
                  error_message: data.message,
                  transcript: data.transcript !== undefined ? data.transcript : u.transcript
                }
              : u
          );
          return { ...prev, utterances: updated };
        });
        break;

      default:
        break;
    }
  };

  const handleRoomCreated = (roomData) => {
    setRoom({
      room_code: roomData.room_code,
      room_id: roomData.room_id,
      topic: roomData.topic,
      host_name: roomData.host_name,
      status: roomData.status,
      participants: [],
      utterances: []
    });
    setHostToken(roomData.host_token);
    setView('host');
    initWebSocket(roomData.room_code, null, 'host');
  };

  const handleJoinRequested = (joinData, roomCode) => {
    setParticipant({
      participant_id: joinData.participant_id,
      room_code: joinData.room_code,
      name: joinData.name,
      status: joinData.status,
      speaker_label: null
    });
    setRoom({
      room_code: roomCode,
      topic: 'Loading debate...',
      status: 'WAITING',
      participants: [],
      utterances: []
    });
    setView('participant');
    initWebSocket(roomCode, joinData.participant_id, 'participant');

    // Fetch initial room info
    getRoom(roomCode).then(data => {
      setRoom(data);
    }).catch(err => {
      console.warn('Initial room fetch error:', err);
    });
  };

  const handleLeave = () => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    setRoom(null);
    setHostToken(null);
    setParticipant(null);
    setView('home');
  };

  return (
    <div style={{ minHeight: '100vh', paddingBottom: '60px' }}>
      <Navbar roomState={room} onLeave={view !== 'home' ? handleLeave : null} role={view} />

      <main>
        {view === 'home' && (
          <HomeView
            onRoomCreated={handleRoomCreated}
            onJoinRequested={handleJoinRequested}
          />
        )}

        {view === 'host' && room && (
          <HostView
            room={room}
            hostToken={hostToken}
            roomError={roomError}
            onClearRoomError={() => setRoomError(null)}
            onRoomUpdated={(settingsUpdate) => {
              setRoom(prev => prev ? { ...prev, ...settingsUpdate } : prev);
            }}
          />
        )}

        {view === 'participant' && room && participant && (
          <ParticipantView
            room={room}
            participant={participant}
            roomSocket={socketRef.current}
            roomError={roomError}
            onClearRoomError={() => setRoomError(null)}
            turnBlockedNotice={turnBlockedNotice}
            onClearTurnBlockedNotice={() => setTurnBlockedNotice(null)}
          />
        )}
      </main>
    </div>
  );
}
