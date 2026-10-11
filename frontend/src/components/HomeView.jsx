import React, { useState } from 'react';
import { Shield, PlusCircle, LogIn, Mic, Sparkles, Radio, AlertCircle } from 'lucide-react';
import { createRoom, joinRoom } from '../services/api';

export default function HomeView({ onRoomCreated, onJoinRequested }) {
  const [mode, setMode] = useState('host'); // 'host' or 'join'
  
  // Host state
  const [hostName, setHostName] = useState('');
  const [topic, setTopic] = useState('');
  const [hostLoading, setHostLoading] = useState(false);
  const [hostError, setHostError] = useState('');

  // Participant state
  const [participantName, setParticipantName] = useState('');
  const [roomCode, setRoomCode] = useState('');
  const [joinLoading, setJoinLoading] = useState(false);
  const [joinError, setJoinError] = useState('');

  const handleCreateRoom = async (e) => {
    e.preventDefault();
    if (!hostName.trim() || !topic.trim()) {
      setHostError('Please provide your name and debate topic.');
      return;
    }
    setHostLoading(true);
    setHostError('');

    try {
      const roomData = await createRoom(hostName.trim(), topic.trim());
      onRoomCreated(roomData);
    } catch (err) {
      setHostError(err.message || 'Failed to create room.');
    } finally {
      setHostLoading(false);
    }
  };

  const handleJoinRoom = async (e) => {
    e.preventDefault();
    if (!participantName.trim() || !roomCode.trim()) {
      setJoinError('Please provide your name and room code.');
      return;
    }
    setJoinLoading(true);
    setJoinError('');

    try {
      const joinData = await joinRoom(roomCode.trim(), participantName.trim());
      onJoinRequested(joinData, roomCode.trim().toUpperCase());
    } catch (err) {
      setJoinError(err.message || 'Failed to join room.');
    } finally {
      setJoinLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '640px', margin: '40px auto', padding: '0 20px' }}>
      {/* Title Hero */}
      <div style={{ textAlign: 'center', marginBottom: '36px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          background: 'rgba(99, 102, 241, 0.12)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          padding: '6px 16px',
          borderRadius: 'var(--radius-full)',
          marginBottom: '16px'
        }}>
          <Sparkles size={16} color="var(--accent-cyan)" />
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
            ReasonIQ Platform
          </span>
        </div>
        <h1 className="font-display" style={{ fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2, marginBottom: '12px' }}>
          Real-Time Argument & Critical Thinking Analysis
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: 1.5 }}>
          Live multi-participant debates with real-time speech transcription, named entity recognition, fallacy detection, bias signaling, and tactical counterargument coaching.
        </p>
      </div>

      {/* Mode Selector Tabs */}
      <div className="glass-panel" style={{ padding: '6px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginBottom: '24px' }}>
        <button
          onClick={() => { setMode('host'); setHostError(''); }}
          style={{
            padding: '12px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: mode === 'host' ? 'linear-gradient(135deg, var(--accent-indigo), var(--accent-purple))' : 'transparent',
            color: mode === 'host' ? '#fff' : 'var(--text-secondary)',
            fontWeight: 600,
            fontSize: '0.95rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'all 0.2s ease'
          }}
        >
          <PlusCircle size={18} />
          Host a Debate
        </button>
        <button
          onClick={() => { setMode('join'); setJoinError(''); }}
          style={{
            padding: '12px',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: mode === 'join' ? 'linear-gradient(135deg, var(--accent-indigo), var(--accent-purple))' : 'transparent',
            color: mode === 'join' ? '#fff' : 'var(--text-secondary)',
            fontWeight: 600,
            fontSize: '0.95rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            transition: 'all 0.2s ease'
          }}
        >
          <LogIn size={18} />
          Join as Participant
        </button>
      </div>

      {/* Form Container */}
      <div className="glass-panel-glow" style={{ padding: '32px' }}>
        {mode === 'host' ? (
          <form onSubmit={handleCreateRoom} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Host Name
              </label>
              <input
                type="text"
                className="input-field"
                placeholder="Enter your name"
                value={hostName}
                onChange={(e) => setHostName(e.target.value)}
                disabled={hostLoading}
                autoFocus
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Debate Topic
              </label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g. Should AI be used in higher education?"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                disabled={hostLoading}
              />
            </div>

            {hostError && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-rose)', fontSize: '0.88rem' }}>
                <AlertCircle size={16} />
                <span>{hostError}</span>
              </div>
            )}

            <button type="submit" className="btn-primary" disabled={hostLoading} style={{ width: '100%', marginTop: '8px' }}>
              {hostLoading ? 'Creating Live Room...' : 'CREATE DEBATE'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleJoinRoom} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Your Name
              </label>
              <input
                type="text"
                className="input-field"
                placeholder="Enter your name"
                value={participantName}
                onChange={(e) => setParticipantName(e.target.value)}
                disabled={joinLoading}
                autoFocus
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Room Code
              </label>
              <input
                type="text"
                className="input-field font-mono"
                placeholder="Enter room code"
                style={{ textTransform: 'uppercase', letterSpacing: '0.1em' }}
                value={roomCode}
                onChange={(e) => setRoomCode(e.target.value)}
                disabled={joinLoading}
              />
            </div>

            {joinError && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-rose)', fontSize: '0.88rem' }}>
                <AlertCircle size={16} />
                <span>{joinError}</span>
              </div>
            )}

            <button type="submit" className="btn-primary" disabled={joinLoading} style={{ width: '100%', marginTop: '8px' }}>
              {joinLoading ? 'Sending Request...' : 'REQUEST TO JOIN'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
