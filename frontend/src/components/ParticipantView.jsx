import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, AlertCircle, Clock, Volume2, ShieldCheck, Sparkles, MessageSquare, Award, Activity } from 'lucide-react';
import { submitUtterance, getMyReport } from '../services/api';
import UtteranceCard from './UtteranceCard';
import ReportModal from './ReportModal';

export default function ParticipantView({
  room,
  participant,
  roomSocket,
  roomError,
  onClearRoomError,
  turnBlockedNotice,
  onClearTurnBlockedNotice
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [pendingProcessingCount, setPendingProcessingCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  
  // Final Report state
  const [showReportModal, setShowReportModal] = useState(false);
  const [myReport, setMyReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Sync turn blocked notice from websocket
  useEffect(() => {
    if (turnBlockedNotice) {
      setErrorMessage(turnBlockedNotice);
      if (isRecording) {
        stopRecording();
      }
    }
  }, [turnBlockedNotice]);

  // Clean up media stream on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const startRecording = async () => {
    setErrorMessage('');
    if (onClearTurnBlockedNotice) onClearTurnBlockedNotice();

    // Check if strict turns mode is active and another speaker is currently speaking
    if (room.allow_interruptions === false) {
      const activeSpeaker = (room.participants || []).find(
        p => p.participant_id !== participant.participant_id && p.mic_active
      );
      if (activeSpeaker) {
        setErrorMessage(`${activeSpeaker.speaker_label || activeSpeaker.name} is currently speaking. Please wait for your turn.`);
        return;
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      const audioTrack = stream.getAudioTracks()[0];
      if (!audioTrack || audioTrack.readyState !== 'live') {
        throw new Error('Microphone audio track is inactive or muted.');
      }
      
      // Determine best supported MIME type
      let mimeType = '';
      const candidateMimes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        'audio/wav'
      ];
      for (const candidate of candidateMimes) {
        if (MediaRecorder.isTypeSupported(candidate)) {
          mimeType = candidate;
          break;
        }
      }

      // Local session-scoped chunks to prevent concurrent turn overwrites
      const sessionChunks = [];

      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          sessionChunks.push(event.data);
        }
      };

      recorder.onstop = async () => {
        // Stop audio tracks to release microphone hardware
        stream.getTracks().forEach(track => track.stop());

        const finalMime = recorder.mimeType || mimeType || 'audio/webm';
        const audioBlob = new Blob(sessionChunks, { type: finalMime });

        if (audioBlob.size > 0) {
          handleAudioSubmission(audioBlob);
        } else {
          setErrorMessage('No audio data recorded. Please ensure your microphone is connected and speak clearly.');
        }
      };

      recorder.start(100); // 100ms slices ensure even short turns have multiple frames
      setIsRecording(true);
      setRecordingSeconds(0);

      // Notify WebSocket of active mic
      if (roomSocket) {
        roomSocket.sendMicStatus(true);
      }

      timerRef.current = setInterval(() => {
        setRecordingSeconds(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Microphone access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Microphone permission is required to speak. Please allow microphone access.');
      } else {
        setErrorMessage('Could not access microphone: ' + (err.message || 'Unknown error'));
      }
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (roomSocket) {
      roomSocket.sendMicStatus(false);
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingSeconds(0);
    }
  };

  const toggleMicrophone = () => {
    // Microphone & recording state is completely decoupled from ongoing processing jobs
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleAudioSubmission = async (audioBlob) => {
    setPendingProcessingCount(prev => prev + 1);
    setErrorMessage('');

    try {
      await submitUtterance(room.room_code, participant.participant_id, audioBlob);
    } catch (err) {
      setErrorMessage(err.message || 'Failed to submit audio.');
    } finally {
      setPendingProcessingCount(prev => Math.max(0, prev - 1));
    }
  };

  const handleViewReport = async () => {
    setLoadingReport(true);
    setErrorMessage('');
    try {
      const data = await getMyReport(room.room_code, participant.participant_id);
      setMyReport(data);
      setShowReportModal(true);
    } catch (err) {
      setErrorMessage(err.message || 'Failed to load performance report.');
    } finally {
      setLoadingReport(false);
    }
  };

  const formatTimer = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const utterances = room.utterances || [];

  // ==========================================================
  // VIEW 1: WAITING FOR HOST ADMISSION
  // ==========================================================
  if (participant.status === 'WAITING') {
    return (
      <div className="container" style={{ maxWidth: '560px', marginTop: '40px' }}>
        <div className="glass-panel-glow" style={{ padding: '36px', textAlign: 'center' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 20px auto'
          }}>
            <Clock size={32} color="var(--accent-amber)" />
          </div>

          <h2 className="font-display" style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: '8px' }}>
            JOIN REQUEST SENT
          </h2>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginBottom: '24px' }}>
            Room: <strong className="font-mono" style={{ color: 'var(--accent-cyan)' }}>{room.room_code}</strong>
          </div>

          <div style={{
            background: 'rgba(13, 18, 31, 0.6)',
            padding: '16px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            fontSize: '0.9rem'
          }}>
            Waiting for the host to approve your request. Please keep this screen open...
          </div>
        </div>
      </div>
    );
  }

  // ==========================================================
  // VIEW 2: REJECTED
  // ==========================================================
  if (participant.status === 'REJECTED') {
    return (
      <div className="container" style={{ maxWidth: '560px', marginTop: '40px' }}>
        <div className="glass-panel" style={{ padding: '36px', textAlign: 'center', borderColor: 'var(--accent-rose)' }}>
          <AlertCircle size={48} color="var(--accent-rose)" style={{ margin: '0 auto 16px auto' }} />
          <h2 className="font-display" style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-rose)', marginBottom: '8px' }}>
            Request Declined
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            The host has declined your request to join this debate room.
          </p>
        </div>
      </div>
    );
  }

  // ==========================================================
  // VIEW 3: ADMITTED, BUT WAITING FOR DEBATE TO START
  // ==========================================================
  if (participant.status === 'ADMITTED' && room.status !== 'LIVE' && room.status !== 'ENDED') {
    return (
      <div className="container" style={{ maxWidth: '560px', marginTop: '40px' }}>
        <div className="glass-panel-glow" style={{ padding: '36px', textAlign: 'center' }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'rgba(16, 185, 129, 0.15)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 20px auto'
          }}>
            <ShieldCheck size={32} color="var(--accent-emerald)" />
          </div>

          <h2 className="font-display" style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: '6px' }}>
            YOU HAVE BEEN ADMITTED
          </h2>
          
          <div style={{ margin: '20px 0' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Your Assigned Identity
            </span>
            <span className="badge badge-speaker" style={{ fontSize: '1.3rem', padding: '8px 24px' }}>
              {participant.speaker_label}
            </span>
          </div>

          <div style={{
            background: 'rgba(13, 18, 31, 0.6)',
            padding: '16px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            fontSize: '0.9rem',
            lineHeight: 1.5
          }}>
            Topic: <strong style={{ color: 'var(--text-primary)' }}>{room.topic}</strong>
            <div style={{ marginTop: '8px', color: 'var(--accent-amber)' }}>
              Waiting for the host to start the debate...
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ==========================================================
  // VIEW 4: LIVE DEBATE & POST-DEBATE REPORT VIEW
  // ==========================================================
  return (
    <div className="container" style={{ maxWidth: '800px' }}>
      
      {/* Room Error Notification Banner (Requirements 11, 14, 45) */}
      {roomError && (
        <div style={{
          marginBottom: '20px',
          background: 'rgba(99, 102, 241, 0.1)',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          borderRadius: 'var(--radius-sm)',
          padding: '14px 18px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          color: 'var(--accent-cyan)'
        }}>
          <div>
            <strong style={{ display: 'block', marginBottom: '2px', color: '#fff', fontSize: '0.95rem' }}>
              DEBATE NOTICE
            </strong>
            <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
              A temporary room error occurred. Your participation is still active. Please wait for the host to continue.
            </span>
          </div>
          {onClearRoomError && (
            <button
              onClick={onClearRoomError}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '0.8rem',
                textDecoration: 'underline'
              }}
            >
              Dismiss
            </button>
          )}
        </div>
      )}

      {/* Participant Header Card */}
      <div className="glass-panel-glow" style={{ padding: '24px', marginBottom: '24px', textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '8px' }}>
          <span className="badge badge-speaker" style={{ fontSize: '1rem', padding: '6px 18px' }}>
            {participant.speaker_label}
          </span>
          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {participant.name}
          </span>
        </div>

        <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '4px' }}>
          Debate Topic
        </div>
        <h2 className="font-display" style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px' }}>
          {room.topic}
        </h2>

        {/* Tactile Microphone Section */}
        {room.status === 'LIVE' && (
          <div className="mic-btn-container">
            <button
              onClick={toggleMicrophone}
              className={`mic-toggle-btn ${
                isRecording
                  ? 'mic-btn-on recording-pulse'
                  : 'mic-btn-off'
              }`}
              aria-label={isRecording ? 'Stop speaking turn' : 'Start speaking turn'}
            >
              {isRecording ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                  <Mic size={38} />
                  <span className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 800 }}>
                    {formatTimer(recordingSeconds)}
                  </span>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                    MIC ON
                  </span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  <MicOff size={36} />
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                    MIC OFF
                  </span>
                </div>
              )}
            </button>

            <div style={{ marginTop: '16px', textAlign: 'center' }}>
              {isRecording ? (
                <span style={{ color: '#fda4af', fontSize: '0.9rem', fontWeight: 600 }}>
                  Recording your point. Tap button again when finished speaking.
                </span>
              ) : (
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  Tap microphone to record your argument. You can speak multiple turns continuously.
                </span>
              )}
            </div>

            {/* Asynchronous Processing Indicator */}
            {pendingProcessingCount > 0 && (
              <div style={{
                marginTop: '12px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(99, 102, 241, 0.12)',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                padding: '6px 14px',
                borderRadius: 'var(--radius-full)',
                color: 'var(--accent-cyan)',
                fontSize: '0.82rem'
              }}>
                <Activity size={14} className="pulse-dot" />
                <span>Processing {pendingProcessingCount} speaking turn{pendingProcessingCount > 1 ? 's' : ''} with Whisper & ReasonIQ NLP...</span>
              </div>
            )}

            {errorMessage && (
              <div style={{
                marginTop: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: 'var(--accent-rose)',
                background: 'rgba(244, 63, 94, 0.1)',
                padding: '10px 16px',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.88rem'
              }}>
                <AlertCircle size={18} />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>
        )}

        {/* Post-Debate Report Trigger */}
        {room.status === 'ENDED' && (
          <div style={{
            marginTop: '20px',
            padding: '20px',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(56, 189, 248, 0.08))',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-glow)'
          }}>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-primary)', marginBottom: '6px' }}>
              Debate Concluded
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>
              The debate has officially concluded. Your authoritative final performance analysis and score breakdown are ready.
            </p>
            <button
              onClick={handleViewReport}
              disabled={loadingReport}
              className="btn-primary"
              style={{ padding: '12px 28px', fontSize: '1rem' }}
            >
              <Award size={18} />
              {loadingReport ? 'Loading Report...' : 'View My Performance Report'}
            </button>
          </div>
        )}
      </div>

      {/* Live Debate Transcript & Analysis Feed */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 700, fontSize: '1.1rem' }}>
            <MessageSquare size={20} color="var(--accent-indigo)" />
            Live Debate Feed
          </div>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {utterances.length} speaking turn{utterances.length === 1 ? '' : 's'}
          </span>
        </div>

        {utterances.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.9rem' }}>
            No speaking turns recorded yet. Turn your mic ON to speak.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {utterances.map(u => (
              <UtteranceCard key={u.utterance_id} utterance={u} />
            ))}
          </div>
        )}
      </div>

      {/* Report Modal */}
      {showReportModal && myReport && (
        <ReportModal
          report={myReport}
          onClose={() => setShowReportModal(false)}
        />
      )}

    </div>
  );
}
