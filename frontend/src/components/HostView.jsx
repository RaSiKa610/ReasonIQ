import React, { useState } from 'react';
import { Copy, Check, Users, Play, Square, UserCheck, UserX, MessageSquare, Sparkles, Sliders, Award, ShieldAlert } from 'lucide-react';
import { admitParticipant, rejectParticipant, startDebate, endDebate, updateSettings, getParticipantReport } from '../services/api';
import UtteranceCard from './UtteranceCard';
import ReportModal from './ReportModal';

export default function HostView({ room, hostToken, onRoomUpdated, roomError, onClearRoomError }) {
  const [copied, setCopied] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Report Modal state
  const [selectedReport, setSelectedReport] = useState(null);
  const [loadingReportId, setLoadingReportId] = useState(null);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(room.room_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handleToggleInterruptions = async () => {
    const currentSetting = room.allow_interruptions ?? true;
    setActionLoading(true);
    setErrorMsg('');
    try {
      const updated = await updateSettings(room.room_code, hostToken, !currentSetting);
      if (onRoomUpdated) {
        onRoomUpdated({ allow_interruptions: updated.allow_interruptions });
      }
    } catch (err) {
      setErrorMsg(
        'Unable to update debate settings. The current debate and participants are still safe. ' +
        'The previous interruption setting has been restored. Please try again.'
      );
    } finally {
      setActionLoading(false);
    }
  };

  const handleAdmit = async (participantId) => {
    setActionLoading(true);
    setErrorMsg('');
    try {
      await admitParticipant(room.room_code, participantId, hostToken);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to admit participant.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleReject = async (participantId) => {
    setActionLoading(true);
    setErrorMsg('');
    try {
      await rejectParticipant(room.room_code, participantId, hostToken);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to reject participant.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleStartDebate = async () => {
    setActionLoading(true);
    setErrorMsg('');
    try {
      await startDebate(room.room_code, hostToken);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to start debate.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleEndDebate = async () => {
    if (!window.confirm('Are you sure you want to end this debate and generate final performance reports?')) return;
    setActionLoading(true);
    setErrorMsg('');
    try {
      await endDebate(room.room_code, hostToken);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to end debate.');
    } finally {
      setActionLoading(false);
    }
  };

  const handleViewParticipantReport = async (participantId) => {
    setLoadingReportId(participantId);
    setErrorMsg('');
    try {
      const report = await getParticipantReport(room.room_code, participantId, hostToken);
      setSelectedReport(report);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to load speaker report.');
    } finally {
      setLoadingReportId(null);
    }
  };

  // Filter pending vs admitted participants
  const participants = room.participants || [];
  const pendingRequests = participants.filter(p => p.status === 'WAITING');
  const admittedParticipants = participants.filter(p => p.status === 'ADMITTED');
  const utterances = room.utterances || [];
  const allowInterruptions = room.allow_interruptions ?? true;

  return (
    <div className="container">
      {/* Top Banner: Topic & Room Code */}
      <div className="glass-panel-glow" style={{ padding: '28px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '20px' }}>
          <div>
            <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-cyan)', fontWeight: 700, marginBottom: '6px' }}>
              Debate Topic
            </div>
            <h2 className="font-display" style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
              {room.topic}
            </h2>
            <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
              Moderated by <strong style={{ color: 'var(--text-primary)' }}>{room.host_name}</strong>
            </div>
          </div>

          {/* Room Code Card */}
          <div style={{
            background: 'rgba(13, 18, 31, 0.9)',
            border: '1px solid var(--border-accent)',
            borderRadius: 'var(--radius-md)',
            padding: '16px 24px',
            textAlign: 'center',
            minWidth: '220px'
          }}>
            <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
              ROOM CODE
            </div>
            <div className="font-mono" style={{ fontSize: '1.8rem', fontWeight: 800, letterSpacing: '0.1em', color: 'var(--accent-cyan)', marginBottom: '8px' }}>
              {room.room_code}
            </div>
            <button
              onClick={handleCopyCode}
              className="btn-secondary"
              style={{ padding: '6px 12px', fontSize: '0.8rem', width: '100%' }}
            >
              {copied ? <Check size={14} color="var(--accent-emerald)" /> : <Copy size={14} />}
              {copied ? 'Copied Code!' : 'Copy Room Code'}
            </button>
          </div>
        </div>

        {/* Global Controls & Interruption Toggle */}
        <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
            {room.status === 'WAITING' && (
              <span style={{ color: 'var(--accent-amber)', fontSize: '0.9rem' }}>
                Waiting for participants to request entry...
              </span>
            )}
            {room.status === 'READY' && (
              <span style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem' }}>
                Participants admitted. Ready to start!
              </span>
            )}
            {room.status === 'LIVE' && (
              <span style={{ color: '#fb7185', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="pulse-dot"></span> Debate is currently LIVE.
              </span>
            )}
            {room.status === 'ENDED' && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Debate ended. Final reports generated below.
              </span>
            )}

            {/* Interruption setting toggle */}
            <button
              onClick={handleToggleInterruptions}
              disabled={actionLoading}
              className="btn-secondary"
              style={{
                padding: '6px 12px',
                fontSize: '0.8rem',
                border: allowInterruptions ? '1px solid var(--border-subtle)' : '1px solid var(--accent-indigo)',
                background: allowInterruptions ? 'transparent' : 'rgba(99, 102, 241, 0.15)'
              }}
              title="Toggle whether participants can speak concurrently"
            >
              <Sliders size={14} />
              Cross-Talk: <strong>{allowInterruptions ? 'Allowed' : 'Strict Turns'}</strong>
            </button>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            {room.status !== 'LIVE' && room.status !== 'ENDED' && (
              <button
                onClick={handleStartDebate}
                className="btn-success"
                disabled={actionLoading || admittedParticipants.length === 0}
              >
                <Play size={16} />
                START DEBATE
              </button>
            )}

            {room.status === 'LIVE' && (
              <button
                onClick={handleEndDebate}
                className="btn-danger"
                disabled={actionLoading}
              >
                <Square size={16} />
                END DEBATE & GENERATE REPORTS
              </button>
            )}
          </div>
        </div>

        {errorMsg && (
          <div style={{ marginTop: '12px', color: 'var(--accent-rose)', fontSize: '0.88rem' }}>
            {errorMsg}
          </div>
        )}

        {roomError && (
          <div style={{
            marginTop: '14px',
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: 'var(--radius-sm)',
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            color: 'var(--accent-rose)',
            fontSize: '0.88rem'
          }}>
            <div>
              <strong>Settings Update Failed: </strong>
              <span>{roomError.message || 'The interruption setting could not be changed. The debate is still active and participants remain connected.'}</span>
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
      </div>

      {/* Main Grid: Left Roster & Join Requests, Right Live Stream */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px' }}>
        
        {/* Left Column: Participants & Requests */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Admitted Participants / Post-Debate Report Triggers */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, fontSize: '1.05rem' }}>
                <UserCheck size={18} color="var(--accent-cyan)" />
                {room.status === 'ENDED' ? 'Debate Performance Reports' : 'Admitted Speakers'}
              </div>
              <span style={{ fontSize: '0.8rem', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '12px' }}>
                {admittedParticipants.length}
              </span>
            </div>

            {admittedParticipants.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem', fontStyle: 'italic', padding: '12px 0' }}>
                No participants admitted yet.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {admittedParticipants.map(p => (
                  <div key={p.participant_id} style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 14px',
                    background: 'rgba(13, 18, 31, 0.6)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)'
                  }}>
                    <div>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {p.name}
                      </div>
                      <span className="badge badge-speaker" style={{ fontSize: '0.72rem', marginTop: '4px' }}>
                        {p.speaker_label}
                      </span>
                    </div>

                    {room.status === 'ENDED' ? (
                      <button
                        onClick={() => handleViewParticipantReport(p.participant_id)}
                        disabled={loadingReportId === p.participant_id}
                        className="btn-primary"
                        style={{ padding: '6px 14px', fontSize: '0.8rem' }}
                      >
                        <Award size={14} />
                        {loadingReportId === p.participant_id ? 'Loading...' : 'View Report'}
                      </button>
                    ) : (
                      <span style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                        Connected
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Pending Join Requests (Only if not ended) */}
          {room.status !== 'ENDED' && (
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, fontSize: '1.05rem' }}>
                  <Users size={18} color="var(--accent-amber)" />
                  Join Requests
                </div>
                <span style={{ fontSize: '0.8rem', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '12px' }}>
                  {pendingRequests.length}
                </span>
              </div>

              {pendingRequests.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem', fontStyle: 'italic', padding: '12px 0' }}>
                  No pending requests.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {pendingRequests.map(p => (
                    <div key={p.participant_id} style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 14px',
                      background: 'rgba(13, 18, 31, 0.6)',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)'
                    }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {p.name}
                      </span>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <button
                          onClick={() => handleAdmit(p.participant_id)}
                          className="btn-success"
                          style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                          disabled={actionLoading}
                        >
                          <UserCheck size={14} /> Accept
                        </button>
                        <button
                          onClick={() => handleReject(p.participant_id)}
                          className="btn-danger"
                          style={{ padding: '6px 10px', fontSize: '0.8rem' }}
                          disabled={actionLoading}
                        >
                          <UserX size={14} /> Reject
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Live Debate Utterances Feed */}
        <div style={{ gridColumn: 'span 2' }}>
          <div className="glass-panel" style={{ padding: '24px', minHeight: '450px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 700, fontSize: '1.15rem' }}>
                <MessageSquare size={20} color="var(--accent-indigo)" />
                Live Debate Feed & ReasonIQ Analysis
              </div>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {utterances.length} speaking turn{utterances.length === 1 ? '' : 's'}
              </span>
            </div>

            {utterances.length === 0 ? (
              <div style={{
                textAlign: 'center',
                padding: '60px 20px',
                color: 'var(--text-muted)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '12px'
              }}>
                <Sparkles size={32} style={{ opacity: 0.3 }} />
                <p style={{ fontSize: '0.95rem' }}>
                  {room.status === 'LIVE'
                    ? 'Debate is live. When participants speak, their speech turns and real-time ReasonIQ analysis will appear here.'
                    : 'Start the debate to allow participants to speak and see real-time argument analysis.'}
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {utterances.map(u => (
                  <UtteranceCard key={u.utterance_id} utterance={u} />
                ))}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Report Modal */}
      {selectedReport && (
        <ReportModal
          report={selectedReport}
          onClose={() => setSelectedReport(null)}
        />
      )}
    </div>
  );
}
