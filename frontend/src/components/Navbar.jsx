import React from 'react';
import { Shield, Radio, Volume2, Sparkles } from 'lucide-react';

export default function Navbar({ roomState, onLeave, role }) {
  return (
    <header className="glass-panel" style={{ margin: '16px auto', maxWidth: '1200px', padding: '14px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Brand Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }} onClick={onLeave}>
          <div style={{
            background: 'linear-gradient(135deg, #6366f1, #38bdf8)',
            padding: '8px',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)'
          }}>
            <Shield size={22} color="#fff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="font-display" style={{ fontSize: '1.3rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(to right, #f8fafc, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                ReasonIQ
              </span>
              <span style={{ fontSize: '0.65rem', background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(56, 189, 248, 0.3)', fontWeight: 700 }}>
                PHASE 1
              </span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Real-Time Argument & Critical Thinking Analysis
            </div>
          </div>
        </div>

        {/* Room State & Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {roomState && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {roomState.status === 'LIVE' && (
                <div className="badge badge-live">
                  <span className="pulse-dot"></span>
                  DEBATE LIVE
                </div>
              )}
              {roomState.status === 'READY' && (
                <div className="badge badge-ready">
                  READY
                </div>
              )}
              {roomState.status === 'WAITING' && (
                <div className="badge badge-waiting">
                  WAITING ROOM
                </div>
              )}
              {roomState.status === 'ENDED' && (
                <div className="badge badge-ended">
                  DEBATE ENDED
                </div>
              )}

              {onLeave && (
                <button className="btn-secondary" style={{ padding: '6px 14px', fontSize: '0.85rem' }} onClick={onLeave}>
                  Leave
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
