import React, { useState } from 'react';
import { X, Award, ShieldAlert, CheckCircle2, AlertTriangle, Zap, Lock, Info, MessageSquare, Clock, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';

export default function ReportModal({ report, onClose }) {
  const [expandedTurns, setExpandedTurns] = useState({});

  if (!report) return null;

  // Defensively unwrap envelope if passed as { success: true, report: { ... } }
  const data = report.report || report;

  const speakerName = data.speaker_name || data.speakerName || 'Speaker';
  const speakerLabel = data.speaker_label || data.speakerLabel || 'Speaker';
  const topic = data.topic || 'General Debate';
  const speakingHistory = data.speaking_history || data.turn_history || [];
  const totalTurns = data.total_turns ?? data.totalTurns ?? speakingHistory.length;
  const analyzedTurns = data.analyzed_turns ?? data.analyzedTurns ?? totalTurns;

  const hasScore = data.final_score !== null && data.final_score !== undefined && !isNaN(Number(data.final_score)) && totalTurns > 0;
  const finalScore = hasScore ? Number(data.final_score) : null;
  const baseScore = (hasScore && data.base_score !== null && data.base_score !== undefined && !isNaN(Number(data.base_score)))
    ? Number(data.base_score)
    : finalScore;
  const penalties = Number(data.intervention_penalties ?? data.intervention_penalty ?? 0);

  const scoreColor = finalScore !== null
    ? (finalScore >= 80 ? 'var(--accent-emerald)'
      : finalScore >= 60 ? 'var(--accent-cyan)'
      : finalScore >= 45 ? 'var(--accent-amber)'
      : 'var(--accent-rose)')
    : 'var(--text-muted)';

  const performanceTier = data.performance_tier || data.performance_level || (
    finalScore !== null
      ? (finalScore >= 75 ? 'Excellent' : finalScore >= 60 ? 'Proficient' : finalScore >= 45 ? 'Moderate' : 'Needs Development')
      : 'No Analyzed Turns'
  );

  const breakdown = data.breakdown || data.score_breakdown || null;
  const deductions = data.deductions || [];
  const strengths = data.strengths || [];
  const whatToAvoid = data.what_to_avoid || data.whatToAvoid || [];
  const recommendations = data.recommendations || data.how_to_improve || data.areas_to_improve || [];

  const formatTurnTime = (ts) => {
    if (!ts) return '';
    const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
    if (isNaN(date.getTime())) return '';
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const toggleTurnExpand = (turnNum) => {
    setExpandedTurns(prev => ({
      ...prev,
      [turnNum]: !prev[turnNum]
    }));
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(7, 9, 14, 0.85)',
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div className="glass-panel-glow" style={{
        maxWidth: '850px',
        width: '100%',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '32px',
        position: 'relative',
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-accent)'
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(255, 255, 255, 0.08)',
            border: 'none',
            color: 'var(--text-secondary)',
            borderRadius: '50%',
            width: '36px',
            height: '36px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.color = '#fff'}
          onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
          aria-label="Close Report"
        >
          <X size={20} />
        </button>

        {/* Header */}
        <div style={{ marginBottom: '24px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <span className="badge badge-speaker" style={{ fontSize: '0.9rem', padding: '4px 14px' }}>
              {speakerLabel}
            </span>
            <h2 className="font-display" style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {speakerName} — Performance Report
            </h2>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Debate: <strong style={{ color: 'var(--text-primary)' }}>{topic}</strong> &bull; Total Turns: <strong style={{ color: 'var(--accent-cyan)' }}>{totalTurns}</strong>
          </div>
        </div>

        {/* Zero turns view */}
        {(!hasScore || totalTurns === 0) ? (
          <div style={{
            background: 'rgba(13, 18, 31, 0.7)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '36px 24px',
            marginBottom: '24px',
            textAlign: 'center'
          }}>
            <Info size={36} color="var(--accent-cyan)" style={{ margin: '0 auto 12px auto' }} />
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
              No completed speaking turns were recorded.
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '500px', margin: '0 auto' }}>
              No completed speaking turns were available for analysis. When this speaker participates in a live debate turn, their argument quality, evidence, and conduct breakdown will be generated.
            </p>
          </div>
        ) : (
          <>
            {/* Final Score Hero Card */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(56, 189, 248, 0.08))',
              border: '1px solid var(--border-glow)',
              borderRadius: 'var(--radius-md)',
              padding: '24px',
              marginBottom: '24px',
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '20px'
            }}>
              <div>
                <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
                  FINAL AGGREGATED DEBATE SCORE
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
                  <span className="font-mono" style={{ fontSize: '3rem', fontWeight: 800, color: scoreColor, lineHeight: 1 }}>
                    {finalScore}
                  </span>
                  <span style={{ fontSize: '1.2rem', color: 'var(--text-muted)' }}>/ 100</span>
                  <span className="badge" style={{
                    background: scoreColor === 'var(--accent-emerald)' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(99, 102, 241, 0.15)',
                    color: scoreColor,
                    border: `1px solid ${scoreColor}`,
                    fontSize: '0.85rem'
                  }}>
                    {performanceTier}
                  </span>
                </div>
              </div>

              {/* Base vs Penalty Details */}
              <div style={{
                background: 'rgba(13, 18, 31, 0.7)',
                padding: '12px 18px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.88rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '24px', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Base Quality Score:</span>
                  <strong style={{ color: 'var(--accent-cyan)' }}>{baseScore !== null ? `${baseScore} / 100` : 'N/A'}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '24px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Intervention Penalties:</span>
                  <strong style={{ color: penalties > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                    {penalties > 0 ? `-${penalties} pts` : '0 pts'}
                  </strong>
                </div>
              </div>
            </div>

            {/* 4 Dimensional Breakdown */}
            {breakdown && (
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Award size={18} color="var(--accent-cyan)" />
                  Performance Dimensions
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                  {/* Dimension 1: Argument Quality */}
                  <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '14px' }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Argument Quality</div>
                    <div className="font-mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                      {breakdown.argument_quality ?? 'N/A'}<span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                  </div>

                  {/* Dimension 2: Evidence Quality */}
                  <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '14px' }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Evidence Usage</div>
                    <div className="font-mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>
                      {breakdown.evidence ?? breakdown.evidence_quality ?? 'N/A'}<span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                  </div>

                  {/* Dimension 3: Clarity & Structure */}
                  <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '14px' }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Clarity & Structure</div>
                    <div className="font-mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--accent-indigo)' }}>
                      {breakdown.clarity ?? breakdown.clarity_and_structure ?? 'N/A'}<span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                  </div>

                  {/* Dimension 4: Debate Conduct */}
                  <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '14px' }}>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>Debate Conduct</div>
                    <div className="font-mono" style={{ fontSize: '1.4rem', fontWeight: 700, color: (breakdown.conduct ?? breakdown.debate_conduct ?? 100) < 80 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                      {breakdown.conduct ?? breakdown.debate_conduct ?? 100}<span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>/100</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Speaking History (Requirement 4) */}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <MessageSquare size={18} color="var(--accent-indigo)" />
                Speaking History
              </h3>

              {speakingHistory.length === 0 ? (
                <div style={{
                  background: 'rgba(13, 18, 31, 0.5)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 18px',
                  color: 'var(--text-muted)',
                  fontSize: '0.88rem',
                  fontStyle: 'italic'
                }}>
                  No completed speaking turns were recorded.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {speakingHistory.map((turn, idx) => {
                    const turnNum = turn.turn_number || idx + 1;
                    const isLong = (turn.transcript || '').length > 180;
                    const isExpanded = expandedTurns[turnNum];
                    const turnScore = turn.final_score ?? turn.quality_score ?? turn.base_score ?? 50;
                    const turnIntervention = turn.intervention;
                    const timeFormatted = formatTurnTime(turn.created_at || turn.timestamp);

                    return (
                      <div key={turn.utterance_id || idx} style={{
                        background: 'rgba(13, 18, 31, 0.65)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '16px 18px',
                        transition: 'border-color 0.2s'
                      }}>
                        {/* Turn Header */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span className="badge" style={{
                              background: 'rgba(99, 102, 241, 0.15)',
                              color: 'var(--accent-cyan)',
                              border: '1px solid rgba(99, 102, 241, 0.3)',
                              fontWeight: 700,
                              fontSize: '0.8rem'
                            }}>
                              Turn {turnNum}
                            </span>
                            {timeFormatted && (
                              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Clock size={12} /> {timeFormatted}
                              </span>
                            )}
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            {turnIntervention && (
                              <span className="badge" style={{
                                background: 'rgba(244, 63, 94, 0.15)',
                                color: 'var(--accent-rose)',
                                border: '1px solid rgba(244, 63, 94, 0.3)',
                                fontSize: '0.75rem'
                              }}>
                                Warning: {turnIntervention.type?.replace(/_/g, ' ') || 'Conduct'} (-{turnIntervention.penalty} pts)
                              </span>
                            )}
                            <span className="font-mono" style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                              Quality Score: <strong style={{ color: '#fff' }}>{turnScore}/100</strong>
                            </span>
                          </div>
                        </div>

                        {/* Transcript */}
                        <div style={{
                          fontSize: '0.9rem',
                          lineHeight: 1.5,
                          color: 'var(--text-primary)',
                          background: 'rgba(7, 9, 14, 0.4)',
                          padding: '10px 14px',
                          borderRadius: '4px',
                          borderLeft: '3px solid var(--accent-indigo)'
                        }}>
                          "{isLong && !isExpanded ? `${turn.transcript.slice(0, 180)}...` : turn.transcript}"
                        </div>

                        {/* Expand / Collapse toggle for long transcripts */}
                        {isLong && (
                          <button
                            onClick={() => toggleTurnExpand(turnNum)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: 'var(--accent-cyan)',
                              fontSize: '0.78rem',
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              marginTop: '6px',
                              padding: 0
                            }}
                          >
                            {isExpanded ? (
                              <><ChevronUp size={14} /> Show Less</>
                            ) : (
                              <><ChevronDown size={14} /> Read Full Transcript</>
                            )}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Deductions Audit Trail */}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldAlert size={18} color="var(--accent-rose)" />
                Deduction Audit Trail
              </h3>

              {deductions.length === 0 ? (
                <div style={{
                  background: 'rgba(16, 185, 129, 0.08)',
                  border: '1px solid rgba(16, 185, 129, 0.25)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 18px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  color: 'var(--accent-emerald)',
                  fontSize: '0.9rem'
                }}>
                  <CheckCircle2 size={20} />
                  <span><strong>Clean Record:</strong> Zero debate conduct violations or major speech clarity penalties were applied.</span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {deductions.map((ded, idx) => (
                    <div key={idx} style={{
                      background: 'rgba(244, 63, 94, 0.06)',
                      border: '1px solid rgba(244, 63, 94, 0.2)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '12px 16px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      gap: '14px'
                    }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                          <span className="badge" style={{ background: 'rgba(244, 63, 94, 0.2)', color: 'var(--accent-rose)', border: '1px solid rgba(244, 63, 94, 0.4)', fontSize: '0.75rem' }}>
                            {ded.category || ded.type || 'Speech Quality Warning'}
                          </span>
                          {ded.utterance_id && (
                            <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              ID: {ded.utterance_id.slice(0, 8)}...
                            </span>
                          )}
                        </div>
                        <div style={{ fontSize: '0.88rem', color: 'var(--text-primary)', marginBottom: '2px' }}>
                          {ded.reason || ded.warning}
                        </div>
                        {(ded.statement || ded.text) && (
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                            "{ded.statement || ded.text}"
                          </div>
                        )}
                      </div>
                      <div className="font-mono" style={{ color: 'var(--accent-rose)', fontWeight: 700, fontSize: '1rem', whiteSpace: 'nowrap' }}>
                        -{ded.points} pts
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Strengths, Areas to Improve, What to Avoid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              {/* Strengths */}
              <div style={{ background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-emerald)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '10px' }}>
                  <CheckCircle2 size={16} />
                  Observed Strengths
                </div>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {(strengths.length > 0 ? strengths : ['Constructive participation throughout the debate']).map((s, i) => (
                    <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
                      <span style={{ color: 'var(--accent-emerald)' }}>&bull;</span>
                      <span>{s}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* What to Avoid */}
              <div style={{ background: 'rgba(245, 158, 11, 0.04)', border: '1px solid rgba(245, 158, 11, 0.15)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-amber)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '10px' }}>
                  <AlertTriangle size={16} />
                  What to Avoid
                </div>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {(whatToAvoid.length > 0 ? whatToAvoid : ['Avoid ungrounded assertions without concrete evidence']).map((w, i) => (
                    <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
                      <span style={{ color: 'var(--accent-amber)' }}>&bull;</span>
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Tactical Recommendations */}
              <div style={{ background: 'rgba(99, 102, 241, 0.04)', border: '1px solid rgba(99, 102, 241, 0.15)', borderRadius: 'var(--radius-sm)', padding: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-indigo)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '10px' }}>
                  <Zap size={16} />
                  Tactical Recommendations
                </div>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  {(recommendations.length > 0 ? recommendations : ['Cite authoritative entities and empirical data to raise evidence score']).map((imp, i) => (
                    <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
                      <span style={{ color: 'var(--accent-indigo)' }}>&bull;</span>
                      <span>{imp}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )}

        {/* Privacy Note */}
        <div style={{
          padding: '12px 16px',
          background: 'rgba(13, 18, 31, 0.6)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          color: 'var(--text-muted)',
          fontSize: '0.8rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '20px'
        }}>
          <Lock size={14} color="var(--accent-cyan)" />
          <span><strong>Server-Side Privacy Guarantee:</strong> Private tactical counterarguments generated during the debate were delivered solely to your private screen.</span>
        </div>

        {/* Action Button */}
        <div style={{ textAlign: 'right' }}>
          <button onClick={onClose} className="btn-secondary">
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
}
