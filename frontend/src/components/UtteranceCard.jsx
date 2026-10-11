import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Flame, Brain, BookOpen, Quote, ShieldAlert, Sparkles, Activity, ShieldX, Tag, Layers, MessageSquare } from 'lucide-react';

export default function UtteranceCard({ utterance }) {
  const isProcessing = utterance.status === 'PROCESSING';
  const isNoSpeech = utterance.status === 'NO_SPEECH';
  const isAudioError = utterance.status === 'AUDIO_ERROR' || utterance.error_type === 'AUDIO_ERROR';
  const isTranscriptionError = utterance.status === 'TRANSCRIPTION_ERROR' || utterance.error_type === 'TRANSCRIPTION_ERROR';
  const isNlpError = utterance.status === 'NLP_ERROR' || utterance.error_type === 'NLP_ERROR';
  const isGenericError = utterance.status === 'ERROR' && !isAudioError && !isTranscriptionError && !isNlpError;
  const isError = isAudioError || isTranscriptionError || isNlpError || isGenericError;
  const nlp = utterance.nlp_analysis;
  const isSocial = nlp?.content_type === 'social_statement';
  const intervention = nlp?.intervention;
  const isInterventionTriggered = Boolean(intervention && (intervention.triggered || intervention.has_intervention));

  // Format time
  const timeString = utterance.created_at
    ? new Date(utterance.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '';

  // Clean speaker label
  const speakerBadge = utterance.speaker_label || 'Speaker';
  const speakerName = utterance.speaker_name || '';

  // Helper for intervention title formatting
  const getInterventionTitle = (type) => {
    switch (type) {
      case 'personal_attack':
        return 'Personal Attack Detected';
      case 'abusive_language':
        return 'Abusive Language Detected';
      case 'unclear_statement':
        return 'Unclear / Fragmented Statement';
      case 'insufficient_argument':
        return 'Insufficient Argument';
      case 'low_information':
        return 'Low-Information Response';
      default:
        return (type || 'Speech Quality Warning').replace(/_/g, ' ');
    }
  };

  // Helper for entity label formatting
  const formatEntityLabel = (label) => {
    switch (label) {
      case 'PER':
        return 'PERSON';
      case 'ORG':
        return 'ORGANIZATION';
      case 'LOC':
        return 'LOCATION';
      default:
        return label || 'ENTITY';
    }
  };

  // Helper for bias display
  const getBiasDisplay = (bias) => {
    if (!bias) return 'Not detected';
    if (bias.label) return bias.label;
    if (bias.status === 'detected' || bias.prediction === '1') return 'Bias-associated language detected';
    if (bias.status === 'possible') return 'Possible bias-associated phrasing';
    return 'Not detected';
  };

  const entities = nlp?.ner || nlp?.evidence?.entities || [];

  return (
    <div className="glass-panel" style={{
      padding: '20px 24px',
      marginBottom: '16px',
      borderLeft: `4px solid ${utterance.speaker_label?.includes('A') ? 'var(--accent-cyan)' : 'var(--accent-purple)'}`,
      transition: 'all 0.2s ease'
    }}>
      {/* Header: Speaker & Status */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="badge badge-speaker">
            {speakerBadge}
          </span>
          {speakerName && (
            <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.95rem' }}>
              {speakerName}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {timeString}
        </div>
      </div>

      {/* Processing State */}
      {isProcessing && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '16px',
          background: 'rgba(99, 102, 241, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px dashed rgba(99, 102, 241, 0.3)'
        }}>
          <Activity size={20} className="pulse-dot" color="var(--accent-cyan)" />
          <span style={{ color: 'var(--accent-cyan)', fontSize: '0.9rem', fontWeight: 500 }}>
            Analyzing your speech...
          </span>
        </div>
      )}

      {/* No Speech State */}
      {isNoSpeech && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(245, 158, 11, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(245, 158, 11, 0.2)',
          color: 'var(--accent-amber)',
          fontSize: '0.88rem'
        }}>
          No recognizable speech detected.
        </div>
      )}

      {/* Audio Error */}
      {isAudioError && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(244, 63, 94, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(244, 63, 94, 0.2)',
          color: 'var(--accent-rose)',
          fontSize: '0.88rem'
        }}>
          Audio could not be processed. Please try again.
        </div>
      )}

      {/* Transcription Error */}
      {isTranscriptionError && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(244, 63, 94, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(244, 63, 94, 0.2)',
          color: 'var(--accent-rose)',
          fontSize: '0.88rem'
        }}>
          Speech transcription failed. Please try again.
        </div>
      )}

      {/* Generic Error */}
      {isGenericError && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(244, 63, 94, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(244, 63, 94, 0.2)',
          color: 'var(--accent-rose)',
          fontSize: '0.88rem'
        }}>
          {utterance.error_message || 'Processing error occurred. Please try again.'}
        </div>
      )}

      {/* Actual Transcript */}
      {!isProcessing && utterance.transcript && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{
            fontSize: '1.05rem',
            lineHeight: 1.5,
            color: 'var(--text-primary)',
            fontStyle: 'normal',
            background: 'rgba(13, 18, 31, 0.6)',
            padding: '14px 18px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)'
          }}>
            <Quote size={16} style={{ display: 'inline', marginRight: '6px', opacity: 0.5 }} />
            "{utterance.transcript}"
          </div>
        </div>
      )}

      {/* NLP Error State (Transcript is preserved above, analysis unavailable) */}
      {isNlpError && (
        <div style={{
          marginBottom: '16px',
          padding: '12px 16px',
          background: 'rgba(245, 158, 11, 0.08)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          color: 'var(--accent-amber)',
          fontSize: '0.88rem'
        }}>
          Speech was transcribed, but analysis could not be completed.
        </div>
      )}

      {/* Speech Intervention Warning Box (if detected) */}
      {isInterventionTriggered && (
        <div style={{
          marginBottom: '16px',
          background: (intervention.severity === 'high' || intervention.severity === 'critical')
            ? 'linear-gradient(135deg, rgba(244, 63, 94, 0.18), rgba(225, 29, 72, 0.1))'
            : 'linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.08))',
          border: `1px solid ${(intervention.severity === 'high' || intervention.severity === 'critical') ? 'rgba(244, 63, 94, 0.5)' : 'rgba(245, 158, 11, 0.5)'}`,
          borderRadius: 'var(--radius-sm)',
          padding: '16px 20px',
          boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {(intervention.severity === 'high' || intervention.severity === 'critical') ? (
                <ShieldX size={20} color="var(--accent-rose)" />
              ) : (
                <AlertTriangle size={20} color="var(--accent-amber)" />
              )}
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{
                  fontSize: '0.72rem',
                  letterSpacing: '0.08em',
                  fontWeight: 800,
                  textTransform: 'uppercase',
                  color: (intervention.severity === 'high' || intervention.severity === 'critical') ? 'var(--accent-rose)' : 'var(--accent-amber)'
                }}>
                  SPEECH WARNING
                </span>
                <span style={{
                  fontWeight: 700,
                  fontSize: '1rem',
                  color: '#fff'
                }}>
                  {getInterventionTitle(intervention.type || intervention.intervention_type)}
                </span>
              </div>
            </div>
            <span className="font-mono" style={{
              background: 'rgba(0,0,0,0.4)',
              padding: '4px 10px',
              borderRadius: '4px',
              color: 'var(--accent-rose)',
              fontWeight: 700,
              fontSize: '0.88rem',
              border: '1px solid rgba(244, 63, 94, 0.3)'
            }}>
              Penalty: -{intervention.penalty}
            </span>
          </div>

          <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: (intervention.details && intervention.details !== intervention.warning) ? '6px' : '0' }}>
            {intervention.warning}
          </div>

          {intervention.details && intervention.details !== intervention.warning && (
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              {intervention.details}
            </div>
          )}
        </div>
      )}

      {/* Non-Argument / Social Statement Banner */}
      {isSocial && (
        <div style={{
          padding: '14px 18px',
          background: 'rgba(56, 189, 248, 0.06)',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          marginTop: '10px'
        }}>
          <MessageSquare size={20} color="var(--accent-cyan)" />
          <div>
            <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--accent-cyan)' }}>
              Non-argument / Social Statement
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              No substantive debate argument detected. logical fallacy and evidence analyses are not applicable.
            </div>
          </div>
        </div>
      )}

      {/* NLP Analytical Breakdown (for substantive arguments & other evaluated turns) */}
      {!isSocial && nlp && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '12px',
          marginTop: '14px'
        }}>
          {/* Argument Quality & Component */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-cyan)', fontSize: '0.85rem', fontWeight: 600 }}>
              <Brain size={16} />
              Argument Structure & Quality
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>Component: <strong style={{ color: 'var(--text-primary)' }}>{nlp.argument_component?.prediction || nlp.argument_component?.label || 'General'}</strong></div>
              <div>Quality Score: <strong style={{ color: 'var(--accent-cyan)' }}>{nlp.quality?.score !== null && nlp.quality?.score !== undefined ? `${nlp.quality.score}/100` : 'N/A'}</strong> ({nlp.quality?.level || 'Moderate'})</div>
              {nlp.quality?.intervention_penalty > 0 && (
                <div style={{ fontSize: '0.75rem', color: 'var(--accent-rose)' }}>
                  Base: {nlp.quality.base_score} &bull; Penalty: -{nlp.quality.intervention_penalty} pts
                </div>
              )}
            </div>
          </div>

          {/* Clarity & Structure */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-indigo)', fontSize: '0.85rem', fontWeight: 600 }}>
              <Layers size={16} />
              Clarity & Structure
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>
                Status: <strong style={{ color: (nlp.quality?.clarity_score < 70) ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                  {nlp.quality?.clarity_status || 'Clear & Structured'} {nlp.quality?.clarity_score ? `(${nlp.quality.clarity_score}/100)` : ''}
                </strong>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {nlp.quality?.clarity_score >= 80 ? 'Coherent premise & syntax' : 'Statement structure needs clarification'}
              </div>
            </div>
          </div>

          {/* Potential Logical Fallacy */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-rose)', fontSize: '0.85rem', fontWeight: 600 }}>
              <ShieldAlert size={16} />
              Logical Fallacy Signal
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>
                Signal: <strong style={{ color: nlp.fallacy?.prediction ? 'var(--accent-rose)' : 'var(--text-primary)' }}>
                  {nlp.fallacy?.prediction ? (nlp.fallacy.prediction.charAt(0).toUpperCase() + nlp.fallacy.prediction.slice(1)) : 'No strong signal detected'}
                </strong>
              </div>
              {nlp.fallacy?.prediction && nlp.fallacy?.decision_strength !== undefined && nlp.fallacy?.decision_strength !== null && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence: {(nlp.fallacy.decision_strength * 100).toFixed(1)}%</div>
              )}
            </div>
          </div>

          {/* Tone & Bias Signals */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-amber)', fontSize: '0.85rem', fontWeight: 600 }}>
              <Flame size={16} />
              Tone & Bias Signals
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>Sentiment: <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>{nlp.sentiment?.label || 'Neutral'}</strong></div>
              <div>Bias-associated: <strong style={{ color: 'var(--text-primary)' }}>{getBiasDisplay(nlp.bias)}</strong></div>
            </div>
          </div>

          {/* Evidence Signals */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-emerald)', fontSize: '0.85rem', fontWeight: 600 }}>
              <BookOpen size={16} />
              Evidence Signals
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <div>Evidence: <strong style={{ color: 'var(--text-primary)' }}>{nlp.evidence?.is_evidence_bearing ? 'Present' : 'Limited'}</strong></div>
              {nlp.evidence?.sources?.length > 0 && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Sources: {nlp.evidence.sources.join(', ')}
                </div>
              )}
            </div>
          </div>

          {/* Named Entity Recognition (NER) */}
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', color: 'var(--accent-cyan)', fontSize: '0.85rem', fontWeight: 600 }}>
              <Tag size={16} />
              Named Entities
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {entities.length === 0 ? (
                <div style={{ fontStyle: 'italic', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                  No notable entities detected.
                </div>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '4px' }}>
                  {entities.map((ent, idx) => (
                    <span key={idx} style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      background: 'rgba(56, 189, 248, 0.1)',
                      border: '1px solid rgba(56, 189, 248, 0.25)',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '0.78rem'
                    }}>
                      <strong style={{ color: 'var(--accent-cyan)', fontSize: '0.7rem', textTransform: 'uppercase' }}>
                        {formatEntityLabel(ent.label)}
                      </strong>
                      <span style={{ color: 'var(--text-primary)' }}>{ent.text}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Private Counterargument Recommendation (Author-Only via Server-Side Privacy) */}
          {nlp.counterargument?.counterargument && (
            <div style={{
              gridColumn: '1 / -1',
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(56, 189, 248, 0.05))',
              padding: '14px 18px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid rgba(99, 102, 241, 0.25)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-indigo)', fontSize: '0.88rem', fontWeight: 600 }}>
                  <Sparkles size={16} />
                  Private Tactical Counterargument & Coaching
                </div>
                <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--accent-cyan)', fontSize: '0.72rem' }}>
                  Private to You
                </span>
              </div>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', fontStyle: 'italic', marginBottom: '4px' }}>
                "{nlp.counterargument.counterargument}"
              </div>
              {nlp.counterargument.reason && (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Strategy: {nlp.counterargument.reason}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
