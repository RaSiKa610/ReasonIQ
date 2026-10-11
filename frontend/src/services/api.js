/**
 * ReasonIQ Frontend API Client
 * Handles communication with the FastAPI backend.
 */

const getApiBase = () => {
  if (import.meta.env?.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  const host = window.location.hostname || 'localhost';
  return `http://${host}:8000/api`;
};

const extractErrorMessage = (errData, fallback) => {
  if (!errData) return fallback;
  if (typeof errData.detail === 'string') return errData.detail;
  if (errData.detail && typeof errData.detail === 'object') {
    return errData.detail.message || errData.detail.code || fallback;
  }
  if (errData.error && typeof errData.error === 'object') {
    return errData.error.message || errData.error.code || fallback;
  }
  if (errData.message) return errData.message;
  return fallback;
};

export async function createRoom(hostName, topic) {
  const res = await fetch(`${getApiBase()}/rooms`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ host_name: hostName, topic })
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to create room' }));
    throw new Error(extractErrorMessage(error, 'Failed to create room'));
  }
  return res.json();
}

export async function getRoom(roomCode) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Room not found' }));
    throw new Error(extractErrorMessage(error, 'Room not found'));
  }
  return res.json();
}

export async function joinRoom(roomCode, participantName) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/join`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ participant_name: participantName })
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to join room' }));
    throw new Error(extractErrorMessage(error, 'Failed to join room'));
  }
  return res.json();
}

export async function admitParticipant(roomCode, participantId, hostToken) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/participants/${participantId}/admit`, {
    method: 'POST',
    headers: {
      'X-Host-Token': hostToken
    }
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to admit participant' }));
    throw new Error(extractErrorMessage(error, 'Failed to admit participant'));
  }
  return res.json();
}

export async function rejectParticipant(roomCode, participantId, hostToken) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/participants/${participantId}/reject`, {
    method: 'POST',
    headers: {
      'X-Host-Token': hostToken
    }
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to reject participant' }));
    throw new Error(extractErrorMessage(error, 'Failed to reject participant'));
  }
  return res.json();
}

export async function startDebate(roomCode, hostToken) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/start`, {
    method: 'POST',
    headers: {
      'X-Host-Token': hostToken
    }
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to start debate' }));
    throw new Error(extractErrorMessage(error, 'Failed to start debate'));
  }
  return res.json();
}

export async function endDebate(roomCode, hostToken) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/end`, {
    method: 'POST',
    headers: {
      'X-Host-Token': hostToken
    }
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to end debate' }));
    throw new Error(extractErrorMessage(error, 'Failed to end debate'));
  }
  return res.json();
}

export async function updateSettings(roomCode, hostToken, allowInterruptions) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/settings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Host-Token': hostToken
    },
    body: JSON.stringify({ allow_interruptions: allowInterruptions })
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to update settings' }));
    throw new Error(extractErrorMessage(error, 'Failed to update settings'));
  }
  return res.json();
}

export async function getMyReport(roomCode, participantId) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/reports/me?participant_id=${participantId}`);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to fetch personal report' }));
    throw new Error(extractErrorMessage(error, 'Failed to fetch personal report'));
  }
  const data = await res.json();
  return data.report || data;
}

export async function getParticipantReport(roomCode, participantId, hostToken = null) {
  const headers = {};
  if (hostToken) {
    headers['X-Host-Token'] = hostToken;
  }
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/reports/${participantId}`, {
    headers
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to fetch report' }));
    throw new Error(extractErrorMessage(error, 'Failed to fetch report'));
  }
  const data = await res.json();
  return data.report || data;
}

export async function getAllReports(roomCode, hostToken) {
  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/reports`, {
    headers: {
      'X-Host-Token': hostToken
    }
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to fetch reports' }));
    throw new Error(extractErrorMessage(error, 'Failed to fetch reports'));
  }
  const data = await res.json();
  return data.reports || data;
}

export async function submitUtterance(roomCode, participantId, audioBlob) {
  const formData = new FormData();
  formData.append('participant_id', participantId);
  const blobType = audioBlob.type || '';
  const ext = blobType.includes('mp4') ? 'utterance.mp4' : blobType.includes('ogg') ? 'utterance.ogg' : blobType.includes('wav') ? 'utterance.wav' : 'utterance.webm';
  formData.append('audio_file', audioBlob, ext);

  const res = await fetch(`${getApiBase()}/rooms/${roomCode.toUpperCase()}/utterance`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to process audio' }));
    throw new Error(extractErrorMessage(error, 'Failed to process audio'));
  }
  return res.json();
}

