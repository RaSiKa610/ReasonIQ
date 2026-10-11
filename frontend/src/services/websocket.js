/**
 * ReasonIQ WebSocket Service
 * Handles live event streaming for rooms.
 */

export class RoomSocket {
  constructor(roomCode, participantId = null, role = 'participant', onMessage = null, onStatusChange = null) {
    this.roomCode = roomCode.toUpperCase();
    this.participantId = participantId;
    this.role = role;
    this.onMessage = onMessage;
    this.onStatusChange = onStatusChange;
    this.ws = null;
    this.pingInterval = null;
    this.shouldReconnect = true;
  }

  connect() {
    const base = import.meta.env?.VITE_WS_BASE_URL || `ws://${window.location.hostname || 'localhost'}:8000/ws`;
    let url = `${base}/rooms/${this.roomCode}?role=${this.role}`;
    if (this.participantId) {
      url += `&participant_id=${this.participantId}`;
    }

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        if (this.onStatusChange) this.onStatusChange('connected');
        // Start ping heartbeat
        this.pingInterval = setInterval(() => {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'PING' }));
          }
        }, 15000);
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (this.onMessage) {
            this.onMessage(payload.type, payload.data);
          }
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      this.ws.onclose = () => {
        if (this.onStatusChange) this.onStatusChange('disconnected');
        clearInterval(this.pingInterval);
        if (this.shouldReconnect) {
          setTimeout(() => this.connect(), 2000);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
    }
  }

  sendMicStatus(micActive) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'MIC_STATUS',
        participant_id: this.participantId,
        mic_active: micActive
      }));
    }
  }

  disconnect() {
    this.shouldReconnect = false;
    clearInterval(this.pingInterval);
    if (this.ws) {
      this.ws.close();
    }
  }
}
