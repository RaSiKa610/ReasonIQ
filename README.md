# ReasonIQ

ReasonIQ is a real-time AI-powered debate and argument analysis system. It captures live spoken discourse, transcribes utterances via faster-whisper, performs argument quality and fallacy detection, analyzes biases, and streams live insights and interventions back to participants over WebSockets.

---

## Architecture Overview

- **Backend**: FastAPI with async WebSocket streaming, faster-whisper speech pipeline, Silero VAD, and Scikit-learn/PyTorch NLP analysis models.
- **Frontend**: Modern React + Vite application with live audio capture, live debate transcription, real-time intervention badges, and critical thinking analytics.
- **Models**: Pre-trained machine learning classifiers for Argument Component (AAEC), Fallacy Detection (LOGIC), and Bias Detection (WNC).

---

## Getting Started

### Prerequisites

- **Python**: 3.10+ (tested on Python 3.11)
- **Node.js**: 18+ and npm
- **FFmpeg**: Required for audio processing with PyAV / faster-whisper

---

### Backend Setup

1. **Create and activate a virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server**:
   ```bash
   python main.py
   ```
   The backend server runs on `http://localhost:8000`.

---

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   The frontend runs on `http://localhost:5173`.

---

### Running Tests

To run the backend test suite:
```bash
pytest
```
