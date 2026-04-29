import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify
from src.inference import ASRPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ASR Pipeline
# We'll load models on demand or at startup
pipeline = ASRPipeline()
try:
    pipeline.load_models()
    logger.info("ASR Models loaded successfully")
except Exception as e:
    logger.warning(f"Could not load models: {e}. Running in mock mode.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400
    
    # Save the file temporarily
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    try:
        # Run transcription
        # If models are not loaded, transcribe() will return placeholders
        result = pipeline.transcribe(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'text': result['text'] or f"Detected speech in {result['language']}. (Model output empty)",
            'language': result['language'].capitalize(),
            'confidence': result['confidence']
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
