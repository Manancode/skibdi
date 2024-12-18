from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice
import torchaudio
import tempfile
import os
import logging

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize TTS
tts = TextToSpeech()

@app.route('/tts', methods=['POST'])
def generate_tts():
    """
    POST request handler for TTS generation.
    Expects:
    - 'text' in the request form (the text to convert to speech)
    - An audio file in the form-data (used for the custom voice)
    """
    try:
        app.logger.info("Received a request to /tts")

        # Extract text and audio file from the request
        text = request.form.get('text')
        uploaded_file = request.files.get('audio_file')

        if not text:
            app.logger.warning("Missing 'text' in request")
            return jsonify({"error": "Text is required."}), 400

        if not uploaded_file:
            app.logger.warning("Missing 'audio_file' in request")
            return jsonify({"error": "Audio file is required."}), 400

        # Process the audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_voice_name = os.path.splitext(uploaded_file.filename)[0]
            audio_file_path = os.path.join(temp_dir, f"{custom_voice_name}.wav")
            uploaded_file.save(audio_file_path)

            # Load voice samples
            voice_samples, conditioning_latents = load_voice(audio_file_path)

            # Generate the speech
            generated_audio = tts.tts_with_preset(
                text=text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset="fast"  # Choose from "ultra_fast", "fast", "standard", "high_quality"
            )

            # Save the generated audio to a file
            output_audio_path = os.path.join(temp_dir, f"generated-{custom_voice_name}.wav")
            torchaudio.save(output_audio_path, generated_audio.squeeze(0).cpu(), 24000)

            # Send the generated audio file back to the client
            return send_file(output_audio_path, mimetype='audio/wav', as_attachment=True)

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred during processing."}), 500

if __name__ == '__main__':
    app.run(debug=True , use_reloader=False)
