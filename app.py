import os
import uuid
import logging
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import google.generativeai as genai

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- App Configuration Class ---
class Config:
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'generated_audio'
    TEMPLATE_FOLDER = '.' # Look for index.html in the root directory
    DEVICE = "cpu"
    DEBUG = True
    PORT = 5000
    # IMPORTANT: It's recommended to set your API key as an environment variable
    # For example: export GOOGLE_API_KEY="YOUR_API_KEY"
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=Config.TEMPLATE_FOLDER)
app.config.from_object(Config)
CORS(app)

# --- Ensure Directories Exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Load Models ---
def load_tts_model():
    logging.info(f"Loading ChatterboxTTS model on device: {app.config['DEVICE']}...")
    try:
        model = ChatterboxTTS.from_pretrained(device=app.config['DEVICE'])
        logging.info("ChatterboxTTS model loaded successfully.")
        return model
    except Exception as e:
        logging.critical(f"FATAL: Could not load ChatterboxTTS model. Error: {e}")
        return None

def configure_gemini():
    if not app.config['GOOGLE_API_KEY']:
        logging.warning("GOOGLE_API_KEY environment variable not set. Gemini features will not work.")
        return None
    try:
        genai.configure(api_key=app.config['GOOGLE_API_KEY'])
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Gemini API configured successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        return None

tts_model = load_tts_model()
gemini_model = configure_gemini()

# --- TTS Route ---
@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    if tts_model is None:
        return jsonify({"error": "TTS model is not available."}), 503
    if 'text' not in request.form or not request.form['text'].strip():
        return jsonify({"error": "No text provided"}), 400
    text_to_generate = request.form['text']
    reference_audio_path = None
    try:
        # We still receive these values from the UI, but we won't use the unsupported ones.
        temperature = float(request.form.get('temperature', 0.75))
        # cfg_scale = float(request.form.get('cfg_scale', 5.0)) # Removed due to error
        # pace = float(request.form.get('pace', 1.0)) # Removed due to error

        if 'reference_audio' in request.files:
            audio_file = request.files['reference_audio']
            if audio_file and audio_file.filename != '':
                temp_ref_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{uuid.uuid4().hex}.wav")
                audio_file.save(temp_ref_filename)
                reference_audio_path = temp_ref_filename
        
        with torch.no_grad():
            # **FIX**: Removed the unsupported 'cfg_scale' and 'pace' arguments.
            wav = tts_model.generate(text_to_generate, 
                                     audio_prompt_path=reference_audio_path, 
                                     temperature=temperature)

        output_filename = f"iris_output_{uuid.uuid4().hex}.wav"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        ta.save(output_filepath, wav, tts_model.sr)
        file_url = f"http://127.0.0.1:{app.config['PORT']}/audio/{output_filename}"
        return jsonify({"message": "TTS generated successfully", "file_url": file_url, "file_name": output_filename})
    except Exception as e:
        logging.error(f"TTS Error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during TTS generation."}), 500
    finally:
        if reference_audio_path and os.path.exists(reference_audio_path):
            os.remove(reference_audio_path)

# --- Gemini API Routes ---
def handle_gemini_request(prompt, is_json=False):
    if not gemini_model:
        return jsonify({"error": "Gemini API is not configured on the server."}), 503
    try:
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json") if is_json else None
        response = gemini_model.generate_content(prompt, generation_config=generation_config)
        return jsonify({"text": response.text})
    except Exception as e:
        logging.error(f"Gemini API request failed: {e}")
        return jsonify({"error": "Failed to get response from Gemini API."}), 500

@app.route('/generate_text', methods=['POST'])
def generate_text_route():
    data = request.get_json()
    prompt = f"Generate a short, creative text based on this prompt: {data.get('prompt')}"
    return handle_gemini_request(prompt)

@app.route('/polish_text', methods=['POST'])
def polish_text_route():
    data = request.get_json()
    prompt = f"You are a professional voice-over script writer. Polish the following text to make it more eloquent, impactful, and natural-sounding when read aloud. Do not add any explanatory text, just provide the polished script.\n\nOriginal Text: \"{data.get('text')}\""
    return handle_gemini_request(prompt)

@app.route('/suggest_scene', methods=['POST'])
def suggest_scene_route():
    data = request.get_json()
    prompt = f"You are a creative director. For the following voice-over text, describe a compelling visual scene where this audio would be used. Be descriptive and evocative.\n\nVoice-over Text: \"{data.get('text')}\""
    return handle_gemini_request(prompt)

@app.route('/analyze_mood', methods=['POST'])
def analyze_mood_route():
    if not gemini_model:
        return jsonify({"error": "Gemini API is not configured on the server."}), 503
    data = request.get_json()
    text = data.get('text')
    mood_analyzer = genai.GenerativeModel(
        'gemini-1.5-flash-latest',
        tools=[genai.protos.Tool(
            function_declarations=[genai.protos.FunctionDeclaration(
                name='submit_voice_parameters',
                description="Submit the detected mood and suggested voice parameters.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        'mood': genai.protos.Schema(type=genai.protos.Type.STRING, description="A single word for the dominant mood (e.g., 'Happy', 'Serious')."),
                        'suggested_expressiveness': genai.protos.Schema(type=genai.protos.Type.NUMBER, description="A value from 0.1 to 1.0 for expressiveness."),
                        'suggested_pace': genai.protos.Schema(type=genai.protos.Type.NUMBER, description="A value from 0.5 to 2.0 for pace.")
                    },
                    required=['mood', 'suggested_expressiveness', 'suggested_pace']
                )
            )]
        )],
        tool_config=genai.protos.ToolConfig(function_calling_config=genai.protos.FunctionCallingConfig(mode="ANY"))
    )
    try:
        response = mood_analyzer.generate_content(f"Analyze the mood of the following text and submit the suggested voice parameters: \"{text}\"")
        func_call = response.candidates[0].content.parts[0].function_call
        args = {key: value for key, value in func_call.args.items()}
        return jsonify(args)
    except Exception as e:
        logging.error(f"Gemini mood analysis error: {e}")
        return jsonify({"error": "Failed to analyze mood with Gemini."}), 500

# --- Static and Utility Routes ---
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/')
def index():
    """Serves the index.html file as the main page."""
    return render_template('hari.html')

# --- Main Execution ---
def open_browser():
    """Opens the default web browser to the application's URL."""
    webbrowser.open_new(f"http://127.0.0.1:{app.config['PORT']}/")

if __name__ == '__main__':
    if tts_model is None or gemini_model is None:
        logging.warning("\nServer is starting, but some features will FAIL. Check logs for model loading or API key errors.")
    
    # Use a timer to open the browser 1 second after the script runs
    Timer(1, open_browser).start()
    
    # Run the Flask app
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])
