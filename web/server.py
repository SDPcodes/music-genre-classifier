from flask import Flask, render_template, request, jsonify
import subprocess
import os

# Set template folder to current directory so it finds index.html
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get JSON data from frontend
    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({"error": "No lyrics provided"}), 400
    
    lyrics = data['lyrics']
    
    # 2. Write lyrics to a temp file for Spark to read
    # Use an absolute path or ensure it's in the project root
    temp_file = "temp_input.txt"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(lyrics)

        # 3. Execute Spark Prediction
        # We run spark-shell in 'non-interactive' mode by passing the scala script
        process = subprocess.Popen(
            ['spark-shell', '--master', 'local[*]', '-i', 'spark/predict.scala'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        stdout, stderr = process.communicate()

        # 4. Parse the specific output tag defined in your predict.scala
        if "RESULT_START|" in stdout:
            parts = stdout.split("RESULT_START|")[1].split("|RESULT_END")
            raw_res = parts[0].strip()
            
            # Convert "pop:0.8,rock:0.1" string into a Python Dictionary
            scores = {}
            for item in raw_res.split(","):
                key, val = item.split(":")
                scores[key] = float(val)
            
            return jsonify(scores)
        else:
            print("Spark Error Output:", stderr)
            return jsonify({"error": "Spark failed to produce a result", "details": stderr}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    # Listen on all interfaces so the 'open' command in run.sh works smoothly
    app.run(host='127.0.0.1', port=8080, debug=True)