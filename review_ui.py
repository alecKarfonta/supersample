from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import shutil
import sqlite3
import datetime

app = Flask(__name__)

INPUT_DIR = 'examples'
OUTPUT_DIR = 'output'
GOOD_DIR = 'good_output'
BAD_DIR = 'bad_output'
DB_PATH = 'review_stats.db'

os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original TEXT,
            noise_level REAL,
            inference_steps INTEGER,
            guidance_scale REAL,
            prompt TEXT,
            rating TEXT,
            timestamp TEXT
        )''')
        conn.commit()

init_db()

def parse_params_from_filename(filename):
    # Example: name_n0.25_i25_g7.5_game_texture.png
    import re
    m = re.match(r"(.+)_n([\d.]+)_i(\d+)_g([\d.]+)_([\w_]+)\.png", filename)
    if m:
        base, noise, steps, guidance, prompt = m.groups()
        return {
            'original': base + '.png',
            'noise_level': float(noise),
            'inference_steps': int(steps),
            'guidance_scale': float(guidance),
            'prompt': prompt.replace('_', ' ')
        }
    return None

@app.route('/')
def index():
    # List all original images
    originals = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.png')]
    # For each original, find all generated images
    data = []
    for orig in originals:
        base, _ = os.path.splitext(orig)
        generated = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(base) and f.lower().endswith('.png')]
        data.append({'original': orig, 'generated': generated})
    return render_template('index.html', data=data)

@app.route('/input/<filename>')
def input_image(filename):
    return send_from_directory(INPUT_DIR, filename)

@app.route('/output/<filename>')
def output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/move_images', methods=['POST'])
def move_images():
    data = request.json
    good = data.get('good', [])
    bad = data.get('bad', [])
    now = datetime.datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for fname in good:
            src = os.path.join(OUTPUT_DIR, fname)
            dst = os.path.join(GOOD_DIR, fname)
            params = parse_params_from_filename(fname)
            if os.path.exists(src):
                shutil.move(src, dst)
            if params:
                c.execute('''INSERT INTO ratings (filename, original, noise_level, inference_steps, guidance_scale, prompt, rating, timestamp)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                          (fname, params['original'], params['noise_level'], params['inference_steps'], params['guidance_scale'], params['prompt'], 'good', now))
        for fname in bad:
            src = os.path.join(OUTPUT_DIR, fname)
            dst = os.path.join(BAD_DIR, fname)
            params = parse_params_from_filename(fname)
            if os.path.exists(src):
                shutil.move(src, dst)
            if params:
                c.execute('''INSERT INTO ratings (filename, original, noise_level, inference_steps, guidance_scale, prompt, rating, timestamp)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                          (fname, params['original'], params['noise_level'], params['inference_steps'], params['guidance_scale'], params['prompt'], 'bad', now))
        conn.commit()
    return jsonify({'status': 'ok'})

@app.route('/stats')
def stats():
    # Group by parameter set, count good/bad
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''SELECT noise_level, inference_steps, guidance_scale, prompt,
                            SUM(rating='good'), SUM(rating='bad'), COUNT(*)
                     FROM ratings
                     GROUP BY noise_level, inference_steps, guidance_scale, prompt
                     ORDER BY SUM(rating='good')*1.0/COUNT(*) DESC''')
        rows = c.fetchall()
    stats = [
        {
            'noise_level': row[0],
            'inference_steps': row[1],
            'guidance_scale': row[2],
            'prompt': row[3],
            'good': row[4],
            'bad': row[5],
            'total': row[6],
            'success_rate': (row[4] / row[6]) if row[6] else 0
        }
        for row in rows
    ]
    return render_template('stats.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') 