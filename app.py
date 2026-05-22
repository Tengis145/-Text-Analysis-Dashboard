from flask import Flask, render_template, request, jsonify, send_file, redirect
import re
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from collections import Counter
import json
import math
import nltk
import pandas as pd
from config import UPLOAD_FOLDER, ALLOWED_VIDEO, ALLOWED_SUBTITLE, ALLOWED_EXCEL, PORT, DEBUG, MAX_FILE_SIZE

# Download required NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    import pysrt
except ImportError:
    pysrt = None

try:
    import webvtt
except ImportError:
    webvtt = None

try:
    import whisper
except ImportError:
    whisper = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# In-memory storage for current transcript
current_transcript = {
    'segments': [],
    'full_text': '',
    'words': [],
    'word_freq': {},
    'filename': '',
    'status': 'idle',  # idle, processing, ready
    'error': None
}

# Common stopwords (English + Mongolian)
STOPWORDS = {
    # English
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'as', 'with', 'by', 'from', 'up', 'out', 'if',
    # Mongolian particles / auxiliaries / filler words
    'нь', 'ба', 'бөгөөд', 'буюу', 'эс', 'юм', 'юмуу', 'байна', 'байсан', 'байх', 'байгаа', 'байж',
    'аль', 'энэ', 'тэр', 'эд', 'тэд', 'би', 'чи', 'та', 'бид', 'тэдэнд', 'үү', 'дүү', 'рүү', 'луу',
    'дээр', 'доор', 'дотор', 'гадна', 'хажуу', 'өмнө', 'хойно', 'зүгт', 'талд',
    'болон', 'боловч', 'гэвч', 'харин', 'тэгвэл', 'тийм', 'үгүй', 'ч', 'л', 'гэж', 'гэх', 'гэсэн',
    'ий', 'ийн', 'ын', 'ийг', 'ыг', 'аас', 'ээс', 'оос', 'өөс', 'аар', 'ээр', 'оор', 'өөр',
    'ийд', 'д', 'т', 'нд', 'нт', 'даа', 'дээ', 'оо', 'өө', 'аа', 'ээ',
    'вэ', 'уу', 'ну', 'нь', 'мөн', 'ч', 'зэрэг', 'гэрт', 'хэн', 'яах', 'яасан',
    # Common Mongolian classroom filler words
    'за', 'заа', 'зав', 'ql', 'ba', 'nan',
}

def parse_srt(filepath):
    """Parse SRT subtitle file"""
    if not pysrt:
        return None
    try:
        subs = pysrt.open(filepath)
        segments = []
        for sub in subs:
            segments.append({
                'start': sub.start.ordinal / 1000.0,
                'end': sub.end.ordinal / 1000.0,
                'text': sub.text.replace('\n', ' ')
            })
        return segments
    except Exception as e:
        print(f"Error parsing SRT: {e}")
        return None

def parse_vtt(filepath):
    """Parse VTT subtitle file"""
    if not webvtt:
        return None
    try:
        vtt = webvtt.read(filepath)
        segments = []
        for cue in vtt:
            # Parse webvtt time format (HH:MM:SS.mmm)
            def parse_time(time_str):
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

            segments.append({
                'start': parse_time(cue.start),
                'end': parse_time(cue.end),
                'text': cue.text.replace('\n', ' ')
            })
        return segments
    except Exception as e:
        print(f"Error parsing VTT: {e}")
        return None

def transcribe_audio(filepath):
    """Transcribe audio/video using Whisper"""
    if not whisper:
        return None
    try:
        model = whisper.load_model("base")
        result = model.transcribe(filepath)
        segments = []
        for seg in result['segments']:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })
        return segments
    except Exception as e:
        print(f"Error transcribing: {e}")
        return None

def segments_to_fulltext(segments):
    full_text = ' '.join([seg['text'] for seg in segments])
    words = re.findall(r'\b[\w]+\b', full_text.lower())
    words = [
        w for w in words
        if len(w) > 1
        and w not in STOPWORDS
        and not w.isdigit()                   # filter pure numbers
        and not re.match(r'^\d+[.,]\d+$', w)  # filter decimals like 1.5
        and not re.match(r'^unnamed', w)       # filter pandas unnamed cols
        and not re.match(r'^\W+$', w)          # filter pure punctuation
    ]
    return full_text, words

def compute_word_freq(words):
    """Compute word frequency counter"""
    return dict(Counter(words).most_common(1000))

def load_transcript(segments):
    """Load transcript into memory"""
    global current_transcript
    full_text, words = segments_to_fulltext(segments)
    current_transcript['segments'] = segments
    current_transcript['full_text'] = full_text
    current_transcript['words'] = words
    current_transcript['word_freq'] = compute_word_freq(words)
    current_transcript['filename'] = 'transcript'
    current_transcript['status'] = 'ready'
    current_transcript['error'] = None

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    """Analysis dashboard"""
    if not current_transcript['full_text']:
        return redirect('/')
    return render_template('analysis.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    """Handle text, video, subtitle, or Excel file upload"""
    global current_transcript

    # Text input
    text_input = request.form.get('text', '').strip()
    file_type = request.form.get('type', 'auto')

    if text_input:
        sentences = nltk.sent_tokenize(text_input)
        segments = []
        time = 0
        for sent in sentences:
            word_count = len(sent.split())
            duration = max(word_count * 0.5, 1)
            segments.append({
                'start': time,
                'end': time + duration,
                'text': sent
            })
            time += duration
        load_transcript(segments)
        return jsonify({'ok': True, 'type': 'text'})

    # Video/Audio/Subtitle file upload
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'ok': False, 'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    filepath = UPLOAD_FOLDER / filename

    print(f"DEBUG: Original filename: {file.filename}")
    print(f"DEBUG: Secure filename: {filename}")
    print(f"DEBUG: Extension: {ext}")
    print(f"DEBUG: ALLOWED_EXCEL: {ALLOWED_EXCEL}")

    try:
        file.save(filepath)

        # CHECK FOR EXCEL FILES FIRST
        if ext == '.xlsx' or ext == '.xls':
            try:
                xls = pd.ExcelFile(filepath)
                combined_text = ""
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    if not df.empty:
                        combined_text += extract_text_columns(df) + " "
                
                if combined_text.strip():
                    sentences = nltk.sent_tokenize(combined_text)
                    segments = []
                    time = 0
                    for sent in sentences:
                        word_count = len(sent.split())
                        duration = max(word_count * 0.5, 1)
                        segments.append({
                            'start': time,
                            'end': time + duration,
                            'text': sent
                        })
                        time += duration
                    load_transcript(segments)
                    return jsonify({'ok': True, 'type': 'excel', 'filename': filename})
                else:
                    return jsonify({'ok': False, 'error': 'No text found in Excel file'}), 400
            except Exception as e:
                return jsonify({'ok': False, 'error': f'Error reading Excel: {str(e)}'}), 500
        
        # SRT file
        if ext == '.srt':
            segments = parse_srt(filepath)
            if segments:
                load_transcript(segments)
                return jsonify({'ok': True, 'type': 'srt', 'filename': filename})

        # VTT file
        if ext == '.vtt':
            segments = parse_vtt(filepath)
            if segments:
                load_transcript(segments)
                return jsonify({'ok': True, 'type': 'vtt', 'filename': filename})

        # TXT file
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text_input = f.read()
            sentences = nltk.sent_tokenize(text_input)
            segments = []
            time = 0
            for sent in sentences:
                word_count = len(sent.split())
                duration = max(word_count * 0.5, 1)
                segments.append({
                    'start': time,
                    'end': time + duration,
                    'text': sent
                })
                time += duration
            load_transcript(segments)
            return jsonify({'ok': True, 'type': 'txt', 'filename': filename})

        # Video/Audio files
        if ext in ALLOWED_VIDEO:
            if whisper:
                current_transcript['status'] = 'processing'
                current_transcript['error'] = None
                try:
                    segments = transcribe_audio(filepath)
                    if segments:
                        load_transcript(segments)
                        return jsonify({'ok': True, 'type': 'video_transcribed', 'filename': filename})
                    else:
                        return jsonify({'ok': False, 'error': 'Transcription failed'}), 500
                except Exception as e:
                    current_transcript['status'] = 'idle'
                    current_transcript['error'] = str(e)
                    return jsonify({'ok': False, 'error': f'Transcription error: {str(e)}'}), 500
            
            current_transcript['filename'] = filename
            current_transcript['status'] = 'ready'
            return jsonify({'ok': True, 'type': 'video', 'filename': filename})

        return jsonify({'ok': False, 'error': f'Unsupported file type: {ext}'}), 400

    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    if not current_transcript['words']:
        return jsonify({'error': 'No transcript loaded'}), 400

    words = current_transcript['words']
    full_text = current_transcript['full_text']

    # Compute statistics
    total_words = len(words)
    unique_words = len(set(words))
    vocab_density = unique_words / total_words if total_words > 0 else 0

    # Readability (Flesch-Kincaid grade level approximation)
    sentences = nltk.sent_tokenize(full_text)
    avg_words_per_sentence = total_words / len(sentences) if sentences else 0

    # Top 5 words
    top_words = dict(sorted(current_transcript['word_freq'].items(), key=lambda x: x[1], reverse=True)[:5])

    return jsonify({
        'total_words': total_words,
        'unique_words': unique_words,
        'vocab_density': round(vocab_density, 3),
        'avg_words_per_sentence': round(avg_words_per_sentence, 1),
        'num_sentences': len(sentences),
        'top_words': top_words
    })

@app.route('/api/word-cloud')
def get_word_cloud():
    """Get word cloud data (top N words)"""
    if not current_transcript['word_freq']:
        return jsonify({})

    limit = request.args.get('limit', 30, type=int)
    data = dict(list(current_transcript['word_freq'].items())[:limit])
    return jsonify(data)

@app.route('/api/trends')
def get_trends():
    """Get word frequency trends across 10 document segments"""
    if not current_transcript['words']:
        return jsonify({})

    words = current_transcript['words']
    segments = current_transcript['segments']
    query_terms = request.args.getlist('term')

    if not query_terms:
        query_terms = list(current_transcript['word_freq'].keys())[:5]

    query_terms = [t.lower() for t in query_terms]

    # Split text into 10 segments
    words_per_segment = max(1, len(words) // 10)
    trends = {}

    for term in query_terms:
        term_data = []
        for i in range(10):
            start_idx = i * words_per_segment
            end_idx = (i + 1) * words_per_segment if i < 9 else len(words)
            segment_words = words[start_idx:end_idx]
            term_count = segment_words.count(term)
            relative_freq = term_count / len(segment_words) if segment_words else 0
            term_data.append(round(relative_freq, 4))
        trends[term] = term_data

    return jsonify(trends)

@app.route('/api/contexts')
def get_contexts():
    """Get keyword-in-context (KWIC) data"""
    if not current_transcript['words']:
        return jsonify([])

    term = request.args.get('term', '').lower()
    if not term:
        return jsonify([])

    words = current_transcript['words']
    contexts = []

    for i, word in enumerate(words):
        if word == term:
            left = words[max(0, i-5):i]
            right = words[i+1:min(len(words), i+6)]
            contexts.append({
                'left': ' '.join(left),
                'term': word,
                'right': ' '.join(right),
                'position': i
            })

    return jsonify(contexts[:100])  # Limit to 100 contexts

@app.route('/api/collocates')
def get_collocates():
    """Get words that frequently co-occur with a term"""
    if not current_transcript['words']:
        return jsonify({})

    term = request.args.get('term', '').lower()
    if not term:
        return jsonify({})

    words = current_transcript['words']
    window_words = []

    for i, word in enumerate(words):
        if word == term:
            # Collect 5 words left and right
            left = words[max(0, i-5):i]
            right = words[i+1:min(len(words), i+6)]
            window_words.extend(left)
            window_words.extend(right)

    # Filter stopwords and exclude the term itself
    window_words = [w for w in window_words if w not in STOPWORDS and w != term and len(w) > 1]
    collocates = dict(Counter(window_words).most_common(20))

    return jsonify(collocates)

@app.route('/upload/<filename>')
def serve_upload(filename):
    """Serve uploaded video file"""
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    if filepath.exists():
        return send_file(filepath)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/segments')
def get_segments():
    """Get all segments (for subtitle sync)"""
    return jsonify(current_transcript['segments'])

@app.route('/api/reader')
def get_reader():
    """Get paginated text for the Reader panel"""
    if not current_transcript['segments']:
        return jsonify({'text': '', 'total_chars': 0, 'page': 1, 'total_pages': 1})

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 1500, type=int)

    full_text = current_transcript['full_text']
    total_chars = len(full_text)
    total_pages = max(1, math.ceil(total_chars / per_page))
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    chunk = full_text[start:end]

    return jsonify({
        'text': chunk,
        'total_chars': total_chars,
        'page': page,
        'total_pages': total_pages
    })

def extract_text_columns(df):
    """Return combined text from columns that contain meaningful text (not numbers/dates)."""
    text_parts = []
    for col in df.columns:
        col_vals = df[col].dropna()
        if len(col_vals) == 0:
            continue
        # Keep only string values with meaningful length
        str_vals = [str(v) for v in col_vals if isinstance(v, str) and len(v.strip()) > 5]
        if not str_vals:
            continue
        # Use this column if >30% of non-null values are meaningful strings
        ratio = len(str_vals) / len(col_vals)
        avg_len = sum(len(v) for v in str_vals) / len(str_vals)
        if ratio > 0.3 and avg_len > 8:
            text_parts.extend(str_vals)
    return ' '.join(text_parts)

@app.route('/api/upload-multi', methods=['POST'])
def upload_multi():
    """Handle multiple Excel file uploads, combining their text content."""
    global current_transcript

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'ok': False, 'error': 'No files provided'}), 400

    combined_text = ""
    filenames = []

    for file in files:
        if not file.filename:
            continue
        filename = secure_filename(file.filename)
        ext = Path(filename).suffix.lower()
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)

        if ext in ('.xlsx', '.xls'):
            try:
                xls = pd.ExcelFile(filepath)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    sheet_text = extract_text_columns(df)
                    if sheet_text.strip():
                        combined_text += sheet_text + " "
                filenames.append(filename)
            except Exception as e:
                return jsonify({'ok': False, 'error': f'Error reading {filename}: {str(e)}'}), 500
        else:
            return jsonify({'ok': False, 'error': f'Unsupported type: {ext}'}), 400

    if not combined_text.strip():
        return jsonify({'ok': False, 'error': 'No text found in uploaded files'}), 400

    sentences = nltk.sent_tokenize(combined_text)
    segments = []
    time = 0
    for sent in sentences:
        word_count = len(sent.split())
        duration = max(word_count * 0.5, 1)
        segments.append({'start': time, 'end': time + duration, 'text': sent})
        time += duration
    load_transcript(segments)
    current_transcript['filename'] = ', '.join(filenames)

    return jsonify({'ok': True, 'type': 'excel', 'filenames': filenames})

@app.route('/api/network')
def get_network():
    """Build a co-occurrence network graph from tokenized words.
    Query params:
        limit  – max number of top nodes to include (default 30)
        window – sliding window size for co-occurrence (default 5)
    Returns JSON with 'nodes' and 'edges'.
    """
    if not current_transcript['words']:
        return jsonify({'nodes': [], 'edges': []})

    limit = request.args.get('limit', 30, type=int)
    window = request.args.get('window', 5, type=int)

    words = current_transcript['words']
    word_freq = current_transcript['word_freq']

    # Pick the top N words to keep the graph readable
    top_words = set(list(word_freq.keys())[:limit])

    # Build co-occurrence counts using a sliding window
    cooc = Counter()
    for i in range(len(words)):
        if words[i] not in top_words:
            continue
        for j in range(i + 1, min(i + window + 1, len(words))):
            if words[j] not in top_words and words[j] != words[i]:
                continue
            if words[j] == words[i]:
                continue
            pair = tuple(sorted([words[i], words[j]]))
            cooc[pair] += 1

    # Build node list (only nodes that appear in at least one edge)
    node_set = set()
    edges = []
    for (w1, w2), weight in cooc.most_common(limit * 3):
        if weight < 2:
            continue
        node_set.add(w1)
        node_set.add(w2)
        edges.append({'source': w1, 'target': w2, 'weight': weight})

    nodes = []
    for w in node_set:
        nodes.append({'id': w, 'freq': word_freq.get(w, 1)})

    return jsonify({'nodes': nodes, 'edges': edges})


@app.route('/status')
def transcription_status():
    """Get transcription status"""
    return jsonify({
        'status': current_transcript.get('status', 'idle'),
        'error': current_transcript.get('error'),
        'has_data': bool(current_transcript.get('full_text', ''))
    })


# ══════════════════════════════════════════════════════════════════════════════
# TIME ANALYSIS — DATA_time & 50_analysis Excel routes
# ══════════════════════════════════════════════════════════════════════════════
F_TIME     = Path(r"C:\Users\Dell\Downloads\DATA_time (1) (1).xlsx")
F_ANALYSIS = Path(r"C:\Users\Dell\Downloads\50_analysis (2).xlsx")

BLOOM_MAP = {
    'Q1': 'Q1: Санах',    'Q2': 'Q2: Ойлгох',  'Q3': 'Q3: Хэрэглэх',
    'Q4': 'Q4: Шинжлэх', 'Q5': 'Q5: Үнэлэх',  'Q6': 'Q6: Бүтээх',
    'QL': 'QL: Доод түвшний', 'QO': 'QO: Нээлттэй', 'QC': 'QC: Хаалттай',
}
BLOOM_COLORS = {
    'Q1': '#3498db', 'Q2': '#2ecc71', 'Q3': '#e67e22',
    'Q4': '#9b59b6', 'Q5': '#e74c3c', 'Q6': '#1abc9c',
    'QL': '#95a5a6', 'QO': '#f39c12', 'QC': '#d35400',
}
SUPPORT_WORDS = [
    'маш сайн',      # very good — unambiguous praise
    'зөв байна',     # that's correct
    'маш зөв',       # very correct
    'яг зөв',        # exactly right
    'яг тийм',       # exactly so
    'зүйтэй',        # appropriate / right
    'болж байна',    # going well / that works
    'сайн байна',    # good
    'зөв ойлгосон',  # understood correctly
    'ойлгосон',      # understood
    'сайхан байна',  # how nice
    'баярлалаа',     # thank you (teacher acknowledging answer)
    'гоё байна',     # great / nice
    'тийм л дээ',    # yes indeed
    'тийм шүү',      # yes indeed
    'үнэн',          # true / correct
    'оновчтой',      # accurate / apt
]
# NOTE: 'зөв' alone removed — too broad (catches "зөв суугаарай" = sit properly).
#       'тийм ээ' removed — also appears inside teacher questions, not just praise.
#       'болсон' removed — "болсон уу?" is a question; ambiguous without context.
#       'сайхан' removed — can be used in non-academic contexts.
Q_TYPES = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'QL', 'QO', 'QC']


def _parse_time_sec(t):
    """Parse MM:SS or MM:SS:cs (not HH:MM:SS) — classroom timestamps are in minutes."""
    s = str(t).strip()
    if s in {'nan', 'NaN', '', 'Цаг', 'минут',
             'Ярьж эхэлсэн хугацаа', 'Ярьж дууссан хугацаа'} or s.startswith('Unnamed'):
        return None
    s = s.rstrip('m').strip()
    parts = s.split(':')
    try:
        if len(parts) == 3:
            # MM:SS:cs  — minutes : seconds : centiseconds
            return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 100
        if len(parts) == 2:
            # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return None


def _extract_q(text):
    return re.findall(r'Q[1-6]|QL|QO|QC', str(text))


def _classify_support(text):
    t = str(text).lower()
    return 'Дэмжсэн' if any(w in t for w in SUPPORT_WORDS) else 'Дэмжээгүй'


def _load_time_rows(sheet_name):
    df_raw = pd.read_excel(F_TIME, sheet_name=sheet_name, header=None)
    rows = []
    for _, row in df_raw.iterrows():
        spk = str(row[2]).strip()
        if spk not in ('T', 'S', 'C'):
            continue
        t0 = _parse_time_sec(row[0])
        t1 = _parse_time_sec(row[1])
        raw_dur = (t1 - t0) if (t0 is not None and t1 is not None) else None
        dur = raw_dur if (raw_dur is not None and 0 < raw_dur <= 3600) else None
        text = str(row[4]) if pd.notna(row[4]) else ''
        qc = _extract_q(text)
        rows.append({'speaker': spk, 'start': t0, 'end': t1,
                     'duration': dur, 'text': text, 'q_codes': qc, 'has_q': bool(qc)})
    return rows


@app.route('/time-analysis')
def time_analysis_page():
    return render_template('time_analysis.html')


@app.route('/api/time/sheets')
def api_time_sheets():
    try:
        return jsonify(pd.ExcelFile(F_TIME).sheet_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time/summary')
def api_time_summary():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_time_rows(sheet)
        def valid_dur(d): return d is not None and 0 < d <= 3600
        t_dur = sum(r['duration'] for r in rows if r['speaker'] == 'T' and valid_dur(r['duration']))
        s_dur = sum(r['duration'] for r in rows if r['speaker'] in ('S', 'C') and valid_dur(r['duration']))
        all_q = [q for r in rows for q in r['q_codes']]
        seg_size = max(1, len(rows) // 10)
        seg_q = []
        for i in range(10):
            chunk = rows[i * seg_size:(i + 1) * seg_size if i < 9 else len(rows)]
            cnt = Counter(q for r in chunk for q in r['q_codes'])
            seg_q.append({qt: cnt.get(qt, 0) for qt in Q_TYPES})
        return jsonify({
            'teacher_time': round(t_dur, 1),
            'student_time': round(s_dur, 1),
            'total_turns': len(rows),
            'q_counts': dict(Counter(all_q)),
            'seg_q': seg_q,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time/segments')
def api_time_segments():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_time_rows(sheet)
        return jsonify([{
            'speaker': r['speaker'],
            'start': r['start'],
            'end': r['end'],
            'duration': round(r['duration'], 1) if r['duration'] else None,
            'q_codes': r['q_codes'],
            'text': r['text'][:100],
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time/teacher-time')
def api_teacher_time():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_time_rows(sheet)
        records = []
        for i, row in enumerate(rows):
            if not (row['speaker'] == 'T' and row['has_q']):
                continue
            j = i + 1
            while j < len(rows) and rows[j]['speaker'] == 'T':
                j += 1
            if j >= len(rows) or rows[j]['speaker'] not in ('S', 'C'):
                continue
            s_row = rows[j]
            k = j + 1
            while k < len(rows) and rows[k]['speaker'] in ('S', 'C'):
                k += 1
            think = None
            if k < len(rows) and rows[k]['speaker'] == 'T':
                t2 = rows[k]
                if s_row['end'] is not None and t2['start'] is not None:
                    gap = t2['start'] - s_row['end']
                    think = round(gap, 1) if gap >= 0 else None
            t_ask = row['duration']
            s_ans = s_row['duration']
            # Skip records with clearly invalid (negative or huge) durations
            if t_ask is not None and t_ask < 0: t_ask = None
            if s_ans is not None and s_ans < 0: s_ans = None
            if t_ask is not None and t_ask > 3600: t_ask = None  # >1 hour impossible
            if s_ans is not None and s_ans > 3600: s_ans = None
            records.append({
                'q_num': i + 1,
                'q_codes': ', '.join(row['q_codes']),
                'teacher_ask': round(t_ask, 1) if t_ask is not None else None,
                'student_ans': round(s_ans, 1) if s_ans is not None else None,
                'teacher_think': think,
                'text': row['text'][:80],
            })
        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time/support')
def api_time_support():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_time_rows(sheet)
        results = []
        for i, row in enumerate(rows):
            if not (row['speaker'] == 'T' and row['has_q']):
                continue
            j = i + 1
            while j < len(rows) and rows[j]['speaker'] == 'T':
                j += 1
            if j >= len(rows) or rows[j]['speaker'] not in ('S', 'C'):
                continue
            s_row = rows[j]
            k = j + 1
            while k < len(rows) and rows[k]['speaker'] in ('S', 'C'):
                k += 1
            if k >= len(rows) or rows[k]['speaker'] != 'T':
                continue
            t2 = rows[k]
            results.append({
                'q_num': i + 1,
                'q_codes': ', '.join(row['q_codes']),
                'teacher_q': row['text'][:80],
                'student_ans': s_row['text'][:80],
                'teacher_followup': t2['text'][:80],
                'support': _classify_support(t2['text']),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time/bloom')
def api_bloom():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_time_rows(sheet)
        all_q = [q for r in rows if r['speaker'] == 'T' for q in r['q_codes']]
        cnt = Counter(all_q)
        return jsonify([{'code': k, 'label': BLOOM_MAP.get(k, k), 'count': v,
                         'color': BLOOM_COLORS.get(k, '#999')}
                        for k, v in sorted(cnt.items(), key=lambda x: -x[1])])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── 50_analysis helpers ───────────────────────────────────────────────────────
# Col4 codes: 1 = closed / lower-order, 2 = open / higher-order
ANALYSIS_Q_LABEL = {'1': 'QL: Хаалттай/Доод', '2': 'QO: Нээлттэй/Дээд'}
ANALYSIS_Q_COLOR = {'1': '#3498db', '2': '#e67e22'}


def _find_analysis_header(df):
    """Return (header_row_idx, speaker_col, text_col, q_col)"""
    for i, row in df.iterrows():
        vals = [str(v).strip() for v in row.tolist()]
        if any(v in ('Хэлсэн хүн', 'Хэлсэн хүн ') for v in vals):
            try:
                sc = next(j for j, v in enumerate(vals) if v.strip() in ('Хэлсэн хүн', 'Хэлсэн хүн '))
                tc = sc + 1
                qc = sc + 2
                return i, sc, tc, qc
            except StopIteration:
                pass
        # Fallback: row contains Багш/Сурагч directly
        if any(v in ('Багш', 'Сурагч', 'Сурагчид') for v in vals):
            sc = next(j for j, v in enumerate(vals) if v in ('Багш', 'Сурагч', 'Сурагчид'))
            return i - 1, sc, sc + 1, sc + 2
    return None, 1, 2, 3  # default fallback


def _load_analysis_rows(sheet_name):
    df_raw = pd.read_excel(F_ANALYSIS, sheet_name=sheet_name, header=None)
    header_row, sc, tc, qc = _find_analysis_header(df_raw)
    rows = []
    for i, row in df_raw.iterrows():
        if header_row is not None and i <= header_row:
            continue
        try:
            spk_raw = str(row.iloc[sc]).strip() if sc < len(row) else ''
        except Exception:
            continue
        if spk_raw not in ('Багш', 'Сурагч', 'Сурагчид', 'T', 'S', 'C'):
            continue
        spk = 'T' if spk_raw == 'Багш' else 'S'
        text = str(row.iloc[tc]).strip() if tc < len(row) and pd.notna(row.iloc[tc]) else ''
        q_raw = str(row.iloc[qc]).strip() if qc < len(row) and pd.notna(row.iloc[qc]) else ''
        q_codes = []
        try:
            v = int(float(q_raw))
            if v in (1, 2):
                q_codes = [str(v)]
        except Exception:
            pass
        # Also extract Q codes embedded in text (some sheets)
        q_codes += _extract_q(text)
        q_codes = list(dict.fromkeys(q_codes))  # dedupe, preserve order
        rows.append({
            'speaker': spk, 'start': None, 'end': None, 'duration': None,
            'text': text, 'q_codes': q_codes, 'has_q': bool(q_codes),
        })
    return rows


@app.route('/api/analysis/sheets')
def api_analysis_sheets():
    try:
        return jsonify(pd.ExcelFile(F_ANALYSIS).sheet_names)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/segments')
def api_analysis_segments():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_analysis_rows(sheet)
        return jsonify([{
            'speaker': r['speaker'], 'start': None, 'end': None,
            'duration': None, 'q_codes': r['q_codes'], 'text': r['text'][:100],
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/summary')
def api_analysis_summary():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_analysis_rows(sheet)
        all_q = [q for r in rows for q in r['q_codes']]
        seg_size = max(1, len(rows) // 10)
        seg_q = []
        for i in range(10):
            chunk = rows[i * seg_size:(i + 1) * seg_size if i < 9 else len(rows)]
            cnt = Counter(q for r in chunk for q in r['q_codes'])
            seg_q.append({qt: cnt.get(qt, 0) for qt in ['1', '2'] + Q_TYPES})
        return jsonify({
            'teacher_time': None, 'student_time': None,
            'total_turns': len(rows),
            'q_counts': dict(Counter(all_q)),
            'seg_q': seg_q,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/support')
def api_analysis_support():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_analysis_rows(sheet)
        results = []
        for i, row in enumerate(rows):
            if not (row['speaker'] == 'T' and row['has_q']):
                continue
            j = i + 1
            while j < len(rows) and rows[j]['speaker'] == 'T':
                j += 1
            if j >= len(rows) or rows[j]['speaker'] != 'S':
                continue
            s_row = rows[j]
            k = j + 1
            while k < len(rows) and rows[k]['speaker'] == 'S':
                k += 1
            if k >= len(rows) or rows[k]['speaker'] != 'T':
                continue
            t2 = rows[k]
            results.append({
                'q_num': i + 1,
                'q_codes': ', '.join(row['q_codes']),
                'teacher_q': row['text'][:80],
                'student_ans': s_row['text'][:80],
                'teacher_followup': t2['text'][:80],
                'support': _classify_support(t2['text']),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/bloom')
def api_analysis_bloom():
    sheet = request.args.get('sheet', '')
    try:
        rows = _load_analysis_rows(sheet)
        all_q = [q for r in rows if r['speaker'] == 'T' for q in r['q_codes']]
        cnt = Counter(all_q)
        result = []
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            label = ANALYSIS_Q_LABEL.get(k, BLOOM_MAP.get(k, k))
            color = ANALYSIS_Q_COLOR.get(k, BLOOM_COLORS.get(k, '#999'))
            result.append({'code': k, 'label': label, 'count': v, 'color': color})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
