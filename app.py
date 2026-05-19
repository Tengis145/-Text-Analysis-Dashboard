from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
from collections import Counter
import re
from pathlib import Path
import io
from datetime import datetime
import time
import os
from werkzeug.utils import secure_filename
from config import ANALYSIS_FILE, TIME_FILE, UPLOAD_FOLDER, CACHE_TTL, DEBUG, PORT

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

data_dict = {}
time_dict = {}
word_counter = None
_cache = {}

def load_data():
    """Load all data from Excel files"""
    global data_dict, time_dict, word_counter

    data_dict = {}
    time_dict = {}

    # Load analysis file
    try:
        xls1 = pd.ExcelFile(ANALYSIS_FILE)
        for sheet in xls1.sheet_names:
            if sheet.strip():
                df = pd.read_excel(ANALYSIS_FILE, sheet_name=sheet)
                if not df.empty:
                    data_dict[f"{sheet}"] = df
    except Exception as e:
        print(f"Error loading analysis file: {e}")

    # Load time file
    try:
        xls2 = pd.ExcelFile(TIME_FILE, engine='openpyxl')
        for sheet in xls2.sheet_names:
            if sheet.strip() and sheet != "1":
                df = pd.read_excel(TIME_FILE, sheet_name=sheet, engine='openpyxl')
                if not df.empty:
                    time_dict[f"{sheet}"] = df
    except Exception as e:
        print(f"Error loading time file: {e}")

    # Pre-compute word frequencies
    compute_word_counter()

def compute_word_counter():
    """Pre-compute word frequency counter from all data"""
    global word_counter

    all_text = ""
    for sheet_name, df in list(data_dict.items()):
        for col in df.columns:
            for val in df[col]:
                if pd.notna(val):
                    all_text += str(val) + " "

    words = re.findall(r'\b[\w]+\b', all_text.lower())
    word_counter = Counter(words)

def get_cached(key, compute_fn):
    """Simple cache with TTL"""
    global _cache
    now = time.time()

    if key in _cache:
        cached_data, cached_time = _cache[key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    data = compute_fn()
    _cache[key] = (data, now)
    return data

def clear_cache():
    """Clear all cached data"""
    global _cache
    _cache = {}

def categorize_answer(value):
    """Categorize answers as 1, 2, or 3"""
    if pd.isna(value):
        return None

    value_str = str(value).strip().lower()

    if value_str in ['1', 'a', 'yes', 'тийм', '1-р']:
        return '1'
    if value_str in ['2', 'b', 'no', 'үгүй', '2-р']:
        return '2'
    if value_str in ['3', 'c', 'maybe', 'магадгүй', '3-р']:
        return '3'

    return 'other'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get overall statistics"""
    def compute():
        return {
            'total_sheets': len(data_dict),
            'total_rows': sum(len(df) for df in data_dict.values()),
            'total_cells': sum(len(df) * len(df.columns) for df in data_dict.values()),
            'max_columns': max(len(df.columns) for df in data_dict.values()) if data_dict else 0
        }

    return jsonify(get_cached('stats', compute))

@app.route('/api/sheets')
def get_sheets():
    """Get list of sheets"""
    sheets = []
    for sheet_name, df in list(data_dict.items()):
        sheets.append({
            'name': sheet_name,
            'rows': len(df),
            'columns': len(df.columns)
        })
    return jsonify(sorted(sheets, key=lambda x: x['rows'], reverse=True))

@app.route('/api/sheet/<sheet_name>')
def get_sheet(sheet_name):
    """Get sheet data with pagination"""
    if sheet_name not in data_dict:
        return jsonify({'error': 'Sheet not found'}), 404

    df = data_dict[sheet_name]
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 50, type=int)

    # Validate pagination params
    page = max(1, page)
    page_size = min(max(10, page_size), 500)  # Between 10 and 500

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    total_rows = len(df)
    data = df.iloc[start_idx:end_idx].to_dict('records')

    return jsonify({
        'name': sheet_name,
        'rows': len(df),
        'columns': len(df.columns),
        'page': page,
        'page_size': page_size,
        'total_rows': total_rows,
        'data': data
    })

@app.route('/api/categories')
def get_categories():
    """Get category analysis (1, 2, 3)"""
    def compute():
        category_counts = {'1': 0, '2': 0, '3': 0, 'other': 0}
        sheet_categories = {}

        for sheet_name, df in list(data_dict.items()):
            sheet_cat_counts = {'1': 0, '2': 0, '3': 0, 'other': 0}

            for col in df.columns:
                for val in df[col]:
                    cat = categorize_answer(val)
                    if cat:
                        category_counts[cat] += 1
                        sheet_cat_counts[cat] += 1

            sheet_categories[sheet_name] = sheet_cat_counts

        return {
            'overall': category_counts,
            'by_sheet': sheet_categories
        }

    return jsonify(get_cached('categories', compute))

@app.route('/api/text-analysis')
def get_text_analysis():
    """Get word frequency analysis from pre-computed counter or specific sheet"""
    sheet_name = request.args.get('sheet', '')
    
    current_counter = None
    if sheet_name and sheet_name in data_dict:
        # Compute counter specifically for this sheet
        df = data_dict[sheet_name]
        all_text = ""
        for col in df.columns:
            for val in df[col]:
                if pd.notna(val):
                    all_text += str(val) + " "
        words = re.findall(r'\b[\w]+\b', all_text.lower())
        current_counter = Counter(words)
    else:
        current_counter = word_counter

    if not current_counter:
        return jsonify({})

    search = request.args.get('search', '').lower()
    min_freq = int(request.args.get('min_freq', 1))

    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                   'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'бөгөөд', 'нь', 'ба', 'эс', 'юм', 'байна', 'байсан'}

    filtered_freq = {k: v for k, v in current_counter.items()
                    if k not in common_words and len(k) > 2 and v >= min_freq}

    if search:
        filtered_freq = {k: v for k, v in filtered_freq.items() if search in k}

    top_words = dict(sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:30])

    return jsonify(top_words)

@app.route('/api/export/csv')
def export_csv():
    """Export data as CSV"""
    sheet_name = request.args.get('sheet')

    if sheet_name and sheet_name in data_dict:
        df = data_dict[sheet_name]
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{sheet_name}.csv'
        )
    return jsonify({'error': 'Sheet not found'}), 404

@app.route('/api/export/all')
def export_all():
    """Export all data"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in list(data_dict.items()):
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )

@app.route('/api/comparison')
def get_comparison():
    """Compare two sheets"""
    sheet1 = request.args.get('sheet1')
    sheet2 = request.args.get('sheet2')

    if sheet1 not in data_dict or sheet2 not in data_dict:
        return jsonify({'error': 'Sheet not found'}), 404

    df1 = data_dict[sheet1]
    df2 = data_dict[sheet2]

    return jsonify({
        'sheet1': {
            'name': sheet1,
            'rows': len(df1),
            'cols': len(df1.columns),
            'memory': df1.memory_usage(deep=True).sum() / 1024
        },
        'sheet2': {
            'name': sheet2,
            'rows': len(df2),
            'cols': len(df2.columns),
            'memory': df2.memory_usage(deep=True).sum() / 1024
        }
    })

@app.route('/api/stats/detailed')
def get_detailed_stats():
    """Get detailed statistics"""
    def compute():
        stats = {
            'total_sheets': len(data_dict),
            'total_rows': sum(len(df) for df in data_dict.values()),
            'total_cols': sum(len(df.columns) for df in data_dict.values()),
            'sheets': []
        }

        for sheet_name, df in list(data_dict.items()):
            missing = df.isna().sum().sum()
            duplicates = df.duplicated().sum()

            stats['sheets'].append({
                'name': sheet_name,
                'rows': len(df),
                'cols': len(df.columns),
                'missing': int(missing),
                'duplicates': int(duplicates),
                'memory_kb': float(df.memory_usage(deep=True).sum() / 1024)
            })

        return stats

    return jsonify(get_cached('detailed_stats', compute))

def time_to_seconds(time_str):
    """Convert HH:MM:SS or HH:MMm format to seconds"""
    if pd.isna(time_str):
        return None

    time_str = str(time_str).strip()

    # Handle HH:MMm format (e.g., "00:58m")
    if time_str.endswith('m'):
        time_str = time_str[:-1]
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except:
            return None

    # Handle HH:MM:SS format
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except:
        return None

    return None

@app.route('/api/time-analysis')
def get_time_analysis():
    """Get time analysis from DATA_time file"""
    time_stats = {}

    # Process each sheet in time_dict (байгалийн ухаан, математик)
    for sheet_name, df in time_dict.items():
        if len(df) < 2:
            continue

        # Find time columns - usually first two columns
        time_col_start = None
        time_col_end = None

        # Look for "Цаг" column or similar
        for col in df.columns:
            if 'цаг' in str(col).lower():
                time_col_start = col
                break

        if time_col_start is None and len(df.columns) > 0:
            time_col_start = df.columns[0]
        if len(df.columns) > 1:
            time_col_end = df.columns[1]

        if time_col_start is None or time_col_end is None:
            continue

        # Calculate durations for each row
        durations = []
        valid_rows = 0

        for idx, row in df.iterrows():
            start_str = row.get(time_col_start)
            end_str = row.get(time_col_end)

            if pd.isna(start_str) or pd.isna(end_str):
                continue

            start_sec = time_to_seconds(start_str)
            end_sec = time_to_seconds(end_str)

            if start_sec is not None and end_sec is not None and end_sec >= start_sec:
                duration = end_sec - start_sec
                durations.append(duration)
                valid_rows += 1

        if len(durations) > 0:
            durations_arr = np.array(durations)
            time_stats[sheet_name] = {
                'min_seconds': float(durations_arr.min()),
                'max_seconds': float(durations_arr.max()),
                'avg_seconds': float(durations_arr.mean()),
                'median_seconds': float(np.median(durations_arr)),
                'std_seconds': float(durations_arr.std()),
                'total_seconds': float(durations_arr.sum()),
                'count': valid_rows
            }

    if not time_stats:
        return jsonify({
            'error': 'No time data could be extracted from DATA_time file.',
            'stats': {},
            'overall': {'total_time': 0, 'avg_time': 0, 'total_responses': 0}
        })

    # Calculate overall stats
    all_durations = []
    for sheet_data in time_stats.values():
        all_durations.append(sheet_data['total_seconds'])

    total_seconds = sum(all_durations)
    total_responses = sum(s['count'] for s in time_stats.values())
    avg_seconds = total_seconds / total_responses if total_responses > 0 else 0

    return jsonify({
        'error': None,
        'stats': time_stats,
        'overall': {
            'total_time': total_seconds,
            'avg_time': avg_seconds,
            'total_responses': total_responses
        }
    })

@app.route('/api/time-by-category')
def get_time_by_category():
    """Get average time by category (1, 2, 3)"""
    category_times = {'1': [], '2': [], '3': [], 'other': []}

    # Match sheets from both files
    for sheet_name in list(data_dict.keys()):
        if sheet_name in time_dict:
            df_answers = data_dict[sheet_name]
            df_times = time_dict[sheet_name]

            # Get time column
            time_col = None
            for col in df_times.columns:
                if 'time' in col.lower() or 'цаг' in col.lower() or 'минут' in col.lower():
                    time_col = col
                    break

            if time_col is None:
                for col in df_times.columns:
                    try:
                        pd.to_numeric(df_times[col], errors='coerce')
                        time_col = col
                        break
                    except:
                        pass

            if time_col:
                times = pd.to_numeric(df_times[time_col], errors='coerce')

                # Match answers with times
                for idx, (time_val, answer_col) in enumerate(zip(times, df_answers.columns)):
                    if pd.notna(time_val):
                        for answer_val in df_answers[answer_col]:
                            cat = categorize_answer(answer_val)
                            category_times[cat].append(float(time_val))

    # Calculate statistics per category
    result = {}
    for cat, times in category_times.items():
        if times:
            times_arr = np.array(times)
            result[cat] = {
                'avg': float(np.mean(times_arr)),
                'median': float(np.median(times_arr)),
                'min': float(np.min(times_arr)),
                'max': float(np.max(times_arr)),
                'count': len(times)
            }

    return jsonify(result)

@app.route('/api/speaker-time')
def get_speaker_time():
    """Get total speaking time for Teacher (T) and Student (S)"""
    import datetime as dt
    
    def parse_time(val):
        if pd.isna(val):
            return None
        if isinstance(val, dt.time):
            return val.hour * 60 + val.minute + (val.second / 60.0)
        if isinstance(val, str):
            val = val.strip().lower()
            if val.endswith('м') or val.endswith('m'):
                val = val[:-1]
            parts = val.split(':')
            if len(parts) >= 2:
                return int(parts[0]) * 60 + int(parts[1])
        return None

    s_time = 0
    t_time = 0
    timeline = []
    
    target_sheet = None
    if 'Sheet1' in time_dict:
        target_sheet = time_dict['Sheet1']
    elif '1' in time_dict:
        target_sheet = time_dict['1']
        
    if target_sheet is not None and 'Цаг' in target_sheet.columns and 'Ярьж буй хүн' in target_sheet.columns:
        df = target_sheet.copy()
        df['seconds'] = df['Цаг'].apply(parse_time)
        valid_df = df.dropna(subset=['seconds', 'Ярьж буй хүн']).copy()
        valid_df['Ярьж буй хүн'] = valid_df['Ярьж буй хүн'].astype(str).str.strip().str.upper()
        
        valid_df = valid_df.sort_values('seconds')
        valid_df['next_seconds'] = valid_df['seconds'].shift(-1)
        valid_df['duration'] = valid_df['next_seconds'] - valid_df['seconds']
        
        for _, row in valid_df.iterrows():
            if pd.notna(row['duration']):
                speaker = row['Ярьж буй хүн']
                dur = float(row['duration'])
                if speaker == 'S':
                    s_time += dur
                elif speaker == 'T':
                    t_time += dur
                timeline.append({'speaker': speaker, 'start': float(row['seconds']), 'duration': dur})
                
    return jsonify({
        's_time': float(s_time),
        't_time': float(t_time),
        'timeline': timeline
    })

@app.route('/api/reload', methods=['POST'])
def reload_data():
    """Reload all data from files"""
    try:
        load_data()
        clear_cache()
        return jsonify({
            'ok': True,
            'sheets_loaded': len(data_dict),
            'time_sheets_loaded': len(time_dict)
        })
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a new Excel file and merge sheets"""
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('type', 'analysis')

    if file.filename == '':
        return jsonify({'ok': False, 'error': 'No file selected'}), 400

    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'ok': False, 'error': 'Only Excel files are allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"

        filepath = UPLOAD_FOLDER / filename
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        file.save(filepath)

        # Load and merge sheets
        xls = pd.ExcelFile(filepath)
        sheets_loaded = 0

        if file_type == 'analysis':
            for sheet in xls.sheet_names:
                if sheet.strip():
                    df = pd.read_excel(filepath, sheet_name=sheet)
                    if not df.empty:
                        data_dict[sheet] = df
                        sheets_loaded += 1
        elif file_type == 'time':
            for sheet in xls.sheet_names:
                if sheet.strip() and sheet != "1":
                    df = pd.read_excel(filepath, sheet_name=sheet, engine='openpyxl')
                    if not df.empty:
                        time_dict[sheet] = df
                        sheets_loaded += 1

        # Recompute word counter with new data
        compute_word_counter()
        clear_cache()

        return jsonify({
            'ok': True,
            'sheets_loaded': sheets_loaded,
            'filename': filename
        })
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    load_data()
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
