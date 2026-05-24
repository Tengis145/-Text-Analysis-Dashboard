================================================================
  ХИЧЭЭЛИЙН ХАРИЛЦААНЫ ШИНЖИЛГЭЭНИЙ ВЕБ АПП
  Монгол ангийн харилцаа дүн шинжилгээний Flask програм
================================================================

Файл: DATA_time (байгалийн ухаан, математик).xlsx
       50_analysis.xlsx
Серверийн хаяг: http://127.0.0.1:8888/time-analysis


================================================================
1. СЕГМЕНТИЙН ХУВААЛТ
================================================================

Зорилго:
  Хичээлийн харилцан ярианы дугаараар (col 3) бүх мөрийг
  3 тэнцүү хэсэгт хуваан, тус бүрд нь Багш (T), Сурагч (S),
  Анги (C) хэдэн удаа ярьсныг тоолж, grouped bar chart-аар
  харуулна.

---------- app.py (backend) ----------

def _load_time_rows(sheet_name):
    df_raw = pd.read_excel(F_TIME, sheet_name=sheet_name, header=None)
    rows = []
    for _, row in df_raw.iterrows():
        spk = str(row[2]).strip()          # багана 2: T / S / C
        if spk not in ('T', 'S', 'C'):
            continue
        try:
            conv_num = int(float(str(row[3])))  # багана 3: харилцан ярианы дугаар
        except Exception:
            conv_num = None
        rows.append({
            'speaker':  spk,
            'conv_num': conv_num,
            ...
        })
    return rows


def api_time_summary():
    rows = _load_time_rows(sheet)
    conv_nums = [r['conv_num'] for r in rows if r['conv_num'] is not None]
    max_conv  = max(conv_nums)

    seg_speakers = []
    for i in range(3):
        lo = max_conv * i / 3
        hi = max_conv * (i + 1) / 3
        chunk = [r for r in rows if r['conv_num'] and lo < r['conv_num'] <= hi]

        spk_cnt = Counter(r['speaker'] for r in chunk)
        seg_speakers.append({
            'T':     spk_cnt.get('T', 0),
            'S':     spk_cnt.get('S', 0),
            'C':     spk_cnt.get('C', 0),
            'label': f'Сег {i+1} ({int(lo)+1}–{int(hi)})',
        })

Тайлбар:
  - max_conv = тухайн хуудасны хамгийн их харилцан ярианы дугаар
  - i = 0,1,2 → гурван ижил хэмжээний интервалд хуваана
  - Тус бүрд T/S/C тооллогыг Counter()-аар авна

---------- time_analysis.html (frontend) ----------

// /api/time/summary-аас seg_speakers массивыг авч Chart.js-ээр зурна
const segs = data.seg_speakers;

new Chart(document.getElementById('segChart'), {
    type: 'bar',
    data: {
        labels:   segs.map(s => s.label),   // "Сег 1 (1–20)" гэх мэт
        datasets: [
            { label: 'Багш (T)',   data: segs.map(s => s.T), backgroundColor: '#3498db' },
            { label: 'Сурагч (S)', data: segs.map(s => s.S), backgroundColor: '#e67e22' },
            { label: 'Анги (C)',   data: segs.map(s => s.C), backgroundColor: '#2ecc71' },
        ]
    },
    options: {
        plugins: { legend: { position: 'top' } },
        scales:  { x: { stacked: false }, y: { stacked: false } }
    }
});

API endpoint:  GET /api/time/summary?sheet=<хуудасны нэр>


================================================================
2. БАГШ / СУРАГЧИЙН ХУГАЦАА
================================================================

Зорилго:
  T→S→T гурвал (triplet) дараалал илрүүлж:
    - Багш асуух хугацаа  (T1 duration)
    - Сурагч хариулах хугацаа (S duration)
    - Багшийн бодох хугацаа (T2.start − S.end)
  Эдгээрийг секундээр тооцож, дэлгэрэнгүй хүснэгтэд харуулна.

---------- app.py (backend) ----------

def _parse_time_sec(t):
    """
    Цагийн форматыг секундад хөрвүүлнэ.
    Дэмжих форматууд:
      MM:SS         → стандарт
      MM:SS:cs      → 1/100 секундтэй
      MM:SS м       → Монгол дагавартай
      '1 day, H:MM:SS' → pandas timedelta (Excel [HH]:MM > 24 цаг үед)
    """
    s = str(t).strip()
    if s in {'nan', 'NaN', '', 'Цаг', 'минут'}:
        return None

    # pandas timedelta: '1 day, 1:45:00' → (1*24 + 1)*60 + 45 = 1545 мин
    td = re.match(r'^(\d+)\s+day[s]?,\s*(\d+):(\d+):(\d+)$', s)
    if td:
        days, h, mm, ss = int(td.group(1)), int(td.group(2)), \
                          int(td.group(3)), int(td.group(4))
        return (days * 24 + h) * 60 + mm + ss / 60

    s = re.sub(r'\s+', '', s)   # '24:11 м' → '24:11м'
    s = s.rstrip('мm').strip()  # 'м' дагавар хасна

    parts = s.split(':')
    try:
        if len(parts) == 3:      # MM:SS:cs
            return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 100
        if len(parts) == 2:      # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return None


def _build_triplets(rows):
    """
    T→S→T дараалал олж гурвал үүсгэнэ.
    Алгоритм:
      1. Q кодтой T мөрийг олно (T1)
      2. Дараагийн S/C мөрийг олно
      3. S-ийн дараагийн T мөрийг олно (T2)
    """
    triplets = []
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
        triplets.append({
            'q_num':  i + 1,
            'row':    row,    # T1
            's_row':  s_row,  # S
            'last_s': rows[k-1],
            't2':     t2,     # T2
        })
    return triplets


def api_teacher_time():
    triplets = _build_triplets(_load_time_rows(sheet))
    records = []
    for t in triplets:
        gap = t['t2']['start'] - t['last_s']['end']  # бодох хугацаа
        records.append({
            'q_num':          t['q_num'],
            'teacher_ask':    round(t['row']['duration'], 1),    # T1 сек
            'student_ans':    round(t['s_row']['duration'], 1),  # S сек
            'teacher_think':  round(gap, 1) if gap >= 0 else None,
            'text':           t['row']['text'][:80],
        })
    return jsonify(records)

---------- time_analysis.html (frontend) ----------

// Секундийг "X мин Y сек" гэж хөрвүүлнэ
function fmtDur(s) {
    if (s == null || isNaN(s)) return '—';
    const m   = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    if (m === 0) return sec + ' сек';
    return m + ' мин' + (sec ? ' ' + sec + ' сек' : '');
}

// Q кодыг текстээс хасна ("яагаад Q2" → "яагаад")
function stripQ(t) {
    return (t || '').replace(/[\s,]*(Q[1-6]|QL|QO|QC)\b[\s,]*/g, ' ')
                    .replace(/\s{2,}/g, ' ').trim();
}

// Metric card-уудад харуулна
document.getElementById('time-metrics').innerHTML = `
    <div class="metric">
        <div class="val">${fmtDur(tSec)}</div>
        <div class="lbl">Багшийн нийт</div>
    </div>
    <div class="metric">
        <div class="val">${fmtDur(sSec)}</div>
        <div class="lbl">Сурагчийн нийт</div>
    </div>`;

API endpoint:  GET /api/time/teacher-time?sheet=<хуудасны нэр>


================================================================
3. БАГШИЙН ДЭМЖЛЭГ
================================================================

Зорилго:
  T→S→T гурвалын T2 (багшийн дараах хариу үйлдэл)-ийг
  машин сургалтаар "Дэмжсэн" эсвэл "Дэмжээгүй" гэж ангилна.

Дэмжсэн гэдэг нь:
  Багш сурагчийн хариултыг магтаж/давтаж/баталж байгаа үе
  Жишээ: "Маш сайн хариулсан байна.", "Зөв байна. Тиймээ."

Дэмжээгүй гэдэг нь:
  Багш сурагчийн хариултыг хүлээн зөвшөөрөлгүй үргэлжлүүлж байгаа үе
  Жишээ: "Одоо дараагийн асуулт руу орцгооё.", "Буруу байна."

---------- ml_classify.py (машин сургалт) ----------

# Сургалтын мэдээлэл — бүтэн Монгол өгүүлбэрүүд
SUPPORT_POS = [   # Дэмжсэн жишээнүүд (35 өгүүлбэр)
    'Маш сайн хариулсан байна. Яг тийм байна.',
    'Зөв байна. Маш сайн.',
    'Тийм ээ, маш сайн байна.',
    'Гоё тайлбарласан байна. Баярлалаа.',
    'Зөв ойлгосон байна. Сайхан тайлбарлав.',
    'Амьд бие хооллодог гэсэн санаа. Тиймээ.',
    'Туулайг хурдан гэж ангилсан байна. Тийм ээ.',
    ...
]

SUPPORT_NEG = [   # Дэмжээгүй жишээнүүд (30 өгүүлбэр)
    'Одоо хичээлдээ анхаарна шүү хүүхдүүдээ.',
    'Зөв суугаарай. Анхаарна уу.',
    'Буруу байна. Дахиад бодно уу.',
    'Тэгвэл дараагийн асуулт руу орцгооё.',
    ...
]


class SupportClassifier:
    def __init__(self):
        X = [_preprocess(t) for t in SUPPORT_POS + SUPPORT_NEG]
        y = [1] * len(SUPPORT_POS) + [0] * len(SUPPORT_NEG)

        # TF-IDF char n-gram векторчлол
        self._vec = TfidfVectorizer(
            analyzer='char_wb',    # тэмдэгтийн n-gram
            ngram_range=(2, 4),    # 2–4 тэмдэгтийн хослол
            sublinear_tf=True,
            max_features=3000,
        )
        X_vec = self._vec.fit_transform(X)

        # Логистик регресс ангилагч
        self._model = LogisticRegression(
            C=1.0,
            class_weight='balanced',  # тэнцвэртэй жинтэй
            max_iter=1000,
            random_state=42
        )
        self._model.fit(X_vec, y)

    def predict(self, text):
        vec   = self._vec.transform([_preprocess(text)])
        proba = self._model.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        label = 'Дэмжсэн' if idx == 1 else 'Дэмжээгүй'
        return label, round(float(proba[idx]), 3)

Тайлбар:
  - char_wb: үгийн дотор болон хилийн дагуу тэмдэгт хэрчнэ
  - ngram_range=(2,4): 2-4 тэмдэгтийн бүх хослолыг онцлог болгоно
  - balanced: Дэмжсэн/Дэмжээгүй тоо ижил биш ч жингээр тэнцвэржүүлнэ
  - predict_proba: 0–1 магадлалаар буцаана

---------- app.py (backend) ----------

def _classify_support(text):
    # ml_classify.py-ийн SupportClassifier ашиглана
    label, _ = _ml.support_classify(text)
    return label   # 'Дэмжсэн' эсвэл 'Дэмжээгүй'


def api_time_support():
    result = []
    for t in _build_triplets(_load_time_rows(sheet)):
        _, conf = _ml.support_classify(t['t2']['text'])
        result.append({
            'q_num':            t['q_num'],
            'teacher_q':        t['row']['text'][:80],    # T1 асуулт
            'student_ans':      t['s_row']['text'][:80],  # S хариулт
            'teacher_followup': t['t2']['text'][:80],     # T2 дараах үйлдэл
            'support':          t['support'],              # 'Дэмжсэн'/'Дэмжээгүй'
            'confidence':       round(conf, 3),            # итгэлцэл
        })
    return jsonify(result)

---------- time_analysis.html (frontend) ----------

// Дэмжлэгийн badge-ийг өнгөөр ялгана
rows.forEach(r => {
    const badge = r.support === 'Дэмжсэн'
        ? `<span class="badge badge-D">Дэмжсэн</span>`
        : `<span class="badge badge-N">Дэмжээгүй</span>`;
    ...
});

// .badge-D → ногоон (#2ecc71)
// .badge-N → улаан (#e74c3c)

API endpoint:  GET /api/time/support?sheet=<хуудасны нэр>


================================================================
4. BLOOM-ЫН АНГИЛАЛ
================================================================

Зорилго:
  Багшийн бүх асуултыг Anderson & Krathwohl (2001)-ийн
  шинэчилсэн Блумын таксономийн 6 түвшинд ангилна.

  Q1 — Сэргээн санах  (Remember)  — нэрлэ, жагсаа, санана уу
  Q2 — Ойлгох         (Understand) — тайлбарла, яагаад, ялгаа нь
  Q3 — Хэрэглэх       (Apply)      — тооцоол, ямар аргаар, бодлого бод
  Q4 — Задлан шинжлэх (Analyze)    — задла, шинжил, учир шалтгаан
  Q5 — Үнэлэх         (Evaluate)   — үнэл, шалга, аль нь илүү
  Q6 — Бүтээх         (Create)     — бүтээ, зохион бүтээ, төлөвл

---------- ml_classify.py (машин сургалт) ----------

# Алхам 1: Keyword matching (үндсэн арга)
# Монгол үйл үгийн үндсийг текстэд хайна
BLOOM_KEYWORDS = {
    'Q1': ['нэрлэ', 'жагсаалт', 'жагсаа', 'тодорхойл',
           'ялгаж тан', 'сэргээн сана', 'байрлуул', 'ажигл',
           'санана уу', 'давтана уу', 'хэд байна', 'хэдэн байна'],

    'Q2': ['тайлбарл', 'яагаад', 'ялгаа нь', 'юугаараа ялгар',
           'ангил', 'бүлэгл', 'харьцуул', 'жишээ тат',
           'жишээ гарга', 'өөрийн үгээр', 'дүгнэлт', 'дүгн',
           'ойлгосон', 'ямар учир', 'юу гэсэн үг', 'ямар холбоо'],

    'Q3': ['хэрэгжүүл', 'хувирган', 'гүйцэтгэ', 'ашигл',
           'бодлого бод', 'тооцоол', 'ямар аргаар',
           'ямар томьёо', 'томьёогоор', 'шийдвэрл', 'дадлага'],

    'Q4': ['задл', 'холбон үз', 'тойм зурга', 'олж илрүүл',
           'бүтэцл', 'хэсгүүдэд', 'учир шалтгаан', 'шинжил',
           'нэгтгэ', 'зохион байгуул', 'нотло', 'хамаарлыг'],

    'Q5': ['шалга', 'таамагл', 'шүүмжил', 'баталгаажуул',
           'хяналт хий', 'үнэл', 'зөвтг', 'аль нь илүү',
           'зөв үү буруу', 'хэрхэн сайжруул', 'турши', 'эргэцүүл'],

    'Q6': ['бүтээ', 'зохион бүтээ', 'төлөвл', 'боловсруул',
           'санаачил', 'загвар гарга', 'шинэ зүйл',
           'санал гарга', 'найруул', 'програм бич'],
}


class BloomClassifier:
    def _kw_scores(self, text):
        # Текстийг жижиг үсгээр хөрвүүлж, тэмдэгт цэвэрлэнэ
        t = _preprocess(text)
        # Тус бүр түвшний keyword тоог тоолно
        return {cls: sum(1 for kw in BLOOM_KEYWORDS[cls] if kw in t)
                for cls in self._classes}

    def _scores(self, text):
        kw       = self._kw_scores(text)
        total_kw = sum(kw.values())

        if total_kw > 0:
            # Keyword олдсон → магадлал = тус бүрийн тоо / нийт тоо
            probs      = {c: kw[c] / total_kw for c in self._classes}
            best_count = max(kw.values())
            # Итгэлцэл = 0.45 + 0.15 × олдсон тоо (хамгийн их 0.92)
            confidence = min(0.45 + 0.15 * best_count, 0.92)
        else:
            # Keyword олдоогүй → TF-IDF cosine similarity fallback
            probs      = self._tfidf_probs(text)
            confidence = max(probs.values())

        return probs, round(confidence, 3)

    def _tfidf_probs(self, text):
        # Сургалтын жишээнүүдийн centroid-тай харьцуулна
        qvec = self._vec.transform([_preprocess(text)])
        raw  = {cls: float(cosine_similarity(qvec, c)[0][0])
                for cls, c in self._centroids.items()}
        # Линейн нормчлол (softmax биш — uniform болохоос сэргийлнэ)
        vals  = np.array([raw[c] for c in self._classes])
        s     = vals.sum()
        probs = vals / s if s > 0 else np.ones(len(vals)) / len(vals)
        return {c: round(float(p), 4) for c, p in zip(self._classes, probs)}

    def predict(self, text):
        probs, confidence = self._scores(text)
        best = max(probs, key=probs.get)
        return best, confidence  # ('Q2', 0.75) гэх мэт

Ангилалын жишээнүүд:
  "Яагаад ингэж ангилсан бэ?"   → Q2 Ойлгох         75%
  "Задлан шинжилнэ үү."         → Q4 Задлан шинжлэх  60%
  "Нэрлэнэ үү."                 → Q1 Сэргээн санах   60%
  "Ямар аргаар бодсон бэ?"      → Q3 Хэрэглэх        60%

---------- app.py (backend) ----------

def _bloom_ml_for_rows(rows):
    clf     = _ml._get_bloom()    # BloomClassifier singleton
    sup_clf = _ml._get_support()  # SupportClassifier singleton
    result  = []
    cnt     = Counter()

    for r in rows:
        if r['speaker'] != 'T' or not r['text'].strip():
            continue  # зөвхөн багшийн асуултыг ангилна

        level, conf = clf.predict(r['text'])    # Q1..Q6, итгэлцэл
        cnt[level] += 1

        support = None
        if sup_clf is not None:
            support, _ = sup_clf.predict(r['text'])  # Дэмжсэн/Дэмжээгүй

        result.append({
            'level':      level,
            'label':      BLOOM_LABELS[level],   # 'Q2: Ойлгох (Understand)'
            'confidence': round(conf, 3),
            'color':      BLOOM_COLORS[level],   # '#2ecc71' гэх мэт
            'conv_num':   r.get('conv_num'),      # харилцан ярианы дугаар
            'support':    support,
            'text':       r['text'][:100],
        })

    distribution = [
        {'code': k, 'label': BLOOM_LABELS[k], 'count': v, 'color': BLOOM_COLORS[k]}
        for k, v in sorted(cnt.items(), key=lambda x: -x[1])
    ]
    return {'distribution': distribution, 'rows': result}

---------- time_analysis.html (frontend) ----------

// Bloom дэлгэрэнгүй хүснэгт үүсгэнэ
tbodyEl.innerHTML = rows.map(r => {
    const supBadge = r.support === 'Дэмжсэн'
        ? `<span class="badge badge-D">Дэмжсэн</span>`
        : r.support === 'Дэмжээгүй'
          ? `<span class="badge badge-N">Дэмжээгүй</span>`
          : '—';

    return `<tr>
        <td style="color:#999">${r.conv_num ?? ''}</td>
        <td><span class="badge" style="background:${r.color}">${r.level}</span></td>
        <td>${r.label}</td>
        <td>${Math.round(r.confidence * 100)}%</td>
        <td>${supBadge}</td>
        <td>${r.text}</td>
    </tr>`;
}).join('');

API endpoint:  GET /api/time/bloom-ml?sheet=<хуудасны нэр>


================================================================
API ENDPOINTS
================================================================

  GET /api/time/summary?sheet=X         → нийт хугацаа, сегментийн T/S/C тоо
  GET /api/time/segments?sheet=X        → мөр бүрийн мэдээлэл + сегмент дугаар
  GET /api/time/teacher-time?sheet=X    → T→S→T гурвалын хугацаанууд
  GET /api/time/support?sheet=X         → ML дэмжлэгийн ангилал
  GET /api/time/bloom-ml?sheet=X        → ML Bloom ангилал + тархалт
  GET /api/time/question-chain?sheet=X  → асуултын сүлжээний node/link


================================================================
ТЕХНОЛОГИ
================================================================

  Backend:    Python 3, Flask, pandas, openpyxl
  ML:         scikit-learn (TfidfVectorizer, LogisticRegression, cosine_similarity)
  Frontend:   Chart.js v3 (grouped bar), D3.js v7 (force graph)
  Суулгах:    pip install flask pandas openpyxl scikit-learn numpy
              python app.py
