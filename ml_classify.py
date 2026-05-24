"""
ML-based classifiers for Mongolian classroom discourse analysis.
- BloomClassifier: keyword matching on official Anderson & Krathwohl (2001) taxonomy verbs
  with TF-IDF cosine similarity fallback
- SupportClassifier: logistic regression binary classifier
"""
import re
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

BLOOM_LABELS = {
    'Q1': 'Q1: Сэргээн санах (Remember)',
    'Q2': 'Q2: Ойлгох (Understand)',
    'Q3': 'Q3: Хэрэглэх (Apply)',
    'Q4': 'Q4: Задлан шинжлэх (Analyze)',
    'Q5': 'Q5: Үнэлэх (Evaluate)',
    'Q6': 'Q6: Бүтээх (Create)',
}

BLOOM_COLORS = {
    'Q1': '#3498db', 'Q2': '#2ecc71', 'Q3': '#e67e22',
    'Q4': '#9b59b6', 'Q5': '#e74c3c', 'Q6': '#1abc9c',
}

# ── Official taxonomy keyword roots ────────────────────────────────────────
# Source: Anderson & Krathwohl (2001) revised taxonomy, Mongolian adaptation
# Substring-matched against lowercased, punctuation-stripped text.
# Q2 / Q4 share "харьцуулах" — assigned Q2 since classroom baseline is understanding.
BLOOM_KEYWORDS = {
    'Q1': [  # Сэргээн санах — recall, name, list, identify, observe
        'нэрлэ',          # нэрлэх – name
        'жагсаалт',       # жагсаалт гаргах – list
        'жагсаа',         # жагсаах – list
        'тодорхойл',      # тодорхойлох – identify/define
        'ялгаж тан',      # ялгаж таних – recognize
        'сэргээн сана',   # сэргээн санах – recall
        'байрлуул',       # байрлуулах – locate/place
        'олж хар',        # олж харах – find and see
        'санана уу',      # remember? (recall prompt)
        'давтана уу',     # repeat/recall
        'нэрийг нь хэл',  # say the name of
        'ажигл',          # ажиглах – observe (олж харах family)
        'хэд байна',      # how many – factual counting/recall
        'хэдэн байна',    # how many
    ],
    'Q2': [  # Ойлгох — explain, classify, compare, summarize
        'тайлбарл',       # тайлбарлах – explain
        'яагаад',         # why – comprehension question word
        'ялгаа нь',       # what is the difference
        'юугаараа ялгар', # how does it differ
        'ангил',          # ангилах – classify
        'бүлэгл',         # бүлэглэх – group
        'харьцуул',       # харьцуулах – compare (Q2/Q4 overlap → Q2 default)
        'жишээ тат',      # жишээ татах – give example
        'жишээ гарга',    # жишээ гаргах – give example
        'өөрийн үгээр',   # in own words
        'дүгнэлт',        # дүгнэлт хийх – conclude
        'дүгн',           # дүгнэх – summarize
        'ойлгосон',       # did you understand
        'ямар учир',      # what reason
        'юу гэсэн үг',    # what does it mean
        'ямар холбоо',    # what connection
    ],
    'Q3': [  # Хэрэглэх — apply, implement, calculate, solve
        'хэрэгжүүл',      # хэрэгжүүлэх – implement
        'хувирган',       # хувирган хэрэглэх – apply/transform
        'гүйцэтгэ',       # гүйцэтгэх – execute
        'ашигл',          # ашиглах – use/apply
        'бодлого бод',    # solve a problem
        'тооцоол',        # тооцоолох – calculate
        'ямар аргаар',    # what method (applying a technique)
        'ямар томьёо',    # what formula
        'томьёогоор',     # using formula
        'шийдвэрл',       # шийдвэрлэх – solve/decide
        'дадлага',        # дадлага – practice
        'хэрэгл',         # хэрэглэх – use (broad apply)
    ],
    'Q4': [  # Задлан шинжлэх — analyze, break down, relate, structure
        'задл',           # задлах – break down/analyze
        'холбон үз',      # холбон үзэх – relate/connect
        'тойм зурга',     # тойм зургийг гаргах – outline/diagram
        'олж илрүүл',     # олж илрүүлэх – discover
        'бүтэцл',         # бүтэцлэх – structure
        'хэсгүүдэд',      # divide into parts
        'учир шалтгаан',  # cause-effect analysis
        'шинжил',         # шинжилэх – examine/analyze
        'нэгтгэ',         # нэгтгэх – integrate (Q4 per revised taxonomy)
        'зохион байгуул', # зохион байгуулах – organize/arrange
        'нотло',          # нотлох – prove
        'оюуны зураглал', # mind map
        'хамаарлыг',      # the relationship (analysis context)
    ],
    'Q5': [  # Үнэлэх — evaluate, judge, critique, validate
        'шалга',          # шалгах – check/verify
        'таамагл',        # таамаглах – hypothesize
        'шүүмжил',        # шүүмжлэх – critique
        'баталгаажуул',   # баталгаажуулах – validate
        'хяналт хий',     # хяналт хийх – monitor/control
        'нээн илрүүл',    # нээн илрүүлэх – discover/reveal
        'үнэл',           # үнэлэх – evaluate
        'зөвтг',          # зөвтгөх – justify
        'аль нь илүү',    # which is better
        'зөв үү буруу',   # right or wrong (judgment)
        'хэрхэн сайжруул',# how to improve
        'турши',          # турших – test
        'эргэцүүл',       # эргэцүүлэх – reflect
    ],
    'Q6': [  # Бүтээх — create, design, plan, produce
        'бүтээ',          # бүтээх – create (double-э distinguishes from бүтэцлэх)
        'зохион бүтээ',   # зохион бүтээх – design
        'төлөвл',         # төлөвлөх – plan
        'боловсруул',     # боловсруулах – develop/elaborate
        'санаачил',       # санаачлах – initiate/originate
        'загвар гарга',   # загвар гаргах – create a model
        'шинэ зүйл',      # new thing (creation context)
        'санал гарга',    # санал гаргах – propose
        'найруул',        # найруулах – compose/direct
        'програм бич',    # write program
    ],
}

# ── Bloom training examples — used only for TF-IDF fallback centroid ──────
BLOOM_TRAIN = {
    'Q1': [
        'Нэрлэнэ үү', 'Жагсаалт гаргана уу', 'Жагсааж бичнэ үү',
        'Тодорхойлно уу', 'Ялгаж таньна уу', 'Сэргээн санана уу',
        'Байрлуулна уу', 'Олно уу', 'Олж харна уу', 'Нэрийг нь хэлнэ үү',
        'Энэ юу вэ', 'Ямар нэртэй вэ', 'Хаана байдаг вэ', 'Хэн бэ',
        'Хэзээ болсон бэ', 'Юу юу байдаг вэ', 'Давтана уу', 'Санана уу',
        'Дарааллыг хэлнэ үү', 'Баримтыг нэрлэнэ үү', 'Тэмдэглэнэ үү',
        'Ямар зүйлс байдаг вэ', 'Жагсаасан байгаарай', 'Хэдэн байна вэ',
    ],
    'Q2': [
        'Тайлбарлана уу', 'Дүгнэнэ үү', 'Дүгнэлт хийнэ үү',
        'Өөрийн үгээр илэрхийлнэ үү', 'Өөрийн үгээр хэлнэ үү',
        'Ангилна уу', 'Бүлэглэнэ үү', 'Харьцуулна уу',
        'Жишээ татана уу', 'Жишээ гаргана уу', 'Тодотгож харуулна уу',
        'Ямар утгатай вэ', 'Яагаад гэж', 'Юу гэсэн үг вэ',
        'Хэрхэн ойлгосон бэ', 'Ялгаа нь юу вэ', 'Ижил тал нь юу вэ',
        'Ямар учиртай вэ', 'Хэрхэн болдог вэ', 'Ойлгосноо хэлнэ үү',
        'Зүйрлэж тайлбарлана уу', 'Агуулгыг ойлгосноо харуулна уу',
    ],
    'Q3': [
        'Хэрэгжүүлнэ үү', 'Хувирган хэрэглэнэ үү', 'Гүйцэтгэнэ үү',
        'Ашиглана уу', 'Хэрэглэнэ үү', 'Хэрхэн хэрэглэх вэ',
        'Тооцоолно уу', 'Шийдвэрлэнэ үү', 'Бодлого бодно уу',
        'Томьёогоор тооцоолно уу', 'Ямар томьёо ашиглах вэ',
        'Ямар аргаар бодох вэ', 'Туршиж үзнэ үү', 'Дадлага хийнэ үү',
        'Практикт ашиглана уу', 'Үр дүнг олно уу', 'Хэрхэн шийдэх вэ',
        'Тооцооллыг хийнэ үү', 'Биелүүлнэ үү',
    ],
    'Q4': [
        'Задлана уу', 'Холбон үзнэ үү', 'Тойм зургийг гаргана уу',
        'Олж илрүүлнэ үү', 'Бүтэцлэнэ үү', 'Нэгтгэнэ үү',
        'Зохион байгуулна уу', 'Шинжилнэ үү', 'Хэсгүүдэд хуваана уу',
        'Ямар хэсгүүдэд бүтсэн вэ', 'Хамаарал нь юу вэ',
        'Ямар шалтгаантай вэ', 'Судлана уу', 'Хэрхэн бүтсэн вэ',
        'Учир шалтгааныг олно уу', 'Нотлоно уу', 'Харьцуулан шинжилнэ үү',
        'Хамаарлыг тодорхойлно уу', 'Оюуны зураглал хийнэ үү',
    ],
    'Q5': [
        'Шалгана уу', 'Таамаглана уу', 'Шүүмжилнэ үү',
        'Туршина уу', 'Шинжлэнэ үү', 'Нээн илрүүлнэ үү',
        'Хяналт хийнэ үү', 'Баталгаажуулна уу', 'Хянаж үзнэ үү',
        'Эргэцүүлнэ үү', 'Үнэлнэ үү', 'Шүүмжлэл хийнэ үү',
        'Ямар байна гэж боддог вэ', 'Зөвтгөнө үү',
        'Хамгийн сайн нь аль вэ', 'Зөв үү буруу үү',
        'Аль нь илүү оновчтой вэ', 'Зөв гэж үзэх үндэслэл нь юу вэ',
        'Хэрхэн сайжруулах байсан бэ', 'Санал бодлоо хэлнэ үү',
    ],
    'Q6': [
        'Бүтээнэ үү', 'Зохионо уу', 'Төлөвлөнэ үү',
        'Боловсруулна уу', 'Санаачилна уу', 'Зохион бүтээнэ үү',
        'Төлөвлөгөө гаргана уу', 'Загвар гаргана уу', 'Шинэ зүйл зохионо уу',
        'Хэрхэн сайжруулах вэ', 'Ямар арга боловсруулах вэ',
        'Санал гаргана уу', 'Шинэ аргаар бүтээнэ үү',
        'Шинэ санаа оруулна уу', 'Загвар зохиона уу', 'Найруулна уу',
        'Шинэ зүйл гаргана уу', 'Програм бичнэ үү',
    ],
}

# ── Support classifier training data — full Mongolian classroom sentences ─────
# Дэмжсэн: teacher affirms/repeats/praises student's answer
# Дэмжээгүй: teacher gives instruction, moves on, or manages class without acknowledging
SUPPORT_POS = [
    # Explicit praise
    'Маш сайн хариулсан байна. Яг тийм байна.',
    'Зөв байна. Маш сайн.',
    'Тийм ээ, маш сайн байна.',
    'Гоё тайлбарласан байна. Баярлалаа.',
    'Яг зөв байна. Чи сайн бодсон байна.',
    'Маш сайн ажигласан байна. Тийм ээ.',
    'Зөв ойлгосон байна. Сайхан тайлбарлав.',
    'Оновчтой байна. Гоё.',
    'Тийм шүү. Сайн байна.',
    'Ухаалаг хариулт байна.',
    'Зөв зөв байна. Тиймээ.',
    'Гоё санасан байна.',
    'Үнэн байна. Сайн тайлбарлалаа.',
    'Чадварлаг хариулсан байна. Баярлалаа.',
    'Тийм л дээ. Зөв байна.',
    'Маш сайн байна баярлалаа.',
    'Болж байна. Яг тийм.',
    'Маш сайн сонсоно шүү.',
    'Та хоёр хөөрхөн ангилсан байсан шүү дээ.',
    'Маш олон янзаар ангилсан байна тиймээ.',
    # Repeat/paraphrase student's answer — supportive validation
    'Амьд бие хооллодог гэсэн санаа. Тиймээ.',
    'Туулайг хурдан гэж ангилсан байна. Тийм ээ.',
    'Хөдөлдөг учраас тиймээ. Зөв байна.',
    'Өсдөг өсдөггүй гэж наасан байна. Маш сайн.',
    'Амьтан гэж ангилж болох юм байна. Тиймээ.',
    'Ургамал гэж ангилж болох юм байна.',
    'Хүн амьтан ургамал гэж ангилсан байна. Тиймээ.',
    'Цэцэг шар гэж ангилсан байна. Зөв байна.',
    'Маш сайн байна. Бусад нь сонссон уу?',
    'Зөв санаа байна. Тиймээ. Маш сайн.',
    # Building on student answer
    'Зөв байна. Тэгвэл бусдыгаа сонсъё.',
    'Тиймээ. Нэмж хэлэх хүн байна уу?',
    'Сайн байна. Энэ санааг дэмждэг хэн байна?',
    'Зөв хариулсан байна. Тэгвэл дараагийн нь.',
    'Маш сайн. За бусдыгаа сонсъё.',
]

SUPPORT_NEG = [
    # Classroom management — no acknowledgment of student answer
    'Одоо хичээлдээ анхаарна шүү хүүхдүүдээ.',
    'Зөв суугаарай. Анхаарна уу.',
    'Урагшаа харна уу. Хосоороо ярилцаарай.',
    'Анхаарна уу. Бичнэ үү.',
    'Дуусгаарай. Хугацаа дуусах гэж байна.',
    'Сонс сонс. Анхааралтай сонсоорой.',
    'Бүгд анхаарна уу. Цааш явцгаая.',
    # Moving to next activity/student without acknowledging
    'Одоо дараагийн асуулт руу орцгооё.',
    'Тэгвэл дараагийн хүнийг сонсъё.',
    'За Сувдынхыг сонсъё. Дараагийн ээлжинд.',
    'Одоо бид дараагийн ажил руу орно.',
    'Цааш үргэлжлүүлцгээе. Тэгвэл дараагийн.',
    'За ерөнхийдөө ийм байна. Тэгвэл дараагийнх.',
    'Хэн хариулах вэ? Бусад нь мэдэх үү?',
    'Тэгвэл нэг минут байна шүү. Дуусгаарай.',
    # Instructions without acknowledging
    'Ажигласан зүйлээ ангилаад дэвтэртээ бичээрэй.',
    'Амьд биеэ дахиад ангилъя. Дотор нь ангилна.',
    'За багш нь даалгаврыг нь дахиад нэг хэлээд өгье.',
    'Одоо хамтдаа ажиллацгаая. Дуусгана уу.',
    'Уншина уу. Тэгээд хариулна уу.',
    'Тэгвэл эндээс харж байгаад тайлбарлаарай.',
    # Correction or redirection — no validation
    'Буруу байна. Дахиад бодно уу.',
    'Алдаатай байна. Дахин хийнэ үү.',
    'Тийм биш. Өөр хариулт байна уу?',
    'Дахиад нэг бодоод харна уу.',
    'Чанга хэлээрэй. Дахин хэлнэ үү.',
    'Ойлгосонгүй юм байна. Нэг дахин хэлнэ.',
    'Хэлж чадахгүй байна уу? Бусад нь.',
    'Нэг удаа давтана уу. Анхааралтай сонсоорой.',
    'Дахиад нэг удаа хийнэ үү.',
    'Тэгвэл өөр санаа байна уу?',
]


def _preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


class BloomClassifier:
    """Bloom's taxonomy classifier.

    Primary: keyword matching on official Anderson & Krathwohl (2001) Mongolian verb roots.
    Fallback: TF-IDF char n-gram cosine similarity to per-level training centroids.

    Confidence:
      keyword match  → 0.45 + 0.15 × match_count  (capped at 0.92)
      TF-IDF only    → proportional cosine score   (typically 0.15–0.25)
    """

    def __init__(self):
        self._classes = list(BLOOM_TRAIN.keys())
        all_texts = [_preprocess(e) for c in self._classes for e in BLOOM_TRAIN[c]]
        self._vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            sublinear_tf=True, max_features=8000,
        )
        self._vec.fit(all_texts)
        self._centroids = {}
        for cls in self._classes:
            vecs = self._vec.transform([_preprocess(e) for e in BLOOM_TRAIN[cls]])
            self._centroids[cls] = np.asarray(vecs.mean(axis=0))

    def _kw_scores(self, text):
        t = _preprocess(text)
        return {cls: sum(1 for kw in BLOOM_KEYWORDS[cls] if kw in t)
                for cls in self._classes}

    def _tfidf_probs(self, text):
        qvec = self._vec.transform([_preprocess(text)])
        raw = {cls: float(cosine_similarity(qvec, c)[0][0])
               for cls, c in self._centroids.items()}
        vals = np.array([raw[c] for c in self._classes])
        s = vals.sum()
        probs = vals / s if s > 0 else np.ones(len(vals)) / len(vals)
        return {c: round(float(p), 4) for c, p in zip(self._classes, probs)}

    def _scores(self, text):
        kw = self._kw_scores(text)
        total_kw = sum(kw.values())
        if total_kw > 0:
            probs = {c: kw[c] / total_kw for c in self._classes}
            best_count = max(kw.values())
            confidence = min(0.45 + 0.15 * best_count, 0.92)
        else:
            probs = self._tfidf_probs(text)
            confidence = max(probs.values())
        return probs, round(confidence, 3)

    def predict(self, text):
        probs, confidence = self._scores(text)
        best = max(probs, key=probs.get)
        return best, confidence

    def predict_all(self, text):
        probs, _ = self._scores(text)
        return {c: round(p, 4) for c, p in probs.items()}


class SupportClassifier:
    """Logistic regression classifier: teacher response supportive (1) or not (0)."""

    def __init__(self):
        X = [_preprocess(t) for t in SUPPORT_POS + SUPPORT_NEG]
        y = [1] * len(SUPPORT_POS) + [0] * len(SUPPORT_NEG)
        self._vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            sublinear_tf=True, max_features=3000,
        )
        X_vec = self._vec.fit_transform(X)
        self._model = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        )
        self._model.fit(X_vec, y)

    def predict(self, text):
        vec = self._vec.transform([_preprocess(text)])
        proba = self._model.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        label = 'Дэмжсэн' if idx == 1 else 'Дэмжээгүй'
        return label, round(float(proba[idx]), 3)


# ── Singletons ────────────────────────────────────────────────────────
_bloom_clf = None
_support_clf = None


def _get_bloom():
    global _bloom_clf
    if _bloom_clf is None and SKLEARN_OK:
        _bloom_clf = BloomClassifier()
    return _bloom_clf


def _get_support():
    global _support_clf
    if _support_clf is None and SKLEARN_OK:
        _support_clf = SupportClassifier()
    return _support_clf


def bloom_classify(text):
    """Return (level, confidence) — level is 'Q1'..'Q6'."""
    clf = _get_bloom()
    if clf is None:
        return 'Q1', 0.0
    return clf.predict(text)


def bloom_proba(text):
    """Return {level: probability} dict for all 6 levels."""
    clf = _get_bloom()
    if clf is None:
        return {c: 1/6 for c in BLOOM_TRAIN}
    return clf.predict_all(text)


def support_classify(text):
    """Return ('Дэмжсэн'/'Дэмжээгүй', confidence)."""
    clf = _get_support()
    if clf is None:
        return 'Дэмжээгүй', 0.5
    return clf.predict(text)
