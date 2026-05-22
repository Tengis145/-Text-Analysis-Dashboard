import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re

st.set_page_config(page_title="Хичээлийн харилцааны шинжилгээ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-box { background:#f0f4f8; border-radius:8px; padding:16px; margin:4px; text-align:center; }
.metric-val { font-size:2em; font-weight:700; color:#2c3e50; }
.metric-lbl { font-size:0.82em; color:#7f8c8d; margin-top:4px; }
h1 { color:#2c3e50; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
F1 = r"C:\Users\Dell\Downloads\DATA_time (1) (1).xlsx"

BLOOM_MAP = {
    'Q1': 'Q1: Санах (Remember)',
    'Q2': 'Q2: Ойлгох (Understand)',
    'Q3': 'Q3: Хэрэглэх (Apply)',
    'Q4': 'Q4: Шинжлэх (Analyze)',
    'Q5': 'Q5: Үнэлэх (Evaluate)',
    'Q6': 'Q6: Бүтээх (Create)',
    'QL': 'QL: Доод түвшний',
    'QO': 'QO: Нээлттэй асуулт',
    'QC': 'QC: Хаалттай асуулт',
}
BLOOM_COLORS = {
    'Q1': '#3498db', 'Q2': '#2ecc71', 'Q3': '#e67e22',
    'Q4': '#9b59b6', 'Q5': '#e74c3c', 'Q6': '#1abc9c',
    'QL': '#95a5a6', 'QO': '#f39c12', 'QC': '#d35400',
}
SUPPORT_WORDS = [
    'маш сайн', 'зөв байна', 'зүйтэй', 'болж байна', 'сайн байна',
    'маш зөв', 'яг зөв', 'яг тийм', 'зөв', 'тийм ээ', 'болсон',
    'ойлгосон', 'дурслалаа', 'сайхан', 'баярлалаа',
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_time(t):
    s = str(t).strip()
    skip = {'nan', 'NaN', '', 'Цаг', 'минут',
            'Ярьж эхэлсэн хугацаа', 'Ярьж дууссан хугацаа'}
    if s in skip or s.startswith('Unnamed'):
        return None
    s = s.rstrip('m').strip()
    parts = s.split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except Exception:
        return None

def extract_q(text):
    return re.findall(r'Q[1-6]|QL|QO|QC', str(text))

def fmt_sec(s):
    if s is None or np.isnan(s):
        return '—'
    m, sec = divmod(int(s), 60)
    return f'{m}:{sec:02d}'

def classify_support(text):
    t = str(text).lower()
    for w in SUPPORT_WORDS:
        if w in t:
            return 'Дэмжсэн'
    return 'Дэмжээгүй'

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_time_data():
    sheets = {}
    try:
        xls = pd.ExcelFile(F1)
    except Exception as e:
        st.error(f"Файл нээхэд алдаа: {e}")
        return {}

    for name in xls.sheet_names:
        df_raw = pd.read_excel(F1, sheet_name=name, header=None)
        rows = []
        for _, row in df_raw.iterrows():
            spk = str(row[2]).strip()
            if spk not in ('T', 'S', 'C'):
                continue
            t_start = parse_time(row[0])
            t_end   = parse_time(row[1])
            dur = (t_end - t_start) if (t_start is not None and t_end is not None) else None
            text = str(row[4]) if pd.notna(row[4]) else ''
            q_codes = extract_q(text)
            rows.append({
                'speaker': spk,
                'start':   t_start,
                'end':     t_end,
                'duration': dur,
                'text':    text,
                'q_codes': q_codes,
                'has_q':   len(q_codes) > 0,
            })
        sheets[name] = pd.DataFrame(rows)
    return sheets


time_data = load_time_data()
if not time_data:
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Навигац")
pages = [
    "Сегментийн хуваалт ба Q тоо",
    "Багш / Сурагчийн хугацаа",
    "Багшийн дэмжлэгийн шинжилгээ",
    "Bloom-ын ангилал",
]
page = st.sidebar.radio("Хуудас:", pages)
sheet_name = st.sidebar.selectbox("Хичээл (хуудас):", list(time_data.keys()))
df = time_data[sheet_name]

st.title("Хичээлийн харилцааны шинжилгээ")
st.caption(f"Сонгосон хичээл: **{sheet_name}** | Нийт ярилцлага: {len(df)} мөр")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Сегментийн хуваалт ба Q тоо
# ══════════════════════════════════════════════════════════════════════════════
if page == "Сегментийн хуваалт ба Q тоо":
    st.header("Сегментийн хуваалт ба Асуултын тоо")

    t_rows = df[df['speaker'] == 'T']
    s_rows = df[df['speaker'].isin(['S', 'C'])]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Нийт ярилцлага", len(df))
    c2.metric("Багшийн тээлт", len(t_rows))
    c3.metric("Сурагчийн тээлт", len(s_rows))
    q_total = sum(len(q) for q in df['q_codes'])
    c4.metric("Нийт Q код", q_total)

    st.divider()

    # Q code per segment (10 buckets)
    st.subheader("Q кодын тархалт — 10 сегментэд хуваасан")
    total = len(df)
    seg_size = max(1, total // 10)
    seg_labels = [f"Сег {i+1}" for i in range(10)]
    q_types = list(BLOOM_MAP.keys())
    seg_counts = {qt: [] for qt in q_types}

    for i in range(10):
        chunk = df.iloc[i*seg_size:(i+1)*seg_size if i < 9 else total]
        all_q = [q for codes in chunk['q_codes'] for q in codes]
        cnt = Counter(all_q)
        for qt in q_types:
            seg_counts[qt].append(cnt.get(qt, 0))

    fig_seg = go.Figure()
    for qt in q_types:
        if sum(seg_counts[qt]) > 0:
            fig_seg.add_trace(go.Bar(
                name=BLOOM_MAP[qt],
                x=seg_labels,
                y=seg_counts[qt],
                marker_color=BLOOM_COLORS.get(qt, '#999'),
            ))
    fig_seg.update_layout(barmode='stack', title='Сегмент тус бүрийн Q кодын тоо',
                          xaxis_title='Сегмент', yaxis_title='Q тоо', height=380)
    st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()

    # Table of segments
    st.subheader("Ярилцлагын жагсаалт")
    tbl = df[['speaker', 'start', 'end', 'duration', 'q_codes', 'text']].copy()
    tbl['start']    = tbl['start'].apply(lambda x: fmt_sec(x) if x is not None else '—')
    tbl['end']      = tbl['end'].apply(lambda x: fmt_sec(x) if x is not None else '—')
    tbl['duration'] = tbl['duration'].apply(lambda x: fmt_sec(x) if x is not None else '—')
    tbl['q_codes']  = tbl['q_codes'].apply(lambda x: ', '.join(x) if x else '')
    tbl['text']     = tbl['text'].str[:80]
    tbl.columns     = ['Ярьсан хүн', 'Эхлэл', 'Төгсгөл', 'Үргэлжлэл', 'Q код', 'Текст']
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=340)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Багш / Сурагчийн хугацаа
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Багш / Сурагчийн хугацаа":
    st.header("Багш / Сурагчийн ярих хугацааны шинжилгээ")

    def total_dur(spk_list):
        return df[df['speaker'].isin(spk_list)]['duration'].dropna().sum()

    t_dur = total_dur(['T'])
    s_dur = total_dur(['S', 'C'])

    c1, c2, c3 = st.columns(3)
    c1.metric("Багшийн нийт хугацаа", fmt_sec(t_dur))
    c2.metric("Сурагчийн нийт хугацаа", fmt_sec(s_dur))
    ratio = round(t_dur / s_dur, 2) if s_dur else 0
    c3.metric("Харьцаа (Б/С)", f"{ratio}x")

    st.divider()

    # Pie: time distribution
    pie_fig = px.pie(
        values=[t_dur, s_dur],
        names=['Багш', 'Сурагч / Анги'],
        title='Ярих хугацааны харьцаа',
        color_discrete_map={'Багш': '#3498db', 'Сурагч / Анги': '#e67e22'},
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    st.divider()
    st.subheader("Багшийн асуултын дараах сурагчийн хариулах хугацаа")
    st.caption("Багш Q код ашиглан асуусны дараа сурагч хариулах хугацаа ба багшийн боловсруулах хугацаа")

    records = []
    rows_list = df.reset_index(drop=True)
    for i, row in rows_list.iterrows():
        if row['speaker'] == 'T' and row['has_q']:
            # Find next student turn
            j = i + 1
            while j < len(rows_list) and rows_list.loc[j, 'speaker'] == 'T':
                j += 1
            if j < len(rows_list) and rows_list.loc[j, 'speaker'] in ('S', 'C'):
                s_row = rows_list.loc[j]
                s_dur_val = s_row['duration']
                # Teacher processing time: gap between S end and next T start
                k = j + 1
                while k < len(rows_list) and rows_list.loc[k, 'speaker'] in ('S', 'C'):
                    k += 1
                think_time = None
                if k < len(rows_list) and rows_list.loc[k, 'speaker'] == 'T':
                    t2 = rows_list.loc[k]
                    if s_row['end'] is not None and t2['start'] is not None:
                        gap = t2['start'] - s_row['end']
                        think_time = gap if gap >= 0 else None
                records.append({
                    'Асуулт #': i + 1,
                    'Q код': ', '.join(row['q_codes']),
                    'Багшийн асуулт (сек)': round(row['duration'], 1) if row['duration'] else None,
                    'Сурагчийн хариулт (сек)': round(s_dur_val, 1) if s_dur_val else None,
                    'Багшийн боловсруулах хугацаа (сек)': round(think_time, 1) if think_time is not None else None,
                    'Асуулт текст': row['text'][:60],
                })

    if records:
        df_rec = pd.DataFrame(records)

        c1, c2, c3 = st.columns(3)
        avg_s = df_rec['Сурагчийн хариулт (сек)'].dropna().mean()
        avg_t = df_rec['Багшийн боловсруулах хугацаа (сек)'].dropna().mean()
        c1.metric("Дундаж сурагчийн хариулт", fmt_sec(avg_s) if avg_s else '—')
        c2.metric("Дундаж багшийн боловсруулах", fmt_sec(avg_t) if avg_t else '—')
        c3.metric("Нийт асуулт-хариулт хос", len(df_rec))

        fig2 = go.Figure()
        x_vals = df_rec['Асуулт #'].tolist()
        fig2.add_trace(go.Bar(name='Сурагч хариулт', x=x_vals,
                              y=df_rec['Сурагчийн хариулт (сек)'], marker_color='#e67e22'))
        fig2.add_trace(go.Bar(name='Багш боловсруулах', x=x_vals,
                              y=df_rec['Багшийн боловсруулах хугацаа (сек)'], marker_color='#3498db'))
        fig2.update_layout(barmode='group', title='Асуулт тус бүрийн хугацаа (сек)',
                           xaxis_title='Асуулт дугаар', yaxis_title='Секунд', height=360)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_rec.drop(columns=['Асуулт текст']), use_container_width=True)
    else:
        st.info("Энэ хичээлд Q кодтой асуулт-хариулт хос олдсонгүй.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Багшийн дэмжлэгийн шинжилгээ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Багшийн дэмжлэгийн шинжилгээ":
    st.header("Багш сурагчийн хариултыг дэмжиж байна уу?")
    st.caption("Багш асуулт тавих → Сурагч хариулах → Багшийн дараагийн яриа дэмжлэг үзүүлж байгаа эсэхийг шинжилнэ")

    rows_list = df.reset_index(drop=True)
    results = []
    for i, row in rows_list.iterrows():
        if row['speaker'] == 'T' and row['has_q']:
            j = i + 1
            while j < len(rows_list) and rows_list.loc[j, 'speaker'] == 'T':
                j += 1
            if j >= len(rows_list) or rows_list.loc[j, 'speaker'] not in ('S', 'C'):
                continue
            s_row = rows_list.loc[j]
            k = j + 1
            while k < len(rows_list) and rows_list.loc[k, 'speaker'] in ('S', 'C'):
                k += 1
            if k >= len(rows_list) or rows_list.loc[k, 'speaker'] != 'T':
                continue
            t2_row = rows_list.loc[k]
            support = classify_support(t2_row['text'])
            results.append({
                'Асуулт #': i + 1,
                'Q код': ', '.join(row['q_codes']),
                'Багшийн асуулт': row['text'][:70],
                'Сурагчийн хариулт': s_row['text'][:70],
                'Багшийн дараагийн яриа': t2_row['text'][:70],
                'Дэмжлэг': support,
            })

    if results:
        df_sup = pd.DataFrame(results)
        counts = df_sup['Дэмжлэг'].value_counts()

        c1, c2, c3 = st.columns(3)
        demj = counts.get('Дэмжсэн', 0)
        ndemj = counts.get('Дэмжээгүй', 0)
        total_r = len(df_sup)
        c1.metric("Дэмжсэн", demj)
        c2.metric("Дэмжээгүй / Чиглүүлсэн", ndemj)
        c3.metric("Нийт Q тавьсан", total_r)

        pie = px.pie(
            values=counts.values,
            names=counts.index,
            title='Багшийн дэмжлэгийн харьцаа',
            color_discrete_map={'Дэмжсэн': '#2ecc71', 'Дэмжээгүй': '#e74c3c'},
        )
        st.plotly_chart(pie, use_container_width=True)

        st.divider()
        st.subheader("Q код тус бүрийн дэмжлэгийн тоо")
        q_sup = defaultdict(lambda: {'Дэмжсэн': 0, 'Дэмжээгүй': 0})
        for _, r in df_sup.iterrows():
            for qc in r['Q код'].split(', '):
                qc = qc.strip()
                if qc:
                    q_sup[qc][r['Дэмжлэг']] += 1
        q_sup_rows = [{'Q код': k, **v} for k, v in q_sup.items()]
        if q_sup_rows:
            df_qs = pd.DataFrame(q_sup_rows)
            fig_qs = px.bar(df_qs, x='Q код', y=['Дэмжсэн', 'Дэмжээгүй'],
                            title='Q код тус бүрийн дэмжлэг',
                            barmode='group',
                            color_discrete_map={'Дэмжсэн': '#2ecc71', 'Дэмжээгүй': '#e74c3c'})
            st.plotly_chart(fig_qs, use_container_width=True)

        st.divider()
        st.subheader("Дэлгэрэнгүй жагсаалт")
        st.dataframe(df_sup, use_container_width=True, height=360)
    else:
        st.info("Энэ хичээлд T→S→T хэлхээс олдсонгүй.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Bloom-ын ангилал
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bloom-ын ангилал":
    st.header("Bloom-ын таксономийн ангилал")
    st.caption("Багшийн асуултуудыг Q1–Q6 (Bloom), QL, QO, QC ангиллаар шинжилнэ")

    st.subheader(f"Сонгосон хичээл: {sheet_name}")
    t_df = df[df['speaker'] == 'T']
    all_q = [q for codes in t_df['q_codes'] for q in codes]
    cnt = Counter(all_q)

    if cnt:
        bloom_rows = [{'Q код': k, 'Нэр': BLOOM_MAP.get(k, k), 'Тоо': v}
                      for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
        df_bloom = pd.DataFrame(bloom_rows)

        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                df_bloom, values='Тоо', names='Нэр',
                title=f'{sheet_name} — Q кодын тархалт',
                color='Q код',
                color_discrete_map=BLOOM_COLORS,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_bar = px.bar(
                df_bloom, x='Q код', y='Тоо', color='Q код',
                color_discrete_map=BLOOM_COLORS,
                title='Q код тус бүрийн тоо',
                text='Тоо',
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(df_bloom, use_container_width=True)
    else:
        st.info("Энэ хичээлд Q код олдсонгүй.")

    st.divider()
    st.subheader("Бүх хичээлийн Bloom ангилалын харьцуулалт")

    all_bloom = []
    for sname, sdf in time_data.items():
        t_only = sdf[sdf['speaker'] == 'T']
        q_list = [q for codes in t_only['q_codes'] for q in codes]
        c = Counter(q_list)
        for qt, n in c.items():
            all_bloom.append({'Хичээл': sname, 'Q код': qt, 'Нэр': BLOOM_MAP.get(qt, qt), 'Тоо': n})

    if all_bloom:
        df_all = pd.DataFrame(all_bloom)
        fig_all = px.bar(
            df_all, x='Хичээл', y='Тоо', color='Q код',
            color_discrete_map=BLOOM_COLORS,
            title='Хичээл тус бүрийн Bloom Q ангилал',
            barmode='stack',
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # Pivot table
        pivot = df_all.pivot_table(index='Хичээл', columns='Q код', values='Тоо', fill_value=0)
        st.dataframe(pivot, use_container_width=True)
