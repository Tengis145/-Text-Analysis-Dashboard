import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .title-main {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-main'>Асуултын Анализ Дашборд</h1>", unsafe_allow_html=True)
st.markdown("**Voyant Tools шиг интерактив дата анализын платформ**")

# Load data
@st.cache_data
def load_excel_files():
    file1 = r"C:\Users\Dell\Downloads\50_analysis (2).xlsx"
    file2 = r"C:\Users\Dell\Downloads\DATA_time (1) (1).xlsx"

    data_dict = {}

    try:
        xls1 = pd.ExcelFile(file1)
        for sheet in xls1.sheet_names:
            if sheet.strip():  # Skip empty names
                df = pd.read_excel(file1, sheet_name=sheet)
                if not df.empty:
                    data_dict[f"{sheet}"] = df
    except Exception as e:
        st.error(f"Error loading file1: {e}")

    try:
        xls2 = pd.ExcelFile(file2)
        for sheet in xls2.sheet_names:
            if sheet.strip() and sheet != "1":  # Skip empty and "1"
                df = pd.read_excel(file2, sheet_name=sheet)
                if not df.empty:
                    data_dict[f"{sheet}"] = df
    except Exception as e:
        st.error(f"Error loading file2: {e}")

    return data_dict

data_dict = load_excel_files()

if not data_dict:
    st.error("Дата ачаалах боломжгүй байна!")
    st.stop()

st.success(f"{len(data_dict)} лист өгөгдөл ачаалсан")

# Sidebar navigation
st.sidebar.title("Навигац")
pages = ["Нийтлэг статистик", "Лист сонгох", "Дугаарын ангилал", "Сравнение", "Текст анализ"]
selected_page = st.sidebar.radio("Хуудас сонгоно уу:", pages)

# Category mapping for questions
def categorize_answer(value):
    """Categorize answers as 1, 2, or 3"""
    if pd.isna(value):
        return None

    value_str = str(value).strip().lower()

    # Category 1
    if value_str in ['1', 'a', 'yes', 'тийм', '1-р']:
        return 'Дугаар 1'

    # Category 2
    if value_str in ['2', 'b', 'no', 'үгүй', '2-р']:
        return 'Дугаар 2'

    # Category 3
    if value_str in ['3', 'c', 'maybe', 'магадгүй', '3-р']:
        return 'Дугаар 3'

    return 'Бусад'

# PAGE 1: Overview Statistics
if selected_page == "Нийтлэг статистик":
    st.header("Нийтлэг Статистик")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Нийтлэг листүүд", len(data_dict))
    with col2:
        total_rows = sum(len(df) for df in data_dict.values())
        st.metric("Нийтлэг мөрүүд", total_rows)
    with col3:
        total_cells = sum(len(df) * len(df.columns) for df in data_dict.values())
        st.metric("Нийтлэг ячейк", total_cells)
    with col4:
        st.metric("Баганууд", max(len(df.columns) for df in data_dict.values()))

    st.divider()

    # Sheet summary
    st.subheader("Лист сүргүүлэлт")
    sheet_info = []
    for sheet_name, df in data_dict.items():
        sheet_info.append({
            'Лист': sheet_name,
            'Мөрүүд': len(df),
            'Баганууд': len(df.columns)
        })

    df_summary = pd.DataFrame(sheet_info).sort_values('Мөрүүд', ascending=False)
    st.dataframe(df_summary, use_container_width=True)

    # Visualization
    fig = px.bar(df_summary, x='Лист', y='Мөрүүд',
                 title='Лист тус бүрийн мөрийн тоо',
                 labels={'Лист': 'Лист', 'Мөрүүд': 'Мөрийн тоо'})
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: Select and view sheet
elif selected_page == "Лист сонгох":
    st.header("Лист сонгох ба харах")

    selected_sheet = st.selectbox("Лист сонгоно уу:", list(data_dict.keys()))

    if selected_sheet:
        df = data_dict[selected_sheet]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Мөрүүд", len(df))
        with col2:
            st.metric("Баганууд", len(df.columns))
        with col3:
            st.metric("Хоосон утга", df.isna().sum().sum())

        st.divider()

        st.subheader("Өгөгдөл")
        st.dataframe(df, use_container_width=True, height=400)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="CSV сохих",
            data=csv,
            file_name=f"{selected_sheet}.csv",
            mime="text/csv"
        )

# PAGE 3: Category Analysis (1, 2, 3)
elif selected_page == "Дугаарын ангилал":
    st.header("Асуултуудыг Дугаар (1, 2, 3) ээр Ангилах")

    st.info("Дугаар 1, 2, 3-аар асуултуудыг ангилан анализлах хуудас")

    # Analyze all sheets
    category_counts = {'Дугаар 1': 0, 'Дугаар 2': 0, 'Дугаар 3': 0, 'Бусад': 0}
    all_categories = []
    sheet_categories = {}

    for sheet_name, df in data_dict.items():
        sheet_cat_counts = {'Дугаар 1': 0, 'Дугаар 2': 0, 'Дугаар 3': 0, 'Бусад': 0}

        for col in df.columns:
            for val in df[col]:
                cat = categorize_answer(val)
                if cat:
                    category_counts[cat] += 1
                    sheet_cat_counts[cat] += 1
                    all_categories.append(cat)

        sheet_categories[sheet_name] = sheet_cat_counts

    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Дугаар 1", category_counts['Дугаар 1'], delta=None)
    with col2:
        st.metric("Дугаар 2", category_counts['Дугаар 2'], delta=None)
    with col3:
        st.metric("Дугаар 3", category_counts['Дугаар 3'], delta=None)
    with col4:
        st.metric("Бусад", category_counts['Бусад'], delta=None)

    st.divider()

    # Pie chart
    fig_pie = px.pie(
        values=list(category_counts.values()),
        names=list(category_counts.keys()),
        title="Дугаарын ангилал",
        color_discrete_map={
            'Дугаар 1': '#1f77b4',
            'Дугаар 2': '#ff7f0e',
            'Дугаар 3': '#2ca02c',
            'Бусад': '#d62728'
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Sheet breakdown
    st.subheader("Лист тус бүрийн ангилал")

    df_categories = pd.DataFrame(sheet_categories).T
    st.dataframe(df_categories, use_container_width=True)

    # Stacked bar chart
    fig_stack = px.bar(
        df_categories.reset_index().rename(columns={'index': 'Лист'}),
        x='Лист',
        y=['Дугаар 1', 'Дугаар 2', 'Дугаар 3', 'Бусад'],
        title="Лист тус бүрийн дугаарын ангилал",
        barmode='stack'
    )
    st.plotly_chart(fig_stack, use_container_width=True)

# PAGE 4: Comparison
elif selected_page == "Сравнение":
    st.header("Листүүдийн Харьцуулалт")

    col1, col2 = st.columns(2)

    with col1:
        sheet1 = st.selectbox("Эхний лист:", list(data_dict.keys()), key='sheet1')

    with col2:
        sheet2 = st.selectbox("Хоёр дахь лист:", list(data_dict.keys()), key='sheet2',
                             index=min(1, len(data_dict)-1))

    if sheet1 and sheet2:
        df1 = data_dict[sheet1]
        df2 = data_dict[sheet2]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(f"{sheet1} - Мөрүүд", len(df1))
            st.metric(f"{sheet1} - Баганууд", len(df1.columns))

        with col2:
            st.metric(f"{sheet2} - Мөрүүд", len(df2))
            st.metric(f"{sheet2} - Баганууд", len(df2.columns))

        st.divider()

        # Comparison tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{sheet1}")
            st.dataframe(df1, use_container_width=True, height=300)

        with col2:
            st.subheader(f"{sheet2}")
            st.dataframe(df2, use_container_width=True, height=300)

# PAGE 5: Text Analysis (Voyant-like)
elif selected_page == "Текст анализ":
    st.header("Текст анализ (Word Frequency)")

    st.info("Бүх дата-аас үгийн давтамжийг анализлаж байна...")

    # Combine all text
    all_text = ""
    for sheet_name, df in data_dict.items():
        for col in df.columns:
            for val in df[col]:
                if pd.notna(val):
                    all_text += str(val) + " "

    # Simple word frequency (Mongolian aware)
    words = re.findall(r'\b[\w]+\b', all_text.lower())
    word_freq = Counter(words)

    # Filter common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                   'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'бөгөөд', 'нь', 'ба', 'эс', 'юм', 'байна', 'байсан'}

    filtered_freq = {k: v for k, v in word_freq.items() if k not in common_words and len(k) > 2}
    top_words = dict(sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:30])

    if top_words:
        col1, col2 = st.columns(2)

        with col1:
            # Word frequency table
            df_words = pd.DataFrame(list(top_words.items()), columns=['Үг', 'Давтамж'])
            st.subheader("Хамгийн олон дахин гарсан үгүүд")
            st.dataframe(df_words, use_container_width=True)

        with col2:
            # Word frequency bar chart
            fig_words = px.bar(
                x=list(top_words.keys()),
                y=list(top_words.values()),
                title="Үгийн давтамжийн топ 30",
                labels={'x': 'Үг', 'y': 'Давтамж'}
            )
            fig_words.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_words, use_container_width=True)

        # Word cloud alternative
        st.subheader("Үгийн үүл")
        fig_scatter = px.scatter(
            df_words.head(50),
            x='Үг',
            y='Давтамж',
            size='Давтамж',
            hover_name='Үг',
            title="Үгийн үүл (Өндөрлөг = давтамж)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Data Analysis Dashboard | Created: 2026 | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
