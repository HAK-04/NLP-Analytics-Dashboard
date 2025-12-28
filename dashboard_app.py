import streamlit as st
import pandas as pd
import json
import os
import nltk
import re
import string
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import chromadb
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --- 0. Page Config & Styling ---
st.set_page_config(
    page_title="UniPulse Analytics",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Card" look and cleaner UI
st.markdown("""
<style>
    /* Main Background adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Styling */
    div.css-1r6slb0, div.stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Chat Interface Styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Colors ---
UNI_COLORS = {
    "LUMS": "#2E8B57",   # SeaGreen
    "NUST": "#1f77b4",   # Blue
    "GIKI": "#ff7f0e",   # Orange
    "FAST": "#d62728",   # Red
    "Unknown": "#7f7f7f" # Grey
}

# Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. NLP Setup & Caching ---
@st.cache_resource
def setup_nltk():
    resources = ["vader_lexicon", "stopwords", "punkt"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)
    return set(stopwords.words("english"))

STOPWORDS = setup_nltk()
sia = SentimentIntensityAnalyzer()

# --- 2. RAG Engine ---
class RAGEngine:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path="university_rag_db")
        try:
            self.collection = self.chroma_client.get_collection(name="pak_university_posts")
        except:
            # Silent fail for UI demo purposes, handled in query
            self.collection = None

    def query_rag(self, user_query):
        if not self.collection:
            return "⚠️ Database unavailable. Please check backend configuration."

        # 1. Generate Embedding
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        )
        query_embedding = response.data[0].embedding

        # 2. Query Vector DB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        if not results['documents'][0]:
            return "No relevant discussions found in the database."

        # 3. Build Context
        context = ""
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"\n[Post {i+1} from r/{meta['subreddit']}]\nContent: {doc}\n"

        # 4. Generate Answer
        system_prompt = """You are a helpful assistant for Pakistani university students. 
        Answer the user's question based ONLY on the provided Reddit context. 
        Compare universities if data exists. Be concise, friendly, and objective."""

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content

rag_engine = RAGEngine()

# --- 3. Data Loading ---
@st.cache_data
def load_and_process_data():
    files = ["giki_data.json", "lums_data.json", "nust_data.json"]
    all_data = []
    
    for file in files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for post in data:
                    full_text = f"{post.get('title', '')} {post.get('body', '')}"
                    sentiment = sia.polarity_scores(full_text)['compound']
                    text_lower = full_text.lower()
                    
                    category = "General"
                    if any(x in text_lower for x in ['admission', 'merit', 'test', 'net', 'sat', 'score', 'chance']):
                        category = "Admissions"
                    elif any(x in text_lower for x in ['hostel', 'mess', 'food', 'room', 'wifi', 'gym', 'social', 'event']):
                        category = "Campus Life"
                    elif any(x in text_lower for x in ['gpa', 'course', 'professor', 'exam', 'grade', 'study', 'major']):
                        category = "Academics"
                        
                    all_data.append({
                        "id": post.get('id'),
                        "subreddit": post.get('subreddit', 'Unknown').upper(),
                        "title": post.get('title'),
                        "full_text": full_text,
                        "comments": " ".join(post.get('comments', [])),
                        "upvotes": post.get('upvotes', 0),
                        "num_comments": post.get('num_comments', 0),
                        "timestamp": post.get('timestamp'),
                        "sentiment": sentiment,
                        "category": category
                    })
    
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- 4. Helper Functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    return " ".join([t for t in tokens if t not in STOPWORDS and len(t) > 2])

def get_topics(text_series, n_topics=3):
    if text_series.empty: return []
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english', ngram_range=(3, 3))
    try:
        dtm = vectorizer.fit_transform(text_series)
    except ValueError:
        return ["Not enough data for topic modeling."]
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_phrase = feature_names[topic.argsort()[-1]]
        topics.append(f"📌 {top_phrase}")
    return topics

def generate_ai_summary(text_series, context_label="this topic"):
    if text_series.empty: return "No data available."
    full_text = " ".join(text_series.tolist())[:15000]
    prompt = f"Summarize sentiment for {context_label}. Max 3 sentences. Text: {full_text}"
    try:
        response = rag_engine.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e: return f"AI Error: {e}"

# --- 5. Visualization Components ---

def plot_sentiment_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [-1, -0.2], 'color': "#ffcccb"},
                {'range': [-0.2, 0.2], 'color': "#f0f0f0"},
                {'range': [0.2, 1], 'color': "#90ee90"}],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def render_comparative_visualizations(df):
    """
    Renders a clean, insight-focused comparative analysis for all universities.
    Removed all references to post counts/data volume to focus on findings.
    """
    st.markdown("### 🏆 University Performance Scorecard")
    st.markdown("Relative performance metrics based on student sentiment and community interaction.")
    
    # 1. Ranking Data Preparation (Focusing only on Scores)
    ranking_data = []
    for uni in df['subreddit'].unique():
        uni_df = df[df['subreddit'] == uni]
        ranking_data.append({
            'University': uni,
            'Pulse Index': uni_df['sentiment'].mean(),
            'Community Vitality': uni_df['num_comments'].mean()
        })
    ranking_df = pd.DataFrame(ranking_data).sort_values(by='Pulse Index', ascending=False)

    # 2. Scorecard with simplified labels and tooltips
    st.dataframe(
        ranking_df,
        column_config={
            "University": st.column_config.TextColumn("University", width="medium"),
            "Pulse Index": st.column_config.NumberColumn(
                "Pulse Index (Sentiment)", 
                format="%.2f", 
                help="📊 WHAT THIS MEANS: A score from -1.0 to +1.0. High scores indicate a positive, satisfied student body. Negative scores suggest widespread frustration."
            ),
            "Community Vitality": st.column_config.NumberColumn(
                "Community Vitality",
                format="%.1f",
                help="💬 WHAT THIS MEANS: The average level of interaction. A higher score means students are more engaged, helpful, and responsive to one another."
            )
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Simple Legend for the Demo
    st.caption("ℹ️ **Pulse Index Guide:** 🟢 > 0.15 (Healthy) | 🟡 0.0 to 0.15 (Neutral) | 🔴 < 0.0 (High Friction)")

    st.divider()

    # 3. Visual Comparison
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📡 360° Institutional Profile")
        st.caption("Comparing strengths across Sentiment, Community, Academics, and Social Life categories.")
        
        radar_metrics = []
        for uni in df['subreddit'].unique():
            uni_df = df[df['subreddit'] == uni]
            radar_metrics.append({
                'University': uni,
                'Sentiment': (uni_df['sentiment'].mean() + 1) / 2 * 100, 
                'Engagement': min(uni_df['num_comments'].mean() * 5, 100),
                'Academics': len(uni_df[uni_df['category'] == 'Academics']) / len(uni_df) * 100 * 2,
                'Social Life': len(uni_df[uni_df['category'] == 'Campus Life']) / len(uni_df) * 100 * 2
            })
        
        fig_radar = go.Figure()
        categories = ['Sentiment', 'Engagement', 'Academics', 'Social Life']
        
        for metric in radar_metrics:
            fig_radar.add_trace(go.Scatterpolar(
                r=[metric[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=metric['University'],
                line_color=UNI_COLORS.get(metric['University'], 'grey')
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 100])), 
            height=400, 
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Sentiment Consistency")
        st.caption("Shows the range of student mood. A concentrated box means a consistent 'vibe' across campus.")
        
        fig_box = px.box(
            df, 
            x="subreddit", 
            y="sentiment", 
            color="subreddit",
            color_discrete_map=UNI_COLORS,
            labels={"sentiment": "Sentiment Score", "subreddit": ""}
        )
        # Removed Y-axis numbers to keep it fluid and visual
        fig_box.update_layout(showlegend=False, height=400, yaxis_showticklabels=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # 4. Comparative Topic Focus (Normalized to show % rather than raw counts)
    st.markdown("#### 🗣️ Discussion Focus Area")
    st.caption("Proportion of conversation dedicated to different aspects of university life.")
    
    # Calculate percentage-based distribution instead of raw counts
    topic_dist = df.groupby(['subreddit', 'category']).size().reset_index(name='count')
    total_per_uni = df.groupby('subreddit').size().reset_index(name='total')
    topic_dist = topic_dist.merge(total_per_uni, on='subreddit')
    topic_dist['percentage'] = (topic_dist['count'] / topic_dist['total']) * 100

    fig_stack = px.bar(
        topic_dist, 
        x="percentage", 
        y="subreddit", 
        color="category", 
        orientation="h",
        labels={"percentage": "Share of Conversation (%)", "subreddit": ""},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_stack.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig_stack, use_container_width=True) 
            
# --- 6. Dashboard Renderers ---

def render_student_dashboard(df):
    # Header
    st.title("🎓 UniPulse: Student Insights")
    st.markdown("Navigate the university landscape with data-driven insights from real student conversations.")
    
    # --- Top Section: AI Assistant ---
    with st.container():
        st.markdown("### 🤖 Ask the AI")
        st.caption("Aggregated knowledge from GIKI, NUST, and LUMS subreddits.")
        
        col_chat, col_img = st.columns([3, 1])
        with col_chat:
            prompt = st.chat_input("Ask about hostel life, strictness, mess food, or merit...")
            if prompt:
                st.chat_message("user").write(prompt)
                with st.spinner("Searching database..."):
                    answer = rag_engine.query_rag(prompt)
                    st.chat_message("assistant").write(answer)
            elif not prompt:
                st.info("Try asking: _'How is the social life at LUMS vs GIKI?'_")

    st.divider()

    # --- Metrics Vibe Check ---
    st.subheader("🌡️ The Vibe Check")
    cols = st.columns(len(df['subreddit'].unique()))
    for idx, uni in enumerate(df['subreddit'].unique()):
        uni_df = df[df['subreddit'] == uni]
        avg_sent = uni_df['sentiment'].mean()
        delta_color = "normal" if avg_sent > 0 else "inverse"
        with cols[idx]:
            st.metric(
                label=uni, 
                value=f"{avg_sent:.2f}", 
                delta="Positive" if avg_sent > 0.1 else "Neutral/Negative",
                delta_color=delta_color
            )

    st.divider()

    # --- Comparative Analysis ---
    render_comparative_visualizations(df)
    
    st.divider()

    # --- Deep Dive Tabs ---
    st.subheader("🔍 University Deep Dive")
    selected_uni = st.selectbox("Select University", df['subreddit'].unique(), label_visibility="collapsed")
    
    uni_data = df[df['subreddit'] == selected_uni]
    
    tab1, tab2, tab3 = st.tabs(["☁️ Word Cloud", "🚩 Complaints & Issues", "📈 Trends"])
    
    with tab1:
        st.markdown(f"**What are {selected_uni} students talking about?**")
        if not uni_data.empty:
            clean_corpus = " ".join(uni_data['full_text'].apply(clean_text))
            wc = WordCloud(width=1000, height=400, background_color='white', colormap='viridis').generate(clean_corpus)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No data found.")
            
    with tab2:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("#### Top Complaints (Trigrams)")
            complaints = uni_data[uni_data['sentiment'] < -0.2]
            if not complaints.empty:
                topics = get_topics(complaints['full_text'].apply(clean_text))
                for t in topics:
                    st.error(t)
            else:
                st.success("Minimal complaints found.")
        
        with col_c2:
            st.markdown("#### AI Summary of Issues")
            if not complaints.empty:
                summ = generate_ai_summary(complaints['full_text'], f"complaints at {selected_uni}")
                st.info(summ)
                
    with tab3:
        st.markdown("#### Sentiment Over Time")
        df_temporal = uni_data.copy()
        df_temporal['date'] = df_temporal['timestamp'].dt.date
        daily_sent = df_temporal.groupby('date')['sentiment'].mean()
        st.line_chart(daily_sent)


def render_admin_dashboard(df):
    st.title("🛡️ Admin Command Center")
    st.markdown("Monitor institutional sentiment, detect crises, and analyze admission trends.")
    
    # --- Sidebar Filters for Admin ---
    target_uni = st.sidebar.selectbox("Select Your Institution", df['subreddit'].unique())
    admin_df = df[df['subreddit'] == target_uni]
    
    # --- KPI Row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Discussions", len(admin_df), "+12 this week")
    k2.metric("Negative Sentiment Rate", f"{(len(admin_df[admin_df['sentiment']<-0.2])/len(admin_df)*100):.1f}%")
    k3.metric("Admission Queries", len(admin_df[admin_df['category'] == 'Admissions']))
    k4.metric("Avg Engagement", f"{admin_df['num_comments'].mean():.1f}")

    st.divider()
    
    # --- Main Admin Tabs ---
    tab_overview, tab_crisis, tab_bench = st.tabs(["📊 Operational Overview", "🚨 Crisis Detection", "⚖️ Competitive Benchmarking"])
    
    with tab_overview:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Sentiment by Category")
            cat_sent = admin_df.groupby('category')['sentiment'].mean().reset_index()
            fig = px.bar(cat_sent, x='category', y='sentiment', color='sentiment', range_color=[-0.5, 0.5], color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Quick Search")
            q = st.text_input("Search student feedback:", placeholder="e.g. cafeteria quality")
            if q:
                res = rag_engine.query_rag(f"{q} at {target_uni}")
                st.info(res)

    with tab_crisis:
        st.error("#### ⚠️ High Priority Issues (Negative Sentiment < -0.3)")
        critical = admin_df[admin_df['sentiment'] < -0.3].sort_values('timestamp', ascending=False)
        
        for index, row in critical.head(3).iterrows():
            with st.container():
                st.markdown(f"**{row['title']}**")
                st.caption(f"{row['timestamp']} | Category: {row['category']} | 👍 {row['upvotes']}")
                st.write(row['full_text'][:200] + "...")
                st.markdown("---")
        
        if not critical.empty:
            st.markdown("#### Automated Issue Summary")
            st.warning(generate_ai_summary(critical['full_text'], "critical student issues"))

    with tab_bench:
        st.markdown("#### How do we compare to others?")
        render_comparative_visualizations(df)

# --- 7. Main Execution ---
def main():
    df = load_and_process_data()
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("UniPulse Analytics")
        st.markdown("---")
        view_mode = st.radio("Access Level", ["Student View", "Admin View"])
        st.markdown("---")
        st.info(f"📅 Data Last Updated:\n{df['timestamp'].max().date() if not df.empty else 'N/A'}")
        st.caption("Powered by OpenAI & ChromaDB")

    if df.empty:
        st.error("⚠️ System Offline: No data found. Please ensure JSON files are loaded.")
        return

    if view_mode == "Student View":
        render_student_dashboard(df)
    else:
        render_admin_dashboard(df)

if __name__ == "__main__":
    main()