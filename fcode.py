import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, string
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import networkx as nx
from textblob import TextBlob

# Load and prepare data
df = pd.read_csv('emails.csv')
df['Body'] = df['Body'].fillna('')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year

# Encrypt email body
key = Fernet.generate_key()
cipher = Fernet(key)
df['Encrypted'] = df['Body'].apply(lambda x: cipher.encrypt(x.encode()).decode())

# Decrypt for analysis
df['Decrypted'] = df['Encrypted'].apply(lambda x: cipher.decrypt(x.encode()).decode())

# Text cleaning
stopwords = ENGLISH_STOP_WORDS
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    return ' '.join(tokens)

print("Cleaning text...")
df['Cleaned'] = df['Decrypted'].apply(clean_text)

# TF-IDF + Clustering
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned'])

# Elbow Method
wcss = []
for i in range(1, min(10, len(df))):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(wcss)+1), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.grid()
plt.savefig("elbow_plot.png")
plt.show()

# Final Clustering
k = 3 if len(df) >= 3 else len(df)
model = KMeans(n_clusters=k, random_state=42, n_init='auto')
df['Cluster'] = model.fit_predict(X)

# PCA for 2D plotting
pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())

plt.figure(figsize=(6, 4))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=df['Cluster'], palette='Set2', s=100)
plt.title("KMeans Clusters")
plt.savefig("cluster_scatter.png")
plt.show()

# Word Cloud
all_text = ' '.join(df['Cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Emails")
plt.savefig("wordcloud.png")
plt.show()

# Emails per Year
plt.figure(figsize=(6, 4))
df.groupby('Year')['From'].count().plot(kind='bar', color='cornflowerblue')
plt.title("Emails Sent Per Year")
plt.ylabel("Count")
plt.savefig("emails_per_year.png")
plt.show()

# Pie Chart - Top Senders
top_senders = df['From'].value_counts().head(5)
plt.figure(figsize=(6, 6))
plt.pie(top_senders, labels=top_senders.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Top 5 Email Senders")
plt.savefig("top_senders.png")
plt.show()

# ====================
# Network Graph
# ====================
edges = df[['From', 'To']].dropna()
G = nx.DiGraph()
for _, row in edges.iterrows():
    sender, receiver = row['From'], row['To']
    if G.has_edge(sender, receiver):
        G[sender][receiver]['weight'] += 1
    else:
        G.add_edge(sender, receiver, weight=1)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue')
weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrows=True)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("📡 Email Communication Network")
plt.axis("off")
plt.savefig("network_graph.png")
plt.show()

# ====================
# Funnel Chart (Sentiment)
# ====================
def get_sentiment_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return 'Positive'
    elif polarity < -0.3:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Decrypted'].apply(get_sentiment_label)
sentiment_counts = df['Sentiment'].value_counts()[['Positive', 'Neutral', 'Negative']]

plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.values, y=sentiment_counts.index, palette="coolwarm")
plt.title("Funnel Chart - Email Sentiments")
plt.xlabel("Count")
plt.ylabel("Sentiment")
plt.savefig("funnel_chart.png")
plt.show()

print("✅ All graphs generated, displayed, and saved.")
