import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import requests
from pyvis.network import Network
from yake import KeywordExtractor
from community import best_partition
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit.components.v1 as components
import io
import xlsxwriter
from collections import Counter

# Initialize models
kw_extractor = KeywordExtractor(lan="en", top=5)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Create static folder for HTML files
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Standard graph height
GRAPH_HEIGHT = 700

class ResearchKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.papers = []
        self.embeddings = None
        self.communities = {}
        
    def fetch_papers_semantic_scholar(self, keywords, max_results=10):
        try:
            query = " ".join(keywords)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,authors,url,abstract,year,venue,publicationVenue,influentialCitationCount,isOpenAccess,openAccessPdf"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return [{
                    "title": paper.get("title", ""),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                    "summary": paper.get("abstract", ""),
                    "link": paper.get("url", ""),
                    "published": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "publicationVenue": paper.get("publicationVenue", {}),
                    "influentialCitationCount": paper.get("influentialCitationCount", 0),
                    "isOpenAccess": paper.get("isOpenAccess", False),
                    "openAccessPdf": paper.get("openAccessPdf", {}),
                    "source": "Semantic Scholar"
                } for paper in data.get("data", [])]
            return []
        except Exception as e:
            st.error(f"Error fetching from Semantic Scholar: {e}")
            return []

    def fetch_papers_openalex(self, keywords, max_results=10):
        try:
            query = " ".join(keywords)
            url = f"https://api.openalex.org/works?search={query}&per-page={max_results}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                papers = []
                for paper in data.get("results", []):
                    journal_info = {}
                    primary_location = paper.get("primary_location", {})
                    source = primary_location.get("source", {})
                    
                    if source:
                        journal_info = {
                            "journal_name": source.get("display_name", ""),
                            "issn": source.get("issn", []),
                            "is_oa": source.get("is_oa", False),
                            "is_in_doaj": source.get("is_in_doaj", False),
                            "is_indexed_in_scopus": source.get("is_indexed_in_scopus", False),
                            "journal_type": source.get("type", ""),
                            "host_organization": source.get("host_organization_name", "")
                        }
                        
                    papers.append({
                        "title": paper.get("title", ""),
                        "authors": [author.get("author", {}).get("display_name", "") for author in paper.get("authorships", [])],
                        "summary": self.process_abstract_inverted_index(paper.get("abstract_inverted_index", {})),
                        "link": paper.get("doi", ""),
                        "published": paper.get("publication_date", ""),
                        "journal_info": journal_info,
                        "landing_page_url": primary_location.get("landing_page_url", ""),
                        "pdf_url": primary_location.get("pdf_url", ""),
                        "is_open_access": primary_location.get("is_oa", False),
                        "source": "OpenAlex"
                    })
                return papers
            return []
        except Exception as e:
            st.error(f"Error fetching from OpenAlex: {e}")
            return []

    def process_abstract_inverted_index(self, inverted_index):
        if not inverted_index:
            return ""
        try:
            all_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    all_positions.append((word, pos))
            
            all_positions.sort(key=lambda x: x[1])
            return " ".join(word for word, _ in all_positions)
        except:
            return ""
    
    def extract_keyphrases(self, text, num_keywords=5):
        if not text:
            return []
        try:
            kw_extractor = KeywordExtractor(lan="en", top=num_keywords)
            keywords = kw_extractor.extract_keywords(text)
            return [kw[0] for kw in keywords]
        except:
            return []

    def fetch_papers(self, keywords, max_results=10):
        self.papers = []
        with st.spinner('Fetching papers based on extracted keywords...'):
            semantic_papers = self.fetch_papers_semantic_scholar(keywords, max_results)
            self.papers.extend(semantic_papers)
            
            openalex_papers = self.fetch_papers_openalex(keywords, max_results)
            self.papers.extend(openalex_papers)
        
        # Create embeddings for similarity calculations
        self.create_embeddings()

    def create_embeddings(self):
        if not self.papers:
            return
            
        texts = []
        for paper in self.papers:
            text = paper["title"]
            if paper.get("summary"):
                text += " " + paper["summary"]
            texts.append(text)
        
        # Create embeddings
        self.embeddings = embedder.encode(texts)

    def build_graph(self):
        with st.spinner('Building knowledge graph...'):
            self.graph.clear()
            
            for paper in self.papers:
                title = paper["title"]
                source = paper.get("source", "Unknown")
                
                self.graph.add_node(title, 
                                   type="paper", 
                                   link=paper["link"], 
                                   source=source,
                                   published=paper.get("published", ""))
                
                for author in paper.get("authors", []):
                    if not author:
                        continue
                    self.graph.add_node(author, type="author")
                    self.graph.add_edge(author, title, relationship="wrote")
                
                keywords = self.extract_keyphrases(paper.get("summary", ""))
                for keyword in keywords:
                    self.graph.add_node(keyword, type="keyword")
                    self.graph.add_edge(title, keyword, relationship="has_keyword")

    def detect_communities(self):
        if not self.graph.nodes:
            return {}
            
        with st.spinner('Detecting communities using Louvain algorithm...'):
            partition = best_partition(self.graph)
            self.communities['louvain'] = partition
            return partition

    def visualize_graph(self, physics_enabled=True):
        if not self.graph.nodes:
            return None
            
        if 'louvain' not in self.communities:
            self.detect_communities()
            
        with st.spinner('Generating visualization...'):
            partition = self.communities.get('louvain', {})
            
            net = Network(notebook=False, height=f"{GRAPH_HEIGHT}px", width="100%", directed=False)
            
            if physics_enabled:
                net.barnes_hut(
                    gravity=-5000,
                    central_gravity=0.1,
                    spring_length=250,
                    spring_strength=0.01,
                    damping=0.09
                )
                
                net.set_options("""
                {
                  "interaction": {
                    "navigationButtons": true,
                    "zoomView": true
                  },
                  "physics": {
                    "stabilization": {
                      "iterations": 100,  
                      "fit": true           
                    }
                  }
                }
                """)
            else:
                net.toggle_physics(False)
            
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            if partition:
                community_ids = set(partition.values())
                paper_colors = plt.cm.Set1(np.linspace(0, 1, len(community_ids)))
                author_colors = plt.cm.Set2(np.linspace(0, 1, len(community_ids)))
                keyword_colors = plt.cm.Set3(np.linspace(0, 1, len(community_ids)))
                
                paper_colors = [mcolors.rgb2hex(paper_colors[i % len(paper_colors)]) for i in range(len(community_ids))]
                author_colors = [mcolors.rgb2hex(author_colors[i % len(author_colors)]) for i in range(len(community_ids))]
                keyword_colors = [mcolors.rgb2hex(keyword_colors[i % len(keyword_colors)]) for i in range(len(community_ids))]
                
                community_to_color = {
                    'paper': {comm_id: paper_colors[i] for i, comm_id in enumerate(community_ids)},
                    'author': {comm_id: author_colors[i] for i, comm_id in enumerate(community_ids)},
                    'keyword': {comm_id: keyword_colors[i] for i, comm_id in enumerate(community_ids)}
                }
            
            for node, data in self.graph.nodes(data=True):
                node_type = data["type"]
                
                if node in partition:
                    comm_id = partition[node]
                    if partition and community_to_color:
                        color = community_to_color[node_type][comm_id]
                    else:
                        color = "#FF6B6B" if node_type == "paper" else "#4ECDC4" if node_type == "author" else "#45B7D1"
                else:
                    color = "#FF6B6B" if node_type == "paper" else "#4ECDC4" if node_type == "author" else "#45B7D1"
                
                if node_type == "paper":
                    shape = "dot"
                    size = 40
                    title = f"Paper: {node}"
                    if data.get('source'):
                        title += f"<br>Source: {data.get('source')}"
                    if data.get('published'):
                        title += f"<br>Published: {data.get('published')}"
                    if node in partition:
                        title += f"<br>Community: {partition[node]}"
                elif node_type == "author":
                    shape = "diamond" 
                    size = 20
                    title = f"Author: {node}"
                    if node in partition:
                        title += f"<br>Community: {partition[node]}"
                else:  # keyword
                    shape = "triangle"
                    size = 20
                    title = f"Keyword: {node}"
                    if node in partition:
                        title += f"<br>Community: {partition[node]}"
                
                url = data.get("link", None) if node_type == "paper" else None
                
                label = node
                if len(label) > 30:
                    label = label[:27] + "..."
                
                net.add_node(node, label=label, color=color, title=title, shape=shape, size=size, url=url)
            
            for source, target, data in self.graph.edges(data=True):
                if partition and source in partition and target in partition:
                    same_community = partition[source] == partition[target]
                    edge_color = "#777777" if same_community else "#cccccc"
                    edge_width = 2 if same_community else 1
                else:
                    edge_color = "#777777"
                    edge_width = 1
                    
                net.add_edge(source, target, title=data["relationship"], color=edge_color, width=edge_width)
            
            graph_path = os.path.join(STATIC_FOLDER, "graph_louvain.html")
            net.save_graph(graph_path)
            
            return graph_path

    def recommend_journals(self, num_journals=5):
        if not self.papers:
            return []
            
        journals = []
        
        # Collect all journal information
        for paper in self.papers:
            if paper['source'] == 'Semantic Scholar' and paper.get('venue'):
                journals.append({
                    'name': paper.get('venue'),
                    'published': paper.get('published'),
                    'open_access': paper.get('isOpenAccess', False),
                    'details': paper.get('publicationVenue', {}),
                    'source': 'Semantic Scholar'
                })
            elif paper['source'] == 'OpenAlex' and paper.get('journal_info'):
                journal_info = paper.get('journal_info')
                if journal_info.get('journal_name'):
                    journals.append({
                        'name': journal_info.get('journal_name'),
                        'published': paper.get('published'),
                        'open_access': journal_info.get('is_oa', False),
                        'in_scopus': journal_info.get('is_indexed_in_scopus', False),
                        'in_doaj': journal_info.get('is_in_doaj', False),
                        'type': journal_info.get('journal_type'),
                        'publisher': journal_info.get('host_organization'),
                        'issn': journal_info.get('issn', []),
                        'source': 'OpenAlex'
                    })
        
        # Count journal frequencies and get top journals
        if journals:
            journal_counter = Counter([j['name'] for j in journals if j['name']])
            top_journals = journal_counter.most_common(num_journals)
            
            # Get detailed information for top journals
            detailed_recommendations = []
            for journal_name, count in top_journals:
                # Find the most complete entry for this journal
                journal_entries = [j for j in journals if j['name'] == journal_name]
                if journal_entries:
                    # Get the entry with most fields filled
                    best_entry = max(journal_entries, key=lambda x: len(x))
                    best_entry['frequency'] = count
                    best_entry['relevance_score'] = count / len(self.papers)  # Simple relevance metric
                    detailed_recommendations.append(best_entry)
            
            return detailed_recommendations
        
        return []

# Streamlit UI
st.set_page_config(layout="wide", page_title="Journal Recommendation System", page_icon="📚")

# Initialize session state variables
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'graph_file' not in st.session_state:
    st.session_state.graph_file = None
if 'selected_paper_index' not in st.session_state:
    st.session_state.selected_paper_index = 0
if 'kg' not in st.session_state:
    st.session_state.kg = ResearchKnowledgeGraph()
if 'extracted_keywords' not in st.session_state:
    st.session_state.extracted_keywords = []
if 'recommended_journals' not in st.session_state:
    st.session_state.recommended_journals = []

# Sidebar for inputs
with st.sidebar:
    st.title("📚 Journal Finder")
    
    st.header("Paper Information")
    paper_title = st.text_input("Paper Title:", key="paper_title")
    paper_abstract = st.text_area("Paper Abstract:", height=200, key="paper_abstract")
    
    num_keywords = st.slider("Number of Keywords to Extract", min_value=3, max_value=10, value=5)
    
    if st.button("Analyze Paper"):
        if paper_title and paper_abstract:
            # Extract keywords using the enhanced method
            kg = st.session_state.kg
            combined_text = paper_title + " " + paper_abstract
            keywords = kg.extract_keyphrases(combined_text, num_keywords)
            
            if keywords:
                st.session_state.extracted_keywords = keywords
                
                # Fetch papers based on keywords
                kg.fetch_papers(keywords, max_results=15)
                
                # Build the graph
                kg.build_graph()
                
                # Detect communities
                kg.detect_communities()
                
                # Visualize the graph
                graph_file = kg.visualize_graph(physics_enabled=True)
                
                # Recommend journals
                recommendations = kg.recommend_journals(num_journals=5)
                
                # Store in session state
                st.session_state.graph_file = graph_file
                st.session_state.graph_generated = True
                st.session_state.papers = kg.papers
                st.session_state.recommended_journals = recommendations
                
                st.success(f"Analysis complete! Found {len(kg.papers)} relevant papers.")
                st.rerun()
            else:
                st.error("Could not extract keywords. Please provide more detailed abstract.")
        else:
            st.error("Please provide both paper title and abstract.")
    
    # Graph visualization settings
    st.title("Graph Settings")
    physics_enabled = st.checkbox("Enable Physics (Interactive Movement)", value=True)
    
    # Color legend
    st.markdown("### 🎨 Graph Legend")
    st.markdown("🔴 **Papers** - Research papers")
    st.markdown("🟢 **Authors** - Paper authors")
    st.markdown("🔵 **Keywords** - Key concepts")

# Main content
st.title("📊 Journal Recommendation System")
st.markdown("""
Enter your paper title and abstract in the sidebar to find the best journals for submission.
The system will extract keywords, analyze similar papers, and recommend relevant journals.
""")

# Display results if analysis is done
if st.session_state.graph_generated and st.session_state.papers:
    # Show extracted keywords
    st.header("Extracted Keywords")
    cols = st.columns(len(st.session_state.extracted_keywords))
    for i, kw in enumerate(st.session_state.extracted_keywords):
        cols[i].markdown(f"**{kw}**")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Journal Recommendations", "Research Graph", "Related Papers", "Data Export"])
    
    with tab1:
        st.header("Top Journal Recommendations")
        
        if st.session_state.recommended_journals:
            for i, journal in enumerate(st.session_state.recommended_journals, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(f"{i}. {journal['name']}")
                        
                        # Display journal metadata
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown(f"**Publisher:** {journal.get('publisher', 'Unknown')}")
                        with cols[1]:
                            st.markdown(f"**Open Access:** {'Yes' if journal.get('open_access') else 'No'}")
                        with cols[2]:
                            if 'in_scopus' in journal:
                                st.markdown(f"**Indexed in Scopus:** {'Yes' if journal.get('in_scopus') else 'No'}")
                    
                    with col2:
                        # Simple relevance score visualization
                        score = journal.get('relevance_score', 0)
                        st.metric("Relevance", f"{int(score * 100)}%")
                    
                    # Journal details in expander
                    with st.expander("Journal Details"):
                        st.markdown(f"**Type:** {journal.get('type', 'Unknown')}")
                        st.markdown(f"**ISSN:** {', '.join(journal.get('issn', ['Unknown']))}")
                        st.markdown(f"**In DOAJ:** {'Yes' if journal.get('in_doaj') else 'Unknown'}")
                        st.markdown(f"**Source:** {journal.get('source', 'Unknown')}")
                
                st.markdown("---")
        else:
            st.info("No journal recommendations available. Add your paper details in the sidebar.")
    
    with tab2:
        st.header("Research Knowledge Graph")
        if st.session_state.graph_file:
            with open(st.session_state.graph_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=GRAPH_HEIGHT)
    
    with tab3:
        st.header("Related Papers")
        if st.session_state.papers:
            papers_df = pd.DataFrame([{
                "Title": p["title"],
                "Authors": ", ".join(p.get("authors", ["Unknown"])),
                "Published": p.get("published", "Unknown"),
                "Journal": p.get("venue", p.get("journal_info", {}).get("journal_name", "Unknown")),
                "Open Access": "Yes" if p.get("isOpenAccess", p.get("is_open_access", False)) else "No",
                "Source": p.get("source", "Unknown"),
                "Link": p.get("link", "")
            } for p in st.session_state.papers])
            
            st.dataframe(papers_df)
            
            # Show paper details
            if st.session_state.papers:
                st.subheader("Paper Details")
                paper_titles = [p["title"] for p in st.session_state.papers]
                selected_paper = st.selectbox(
                    "Select a paper to view details:",
                    options=paper_titles,
                    index=st.session_state.selected_paper_index
                )
                
                selected_index = paper_titles.index(selected_paper)
                paper = st.session_state.papers[selected_index]
                
                with st.container():
                    st.markdown(f"**Title:** {paper['title']}")
                    st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                    st.markdown(f"**Published:** {paper.get('published', 'Unknown')}")
                    st.markdown(f"**Source:** {paper.get('source', 'Unknown')}")
                    
                    with st.expander("Abstract"):
                        st.markdown(paper.get("summary", "No abstract available"))
                    
                    if paper.get("link"):
                        st.markdown(f"[Open Paper]({paper['link']})")
    
    with tab4:
        st.header("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export journal recommendations
            if st.session_state.recommended_journals:
                journals_df = pd.DataFrame([{
                    "Journal Name": j["name"],
                    "Publisher": j.get("publisher", "Unknown"),
                    "Open Access": "Yes" if j.get("open_access") else "No",
                    "In Scopus": "Yes" if j.get("in_scopus", False) else "No",
                    "In DOAJ": "Yes" if j.get("in_doaj", False) else "No",
                    "Type": j.get("type", "Unknown"),
                    "Relevance Score": f"{int(j.get('relevance_score', 0) * 100)}%"
                } for j in st.session_state.recommended_journals])
                
                csv = journals_df.to_csv(index=False)
                st.download_button(
                    label="Download Journal Recommendations (CSV)",
                    data=csv,
                    file_name="journal_recommendations.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export graph as HTML
            if st.session_state.graph_file:
                with open(st.session_state.graph_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.download_button(
                    label="Download Research Graph (HTML)",
                    data=html_content,
                    file_name="research_graph.html",
                    mime="text/html"
                )
else:
    # Instructions when first loading
    st.info("Enter your paper title and abstract in the sidebar, then click 'Analyze Paper' to get journal recommendations.")
    
    # Example with sample data
    with st.expander("See Example"):
        st.markdown("""
        **Example Paper Title:**
        "Deep Learning Approaches for COVID-19 Detection in Chest X-ray Images"
        
        **Example Abstract:**
        "The COVID-19 pandemic has created an urgent need for rapid diagnostic tools. This paper presents a novel deep learning approach for detecting COVID-19 from chest X-ray images. We propose a convolutional neural network architecture that achieves 97% accuracy in distinguishing COVID-19 cases from normal and pneumonia cases. Our model utilizes transfer learning and data augmentation techniques to overcome the limited availability of COVID-19 X-ray samples. Experimental results demonstrate that the proposed method outperforms existing approaches in terms of sensitivity and specificity. This research contributes to the development of AI-assisted diagnostic tools that can help medical professionals in resource-constrained environments."
        """)