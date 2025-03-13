import streamlit as st
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
from typing import List

def get_available_cluster_files(layer_dir: str) -> List[str]:
    """Get list of available cluster files and extract their types and sizes."""
    cluster_files = []
    for file in os.listdir(layer_dir):
        if file.startswith('clusters-') and file.endswith('.txt'):
            # Parse files like 'clusters-agg-10.txt' or 'clusters-kmeans-500.txt'
            parts = file.replace('.txt', '').split('-')
            if len(parts) == 3 and parts[2].isdigit():
                cluster_files.append(file)
    for file in sorted(cluster_files):
        st.sidebar.write(f"- {file}")
    return sorted(cluster_files)

def parse_cluster_filename(filename: str) -> tuple:
    """Parse cluster filename to get algorithm and size."""
    parts = filename.replace('.txt', '').split('-')
    return parts[1], int(parts[2])  # returns (algorithm, size)

def load_cluster_sentences(model_dir: str, language: str, cluster_type: str, layer: int, cluster_file: str):
    """Load sentences and their cluster assignments for a given model and layer."""
    # Input file is in the language directory
    sentence_file = os.path.join(model_dir, language, "input.in")
    cluster_file_path = os.path.join(model_dir, language, f"layer{layer}", cluster_type, cluster_file)
    
    # Load all sentences first
    with open(sentence_file, 'r', encoding='utf-8') as f:
        all_sentences = [line.strip() for line in f]
    
    # Process cluster file to get sentence mappings
    cluster_sentences = defaultdict(list)
    
    with open(cluster_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            pipe_count = stripped_line.count('|')
            
            # Handle special cases for pipe tokens
            if pipe_count == 13:
                token = '|'
                parts = stripped_line.split('|||')
                occurrence = 1  # Default occurrence for special tokens
                sentence_id = int(parts[2])
                token_idx = int(parts[3])
                cluster_id = parts[4].strip()
            elif pipe_count == 14:
                token = '||'
                parts = stripped_line.split('|||')
                occurrence = 1  # Default occurrence for special tokens
                sentence_id = int(parts[2])
                token_idx = int(parts[3])
                cluster_id = parts[4].strip()
            else:
                # Normal case
                parts = stripped_line.split('|||')
                if len(parts) != 5:
                    continue
                    
                token = parts[0].strip()
                try:
                    occurrence = int(parts[1])
                    sentence_id = int(parts[2])
                    token_idx = int(parts[3])
                    cluster_id = parts[4].strip()
                except ValueError:
                    # Skip lines with invalid number formats
                    continue
            
            if 0 <= sentence_id < len(all_sentences):
                cluster_sentences[f"c{cluster_id}"].append({
                    "sentence": all_sentences[sentence_id],
                    "token": token,
                    "token_idx": token_idx,
                    "occurrence": occurrence,
                    "sentence_id": sentence_id
                })
    
    return cluster_sentences

def create_sentence_html(sentence, sent_info, cluster_tokens=None):
    """Create HTML for sentence with highlighted tokens
    Args:
        sentence: The full sentence text
        sent_info: Dictionary containing token and position info
        cluster_tokens: Set of all unique tokens in this cluster
    """
    html = """
    <div style='font-family: monospace; padding: 10px; margin: 5px 0; background-color: #f5f5f5; border-radius: 5px;'>
        <div style='margin-bottom: 5px;'>
    """
    
    # Get token information
    target_token = sent_info["token"]
    target_idx = sent_info["token_idx"]
    line_number = sent_info["sentence_id"]
    
    # Split the tokenized sentence
    tokens = sentence.split()
    
    # Create set of cluster tokens if provided, excluding the target token
    other_cluster_tokens = set(cluster_tokens or []) - {target_token}
    
    # Highlight tokens based on their type
    for i, token in enumerate(tokens):
        if i == target_idx:
            # Target token in red
            html += f"<span style='color: red; font-weight: bold;'>{token}</span> "
        elif token in other_cluster_tokens:
            # Other cluster tokens in a milder green
            html += f"<span style='color: #2e8b57; font-weight: bold;'>{token}</span> "  # Using SeaGreen color
        else:
            # Regular tokens
            html += f"{token} "
    
    html += f"""
        </div>
        <div style='color: #666; font-size: 0.9em;'>Token: <code>{target_token}</code> (Line: {line_number}, Index: {target_idx})</div>
    </div>
    """
    return html

def display_cluster_analysis(model_name: str, language: str, cluster_type: str, selected_layer: int, cluster_file: str):
    """Display cluster analysis for selected model and layer."""
    # Load cluster data
    cluster_sentences = load_cluster_sentences(model_name, language, cluster_type, selected_layer, cluster_file)
    
    # Get clustering algorithm and size from filename
    algorithm, size = parse_cluster_filename(cluster_file)
    st.write(f"### Analyzing {algorithm.upper()} clustering with {size} clusters")
    
    # Create cluster selection with navigation buttons
    cluster_ids = sorted(cluster_sentences.keys(), key=lambda x: int(x[1:]))  # Sort by numeric ID
    
    # Initialize session state if needed
    if 'cluster_index' not in st.session_state:
        st.session_state['cluster_index'] = 0
    
    # Create columns with adjusted widths for better spacing
    col1, col2, col3, col4 = st.columns([3, 1, 1, 7])  # Adjusted column ratios
    
    # Add some vertical space before the controls
    st.write("")
    
    with col1:
        selected_cluster = st.selectbox(
            "Select cluster",
            range(len(cluster_ids)),
            index=st.session_state['cluster_index'],
            format_func=lambda x: f"Cluster {cluster_ids[x]}",
            label_visibility="collapsed"  # Hides the label but keeps accessibility
        )
        # Update session state when dropdown changes
        st.session_state['cluster_index'] = selected_cluster
    
    # Previous cluster button with custom styling
    with col2:
        if st.button("◀", use_container_width=True):
            st.session_state['cluster_index'] = max(0, st.session_state['cluster_index'] - 1)
            st.rerun()
    
    # Next cluster button with custom styling
    with col3:
        if st.button("▶", use_container_width=True):
            st.session_state['cluster_index'] = min(len(cluster_ids) - 1, st.session_state['cluster_index'] + 1)
            st.rerun()

    # Add some vertical space after the controls
    st.write("")
    
    # Get the current cluster
    cluster_id = cluster_ids[st.session_state['cluster_index']]
    sentences_data = cluster_sentences[cluster_id]
    
    # Get all unique tokens in this cluster
    cluster_tokens = {sent_info["token"] for sent_info in sentences_data}
    
    # Display word cloud for this cluster
    st.write("### Word Cloud")
    wc = create_wordcloud(cluster_tokens)
    if wc:
        # Create a centered column for the word cloud
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Clear any existing matplotlib figures
            plt.clf()
            
            # Create new figure with smaller size
            fig, ax = plt.subplots(figsize=(5, 3))  # Reduced width from 10 to 5
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            
            # Display the figure
            st.pyplot(fig)
            
            # Clean up
            plt.close(fig)
    
    # Display context sentences for this cluster only
    st.write("### Context Sentences")
    seen_sentences = set()  # Track unique sentences
    for sent_info in sentences_data:
        # Only show each unique sentence once
        if sent_info["sentence"] not in seen_sentences:
            html = create_sentence_html(sent_info["sentence"], sent_info, cluster_tokens)
            st.markdown(html, unsafe_allow_html=True)
            seen_sentences.add(sent_info["sentence"])

def main():
    # Set page to use full width
    st.set_page_config(layout="wide")
    
    st.title("Coconet Visual Analysis")
    
    # Initialize session state for selections if they don't exist
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = None
    if 'selected_layer' not in st.session_state:
        st.session_state.selected_layer = None
    if 'selected_cluster_type' not in st.session_state:
        st.session_state.selected_cluster_type = None
    if 'selected_cluster_file' not in st.session_state:
        st.session_state.selected_cluster_file = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Individual Clusters"
    
    # Get available models (directories in the current directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    available_models = [d for d in os.listdir(current_dir) 
                       if os.path.isdir(os.path.join(current_dir, d))]
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Data",
        available_models,
        key="model_select",
        index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0
    )
    
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name
        st.session_state.selected_language = None
        st.session_state.selected_layer = None
        st.session_state.selected_cluster_type = None
        st.session_state.selected_cluster_file = None
    
    if not model_name:
        st.error("No models found")
        return
    
    # Get available languages for the selected model
    model_dir = os.path.join(current_dir, model_name)
    available_languages = [d for d in os.listdir(model_dir) 
                         if os.path.isdir(os.path.join(model_dir, d))]
    
    # Language selection
    selected_language = st.sidebar.selectbox(
        "Select Language",
        available_languages,
        key="language_select",
        index=available_languages.index(st.session_state.selected_language) if st.session_state.selected_language in available_languages else 0
    )
    
    if selected_language != st.session_state.selected_language:
        st.session_state.selected_language = selected_language
        st.session_state.selected_layer = None
        st.session_state.selected_cluster_type = None
        st.session_state.selected_cluster_file = None
    
    if not selected_language:
        st.error("No languages found for selected model")
        return
    
    # Get available layers
    language_dir = os.path.join(model_dir, selected_language)
    layer_dirs = [d for d in os.listdir(language_dir) 
                 if d.startswith('layer') and os.path.isdir(os.path.join(language_dir, d))]
    
    if not layer_dirs:
        st.error("No layer directories found")
        return
        
    # Extract layer numbers
    available_layers = sorted([int(d.replace('layer', '')) for d in layer_dirs])
    
    # Layer selection
    selected_layer = st.sidebar.selectbox(
        "Select Layer",
        available_layers,
        key="layer_select",
        index=available_layers.index(st.session_state.selected_layer) if st.session_state.selected_layer in available_layers else 0
    )
    
    if selected_layer != st.session_state.selected_layer:
        st.session_state.selected_layer = selected_layer
        st.session_state.selected_cluster_type = None
        st.session_state.selected_cluster_file = None
    
    # Get available clustering types
    layer_dir = os.path.join(language_dir, f"layer{selected_layer}")
    available_cluster_types = [d for d in os.listdir(layer_dir) 
                             if os.path.isdir(os.path.join(layer_dir, d))]
    
    # Clustering type selection
    selected_cluster_type = st.sidebar.selectbox(
        "Select Clustering Type",
        available_cluster_types,
        key="cluster_type_select",
        index=available_cluster_types.index(st.session_state.selected_cluster_type) if st.session_state.selected_cluster_type in available_cluster_types else 0
    )
    
    if selected_cluster_type != st.session_state.selected_cluster_type:
        st.session_state.selected_cluster_type = selected_cluster_type
        st.session_state.selected_cluster_file = None
    
    if not selected_cluster_type:
        st.error("No clustering types found for selected layer")
        return
    
    # Get available cluster files
    cluster_dir = os.path.join(layer_dir, selected_cluster_type)
    available_cluster_files = get_available_cluster_files(cluster_dir)
    
    if not available_cluster_files:
        st.error("No cluster files found in the selected layer")
        return
    
    # Cluster file selection
    selected_cluster_file = st.sidebar.selectbox(
        "Select Clustering",
        available_cluster_files,
        key="cluster_file_select",
        format_func=lambda x: f"{parse_cluster_filename(x)[0].upper()} (k={parse_cluster_filename(x)[1]})",
        index=available_cluster_files.index(st.session_state.selected_cluster_file) if st.session_state.selected_cluster_file in available_cluster_files else 0
    )
    
    st.session_state.selected_cluster_file = selected_cluster_file

    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ["Individual Clusters", "Search And Analysis", "Token Pairs"],
        key="analysis_mode_select",
        index=["Individual Clusters", "Search And Analysis", "Token Pairs"].index(st.session_state.analysis_mode)
    )
    
    st.session_state.analysis_mode = analysis_mode

    # Call appropriate analysis function based on mode
    if analysis_mode == "Individual Clusters":
        display_cluster_analysis(model_name, selected_language, selected_cluster_type, selected_layer, selected_cluster_file)
    elif analysis_mode == "Search And Analysis":
        handle_token_search(model_name, selected_language, selected_cluster_type, selected_layer, selected_cluster_file)
    elif analysis_mode == "Token Pairs":
        display_token_pair_analysis(model_name, selected_language, selected_cluster_type, selected_layer, selected_cluster_file)

def display_token_evolution(evolution_data: dict, tokens: List[str]):
    """Display evolution analysis for tokens"""
    st.write(f"### Evolution Analysis for Token(s)")
    
    # Create main evolution graph
    fig = go.Figure()
    
    # Colors for different types of lines
    colors = {
        'individual': ['#3498db', '#e74c3c', '#2ecc71'],  # Blue, Red, Green
        'exclusive': ['#9b59b6', '#f1c40f', '#1abc9c'],   # Purple, Yellow, Turquoise
        'combined': '#34495e'                              # Dark Gray
    }
    
    # Add individual count lines
    for i, token in enumerate(tokens):
        fig.add_trace(go.Scatter(
            x=evolution_data['layers'],
            y=evolution_data['individual_counts'][token],
            name=f"'{token}' (Total)",
            mode='lines+markers',
            line=dict(color=colors['individual'][i], width=2),
            marker=dict(size=8)
        ))
        
        # Add exclusive count lines only for multiple tokens
        if len(tokens) > 1:
            fig.add_trace(go.Scatter(
                x=evolution_data['layers'],
                y=evolution_data['exclusive_counts'][token],
                name=f"'{token}' (Exclusive)",
                mode='lines+markers',
                line=dict(color=colors['exclusive'][i], width=2, dash='dot'),
                marker=dict(size=8)
            ))
    
    # Add combined counts if multiple tokens
    if len(tokens) > 1:
        fig.add_trace(go.Scatter(
            x=evolution_data['layers'],
            y=evolution_data['combined_counts'],
            name='Co-occurring',
            mode='lines+markers',
            line=dict(color=colors['combined'], width=2),
            marker=dict(size=8)
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Token Evolution Across Layers',
            font=dict(size=20)
        ),
        xaxis_title=dict(
            text='Layer',
            font=dict(size=14)
        ),
        yaxis_title=dict(
            text='Number of Clusters',
            font=dict(size=14)
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add gridlines
    fig.update_xaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
    fig.update_yaxes(gridcolor='LightGray', gridwidth=0.5, griddash='dot')
    
    st.plotly_chart(fig, use_container_width=True)

def find_clusters_for_token(model_name: str, language: str, cluster_type: str, layer: int, cluster_file: str, search_token: str) -> dict:
    """Find clusters containing the specified token"""
    matching_tokens = set()
    clusters = defaultdict(set)
    
    try:
        cluster_file_path = os.path.join(model_name, language, f"layer{layer}", cluster_type, cluster_file)
        
        # First collect all matching tokens and their clusters
        with open(cluster_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 5:  # token|||occurrence|||line_number|||column_number|||cluster_id
                    token = parts[0].strip()
                    cluster_id = f"c{parts[4].strip()}"
                    
                    # Only process if token contains search term
                    if search_token.lower() in token.lower():
                        matching_tokens.add(token)
                        clusters[cluster_id].add(token)
                                                
    except Exception as e:
        st.error(f"Error reading cluster file: {e}")
        return {}
    
    # Only return clusters that have matching tokens
    return {k: {'matching_tokens': sorted(v)} for k, v in clusters.items() if v}

def handle_token_search(model_name: str, language: str, cluster_type: str, layer: int, cluster_file: str):
    """Handle token search functionality"""
    st.write("### Token Search")
    
    # Initialize session state for search results if needed
    if 'search_results_state' not in st.session_state:
        st.session_state.search_results_state = {
            'matching_tokens': [],
            'matching_tokens2': [],
            'last_search': None,
            'last_search2': None,
            'search_mode': 'single'  # Default to single token search
        }
    elif 'search_mode' not in st.session_state.search_results_state:
        # Add search_mode if it doesn't exist in existing state
        st.session_state.search_results_state['search_mode'] = 'single'
    
    # Radio button for search mode
    search_mode = st.radio(
        "Search Mode",
        ["Single Token", "Token Pair"],
        key="search_mode_radio",  # Added unique key
        index=0 if st.session_state.search_results_state['search_mode'] == 'single' else 1
    )
    
    # Update search mode in session state
    st.session_state.search_results_state['search_mode'] = 'single' if search_mode == "Single Token" else 'pair'
    
    if search_mode == "Single Token":
        # Single token search interface
        search_token = st.text_input("Search for token:")
        
        if search_token:
            # Find matching tokens
            all_matching_tokens = set()
            clusters = find_clusters_for_token(
                model_name,
                language,
                cluster_type,
                layer,
                cluster_file,
                search_token
            )
            
            for cluster_data in clusters.values():
                all_matching_tokens.update(cluster_data['matching_tokens'])
            
            matching_tokens = sorted(all_matching_tokens)
            
            if matching_tokens:
                selected_token = st.selectbox(
                    "Select token:",
                    matching_tokens,
                    key="token_select"
                )
                
                if selected_token:
                    # Update state
                    st.session_state.search_results_state.update({
                        'matching_tokens': matching_tokens,
                        'last_search': selected_token
                    })
                    
                    # Display results
                    st.write("### Search Results")
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Evolution Analysis", "Cluster Details"])
                    
                    with tab1:
                        if st.button("Analyze Evolution", type="primary"):
                            evolution_data = analyze_token_evolution(
                                model_name,
                                language,
                                cluster_type,
                                layer,
                                [selected_token],
                                cluster_file
                            )
                            
                            if evolution_data:
                                display_token_evolution(evolution_data, [selected_token])
                    
                    with tab2:
                        display_cluster_details(
                            model_name,
                            language,
                            cluster_type,
                            selected_token,
                            cluster_file
                        )
            else:
                st.warning("No matching tokens found")
    
    else:  # Token Pair search
        col1, col2 = st.columns(2)
        
        with col1:
            search_token1 = st.text_input("Search for first token:")
        with col2:
            search_token2 = st.text_input("Search for second token:")
        
        if search_token1 and search_token2:
            # Find matching tokens for both searches
            matching_tokens1 = set()
            matching_tokens2 = set()
            
            clusters1 = find_clusters_for_token(
                model_name,
                language,
                cluster_type,
                layer,
                cluster_file,
                search_token1
            )
            
            clusters2 = find_clusters_for_token(
                model_name,
                language,
                cluster_type,
                layer,
                cluster_file,
                search_token2
            )
            
            for cluster_data in clusters1.values():
                matching_tokens1.update(cluster_data['matching_tokens'])
            
            for cluster_data in clusters2.values():
                matching_tokens2.update(cluster_data['matching_tokens'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if matching_tokens1:
                    selected_token1 = st.selectbox(
                        "Select first token:",
                        sorted(matching_tokens1),
                        key="token1_select"
                    )
                else:
                    st.warning("No matching tokens found for first search")
                    selected_token1 = None
            
            with col2:
                if matching_tokens2:
                    selected_token2 = st.selectbox(
                        "Select second token:",
                        sorted(matching_tokens2),
                        key="token2_select"
                    )
                else:
                    st.warning("No matching tokens found for second search")
                    selected_token2 = None
            
            if selected_token1 and selected_token2:
                # Update state
                st.session_state.search_results_state.update({
                    'matching_tokens': sorted(matching_tokens1),
                    'matching_tokens2': sorted(matching_tokens2),
                    'last_search': selected_token1,
                    'last_search2': selected_token2
                })
                
                # Display results
                st.write("### Search Results")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Evolution Analysis", "Co-occurring Clusters"])
                
                with tab1:
                    if st.button("Analyze Evolution", type="primary"):
                        evolution_data = analyze_token_evolution(
                            model_name,
                            language,
                            cluster_type,
                            layer,
                            [selected_token1, selected_token2],
                            cluster_file
                        )
                        
                        if evolution_data:
                            display_token_evolution(evolution_data, [selected_token1, selected_token2])
                
                with tab2:
                    display_cluster_details(
                        model_name,
                        language,
                        cluster_type,
                        selected_token1,
                        cluster_file,
                        second_token=selected_token2
                    )

def analyze_token_evolution(model_name: str, language: str, cluster_type: str, layer: int, tokens: List[str], cluster_file: str) -> dict:
    """Analyze token evolution across all available layers"""
    # Get all available layers by checking directories
    language_dir = os.path.join(model_name, language)
    available_layers = []
    for d in os.listdir(language_dir):
        if d.startswith('layer') and os.path.isdir(os.path.join(language_dir, d)):
            try:
                layer_num = int(d.replace('layer', ''))
                available_layers.append(layer_num)
            except ValueError:
                continue
    
    available_layers.sort()  # Sort layers numerically
    
    evolution_data = {
        'layers': available_layers,
        'individual_counts': {token: [] for token in tokens},
        'exclusive_counts': {token: [] for token in tokens},  # New: track exclusive counts
        'combined_counts': [] if len(tokens) > 1 else None
    }
    
    # Extract cluster size from the filename (e.g., "clusters-agg-500.txt" -> "500")
    cluster_size = cluster_file.split('-')[-1].replace('.txt', '')
    
    # Handle shortened form for agglomerative clustering
    cluster_type_short = "agg" if cluster_type == "agglomerative" else cluster_type
    
    for current_layer in available_layers:
        # Get clusters for each token
        token_clusters = {}
        cluster_file_path = os.path.join(
            model_name, 
            language, 
            f"layer{current_layer}", 
            cluster_type,
            f"clusters-{cluster_type_short}-{cluster_size}.txt"
        )
        
        # Skip layer if cluster file doesn't exist
        if not os.path.exists(cluster_file_path):
            continue
            
        for token in tokens:
            clusters = find_clusters_for_token(
                model_name,
                language,
                cluster_type,
                current_layer,
                f"clusters-{cluster_type_short}-{cluster_size}.txt",
                token
            )
            token_clusters[token] = set(clusters.keys())
            evolution_data['individual_counts'][token].append(len(clusters))
        
        # Calculate exclusive and co-occurring clusters
        if len(tokens) > 1:
            # Calculate co-occurrences
            cooccurring_clusters = set.intersection(*[token_clusters[token] for token in tokens])
            evolution_data['combined_counts'].append(len(cooccurring_clusters))
            
            # Calculate exclusive counts for each token
            for token in tokens:
                other_tokens = set(tokens) - {token}
                other_clusters = set.union(*[token_clusters[t] for t in other_tokens]) if other_tokens else set()
                exclusive_clusters = token_clusters[token] - other_clusters
                evolution_data['exclusive_counts'][token].append(len(exclusive_clusters))
        else:
            # For single token, exclusive count is the same as individual count
            evolution_data['exclusive_counts'][tokens[0]] = evolution_data['individual_counts'][tokens[0]]
    
    return evolution_data

def find_clusters_with_multiple_tokens(model_name: str, language: str, cluster_type: str, layer: int, cluster_file: str, tokens: List[str]) -> dict:
    """Find clusters containing multiple specified tokens"""
    clusters = defaultdict(lambda: {'matching_tokens': {token: set() for token in tokens}})
    
    try:
        cluster_file_path = os.path.join(model_name, language, f"layer{layer}", cluster_type, cluster_file)
        with open(cluster_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||')
                if len(parts) == 5:
                    token = parts[0].strip()
                    cluster_id = parts[4].strip()
                    
                    for search_token in tokens:
                        if token == search_token:
                            # Add all tokens from this cluster
                            with open(cluster_file_path, 'r', encoding='utf-8') as f2:
                                for line2 in f2:
                                    parts2 = line2.strip().split('|||')
                                    if len(parts2) == 5 and parts2[4].strip() == cluster_id:
                                        clusters[cluster_id]['matching_tokens'][search_token].add(parts2[0].strip())
    except Exception as e:
        st.error(f"Error reading cluster file: {e}")
        return {}
    
    # Filter to only keep clusters with all tokens
    return {k: v for k, v in clusters.items() if all(v['matching_tokens'][token] for token in tokens)}

def handle_semantic_tag_search(model_name: str, language: str, cluster_type: str, layer: int, cluster_file: str):
    """Handle semantic tag search functionality"""
    st.write("### Semantic Tag Search")
    st.info("This feature will be implemented soon.")

def display_token_pair_analysis(model_name: str, language: str, cluster_type: str, layer: int, cluster_file: str):
    """Display analysis for predefined token pairs"""
    st.write("### Token Pair Analysis")
    
    # Get predefined token pairs
    token_pairs = get_predefined_token_pairs()
    
    # Create tabs for each category
    tabs = st.tabs(list(token_pairs.keys()))
    
    for tab, (category, data) in zip(tabs, token_pairs.items()):
        with tab:
            st.write(f"### {category}")
            st.write(data["description"])
            
            for token1, token2 in data["pairs"]:
                with st.expander(f"{token1} vs {token2}"):
                    # Update state for token pair search
                    st.session_state.search_results_state = {
                        'matching_tokens': [token1],
                        'matching_tokens2': [token2],
                        'last_search': token1,
                        'last_search2': token2,
                        'search_mode': 'pair'
                    }
                    
                    # Display results
                    st.write("### Search Results")
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Evolution Analysis", "Co-occurring Clusters"])
                    
                    with tab1:
                        evolution_data = analyze_token_evolution(
                            model_name,
                            language,
                            cluster_type,
                            layer,
                            [token1, token2],
                            cluster_file
                        )
                        
                        if evolution_data:
                            display_token_evolution(evolution_data, [token1, token2])
                    
                    with tab2:
                        display_cluster_details(
                            model_name,
                            language,
                            cluster_type,
                            token1,
                            cluster_file,
                            second_token=token2
                        )

def get_predefined_token_pairs():
    """Return predefined token pairs organized by categories"""
    return {
        "Control Flow": {
            "description": "Different control flow constructs",
            "pairs": [
                ("for", "while"),
                ("if", "switch"),
                ("break", "continue"),
                ("try", "catch")
            ]
        },
        "Access Modifiers": {
            "description": "Access and modifier keywords",
            "pairs": [
                ("public", "private"),
                ("static", "final"),
                ("abstract", "interface")
            ]
        },
        "Variable/Type": {
            "description": "Variable and type-related tokens",
            "pairs": [
                ("int", "Integer"),
                ("null", "Optional"),
                ("var", "String")  # Example of var vs explicit type
            ]
        },
        "Collections": {
            "description": "Collection-related tokens",
            "pairs": [
                ("List", "Array"),
                ("ArrayList", "LinkedList"),
                ("HashMap", "TreeMap"),
                ("Set", "List")
            ]
        },
        "Threading": {
            "description": "Threading and concurrency tokens",
            "pairs": [
                ("synchronized", "volatile"),
                ("Runnable", "Callable"),
                ("wait", "sleep")
            ]
        },
        "Object-Oriented": {
            "description": "Object-oriented programming tokens",
            "pairs": [
                ("extends", "implements"),
                ("this", "super"),
                ("new", "clone")
            ]
        }
    }

def create_wordcloud(tokens, token1=None, token2=None):
    """Create and return a word cloud from tokens"""
    if not tokens:
        return None
        
    # Create frequency dict
    freq_dict = {}
    
    if token1 and token2:
        # Ensure the searched tokens have higher frequency
        freq_dict = {token: 1 for token in tokens}
        freq_dict[token1] = 5  # Give higher weight to searched tokens
        freq_dict[token2] = 5
    else:
        freq_dict = {token: 1 for token in tokens}
    
    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100
    ).generate_from_frequencies(freq_dict)
    
    return wc

def display_cluster_details(model_name: str, language: str, cluster_type: str, token: str, cluster_file: str, second_token: str = None):
    """Display detailed cluster information organized by layers"""
    # Get all available layers
    language_dir = os.path.join(model_name, language)
    available_layers = []
    for d in os.listdir(language_dir):
        if d.startswith('layer') and os.path.isdir(os.path.join(language_dir, d)):
            try:
                layer_num = int(d.replace('layer', ''))
                available_layers.append(layer_num)
            except ValueError:
                continue
    
    available_layers.sort()
    
    # Create tabs for each layer
    layer_tabs = st.tabs([f"Layer {layer}" for layer in available_layers])
    
    # Handle shortened form for agglomerative clustering
    cluster_type_short = "agg" if cluster_type == "agglomerative" else cluster_type
    cluster_size = cluster_file.split('-')[-1].replace('.txt', '')
    
    for layer, tab in zip(available_layers, layer_tabs):
        with tab:
            # Find clusters for this layer for both tokens
            clusters1 = find_clusters_for_token(
                model_name,
                language,
                cluster_type,
                layer,
                f"clusters-{cluster_type_short}-{cluster_size}.txt",
                token
            )
            
            if second_token:
                clusters2 = find_clusters_for_token(
                    model_name,
                    language,
                    cluster_type,
                    layer,
                    f"clusters-{cluster_type_short}-{cluster_size}.txt",
                    second_token
                )
                # Find common clusters between both tokens
                common_cluster_ids = set(clusters1.keys()) & set(clusters2.keys())
                clusters = {k: clusters1[k] for k in common_cluster_ids}
            else:
                clusters = clusters1
            
            if clusters:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create unique key for cluster selection
                    token_key = f"{token}_{second_token}" if second_token else token
                    select_key = f"cluster_select_{token_key}_{layer}_{cluster_type_short}_{cluster_size}"
                    # Create dropdown for cluster selection
                    cluster_ids = sorted(clusters.keys(), key=lambda x: int(x[1:]))  # Sort by numeric ID
                    selected_cluster = st.selectbox(
                        f"Select cluster from Layer {layer}",
                        cluster_ids,
                        format_func=lambda x: f"Cluster {x}",
                        key=select_key
                    )
                
                with col2:
                    # Add checkbox for context sentences with unique key
                    context_key = f"show_context_{token_key}_{layer}_{cluster_type_short}_{cluster_size}"
                    show_context = st.checkbox("Show Context", key=context_key)
                
                if selected_cluster:
                    st.write(f"### Cluster {selected_cluster}")
                    
                    # Get tokens for this cluster
                    cluster_tokens = clusters[selected_cluster]['matching_tokens']
                    
                    # Display word cloud
                    st.write("#### Word Cloud")
                    if second_token:
                        wc = create_wordcloud(cluster_tokens, token, second_token)
                    else:
                        wc = create_wordcloud(cluster_tokens)
                    
                    if wc:
                        # Create a centered column for the word cloud
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            # Clear any existing matplotlib figures
                            plt.clf()
                            
                            # Create new figure with smaller size
                            fig, ax = plt.subplots(figsize=(5, 3))  # Reduced width from 10 to 5
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            
                            # Display the figure
                            st.pyplot(fig)
                            
                            # Clean up
                            plt.close(fig)
                    
                    # Display context sentences only if checkbox is selected
                    if show_context:
                        sentences = load_cluster_sentences(
                            model_name,
                            language,
                            cluster_type,
                            layer,
                            f"clusters-{cluster_type_short}-{cluster_size}.txt"
                        )
                        
                        if selected_cluster in sentences:
                            if second_token:
                                # Show sentences containing either token
                                st.write(f"#### Context Sentences for '{token}' and '{second_token}'")
                                seen_sentences = set()
                                relevant_sentences = []
                                
                                for sent_info in sentences[selected_cluster]:
                                    if (sent_info["token"] in [token, second_token] and 
                                        sent_info["sentence"] not in seen_sentences):
                                        relevant_sentences.append(sent_info)
                                        seen_sentences.add(sent_info["sentence"])
                                
                                if relevant_sentences:
                                    for sent_info in relevant_sentences:
                                        html = create_sentence_html(sent_info["sentence"], sent_info, cluster_tokens)
                                        st.markdown(html, unsafe_allow_html=True)
                                else:
                                    st.info(f"No sentences found containing the tokens in this cluster")
                            else:
                                # Original single token display logic
                                st.write(f"#### Context Sentences for '{token}'")
                                seen_sentences = set()
                                relevant_sentences = []
                                
                                for sent_info in sentences[selected_cluster]:
                                    if (sent_info["token"] == token and 
                                        sent_info["sentence"] not in seen_sentences):
                                        relevant_sentences.append(sent_info)
                                        seen_sentences.add(sent_info["sentence"])
                                
                                if relevant_sentences:
                                    for sent_info in relevant_sentences:
                                        html = create_sentence_html(sent_info["sentence"], sent_info, cluster_tokens)
                                        st.markdown(html, unsafe_allow_html=True)
                                else:
                                    st.info(f"No sentences found containing '{token}' in this cluster")
                            
                            # Show other sentences in the cluster
                            st.write("#### Other Context Sentences in Cluster")
                            other_sentences = []
                            
                            for sent_info in sentences[selected_cluster]:
                                if sent_info["sentence"] not in seen_sentences:
                                    other_sentences.append(sent_info)
                                    seen_sentences.add(sent_info["sentence"])
                            
                            if other_sentences:
                                for sent_info in other_sentences:
                                    html = create_sentence_html(sent_info["sentence"], sent_info, cluster_tokens)
                                    st.markdown(html, unsafe_allow_html=True)
                            else:
                                st.info("No additional unique sentences in this cluster")
            else:
                if second_token:
                    st.info(f"No clusters containing both '{token}' and '{second_token}' found in Layer {layer}")
                else:
                    st.info(f"No clusters containing '{token}' found in Layer {layer}")

if __name__ == "__main__":
    main()
