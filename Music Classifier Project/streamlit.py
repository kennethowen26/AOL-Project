# app_ui_ux_enhanced.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

st.set_page_config(
    page_title="🎶 Personal Music Classifier AI 🎧",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih modern
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .cluster-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        padding: 12px 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-weight: 600;
        margin: 0 4px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        box-shadow: 0 6px 25px rgba(255, 107, 107, 0.4);
        transform: translateY(-3px);
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
        background: linear-gradient(135deg, #feca57 0%, #ff6b6b 100%);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi untuk Memuat Model dan Metadata ---
@st.cache_resource
def load_model_and_metadata(file_path='music_classifier_model.pkl'):
    """Memuat model K-Means, scaler, dan metadata dari file .pkl."""
    try:
        with open(file_path, 'rb') as f:
            model_metadata = pickle.load(f)
        return model_metadata
    except FileNotFoundError:
        st.error(f"⚠️ Error: File model '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {e}")
        st.stop()

# --- Fungsi Visualisasi Enhanced ---
def create_radar_chart(cluster_data, cluster_name):
    """Membuat radar chart untuk karakteristik cluster."""
    features = list(cluster_data.columns)
    values = list(cluster_data.iloc[0])
    
    # Normalisasi nilai untuk radar chart (0-1)
    normalized_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],  # Tutup lingkaran
        theta=features + [features[0]],
        fill='toself',
        name=cluster_name,
        line=dict(color='#667eea'),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Profil Audio: {cluster_name}",
        title_x=0.5
    )
    
    return fig

def create_cluster_scatter_plot(df_with_clusters, numeric_features):
    """Membuat scatter plot interaktif untuk visualisasi cluster."""
    # Gunakan PCA untuk reduksi dimensi jika diperlukan
    from sklearn.decomposition import PCA
    
    if len(numeric_features) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(df_with_clusters[numeric_features])
        df_plot = df_with_clusters.copy()
        df_plot['PC1'] = coords[:, 0]
        df_plot['PC2'] = coords[:, 1]
        x_col, y_col = 'PC1', 'PC2'
        x_title, y_title = f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
    else:
        df_plot = df_with_clusters.copy()
        x_col, y_col = numeric_features[0], numeric_features[1] if len(numeric_features) > 1 else numeric_features[0]
        x_title, y_title = x_col.replace('_', ' ').title(), y_col.replace('_', ' ').title()
    
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color='cluster',
        hover_data=['track_name', 'artist_name', 'genre'],
        title="Distribusi Cluster dalam Ruang Fitur",
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=600
    )
    
    return fig

def create_genre_sunburst(cluster_genre_distribution, display_cluster_labels):
    """Membuat sunburst chart untuk distribusi genre per cluster."""
    data = []
    
    for cluster_id, genres in cluster_genre_distribution.iterrows():
        cluster_name = display_cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        top_genres = genres.sort_values(ascending=False).head(5)
        
        for genre, proportion in top_genres.items():
            if proportion > 0.01:  # Hanya tampilkan genre dengan proporsi > 1%
                data.append({
                    'cluster': cluster_name,
                    'genre': genre,
                    'proportion': proportion
                })
    
    df_sunburst = pd.DataFrame(data)
    
    if not df_sunburst.empty:
        fig = px.sunburst(
            df_sunburst,
            path=['cluster', 'genre'],
            values='proportion',
            title="Distribusi Genre per Cluster",
            color='proportion',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        return fig
    return None

def create_cluster_heatmap(cluster_centers_unscaled_df, display_cluster_labels):
    """Membuat heatmap untuk perbandingan karakteristik antar cluster."""
    df_heatmap = cluster_centers_unscaled_df.copy()
    df_heatmap['Aktivitas'] = df_heatmap['cluster_id'].map(display_cluster_labels)
    df_heatmap = df_heatmap.set_index('Aktivitas').drop(columns=['cluster_id'])
    
    # Normalisasi untuk heatmap
    df_normalized = (df_heatmap - df_heatmap.min()) / (df_heatmap.max() - df_heatmap.min())
    
    fig = px.imshow(
        df_normalized.values,
        x=df_normalized.columns,
        y=df_normalized.index,
        color_continuous_scale='viridis',
        title="Heatmap Karakteristik Cluster (Normalized)",
        aspect="auto"
    )
    
    fig.update_layout(
        xaxis_title="Fitur Audio",
        yaxis_title="Aktivitas",
        height=400
    )
    
    return fig

def show_onboarding():
    """Personal Music Profile Onboarding"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 2rem 0; color: white;">
        <h2>🎵 Let's Get to Know Your Music Taste!</h2>
        <p>Help us personalize your music experience in just a few steps</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("user_onboarding"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 About You")
            user_name = st.text_input("What's your name?", placeholder="Enter your name")
            
            st.subheader("🎵 Music Preferences")
            # Get available genres from dataset
            available_genres = sorted(df_with_clusters['genre'].unique())
            fav_genres = st.multiselect(
                "Your favorite genres (select 3-5):",
                options=available_genres,
                help="This helps us understand your taste"
            )
            
        
        with col2:
            st.subheader("🎭 Your Vibe")
            mood_preference = st.radio(
                "I usually prefer music that is:",
                ["Energetic & Upbeat", "Calm & Relaxed", "Emotional & Deep", "Mixed - depends on mood"]
            )
            
            activity_prefs = st.multiselect(
                "I listen to music while:",
                ["Working/Studying", "Exercising", "Relaxing", "Driving", "Partying", "Sleeping", "Cooking"]
            )
            
        
        submitted = st.form_submit_button("🚀 Start My Music Journey", use_container_width=True)
        
        if submitted and user_name and fav_genres:
            # Save user profile
            st.session_state.user_profile = {
                'name': user_name,
                'favorite_genres': fav_genres,
                'mood_preference': mood_preference,
                'activity_preferences': activity_prefs
            }
            st.session_state.onboarding_complete = True
            st.rerun()
        elif submitted:
            st.warning("Please fill in your name and select at least one favorite genre!")

def add_to_playlist(song):
    """Add song to user's playlist - WITHOUT st.rerun()"""
    if 'user_playlist' not in st.session_state:
        st.session_state.user_playlist = []
    
    # Check if song already exists
    existing = any(
        s['track_name'] == song['track_name'] and s['artist_name'] == song['artist_name'] 
        for s in st.session_state.user_playlist
    )
    
    if not existing:
        st.session_state.user_playlist.append({
            'track_name': song['track_name'],
            'artist_name': song['artist_name'],
            'genre': song['genre'],
            'energy': song.get('energy', 0),
            'valence': song.get('valence', 0),
            'danceability': song.get('danceability', 0)
        })
        return True  # Successfully added
    return False  # Already exists

def analyze_user_playlist(playlist):
    """Analyze user's playlist and show insights"""
    if not playlist:
        return
    
    # Convert to DataFrame for analysis
    playlist_df = pd.DataFrame(playlist)
    
    st.subheader("📊 Your Music Taste Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average characteristics
        avg_energy = playlist_df['energy'].mean()
        avg_valence = playlist_df['valence'].mean()
        avg_dance = playlist_df['danceability'].mean()
        
        st.metric("Average Energy", f"{avg_energy:.2f}", help="How intense your music taste is")
        st.metric("Average Positivity", f"{avg_valence:.2f}", help="How upbeat your music is")
        st.metric("Average Danceability", f"{avg_dance:.2f}", help="How danceable your playlist is")
        
        # Music personality
        personality = get_music_personality(avg_energy, avg_valence, avg_dance)
        st.info(f"🎭 **Your Music Personality:** {personality}")
    
    with col2:
        # Genre distribution
        genre_counts = playlist_df['genre'].value_counts()
        
        if len(genre_counts) > 0:
            fig = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Your Genre Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def get_music_personality(energy, valence, dance):
    """Determine music personality based on averages"""
    if energy > 0.7 and valence > 0.7:
        return "🎉 Party Animal - You love energetic, upbeat music!"
    elif energy < 0.4 and valence < 0.5:
        return "🌙 Contemplative Soul - You prefer deep, emotional music"
    elif energy > 0.6 and dance > 0.7:
        return "💃 Dance Enthusiast - You can't resist a good beat!"
    elif energy < 0.5 and valence > 0.6:
        return "☀️ Chill Vibes - You enjoy relaxed, positive music"
    elif valence < 0.4:
        return "🎭 Emotional Explorer - You connect with melancholic sounds"
    else:
        return "🎵 Balanced Listener - You enjoy diverse musical experiences"

def find_mood_based_music(mood, energy, valence, dance):
    """Find music based on mood and parameters - WITH PERSISTENT STATE"""
    user = st.session_state.user_profile
    
    # Filter songs based on mood parameters
    filtered_songs = df_with_clusters.copy()
    
    # Apply mood preference bias
    mood_pref = user.get('mood_preference', '')
    if mood_pref == "Energetic & Upbeat":
        # Boost energy and valence
        energy = min(1.0, energy + 0.1)
        valence = min(1.0, valence + 0.1)
    elif mood_pref == "Calm & Relaxed":
        # Lower energy, maintain valence
        energy = max(0.0, energy - 0.2)
    elif mood_pref == "Emotional & Deep":
        # Lower valence, variable energy
        valence = max(0.0, valence - 0.2)
    # "Mixed - depends on mood" = no adjustment

    # Apply mood-based filtering
    if 'energy' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['energy'] >= energy - 0.2) & 
            (filtered_songs['energy'] <= energy + 0.2)
        ]
    
    if 'valence' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['valence'] >= valence - 0.2) & 
            (filtered_songs['valence'] <= valence + 0.2)
        ]
    
    if 'danceability' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['danceability'] >= dance - 0.2) & 
            (filtered_songs['danceability'] <= dance + 0.2)
        ]
    
    # Prefer user's favorite genres
    if user['favorite_genres']:
        genre_boost = filtered_songs[filtered_songs['genre'].isin(user['favorite_genres'])]
        if not genre_boost.empty:
            # Mix genre preferences with mood-based results
            genre_songs = genre_boost.sample(n=min(5, len(genre_boost)), random_state=42)
            other_songs = filtered_songs[~filtered_songs['genre'].isin(user['favorite_genres'])]
            if not other_songs.empty:
                other_songs = other_songs.sample(n=min(5, len(other_songs)), random_state=42)
                filtered_songs = pd.concat([genre_songs, other_songs])
            else:
                filtered_songs = genre_songs
    
    if not filtered_songs.empty:
        st.success(f"🎉 Found {len(filtered_songs)} songs perfect for your {mood} mood!")
        
        # STORE RECOMMENDATIONS IN SESSION STATE
        sample_size = min(8, len(filtered_songs))
        recommended_songs = filtered_songs.drop_duplicates(subset=['track_name', 'artist_name']).sample(
            n=sample_size, random_state=42
        )
        
        # Save to session state to persist across reruns
        st.session_state.current_recommendations = recommended_songs.to_dict('records')
        
        # Display recommendations
        display_recommendations(st.session_state.current_recommendations, source="mood")
    else:
        st.warning("No songs found for your current mood. Try adjusting the sliders!")


from hashlib import md5

def display_recommendations(recommendations, source="mood"):
    for idx, song in enumerate(recommendations):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"""
                🎧 **{song['track_name']}**  
                by *{song['artist_name']}* • {song['genre']}  
                Energy: {song.get('energy', 0):.1f} | Valence: {song.get('valence', 0):.1f} | Dance: {song.get('danceability', 0):.1f}
            """, unsafe_allow_html=True)

        with col2:
            track_id = f"{song['track_name']}_{song['artist_name']}"
            stable_id = md5(track_id.encode()).hexdigest()[:10]
            button_key = f"add_{source}_{idx}_{stable_id}"
            success_key = f"added_{source}_{idx}_{stable_id}"

            if st.button("➕ Add to Playlist", key=button_key):
                success = add_to_playlist(song)
                st.session_state[success_key] = "added" if success else "duplicate"

            if st.session_state.get(success_key) == "added":
                st.success("✅ Added!")
            elif st.session_state.get(success_key) == "duplicate":
                st.warning("⚠️ Already in playlist!")



def show_main_app():
    """Main application with personalized experience"""
    user = st.session_state.user_profile
    
    # Personalized welcome
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; color: white;">
        <h3>Welcome back, {user['name']}! 🎵</h3>
        <p>Your music taste: {', '.join(user['favorite_genres'][:3])} • Preferred vibe: {user['mood_preference']}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs dengan Enhanced Features ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Mood Selector", "🎵 Smart Recommendations", "🔎 Song Classifier", "📊 Cluster Analysis", "🎧 My Playlist"])

    with tab1:
        st.header("🎨 Visual Mood Selector")
        st.markdown("Tell us how you're feeling, and we'll find the perfect music for you!")
        
        # Visual Mood Selector
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Mood Emoji Selector
            st.subheader("How are you feeling right now?")
            mood_options = {
                "😴 Sleepy/Relaxed": {"energy": (0.0, 0.3), "valence": (0.0, 0.5), "danceability": (0.0, 0.4)},
                "😊 Happy/Upbeat": {"energy": (0.6, 1.0), "valence": (0.7, 1.0), "danceability": (0.6, 1.0)},
                "😤 Energetic/Workout": {"energy": (0.8, 1.0), "valence": (0.5, 1.0), "danceability": (0.7, 1.0)},
                "😔 Sad/Melancholic": {"energy": (0.0, 0.4), "valence": (0.0, 0.3), "danceability": (0.0, 0.4)},
                "🤔 Focus/Study": {"energy": (0.3, 0.7), "valence": (0.4, 0.8), "danceability": (0.2, 0.6)},
                "🎉 Party/Social": {"energy": (0.7, 1.0), "valence": (0.8, 1.0), "danceability": (0.8, 1.0)},
                "🌅 Chill/Morning": {"energy": (0.4, 0.7), "valence": (0.6, 0.9), "danceability": (0.3, 0.7)},
                "🌙 Night/Intimate": {"energy": (0.2, 0.5), "valence": (0.3, 0.7), "danceability": (0.2, 0.5)}
            }
            
            # Create mood buttons in a grid
            mood_cols = st.columns(2)
            selected_mood = None
            
            for i, (mood, params) in enumerate(mood_options.items()):
                with mood_cols[i % 2]:
                    if st.button(mood, key=f"mood_{i}", use_container_width=True):
                        selected_mood = mood
                        st.session_state.selected_mood = mood
                        st.session_state.mood_params = params
            
            # Use previously selected mood if available
            if 'selected_mood' in st.session_state and selected_mood is None:
                selected_mood = st.session_state.selected_mood
            
            if selected_mood:
                st.success(f"Selected mood: **{selected_mood}**")
                
                # Fine-tune controls
                st.subheader("🎛️ Fine-tune Your Vibe")
                
                mood_params = st.session_state.get('mood_params', mood_options[selected_mood])
                
                col_energy, col_valence, col_dance = st.columns(3)
                
                with col_energy:
                    energy_level = st.slider(
                        "Energy Level",
                        0.0, 1.0, 
                        (mood_params['energy'][0] + mood_params['energy'][1]) / 2,
                        help="How intense should the music be?"
                    )
                
                with col_valence:
                    positivity = st.slider(
                        "Positivity",
                        0.0, 1.0,
                        (mood_params['valence'][0] + mood_params['valence'][1]) / 2,
                        help="How upbeat/positive should it sound?"
                    )
                
                with col_dance:
                    danceability = st.slider(
                        "Danceability",
                        0.0, 1.0,
                        (mood_params['danceability'][0] + mood_params['danceability'][1]) / 2,
                        help="How danceable should it be?"
                    )
                
                
                # Generate recommendations
                if st.button("🎵 Find My Perfect Music", use_container_width=True):
                    find_mood_based_music(selected_mood, energy_level, positivity, danceability)
        
        # Display current recommendations if they exist
        if 'current_recommendations' in st.session_state:
            st.subheader("🎵 Your Current Recommendations")
            display_recommendations(st.session_state.current_recommendations, source="mood")

    with tab2:
        st.header("🎵 Smart Music Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_activity = st.selectbox(
                "Pilih mood atau aktivitas:",
                options=list(activity_to_cluster_id.keys()),
                key='activity_selector'
            )
            
            # Advanced filters
            st.subheader("🎛️ Filter Lanjutan")
            
            if 'danceability' in numeric_features:
                dance_range = st.slider(
                    "Tingkat Danceability",
                    0.0, 1.0, (0.0, 1.0),
                    help="Seberapa cocok lagu untuk menari"
                )
            
            if 'energy' in numeric_features:
                energy_range = st.slider(
                    "Tingkat Energy",
                    0.0, 1.0, (0.0, 1.0),
                    help="Intensitas dan kekuatan lagu"
                )
            
            # Genre filter
            available_genres = df_with_clusters['genre'].unique()
            selected_genres = st.multiselect(
                "Filter Genre (opsional)",
                options=available_genres,
                default=None
            )

        with col2:
            if selected_activity:
                cluster_id = activity_to_cluster_id.get(selected_activity)
                
                if cluster_id is not None:
                    st.success(f"✨ Aktivitas: **{selected_activity}**")
                    
                    # Radar Chart
                    cluster_data = cluster_centers_unscaled_df[
                        cluster_centers_unscaled_df['cluster_id'] == cluster_id
                    ].drop(columns=['cluster_id'])
                    
                    if not cluster_data.empty:
                        radar_fig = create_radar_chart(cluster_data, selected_activity)
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Generate and store smart recommendations
                    if st.button("🎵 Generate Smart Recommendations", use_container_width=True):
                        # Filtered Recommendations
                        songs_in_cluster = df_with_clusters[df_with_clusters['cluster'] == cluster_id].copy()
                        
                        # Apply filters
                        if 'danceability' in numeric_features:
                            songs_in_cluster = songs_in_cluster[
                                (songs_in_cluster['danceability'] >= dance_range[0]) &
                                (songs_in_cluster['danceability'] <= dance_range[1])
                            ]
                        
                        if 'energy' in numeric_features:
                            songs_in_cluster = songs_in_cluster[
                                (songs_in_cluster['energy'] >= energy_range[0]) &
                                (songs_in_cluster['energy'] <= energy_range[1])
                            ]
                        
                        if selected_genres:
                            songs_in_cluster = songs_in_cluster[
                                songs_in_cluster['genre'].isin(selected_genres)
                            ]
                        
                        if not songs_in_cluster.empty:
                            # Sampling rekomendasi dengan diversitas
                            sample_songs = songs_in_cluster.drop_duplicates(
                                subset=['track_name', 'artist_name']
                            ).sample(
                                n=min(8, len(songs_in_cluster.drop_duplicates(subset=['track_name', 'artist_name']))),
                                random_state=42
                            )
                            
                            # Store in session state
                            st.session_state.smart_recommendations = sample_songs.to_dict('records')
                            st.success(f"🎉 Generated {len(sample_songs)} recommendations!")
                        else:
                            st.info("Tidak ada lagu yang sesuai dengan filter yang dipilih.")
                    
                    # Display smart recommendations if they exist
                    if 'smart_recommendations' in st.session_state:
                        st.subheader(f"🎶 Rekomendasi untuk {selected_activity}")
                        display_recommendations(st.session_state.smart_recommendations, source="smart")

    with tab3:
        st.header("🔎 Advanced Song Classifier")
        
        # Real-time prediction dengan visualisasi
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Fitur Audio")
            
            feature_min_max = {
                feature: (df_with_clusters[feature].min(), df_with_clusters[feature].max())
                for feature in numeric_features
            }

            input_features = {}
            
            for feature in numeric_features:
                min_val, max_val = feature_min_max.get(feature, (0.0, 1.0))
                default_val = float(df_with_clusters[feature].mean())
                
                if max_val - min_val > 100:
                    step_val = 1.0
                elif max_val - min_val > 10:
                    step_val = 0.1
                else:
                    step_val = 0.01

                input_features[feature] = st.slider(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=default_val,
                    step=step_val,
                    help=f"Rentang dataset: {min_val:.2f} - {max_val:.2f}"
                )

        with col2:
            st.subheader("Prediksi Real-time")
            
            # Real-time prediction
            input_df = pd.DataFrame([input_features])
            
            try:
                scaled_input = scaler.transform(input_df[numeric_features])
                predicted_cluster_id = kmeans_model.predict(scaled_input)[0]
                predicted_activity = display_cluster_labels.get(np.int64(predicted_cluster_id), "Tidak Diketahui")
                
                # Confidence calculation
                distances = kmeans_model.transform(scaled_input)[0]
                confidence = 1 - (min(distances) / max(distances)) if max(distances) > 0 else 1
                
                st.success(f"🎯 **Prediksi: {predicted_activity}**")
                st.info(f"Confidence: {confidence:.1%}")
                
                # Progress bars untuk probabilitas
                st.subheader("Similarity Score")
                cluster_distances = kmeans_model.transform(scaled_input)[0]
                normalized_scores = 1 - (cluster_distances / max(cluster_distances))
                
                for i, score in enumerate(normalized_scores):
                    activity_name = display_cluster_labels.get(i, f"Cluster {i}")
                    st.write(f"{activity_name}")
                    st.progress(float(score))
                
                # Visualization of input vs cluster centers
                input_radar_data = pd.DataFrame([list(input_features.values())], columns=numeric_features)
                input_radar_fig = create_radar_chart(input_radar_data, "Input Lagu")
                st.plotly_chart(input_radar_fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error dalam prediksi: {e}")

    with tab4:
        st.header("📊 Advanced Cluster Analysis")
        
        # Cluster comparison heatmap
        st.subheader("🌡️ Perbandingan Karakteristik Cluster")
        heatmap_fig = create_cluster_heatmap(cluster_centers_unscaled_df, display_cluster_labels)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Genre distribution sunburst
        st.subheader("🎭 Distribusi Genre per Cluster")
        sunburst_fig = create_genre_sunburst(cluster_genre_distribution, display_cluster_labels)
        if sunburst_fig:
            st.plotly_chart(sunburst_fig, use_container_width=True)
        
        # Detailed cluster stats
        st.subheader("📈 Statistik Detail Cluster")
        
        for cluster_id in sorted(display_cluster_labels.keys()):
            activity_name = display_cluster_labels[cluster_id]
            songs_count = len(df_with_clusters[df_with_clusters['cluster'] == cluster_id])
            
            with st.expander(f"📊 {activity_name} ({songs_count} lagu)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Karakteristik rata-rata
                    char_df = cluster_centers_unscaled_df[
                        cluster_centers_unscaled_df['cluster_id'] == cluster_id
                    ].drop(columns=['cluster_id'])
                    
                    if not char_df.empty:
                        # Bar chart untuk karakteristik
                        char_melted = char_df.T.reset_index()
                        char_melted.columns = ['Feature', 'Value']
                        
                        fig = px.bar(
                            char_melted,
                            x='Feature',
                            y='Value',
                            title=f'Karakteristik Audio: {activity_name}',
                            color='Value',
                            color_continuous_scale='viridis'
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top genres
                    if cluster_id in cluster_genre_distribution.index:
                        top_genres = cluster_genre_distribution.loc[cluster_id].sort_values(ascending=False).head(8)
                        
                        fig = px.pie(
                            values=top_genres.values,
                            names=top_genres.index,
                            title=f'Distribusi Genre: {activity_name}'
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.header("🎧 My Personal Playlist")
        
        if 'user_playlist' in st.session_state and st.session_state.user_playlist:
            # Playlist analysis
            analyze_user_playlist(st.session_state.user_playlist)
            
            st.subheader("🎵 Your Saved Songs")
            
            # Display playlist
            for i, song in enumerate(st.session_state.user_playlist):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: #ff9800; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                        <strong>{song['track_name']}</strong><br>
                        <small>by {song['artist_name']} • {song['genre']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.user_playlist.pop(i)
                        st.rerun()
            
            # Playlist controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 Export Playlist", use_container_width=True):
                    playlist_df = pd.DataFrame(st.session_state.user_playlist)
                    csv = playlist_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download as CSV",
                        data=csv,
                        file_name=f"{user['name']}_playlist.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear Playlist", use_container_width=True):
                    st.session_state.user_playlist = []
                    st.rerun()
        else:
            st.info("🎵 Your playlist is empty. Add some songs from the recommendations!")
            
            # Quick add popular songs
            st.subheader("🔥 Popular Songs to Get Started")
            popular_songs = df_with_clusters.sample(n=5, random_state=42)
            
            for idx, row in popular_songs.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                        <strong>🎧 {row['track_name']}</strong><br>
                        <small>by {row['artist_name']} • {row['genre']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("➕ Add", key=f"popular_{idx}"):
                        add_to_playlist(row)
                        st.success("Added!")


# FIXED FUNCTIONS - Perbaikan utama ada di sini
def add_to_playlist(song):
    """Add song to user's playlist - FIXED VERSION without st.rerun()"""
    if 'user_playlist' not in st.session_state:
        st.session_state.user_playlist = []
    
    # Check if song already exists
    existing = any(
        s['track_name'] == song['track_name'] and s['artist_name'] == song['artist_name'] 
        for s in st.session_state.user_playlist
    )
    
    if not existing:
        st.session_state.user_playlist.append({
            'track_name': song['track_name'],
            'artist_name': song['artist_name'],
            'genre': song['genre'],
            'energy': song.get('energy', 0),
            'valence': song.get('valence', 0),
            'danceability': song.get('danceability', 0)
        })
        return True  # Successfully added
    return False  # Already exists

def find_mood_based_music(mood, energy, valence, dance):
    """Find music based on mood and parameters - IMPROVED VERSION"""
    user = st.session_state.user_profile
    
    # Filter songs based on mood parameters
    filtered_songs = df_with_clusters.copy()
    
    # Apply mood preference bias
    mood_pref = user.get('mood_preference', '')
    if mood_pref == "Energetic & Upbeat":
        # Boost energy and valence
        energy = min(1.0, energy + 0.1)
        valence = min(1.0, valence + 0.1)
    elif mood_pref == "Calm & Relaxed":
        # Lower energy, maintain valence
        energy = max(0.0, energy - 0.2)
    elif mood_pref == "Emotional & Deep":
        # Lower valence, variable energy
        valence = max(0.0, valence - 0.2)
    # "Mixed - depends on mood" = no adjustment

    # Apply mood-based filtering
    if 'energy' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['energy'] >= energy - 0.2) & 
            (filtered_songs['energy'] <= energy + 0.2)
        ]
    
    if 'valence' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['valence'] >= valence - 0.2) & 
            (filtered_songs['valence'] <= valence + 0.2)
        ]
    
    if 'danceability' in filtered_songs.columns:
        filtered_songs = filtered_songs[
            (filtered_songs['danceability'] >= dance - 0.2) & 
            (filtered_songs['danceability'] <= dance + 0.2)
        ]
    
    # Prefer user's favorite genres
    if user['favorite_genres']:
        genre_boost = filtered_songs[filtered_songs['genre'].isin(user['favorite_genres'])]
        if not genre_boost.empty:
            # Mix genre preferences with mood-based results
            genre_songs = genre_boost.sample(n=min(5, len(genre_boost)), random_state=42)
            other_songs = filtered_songs[~filtered_songs['genre'].isin(user['favorite_genres'])]
            if not other_songs.empty:
                other_songs = other_songs.sample(n=min(5, len(other_songs)), random_state=42)
                filtered_songs = pd.concat([genre_songs, other_songs])
            else:
                filtered_songs = genre_songs
    
    if not filtered_songs.empty:
        # Create a unique key for this mood search to prevent conflicts
        mood_key = f"{mood}_{energy:.1f}_{valence:.1f}_{dance:.1f}"
        
        # STORE RECOMMENDATIONS IN SESSION STATE with unique key
        sample_size = min(8, len(filtered_songs))
        recommended_songs = filtered_songs.drop_duplicates(subset=['track_name', 'artist_name']).sample(
            n=sample_size, random_state=42
        )
        
        # Save to session state to persist across interactions
        st.session_state.current_recommendations = recommended_songs.to_dict('records')
        st.session_state.current_mood_key = mood_key
        
        st.success(f"🎉 Found {len(recommended_songs)} songs perfect for your {mood} mood!")
        
        # Display recommendations
        #display_recommendations(st.session_state.current_recommendations, source="mood")
    else:
        st.warning("No songs found for your current mood. Try adjusting the sliders!")


# Load model and data
model_metadata = load_model_and_metadata()

# Extract components with safe access
kmeans_model = model_metadata.get('kmeans_model')
scaler = model_metadata.get('scaler')
df_with_clusters = model_metadata.get('df_with_clusters')
numeric_features = model_metadata.get('numeric_features', [])
cluster_centers_unscaled_df = model_metadata.get('cluster_centers_unscaled_df')
cluster_genre_distribution = model_metadata.get('cluster_genre_distribution')

# Fixed cluster labels for 5 clusters
display_cluster_labels = {
    0: 'Instrumental / Cinematic',
    1: 'Dance / Groove', 
    2: 'Upbeat / Alternative',
    3: 'Vocal / Dramatic',
    4: 'Podcast / Spoken Word'
}

# Override with model labels if available but ensure we have 5 clusters
if 'display_cluster_labels' in model_metadata:
    model_labels = model_metadata['display_cluster_labels']
    # Only use model labels if it has exactly 5 clusters
    if len(model_labels) == 5:
        display_cluster_labels = model_labels

# Create activity mapping from display labels
activity_to_cluster_id = {label: cluster_id for cluster_id, label in display_cluster_labels.items()}

# Override with model mapping if available
if 'activity_to_cluster_id' in model_metadata:
    model_mapping = model_metadata['activity_to_cluster_id']
    # Only use if it matches our 5 cluster structure
    if len(model_mapping) == 5:
        activity_to_cluster_id = model_mapping

# --- Main App Logic ---
# Initialize session state
if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False

if not st.session_state.onboarding_complete:
    show_onboarding()
else:
    show_main_app()

# --- Enhanced Sidebar ---
with st.sidebar:
    st.title("🎯 Navigation & Info")
    
    # User profile summary
    if 'user_profile' in st.session_state:
        user = st.session_state.user_profile
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong>👤 {user['name']}</strong><br>
            <small>🎵 {', '.join(user['favorite_genres'][:2])}</small><br>
            <small>🎭 {user['mood_preference']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state.onboarding_complete = False
            if 'user_profile' in st.session_state:
                del st.session_state.user_profile
            if 'user_playlist' in st.session_state:
                del st.session_state.user_playlist
            st.rerun()
    
    # Quick stats
    st.subheader("📈 Quick Stats")
    st.write(f"🎵 **{len(df_with_clusters):,}** total songs")
    st.write(f"🎭 **{df_with_clusters['genre'].nunique()}** unique genres")
    st.write(f"👨‍🎤 **{df_with_clusters['artist_name'].nunique():,}** artists")
    st.write(f"🏷️ **{len(display_cluster_labels)}** activity clusters")
    st.write("Silhouette Score: 0.6610")
    st.write("Davies-Bouldin Index: 0.5171")
    st.write("Calinski-Harabasz Index: 202660.7729")

    
    # Playlist stats
    if 'user_playlist' in st.session_state:
        playlist_count = len(st.session_state.user_playlist)
        st.write(f"🎧 **{playlist_count}** songs in playlist")
    
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Information")
    st.info("""
    **Algorithm:** K-Means Clustering
    **Features:** Audio characteristics from Spotify API
    **Clusters:** Automatically grouped by musical similarity
    **Purpose:** Match music to activities/moods
    """)
    
    st.markdown("---")
    
    # Feature explanations
    st.subheader("🎵 Audio Features Guide")
    feature_explanations = {
        'danceability': '💃 How suitable for dancing (0-1)',
        'energy': '⚡ Intensity and power (0-1)',
        'valence': '😊 Musical positivity (0-1)',
        'acousticness': '🎸 Acoustic vs electronic (0-1)',
        'instrumentalness': '🎼 Vocal vs instrumental (0-1)',
        'liveness': '🎤 Live performance feel (0-1)',
        'speechiness': '🗣️ Spoken words presence (0-1)',
        'tempo': '🥁 Speed in BPM',
        'loudness': '🔊 Overall loudness in dB'
    }
    
    for feature, explanation in feature_explanations.items():
        if feature in numeric_features:
            st.write(f"**{feature.title()}:** {explanation}")
    
    st.markdown("---")
    
    # Credits
    st.markdown("""
    ### 💡 About This App
    
    This AI-powered music classifier uses machine learning to:
    - 🎯 Match songs to activities/moods
    - 📊 Analyze musical patterns
    - 🎵 Provide personalized recommendations
    
    **Data Source:** Spotify Web API  
    **Built with:** Streamlit + Plotly + Scikit-learn
    
    ---
    
    Made with ❤️ for music lovers
    """)