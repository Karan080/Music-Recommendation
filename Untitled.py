#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image 
import base64
import streamlit as st
import pickle
#import joblib

import os
import numpy as np
import pandas as pd
import time

#import seaborn as sns
#import plotly.express as px 
#import matplotlib.pyplot as plt

#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#from sklearn.metrics import euclidean_distances
#from scipy.spatial.distance import cdist




import os
#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials
#from collections import defaultdict

#from sklearn.metrics import euclidean_distances
#from scipy.spatial.distance import cdist
#import difflib

import streamlit as st
#from streamlit_modal import Modal
import streamlit.components.v1 as components

#from musixmatch import Musixmatch

st.set_page_config(page_title="Music Recommendation", page_icon="🎵", layout='wide') 

# @st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


spotify = Image.open('spotify.png')
empty1, spot, title, empty = st.columns([1.1,.2,2,1.1])
spot.image(spotify, width = 150)
title.markdown('<div style="text-align: center;font-family: Sans serif; font-size:62px; color:green;">Music Recommendation</div>', unsafe_allow_html=True)
#st.title("Music Recommendation")


buff, col_1, buff2 = st.columns([1,2,1])
song_name = col_1.text_input(label = 'Enter a song', value = "",placeholder  = 'Enter a song', label_visibility = 'hidden')

st.divider()

buff5, col_3, buff6 = st.columns([1,2,1])
col_3.markdown('<div style="font-size:35px;font-family: Sans serif; color:white;">Select release year</div>',unsafe_allow_html=True)
my_range = range(1921,2021)


buff3, col_2, buff4 = st.columns([1,2,1])
year = col_2.select_slider(label = "Choose a number", options = my_range, label_visibility = 'hidden')
#st.write('Release year: ', year)

song_selected = [{"name": song_name,"year": year}]

#scr : https://towardsdatascience.com/how-to-use-streamlits-st-write-function-to-improve-your-streamlit-dashboard-1586333eb24d
#st.write('<p style="font-size:26px; color:red;"> 'Song :' song_name </p>',unsafe_allow_html=True)
#st.write('<p style="color:red;">'Song :' song_name </p>',unsafe_allow_html=True)



st.markdown(
    """
    <style>
    input {
        font-size: 2rem !important;
        
    }
    
    .big-font {
    font-size: 2rem !important;
    }
    div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
    height : 50px;
    width : 150px;
    font-size : 10rem !important;
    }
    
    """,
    unsafe_allow_html=True,
)


buff7, col_4, buff8 = st.columns([10,3,10])
search = col_4.button('Search')


if search:

    if song_name:

        with st.spinner("Just a moment ..."):
            time.sleep(1)
            #model = open('recom_model.pkl','rb')
            #test = joblib.load(model)

            data = pd.read_csv("data.csv")
            song_selected = [{"name": song_name,"year": year}]


            # Building model
            song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                          ('kmeans', KMeans(n_clusters=20, 
                                           verbose=False))
                                         ], verbose=False)

            X = data.select_dtypes(np.number)
            number_cols = list(X.columns)
            song_cluster_pipeline.fit(X)
            song_cluster_labels = song_cluster_pipeline.predict(X)
            data['cluster_label'] = song_cluster_labels

            sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='4402224b44c74fec98742f1a8407d7f3',
                                                                  client_secret='b67cf13a0eb24f2e8db6703c9572ba84'))


            def find_song(name, year):
                song_data = defaultdict()
                results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
                if results['tracks']['items'] == []:
                    return None

                results = results['tracks']['items'][0]
                track_id = results['id']
                audio_features = sp.audio_features(track_id)[0]

                song_data['name'] = [name]
                song_data['year'] = [year]
                song_data['explicit'] = [int(results['explicit'])]
                song_data['duration_ms'] = [results['duration_ms']]
                song_data['popularity'] = [results['popularity']]

                for key, value in audio_features.items():
                    song_data[key] = value

                return pd.DataFrame(song_data)

            number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                            'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']



            def get_song_data(song, spotify_data):

                try:
                    song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                            & (spotify_data['year'] == song['year'])].iloc[0]
                    return song_data

                except IndexError:
                    return find_song(song['name'], song['year'])


            def get_mean_vector(song_list, spotify_data):

                song_vectors = []

                for song in song_list:
                    song_data = get_song_data(song, spotify_data)
                    if song_data is None:
                        print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                        continue
                    song_vector = song_data[number_cols].values
                    song_vectors.append(song_vector)  

                song_matrix = np.array(list(song_vectors))
                return np.mean(song_matrix, axis=0)


            def flatten_dict_list(dict_list):

                flattened_dict = defaultdict()
                for key in dict_list[0].keys():
                    flattened_dict[key] = []

                for dictionary in dict_list:
                    for key, value in dictionary.items():
                        flattened_dict[key].append(value)

                return flattened_dict


            def recommend_songs( song_list, spotify_data, n_songs=4):

                metadata_cols = ['name', 'year', 'artists']
                song_dict = flatten_dict_list(song_list)

                song_center = get_mean_vector(song_list, spotify_data)
                scaler = song_cluster_pipeline.steps[0][1]
                scaled_data = scaler.transform(spotify_data[number_cols])
                scaled_song_center = scaler.transform(song_center.reshape(1, -1))
                distances = cdist(scaled_song_center, scaled_data, 'cosine')
                index = list(np.argsort(distances)[:, :n_songs][0])

                rec_songs = spotify_data.iloc[index]
                rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
                return rec_songs[metadata_cols].to_dict(orient='records')

            lyrics_list = pd.DataFrame(recommend_songs(song_selected,  data))

            musixmatch = Musixmatch('e5d59669284675aa6d3e5e9dfb867b2c')

            lst = []
            for index, row in lyrics_list.iterrows():
                sng = musixmatch.matcher_lyrics_get(row['name'], row['artists'])
                lst.append(sng["message"]["body"]["lyrics"]["lyrics_body"].split('\n'))

            lyrics_list["lyrics"] = -1 
            for i in range(len(lst)):
                lyrics_list["lyrics"].iloc[i] = lst[i]


        #st.write(lyrics_list)

    else:
        st.warning('Enter a song', icon="🎵")
    #st.text_area(label = "no song selected", placeholder="Enter a song to search", label_visibility = 'hidden')



col1, col2, col3, col4 = st.columns([10,10,10,10])

original = Image.open('music.jpg')
col1.header(lyrics_list['name'][0][0:27])
#col1.image(original, use_column_width=True)
col1.image("https://upload.wikimedia.org/wikipedia/en/b/b4/Shape_Of_You_%28Official_Single_Cover%29_by_Ed_Sheeran.png?20170317220726",use_column_width=True)

col2.header(lyrics_list['name'][1][0:27])
col2.image(original, use_column_width=True)

col3.header(lyrics_list['name'][2][0:27])
col3.image("https://i.scdn.co/image/ab67616d0000b273e572168cd6db773ee20b3619", use_column_width=True)


col4.header(lyrics_list['name'][3][0:27])
col4.image("https://upload.wikimedia.org/wikipedia/en/0/02/Rauw_Alejandro%2C_Chencho_Corleone_-_El_Efecto.jpeg",use_column_width=True)
#col4.image(original, use_column_width=True)


hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

col6, col7, col8, col9 = st.columns([10,10,10,10])
#col6 = st.container()

with col6:
    col6.write(lyrics_list['lyrics'][0])
    #col6.write(lyrics_list['lyrics'][0])
with col7:
    col7.write(lyrics_list['lyrics'][1])    

with col8:
    col8.write(lyrics_list['lyrics'][2])
    
with col9:
    col9.write(lyrics_list['lyrics'][3])    
    
    
    
    
    
    
    
    
    
    
    
