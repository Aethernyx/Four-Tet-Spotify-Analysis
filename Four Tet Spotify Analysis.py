#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials


# In[2]:


# Auhtorization details

client_id = 'id'
client_secret = 'secret'


# In[3]:


client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)


# In[4]:


sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[5]:


name = 'Four Tet'


# In[6]:


result = sp.search(name)


# In[7]:


result['tracks']['items'][0]['artists']


# In[8]:


artist_uri = result['tracks']['items'][0]['artists'][0]['uri']


# In[9]:


sp_albums = sp.artist_albums(artist_uri, album_type='album')


# In[ ]:





# In[10]:


album_names = []
album_uris = []

for i in range(len(sp_albums['items'])):
    album_names.append(sp_albums['items'][i]['name'])
    album_uris.append(sp_albums['items'][i]['uri'])


# In[11]:


album_names


# In[12]:


album_uris


# In[13]:




def album_songs(uri):
    album = uri
    
    spotify_albums[album] = {}
    
    #Create keys-values of empty lists inside nested dictionary for album
    spotify_albums[album]['album'] = [] #create empty list
    spotify_albums[album]['track_number'] = []
    spotify_albums[album]['id'] = []
    spotify_albums[album]['name'] = []
    spotify_albums[album]['uri'] = []
    
    tracks = sp.album_tracks(album)
    
    for n in range(len(tracks['items'])): #for each song track
        spotify_albums[album]['album'].append(album_names[album_count]) #append album name tracked via album_count
        spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
        spotify_albums[album]['id'].append(tracks['items'][n]['id'])
        spotify_albums[album]['name'].append(tracks['items'][n]['name'])
        spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])


# In[14]:


spotify_albums = {}

album_count = 0

for i in album_uris:
    album_songs(i)
    print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary")
    album_count+=1 #Updates album count once all tracks have been added


# In[15]:


spotify_albums


# In[16]:


def audio_features(album):
    spotify_albums[album]['acousticness'] = []
    spotify_albums[album]['danceability'] = []
    spotify_albums[album]['energy'] = []
    spotify_albums[album]['instrumentalness'] = []
    spotify_albums[album]['liveness'] = []
    spotify_albums[album]['loudness'] = []
    spotify_albums[album]['speechiness'] = []
    spotify_albums[album]['tempo'] = []
    spotify_albums[album]['valence'] = []
    spotify_albums[album]['popularity'] = []
    spotify_albums[album]['key'] = []
    spotify_albums[album]['mode'] = []
    
    track_count = 0
    for track in spotify_albums[album]['uri']:
        #pull audio features per track
        features = sp.audio_features(str(track))
        
        spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
        spotify_albums[album]['danceability'].append(features[0]['danceability'])
        spotify_albums[album]['energy'].append(features[0]['energy'])
        spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
        spotify_albums[album]['key'].append(features[0]['key'])
        spotify_albums[album]['liveness'].append(features[0]['liveness'])
        spotify_albums[album]['loudness'].append(features[0]['loudness'])
        spotify_albums[album]['mode'].append(features[0]['mode'])
        spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
        spotify_albums[album]['tempo'].append(features[0]['tempo'])
        spotify_albums[album]['valence'].append(features[0]['valence'])
        
        pop = sp.track(track)
        spotify_albums[album]['popularity'].append(pop['popularity'])
        track_count+=1


# In[17]:


import time
import numpy as np
sleep_min = 2
sleep_max = 5
start_time = time.time()
request_count = 0

for i in spotify_albums:
    audio_features(i)
    request_count+=1
    if request_count % 5 == 0:
        print(str(request_count) + " playlists completed")
        time.sleep(np.random.uniform(sleep_min, sleep_max))
        print('Loop #: {}'.format(request_count))
        print('Elapsed Time: {} seconds'.format(time.time() - start_time))


# In[ ]:





# In[18]:


dic_df = {}

dic_df['album'] = []
dic_df['track_number'] = []
dic_df['id'] = []
dic_df['name'] = []
dic_df['uri'] = []
dic_df['acousticness'] = []
dic_df['danceability'] = []
dic_df['energy'] = []
dic_df['instrumentalness'] = []
dic_df['liveness'] = []
dic_df['loudness'] = []
dic_df['speechiness'] = []
dic_df['tempo'] = []
dic_df['valence'] = []
dic_df['popularity'] = []
dic_df['popularity'] = []
dic_df['key'] = []
dic_df['mode'] = []

for album in spotify_albums: 
    for feature in spotify_albums[album]:
        dic_df[feature].extend(spotify_albums[album][feature])


# In[ ]:





# In[19]:


import pandas as pd


# In[20]:


df = pd.DataFrame.from_dict(dic_df)
df


# In[21]:


df.drop(['id', 'uri'], axis=1, inplace=True)


# In[22]:


df.name = df.name.apply(lambda x: x.lower())
df.drop_duplicates(inplace=True)


# In[23]:


df.sort_values('popularity', ascending=True, inplace=True)


# In[24]:


# For each album I want to add the year the album was released. I will obtain the release year for each album
# by using the Spotipy API, and for each album I will put a list of the album name, release year in a list
# named album_release_year

album_release_year = []

for i in range(len(sp_albums['items'])):
    album_release_year.append([sp_albums['items'][i]['name'], sp_albums['items'][i]['release_date'][:4]])


# In[25]:


# I will create a function that will add the release year for each album in a separate column named 'release_year'
# based on what album is listed.

def add_release_year(column):
    for album in album_release_year:
        if column == album[0]:
            return album[1]
            


# In[26]:


df['release_year'] = df['album'].apply(add_release_year)


# In[27]:


# One of Four Tet's albums is caled 'There is Love in You (Expanded Edition)'. Since I don't see a regular edition
# listed for this album I want to remove the '(Expanded Edition)' part from the album name to clean things up.

df['album'].replace('There Is Love in You (Expanded Edition)', 'There Is Love in You', inplace=True)


# In[28]:


df.album.unique()


# In[29]:


# Checking to see how many tracks each album has

df['album'].value_counts()


# Here are the attributes for the features in the DataFrame:
# - Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. 
# 
# 
# - Danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
# 
# 
# - Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# 
# 
# - Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. 
# 
# 
# - Key: 	The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
# 
# 
# - Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. 
# 
# 
# - Loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
# 
# 
# - Mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# 
# 
# - Popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.
# Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note that the popularity value may lag actual popularity by a few days: the value is not updated in real time.
# 
# 
# - Speechiness: 	Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# 
# 
# - Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# 
# 
# - Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
# 

# In[30]:


# I want to look at the distribution of acousticness from Four Tet's Spotify album discography

# Importing Seaborn and matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[31]:


# Plotting the acousticness distribution

plt.figure(figsize=(10, 5))
sns.distplot(df['acousticness'], bins=20, color='blue', kde=False)
plt.xlim(0, 1)


# From the plot above it looks like Four Tet's tracks vary evenly across the acoustice spectrum, with the exception of very unacoustic tracks being an extreme, with a majority of his tracks being considered highly unacoustic.

# In[32]:


# I'm curious if there's a trend between albums and acousticness, so I will group by album, and find the 
# mean acousticness per album. I will also include the release_year, and sort the new dataframe by release_year

acousticness_by_album = df.groupby(['album', 'release_year'])['acousticness'].mean().reset_index()
acousticness_by_album.sort_values('release_year', inplace=True)
acousticness_by_album


# In[33]:


# Plotting the acousticness per album

plt.figure(figsize=(12, 8))
g = sns.barplot(x='album', y='acousticness', data=acousticness_by_album, palette='Set1')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()


# We can see an interesting trend here as on average Four Tet's last 4 albums have been his most acoustic pieces of work. His album New Energy which he released in 2017 is his most acoustic album which, if you've heard that album makes a lot of sense as he's got a lot of organic and acoustic elements in that album.
# 
# His first album was also one of his more acoustic albums on average. His album Pink which was released in 2012 was considered his least acoustic album which is something I find intersting as there are a decent amount of acoustic elements in that album. It's possible that the album was just very uncoventional in it's arrangement and added a lot of unqiue electronic elements that it was hard for the Spotify acoustic algorithm to determine it as a more acoustic album.

# In[34]:


# Next I will look at the Danceability among Four Tet's albums. First I want to look at the distribution of 
# Danceability among his albums discography.

plt.figure(figsize=(12, 7))
sns.distplot(df['danceability'], bins=30)
plt.xlim(0, 1)
plt.title('Distribution of Danceability of Discography')
plt.show()


# In[35]:


# Next I want to look at the danceability over albums, and sort ascending from album release year

danceability_by_album = df.groupby(['album', 'release_year'])['danceability'].mean().reset_index()
danceability_by_album.sort_values('release_year', inplace=True)
danceability_by_album


# In[36]:


# Plotting the danceability of each album

plt.figure(figsize=(12, 7))
g = sns.barplot(x='album', y='danceability', data=danceability_by_album, palette='Set2')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.title('Average Danceability per album')
plt.show()


# It looks like the average danceability amongst Four Tet's albums are relatively consistent, with exception to Everything Ecstatic Part 2, which is  denoted as his least danceable album on average. That being said it looks like the rest of his albums on average are considered more danceable albums with an average danceability greater than 0.5. A lot of his albums hover around 0.5 danceability though and this could be due to the diversity most of his albums have with a combination of dance and ambient tracks.

# In[37]:


# Now I want to look at the danceability for tracks for each album
sns.set_style('white')

plt.figure(figsize=(12, 8))
g = sns.stripplot(x='album', y='danceability', data=df, alpha=0.5, jitter=True, palette='Set1', size=8)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title('Danceability per track per album')
plt.show()


# In[38]:


# From this it looks like There is Love in You has the track with the most danceability amongst all albums, 
# but also the track with the least danceability amongst all albums. I will double check here to confirm

df[(df['danceability'] == df['danceability'].max()) | (df['danceability'] == df['danceability']
                                                       .min())].sort_values('danceability', ascending=False)


# So interestingly both the most danceable track, and the least danceable track are borht on the 'There is Love in You' album. Something that's also interesting is that the track 'sing' which is the most danceable track has a very low acousticness, while the track 'pablo's heart' which is the least danceable track has an extremely high acousticness.

# In[39]:


# I want to plot out 4 graphs with 2 graphs per row, and 2 graphs per column showing the top 10 percentile for
# acousticness and the top 10 percentile for danceability plotting the acousticness and danceability
# for both.

# First I will create dataframes of only the top 10 percentile of both acousticness and danceability

aq = np.percentile(df['acousticness'], 90)
dq = np.percentile(df['danceability'], 90)

top_10_p_acousticness = df[df['acousticness'] >= aq]
top_10_p_danceability = df[df['danceability'] >= dq]


# In[40]:


# Plotting them

plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
g = sns.barplot(x='name', y='acousticness', data=top_10_p_acousticness)
g.set_xticklabels(g.get_xticklabels(), rotation=70)

plt.subplot(2, 2, 2)
g = sns.barplot(x='name', y='danceability', data=top_10_p_acousticness)
g.set_xticklabels(g.get_xticklabels(), rotation=70)

plt.subplot(2, 2, 3)
g = sns.barplot(x='name', y='acousticness', data=top_10_p_danceability)
g.set_xticklabels(g.get_xticklabels(), rotation=70)

plt.subplot(2, 2, 4)
g = sns.barplot(x='name', y='danceability', data=top_10_p_danceability)
g.set_xticklabels(g.get_xticklabels(), rotation=70)

plt.show()


# It looks like the most acoustic tracks can still be very danceable tracks, although it varies from track to track. However it looks like the most danceable are extremely unacoustic with the exception of one track **hi hello**.

# In[41]:


# To further investigate I am going to create a Joint Plot between acousticness and danceability

sns.jointplot(x='acousticness', y='danceability', data=df, height=10, alpha=0.4)


# Overall it doesn't look like there's a clear cut relationship between acousticness and danceability as I originally might have thought.

# In[126]:


# Lastly with regards to danceability, I want to see the count of tracks that have above 0.5 danceability, vs.
# tracks that have below 0.5. I will make a new column with a 0 for all tracks under 0.5 and a 1 for all tracks
# above 0.5

danceability = df.copy()
danceability['danceable'] = df['danceability'].apply(lambda x: 0 if x < 0.5 else 1)

sns.countplot(x='danceable', data=danceability)
plt.show()


# In[130]:


# Here I will calculate the percentage of Four Tet's that are considered danceable. My threshold for a track being
# danceable is above or equal to 0.5 in Spotify's danceability metric

round(100.0 * len(danceability[danceability['danceable'] == 1])/len(danceability), 2)


# So from this it looks likemore than half of Four Tet's tracks are considered Danceable - roughly 64 percent of his tracks are considered to be danceable.

# In[ ]:





# In[42]:


# Next I will look at the overall energy of Four Tet's discography. Plotting the distribution of energy here
sns.set_style('whitegrid')

plt.figure(figsize=(10, 7))
sns.distplot(df['energy'], bins=20)
plt.xlim(0, 1)
plt.title('Distribution of Track Energy')
plt.show()


# In[43]:


# Grouping by album to see which album on average has the most energy

album_by_energy = df.groupby(['album', 'release_year'])['energy'].mean().reset_index()
album_by_energy.sort_values('release_year', inplace=True)
album_by_energy


# In[44]:


# Plotting average album energy

plt.figure(figsize=(14, 8))
g = sns.barplot(x='album', y='energy', data=album_by_energy, palette='coolwarm')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title('Average Energy per Album')
plt.show()


# In[45]:


# Checking to see relationships between energy and acousticness, and energy and danceability

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
sns.scatterplot(x='energy', y='acousticness', data=df, alpha=0.4, color='black')
plt.xlim(0, 1)

plt.subplot(1, 2, 2)
sns.scatterplot(x='energy', y='danceability', data=df, alpha=0.4, color='red')
plt.xlim(0, 1)

plt.show()


# It looks like there are clusters within both plots above that have somewhat of a relationship.
# For example, in the energy vs. acousticness plot, there's a cluster of highly acoustic, lower energetic tracks, highly acoustic and highly energetic tracks, medium acoustic and high energy tracks, and low acoustic and high energy tracks

# In[46]:


# Plotting the energy for each track on each album
sns.set_style('white')

plt.figure(figsize=(12, 8))
g = sns.stripplot(x='album', y='energy', data=df, alpha=0.5, jitter=True, palette='viridis', size=8)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title('Danceability per track per album')
plt.show()


# It looks like a majority of Four Tet's album have a widespread range of energy. His two lateset album Sixteen Oceans and New Energy have the biggest range when it comes to track energy. I don't see much of a trend in the change of his album's energy though, because the energy in his albums are all relatively consistent. His two latest albums do show that he has started to incorporate some more less energetic music into his albums, however he keeps the energy widespread through both albums.

# In[47]:


# Looking at Four Tet's least energetic track, and his most energetic track

df[(df['energy'] == df['energy'].max()) | (df['energy'] == df['energy'].min())]


# In[ ]:





# In[48]:


# Checking out the distribution of Instrumentalness across his albums

plt.figure(figsize=(12, 7))
sns.distplot(df['instrumentalness'], bins=30)
plt.xlim(0, 1)
plt.show()


# As I anticipated, most of his tracks are highly instrumental, and contain very limited vocal content. It does look like a lot of his tracks are also considered to contain a lot of vocal content. I want to see how this varies per track over albums.

# In[49]:


plt.figure(figsize=(12, 8))
g = sns.stripplot(x='album', y='instrumentalness', data=df, size=8, palette='viridis', alpha=0.7)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# I looks like the majority of Four Tet's music is highly instrumental with exception to his first album 'Everything Ecstatic'. It looks like his first album contained a lot of music with high vocal content. As his career has progressed though it would appear that his music became more instrumental based with limited amount of vocal content. His album 'Rounds' has only highly instrumental tracks. It would appear that for that album he focusued on more of a club style electonic beat heavy album.

# In[50]:


# I am curious to check and see what keys most of Four Tet's music is in

df['key'].value_counts()


# In[51]:


# Since there were no -1 values present for Key it looks like each key value was detected. Now I want to see how
# many track keys were in major vs. minor

df['mode'].value_counts()


# In[52]:


g=sns.countplot(x='mode', data=df)
g.set_xticklabels(['Minor', 'Major'])
plt.title('Total Count of Minor and Major Tracks')
plt.show()


# In[53]:


# So it looks like most of Four Tet's tracks were in Major keys. I want to see how that is reflected per album

plt.figure(figsize=(12, 8))
sns.countplot(x='mode', hue='album', data=df, palette='coolwarm')
plt.show()


# In[54]:


# I want to see the actual keys and visualize what are the most frequent keys in his music, and per albums.
# I am going to create a function to add a column to df.

def key_and_mode(row):
    if row['key'] == 0 and row['mode'] == 1:
        return 'C-Major'
    elif row['key'] == 0 and row['mode'] == 0:
        return 'C-minor'
    elif row['key'] == 1 and row['mode'] == 1:
        return 'C#-Major'
    elif row['key'] == 1 and row['mode'] == 0:
        return 'C#-minor'
    elif row['key'] == 2 and row['mode'] == 1:
        return 'D-Major'
    elif row['key'] == 2 and row['mode'] == 0:
        return 'D-minor'
    elif row['key'] == 3 and row['mode'] == 1:
        return 'D#-Major'
    elif row['key'] == 3 and row['mode'] == 0:
        return 'D#-minor'
    elif row['key'] == 4 and row['mode'] == 1:
        return 'E-Major'
    elif row['key'] == 4 and row['mode'] == 0:
        return 'E-minor'
    elif row['key'] == 5 and row['mode'] == 1:
        return 'F-Major'
    elif row['key'] == 5 and row['mode'] == 0:
        return 'F-minor'
    elif row['key'] == 6 and row['mode'] == 1:
        return 'F#-Major'
    elif row['key'] == 6 and row['mode'] == 0:
        return 'F#-minor'
    elif row['key'] == 7 and row['mode'] == 1:
        return 'G-Major'
    elif row['key'] == 7 and row['mode'] == 0:
        return 'G-minor'
    elif row['key'] == 8 and row['mode'] == 1:
        return 'G#-Major'
    elif row['key'] == 8 and row['mode'] == 0:
        return 'G#-minor'
    elif row['key'] == 9 and row['mode'] == 1:
        return 'A-Major'
    elif row['key'] == 9 and row['mode'] == 0:
        return 'A-minor'
    elif row['key'] == 10 and row['mode'] == 1:
        return 'A#-Major'
    elif row['key'] == 10 and row['mode'] == 0:
        return 'A#-minor'
    elif row['key'] == 11 and row['mode'] == 1:
        return 'B-Major'
    elif row['key'] == 11 and row['mode'] == 0:
        return 'B-minor'


# In[55]:


# Creating a new column to show the key and mode for each track

df['key_n_mode'] = df.apply(lambda row: key_and_mode(row), axis=1)


# In[56]:


df.head()


# In[57]:


# Looking at the value counts for each key_n_mode

df['key_n_mode'].value_counts()


# In[58]:


# Visualizing with a countplot
sns.set_style('darkgrid')

plt.figure(figsize=(12, 7))
g = sns.countplot(x='key_n_mode', data=df.sort_values('key_n_mode'))
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[59]:


# Plotting out the different keys and modes per song for each album. The 0 is for a minor mode, and the 1 is major

plt.figure(figsize=(15, 8))
g=sns.stripplot(x='album', y='key', hue='mode', data=df, alpha=0.5, size=10, palette='Set1')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
g.set_yticks(range(len(df['key'].unique())))
g.set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.show()


# In[60]:


# Counting the amount of Major vs. Minor tracks per album

plt.figure(figsize=(15, 8))
g=sns.countplot(x='album', data=df, hue='mode')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# Overall it looks like Four Tet's music spans between major and minor modes almost evenly with a slight edge to more major songs over minor. His two most recent album along with his first two albums and his album 'Pause' have a lot more tracks in the Major mode then the minor mode it would seem. He loves the keys C-major, and A-major, however D-major and E-minor are on almost every single album. Something intersting to note is one his two track album 'Morning / Evening' both tracks are in the key C, with one track being C-major, and the other track being C-minor.
# 
# His album Rounds which has low vocal content and one that I presumed to be a more club style album has an even amount of minor mode tracks and major mode tracks. I would've expected it to have more minor mode tracks. With exception to his albums 'Ringer' and 'Everything Ecstatic Part 2' all of his albums are either evenly spread between major and minor tracks, or favor major tracks.

# In[ ]:





# In[61]:


# Next I will look at the overall Liveness of Four Tet's discography. Here I am plotting the distribution of
# liveness

plt.figure(figsize=(12,7))
sns.distplot(df['liveness'], bins=30, color='red')


# In[62]:


# It looks like the majority of his tracks are not preformed live as anticipated. I will look at a boxplot to
# further investigate this

plt.figure(figsize=(12, 7))
sns.boxplot(df['liveness'], orient='h', width=0.5)
plt.show()


# In[63]:


# I want to look at the boxplot distributions of liveness for each album with a hue of Major vs. Minor mode tracks.

plt.figure(figsize=(12, 10))
sns.boxplot(y='album', x='liveness', data=df, orient='h', width=0.5, hue='mode')
plt.show()


# Based off of our distribution plot and boxplot it looks like the large majority of Four Tet's tracks lie between the 0.1 - 0.25 range of track liveness. The outliers could be due to certain samples he used that contain a live audience atmosphere or element. It also looks like the majority of his Liveness in his albums are minor tracks, with 'Everything Ecstatic' and 'Sixteen Oceans' containing the largest range of Minor mode liveness, and 'Everything Ecstatic' and 'Pause' containing the largest range of Major mode liveness. It looks like on average most Minor mode tracks have very little Liveness in their tracks with exception to the tracks on the 'Everything Ecstatic' album.

# In[ ]:





# In[64]:


# Looking at loudness next. Plotting the loudness distribution here across all tracks

plt.figure(figsize=(12, 8))
sns.distplot(df['loudness'], bins=30, color='black')


# In[65]:


loudness = df.groupby(['album', 'release_year'])['loudness'].mean().reset_index()
loudness.sort_values('release_year', inplace=True)
loudness


# In[66]:


# Checking the average Loudness per album

plt.figure(figsize=(12, 5))
g=sns.barplot(x='album', y='loudness', data=loudness, palette='muted')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[67]:


# Plotting Loudness vs. Danceability

sns.jointplot(x='loudness', y='danceability', data=df, alpha=0.5, height=8)


# In[68]:


# Plotting loudness vs. acousticness

sns.jointplot(x='loudness', y='acousticness', data=df, alpha=0.5, height=8, color='green')


# In[69]:


plt.figure(figsize=(12, 8))
g=sns.stripplot(x='album', y='loudness', data=df, size=8, alpha=0.5)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# An interesting insight from the loudness attributes is that on average Four Tet's last 4 albums are his quietest, with his last two albums being the quietest albums he's ever released on average. I find that interesting that his earlier albums are his loudess albums on average because with the advancement of audio engineering music is becoming louder and louder, so I would've anticipated that his latest albums were his loudest on average, not the quietest. He definitely has more quieter tracks on his recent two albums than on any other album he released, with exception to 'Rounds'. However with 'New Energy' and 'Sixteen Oceans' he has more tracks that that fall below his average threshold of track loudness. It could be because these albums are his most acoustic, and he elected for a more organic feel by keeping the dynamic range intact on a majority of the tracks, and compensated loudness for organicness. These albums also had the lowest energy out of all of his albums - this could be due in part to the more acousticness of these albums and organic feel, or it could be in part because of the quietness of the albums.

# In[ ]:





# In[70]:


# Next I will look at the popularity of his tracks. First I want to find the most popular and least popular tracks.

df[(df['popularity'] == df['popularity'].
    max()) | (df['popularity'] == df['popularity'].min())].sort_values('popularity', ascending=False)


# In[71]:


# I want to look at the average popularity based on album

popularity_by_album = df.groupby(['album', 'release_year'])['popularity'].mean().reset_index()
popularity_by_album.sort_values('release_year', inplace=True)
popularity_by_album


# In[72]:


# Plotting the average popularity per album

plt.figure(figsize=(12, 8))
g=sns.barplot(x='album', y='popularity', data=popularity_by_album, palette='viridis')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[73]:


# Plotting track popularity for each album

plt.figure(figsize=(12, 8))
g=sns.stripplot(x='album', y='popularity', data=df, size=8, alpha=0.4)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[74]:


# I am curious to see if there's a disparity between popularity and track mode

plt.figure(figsize=(15, 8))
sns.boxplot(x='popularity', y='album', data=df, hue='mode')
plt.show()


# In[75]:


# I'm also curious to see if there's any one musical key that has a higher popularity than others

plt.figure(figsize=(10, 5))
g=sns.barplot(x='key', y='popularity', data=df,)
g.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.show()


# In[76]:


# I also want to see what the most popular key_n_modes were. I can use the plot above and just separate by 'mode' hue

plt.figure(figsize=(12, 5))
g=sns.barplot(x='key', y='popularity', data=df, hue='mode')
g.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.show()


# In[77]:


# I want to look at the audio features correlation with popularity

popularity_corr = df.corr()['popularity'].reset_index()
popularity_corr


# In[78]:


# Plotting a heatmap to see the correlation across the dataframe features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# As anticipated the most popular albums on average are Four Tet's recent albums 'Sixteen Oceans' and 'New Energy'. His least popular albums on average are his fourth album 'Everything Ecstatic Part 2', and third album 'Everything Ecstatic'. Popularity is partially based off of the most recent plays, so it's no surprise that his latest albums are on average the most popular as they probably have been getting the most recent plays being that they are his newest releases. 'Everything Ecstatic' has two tracks that were rated a 0 popularity, and have the most tracks that fall below 20 in populairty (almost all the tracks on the album). When investigating popularity based on mode, it looks like the distributions for each album are pretty even on a popularity scale when comparing Minor to Major mode tracks. When comparing keys it looks like F-Major and F-Minor were the most popular Major and Minor keys, however the distribution was pretty even amongst keys, which leads me to believe that song key isn't a big influence on song popularity. Upon further investigation it looks like acousticness has the largest correlation to song popularity, but it is only a correlation of 0.26 which is not large at all, and is probably only due to the fact that Four Tet's lateset albums are highly acoustic, and are his most popular albums because they are his most recent releases. The track 'Two Thousand and Seventeen' is his most popular track off of his album 'New Energy'. I have a feeling this is in some part due to the Spotify algorithm pushing it to the first track under his profile.

# In[85]:


# I anticipate that the speechiness on Four Tet's tracks will be very low, so I will plot the distribution

plt.figure(figsize=(12, 5))
sns.distplot(df['speechiness'], bins=30)


# As anticipated most of his tracks fall between 0.0 and 0.1 on the speechiness scale, and his tracks don't go above 0.5 on the scale.

# In[115]:


# I want to view the distribution of his track tempo's. I'm curious to see what's his most frequently used tempo.
# For this plot I want to use plotly.

import plotly
import cufflinks as cf
cf.go_offline()

df['tempo'].iplot(kind='hist', xTitle='Tempo', yTitle='Count', bins=50)


# In[108]:


# It's strange to see that he has one track that's lower than 3 bpm, and 2 tracks above 190. I want to further 
# investigate these

df[(df['tempo'] <= 0) | (df['tempo'] > 190)]


# In[100]:


# Upon further investigation the track 'Teenage Birdsong' is listed as 98 bpm, and 'Love Cry Joy Orbison remix' is
# listed as 132 bpm, so I will update both in the dataframe.

df.at[58,'tempo'] = 132
df.at[78, 'tempo'] = 98


# In[119]:


# The track "Pablo's Heart" turned out to be a field recording that was only 12 seconds long, hence the inability
# to define a tempo, thus calling the tempo 0. I will remove this instance.

new_df = df[df['tempo'] > 0]


# In[123]:


# I will replot the distribution based on the new_df

new_df['tempo'].iplot(kind='hist', xTitle='Tempo', yTitle='Count', bins=50, colors='green')


# From the distribution above it looks like Four Tet knows no bounds when it comes to track tempo's. It really shows his versatility as an artist. While he most frequently creates tracks between 125-130 bpm, he has quite a few releases within the 70-100 bpm range. The large portion of his tracks being between 115-140 bpm make sense due to the fact that about 64% of his tracks are considered danceable, and electronic club music usually falls between 115-140 bpm
# 
# The lower bpms are most likely his ambient and acoustic songs. I am curious to see if on average his two most recent albums have lower bpms. I would also be curious to map out danceability vs. tempo.

# In[135]:


# Looking at average tempo's over albums

album_tempo_average = new_df.groupby(['album', 'release_year'])['tempo'].mean().reset_index()
album_tempo_average.sort_values('release_year', inplace=True)
album_tempo_average


# In[137]:


plt.figure(figsize=(12, 8))
g=sns.barplot(x='album', y='tempo', data=album_tempo_average, palette='viridis')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.show()


# In[144]:


# Plotting a jointplot of acousticness vs. tempo

sns.jointplot(x='tempo', y='acousticness', data=new_df, kind='hex', color='red', height=7, alpha=0.7)


# In[145]:


# Plotting a jointplot between tempo and danceability

sns.jointplot(x='tempo', y='danceability', data=new_df, height=7, alpha=0.5)


# It looks like there might be a small relationship between acousticness and tempo and danceability and tempo, but nothing major that pops out. When looking at the correlation heatmap there's around a 0.2 correlation for both relationships which isn't that strong of a relationship. That being said, on average the album with the second lowest tempo was 'New Energy' which also happened to be the most acoustic album, but otherwise the average album tempos all fell between 110-125 with exception to 'Rounds' which had the lowest average tempo.

# In[148]:


# I'm curious to look at the tempo distributions over albums. I will plot some boxplots to see that.

plt.figure(figsize=(12, 8))
sns.boxplot(x='tempo', y='album', data=new_df)


# It looks like his earlier albums and his two recent albums had the widest tempo distribution range. Maybe that's an indicator along with the high acoustic levels of his recent two albums that he's starting to create more diverse albums, and potentially incorporating a certain style to tracks that was more prevalent in his earlier album releases. His album Beautiful Rewind seems to have quite a few outliers, with his remaining albums have a much smaller range of tempos.

# In[169]:


# I want to see how many tracks in the album Beautiful Rewind are between 100 and 140 bpm

len(new_df[(new_df['album'] == 'Beautiful Rewind') & (new_df['tempo'].between(100, 140))])


# In[ ]:





# In[170]:


# The last feature I want to explore is 'valence'. Here I will plot the distribution of valence using plotly.

df['valence'].iplot(kind='hist', xTitle='Valence', yTitle='Count', colors='black', bins=50)


# What's interesting is that a large portion of his tracks are considered extremely sad. I would have actually expected the opposite, especially since he has more Major tracks than Minor tracks. His music does have a melancholy-ness to it, so perhaps this is why he has a lot of songs that have a very low valence. Otherwise like many other attributes the valence of his discography is truly spread out across the spectrum. It does seem though that there are more tracks  that are on the lower side of the valence spectrum.

# In[171]:


# I want to plot the average valence over his albums

average_album_valence = df.groupby(['album', 'release_year'])['valence'].mean().reset_index()
average_album_valence.sort_values('release_year', inplace=True)
average_album_valence


# In[174]:


plt.figure(figsize=(14, 8))
g=sns.barplot(x='album', y='valence', data=average_album_valence, palette='Set1')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[176]:


# I also want to look at where each track falls on the valence spectrum per album

plt.figure(figsize=(12, 7))
g=sns.stripplot(x='album', y='valence', data=df, palette='Set1', size=8, alpha=0.5)
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.show()


# In[178]:


# Plotting the distribution of valence over albums

plt.figure(figsize=(14, 7))
sns.boxplot(y='album', x='valence', data=df, palette='Set1')
plt.show()


# It looks like his 3 most recent album releases all fall more on the sadder/ melancholy side. This further shows that his music has gone from being less clubby and more melancholoy, ambient and acoustic. The distribution of his most recent album Sixteen Oceans falls primarily between 0 and 0.25 on the valence scale with 3 major outliers. The overall distribution of his tracks filling the spectrum of valence is also evident in the majority of his albums as they range consistently from very low valence tracks to very high valence. It furhter confirms his versatility as an artist, and further proves why his sound is so unique and so distinguishable.

# In[ ]:





# ## Building a model

# After performing a lot of EDA on the dataset built off of the Spotify Audio Features metrics, I want to see if I can build a model that can predict a 'Four Tet style' songs' popularity based off of the Audio Features metrics I have.

# In[180]:


# Setting up the variables

X = new_df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
                          'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']].values
y = new_df['popularity'].values


# In[181]:


# Importing train_test_split

from sklearn.model_selection import train_test_split


# In[182]:


# Creating train and test variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=104)


# In[183]:


# Importing MinMaxScaler and creating an object for it

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[192]:


# Scaling and transforming the X_train and X_test data

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# In[193]:


# Importing Sequential, Dense, Dropout and EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[194]:


# Building the model

model = Sequential()

model.add(Dense(11, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[195]:


# Creating an early stop

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[196]:


# Fitting the model

model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=200, callbacks=[early_stop])


# In[197]:


# Putting the loss values throughout the model training into a dataframe

losses = pd.DataFrame(model.history.history)


# In[198]:


# Plotting the losses 

losses.plot()


# In[199]:


# Importing mean_squared_error, and mean_absolute_error to evaluate the model

from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[200]:


# Creating predictions from the X_test_scaled data

predictions = model.predict(X_test_scaled)


# In[201]:


# Evaluation the model by calculating the mean_squared_error, mean_absolute_error, and root_mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)


# In[202]:


print('Mean Absolute Error: {}'.format(mae))
print('Root Mean Squared Error: {}'.format(rmse))
print('Mean Squared Error: {}'.format(mse))


# In[234]:


# Plotting the Predictions vs. the Actual with a red line that shows were a perfect models predictions would be.
# The blue line reflects the linear line of the model's precitions vs. the actual popularity level

results = pd.DataFrame(predictions, columns=['Predictions'])
results['Actual'] = y_test

x = np.linspace(0, 60, 100)
y = x

sns.lmplot(x= 'Actual', y='Predictions', data=results, height=8)
plt.plot(x, y, color='r')
plt.xlim(-5, 60)
plt.ylim(0, 35)
plt.show()


# In[ ]:





# In[79]:





# In[80]:





# In[ ]:





# In[ ]:





# In[ ]:




