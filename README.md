# Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio

## Facial Expression Recognition
Original dataset from Kaggle: <br>
[original dataset](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/releases/tag/v0)<br>
After data preprocessing, dataset used in building models: <br>
[data: face-data.csv](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/releases/tag/v1)<br>

Code: [code](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/tree/main/Facial%20Expression%20Recognition)<br>
1. EDA <br>
2. CNN model <br>
3. DCNN model <br>

## Audio Emotion Recognition
Original dataset from Kaggle: <br>
[original dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)<br>
After data preprocessing, dataset used in building models: <br>
[features.csv](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/tree/main/Audio%20Emotion%20Recognition)<br>

Code: [code](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/tree/main/Audio%20Emotion%20Recognition)<br>
1. audio-emotion-recognition-model <br>
2. grid-search-Hyperopt <br>

## Mood-based Recommender
### Input Data:

1. Facial recognition model results:<br>
[Facial data](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/blob/main/Mood-based%20Recommender/facial-recommendation.csv)<br>
2. Audio recognition model results<br>
[Audio data](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/blob/main/Mood-based%20Recommender/audio-recommendation.csv)<br>

### Code: [A demo for mood-based music recommender](https://github.com/tsaokaiting/Music-Recommendation-Based-on-Emotions-Extracted-from-Facial-Expression-and-Speech-Audio/blob/main/Mood-based%20Recommender/5153_recommender_mk.ipynb)<br>

### Functions:
#### 1. get_valence_arousal()
```
def get_valence_arousal(data, map):
  '''
  Input:
    data: User emotion prediction output probability vector including 6 dimensions (angry	fear	happy	neutral	sad	surprise)
          Togther with two additional features: pred_label, label.
    map: Valence-arousal score for six emotions
  Output:
    Dataframe contains valence-arousal scores.
    Features: Valence, Arousal,	pred_label, label
  '''
  map_dict = dict.fromkeys(map.Mood)
  for key in map_dict.keys():
    map_dict[key] = np.array(map.loc[map['Mood'] == key, ['Valence','Arousal']])
  ## Prepare emotion Valence-arousal vector and probability matrix
  emotion = map_dict.keys()
  data_m = data.loc[:,emotion]
  emotion_v = np.array([i[0] for i in map_dict.values()])
  data_score = data_m @ emotion_v
  data_score_df = pd.DataFrame(data_score.values, columns = ['Valence','Arousal'])
  data_score_df['pred_label'] = data['pred_label']
  data_score_df['label'] = data['label']
  return data_score_df
```

#### 2. sample_recommend()


```
def sample_recommend(user_emotion, music_dataset, music_tracklist, top = 3):
  '''
  Input:
    user_emotion: valence_arousal score for one record, 2D list-like
    music_dataset: music dataset with columns, Number, Valence, Energy at least
    music_tracklist: music list with number as index, is used for sourcing details of mucic recommended
    top: number of top similar music recommended
  Output:
    Dataframe contains topN music detailed information
  '''
  music_dataset = music_dataset.copy()
  music_dataset['similarity'] = music_dataset.apply(lambda x: cos_similarity(user_emotion, x[['valence','energy']]), axis = 1)
  top_music = music_dataset.sort_values('similarity', ascending = False).iloc[:top,:]
  top_num = top_music['number']
  results = music_tracklist.loc[music_tracklist['Number'].isin(top_num), :]
  return top_music, results
```

