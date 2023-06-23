#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

# # a, an, the, and, but gibi bizim icin bir anlam teskil etmeyen yapilari DataFramimizden cikartalim.
tfidf = TfidfVectorizer(stop_words="english")

#ilerleyen asamalarda hata almamak cin overview degiskenindeki null degerli hicbirseyle dolduralim
# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')

# tfidf nesnesine gore fit et ve donustur
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape # satirlardakiler metinlerdir 'overviewdir'. Sutunlardakiler essiz kelimelerdir.


df['title'].shape

#sutunlardaki butun essiz kelimeleri gormek istersek
tfidf.get_feature_names()

# tfidf skorlari
tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

# Butun olasi dokuman ciftleri icin tek tek cos sim hesabi yapar.cosine_sim matrisinde herbir filmin birbirleriyle
# benzerlikleri var
cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape

# 1. indexteki filmin diger tum filmlerle olan benzerliklerini gormek icin
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# indexlerden ve film isimlerinden olusan bir pd Series olusturalim
indices = pd.Series(df.index, index=df['title'])

# filmlerin index bilgilerini saydiralim ve fazla tekrar eden filmleri en gunceli kalacak sekilde sadelestirelim
indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

# 'Sherlock Holmes' filminin indexini degiskene atiyorum
movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

# 'Sherlock Holmes' filmi ile diger filmerin benzerliklerini ifade eden Smilarity Score larini gorelim
similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

# 'Sherlock Holmes' filminin benzerlik skorlarini azalan seklinde listeleyelim. 1den basliyor cunku 1. film kendisi
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# Ilk veri setimizde sectigimiz indexlere git
df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
