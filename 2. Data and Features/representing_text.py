from sklearn.feature_extraction.text import CountVectorizer

corpus = [
          "Authman ran faster than Harry because he is an athlete.",
          "Authman and Harry ran faster and faster"
          ]
          
bow = CountVectorizer() # bag of words
X = bow.fit_transform(corpus) # Sparse matrix

print(bow.get_feature_names()) # ['an', 'and', ...]
print(X.toarray()) # [[1 0 1 ...] [0 2 0 ...] ]