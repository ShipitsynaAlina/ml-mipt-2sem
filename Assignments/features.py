from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        
        
        num_tokens = {}
        for text in X:
            tokens = text.split()
            for token in tokens:
                if (token in num_tokens):
                    num_tokens[token] = num_tokens[token] + 1
                else:
                    num_tokens[token] = 1
                    
        num_tokens_list = list(num_tokens.items())
        num_tokens_list.sort(key=lambda i: i[1], reverse=True)
        voc = []
        
        for i in range(min(self.k, len(num_tokens_list))):
            voc.append(num_tokens_list[i][0])
        self.bow = voc    

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        voc = self.bow
        tokens = text.split()
        voc_map = {}
        
        for i, term in enumerate(voc, start=0):
            voc_map[term] = i

        bow = [0]*len(voc)
        for token in tokens:
            if token in voc_map:
                bow[voc_map[token]] = bow[voc_map[token]]+1
            else:
                pass
        
        result = bow
            
        
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        
        text = []
        for words in list(X):
            text += list(set(words.split()))
            
        words_freq = Counter(text)
        if self.k is not None:
            top_words = [word for word, freq in words_freq.most_common(self.k)]
            words_freq = {word : words_freq[word] for word in words_freq if word in top_words}
        
        self.idf = {word : np.log(len(X) / words_freq[word]) for word in words_freq}
        
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        
        tf = Counter(text.split())
        
        if self.normalize:
            tf = {word : tf[word] / len(tf) for word in tf}
            
        result = [tf[word] * self.idf[word] if word in tf else 0 for word in self.idf]
        
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])