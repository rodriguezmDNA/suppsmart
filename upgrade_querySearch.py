### General use
import os #interact with the system
import pickle #save/load
import re # regular expressions
import pandas as pd #data frames
import numpy as np #math
from scipy import sparse #sparse matrices

### For lemmatization
import spacy #Language model
"""
See: https://spacy.io/models/en
English multi-task CNN trained on OntoNotes. Assigns context-specific token vectors, 
POS tags, dependency parse and named entities
"""

### For stop words
from wordcloud import STOPWORDS
from nltk.corpus import stopwords

### To create the matrices and perform similarity searches for the query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

### gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

### Dviz
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt

##### Define unwanted/stop words

"""
-PRON- is a POS (part of speech) tag. In this case any pronoun (my, she, they, etc) is 
generalized (through lemmatization) to a -PRON- tag.
This and other tags could be removed using another approach, but for simplicity I'm removing them along with
other stop words
"""
## 
stop_words = stopwords.words('english')
extraStop=["mg","erowid", "-PRON-","june",'i知'] 
stop_words.extend(STOPWORDS)
stop_words.extend(extraStop)
stop_words.extend(stopwords.words('french')) # Just in case
stop_words=set(stop_words)

# Some words are actually desired, removing them from the set will keep them in the data.
stop_words.discard("no")

nlp = spacy.load('en', disable=['parser', 'ner']) 
# Functions for query search
def cleanQuery(query):
    """
    A function to clean the input text
    """
    cleanQuery = " ".join([_.lemma_ for _ in nlp(" ".join(simple_preprocess(query)))])
    cleanQuery = removeStopWords(cleanQuery,stop_words)
    return cleanQuery

def d2v_inferVector(searchTerms):
    activationVector = doc2vec_model.infer_vector(doc_words=searchTerms.split(" "), epochs=200)
    activationVector = activationVector.reshape(1, -1)
    return activationVector #Activated neurons in NN in the doc2vec model

def svd_inferVector(searchTerms):
    """
    Find the embedding vector from the SVD decomposition
    """
    bowSearch = tf.transform([searchTerms]).toarray() #no need to split here
    bowSearch = svd.transform(bowSearch)
    bowSearch = bowSearch.reshape(1, -1)
    return bowSearch

def removeStopWords(listOfText,stopWords):
    return ' '.join([_ for _ in listOfText.split() if not _ in stopWords])


def getSimilarDocument(matrix,vector):
    """
    Calculate cosine similarity between the target matrix and the calculated vector
    """
    cosSimilarity = pd.DataFrame(cosine_similarity(X=matrix,
                                                    Y=vector,
                                                    dense_output=True))
    cosSimilarity.set_index(matrix.index, inplace=True)
    cosSimilarity.columns = ["cosine_similarity"]
    return cosSimilarity

def aggregateCosSims(svd,d2v,topN=-1):
    ### Average cosine similarity from doc2vec and also bag of words. 
    aggSimTable = pd.merge(d2v, svd, left_index=True, right_index=True)
    aggSimTable.columns = ["Doc2Vec", "BoW"]
    aggSimTable['aggCosSim'] = (aggSimTable["Doc2Vec"] + aggSimTable["BoW"])/2
    aggSimTable.sort_values(by="aggCosSim", ascending=False, inplace=True)
    return aggSimTable.head(topN)

def wrapperSimilarSupplements(query,topN=-1):
    searchTerms = cleanQuery(query)
    d2v_searchOut = d2v_inferVector(searchTerms)
    svd_searchOut = svd_inferVector(searchTerms)
    ##
    svd_CosSim = getSimilarDocument(svd_feature_matrix,svd_searchOut)
    d2v_CosSim = getSimilarDocument(doctovec_feature_matrix,d2v_searchOut)
    topSimilarSupplements = aggregateCosSims(svd_CosSim,d2v_CosSim,topN)
    namesOfSupps = getSupplementNames(topSimilarSupplements)
    return namesOfSupps,topSimilarSupplements,searchTerms

def loadModelData(dictOfSaves,pickleShelf = "./modelData/",verbose=False):
    """
    Load results from model training
    """
    tmpList = []
    for fileName in dictOfSaves.values():
        #
        picklePath=pickleShelf + fileName
        if verbose:
            print("loading",picklePath)       
        with open(picklePath, 'rb') as handle:
            tmpList.append(pickle.load(handle))
    return tmpList

###
def make_bigrams(listOfTokens,bimodel):
    return [bimodel[_] for _ in listOfTokens]

def make_trigrams(listOfTokens,bimodel,trimodel):
    return [trimodel[bimodel[_]] for _ in listOfTokens]

def make_quadgrams(listOfTokens,bimodel,trimodel,quadmodel):
    return [quadmodel[trimodel[bimodel[_]]] for _ in listOfTokens]

def buildTopicModel(supplement,ngram="tri",nTopics=8,modelType="lda"):
    tmpTokens = scrappedCleanTopic[supplement]
    # ## For debugging
    # tmpTokens = [("my cat is the best").split(), 
    #              ("my dog is the worst").split(),
    #              ("my fish is the worst").split(),
    #              ("i have one wish").split(),
    #             ("i have one kind of shoe").split(),
    #             ("i have one kind of dream").split(),
    #             ("i have one kind of pillow").split()]
    # Build models
    bigram = gensim.models.Phrases(tmpTokens, min_count=3, threshold=2) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tmpTokens],min_count=3, threshold=2)  
    quadgram = gensim.models.Phrases(trigram[bigram[tmpTokens]],min_count=3, threshold=2)  

    # Get bi and trigrams
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    quadgram_model = gensim.models.phrases.Phraser(quadgram)
    
    #print(quadgram_model[trigram_model[bigram_model[tmpTokens[4]]]])
    tmp_bi = make_bigrams(tmpTokens,bigram_model)
    tmp_tri = make_trigrams(tmpTokens,bigram_model,trigram_model)
    tmp_quad = make_quadgrams(tmpTokens,bigram_model,trigram_model,quadgram_model)

    ngramChoice = {"bi":tmp_bi, 
                   "tri":tmp_tri,
                   "quad":tmp_quad}
    
    ngram = ngramChoice[ngram]
    ###
    id2word = gensim.corpora.Dictionary(ngram) # Dictionary
    allWords = ngram #Corpus
    corpus = [id2word.doc2bow(text) for text in allWords] #Freq of tokens
    ##########
    if(modelType == "lda"):
        ldaMod = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=nTopics, 
                                                   random_state=42,
                                                   update_every=1,
                                                   chunksize=200,
                                                   passes=30,
                                                   alpha='auto',
                                                   per_word_topics=True)
    else:
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=nTopics, id2word=id2word)
        ldaMod = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
    return ldaMod,id2word,corpus


def topicModelSupp(supp,nTopics=8,modelType='lda',ngram='tri'):
    resLDA, resI2W, resCorp = buildTopicModel(supp,ngram,nTopics,modelType)
    # Visualize the topics-keywords
    #pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(resLDA, resCorp, resI2W)
    return pyLDAvis.show(vis,port=8501) ## Working well with st.altair_chart
    #return pyLDAvis.display(vis) #no pyplot,
    #return pyLDAvis.prepared_data_to_html(vis) ## no st.write or mdown

getSupplementNames = lambda df: list(df.index)


loadPickles = {'doc2vec_model': 'doc2vec_model.pkl',
               'svd_feature_matrix': 'svd_feature_matrix.pkl',
               'doctovec_feature_matrix': 'doctovec_feature_matrix.pkl',
               'tfidf_matrix': 'tfidf_matrix.pkl',
               'tf': 'tf.pkl',
               'svd': 'svd.pkl',
               'svdMatrix': 'svdMatrix.pkl',
               'scrappedCleanTopic':"scrappedCleanTopic.pkl"
               }

tmpList = loadModelData(loadPickles,verbose=False)
doc2vec_model,svd_feature_matrix,doctovec_feature_matrix, tfidf_matrix, tf, svd, svdMatrix, scrappedCleanTopic= tmpList
del tmpList #Stop holding into memory




