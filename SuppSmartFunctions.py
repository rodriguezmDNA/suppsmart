
"""
Load libs
"""
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


##DatViz
import seaborn as sns
import matplotlib.pyplot as plt

######################################################
stop_words = stopwords.words('english')
extraStop=["mg","erowid", "-PRON-"] 
stop_words.extend(STOPWORDS)
stop_words.extend(extraStop)
stop_words.extend(stopwords.words('french')) # Just in case
stop_words=set(stop_words)

# Some words are actually desired, removing them from the set will keep them in the data.
stop_words.discard("no")

nlp = spacy.load('en', disable=['parser', 'ner']) 
######################################################

#############
def cleanQuery(query):
    """
    A function to clean the input text
    """
    cleanQuery = " ".join([_.lemma_ for _ in nlp(" ".join(simple_preprocess(query)))])
    cleanQuery = removeStopWords(cleanQuery,stop_words)
    return cleanQuery

def removeStopWords(listOfText,stopWords):
    return ' '.join([_ for _ in listOfText.split() if not _ in stopWords])

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

def getSimilarDocument(matrix,vector):
    """
    Calculate cosine similarity using the BoW approach
    """
    cosSimilarity = pd.DataFrame(cosine_similarity(X=matrix,
                                                    Y=vector,
                                                    dense_output=True))
    cosSimilarity.set_index(matrix.index, inplace=True)
    cosSimilarity.columns = ["cosine_similarity"]
    return cosSimilarity

def aggregateCosSims(svd,d2v,topN=-1):
    ### Aggregate
    aggSimTable = pd.merge(d2v, svd, left_index=True, right_index=True)
    aggSimTable.columns = ["Doc2Vec", "BoW"]
    aggSimTable['aggCosSim'] = (aggSimTable["Doc2Vec"] + aggSimTable["BoW"])/2
    aggSimTable.sort_values(by="aggCosSim", ascending=False, inplace=True)
    return aggSimTable.head(topN)

getSupplementNames = lambda df: list(df.index)


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


def showDocEmbedding(embedding,x):
    fig = plt.figure(figsize = (12,12))
    sns.set(style='white')

    umX = embedding[:, 0],
    umY = embedding[:, 1],


    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('UMAP1', fontsize = 15)
    ax.set_ylabel('UMAP2', fontsize = 15)
    ax.set_title('UMAP', fontsize = 20)

    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['bottom'].set_color(None)
    ax.spines['top'].set_color(None)

    ax.scatter(umX, umY, c = "black")

    for i, txt in enumerate( list(x.index) ):
        #print(txt, (umX[i], umY[i]))
        ax.annotate(txt, (embedding[i][0], embedding[i][1]),
                    color="skyblue",size=18)

    plt.title('UMAP projection of the Doc2Vec doc embeddings')
    plt.show()


def queryUMAP(query,doc2vec_model,umap_wordvecs):
    query = query.split()
    docSimWords = pd.DataFrame.from_dict(dict(doc2vec_model.wv.most_similar( query )),
                                     orient='index',columns=[""])
    AllSimWords = [_ for _ in docSimWords.index if _ in umap_wordvecs.index]
    AllQueryWords = [_ for _ in query if _ in umap_wordvecs.index]
    reduxUmap = umap_wordvecs.loc[AllSimWords + AllQueryWords,:]
    return showDocEmbedding(reduxUmap,query=AllQueryWords)
    

def plotCosinSim(topSimilarSupplements):
    #plt.rcParams['figure.figsize'] = [4, 8]
    sns.set(style="darkgrid")
    figOut = sns.barplot(y="index",x='aggCosSim',
            data=topSimilarSupplements.reset_index(),color="skyblue")
    return figOut


def showDocEmbedding(embeddingDF,query=""):
    fig = plt.figure(figsize = (12,12))
    sns.set(style='white')

    umX = embeddingDF["UMAP1"]
    umY = embeddingDF["UMAP2"]


    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('UMAP1', fontsize = 15)
    ax.set_ylabel('UMAP2', fontsize = 15)
    ax.set_title('UMAP', fontsize = 20)

    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['bottom'].set_color(None)
    ax.spines['top'].set_color(None)

    ax.scatter(umX, umY, c = "black")

    for i, txt in enumerate( list(embeddingDF.index) ):
        if txt not in query: 
            txtCol = "skyblue"
        else:
            txtCol = "orange"
        ax.annotate(txt, (umX[i], umY[i]),
                    color=txtCol,size=18)
    plt.title('UMAP projection of the embeddings')
    plt.show()

### Load stuff

loadPickles = {'doc2vec_model': 'doc2vec_model.pkl',
               'svd_feature_matrix': 'svd_feature_matrix.pkl',
               'doctovec_feature_matrix': 'doctovec_feature_matrix.pkl',
               'tfidf_matrix': 'tfidf_matrix.pkl',
               'tf': 'tf.pkl',
               'svd': 'svd.pkl',
               'svdMatrix': 'svdMatrix.pkl',
               'umap_doc':"umap_doc.pkl",
               'umap_wordvecs':"umap_wordvecs.pkl"
               }

tmpList = loadModelData(loadPickles,verbose=False)
doc2vec_model,svd_feature_matrix,doctovec_feature_matrix, tfidf_matrix, tf, svd, svdMatrix, umap_doc, umap_wordvecs = tmpList
del tmpList #Stop holding into memory
