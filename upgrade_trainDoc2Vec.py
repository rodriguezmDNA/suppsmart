# Functions to train the model 

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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
#############################################

### Initialize the spacy model
"""
Converts words to their root word: 
"""
nlp = spacy.load('en', disable=['parser', 'ner']) 

###### Define functions ######
##############################

## Flatten lists of lists
flatten = lambda l: [item for sublist in l for item in sublist]

def loadScrappedData(pickleShelf = "scrappedData/"):
    """
    After scraping each supplement, the data is stored in a "scrappedData/" folder. This function loads each pickled file into memory
    """
    supplementsScrapped = {}
    for file in os.listdir(pickleShelf):    
        if 'pkl' in file: ## Make sure to load only pickled files. '.DS_Store' causes problems
            picklePath=pickleShelf + file
            supplement = file.replace(".pkl",'')
            #print(supplement, picklePath)
            with open(picklePath, 'rb') as handle:
                tmpScrapped = pickle.load(handle)
            supplementsScrapped[supplement]=tmpScrapped
    return supplementsScrapped

def regexClean(s):
    """
    Clean leftover HTML from scraped data
    """
    s_clean=re.sub(r'\<span class=\"erowid-caution\">\[Erowid Note: \nTwo samples of powder \(even of the same chemical\) with equivalent volumes won\'t necessarily weigh the same\. For this reason, eyeballing is an inaccurate and potentially dangerous method of measuring, particularly for substances that are active in very small amounts\.\nSee \<a href=\"\/psychoactives\/basics\/basics_measuring1\.shtml\"\>this article on The Importance of Measured Doses\<\/a\>\.\]\<\/span\>','',s)
    s_clean=re.sub(r"(-->|<!--|<br/>|\\n|\\r|\[|\"|\(|\)|-|–|:|[0-9]|\$|\@|\%|<a href.+ >|mg|<span.+span>|<span.+>|<a.+>|<\/.>|<\/span>|]|\+|\n|\.\.\.|<div.+>|<div.+>|<\/div>|\r)", ' ', s_clean) 
    #s_clean = re.sub(r"(-->|<!--|<br/>|\\n|\\r|\[|\"|\(|\)|-|–|:|[0-9]|\$|\@|<a href.+ >|<span.+span>|<span.+>|<a.+>|<\/.>|<\/span>|]|\+|\n|<div.+>|<div.+>|<\/div>|\r)", ' ', s) #Clean
    #s_clean = re.sub(r'\<span class=\"erowid-caution\">\[Erowid Note: \nTwo samples of powder \(even of the same chemical\) with equivalent volumes won\'t necessarily weigh the same\. For this reason, eyeballing is an inaccurate and potentially dangerous method of measuring, particularly for substances that are active in very small amounts\.\nSee \<a href=\"\/psychoactives\/basics\/basics_measuring1\.shtml\"\>this article on The Importance of Measured Doses\<\/a\>\.\]\<\/span\>','',s_clean)
    #s_clean = re.split('<br/>|\\.',s_clean)
    #s_clean = list(compress(s_clean, [len(_)> 4 for _ in s_clean]))
    return(s_clean)

def preProcessText(dictOfText):
    """
    From a dictionary containing a string of text:
    Tokenizes the list, converts to lower case and removes words that are too short or too long
    """
    cleanDict={}
    for k,blurb in dictOfText.items():
        cleanDict[k]=" ".join([_.lemma_ for _ in nlp(" ".join(simple_preprocess(blurb)))])
    return(cleanDict)

def removeStopWords(listOfText,stopWords):
    return ' '.join([_ for _ in listOfText.split() if not _ in stopWords])

def saveData(fileToSave,Name,pathSave='./modelData/'):
    """
    A function to save files to a specific folder. If dir doesn't exit,create
    """
    ## Make dir
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    filepath = os.path.join(pathSave+Name+".pkl")
    with open(filepath, 'wb') as handle:
        pickle.dump(fileToSave, handle, protocol=pickle.HIGHEST_PROTOCOL)

def cleanList(listOfSentences):
    return [[_.lemma_ for _ in nlp(" ".join(simple_preprocess(l)))] for l in listOfSentences]

def cleanDict(dictToClean):
    return {k:cleanList(v) for k,v in dictToClean.items()}

def remStopWords(listOfSentences,stpwrd=stop_words):
    return [[_ for _ in l if _ not in stpwrd] for l in listOfSentences]

def make_bigrams(listOfTokens):
    return [bigram_model[_] for _ in listOfTokens]

def make_trigrams(listOfTokens):
    return [trigram_model[bigram_model[_]] for _ in listOfTokens]

def make_quadgrams(listOfTokens):
    return [quadgram_model[trigram_model[bigram_model[_]]] for _ in listOfTokens]

"""
Where the fun part begins
"""    
################################################################################
#### Load data
supplementsScrapped = loadScrappedData("scrappedData/")
## Since posts are stored into lists of multiple categories per supplement, flatten these lists
supplementsScrapped = {k : flatten(supplementsScrapped[k].values()) for k in supplementsScrapped.keys()}
# Each supplement should be stored into one dict key
############
from itertools import compress
deCaff = supplementsScrapped["Caffeine"]
#print(len(supplementsScrapped["Caffeine"]))
supplementsScrapped["Caffeine"] = list(compress(deCaff,["Après plusieur" not in _ for _ in deCaff]))
#print(len(supplementsScrapped["Caffeine"]))


### Clean data
scrappedClean = {key:[regexClean(_) for _ in val] for key, val in supplementsScrapped.items()}
scrappedCleanLemma = {k:cleanList(v) for k,v in scrappedClean.items()}
scrappedCleanTopic = {k:remStopWords(v) for k,v in scrappedCleanLemma.items()}
scrappedCleanDoc2Vec = {k:" ".join(flatten(v)) for k,v in scrappedCleanTopic.items()}


saveData(scrappedCleanTopic,"scrappedCleanTopic") #Save the matrix

### Convert to DF to make it easy to iterate and get indices
supplementDF = pd.DataFrame.from_dict(scrappedCleanDoc2Vec,orient='index',columns=['Pooled'])



###################### BoW approach
#### Fit a TF-IDF
# get vocabulary and tfidf from all strings.
tf = TfidfVectorizer(analyzer='word', 
                     min_df=5,
                     ngram_range=(2,4),
                     #max_features=1000,
                     stop_words='english')
tf.fit(supplementDF['Pooled'])

tfidf_matrix = tf.transform(supplementDF['Pooled'])
# Save
saveData(tf,"tf") #Save the tf model
saveData(tfidf_matrix,"tfidf_matrix") #Save the matrix



#### Dimension reduction for BoW approach
svd = TruncatedSVD(n_components=500)
svdMatrix = svd.fit_transform(tfidf_matrix)
## Save
saveData(svdMatrix,"svdMatrix")
saveData(svd,"svd")


### Get the feature matrix as a dataframe
# this will be used to compute similarities later
n = len(supplementDF.index)
supplements = supplementDF.index
svd_feature_matrix = pd.DataFrame(svdMatrix[:,0:n] ,index=supplements)

saveData(svd_feature_matrix,"svd_feature_matrix")

###################### Doc2Vec approach
### Doc2Vec model: basically a shallow NN
# Prep
supplementDF["listedDocs"] = [_.split() for _ in supplementDF["Pooled"]]
documents = supplementDF["listedDocs"]
# convert to tagged documents: each string of text is indexed
taggedDocuments = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

## instance of Doc2Vec model
doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=100, # Of layers
                                              min_count=5, 
                                              epochs=2000, #Train 2000 times
                                              seed=0, #random seed
                                              window=7, # Distance between the current and predicted word within a sentence
                                              dm=1)
## build vocabulary
doc2vec_model.build_vocab(taggedDocuments)

##### Train the actual model
doc2vec_model.train(taggedDocuments, 
                    total_examples=doc2vec_model.corpus_count, 
                    epochs=doc2vec_model.epochs)
############


## Create doc2vec matrix
doctovec_feature_matrix = pd.DataFrame(doc2vec_model.docvecs.vectors_docs, 
                                       index=supplementDF.index)

saveData(doc2vec_model,"doc2vec_model")
saveData(doctovec_feature_matrix,"doctovec_feature_matrix")

