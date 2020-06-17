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

### dim red
import umap.umap_ as umap #https://github.com/lmcinnes/umap/issues/24
#conda install -c conda-forge umap-learn
from sklearn.preprocessing import StandardScaler

##### Define unwanted/stop words

"""
-PRON- is a POS (part of speech) tag. In this case any pronoun (my, she, they, etc) is 
generalized (through lemmatization) to a -PRON- tag.
This and other tags could be removed using another approach, but for simplicity I'm removing them along with
other stop words
"""
## 
stop_words = stopwords.words('english')
extraStop=["mg","erowid", "-PRON-"] 
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
        
def cleanQuery(query):
    """
    A function to clean the input text
    """
    cleanQuery = " ".join([_.lemma_ for _ in nlp(" ".join(simple_preprocess(query)))])
    cleanQuery = removeStopWords(cleanQuery,stop_words)
    return cleanQuery

def d2v_inferVector(searchTerms):
    activationVector = doc2vec_model.infer_vector(doc_words=searchTerms.split(" "), epochs=200)
    activationVector = semantic_message_array.reshape(1, -1)
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
    aggSimTable['similarity'] = (aggSimTable["Doc2Vec"] + aggSimTable["BoW"])/2
    aggSimTable.sort_values(by="aggCosSim", ascending=False, inplace=True)
    return aggSimTable.head(topN)

getSupplementNames = lambda df: list(df.index)

################################################################################

#### Load data
supplementsScrapped = loadScrappedData()
## Since posts are stored into lists of multiple categories per supplement, flatten these lists
supplementsScrapped = {k : flatten(supplementsScrapped[k].values()) for k in supplementsScrapped.keys()}
# Each supplement should be stored into one dict key
############

#### Clean
## Remove leftover HTML code in text
scrappedClean = {key:[regexClean(_) for _ in val] for key, val in supplementsScrapped.items()}
# Convert back to list
scrappedClean = {k:' '.join(v) for k,v in scrappedClean.items()}
# tokenize, make lowecase, remove too short or too long words 
scrappedClean = preProcessText(scrappedClean)
scrappedClean = {k:removeStopWords(v,stop_words) for k,v in scrappedClean.items()}
############


### Convert to DF to make it easy to iterate and get indices
supplementDF = pd.DataFrame.from_dict(scrappedClean,orient='index',columns=['full'])
###################### BoW approach
#### Fit a TF-IDF
# get vocabulary and tfidf from all strings.
tf = TfidfVectorizer(analyzer='word', 
                     min_df=5,
                     ngram_range=(2,3),
                     #max_features=1000,
                     stop_words='english')
tf.fit(supplementDF['full'])

#Transform style_id products to document-term matrix.
tfidf_matrix = tf.transform(supplementDF['full'])

## Save
saveData(tf,"tf") #Save the tf model
saveData(tfidf_matrix,"tfidf_matrix") #Save the matrix
############


#### Dimension reduction for BoW approach
svd = TruncatedSVD(n_components=500)
svdMatrix = svd.fit_transform(tfidf_matrix)
## Save
saveData(svdMatrix,"svdMatrix")
saveData(svd,"svd")

### Get the feature matrix as a dataframe
# this will be used to compute similarities later
n = 25 
supplements = supplementDF.index
svd_feature_matrix = pd.DataFrame(svdMatrix[:,0:n] ,index=supplements)

saveData(svd_feature_matrix,"svd_feature_matrix")
############


###################### Doc2Vec approach
### Doc2Vec model: basically a shallow NN

# Prep

supplementDF["listedDocs"] = [_.split() for _ in supplementDF["full"]]
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

####
############


## Create UMAP representation of supp distances
reducer = umap.UMAP(n_neighbors=23) 
x = doctovec_feature_matrix
docEmbedding = reducer.fit_transform(x)
umap_doc = pd.DataFrame(docEmbedding,index=doctovec_feature_matrix.index,columns=["UMAP1","UMAP2"])
saveData(umap_doc,"umap_doc")

## Create UMAP representation of word distances
VectorsPerWord=pd.DataFrame(doc2vec_model.wv.vectors,index=doc2vec_model.wv.vocab.keys())
reducer = umap.UMAP(n_neighbors=23) 
x = VectorsPerWord
wordvecsEmbedding = reducer.fit_transform(x)
umap_wordvecs = pd.DataFrame(wordvecsEmbedding,index=VectorsPerWord.index,columns=["UMAP1","UMAP2"])
saveData(umap_wordvecs,"umap_wordvecs")

