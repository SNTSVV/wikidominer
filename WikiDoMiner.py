import wikipediaapi, calendar, spacy, nltk, os, re, en_core_web_md, pickle, sys, argparse
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download(['punkt','stopwords','wordnet','omw-1.4'],quiet=True)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PyPDF2 import PdfReader
import wikipedia as wiki

wp = wikipediaapi.Wikipedia('en')


def main():
    parser = argparse.ArgumentParser(description='anaphora')
    parser.add_argument('--doc', dest='doc', type=str, help='Path to an input requirements document in txt or pdf format.')
    parser.add_argument('--output-path', dest='path', type=str, default='./', help='set the path where to save the extracted corpus')
    parser.add_argument('--output-dir', dest='dir', type=str, default='Corpus', help='set the folder name of the corpus')
    parser.add_argument('--include-nouns', dest='include_nouns', type=bool, default=False, help='Set to true to include nouns as keywords.')

    parser.add_argument('--title-similarity', dest='title_similarity', type=bool, default=False, help='filter using similarity between the keywords and the matching titles')
    parser.add_argument('--sim_threshold', dest='sim_threshold', type=float, default=0.5, help='set similarity threshold when using title similarity')
    parser.add_argument('--filter-cats', dest='filter_cats', type=bool, default=True, help='filter generic categories')
    parser.add_argument('--use-tfidf', dest='tfidf', type=bool, default=True, help='use TFIDF to rank keywords')
    parser.add_argument('--limit-keywords', dest='K', type=int, default=None, help='Set to limit the number of extracted keywords if TFIDF is used.')
    parser.add_argument('--wiki-depth', dest='depth', type=int, default=0, help='Set the depth of wikiedia search. 0: only the articles that matches the keywords. 1: articles from the the categories of matching keywords. 2 articles from the subcategories..')
    parser.add_argument('--max-limit', dest='maxlimit', type=int, default=200, help='set limit to filter large categories')
    parser.add_argument('--max-cats', dest='maxcats', type=int, default=50, help='skip articles with too many categories')
    parser.add_argument('--wiki-auto-suggest', dest='auto_suggest', type=bool, default=False, help='allow auto suggest from wikipedia if the keyword doesn\'t match any article')
    parser.add_argument('--wordcloud', dest='wc', type=bool, default=False, help='create a wordcloud of the created corpus')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=True, help='activate the running information.')
    args = parser.parse_args()
    
    nlp = en_core_web_md.load()

    tfidf=args.tfidf
    if args.doc.split('.')[-1].lower()=='txt':
      doc = ' '.join(open(args.doc))
    elif args.doc.split('.')[-1].lower()=='pdf':
      doc = readPDF(args.doc)
    else:
      print('Invalid input file', args.doc ,'Please input a txt or a pdf file')
      return

    if args.tfidf:
      tfidf=buildTFIDFvector([doc]).loc[0]
    else:
      tfidf=[]

    keywords = getKeywords(doc,nlp,include_nouns=args.include_nouns,tfidf=tfidf,K=args.K) 
    if args.verbose:
      print(len(keywords),' keywords extracted')
    if args.filter_cats:
      filters=['Articles','Commons','WikiData','Wikipedia','Webarchive','disputes','bot:','CS1','errors','pages','births','deaths','disambiguation','elements']+[calendar.month_name[i] for i in range(1,12)]+[calendar.month_abbr[i] for i in range(1,12)]
    else: 
      filters=[]
    corpus = getCorpus(keywords, title_similarity=args.title_similarity,sim_threshold=args.sim_threshold,filtered_cats=filters, auto_suggest = args.auto_suggest, depth=args.depth,maxcats=args.maxcats, maxlimit=args.maxlimit, verbose=args.verbose)
    getTotal_nbr_words(corpus)
    if args.wc:
      createWordCloud(corpus,nlp)
    saveCorpus(corpus,args.path,folder=args.dir)

def calculate_jaccard(text1,text2):  # calculates jaccard similarity between two string
  word_tokens1=word_tokenize(text1.lower())
  word_tokens2=word_tokenize(text2.lower())
  both_tokens = word_tokens1 + word_tokens2
  union = set(both_tokens)
  # Calculate intersection.
  intersection = set()
  for w in word_tokens1:
    if w in word_tokens2:
      intersection.add(w)
  jaccard_score = len(intersection)/len(union)
  return jaccard_score

def stemlemma(text):
  return ' '.join([stemmer.stem(wordnet_lemmatizer.lemmatize(word)) for word in word_tokenize(text.lower())])

def openFiles(files,path):
  li=[]
  for f in files:
    with open(path+f,"r") as tf:
      li.append(tf.read().replace('\n', ''))
  return li

# get all nouns and noun phrases from the input sentence
def getAllNPsFromSent(sent,include_nouns=False):
    npstr=[]
    chunks = list(sent.noun_chunks)
    for i in range(len(chunks)):
        np=chunks[i]
        if len(np)==1:
            if np[0].pos_!="NOUN":
                continue
        if np.text.lower() not in npstr:
            npstr.append(np.text.lower())      
        if i < len(chunks)-1:
            np1=chunks[i+1]
            if np1.start-np.end==1:
                if sent.doc[np.end].tag_=="CC":
                    newnp = sent[np.start:np1.end]
                    if newnp.text.lower() not in npstr:
                        npstr.append(newnp.text.lower())
    if include_nouns:
        for t in sent:
            if "subj" in t.dep_ and t.pos_=="NOUN": 
                if t.text.lower() not in npstr:
                    npstr.append(t.text.lower())
    return npstr    

def getTopK(di,K=50):
  tempdf=pd.DataFrame.from_dict(di,columns=["tfidf"], orient='index')
  return list(tempdf.sort_values(by=['tfidf'],ascending=False)[:K].index)

def getKeywords(doc,nlp,include_nouns=False,tfidf=[],K=None): # K: free parameter
  keywords=[]
  for s in doc.split('\n'):
    s=nlp(s)
    keywords.extend([n.text for n in list(s.ents)])
    keywords.extend(list(getAllNPsFromSent(s,include_nouns)))
  
  keywords=list(set(keywords))

  keywords_wn={}

  if len(tfidf)>0:
    tfidf_threshold=np.mean([t for t in tfidf if t>0])

  for k in keywords:
    keyword=' '.join([word for word in word_tokenize(k) if not word.lower() in stopwords.words('english')])
    if not wn.synsets(keyword) and keyword.replace(' ','').isalpha() and not keyword.isupper() and not np.array([k.isupper() for k in [ky[:-1] for ky in keyword.split()]]).any():
      keyword=keyword.lower()
      if len(tfidf)>0:
        if stemlemma(keyword) in tfidf.index:# and len(keyword)>2:
          if tfidf[stemlemma(keyword)]>tfidf_threshold:
            if keyword not in keywords_wn:
              keywords_wn[keyword]=tfidf[stemlemma(keyword)]
            else:
              keywords_wn[keyword]=max(keywords_wn[keyword],tfidf[stemlemma(keyword)])
      else:
        keywords_wn[keyword]=0

  if K and len(tfidf)>0:
    return getTopK(keywords_wn,K=K)

  else:
    return list(keywords_wn.keys())

def buildTFIDFvector(docs,use_ngrams=True,ngrams=4):
  if use_ngrams:
    vectorizer = TfidfVectorizer(ngram_range=(1,ngrams),min_df=0,stop_words=stopwords.words('english'))
  else:
    vectorizer = TfidfVectorizer(min_df=0,stop_words=stopwords.words('english'))
  vectors = vectorizer.fit_transform(docs)
  return pd.DataFrame(vectors.todense().tolist(), columns=vectorizer.get_feature_names_out())
def buildTFIDF(domains,files,use_ngrams=True,ngrams=3):
  docs={}
  for d in domains:
    docs[d]=stemlemma(' '.join([files[doc] for doc in domains[d]]))
  return buildTFIDFvector(list(docs.values()),use_ngrams=use_ngrams,ngrams=ngrams)

def getTFIDFscore(q,id,tfidf):
  score=0
  for t in q.split():
    if t in tfidf[q].columns:
      score+=tfidf[q][t][id]
  return score  


def get_articles_in_category(cat_title, category,filtered_cats, level=1, max_level=2,maxlimit=200, verbose=False):
  articles=[]
  try:
    categorymembers=category.categorymembers
    if len(categorymembers)>maxlimit:
      return articles
    if verbose:
              print("\t== Extracting articles from the category:",cat_title)
    for cat_title, c in tqdm(categorymembers.items(), disable=not verbose):

      if c.ns != wikipediaapi.Namespace.CATEGORY:
        try:
          articles.append(c.text)
        except:
          articles.append(wiki.page(cat_title).content)
      elif level < max_level and not match(cat_title, filtered_cats):
          articles.extend(get_articles_in_category(c,filtered_cats, level=level + 1, max_level=max_level))
  except Exception as e: 
    print(e)
  return articles

def getCorpusFromTitle(title,filtered_cats,depth=1,maxcats=10, maxlimit=200,verbose=False):
  corpus= [] # here we store extracted articles
  page = wp.page(title) # get the page that matches the title
  if page:
    try:
      try:
        corpus.append(page.text) # add matched page to the corpus list
      except:
        corpus.append(wiki.page(wiki.search(title)[0]).content)
    except:
      None
    # browse the categories of the page
    try:
      cats=page.categories
    except:
      return corpus
    if depth>0:
      if len(cats)<maxcats: #max number of categories
        for cat_title, category in cats.items(): # There are some generic categories that we want to filter out (e.g, Category:articles from August 2019).
          if not match(cat_title, filtered_cats): 
            # depth=1 get all articles in each category, depth=2: include the articles in subcategories, depth=3: include the articles in subsubcategories. 
            corpus.extend(get_articles_in_category(cat_title, category,filtered_cats, max_level=depth,maxlimit=maxlimit,verbose=verbose)) 
  return list(set(corpus))

def match(title,filters):
  for filter in filters:
    if filter.lower() in title.lower():
      return True
  return False

def getCorpus(list_of_keywords, title_similarity=False,sim_threshold=0.5,filtered_cats=[], auto_suggest = False, maxcats=10, maxlimit=200, depth=1,verbose=False):
  processed_titles=[] # we store processed titles of wikipedia articles to avoid processing them more than once
  corpus= [] # here we store extracted articles
  c=0
  for keyword in tqdm(list_of_keywords, disable= verbose):
    c+=1
    #if verbose:
    #  print('=== Processing keyword:',keyword, c,'/',len(list_of_keywords))
    # we search for the closest titles matching our keyword
    if auto_suggest:
      matching_titles=wiki.search(keyword,suggestion=True)  
      if not matching_titles:
        continue
      for title in matching_titles:
        # you can add a similarity criteria between the keyword and the matching article before proceeding
        # for example use jaccard with a threshold: if calculate_jaccard(keyword,title)>0.5
        if title not in processed_titles:
          if title_similarity:
            if calculate_jaccard(title,keyword)<sim_threshold:
              continue
          corpus.extend(getCorpusFromTitle(title,filtered_cats,depth=depth,maxcats=maxcats, maxlimit=maxlimit,verbose=verbose))
          title.append(processed_titles)
    else:
      corpus.extend(getCorpusFromTitle(keyword,filtered_cats,depth=depth,maxcats=maxcats, maxlimit=maxlimit,verbose=verbose))
  return list(set(corpus))

def saveCorpus(docs,parent_dir,folder='Corpus'):
  path = os.path.join(parent_dir, folder)
  os.makedirs(path,exist_ok=True)
  for i in range(0,len(docs)):
    doc=docs[i]    
    filename='doc'+str(i)+'.txt'
    filepath = os.path.join(path, filename)
    text_file = open(filepath, "w")
    n = text_file.write(doc)
    text_file.close()

def docSimilarity(doc1,doc2,nlp):
  return nlp(doc1).similarity(nlp(doc2))

def getTotal_nbr_words(corpus): 
  total_nbr_words=0
  for article in corpus:
    total_nbr_words+= len(word_tokenize(article))
  
  print("total number of words:",total_nbr_words)
  return total_nbr_words

def createWordCloud(corpus,nlp,show=False,image_name='Word Cloud'):
  WC=WordCloud(stopwords=set(nlp.Defaults.stop_words), #width = 1000, height = 500,
                        max_font_size=50, max_words=100,background_color="white")
  wordcloud = (WC.generate(' '.join(corpus)))

  plt.figure(figsize=(15,8))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.savefig(image_name+".png", bbox_inches='tight')
  if show:
    plt.show()
  plt.close()

def readPDF(file):
  reader = PdfReader(file)
  return ' '.join([re.sub(r"\s+", " ",page.extract_text().replace('\n',' ')).strip() for page in reader.pages])

def simCheck(doc,corpus,nlp):
  doc = nlp(doc)
  c=[]
  for article in corpus:
    sim=doc.similarity(nlp(article))
    if sim>0:
      c.append(sim)
  # print(min(c),np.average(c),max(c))
  return min(c),np.average(c),max(c)


def simCheckV2(doc1,doc2,corpus,nlp):
  doc1 = nlp(doc1)
  doc2 = nlp(doc2)
  c1=[]
  c2=[]
  for article in tqdm(corpus):
    article=nlp(article)
    sim1=doc1.similarity(article)
    sim2=doc2.similarity(article)
    if sim1>0:
      c1.append(sim1)
    if sim2>0:
      c2.append(sim2)
  c3=c1+c2
  return min(c3),np.average(c3),max(c3)

def docSimilarity(doc1,doc2,nlp):
  return nlp(doc1).similarity(nlp(doc2))


def saveObj(ob,filename):
    filehandler = open(filename, 'wb') 
    pickle.dump(ob, filehandler)
    filehandler.close()

if __name__ == '__main__':
    sys.exit(main())
