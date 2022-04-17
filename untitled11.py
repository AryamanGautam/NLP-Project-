
# cd Desktop\Sem 6\NLP project
# streamlit run .\untitled7.py

import time
from PIL import Image
import streamlit as st

import random
import transformers
import sentencepiece
import textwrap3
import nltk
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
import numpy as np
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('s2v_old')
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

#To ensure same output every time
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final
#Just for the purpose of capitalising the first letter of the start of every scentence


def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary


def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'] #deal with paranthesis etc
        # DIfferent types of brackets (){}[] respectively
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        #    alpha controls the weight adjustment mechanism, see TopicRank for threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out



def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
  print ("keywords unsummarized: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))
  print ("keywords_found in summarized: ",keywords_found)

  important_keywords =[]
  for keyword in keywords:
    if keyword in keywords_found:
      important_keywords.append(keyword)

  return important_keywords[:4]

question_model = T5ForConditionalGeneration.from_pretrained('C:/Users/Usha Gautam/Desktop/Sem 6/NLP project/t5/model/')
question_tokenizer = T5Tokenizer.from_pretrained('C:/Users/Usha Gautam/Desktop/Sem 6/NLP project/t5/tokenizer/')
question_model = question_model.to(device)

def get_question(context,answer,model,tokenizer):
  text = "context: {} answer: {}".format(context,answer)
  encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question





from sentence_transformers import SentenceTransformer
sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
#A scentence tranformer model, it maps scentences and pararaphs to a 768 dimensional dense 
#vector space, used for tasks like clustering & semantic search.

from similarity.normalized_levenshtein import NormalizedLevenshtein
normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(original,wordlist):
  filtered_words=[]
  base_sense =original.split('|')[1] 
  print (base_sense)
  for eachword in wordlist:
    if eachword[0].split('|')[1] == base_sense:
      filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
  return filtered_words

def get_highest_similarity_score(wordlist,wrd):
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def sense2vec_get_words(word,s2v,topn,question):
    output = []
    print ("word ",word)
    try:
      sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
      most_similar = s2v.most_similar(sense, n=topn)
      # print (most_similar)
      output = filter_same_sense_words(sense,most_similar)
      print ("Similar ",output)
    except:
      output =[]

    threshold = 0.6
    final=[word]
    checklist =question.split()
    for x in output:
      if get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
        final.append(x)
    
    return final[1:]

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def get_distractors_wordnet(word):
    distractors=[]
    try:
      syn = wn.synsets(word,'n')[0]
      
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

def get_distractors (word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
  distractors = sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
  print ("distractors ",distractors)
  if len(distractors) ==0:
    return distractors
  distractors_new = [word.capitalize()]
  distractors_new.extend(distractors)
  # print ("distractors_new .. ",distractors_new)

  embedding_sentence = origsentence+ " "+word.capitalize()
  # embedding_sentence = word
  keyword_embedding = sentencemodel.encode([embedding_sentence])
  distractor_embeddings = sentencemodel.encode(distractors_new)

  # filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors,4,0.7)
  max_keywords = min(len(distractors_new),5)
  filtered_keywords = mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
  # filtered_keywords = filtered_keywords[1:]
  final = [word.capitalize()]
  for wrd in filtered_keywords:
    if wrd.lower() !=word.lower():
      final.append(wrd.capitalize())
  final = final[1:]
  return final

def get_distractors_wordnetforuse(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


placeholder = st.empty()

with placeholder.container():
    st.title("Question Generation")
    
    image = Image.open('mcq.png')

    st.image(image)

    st.subheader("Given an Input of text we intend on providing MCQs")
        
    data = st.text_input("Enter your text :  ")
    
    btn = st.button("Proceed")

input_txt = data


if btn:
    time.sleep(2)
    placeholder.empty()
    
    
    placeholder2 = st.empty()
    with placeholder2.container():
        summarized_text = summarizer(input_txt,summary_model,summary_tokenizer)
        imp_keywords = get_keywords(input_txt,summarized_text)
        final_qs = []
        final_ans = []
        final_other_opns = []
        for answer in imp_keywords:
            ques = get_question(summarized_text,answer,question_model,question_tokenizer)
            final_qs.append(ques)
            final_ans.append(answer.capitalize()) 
        
            
        for i in range(len(final_qs)):
            sent = final_qs[i]
            keyword = final_ans[i]
            a = get_distractors(keyword,sent,s2v,sentence_transformer_model,40,0.2)
            
            if a == []:
                word = keyword
                word = word.lower()
                syns = wn.synsets(word,'n')
                
                opns = get_distractors_wordnet(keyword)[:4]
            else:
                opns = a
            final_other_opns.append(opns)

        st.title("Output : ")
        st.subheader("")
        st.subheader("")
        st.subheader("Original : ")
        st.subheader("")
        st.subheader("")
        st.text(input_txt)
        st.subheader("")
        st.subheader("")
        st.subheader("Summarised : ")
        st.subheader("")
        st.subheader("")
        st.text(summarized_text)
        st.subheader("")
        st.subheader("")
        st.subheader("Important Keywords : ")
        st.subheader("")
        st.subheader("")
        st.text(imp_keywords)
        for itrt in range(len(final_qs)):
            tmp1 = final_other_opns[itrt]
            tmp2 = final_ans[itrt]
            fin_ans_op = tmp1 + [tmp2]
            random.shuffle(fin_ans_op)
            option = st.selectbox(final_qs[itrt],fin_ans_op)

        st.title("Answers")
        for itrt in range(len(final_qs)):
            st.text(final_qs[itrt])
            st.text(final_ans[itrt])
            




