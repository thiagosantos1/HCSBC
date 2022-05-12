from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords
import re 
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
import pandas as pd 

def remove_noise_text(txt):

  txt = txt.lower()
  txt = re.sub('right|left', '', txt) # remove right/left spaces
  txt = re.sub("primary site:", '', txt)

  #txt = re.sub('post-surgical changes', ' ', txt.lower()) 

  # Remove any mentions to " Findings were discussed with...."
  txt = txt.split("findings were discussed with")[0] 

  # Remove any other occurance of PI's Information
  txt = txt.split("this study has been reviewed and interpreted")[0] 
  txt = txt.split("this finding was communicated to")[0] 
  txt = txt.split("important findings were identified")[0] 
  txt = txt.split("these findings")[0] 
  txt = txt.split("findings above were")[0] 
  txt = txt.split("findings regarding")[0] 
  txt = txt.split("were discussed")[0] 
  txt = txt.split("these images were")[0] 
  txt = txt.split("important finding")[0] 

  # remove any section headers
  txt = re.sub("post-surgical changes:", '', txt)
  txt = re.sub("post surgical changes:", '', txt)
  txt = re.sub("primary site:", '', txt)
  txt = re.sub("primary site", '', txt)
  txt = re.sub("neck:", '', txt)
  txt = re.sub("post-treatment changes:", '', txt)
  txt = re.sub("post treatment changes:", '', txt)
  txt = re.sub("brain, orbits, spine and lungs:", '', txt)
  txt = re.sub("primary :", '', txt)
  txt = re.sub("neck:", '', txt)
  txt = re.sub("aerodigestive tract:", '', txt)
  txt = re.sub("calvarium, skull base, and spine:", '', txt)
  txt = re.sub("other:", '', txt)
  txt = re.sub("upper neck:", '', txt)
  txt = re.sub("perineural disease:", '', txt)
  txt = re.sub("technique:", '', txt)
  txt = re.sub("comparison:", '', txt)
  txt = re.sub("paranasal sinuses:", '', txt)
  txt = re.sub("included orbits:", '', txt)
  txt = re.sub("nasopharynx:", '', txt)
  txt = re.sub("tympanomastoid cavities:", '', txt)
  txt = re.sub("skull base and calvarium:", '', txt)
  txt = re.sub("included intracranial structures:", '', txt)
  txt = re.sub("abnormal enhancement:", '', txt)
  txt = re.sub("lymph nodes:", '', txt)
  txt = re.sub("impression:", '', txt)
  txt = re.sub("nodes:", '', txt)
  txt = re.sub("mri orbits:", '', txt)
  txt = re.sub("mri brain:", '', txt)
  txt = re.sub("brain:", '', txt)
  txt = re.sub("ct face w/:", '', txt)
  txt = re.sub("transspatial extension:", '', txt)
  txt = re.sub("thyroid bed:", '', txt)
  txt = re.sub("additional findings:", '', txt)
  txt = re.sub("series_image", '', txt) 
  txt = re.sub("series image", '', txt)
  txt = re.sub("image series", '', txt)
  txt = re.sub("series", '', txt)

  txt = re.sub(" mm | mm|mm ", " ", txt)
  txt = re.sub(" series | series|series ", "", txt)
  txt = re.sub(" cm | cm|cm ", " ", txt)
  txt = re.sub(" cc | cc|cc ", " ", txt)
  txt = re.sub(" ct | ct|ct ", " ", txt)
  txt = re.sub(" mri | mri|mri ", " ", txt)
  txt = re.sub(" see | see|see ", " ", txt)
  txt = re.sub(" iia | iia|iia ", " ", txt)
  txt = re.sub("comment", "", txt)


  txt = re.sub("post treatment", '', txt)
  txt = re.sub("post_treatment", '', txt)
  txt = re.sub("post-treatment", '', txt)
  txt = re.sub("findings suggest", '', txt)
  txt = re.sub("findings", '', txt)
  txt = re.sub("suggest", '', txt)
  txt = re.sub("study reviewed", '', txt)
  txt = re.sub("study", '', txt)
  txt = re.sub("reviewed", '', txt)
  txt = re.sub("please see", '', txt)
  txt = re.sub("please", '', txt)
  
  txt = re.sub("skull base", '', txt)
  txt = re.sub("fdg avid", '', txt)
  txt = re.sub("fdg aivity", '', txt)
  txt = re.sub("please see chest ct for further evaluation of known lung mass", '', txt)
  
  txt = re.sub("status_post", '', txt)
  txt = re.sub("status post|clock|/|'/'", '', txt)
  txt = re.sub("statuspost|:", '', txt)
  txt = re.sub(" cm | cm|cm ", " centimeters ", txt)
  txt = re.sub(" cc | cc|cc ", " cubic centimeters ", txt)
  txt = re.sub(" ct | ct|ct ", " carat metric ", txt)
  txt = re.sub(" mm | mm|mm ", " millimeters ", txt)
  #txt = re.sub("(\\d*\\.\\d+)|(\\d+\\.[0-9 ]+)","",txt)

  # in the worst case, just replace the name from PI to empty string
  txt = re.sub("dr\\.\\s[^\\s]+", '', txt)

  
  txt = re.sub('\\;', ' .', txt)  
  txt = re.sub('\\.', ' .', txt)  

  # Removing multiple spaces
  txt = re.sub(r'\s+', ' ', txt)


  return txt

def add_bigrams(txt, fixed_bigrams):

  for b in fixed_bigrams:
    sub = ""
    not_first = False
    for x in b[1:]:
      if not_first:
        sub += "|"
        not_first = True

      sub += str(x) + "|" + str(x) + " " + "|" +  " " + str(x) + "|" + " " + str(x)   
    txt = re.sub(sub, b[0], txt)

  return txt


def clean_text(txt_orig,filters,stop_words,non_stop_words,freq_words,fixed_bigrams,steam, lemma , clean, min_lenght, eightify=False):
  txt = remove_noise_text(txt_orig)

  #print("\n\t\tOriginal\n", txt)
  txt = add_bigrams(txt, fixed_bigrams)
  #print("\n\t\tCleaned\n", txt)
  words = preprocessing.preprocess_string(txt, filters)
  words = add_bigrams(" ".join(w for w in words), fixed_bigrams).split()

  txt = " ".join(w for w in words)
  
  # eightify 
  # 
  if eightify:
    replaces = [ ["her2|her 2|her two", " hertwo "], ["0", "8"], ["1", "8"], ["2", "8"], ["3", "8"],["4", "8"],
                ["5", "8"],["6", "8"] ,["7", "8"] ,["8", "8"] ,["9", "8"] ,
                ["\\>", " greather "], ["\\<", " less "]]

  else:
    replaces = [ ["her2|her 2|her two", " hertwo "], ["0", "zero "], ["1", "one "], ["2", "two "], ["3", "three "],["4", "four "],
                ["5", "five "],["6", "six "] ,["7", "seven "] ,["8", "eight "] ,["9", "nine " ] ,
                ["\\>", " greather "], ["\\<", " less "]]


  for sub in replaces:
    txt = re.sub(sub[0], sub[1], txt)

  # Removing multiple spaces
  txt = re.sub(r'\s+', ' ', txt)

  words = txt.split()

  if clean:
    words = [w for w in words if (not w in stop_words and re.search("[a-z-A-Z]+\\w+",w) != None and (len(w) >min_lenght or w in non_stop_words) or w=='.') ] 
  else:
    words = [w for w in words if (re.search("[a-z-A-Z]+\\w+",w) != None and (len(w) >min_lenght or w in non_stop_words) or w=='.')] 

  c_words = words.copy()

  if steam:
    porter = PorterStemmer()
    c_words = [porter.stem(word) for word in c_words if not porter.stem(word) in freq_words and (len(porter.stem(word)) >min_lenght or word in non_stop_words or word=='.')]

  if lemma:
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    c_words = [lem.lemmatize(word) for word in c_words if not lem.lemmatize(word) in freq_words and (len(lem.lemmatize(word)) >min_lenght or word in non_stop_words or word=='.')]

  return c_words
  

def text_cleaning(data, steam=False, lemma = True, clean=True, min_lenght=2, remove_punctuation=True,
                  freq_words_analysis=False, single_input=False,eightify=True):

  clean_txt = []


  freq_words = ["breast","biopsy","margin","dual","tissue","excision","change","core","identified",
                "mastectomy","site","report","lesion","superior","anterior","inferior","medial",
                "lateral","synoptic","evidence","slide", "brbx"]

                    # position 0 means the bigram output - 1:end means how they may come on text
  fixed_bigrams = [ [' grade_one ', 'grade 1', 'grade i', 'grade I', 'grade one',],
                    [' grade_two ', 'grade 2', 'grade ii', 'grade II', 'grade two', ],
                    [' grade_three ', 'grade 3' , 'grade iii', 'grade III', 'grade three']]


  if remove_punctuation:
    filters = [lambda x: x.lower(), strip_tags, strip_punctuation]
  else:
    filters = [lambda x: x.lower(), strip_tags]
    
  stop_words = set(stopwords.words('english'))
  non_stop_words = ['no', 'than', 'not']
  for x in non_stop_words:
    stop_words.remove(x)

  if single_input:
    c_words = clean_text(data,filters,stop_words,non_stop_words,freq_words,fixed_bigrams,steam, lemma, clean, min_lenght,eightify=eightify)
    if len(c_words)>0:
      if c_words[0] =='.':
        c_words = c_words[1:]
    clean_txt.append(c_words)

  else:
    for i in range(data.shape[0]):
      txt_orig = data.iloc[i].lower()
      c_words = clean_text(txt_orig,filters,stop_words,non_stop_words,freq_words,fixed_bigrams,steam, lemma, clean, min_lenght,eightify=eightify)
      if len(c_words)>0:
        if c_words[0] =='.':
          c_words = c_words[1:]
      clean_txt.append(c_words)


  if freq_words_analysis:
    flatten_corpus = [j for sub in clean_txt for j in sub] 
    clean_txt = []
    unique = list(set(flatten_corpus))
    wordfreq = [flatten_corpus.count(p) for p in unique]
    wordfreq =  dict(list(zip(unique,wordfreq)))

    freqdict = [(wordfreq[key], key) for key in wordfreq]
    freqdict.sort()
    freqdict.reverse()

    df = pd.DataFrame(freqdict,columns = ['Frequency','Word']) 


    df.to_excel('../mammo_word_count.xls')
    
  return clean_txt

if __name__ == '__main__':
  exit()

