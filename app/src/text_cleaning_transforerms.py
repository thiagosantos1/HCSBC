import pandas as pd
from os import walk
from os import listdir
from os.path import isfile, join
import numpy as np
import re 

from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
import math
from tqdm import tqdm

def remove_noise_text(txt):

  txt = txt.lower()
  txt = re.sub("primary site:", ' ', txt)

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
  txt = re.sub("post-surgical changes:", ' ', txt)
  txt = re.sub("post surgical changes:", ' ', txt)
  txt = re.sub("primary site:", ' ', txt)
  txt = re.sub("primary site", ' ', txt)
  txt = re.sub("neck:", ' ', txt)
  txt = re.sub("post-treatment changes:", ' ', txt)
  txt = re.sub("post treatment changes:", ' ', txt)
  txt = re.sub("brain, orbits, spine and lungs:", ' ', txt)
  txt = re.sub("primary :", ' ', txt)
  txt = re.sub("neck:", ' ', txt)
  txt = re.sub("aerodigestive tract:", ' ', txt)
  txt = re.sub("calvarium, skull base, and spine:", ' ', txt)
  txt = re.sub("other:", ' ', txt)
  txt = re.sub("upper neck:", ' ', txt)
  txt = re.sub("perineural disease:", ' ', txt)
  txt = re.sub("technique:", ' ', txt)
  txt = re.sub("comparison:", ' ', txt)
  txt = re.sub("paranasal sinuses:", ' ', txt)
  txt = re.sub("included orbits:", ' ', txt)
  txt = re.sub("nasopharynx:", ' ', txt)
  txt = re.sub("tympanomastoid cavities:", ' ', txt)
  txt = re.sub("skull base and calvarium:", ' ', txt)
  txt = re.sub("included intracranial structures:", ' ', txt)
  txt = re.sub("impression:", ' ', txt)
  txt = re.sub("nodes:", ' ', txt)
  txt = re.sub("mri orbits:", ' ', txt)
  txt = re.sub("mri brain:", ' ', txt)
  txt = re.sub("brain:", ' ', txt)
  txt = re.sub("ct face w/:", ' ', txt)
  txt = re.sub("transspatial extension:", ' ', txt)
  txt = re.sub("thyroid bed:", ' ', txt)
  txt = re.sub("additional findings:", ' ', txt)
  
  txt = re.sub("brstwo|brstmarun|brstwln|brlump|lnbx", ' ', txt)
  
  txt = re.sub("post_treatment", 'post treatment', txt)
  txt = re.sub("post-treatment", 'post treatment', txt)

  txt = re.sub("nonmasslike", 'non mass like', txt)
  txt = re.sub("non_mass_like", 'non mass like', txt) 
  txt = re.sub("non-mass-like", 'non mass like', txt)
  txt = re.sub("statuspost", 'status post', txt)


  # in the worst case, just replace the name from PI to empty string
  txt = re.sub("dr\\.\\s[^\\s]+", ' ', txt)  

  txt = re.sub(" series | series|series ", "", txt)
  txt = re.sub(" cm | cm|cm ", " centimeters ", txt)
  txt = re.sub(" cc | cc|cc ", " cubic centimeters ", txt)
  txt = re.sub(" ct | ct|ct ", " carat metric ", txt)
  txt = re.sub(" mm | mm|mm ", " millimeters ", txt)
  
  txt = re.sub("status_post|o\'", '', txt)
  txt = re.sub("status post|clock|/|'/'", '', txt)
  txt = re.sub("statuspost", '', txt)
  txt = re.sub("brstwo|brlump|brstmarun|brwire|brstcap|", '', txt)

  txt = re.sub("\\(|\\)", ',', txt)
  txt = re.sub(",,", ',', txt)
  txt = re.sub(",\\.", '.', txt)
  txt = re.sub(", \\.", '.', txt)

  txt = re.sub(" ,", ', ', txt)
  txt = re.sub("a\\.", ' ', txt[0:5]) + txt[5:]
  txt = re.sub("b\\.", ' ', txt[0:5]) + txt[5:]
  txt = re.sub("c\\.", ' ', txt[0:5]) + txt[5:]
  txt = re.sub("d\\.", ' ', txt[0:5]) + txt[5:]
  txt = re.sub("e\\.", ' ', txt[0:5]) + txt[5:]
  txt = re.sub("f\\.", ' ', txt[0:5]) + txt[5:]
  

  # in the worst case, just replace the name from PI to empty string
  txt = re.sub("dr\\.\\s[^\\s]+", '', txt)

  # Removing multiple spaces
  txt = re.sub(r'\s+', ' ', txt)
  txt = re.sub(' +', ' ', txt)

  txt = txt.rstrip().lstrip()

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

def extra_clean_text(clean_t,fixed_bigrams):

  txt = add_bigrams(clean_t, fixed_bigrams)
  replaces = [ ["her2|her 2|her two", " hertwo "], 
              # ["0", "zero "], ["1", "one "], ["2", "two "], ["3", "three "],["4", "four "],
              # ["5", "five "],["6", "six "] ,["7", "seven "] ,["8", "eight "] ,["9", "nine " ] ,
              ["\\>", " greather "], ["\\<", " less "]]

  for sub in replaces:
    txt = re.sub(sub[0], sub[1], txt)

  return txt


def text_cleaning(data,min_lenght=2,extra_clean=True, remove_punctuation=False):

                  # position 0 means the bigram output - 1:end means how they may come on text
  fixed_bigrams = [ [' gradeone ', 'grade 1', 'grade i', 'grade I', 'grade one',],
                    [' gradetwo ', 'grade 2', 'grade ii', 'grade II', 'grade two', ],
                    [' gradethree ', 'grade 3' , 'grade iii', 'grade III', 'grade three']]

  clean_txt = []

  clean_t = remove_noise_text(data)
  if extra_clean:
    clean_t = extra_clean_text(clean_t,fixed_bigrams)
    if remove_punctuation:
      filters = [lambda x: x.lower(), strip_tags, strip_punctuation]
    else:
      filters = [lambda x: x.lower(), strip_tags]

    clean_t = " ".join(x for x in preprocessing.preprocess_string(clean_t, filters) if len(x) >=min_lenght)


  # Removing multiple spaces
  clean_t = re.sub(r'\s+', ' ', clean_t)

  return clean_t

def split_by_chuncks(data,min_lenght=2,max_size=64, extra_clean=True, remove_punctuation=False):
  pre_processed_chunks = []
  words = word_tokenize(data)
  lower_b, upper_b = 0, max_size
  for x in range(math.ceil(len(words)/max_size)):
    sample = " ".join(x for x in words[lower_b:upper_b])
    lower_b, upper_b = upper_b, upper_b+max_size
    clean_data = text_cleaning(sample,min_lenght=min_lenght,extra_clean=extra_clean, remove_punctuation=remove_punctuation)

    pre_processed_chunks.append(clean_data)

  return pre_processed_chunks

# set only_data = True if no need to get scores or if dataaset doesn't have a score
def pre_process(data,min_lenght=2,max_size=64, extra_clean=True, remove_punctuation=False): 


  data_pre_processed = text_cleaning(data,min_lenght=min_lenght,extra_clean=extra_clean, remove_punctuation=remove_punctuation)

  """
    Partion the data into max_size chunks
  """
  sentences = sent_tokenize(data)
  data_pre_processed_chunks,sample = [],""

  # Were able to split into sentences
  if len(sentences)>2:
    for index,sentence in enumerate(sentences):
      if len(sentence.split()) + len(sample.split()) <= max_size:
        sample += sentence
      else:
        if len(sample.split())>1:
          clean_data = text_cleaning(sample,min_lenght=min_lenght,extra_clean=extra_clean, remove_punctuation=remove_punctuation)
          if len(clean_data.split()) > max_size:
            pre_processed_chunks = split_by_chuncks(sample,min_lenght=min_lenght,max_size=max_size, extra_clean=extra_clean, remove_punctuation=remove_punctuation)
            data_pre_processed_chunks.extend(pre_processed_chunks)
          else:
            data_pre_processed_chunks.append(clean_data)
        sample = sentence if index < len(sentences)-1 else ""

    if len(sample) ==0:
      clean_data = text_cleaning(sentences[-1],min_lenght=min_lenght,extra_clean=extra_clean, remove_punctuation=remove_punctuation)  
    else:
      clean_data = text_cleaning(sample,min_lenght=min_lenght,extra_clean=extra_clean, remove_punctuation=remove_punctuation)
      
    #if len(clean_data.split()) >3:
    if len(clean_data.split()) > max_size:
      pre_processed_chunks = split_by_chuncks(clean_data,min_lenght=min_lenght,max_size=max_size, extra_clean=extra_clean, remove_punctuation=remove_punctuation)
      data_pre_processed_chunks.extend(pre_processed_chunks)
    else:
      data_pre_processed_chunks.append(clean_data)

  # Split by get max size chunks
  else:
    pre_processed_chunks = split_by_chuncks(data,min_lenght=min_lenght,max_size=max_size, extra_clean=extra_clean, remove_punctuation=remove_punctuation)
    data_pre_processed_chunks.extend(pre_processed_chunks)

  # return the pre_processed of whoole text and chunks
  return data_pre_processed,data_pre_processed_chunks

if __name__ == '__main__':
  exit(1)





