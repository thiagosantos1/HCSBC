import os
import sys 

import text_cleaning_transforerms as tc  
import text_cleaning

import logging
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import json
import joblib
from gensim.models import phrases

import math

import xgboost
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import html

from config import config_file


from lime import lime_text
from lime.lime_text import LimeTextExplainer


from transformers import AutoModelForSequenceClassification,AutoTokenizer

from nltk.tokenize import word_tokenize


"""
	Cancer Severity Class. 

	export env_name="path"
"""
class BERT_Model(object):
	def __init__(self, config,bert_option:str="clinicalBERT"):

		try:
			self.config = config
			self.project_dir = os.path.dirname(os.path.abspath(__file__))
			self.bert_option = bert_option
			# check if a path was alreadey added to os env table

			if "model_folder" in os.environ:
				self.config['model_folder'] = os.environ['model_folder']
			else:
				self.config['model_folder'] = os.path.join(self.project_dir, self.config['model_option'][self.bert_option]['model_folder'])

			self.initialize()
		except Exception as e:
			logging.exception("Error occurred while Initializing BERT Model, please double check you have a config file " +" Info: " + str(e))
			exit()

	def initialize(self):
		# Set up logging
		logging.basicConfig(
			format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
			datefmt="%d/%m/%Y %H:%M:%S",
			level=logging.INFO)

		# Check for GPUs
		if torch.cuda.is_available():
			self.config["use_cuda"] = True
			self.config["cuda_device"] = torch.cuda.current_device()
			logging.info("Using GPU (`%s`)", torch.cuda.get_device_name())
		else:
			self.config["use_cuda"] = False
			self.config["cuda_device"] = "cpu"
			logging.info("Using CPU")


		self.model  = AutoModelForSequenceClassification.from_pretrained(self.config["model_folder"], num_labels=len(self.config['classes']),output_hidden_states=True).to(self.config["cuda_device"])
		self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_folder"])


	def clean_data(self,text:str):
		return tc.pre_process(text,max_size=int(self.config["max_seq_length"]),remove_punctuation=True )

	def sigmoid(self,x):
		return 1 / (1 + math.exp(-x))

	"""
		Convert output of multi-class to probabilities between 0-1
	"""
	def raw_to_probs(self,vector):
		return [self.sigmoid(x) for x in vector]


	"""
		Given a threshold, convert a vector of probabiities into predictions (0 or 1)
	"""
	def _threshold(self, vector:list, threshold:float=0.5) -> list:
		logit_vector = [1 if x >=threshold else 0 for x in vector]
		return logit_vector

	"""
		Pre-Process the data according to the same strategy used during training
	"""
	def pre_process(self,texts:list)-> list:
		transformer_clean_data,transformer_clean_data_chunks = [],[]
		for index,t in enumerate(texts):
			clean_data, clean_data_chunks = self.clean_data(t)
			transformer_clean_data.append(clean_data)
			transformer_clean_data_chunks.append(clean_data_chunks)

		return transformer_clean_data,transformer_clean_data_chunks


	"""
		Giving a list of texts, return the sentence embedding (CLS token from last BERT layer)
	"""
	def get_embeddings(self,texts:list)-> list:

		transformer_clean_data,_ = self.pre_process(texts)
		
		inputs = self.tokenizer(transformer_clean_data, return_tensors="pt", padding=True).to(self.config["cuda_device"])
		outputs = self.model(**inputs,output_hidden_states=True)
		last_hidden_states = outputs[1][-1].detach().cpu().numpy()
		embeddings_output = np.asarray(last_hidden_states[:, 0])

		return embeddings_output

	"""
		Giving a list of texts, run BERT prediction for each sample
		If use_chunks is set to True (default), it chunks de data into chunks of max_size (set on config.py)
		The final prediction for that sample is the concatenation of predictions from every chunck

		Returns:
			* Predictions
			* Probabiities
			* Sentence Embedding (CLS token from last BERT layer)
			* Pre-Processed data used for Prediction
	"""
	def predict(self,texts:list, use_chunks=True)-> list:
		
		transformer_clean_data,transformer_clean_data_chunks = self.pre_process(texts)
		ids_chunks = []
		# Flat all chunks (2d list) into 1d List (each chunck is feed separetly to prediction)
		if use_chunks:

			flatten_chunks = [j for sub in transformer_clean_data_chunks for j in sub]
			ids = [[x]*len(transformer_clean_data_chunks[x]) for x in range(len(transformer_clean_data_chunks))]
			ids_chunks = [j for sub in ids for j in sub]
			data = flatten_chunks.copy()

		else:
			data = transformer_clean_data.copy()

		inputs = self.tokenizer(data, return_tensors="pt", padding=True).to(self.config["cuda_device"])
		outputs = self.model(**inputs,output_hidden_states=True)

		# Post-Process output if using chunks --> Merge chunck Predictions into 1
		if use_chunks:
			raw_probs_chunks = outputs[0].detach().cpu().numpy()
			probs_chunks = [self.raw_to_probs(x) for x in raw_probs_chunks]
			probs = np.asarray([[0 for x in range(len(probs_chunks[0]))] for x in range(len(texts))],dtype=float)
			for index, prob in enumerate(probs_chunks):
				id_ = ids_chunks[index]

				# if no predictions for such index yet, add (this is the base - avoid zero preds)
				if np.sum(probs[id_])<=0:
					probs[id_] = prob
				else: # update to merge predictions
					pred = np.asarray(self._threshold(vector=prob,threshold=self.config["threshold_prediction"]))
					pos_pred_index  = np.where(pred>0)[0]
					if len(pos_pred_index)>0:
						for pos in pos_pred_index:
							probs[id_][pos] = prob[pos]

		else:
			raw_probs = outputs[0].detach().cpu().numpy()
			probs = [self.raw_to_probs(x) for x in raw_probs]
		
		predictions = [self._threshold(vector=pred,threshold=self.config["threshold_prediction"]) for pred in probs]



		last_hidden_states = outputs[1][-1].detach().cpu().numpy()
		embeddings_output = np.asarray(last_hidden_states[:, 0])

		return predictions, probs, embeddings_output, transformer_clean_data

		

	""" 
		Giving a list of text, it executes the branch prediction
		This function call BERT Predict, pre-process predictions, and return the post-process branch prediction
		Returns:
			* Branch Prediction
			* Sentence Embedding (CLS token from last BERT layer)
	"""
	def branch_prediction(self,texts:list)-> list:
		out_pred = []
		
		predictions, probs, embeddings_output, transformer_clean_data = self.predict(texts,use_chunks=True)

		try:
			for index, preds in enumerate(probs):
				preds = np.asarray(preds)
				pos = np.where(preds > 0.5)[0]
				pred = []
				if len(pos) >0:
					for ind in pos:
						pred.append({self.config['classes'][ind]: {"probability":preds[ind], "data":texts[index], "transformer_data": transformer_clean_data[index] }})
				else:
					pred.append({"No Prediction": {"probability":0, "data":texts[index], "transformer_data": transformer_clean_data[index]}})

				out_pred.append(pred)
		except Exception as e:
			logging.exception("Error occurred on BERT model prediction" +" Info: " + str(e))
			exit()

		return out_pred,embeddings_output


"""
	Cancer Diagnose Prediction Class.
	This class is used to load each individual branch classifier
"""
class Branch_Classifier(object):
	def __init__(self, config, branch_option:str="single_tfidf"):
		self.config = config
		self.branch_option = branch_option
		self.project_dir = os.path.dirname(os.path.abspath(__file__))

		try:
			if "path_model" in os.environ:
				self.config['path_model'] = os.environ['path_model']
			else:
				self.config['path_model'] = os.path.join(self.project_dir, self.config['model_option'][self.branch_option]['path_model'])

			if "path_vectorizer" in os.environ:
				self.config['path_vectorizer'] = os.environ['path_vectorizer']
			else:
				self.config['path_vectorizer'] = os.path.join(self.project_dir, self.config['model_option'][self.branch_option]['path_vectorizer'])

			if "path_bigrmas" in os.environ:
				self.config['path_bigrmas'] = os.environ['path_bigrmas']
			else:
				self.config['path_bigrmas'] = os.path.join(self.project_dir, self.config['model_option'][self.branch_option]['path_bigrmas'])

			if "path_phrase_bigrams" in os.environ:
				self.config['path_phrase_bigrams'] = os.environ['path_phrase_bigrams']
			else:
				self.config['path_phrase_bigrams'] = os.path.join(self.project_dir, self.config['model_option'][self.branch_option]['path_phrase_bigrams'])

		except Exception as e:
			logging.exception("Error occurred while reading config file. Please read config instructions" +" Info: " + str(e))
			exit()

		self.initialize()
		

	def initialize(self):

		try:
			self.model = joblib.load(os.path.join(self.config['path_model'],self.config['model_option'][self.branch_option]['model']))
			self.vectorizer = joblib.load(os.path.join(self.config['path_vectorizer'],self.config['model_option'][self.branch_option]['vectorizer']))
			self.good_bigrams = pd.read_csv(os.path.join(self.config["path_bigrmas"],self.config['model_option'][self.branch_option]['bigrams']))['bigram'].to_list()
			self.phrase_bigrams = phrases.Phrases.load(os.path.join(self.config["path_phrase_bigrams"],self.config['model_option'][self.branch_option]['phrase_bigrams']))

		except Exception as e:
			logging.exception("Error occurred while initializing models and vectorizer" +" Info: " + str(e))
			exit()

	"""
		Only add specific Bi-grams (Pre-calculated during Training)
	"""
	def clean_bigram(self,data:list)-> list:

		data_clean = []

		for word in data:
			if re.search("_",word) == None:
				data_clean.append(word)
			else: # gotta add the word without _ as well
				if word in self.good_bigrams:
					data_clean.append(word)
				else:
					data_clean.append(word.split("_")[0])
					data_clean.append(word.split("_")[1])

		return np.asarray(data_clean)

	"""
		Giving a list of text, pre-process and format the data
	"""
	def format_data(self,data:list)-> list:
		try:
			X = text_cleaning.text_cleaning(data, steam=False, lemma=True,single_input=True)[0]

			### Add Bigrams and keep only the good ones(pre-selected)
			X_bigrmas  = self.phrase_bigrams[X] 
			data_clean = self.clean_bigram(X_bigrmas)
			X_bigrams_clean = ' '.join(map(str, data_clean)) 
			pre_processed = self.vectorizer.transform([X_bigrams_clean]).toarray(),X_bigrams_clean

		except Exception as e:
			logging.exception("Error occurred while formatting and cleaning data" +" Info: " + str(e))
			exit()

		return pre_processed


	def html_escape(self,text):
		return html.escape(text)

	def predict(self, texts:list)-> list:
		"""
			Steps:
				1) Run the predictions from higher-order
				2) Based on the prediction, activate which brach(es) to send for final prediction (cancer characteristics)
				3) For final prediction, create a word importance HTML for each input
		"""
		out_pred = {'predictions': {}, 'word_analysis':{},}

		color = "234, 131, 4" # orange
		try:
			for t in texts:
				text_tfidf,clean_data = self.format_data(t)
				probs = self.model.predict_proba(text_tfidf).toarray()
				predictions = self.model.predict(text_tfidf).toarray()
				for index,preds in enumerate(predictions):
					pos = np.where(preds > 0.5)[0]
					pred = []
					if len(pos) >0:
						for ind in pos:
							highlighted_html_text = []
							weigts = self.model.classifiers_[ind].feature_importances_
							word_weights = {}
							words = clean_data.split()
							min_new = 0
							max_new = 100
							min_old = np.min(weigts)
							max_old = np.max(weigts)
							for w in words:
								found = False
								for word, key in self.vectorizer.vocabulary_.items():
									if w == word:
										found = True
										# rescale weights
										weight = ( (max_new - min_new) / (max_old - min_old) * (weigts[key] - max_old) + max_new)
										if weight <0.5:
											weight = 0

										 
										if "_" in w: # add for each word
											w1,w2 = w.split("_")
											word_weights[w1] =  weight
											word_weights[w2] =  weight
											if w2 =="one":
												word_weights["1"] =  weight
												word_weights["i"] =  weight
											if w2 =="two":
												word_weights["2"] =  weight
												word_weights["ii"] =  weight
											if w2 =="three":
												word_weights["3"] =  weight
												word_weights["iii"] =  weight
										else:
											word_weights[w] =  weight
								if found == False: # some words aren't presented in the model
									word_weights[w] =  0

							words = word_tokenize(t.lower().replace("-", " - ").replace("_", " ").replace(".", " . ").replace(",", " , ").replace("(", " ( ").replace(")", " ) "))
							for i,w in enumerate(words):
								if w not in word_weights or w=='-' or w==',' or w=='.' or w=="(" or w==")":
									word_weights[w] =  0
									highlighted_html_text.append(w)
								else:
									weight = 0 if word_weights[w] <1 else word_weights[w]
									highlighted_html_text.append('<span font-size:40px; ; style="background-color:rgba(' + color + ',' + str(weight) + ');">' + self.html_escape(w) + '</span>')

										

							highlighted_html_text = ' '.join(highlighted_html_text)
							#pred.append({ "predictions": {self.config['classes'][ind]: {"probability":probs[index][ind]}},"word_analysis": {"discriminator_data": clean_data,"word_importance": word_weights, "highlighted_html_text":highlighted_html_text}})
							out_pred["predictions"][self.config['classes'][ind]] = {"probability":probs[index][ind]}
							out_pred["word_analysis"] = {"discriminator_data": clean_data,"word_importance": word_weights, "highlighted_html_text":highlighted_html_text}
					
					else:
						out_pred["predictions"] = {"Unkown": {"probability":0.5}}
						out_pred["word_analysis"] = {"discriminator_data": clean_data,"word_importance": {x:0 for x in t.split()}, "highlighted_html_text": " ".join(x for x in t.split())} 

						#pred.append({"predictions": {"Unkown": {"probability":0.5}}, "word_analysis": {"discriminator_data": clean_data,"word_importance": {x:0 for x in t.split()}, "highlighted_html_text": " ".join(x for x in t.split())}}) 

					#out_pred.append(pred)

		except Exception as e:
			logging.exception("Error occurred on model prediction" +" Info: " + str(e))
			exit()

		return out_pred


class LIME_Interpretability(object):

	"""
		Class for LIME Analysis

	"""

	def __init__(self, label_colors = { "positive": "234, 131, 4",  # orange
										 "negative":'65, 137, 225',  # blue
										}):

		self.color_classes = label_colors

	# function to normalize, if applicable
	def __normalize_MinMax(self,arr, t_min=0, t_max=1):
		norm_arr = []
		diff = t_max - t_min
		diff_arr = max(arr) - min(arr)
		for i in arr:
			temp = (((i - min(arr)) * diff) / diff_arr) + t_min
			norm_arr.append(temp)
		return norm_arr


	def __html_escape(self,text):
		return html.escape(text)


	def __add_bigrams(self,txt):
		fixed_bigrams = [ [' gradeone ', 'grade 1', 'grade i', 'grade I', 'grade one',],
						[' gradetwo ', 'grade 2', 'grade ii', 'grade II', 'grade two', ],
						[' gradethree ', 'grade 3' , 'grade iii', 'grade III', 'grade three']]
		for b in fixed_bigrams:
			sub = ""
			not_first = False
			for x in b[1:]:
				if not_first:
					sub += "|"
					not_first = True

				sub += str(x) + "|" + str(x) + " " + "|" +  " " + str(x) + "|" + " " + str(x)   
			txt = re.sub(sub, b[0], txt)
			# Removing multiple spaces
			txt = re.sub(r'\s+', ' ', txt)
			txt = re.sub(' +', ' ', txt)
		return txt

	def __highlight_full_data(self,lime_weights, data, exp_labels,class_names):
		words_p = [x[0] for x in lime_weights if x[1]>0]
		weights_p = np.asarray([x[1] for x in lime_weights if x[1] >0])
		if len(weights_p) >1:
			weights_p = self.__normalize_MinMax(weights_p, t_min=min(weights_p), t_max=1)
		else:
			weights_p = [1]
		words_n = [x[0] for x in lime_weights if x[1]<0]
		weights_n = np.asarray([x[1] for x in lime_weights if x[1] <0])
	#     weights_n = self.__normalize_MinMax(weights_n, t_min=max(weights_p), t_max=-0.8)
		
		labels = exp_labels
		pred = class_names[labels[0]]
		corr_pred = class_names[labels[1]] # negative lime weights
		
		# positive values
		df_coeff = pd.DataFrame(
			{'word': words_p,
			 'num_code': weights_p
			})
		word_to_coeff_mapping_p = {}
		for row in df_coeff.iterrows():
			row = row[1]
			word_to_coeff_mapping_p[row[0]] = row[1]
		
		# negative values
		df_coeff = pd.DataFrame(
			{'word': words_n,
			 'num_code': weights_n
			})
		
		word_to_coeff_mapping_n = {}
		for row in df_coeff.iterrows():
			row = row[1]
			word_to_coeff_mapping_n[row[0]] = row[1]
		
		max_alpha = 1
		highlighted_text = []
		data = re.sub("-"," ", data)
		data = re.sub("/","", data)
		for word in word_tokenize(self.__add_bigrams(data)):
			if word.lower() in word_to_coeff_mapping_p or word.lower() in word_to_coeff_mapping_n:
				if word.lower() in word_to_coeff_mapping_p:
					weight = word_to_coeff_mapping_p[word.lower()]
				else:
					weight = word_to_coeff_mapping_n[word.lower()]
					
				if weight >0:
					color = self.color_classes["positive"]
				else:
					color = self.color_classes["negative"]
					weight *= -1
					weight *=10
				
				highlighted_text.append('<span font-size:40px; ; style="background-color:rgba(' + color + ',' + str(weight) + ');">' + self.__html_escape(word) + '</span>')

			else:
				highlighted_text.append(word)
				
		highlighted_text = ' '.join(highlighted_text)

		return highlighted_text


	def lime_analysis(self,model,data_original, data_clean, num_features=30, num_samples=50, top_labels=2,
					class_names=['ibc', 'nbc', 'isc', 'bll', 'hrl', 'benign', 'negative']):

		# LIME Predictor Function
		def predict(texts):
			results = []
			for text in texts:
				predictions, probs, embeddings_output, transformer_clean_data = model.predict([text],use_chunks=False)
				results.append(probs[0])

			return np.array(results)

		explainer = LimeTextExplainer(class_names=class_names)
		exp = explainer.explain_instance(data_clean, predict, num_features=num_features,
										 num_samples=num_samples, top_labels=top_labels)
		l = exp.available_labels()
		run_info = exp.as_list(l[0])
		return self.__highlight_full_data(run_info, data_original, l,class_names)


"""
	The pipeline is responsible to consolidate the output of all models (higher order and all labels hierarchy)
	It takes a string as input, and returns a jason with higher-order(Severity) and all labels(Diagnose) predictions and their probability score
"""
class Pipeline(object):

	def __init__(self, bert_option:str="clinicalBERT", branch_option:str="single_tfidf"):
		logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",datefmt="%d/%m/%Y %H:%M:%S",level=logging.INFO)
		
		if branch_option =="single_vectorizer":
			self.branch_option = "single_tfidf"
		elif branch_option =="branch_vectorizer":
			self.branch_option = "branch_tfidf"
		else:
			self.branch_option=branch_option

		self.bert_option=bert_option
		
		try:
			self.config = config_file()
			self.BERT_config = self.config['BERT_config']
			self.ibc_config = self.config['ibc_config']
			self.isc_config = self.config['isc_config']
			self.hrl_config = self.config['hrl_config']
			self.bll_config = self.config['bll_config']
			self.benign_config = self.config['benign_config']
			self.nbc_config = self.config['nbc_config']

		except Exception as e:
			logging.exception("Error occurred while initializing models and vectorizer" +" Info: " + str(e))
			exit()

		self.lime_interpretability = LIME_Interpretability()

		self.initialize()


	def initialize(self):
		try:
			self.bert_model = BERT_Model(self.BERT_config, self.bert_option)
			try:
				self.ibc_branch = Branch_Classifier(self.ibc_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing IBC branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()
			
			try:
				self.isc_branch = Branch_Classifier(self.isc_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing isc branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()

			try:
				self.hrl_branch = Branch_Classifier(self.hrl_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing hrl branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()

			try:
				self.bll_branch = Branch_Classifier(self.bll_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing bll branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()

			try:
				self.benign_branch = Branch_Classifier(self.benign_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing benign branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()

			try:
				self.nbc_branch = Branch_Classifier(self.nbc_config,branch_option=self.branch_option)
			except Exception as e:
				logging.exception("Error occurred while Initializing nbc branch Model, please double check you have a config file " +" Info: " + str(e))
				exit()

			self.all_label_models = [self.ibc_branch,self.nbc_branch,self.isc_branch,self.bll_branch,self.hrl_branch,self.benign_branch]
	

		except Exception as e:
			logging.exception("Error occurred while Initializing Pipeline, please double check you have a config file " +" Info: " + str(e))
			exit()

	
	"""
		Run the entire pipeline
		Steps:
			1) First, we run the Severity Prediction (BERT)
			2) Given each prediction for each sample, we then:
				2.1) Run the corresponding Diagnose Branch Prediction
				2.2) Merge every branch prediction
			3) Merge Every Severity and Branch Prediction
		
		Inputs:
			* Text

		Output:
			* Predictions (Predictions + Probabilites)
			* Sentence Embedding
	"""
	def run(self,input_text:str):

		"""
			First, get the severity prediction (higher order branch)
		"""
		predictions,embeddings_output =  self.bert_model.branch_prediction([input_text])
		predictions = predictions[0]
		for pred in predictions:
			for higher_order, sub_arr in pred.items():
				# Check which branch it belongs to
				if higher_order in ["Negative","No Prediction"]:
					pred[higher_order]['diagnose'] = {higher_order: {"probability":sub_arr['probability']}}
					pred[higher_order]["word_analysis"] = {"discriminator_data": "Not Used", "word_importance": {x:0 for x in input_text.split()}, "highlighted_html_text": " ".join(x for x in input_text.split())}

				# For each Severity, run the corresponding Branch Prediction
				else: 
					model = self.all_label_models[self.bert_model.config['classes'].index(higher_order)]
					out_pred = model.predict([input_text])
					
					pred[higher_order]['diagnose'] = out_pred['predictions']
					pred[higher_order]['word_analysis'] = out_pred['word_analysis']

		return predictions,embeddings_output

	def bert_interpretability(self, input_text:str):
		clean_data,_ = self.bert_model.clean_data(input_text)
		return self.lime_interpretability.lime_analysis(self.bert_model,input_text, clean_data, class_names=self.bert_model.config['classes'])


if __name__ == '__main__':
	exit()




