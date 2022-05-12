"""
	Input config for pipeline 
"""

def config_file() -> dict:   
	config = {
				"BERT_config": { 
						"model_emb": 'bert',

						"model_option": {
											"PathologyEmoryPubMedBERT": {
															 "model_folder":"../models/higher_order_hierarchy/PathologyEmoryPubMedBERT/"
															},
											"PathologyEmoryBERT": {
															 "model_folder":"../models/higher_order_hierarchy/PathologyEmoryBERT/"
															},
											"ClinicalBERT": {
															 "model_folder":"../models/higher_order_hierarchy/ClinicalBERT/"
															},
											"BlueBERT": {
															 "model_folder":"../models/higher_order_hierarchy/BlueBERT/"
															},
											"BioBERT": {
															 "model_folder":"../models/higher_order_hierarchy/BioBERT/"
															},
											"BERT": {
															 "model_folder":"../models/higher_order_hierarchy/BERT/"
															},

										},
						"max_seq_length": "64",
						"threshold_prediction":0.5,
						"classes": ['Invasive breast cancer-IBC','Non-breast cancer-NBC','In situ breast cancer-ISC',
									 'Borderline lesion-BLL','High risk lesion-HRL','Benign-B','Negative'],
						"worst_rank" : ['Invasive breast cancer-IBC', 'In situ breast cancer-ISC', 'High risk lesion-HRL',
										'Borderline lesion-BLL','Benign-B','Non-breast cancer-NBC','Negative']
				},


				"ibc_config": {

						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "ibc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "ibc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"ibc_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},

						"classes": ['apocrine carcinoma','grade i','grade ii','grade iii','invasive ductal carcinoma','invasive lobular carcinoma','medullary carcinoma','metaplastic carcinoma','mucinous carcinoma','tubular carcinoma','lymph node - metastatic']
				
				},

				"isc_config": {
						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "isc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "isc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"isc_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},


						"classes": ['ductal carcinoma in situ','high','intermediate','intracystic papillary carcinoma','intraductal papillary carcinoma','low','pagets','fna - malignant']
				
				},

				"hrl_config": { 
						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "hrl_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "hrl_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"hrl_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},


						"classes": ['atypical ductal hyperplasia','atypical lobular hyperplasia','atypical papilloma','columnar cell change with atypia','flat epithelial atypia','hyperplasia with atypia','intraductal papilloma','lobular carcinoma in situ','microscopic papilloma','radial scar']
				},

				"bll_config": { 
						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "bll_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "bll_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"bll_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},


						"classes": ['atypical phyllodes', 'granular cell tumor', 'mucocele']
				},

				"benign_config": { 
						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "benign_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "benign_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"benign_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},


						"classes": ['apocrine metaplasia','biopsy site changes','columnar cell change without atypia','cyst','excisional or post-surgical change','fat necrosis','fibroadenoma','fibroadenomatoid','fibrocystic disease','fibromatoses','fibrosis','hamartoma','hemangioma','lactational change','lymph node - benign','myofibroblastoma','myxoma','phyllodes','pseudoangiomatous stromal hyperplasia','sclerosing adenosis','usual ductal hyperplasia','fna - benign','seroma']
				},

				"nbc_config": { 
						"model_option": {
											"single_tfidf": {
																"path_model":"../models/all_labels_hierarchy/single_tfidf/classifiers",
																"model": "nbc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"vectorizer":"vectorizer_all_branches.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/single_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																},

											"branch_tfidf": {
																"path_model":"../models/all_labels_hierarchy/branch_tfidf/classifiers",
																"model": "nbc_xgboost_classifier.pkl",
																"path_vectorizer":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"vectorizer":"nbc_vectorizer.pkl",
																"path_bigrmas":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"bigrams":"best_bigrams.csv",
																"path_phrase_bigrams":"../models/all_labels_hierarchy/branch_tfidf/vectorizers",
																"phrase_bigrams" : "phrase_bigrams.pkl"
																}
										},


						"classes": ['lymphoma', 'malignant(sarcomas)', 'non-breast metastasis']
				},
	}

	return config

if __name__ == '__main__':
	pass

