
""" Download pre-trained models from Google drive. """
import os
import argparse
import zipfile 
import logging
import requests
from tqdm import tqdm
import fire
import re 

logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
		datefmt="%d/%m/%Y %H:%M:%S",
		level=logging.INFO)


"", "", "", "","","" 


MODEL_TO_URL = {
	
	'PathologyEmoryPubMedBERT': 'https://drive.google.com/open?id=1l_el_mYXoTIQvGwKN2NZbp97E4svH4Fh', 
	'PathologyEmoryBERT': 'https://drive.google.com/open?id=11vzo6fJBw1RcdHVBAh6nnn8yua-4kj2IX', 
	'ClinicalBERT': 'https://drive.google.com/open?id=1UK9HqSspVneK8zGg7B93vIdTGKK9MI_v', 
	'BlueBERT': 'https://drive.google.com/open?id=1o-tcItErOiiwqZ-YRa3sMM3hGB4d3WkP', 
	'BioBERT': 'https://drive.google.com/open?id=1m7EkWkFBIBuGbfwg7j0R_WINNnYk3oS9',
	'BERT': 'https://drive.google.com/open?id=1SB_AQAAsHkF79iSAaB3kumYT1rwcOJru',

	'single_tfidf': 'https://drive.google.com/open?id=1-hxf7sKRtFGMOenlafdkeAr8_9pOz6Ym',
	'branch_tfidf': 'https://drive.google.com/open?id=1pDSnwLFn3YzPRac9rKFV_FN9kdzj2Lb0'
}

"""
	For large Files, Drive requires a Virus Check.
	This function is reponsivle to extract the link from the button confirmation
"""
def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('id="downloadForm" action="(.+?)"', line)
        if m:
            url = m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise RuntimeError(error)
    if not url:
        return None
    return url

def download_file_from_google_drive(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	
	response = session.get(URL, params={ 'id' : id }, stream=True)
	URL_new = get_url_from_gdrive_confirmation(response.text)

	if URL_new != None:
		URL = URL_new
		response = session.get(URL, params={ 'id' : id }, stream=True)

	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def check_if_exist(model:str = "single_tfidf"):

	if model =="single_vectorizer":
		model = "single_tfidf"
	if model =="branch_vectorizer":
		model = "branch_tfidf"

	project_dir = os.path.dirname(os.path.abspath(__file__))
	if model != None:
		if model in ['single_tfidf', 'branch_tfidf' ]:
			path='models/all_labels_hierarchy/'
			path_model = os.path.join(project_dir, "..", path,  model,'classifiers')
			path_vectorizer = os.path.join(project_dir, "..", path,  model,'vectorizers')
			if os.path.exists(path_model) and os.path.exists(path_vectorizer):
				if len(os.listdir(path_model)) >0 and len(os.listdir(path_vectorizer)) >0:
					return True
		else:
			path='models/higher_order_hierarchy/'
			path_folder = os.path.join(project_dir, "..", path,  model)
			if os.path.exists(path_folder):
				if len(os.listdir(path_folder + "/" )) >1:
					return True
	return False

def download_model(all_labels='single_tfidf', higher_order='PathologyEmoryPubMedBERT'):
	project_dir = os.path.dirname(os.path.abspath(__file__))

	path_all_labels='models/all_labels_hierarchy/'
	path_higher_order='models/higher_order_hierarchy/'
	
	def extract_model(path_file, name):

		os.makedirs(os.path.join(project_dir, "..", path_file), exist_ok=True)

		file_destination = os.path.join(project_dir, "..", path_file, name + '.zip')

		file_id = MODEL_TO_URL[name].split('id=')[-1]

		logging.info(f'Downloading {name} model (~1000MB tar.xz archive)')
		download_file_from_google_drive(file_id, file_destination)

		logging.info('Extracting model from archive (~1300MB folder) and saving to ' + str(file_destination))
		with zipfile.ZipFile(file_destination, 'r') as zip_ref:
			zip_ref.extractall(path=os.path.dirname(file_destination))

		logging.info('Removing archive')
		os.remove(file_destination)
		logging.info('Done.')


	if higher_order != None:
		if not check_if_exist(higher_order):
			extract_model(path_higher_order, higher_order)
		else:
			logging.info('Model ' + str(higher_order) + ' already exist')

	if all_labels!= None:
		if not check_if_exist(all_labels):
			extract_model(path_all_labels, all_labels)
		else:
			logging.info('Model ' + str(all_labels) + ' already exist')

	
	

def download(all_labels:str = "single_tfidf", higher_order:str = "PathologyEmoryPubMedBERT"):
	"""
		Input Options:
			all_labels : single_tfidf, branch_tfidf
			higher_order : clinicalBERT, blueBERT, patho_clinicalBERT, patho_blueBERT, charBERT
	"""
	all_labels_options  = [ "single_tfidf", "branch_tfidf"]
	higher_order_option = [ "PathologyEmoryPubMedBERT", "PathologyEmoryBERT", "ClinicalBERT", "BlueBERT","BioBERT","BERT" ]

	if all_labels not in all_labels_options or higher_order not in higher_order_option:
		print("\n\tPlease provide a valid model for downloading")
		print("\n\t\tall_labels: " + " ".join(x for x in all_labels_options))
		print("\n\t\thigher_order: " + " ".join(x for x in higher_order))
		exit()

	download_model(all_labels,higher_order)

if __name__ == "__main__":
	fire.Fire(download)



