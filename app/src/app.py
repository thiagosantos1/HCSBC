# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import streamlit as st
import streamlit.components.v1 as components 
import time
import matplotlib.pyplot as plt
import pandas as pd
from pipeline import Pipeline 
import html
from IPython.core.display import display, HTML
import json
from PIL import Image
from tqdm import tqdm
import logging
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import copy
from download_models import check_if_exist
import re
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as plotpx
import umap

def image(src_as_string, **style):
	return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
	return a(_href=link, _target="_blank", style=styles(**style))(text)

def update_highlight(current,old):
	out = current
	matches_background_new = [(m.start(0), m.end(0)) for m in re.finditer("background-color:rgba\\(234, 131, 4,", out)]
	matches_background_old = [(m.start(0), m.end(0)) for m in re.finditer("background-color:rgba\\(234, 131, 4,", old)]
	for x,y in zip(matches_background_old,matches_background_new):
		try:
			old_importance = re.search("\\d+\\.\\d+",old[x[1]:x[1]+20])
			new_importance = re.search("\\d+\\.\\d+",current[y[1]:y[1]+20])

			if int(out[y[1]]) ==0 and float(old[x[1]]) != 0:
				out = out[0:y[1]] + str(old_importance.group(0))  + out[y[1]:]
				return False,out
			if float(out[y[1]]) !=0 and float(old[x[1]]) != 0:
				if float(old[x[1]]) > float(out[y[1]]):
					out = out[0:y[1]] + str(old_importance.group(0))[0] + out[y[1]:]
					return False,out
		except Exception as e:
			return True, out

	return True,out

def hidde_menu():

	footer_style = """<style>
					footer {
					visibility: hidden;
					}
					footer:after {
						content:"An end-to-end Breast Pathology Classification System to infer Breast Cancer Diagnosis and Severity"; 
						visibility: visible;
						display: block;
						position: center;
						#background-color: red;
						padding: 5px;
						top: 2px;
					}
					</style>
				"""

	st.markdown(footer_style, unsafe_allow_html=True)

def main(myargs):
	project_dir = os.path.dirname(os.path.abspath(__file__))


	def add_content(columns): 
		if 'hg_df' in st.session_state:
			columns[1].dataframe(st.session_state.hg_df)
		if 'all_l' in st.session_state:
			columns[2].dataframe(st.session_state.all_l)

		if "highlight_samples" in st.session_state:

			if "selected_indices" in st.session_state:
				if len(st.session_state.selected_indices) >0:
					out = ""
					l = st.session_state.selected_indices
					l.sort()
					for ind in l:
						out += st.session_state.highlight_samples[ind] +  "<br><br>"
					components.html(out,scrolling=True)
				else:
					components.html(st.session_state.highlight_samples[0])
			else:
				components.html(st.session_state.highlight_samples[0])


		# Add Plot - Only for File version
		if st.session_state['input_type'] == 'File' and "embeddings_all" in st.session_state and st.session_state.embeddings_plot in ["2D", "3D"]:
			indices = [x for x in range(st.session_state.data_df[st.session_state.input_column].values.shape[0])]
			if "selected_indices" in st.session_state:
				if len(st.session_state.selected_indices) >4:
					l = st.session_state.selected_indices
					l.sort()
					indices = l

			if st.session_state.data_df[st.session_state.input_column].values.shape[0] >=2:
				sub_embeddings = st.session_state.embeddings_all[indices]
				sentences = st.session_state.data_df[st.session_state.input_column].values[indices]
				sentences_parses = []
				break_size = 20
				for data in sentences:
					d = data.split()
					size_sentence = len(d)
					if len(d) >break_size:
						out = ""
						for lower_bound in range(0,size_sentence, break_size):
							upper_bound = lower_bound + break_size if lower_bound + break_size <= size_sentence else size_sentence
							out += " ".join(x for x in d[lower_bound:upper_bound]) + "<br>"
						sentences_parses.append(out)
					else:
						sentences_parses.append(data)

    

				prediction_label = st.session_state.hg_df["Prediction"].values[indices]
				prediction_worst_label = []
				for pred in prediction_label:
					preds = pred.split(" | ")
					if len(preds) ==1:
						prediction_worst_label.extend(preds)
					else:
						worst_index = min([st.session_state.predictor.bert_model.config['worst_rank'].index(x) for x in preds])
						prediction_worst_label.append(st.session_state.predictor.bert_model.config['worst_rank'][worst_index])
				

				if st.session_state.embeddings_type == "PCA":

					low_dim_embeddings = PCA(n_components=3).fit_transform(sub_embeddings)
				elif st.session_state.embeddings_type == "TSNE":
					low_dim_embeddings = TSNE(n_components=3,init="pca",perplexity=st.session_state.perplexity,learning_rate=st.session_state.learning_rate).fit_transform(sub_embeddings)

				else:
					n_neighbors = min(st.session_state.n_neighbors, len(sub_embeddings)-1 )
					low_dim_embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=st.session_state.min_dist,n_components=3).fit(sub_embeddings).embedding_

				df_embeddings = pd.DataFrame(low_dim_embeddings)
				df_embeddings = df_embeddings.rename(columns={0:'x',1:'y',2:'z'})
				df_embeddings = df_embeddings.assign(severity=prediction_worst_label)
				df_embeddings = df_embeddings.assign(text=sentences_parses)
				df_embeddings = df_embeddings.assign(data_index=indices)
				df_embeddings = df_embeddings.assign(all_predictions=prediction_label)
				

				if st.session_state.embeddings_plot == "2D":
					# 2D
					plot = plotpx.scatter(
								df_embeddings, x='x', y='y',
								color='severity', labels={'color': 'severity'},
								hover_data=['text','all_predictions','data_index'],title = 'BERT Embeddings Visualization - Please select rows (at least 5) to display specific examples'
					)

				else:
					# 3D
					plot = plotpx.scatter_3d(
								df_embeddings, x='x', y='y', z='z',
								color='severity', labels={'color': 'severity'},
								hover_data=['text','all_predictions','data_index'],title = 'BERT Embeddings Visualization - Please select rows (at least 5) to display specific examples'
					)

				st.plotly_chart(plot,use_container_width=True,)


			#worst_rank_ind = [classes.index(x) for x in worst_rank]

		if 'bert_lime_output' in st.session_state and st.session_state.bert_lime:
			if len(st.session_state.bert_lime_output) >0: # need to re-run prediction
				st.markdown("BERT Interpretability")
				components.html(st.session_state.bert_lime_output[0])

		if 'json_output' in st.session_state and st.session_state.json_out:

			st.markdown("Here are your analysis results in JSON format:")
			out = {}
			if "selected_indices" in st.session_state:
				
				if len(st.session_state.selected_indices) >0:
					l = st.session_state.selected_indices
					l.sort()
					for ind in l:
						out['sample_'+str(ind)] = st.session_state.json_output['sample_'+str(ind)]
					st.json(out)
				else:
					out['sample_'+str(0)] = st.session_state.json_output['sample_'+str(0)]
					st.json(out)
			else:
				# Display JSON
				out['sample_'+str(0)] = st.session_state.json_output['sample_'+str(0)]
				st.json(out)


	def delete_var_session(keys:list):
		for key in keys:
			if key in st.session_state:
				del st.session_state[key]

	im = Image.open(os.path.join(project_dir, "../imgs/icon.png"))


	# Wide mode
	st.set_page_config(page_title='HCSBC', layout = 'wide',page_icon=im,menu_items={
		'Get Help': 'https://github.com/thiagosantos1/BreastPathologyClassificationSystem',
		'Report a bug': "https://github.com/thiagosantos1/BreastPathologyClassificationSystem",
		'About': "An end-to-end breast pathology classification system https://github.com/thiagosantos1/BreastPathologyClassificationSystem"
	})
	st.sidebar.image(os.path.join(project_dir,"../imgs/doctor.png"),use_column_width=False)

	# Designing the interface
	st.markdown("<h1 style='text-align: center; color: black;'>HCSBC: Hierarchical Classification System for Breast Cancer Specimen Report</h1>", unsafe_allow_html=True)
	st.markdown("System Pipeline: Pathology Emory Pubmed BERT + 6 independent Machine Learning discriminators")
	# For newline
	st.write('\n')
	# Instructions
	st.markdown("*Hint: click on the top-right corner to enlarge it!*")
	# Set the columns

	cols = st.columns((1, 1, 1))
	#cols = st.columns(4)
	cols[0].subheader("Input Data")
	cols[1].subheader("Severity Predictions")
	cols[2].subheader("Diagnose Predictions")

	# Sidebar
	# File selection
	st.sidebar.title("Data Selection")
			
	st.session_state['input_type'] = st.sidebar.radio("Input Selection", ('File', 'Text'), key="data_format",index=1)
	if "prev_input_type" not in st.session_state:
		st.session_state['prev_input_type'] = st.session_state.input_type

	st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


	# Disabling warning
	st.set_option('deprecation.showfileUploaderEncoding', False)


	if st.session_state['input_type'] == 'File': 
		if st.session_state['prev_input_type'] == 'Text':
			delete_var_session(keys=["data_df","data_columns","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])
		st.session_state['prev_input_type'] = "File"

		# Choose your own file
		new_file = st.sidebar.file_uploader("Upload Document", type=['xlsx','csv'])
		if 'uploaded_file' in st.session_state and st.session_state.uploaded_file != None and new_file != None:
			if st.session_state.uploaded_file.name != new_file.name and st.session_state.uploaded_file.id != new_file.id:
				delete_var_session(keys=["data_df","data_columns","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])
		
		st.session_state['uploaded_file'] = new_file

		data_columns = ['Input']
		if 'data_columns' not in st.session_state:
			st.session_state['data_columns'] = data_columns

		if st.session_state.uploaded_file is not None:
			if 'data_df' not in st.session_state:
				if st.session_state.uploaded_file.name.endswith('.xlsx'):
					df = pd.read_excel(st.session_state.uploaded_file)
				else:
					df = pd.read_csv(st.session_state.uploaded_file)
	
				df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
				df = df.fillna("NA")
				data_columns = df.columns.values
				st.session_state['data_df'] = df
				st.session_state['data_columns'] = data_columns
	else:
		if st.session_state['prev_input_type'] == 'File':
			delete_var_session(keys=["data_df","input_column","user_input","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])
		st.session_state['prev_input_type'] = "Text"

		input_column = "Input"
		data = st.sidebar.text_area("Please enter a breast cancer pathology diagnose",value="BRWIRE Left wire directed segmntal mastectomy; short suture, superior; long suture, lateral   breast, left, wire-directed segmental mastectomy:   - infiltrating ductal carcinoma, nottingham grade i, 0.8 cm in maximum gross dimension.   - ductal carcinoma in situ, low nuclear grade, solid and cribriform types, associated with  microcalcifications and partially involving a small intraductal papilloma (0.2 cm). - invasive and in situ carcinoma extend to within 0.2 cm of the anterior specimen edge  separately submitted margin specimen below).  - no angiolymphatic invasion identifie  - adjacent breast with biopsy site changes, a small intraductal papilloma (0.2 cm), and fibrocystic changes.      - see synoptic report.")
		if "user_input" in st.session_state:
			if data != st.session_state.user_input:
				delete_var_session(keys=["data_df","input_column","user_input","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])

		st.session_state['user_input'] = data
		if len(st.session_state.user_input.split()) >0:
			st.session_state['data_df'] = pd.DataFrame([st.session_state['user_input']], columns =[input_column])
			st.session_state['input_column'] = input_column
			st.session_state['uploaded_file'] = True
		else:
			delete_var_session(keys=["data_df","input_column","user_input","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])


	if 'data_df' in st.session_state:
		cols[0].dataframe(st.session_state.data_df)


	if st.session_state['input_type'] == 'File':
		# Columns selection
		st.sidebar.write('\n')
		st.sidebar.title("Column For Prediction")
		input_column = st.sidebar.selectbox("Columns", st.session_state.data_columns)

		st.session_state['input_column'] = input_column


	st.sidebar.write('\n')
	st.sidebar.title("Severity Model")
	input_higher = st.sidebar.selectbox("Model", ["PathologyEmoryPubMedBERT", "PathologyEmoryBERT", "ClinicalBERT", "BlueBERT","BioBERT","BERT"])
	st.session_state['input_higher'] = input_higher

	if "prev_input_higher" not in st.session_state:
		st.session_state['prev_input_higher'] = st.session_state.input_higher
		st.session_state['input_higher_exist'] = check_if_exist(st.session_state.input_higher)
		st.session_state['load_new_higher_model'] = True
	elif st.session_state.prev_input_higher != st.session_state.input_higher:
		st.session_state['input_higher_exist'] = check_if_exist(st.session_state.input_higher)
		st.session_state['prev_input_higher'] = st.session_state.input_higher
		st.session_state['load_new_higher_model'] = True
		delete_var_session(keys=["data_df","input_column","user_input","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])
		

	st.sidebar.write('\n')
	st.sidebar.title("Diagnosis Model")
	input_all_labels = st.sidebar.selectbox("Model", ['single_vectorizer', 'branch_vectorizer'])
	st.session_state['input_all_labels'] = input_all_labels

	if "prev_input_all_labels" not in st.session_state:
		st.session_state['prev_input_all_labels'] = st.session_state.input_all_labels
		st.session_state['input_all_labels_exist'] = check_if_exist(st.session_state.input_all_labels)
		st.session_state['load_new_all_label_model'] = True
	elif st.session_state.prev_input_all_labels != st.session_state.input_all_labels:
		st.session_state['input_all_labels_exist'] = check_if_exist(st.session_state.input_all_labels)
		st.session_state['prev_input_all_labels'] = st.session_state.input_all_labels
		st.session_state['load_new_all_label_model'] = True
		delete_var_session(keys=["data_df","input_column","user_input","hg_df","all_l","highlight_samples","selected_indices","json_output","bert_lime_output","embeddings_all"])
		

	# For newline
	st.sidebar.write('\n')
	st.sidebar.title("Analysis Options")

	predictions, json_output, higher_order_pred,all_labels_pred,higher_order_prob,all_labels_prob = {},[],[],[],[],[]
	hg_df, all_l,highlight_samples, bert_lime_output, embeddings_all= [],[],[],[],[]


	if st.session_state['input_type'] == 'File':
		embeddings_plot = st.sidebar.radio('Display embeddings plot',
					  ['2D',
					   '3D',
					   'Dont Display'],index=1)

		st.session_state['embeddings_plot'] = embeddings_plot

	else:
		st.session_state['embeddings_plot'] = 'Dont Display'

	if st.session_state['input_type'] == 'File':
		embeddings_type = st.sidebar.radio('Dimensionality Reduction',
					  ['PCA',
					   'TSNE','UMAP'],index=0)

		st.session_state['embeddings_type'] = embeddings_type

		if st.session_state.embeddings_type == "TSNE":
			perplexity = st.sidebar.slider("Perplexity", min_value=5, max_value=100, step=5, value=30)
			st.session_state['perplexity'] = perplexity

			learning_rate = st.sidebar.slider("Learning Rate", min_value=10, max_value=1000, step=10, value=100)
			st.session_state['learning_rate'] = learning_rate

		if st.session_state.embeddings_type == "UMAP":
			n_neighbors = st.sidebar.slider("Neighbors", min_value=2, max_value=100, step=1, value=2)
			st.session_state['n_neighbors'] = n_neighbors

			min_dist = st.sidebar.slider("Minimal Distance", min_value=0.1, max_value=0.99, step=0.05, value=0.1)
			st.session_state['min_dist'] = min_dist

	json_out = st.sidebar.checkbox('Display Json',value = True,key='check3')
	st.session_state['json_out'] = json_out

	if st.session_state['input_type'] == 'Text':
		bert_lime = st.sidebar.checkbox('Display BERT Interpretability',value = False,key='check3')
		st.session_state['bert_lime'] = bert_lime
	else:
		st.session_state['bert_lime'] = False


	# For newline
	st.sidebar.write('\n')
	st.sidebar.title("Prediction")


	if st.sidebar.button("Run Prediction"):

		if st.session_state.uploaded_file is None:
			st.sidebar.write("Please upload a your data")

		else:
			st.session_state['input_all_labels_exist'] = check_if_exist(st.session_state.input_all_labels)
			if not st.session_state.input_all_labels_exist:
				st.sidebar.write("Please Download Model: " + str(st.session_state.input_all_labels))

			st.session_state['input_higher_exist'] = check_if_exist(st.session_state.input_higher)
			if not st.session_state.input_higher_exist:
				st.sidebar.write("Please Download Model: " + str(st.session_state.input_higher))

			if st.session_state.input_all_labels_exist and st.session_state.input_higher_exist:
				if "predictor" not in st.session_state or st.session_state.load_new_higher_model or st.session_state.load_new_all_label_model:
					with st.spinner('Loading model...'):
						print("\n\tLoading Model")
						st.session_state["predictor"] = Pipeline(bert_option=str(st.session_state.input_higher), branch_option=str(st.session_state.input_all_labels))
						st.session_state['load_new_higher_model'] = False
						st.session_state['load_new_all_label_model'] = False

				with st.spinner('Transforming Data...'):
					data = st.session_state.data_df[st.session_state.input_column].values

				with st.spinner('Analyzing...'):
					time.sleep(0.1)
					prog_bar = st.progress(0)
					logging.info("Running Predictions for data size of: " + str(len(data)))
					logging.info("\n\tRunning Predictions with: " + str(st.session_state.input_higher) + str(st.session_state.input_all_labels))
					for index in tqdm(range(len(data))):
						d = data[index]
						time.sleep(0.1)
						prog_bar.progress(int( (100/len(data)) * (index+1) ))
						# refactor json
						preds,embeddings_output = st.session_state.predictor.run(d)
						embeddings = embeddings_output.tolist()
						embeddings_all.append(embeddings[0])
						if st.session_state.bert_lime:
							logging.info("Running BERT LIME Interpretability Predictions")
							bert_lime_output.append(st.session_state.predictor.bert_interpretability(d))

						predictions["sample_" + str(index)] = {}
						for ind,pred in enumerate(preds):
							predictions["sample_" + str(index)]["prediction_" + str(ind)] = pred
							

					prog_bar.progress(100)
					time.sleep(0.1)
	
					for key,sample in predictions.items():
						higher,all_p, prob_higher, prob_all = [],[],[],[]
						for key,pred in sample.items():
							for higher_order, sub_arr in pred.items():
								higher.append(higher_order)
								prob_higher.append(round(sub_arr["probability"], 2))
								for label,v in sub_arr['diagnose'].items():
									all_p.append(label)  
									prob_all.append(round(v["probability"], 2))

						higher_order_pred.append(" | ".join(x for x in higher))
						all_labels_pred.append(" | ".join(x for x in all_p))

						higher_order_prob.append(" | ".join(str(x) for x in prob_higher))
						all_labels_prob.append(" | ".join(str(x) for x in prob_all))

					predictions_refact = copy.deepcopy(predictions)

					for index in tqdm(range(len(data))):
						highlights = ""
						key = "sample_" + str(index)
						for k,v in predictions[key].items():
							for k_s, v_s in v.items():
								predictions_refact["sample_" + str(index)]["data"] = v_s['data']
								predictions_refact["sample_" + str(index)]["transformer_data"] = v_s['transformer_data']
								predictions_refact["sample_" + str(index)]["discriminator_data"] = v_s['word_analysis']['discriminator_data']
								highlight = v_s['word_analysis']['highlighted_html_text']
								
								if len(highlights) >0:
									done = False
									merged = highlight
									while not done:
										done,merged = update_highlight(merged,highlights)

									highlights = merged
								else:
									highlights = highlight

								del predictions_refact[key][k][k_s]['data']
								del predictions_refact[key][k][k_s]['transformer_data']
								del predictions_refact[key][k][k_s]['word_analysis']['discriminator_data']

						highlight_samples.append(highlights)	

					json_output = predictions_refact
					
					hg_df = pd.DataFrame(list(zip(higher_order_pred, higher_order_prob)), columns =['Prediction', "Probability"])
					all_l = pd.DataFrame(list(zip(all_labels_pred,all_labels_prob)), columns =['Prediction',"Probability"])
					all_preds = pd.DataFrame(list(zip(higher_order_pred, all_labels_pred)), columns =['Severity Prediction',"Diagnose Prediction"])

					st.session_state['hg_df'] = hg_df
					st.session_state['all_l'] = all_l
					st.session_state['all_preds'] = all_preds
					st.session_state['json_output'] = json_output
					st.session_state['highlight_samples'] = highlight_samples
					st.session_state['highlight_samples_df'] = pd.DataFrame(highlight_samples, columns =["HTML Word Importance"])
					st.session_state['bert_lime_output'] = bert_lime_output
					st.session_state['embeddings_all'] = np.asarray(embeddings_all)

	if 'data_df' in st.session_state and 'json_output' in st.session_state: 
		st.markdown("<h1 style='text-align: center; color: purple;'>Model Analysis</h1>", unsafe_allow_html=True)              
		selected_indices = st.multiselect('Select Rows to Display Word Importance, Embeddings Visualization, and Json Analysis:', [x for x in range(len(st.session_state.data_df))])
		st.session_state['selected_indices'] = selected_indices   

	add_content(cols)
	

	if 'json_output' in st.session_state:
		st.sidebar.write('\n')
		st.sidebar.title("Save Results")

		st.sidebar.write('\n')
		st.sidebar.download_button(
			label="Download Output Json",
			data=str(st.session_state.json_output),
			file_name="output.json",
		 )
		st.sidebar.download_button(
			label="Download Predictions",
			data=st.session_state.all_preds.to_csv(),
			file_name="predictions.csv",
		 )
		st.sidebar.download_button(
			label="Download Data + Predictions",
			data = pd.concat([st.session_state.data_df, st.session_state.all_preds,st.session_state.highlight_samples_df], axis=1, join='inner').to_csv(),
			file_name="data_predictions.csv",
		 )
			
	st.sidebar.write('\n')
	st.sidebar.title("Contact Me")
	sub_colms = st.sidebar.columns([1, 1, 1])
	sub_colms[0].markdown('''<a href="https://github.com/thiagosantos1/BreastPathologyClassificationSystem">
						<img src="https://img.icons8.com/fluency/48/000000/github.png" /></a>''',unsafe_allow_html=True)
	sub_colms[1].markdown('''<a href="https://twitter.com/intent/follow?original_referer=https%3A%2F%2Fgithub.com%2Ftsantos_maia&screen_name=tsantos_maia">
						<img src="https://img.icons8.com/color/48/000000/twitter--v1.png" /></a>''',unsafe_allow_html=True)
	sub_colms[2].markdown('''<a href="https://www.linkedin.com/in/thiagosantos-cs/">
						<img src="https://img.icons8.com/color/48/000000/linkedin.png" /></a>''',unsafe_allow_html=True)
	

	hidde_menu()

	


if __name__ == '__main__':

	myargs = [
		"Made in ",
		image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
			  width=px(25), height=px(25)),
		" with ❤️ by ",
		link("https://www.linkedin.com/in/thiagosantos-cs/", "@thiagosantos-cs"),
		br(),
		link("https://www.linkedin.com/in/thiagosantos-cs/", image('https://img.icons8.com/color/48/000000/twitter--v1.png')),
		link("https://github.com/thiagosantos1/BreastPathologyClassificationSystem", image('https://img.icons8.com/fluency/48/000000/github.png')),
	]
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
		datefmt="%d/%m/%Y %H:%M:%S",
		level=logging.INFO)
	main(myargs)


