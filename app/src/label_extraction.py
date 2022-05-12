import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import fire
import json
from tqdm import tqdm
import logging
from pipeline import Pipeline 
import copy
from download_models import check_if_exist

"""
    Install dependecies by running: pip3 install -r requirements.txt

    Running command example:
    python3 label_extraction.py --path_to_file data.xlsx --column_name report --save_predictions predictions.xlsx --save_json output.json
"""

def data_extraction(path_to_file:str, column_name:str, higher_model:str="clinicalBERT", all_label_model="single_tfidf", save_predictions:str=None, output_model_data=None,save_input=None, save_json:str=None):

    """
        This program takes an excell/csv sheet and extract the higher order and cancer characteristics from pathology reports 

        Input Options:
            1) path_to_file - Path to an excel/csv with pathology diagnosis: String (Required)
            2) column_name - Which column has the pathology diagnosis: String (Required)
            3) higher_model - Which version of higher order model to use: String (Required)
            4) all_label_model - Which version of all labels model to use: String (Required)
            5) save_predictions - Path to save output: String (Optional)
            6) output_model_data - Option to output model data to csv True/False (Optional)
            7) save_input - Option to output the input fields True/False (Optional)
            8) save_json - Path to save json analyis: String (Optional)
            

    """

    data_orig = read_data(path_to_file)
    data_orig = data_orig.fillna("NA")
    data = data_orig.loc[:, ~data_orig.columns.str.contains('^Unnamed')][column_name].values
    
    predictions, json_output, higher_order_pred,all_labels_pred,higher_order_prob,all_labels_prob = {},[],[],[],[],[]

    if not check_if_exist(higher_model):
        print("\n\t ##### Please Download Model: " + str(higher_model) + "#####")
        exit()
    if not check_if_exist(all_label_model):
        print("\n\t ##### Please Download Model: " + str(all_label_model) + "#####")
        exit()

    model = Pipeline(bert_option=higher_model, branch_option=all_label_model)

    logging.info("\nRunning Predictions for data size of: " + str(len(data)))
    for index in tqdm(range(len(data))):
        d = data[index]
        # refactor json
        preds,all_layer_hidden_states = model.run(d)
        predictions["sample_" + str(index)] = {}
        for ind,pred in enumerate(preds):
            predictions["sample_" + str(index)]["prediction_" + str(ind)] = pred

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
    transformer_data, discriminator_data= [0 for x in range(len(data))], [0 for x in range(len(data))]

    for index in tqdm(range(len(data))):
        key = "sample_" + str(index)
        for k,v in predictions[key].items():
            for k_s, v_s in v.items():
                predictions_refact["sample_" + str(index)]["data"] = v_s['data']
                predictions_refact["sample_" + str(index)]["transformer_data"] = v_s['transformer_data']
                predictions_refact["sample_" + str(index)]["discriminator_data"] = v_s['word_analysis']['discriminator_data']
                transformer_data[index] = v_s['transformer_data']
                discriminator_data[index] = v_s['word_analysis']['discriminator_data']

                del predictions_refact[key][k][k_s]['data']
                del predictions_refact[key][k][k_s]['transformer_data']
                del predictions_refact[key][k][k_s]['word_analysis']['discriminator_data']

    json_output = predictions_refact
    

    if save_predictions!= None:
        logging.info("Saving Predictions")
        if output_model_data != None:
            all_preds = pd.DataFrame(list(zip(higher_order_pred, all_labels_pred,transformer_data,discriminator_data,data)), columns =['Severity Prediction',"Diagnose Prediction", 'Severity Model Data','Diagnose Model Data',column_name])
        else:
            all_preds = pd.DataFrame(list(zip(higher_order_pred, all_labels_pred)), columns =['Severity Prediction',"Diagnose Prediction"])

        if save_input != None:
            all_preds = pd.concat([data_orig, all_preds], axis=1)
        try:
            all_preds.to_excel(save_predictions)
        except ValueError:
            try:
                all_preds.to_csv(save_predictions)
            except ValueError:
                logging.exception("Error while saving predictions " + str(e))
                exit()
        logging.info("Done")

    if save_json!= None:
        logging.info("Saving Json")
        try:
            with open(save_json, 'w') as f:
                for k, v in json_output.items():
                    f.write('{'+str(k) + ':'+ str(v) + '\n')
        
        except ValueError:
            logging.exception("Error while saving json analysis " + str(e))
            exit()
        logging.info("Done")


def read_data(path_to_file):

    try:
        df = pd.read_excel(path_to_file)
        return df
    except ValueError:
        try:
            df = pd.read_csv(path_to_file)
            return df
        except ValueError:
            logging.exception("### Error occurred while splitting document. Info: " + str(e))
            exit()



def run():
    fire.Fire(data_extraction)

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",datefmt="%d/%m/%Y %H:%M:%S",level=logging.INFO)
    run()



    
