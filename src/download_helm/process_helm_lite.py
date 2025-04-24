import json
import os
import pickle
import numpy as np
from pathlib import Path

HELM_LITE_SCENARIOS = {
    'commonsense:dataset=openbookqa,method=multiple_choice_joint,': ['commonsense:dataset=openbookqa,method=multiple_choice_joint,'],
    'gsm:': ['gsm:'],
    'med_qa:': ['med_qa:'],
    'legalbench': [
        'legalbench:subset=abercrombie,',
        'legalbench:subset=corporate_lobbying,',
        'legalbench:subset=function_of_decision_section,',
        'legalbench:subset=proa,',
        'legalbench:subset=international_citizenship_questions,'
    ],
    'math': [
        'math:subject=algebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=counting_and_probability,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=geometry,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=intermediate_algebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=number_theory,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=prealgebra,level=1,use_official_examples=False,use_chain_of_thought=True,',
        'math:subject=precalculus,level=1,use_official_examples=False,use_chain_of_thought=True,'
    ],
    'mmlu': [
        'mmlu:subject=abstract_algebra,method=multiple_choice_joint,',
        'mmlu:subject=college_chemistry,method=multiple_choice_joint,',
        'mmlu:subject=computer_security,method=multiple_choice_joint,',
        'mmlu:subject=econometrics,method=multiple_choice_joint,',
        'mmlu:subject=us_foreign_policy,method=multiple_choice_joint,'
    ],
    'narrative_qa:': ['narrative_qa:'],
    'natural_qa:mode=closedbook,': ['natural_qa:mode=closedbook,'],
    'natural_qa:mode=openbook_longans,': ['natural_qa:mode=openbook_longans,'],
    'wmt_14': [
        'wmt_14:language_pair=cs-en,',
        'wmt_14:language_pair=de-en,',
        'wmt_14:language_pair=fr-en,',
        'wmt_14:language_pair=hi-en,',
        'wmt_14:language_pair=ru-en,'
    ]
}

def process_helm_data(path='/llmthonskdir/felipe/helm/lite/v1.0.0'):
    """Process HELM Lite data and save results."""
    directory = Path(path)
    runs = [item.name for item in directory.iterdir() if item.is_dir()]
    
    data2 = {}
    
    for run in runs:
        # Get valid test IDs
        with open(f'{path}/{run}/instances.json') as f:
            instances_data = json.load(f)
        valid_ids = [d['id'] for d in instances_data if d['split']=='test']

        # Get predictions
        with open(f'{path}/{run}/display_predictions.json') as f:
            pred_data = json.load(f)
            
        metric = list(pred_data[0]['stats'].keys())[-1]
        
        subscenario = run[:run.find('model=')]
        model = run[run.find('model=')+6:]
        
        if subscenario not in data2:
            data2[subscenario] = {}

        if any(s in subscenario for s in ['med_qa', 'mmlu', 'narrative_qa', 'wmt_14']):
            data2[subscenario][model] = [d['stats'][metric] for d in pred_data if d['instance_id'] in valid_ids]
        else:
            data2[subscenario][model] = [d['stats'][metric] for d in pred_data]

    # Format final data structure
    data = {
        'data': {},
        'models': list(np.unique([list(data2[subscenario].keys()) for subscenario in data2.keys()]))
    }

    for sub in data2:
        data['data'][sub] = {
            'correctness': np.array([data2[sub][model] for model in data['models']]).T
        }

    # Save processed data
    with open('helm_lite.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return data

def print_scenario_stats(data):
    """Print statistics for each scenario."""
    for scenario in HELM_LITE_SCENARIOS:
        print(scenario)
        print(len(HELM_LITE_SCENARIOS[scenario]))
        means = np.sort(np.vstack([
            data['data'][sub]['correctness'].mean(axis=0) 
            for sub in HELM_LITE_SCENARIOS[scenario]
        ]).mean(axis=0))
        print(np.round(means, 3))
        print('\n')

if __name__ == "__main__":
    data = process_helm_data()
    print_scenario_stats(data) 