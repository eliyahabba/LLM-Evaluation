import streamlit as st
import evaluate
metric = evaluate.load("unitxt/metric")


def display_code_example_1():
    code_example_1 = """
    predictions_1 = ['\nThe correct answer is D.']
    references_1 = [{'metrics': ['metrics.accuracy'],
                     'source': 'Context: Tree rings, ice cores, and varves indicate the environmental conditions at the time they were made. Question: Ice cores, varves and what else indicate the environmental conditions at the time of their creation? Choices: A. magma\nB. fossils\nC. mountain ranges\nD. tree rings Answer:\n',
                     'target': 'D. tree rings', 'references': ['D. tree rings'],
                     'task_data': '{"context": "Tree rings, ice cores, and varves indicate the environmental conditions at the time they were made.", "context_type": "paragraph", "question": "Ice cores, varves and what else indicate the environmental conditions at the time of their creation?", "choices": ["magma", "fossils", "mountain ranges", "tree rings"], "answer": 3, "options": ["A. magma", "B. fossils", "C. mountain ranges", "D. tree rings"]}',
                     'group': 'unitxt',
                     'postprocessors': ['processors.to_string_stripped', 'processors.take_first_non_empty_line',
                                        'processors.match_closest_option']}]
    # answer is 'D. tree rings'
    scores_1 = metric.compute(predictions=predictions_1, references=references_1)
    results_1 = scores_1[0]['score']['instance']
    {'accuracy': 0.0, 'score': 0.0, 'score_name': 'accuracy'}
    """
    st.code(code_example_1)
def display_code_example_2():
    code_example_2 = """
    predictions_2 = ['\nThe correct answer is D.']
    references_2 = [{'metrics': ['metrics.accuracy'],
                     'source': 'Context: Tree rings, ice cores, and varves indicate the environmental conditions at the time they were made. Question: Ice cores, varves and what else indicate the environmental conditions at the time of their creation? Choices: A. magma\nB. fossils\nC. mountain ranges\nD. tree rings Answer:\n',
                     'target': 'D. tree rings', 'references': ['D. tree rings'],
                     'task_data': '{"context": "Tree rings, ice cores, and varves indicate the environmental conditions at the time they were made.", "context_type": "paragraph", "question": "Ice cores, varves and what else indicate the environmental conditions at the time of their creation?", "choices": ["magma", "fossils", "mountain ranges", "tree rings"], "answer": 3, "options": ["A. magma", "B. fossils", "C. mountain ranges", "D. tree rings"]}',
                     'group': 'unitxt', 'postprocessors': ["processors.first_character"]}]
    # answer is 'D. tree rings'
    scores_2 = metric.compute(predictions=predictions_2, references=references_2)
    results_2 = scores_2[0]['score']['instance']
    {'accuracy': 0.0, 'score': 0.0, 'score_name': 'accuracy'}
    """
    st.code(code_example_2)
def main():
    display_code_example_1()
    display_code_example_2()

if __name__ == "__main__":
    main()
