import sys
from pathlib import Path

import streamlit as st

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class SamplesNavigator:
    @staticmethod
    def next_sentence():
        file_index = st.session_state["file_index"]
        if file_index < st.session_state["files_number"] - 1:
            st.session_state["file_index"] += 1

        else:
            st.warning('This is the last sentence.')

    @staticmethod
    def previous_sentence():
        file_index = st.session_state["file_index"]
        if file_index > 0:
            st.session_state["file_index"] -= 1
        else:
            st.warning('This is the first sentence.')

    @staticmethod
    def go_to_sentence():
        # split the number of the sentence from the string of st.session_state["sentence_for_tagging"]
        # and then convert it to int
        sentence_number = int(st.session_state["selected_sentence"].split(" ")[1]) - 1
        st.session_state["file_index"] = sentence_number

