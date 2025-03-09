import sys
from pathlib import Path

import streamlit as st

file_path = Path(__file__).parents[3]
sys.path.append(str(file_path))

from src.streamlit_app.ui_components.FindCombinations import FindCombinations
from src.experiments.experiment_preparation.configuration_generation.TemplateVariationDimensions import TemplateVariationDimensions
from src.utils.Constants import Constants

BestOrWorst = Constants.BestOrWorst

if __name__ == "__main__":
    title = "Select whether you want to see the best or the worst prompt formatting variations for the models."
    # Customize the look of the selectbox with color and font size using CSS
    st.sidebar.markdown(f'<span style="color:blue ; font-size: 16px;">{title}</span>', unsafe_allow_html=True)
    # Dropdown for selecting best or worst combinations using the Enum directly
    best_or_worst = st.sidebar.selectbox(
        "Choose the best or worst combinations",
        options=list(BestOrWorst),
        format_func=lambda x: x.value  # This will display the string value of the Enum in the selectbox
    )

    # Use BestOrWorst class to get the value of the selected option
    best_combinations_displayer = FindCombinations(
        best_or_worst,
        TemplateVariationDimensions.override_options,
        families=True
    )
    best_combinations_displayer.evaluate()
