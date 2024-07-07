import pandas as pd
import streamlit as st

from src.CreateData.TemplatesGenerator.ConfigParams import ConfigParams
from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants


class SelectAxes:
    def __init__(self):
        self.override_options = ConfigParams.override_options
        if 'selected_best_value_axes' not in st.session_state:
            st.session_state['selected_best_value_axes'] = list(self.override_options.keys())
        if 'selected_average_value_axes' not in st.session_state:
            st.session_state['selected_average_value_axes'] = []

    def select_causal_axes(self):
        """
        Creates multiselect widgets for selecting axis options.

        The user can select which axis options to use for getting the best value
        and for getting the average value.
        """

        st.markdown("## Heatmap of the accuracy of the templates")
        st.markdown("""
        To determine the optimal set of values for this model and dataset, 
        please select the axes for calculating the best value.
        For each selected axis, a specific value will be recommended.
        For the axes you do not select, an average over all other possible choices will be calculated. 
        """)

        # Multiselect for axis options to get the best value
        self.select_best_value_axes()

        # Multiselect for axis options to get the average value
        self.select_average_value_axes()

        return st.session_state['selected_best_value_axes']

    def select_best_value_axes(self) -> None:
        """
        Allows the user to select axis options to get the best value.

        Automatically updates the selection in case of changes.
        """
        select_best_values_title = ("Select axes to determine best value "
                                    "(we recommend the best value for these axes)")
        st.multiselect(select_best_values_title,
                       list(self.override_options.keys()),
                       key="selected_best_value_axes",
                       on_change=self.update_average_value_axes)

    def select_average_value_axes(self) -> None:
        """
        Allows the user to select axis options to get the average value.

        Automatically updates the selection in case of changes.
        """
        select_avg_values_title = ("Select Axes for which no specific value is recommended, "
                                   "but an average over all possible choices will be calculated.  \n"
                                   "*(The complement set of the selected axes)")
        st.multiselect(select_avg_values_title,
                       list(self.override_options.keys()),
                       key="selected_average_value_axes",
                       on_change=self.update_best_value_axes)

    def update_average_value_axes(self) -> None:
        """
        Updates the session state variable for axis options without a value.

        This function is triggered when the user selects/deselects options
        in the multiselect for axis options to get the best value.
        """
        st.session_state['selected_average_value_axes'] = \
            list(set(self.override_options.keys()) - set(st.session_state['selected_best_value_axes']))

    def update_best_value_axes(self) -> None:
        """
        Updates the session state variable for axis options with a value.

        This function is triggered when the user selects/deselects options
        in the multiselect for axis options to get the average value.
        """
        st.session_state['selected_best_value_axes'] = \
            list(set(self.override_options.keys()) - set(st.session_state['selected_average_value_axes']))

    def write_best_combination(self, best_row: pd.Series) -> None:
        """
        Write the best combination of the values of the axes to a file.
        @param best_row: The best combination of the values of the axes.
        @return: None
        """
        st.markdown("#### The best set of caused factors:")
        red_color = "#FF5733"
        colors = ['blue', 'green', 'purple']
        for i, option in enumerate(st.session_state['selected_best_value_axes']):
            value = best_row[option]
            if isinstance(value, str):
                value = value.strip()
            st.markdown(f'<span style="color:{colors[i % len(colors)]}">{option}: **{value}**</span>',
                        unsafe_allow_html=True)
        # take a red color for the accuracy
        selected_average_value_axes_str = ', '.join(st.session_state['selected_average_value_axes'])
        # take only the 2 digits after the point
        accuracy = f"{best_row['accuracy']:.2f}"
        acc_text = (f"These factors " +
                    (f"(when we average the values of the other axes:"
                     f" <strong>{selected_average_value_axes_str}</strong>) " if selected_average_value_axes_str else "") \
                    + f"contribute to the **accuracy** <strong style='color:{red_color}'>{accuracy}</strong>")
        st.markdown(acc_text, unsafe_allow_html=True)
