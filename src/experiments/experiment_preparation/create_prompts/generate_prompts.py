from src.experiments.experiment_preparation.create_prompts.prompt_generation import generate_prompts

if __name__ == "__main__":
    generate_prompts(
        prompts_outpath="generated_prompts/my_multiple_choice",
        task_info_folder_path="my_task_info",  # תיקייה שתיצור
        default_prompts_and_hints_json_path="my_default_prompts.json", # הקובץ JSON שיצרנו למעלה
        meta_prompts_module_path="meta_prompt_templates_lmentry", # אפשר להשתמש בתבניות הקיימות
        task_names=["multiple_choice"], # שם המשימה שלך
        num_calls_rephrase=9,  # כמה וריאציות אתה רוצה מכל סוג
        num_calls_chain_of_thought=0,
        num_calls_task_description_generation=1,
        num_calls_gradual_generation=1,
        derive_prompt_template_given_few_shots_examples_folder_path="derive_prompt_template_few_shots",
        derive_prompt_template_given_few_shots_examples_file_names={
            "multi_choice": "derive_prompt_template_few_shots_bbh_mc.txt"
        },
        examples_with_explanations=False
    )