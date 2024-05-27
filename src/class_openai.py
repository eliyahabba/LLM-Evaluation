import os

import pandas as pd

os.environ["HF_HOME"] = "/cs/snapless/gabis/gabis/shared/huggingface2"
os.environ["OPENAI_API_KEY"] = "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"
model = "GPT-4o-2024-05-13"
# from openai import OpenAI

from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# client = OpenAI()
system_prompt = "You are an AI assistant skilled in categorizing scenes based on given descriptions, questions, and answers. Your task is to identify the most appropriate category from a predefined list."

def create_prompt(caption: str, question: str, answer: str, categories: list,
                  response_format:str) -> str:
    category_list = ', '.join(categories)
    return f"""Given the caption, question, and answer, identify the category that best describes the scene.

    Caption: {caption}
    
    Question: {question}
    
    Answer: {answer}
    
    Categories: {category_list}
    
    {response_format}"""


def get_answer(prompt: str, model: str = "gpt-4") -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def ask_model(cap: str, q: str, a: str, cats: list, response_format:str) -> str:
    prompt = create_prompt(cap, q, a, cats, response_format)
    return get_answer(prompt)


if __name__ == '__main__':
    file = "all_346_annotation_visual_riddles_dataset - all_346_annotation_visual_riddles_dataset.csv"
    df = pd.read_csv(file)
    cols = ["question", "ground_truth_answer", "Image caption", "Prompt"]
    df = df[cols]
    df.rename(columns={"Image caption": "caption"}, inplace=True)
    df["category"] = ""
    categories = [
        "animal behavior", "biological principles", "common cultural beliefs", "cultural principles",
        "engineering principles", "humor", "nutritional inconsistencies", "physical principles",
        "religion principles", "safety", "social conventions", "temporal principles", "weather conditions"
    ]
    new_suggest = "If you believe there is another category that would be more suitable but is not in the list, please still choose the most appropriate category from the existing ones and then suggest a new category on a new line. Respond only with the category name and, if applicable, the new category suggestion"
    response_foramt = "Please respond with only the category name and nothing else."

    c= 1
    for i, row in df.iterrows():
        # if i > 20:
        #     break
        image_prompt = row["Prompt"]
        if not pd.isna(image_prompt) and image_prompt != "*":
            cap = image_prompt
            print(f"{c}: {cap}")
            c += 1
        else:
            cap = row["caption"]
        q = row["question"]
        a = row["ground_truth_answer"]
        # cat = ask_model(cap, q, a, categories, new_suggest)
        # cat = None
        # df.loc[i, "category"] = cat
        # print(f"Category for row {i}: {cat}")
    # df.to_csv("visual_riddles_dataset.csv")
