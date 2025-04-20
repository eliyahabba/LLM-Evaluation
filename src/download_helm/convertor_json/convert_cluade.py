import json
import os
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# יצירת מבנה נתונים לאחסון כל הדוגמאות
all_examples = []

# מידע על מודלים ידוע מראש - אפשר להחליף זאת בקריאה מקובץ חיצוני
MODEL_INFO = {
    "01-ai/yi-34b": {
        "family": "Yi",
        "architecture": "transformer",
        "parameters": 34,
        "context_window": 8192,
        "is_instruct": False,
        "hf_path": "01-ai/yi-34b",
        "revision": None
    },
    # אפשר להוסיף מודלים נוספים כאן
}


def load_json_file(filepath: str) -> dict:
    """טעינת קובץ JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_evaluation_id(run_spec: dict, instance_id: str) -> str:
    """יצירת מזהה ייחודי להערכה"""
    # שילוב של שם הריצה ומזהה הדוגמה
    run_name = run_spec.get("name")
    if not run_name:
        run_name = "unknown"
    return f"{run_name}-{instance_id}"


def get_model_info(model_name: str) -> dict:
    """השגת מידע על המודל"""
    # בדיקה אם המודל מוכר במיפוי הידוע מראש
    if model_name in MODEL_INFO:
        return {
            "model_info": {
                "name": model_name,
                "family": MODEL_INFO[model_name]["family"]
            },
            "configuration": {
                "context_window": MODEL_INFO[model_name]["context_window"],
                "architecture": MODEL_INFO[model_name]["architecture"],
                "parameters": MODEL_INFO[model_name]["parameters"],
                "is_instruct": MODEL_INFO[model_name]["is_instruct"],
                "hf_path": MODEL_INFO[model_name]["hf_path"],
                "revision": MODEL_INFO[model_name]["revision"]
            },
            "inference_settings": {
                "quantization": {
                    "bit_precision": "none",
                    "method": "None"
                }
            }
        }
    else:
        # אם המודל לא מוכר, נחזיר מידע בסיסי בלבד
        return {
            "model_info": {
                "name": model_name,
                "family": None
            },
            "configuration": {
                "context_window": None,  # אין מידע
                "architecture": None,
                "parameters": None,
                "is_instruct": None,
                "hf_path": model_name,
                "revision": None
            },
            "inference_settings": {
                "quantization": {
                    "bit_precision": "none",
                    "method": "None"
                }
            }
        }


def create_model_section(run_spec: dict, display_request: dict) -> dict:
    """יצירת חלק המודל בסכמה החדשה"""
    model_name = run_spec.get("adapter_spec", {}).get("model")
    model_info = get_model_info(model_name)

    # הוספת פרמטרי היצירה מתוך נתוני ה-display_request
    # כאן אנחנו לוקחים את הערכים אך ורק מהנתונים עצמם, ללא ערכי ברירת מחדל
    model_info["inference_settings"]["generation_args"] = {
        "use_vllm": None,  # מידע שלא קיים ב-HELM
        "temperature": display_request.get("request", {}).get("temperature"),
        "top_p": display_request.get("request", {}).get("top_p"),
        "max_tokens": display_request.get("request", {}).get("max_tokens"),
        "stop_sequences": display_request.get("request", {}).get("stop_sequences")
    }

    return model_info


def detect_prompt_format(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    מזהה את פורמט הפרומפט מתוך הטקסט - אילו אותיות משמשות למספור ואיזה תו משמש להפרדה
    מחזיר: (enumerator_type, separator, instruction_name)
    """
    if not prompt:
        return None, None, None

    # בדיקה יותר מדויקת של הפורמט של האופציות
    # מחפשת A., B., C., D. ברצף בתוך הטקסט

    # בדיקת אותיות גדולות (A., B., C., D.)
    if re.search(r'A\. .+\s+B\. .+\s+C\. .+\s+D\.', prompt, re.DOTALL):
        enumerator = "capitals"
    # בדיקת אותיות קטנות (a., b., c., d.)
    elif re.search(r'a\. .+\s+b\. .+\s+c\. .+\s+d\.', prompt, re.DOTALL):
        enumerator = "lowercase"
    # בדיקת מספרים (1., 2., 3., 4.)
    elif re.search(r'1\. .+\s+2\. .+\s+3\. .+\s+4\.', prompt, re.DOTALL):
        enumerator = "numbers"
    # בדיקת ספרות רומיות (i., ii., iii., iv.)
    elif re.search(r'i\. .+\s+ii\. .+\s+iii\. .+\s+iv\.', prompt, re.DOTALL) or \
            re.search(r'I\. .+\s+II\. .+\s+III\. .+\s+IV\.', prompt, re.DOTALL):
        enumerator = "roman"
    # בדיקת מקשי מקלדת (@, #, $, %)
    elif re.search(r'@\. .+\s+#\. .+\s+\$\. .+\s+%\.', prompt, re.DOTALL):
        enumerator = "keyboard"
    else:
        enumerator = None

    # בדיקת המפריד בין אפשרויות
    # מחפשת את התבנית שבה האופציות מופרדות

    # בדיקה האם יש שורות חדשות בין האופציות
    if re.search(r'A\. .+\nB\. ', prompt) or re.search(r'1\. .+\n2\. ', prompt):
        separator = "\\n"
    # בדיקה האם יש פסיקים בין האופציות
    elif re.search(r'A\. .+, B\. ', prompt) or re.search(r'1\. .+, 2\. ', prompt):
        separator = ", "
    # בדיקה האם יש נקודה-פסיק בין האופציות
    elif re.search(r'A\. .+; B\. ', prompt) or re.search(r'1\. .+; 2\. ', prompt):
        separator = "; "
    # בדיקה האם יש מקף אנכי בין האופציות
    elif re.search(r'A\. .+\| B\. ', prompt) or re.search(r'1\. .+\| 2\. ', prompt):
        separator = " | "
    # בדיקה האם יש "OR" בין האופציות
    elif re.search(r'A\. .+ OR B\. ', prompt) or re.search(r'1\. .+ OR 2\. ', prompt):
        separator = " OR "
    # בדיקה האם יש "or" בין האופציות
    elif re.search(r'A\. .+ or B\. ', prompt) or re.search(r'1\. .+ or 2\. ', prompt):
        separator = " or "
    else:
        separator = None

    # זיהוי סוג ההנחיות (אם יש)
    instruction_name = None


    return enumerator, separator, instruction_name


def extract_dataset_name(run_spec: dict, scenario: dict) -> str:
    """מנסה לחלץ את שם הדאטה-סט מתוך הנתונים"""
    # נסיון לחלץ מה-run_spec
    if run_spec and "scenario_spec" in run_spec:
        spec = run_spec.get("scenario_spec", {})
        if "args" in spec and "dataset" in spec["args"]:
            return spec["args"]["dataset"]

    # נסיון לחלץ מה-scenario
    if scenario and "name" in scenario:
        return scenario["name"]

    return None  # אם לא הצלחנו לחלץ

def create_prompt_config(run_spec: dict, first_request: dict) -> dict:
    """יצירת חלק הגדרות הפרומפט בסכמה החדשה"""
    adapter_spec = run_spec.get("adapter_spec", {})

    # זיהוי הפורמט מתוך הפרומפט של הבקשה הראשונה
    prompt = first_request.get("request", {}).get("prompt")
    enumerator, separator, instruction_name = detect_prompt_format(prompt)

    return {
        "prompt_class": "MultipleChoice",  # לפי הסכמה שלך, זה חייב להיות אחד מהערכים המוגדרים
        "dimensions": {
            "choices_order": {
                "method": "fixed",
                "description": "Fixed order as provided in dataset"
            },
            "enumerator": enumerator,  # כפי שזוהה מהנתונים
            "instruction_phrasing": {
                "name": None,
                "text": adapter_spec.get("instructions")
            },
            "separator": separator,  # כפי שזוהה מהנתונים
            "shots": adapter_spec.get("max_train_instances")
        }
    }


def extract_correct_answer(references: List[dict]) -> Optional[str]:
    """מציאת התשובה הנכונה מתוך רשימת האפשרויות"""
    for ref in references:
        if "correct" in ref.get("tags", []):
            return ref.get("output", {}).get("text")
    return None


def create_instance_section(instance: dict, display_request: dict) -> dict:
    """יצירת חלק הדוגמה בסכמה החדשה"""
    question_text = instance.get("input", {}).get("text")
    instance_id = instance.get("id")

    # יצירת המיפוי בין האותיות לטקסט התשובות
    references = instance.get("references", [])
    choices = []
    correct_text = extract_correct_answer(references)
    correct_id = None

    for i, ref in enumerate(references):
        choice_id = chr(65 + i)  # A, B, C, D...
        choice_text = ref.get("output", {}).get("text")
        choices.append({"id": choice_id, "text": choice_text})
        if choice_text == correct_text:
            correct_id = choice_id

    # חילוץ מספר מזהה מ-ID
    instance_num = int(instance_id.replace("id", "")) if instance_id and instance_id.startswith("id") else None

    # נסיון לזהות את שם הדאטה-סט מתוך ה-scenario
    scenario_name = None
    if "scenario_spec" in display_request:
        scenario_name = display_request.get("scenario_spec", {}).get("name")

    if not scenario_name:
        scenario_name = "openbookqa"  # אם לא הצלחנו לזהות

    result = {
        "task_type": "classification",  # הכרחי לפי הסכמה
        "raw_input": question_text,
        "language": None,  # אין מידע ישיר בקבצים
        "sample_identifier": {
            "dataset_name": scenario_name,
            "hf_repo": scenario_name,  # בהנחה שזה אותו שם, אפשר לשים None
            "hf_split": instance.get("split"),
            "hf_index": instance_num
        },
        "classification_fields": {
            "full_input": display_request.get("request", {}).get("prompt"),
            "question": question_text,
            "choices": choices,
            "ground_truth": {
                "id": correct_id,
                "text": correct_text
            }
        }
    }

    return result


def create_output_section(prediction: dict) -> dict:
    """יצירת חלק הפלט בסכמה החדשה"""
    predicted_text = prediction.get("predicted_text")
    if predicted_text:
        predicted_text = predicted_text.strip()

    return {
        "response": predicted_text,
        "cumulative_logprob": None  # מידע שלא קיים ב-HELM
    }


def create_evaluation_section(prediction: dict, instance: dict) -> dict:
    """יצירת חלק ההערכה בסכמה החדשה"""
    # מציאת התשובה הנכונה
    references = instance.get("references", [])
    correct_id = None

    # איתור האות (A,B,C,D) של התשובה הנכונה
    for i, ref in enumerate(references):
        if "correct" in ref.get("tags", []):
            correct_id = chr(65 + i)  # A, B, C, D...
            break

    # חישוב הציון - 1.0 אם התשובה נכונה, 0.0 אחרת
    exact_match = prediction.get("stats", {}).get("exact_match")

    return {
        "ground_truth": correct_id,
        "evaluation_method": {
            "method_name": "label_only_match",  # הכרחי לפי הסכמה
            "description": "Compares only the choice identifier/label to evaluate the response."
        },
        "score": exact_match
    }


def process_helm_data(data_dir: str) -> List[dict]:
    """עיבוד נתוני HELM ויצירת דוגמאות בפורמט הסכמה החדשה"""
    # טעינת הקבצים העיקריים
    run_spec = load_json_file(os.path.join(data_dir, "run_spec.json"))
    scenario = load_json_file(os.path.join(data_dir, "scenario.json"))
    instances = load_json_file(os.path.join(data_dir, "instances.json"))
    display_requests = load_json_file(os.path.join(data_dir, "display_requests.json"))
    display_predictions = load_json_file(os.path.join(data_dir, "display_predictions.json"))

    # יצירת מיפוי בין מזהי דוגמאות לבקשות ותחזיות
    request_map = {req["instance_id"]: req for req in display_requests}
    prediction_map = {pred["instance_id"]: pred for pred in display_predictions}

    examples = []

    # יצירת קונפיגורציה של הפרומפט על בסיס הבקשה הראשונה (אם יש)
    first_request = display_requests[0] if display_requests else None
    prompt_config = create_prompt_config(run_spec, first_request)

    # עיבוד כל דוגמה
    for instance in instances:
        instance_id = instance.get("id")

        # בדיקה אם קיימים נתוני בקשה ותחזית עבור דוגמה זו
        if instance_id in request_map and instance_id in prediction_map:
            request = request_map[instance_id]
            prediction = prediction_map[instance_id]

            # יצירת דוגמה בפורמט החדש
            example = {
                "evaluation_id": create_evaluation_id(run_spec, instance_id),
                "model": create_model_section(run_spec, request),
                "prompt_config": prompt_config,  # אותו קונפיג לכל הדוגמאות
                "instance": create_instance_section(instance, request),
                "output": create_output_section(prediction),
                "evaluation": create_evaluation_section(prediction, instance)
            }

            examples.append(example)

    return examples


def main(data_dir: str, output_file: str):
    """פונקציה ראשית לעיבוד והמרת הנתונים"""
    examples = process_helm_data(data_dir)

    # שמירת התוצאות כקובץ JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"הומרו {len(examples)} דוגמאות ונשמרו בקובץ {output_file}")

    # אפשר גם להדפיס דוגמה אחת בפורמט JSON לצורך בדיקה
    if examples:
        print("\nדוגמה ראשונה:")
        print(json.dumps(examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # הגדרת נתיבי הקבצים
    data_directory = r"/Users/ehabba/PycharmProjects/LLM-Evaluation/src/download_helm/helm/lite/v1.0.0/commonsense:dataset=openbookqa,method=multiple_choice_joint,model=01-ai_yi-34b"  # תיקייה שמכילה את קבצי ה-HELM
    output_filepath = "helm_converted.json"  # קובץ הפלט - סיומת json

    main(data_directory, output_filepath)