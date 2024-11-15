# Prompt Paraphrasing for MMLU Dataset

## Background
Our goal is to publish a large-scale dataset containing extensive examples from different datasets, with a particular focus on MMLU (Massive Multitask Language Understanding). While the core value of our dataset - its ability to test various phenomena like model robustness and capabilities - doesn't strictly depend on how we word the instructions, and we technically have the freedom to choose any prompt format, it makes sense to align with the conventions established in the literature. This is especially important since we know (from our own research!) how significantly different prompt formulations can affect model performance.
Therefore, we need to be particularly thoughtful in choosing our prompt formats, ensuring they're both precise for robustness testing and useful for other researchers working with the dataset.
## Key considerations

### Zero-shot challenges
When examining zero-shot scenarios, we must consider that responses might contain additional text since we haven't explicitly defined the answer structure.


_Note: Double curly braces {{}} indicate placeholders that will be dynamically replaced with actual values from each specific sample._

## Currently used prompt format
```python
f"Question: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
```
follows a key principle: we wanted to enable effective zero-shot performance by demonstrating the expected response structure through a "clean" example. Instead of providing content-based examples that might influence the model's understanding of the subject matter, we use placeholder text ([question], [choices], [answer]) to show the desired format. This approach is particularly interesting when compared to few-shot methods, where example content inevitably introduces bias through the specific examples chosen. By using "clean" placeholders, we maintain the model's ability to understand the desired response structure while avoiding the arbitrary influence of specific example content. This approach helps models understand how to structure their responses without being primed by specific example content.


### Practical experience
Based on my experience working with our defined prompt format, models consistently output answers in the correct format (a letter followed by the answer), though I haven't measured this formally.

## Review of prompt formats in literature

### 1. MMLU paper (GitHub)
```python
f"The following are multiple choice questions (with answers) about  {{topic}}.\n\n{{question}}\n{{choices}}\nAnswer:"
```

### 2. HELM implementation
```python
f"The following are multiple choice questions (with answers) about {{topic}}.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"
```

### 3. LLM-eval-harness
Initially, they used:
```python
f"Question: {{question}}\n\nChoices: {{choices}}\nAnswer:"
```

This was later identified as inconsistent with the original prompt format and was corrected on May 13, 2023, to align with the version of MMLU paper:
[Link to commit](https://github.com/EleutherAI/lm-evaluation-harness/pull/497/commits/48c6bd6580d03005f67e87015bb1e172a4b0f76d)

## Notable technical considerations

### 1. Double space bug
The original code contains an extra space issue:
The original code contains an extra space issue:
- `about {{topic}}` -> `about  {{topic}}`

Andrej Karpathy identified this issue, but it was decided to maintain the bug to preserve result reproducibility.
[Link to discussion](https://github.com/hendrycks/test/pull/13#issuecomment-1967494573)

### 2. Few-shot format and space impact
Another consideration relates to few-shot formatting. Consider this input example:
- `Answer:` or `Answer:[space]`

```
{{Question}}:
1. cat
2. dog
3. bird
4. lion
Answer: 1.

{{Question}}:
1. one
2. two
3. three
4. four
Answer: or maybe Answer:[space]  
```

The original code didn't include an additional space after the last "Answer:", leading to concerns about model prediction. If the evaluation method checks token probabilities for ["1","2","3","4"], it's incorrect. The proper approach should examine probabilities for [" 1", " 2", " 3", " 4"] (with spaces).
[Link to discussion](https://github.com/hendrycks/test/pull/13#issuecomment-1618797712)

### 3. MMLU pro and Chain of Thought integration
Following Ofir's question, we noted that MMLU Pro typically employs prompts requesting step-by-step reasoning:
```python
f"The following are multiple choice questions (with answers) about biology. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n"
```

Accordingly, the dataset includes detailed step-by-step reasoning annotations. 

Example of a CoT response format:

```
A: Let's think step by step. We refer to ………………………, we conclude that (G) must be the correct answer. The answer is (G).
```

I think it's still fine to run MMLU Pro with regular MMLU instructions potmps. However, we could also add this CoT prompt format to our experiments and even try it with MMLU. One note: evaluating responses that follow the CoT format will require a different approach, as we'll need more sophisticated parsing.

The corresponding prompt template:
```python
def get_mmlu_instructions_with_topic_and_cot(self):
    return (
        f"The following are multiple choice questions (with answers) about {{topic}}. Think step by"
        f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
    )
```


### 4. Topic influence
An underexplored aspect is the impact of the "topic" on responses. I think that it an interesting axis for investigation so 
I recommend testing prompts where the only variation is the presence or absence of the topic.

## Recommended prompts for experiments
[Link to code](https://github.com/eliyahabba/LLM-Evaluation/blob/main/src/experiments/experiment_preparation/datasets_configurations/MMLUProConfig.py)

### 1. Basic with/without topic
```python
def get_mmlu_instructions_with_topic(self) -> str:
    return f"The following are multiple choice questions (with answers) about {{topic}}.\n\n{{question}}\n{{choices}}\nAnswer:"

def get_mmlu_instructions_without_topic(self) -> str:
    return f"The following are multiple choice questions (with answers).\n\n{{question}}\n\n{{choices}}\nAnswer:"
```

### 2. HELM variations
```python
def get_mmlu_instructions_with_topic_helm(self) -> str:
    return f"The following are multiple choice questions (with answers) about {{topic}}.\n\nQuestion: {{question}}\n{{choices}}\nAnswer:"

def get_mmlu_instructions_without_topic_helm(self) -> str:
    return f"The following are multiple choice questions (with answers).\n\nQuestion: {{question}}\n\n{{choices}}\nAnswer:"
```

### 3. MMLU Pro with CoT
```python
def get_mmlu_instructions_with_topic_and_cot(self) -> str:
    return f"The following are multiple choice questions (with answers) about {{topic}}. Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n{{question}}\n{{choices}}\nAnswer:"
```


### 4. Our original prompt
We might not generate new data with it, but we can use the existing data and include in our analysis (even if we don't complete all examples with this prompt).
```python
def get_structured_instruction_text_with_topic(self) -> str:
    return f"Topic: {{topic}}\nQuestion: [question] Choices: [choices] Answer: [answer]\nQuestion: {{question}} Choices: {{choices}} Answer:"
```
