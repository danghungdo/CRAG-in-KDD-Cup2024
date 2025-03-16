#!/usr/bin/env python3

INSTRUCTIONS = """
# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# Output: 
Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".
"""

IN_CONTEXT_EXAMPLES = """
# Examples:
Question: how many seconds is 3 minutes 15 seconds?
Ground truth: ["195 seconds"]
Prediction: 3 minutes 15 seconds is 195 seconds.
Accuracy: True

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
Accuracy: False

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I am sorry I don't know.
Accuracy: False
"""

COT_PROMPT = """_For the given question and multiple references from web pages, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""


FEWSHOT_COT_MOVIE_KG = """_For the given question and multiple references from web pages and Mock API, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_MUSIC_KG = """_For the given question and multiple references from web pages and Mock API, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_SPORTS_KG = """_For the given question and multiple references from Mock API, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_FINANCE_KG = """_For the given question and multiple references from Mock API, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_MOVIE = """_For the given question and multiple references from web pages, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_MUSIC = """_For the given question and multiple references from web pages, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_SPORTS = """_For the given question and multiple references from web pages, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""

FEWSHOT_COT_FINANCE = """_For the given question and multiple references from web pages, follow a step-by-step reasoning approach before providing the final answer._  


## **Instructions for Answering the Question:**  

### **1. Analyze the Question & References:**  
- Read the question carefully and examine all provided references.  
- Determine if the references contain sufficient and relevant information to answer the question.  

### **2. Classify the Question Type:**  
- **Static / Slow-Changing:** If the information in references is adequate, use your prior knowledge but only if you are **100% certain**.  
  - When using prior knowledge, always consider the **query time** and ensure the information remains relevant as of that date.  
- **Fast-Changing / Real-Time:** You **must rely on the provided references** to answer the question, even if you have prior knowledge.  

### **3. Step-by-Step Reasoning:**  
- If references provide conflicting or ambiguous information, prioritize credibility and cross-verify details.  
- Ensure the answer is fact-based, concise, and neutral. Avoid assumptions beyond the given information unless required for clarity.  
- When using prior knowledge, explicitly consider and mention whether the information is still valid given the **query time**.  

### **4. Deliver the Final Answer:**  
- If the references are sufficient, provide a **precise and well-supported response**.  
- If the references are **insufficient or unclear**, respond with **"I don't know"** rather than speculating.
- The final answer should be derived from reasoning and few words as possible.

---  

## **Response Format:**  

```  
`## Thought\n`  
(Your logical reasoning process based on the provided instructions.)  

`## Final Answer\n`
(The concise and objective final answer, it must be few words as possible.)  
```  
> **Important:** The final answer should be consistent with your `## Thought` and clearly supported by the references or by your validated prior knowledge (taking into account querry time).  

---  

### Question
{query}

### Query Time
{query_time}

### Question Type
{dynamic}

### References
{references}
"""