prompt1 = """Your task is to determine whether the input log template is normal or anomaly. 
Input Log Template: {question}
{context}
NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. You should generate reasons for your judgment.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""


prompt2 = """Your task is to determine whether the input log template is normal or anomaly. 
Input Log Template: {question}
{context}
Analyze the Input Log Template independently:
 - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
 - In the log template, the parameters are replaced by <*>, so you should never consider <*> and missing values as the reason for abnormal log.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. You should generate reasons for your judgment.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""


prompt3 = """Your task is to determine whether the input log template is normal or anomaly. 

Input Log Template: {question}
1 Known Normal Log Templates: {context}

Perform the following steps to check whether the input log template is abnormal:
1. Compare the Input Log Template with each of the Known Normal Log Templates.
2. Determine if the input Log Template is structurally and semantically identical or very similar to the Known Normal Log Templates.
   - If yes, classify the Input Log Template as Normal.
   - If no, proceed to step 3.
3. Analyze the Input Log Template independently and provide an assessment:
   - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
   - In the log template, the parameters are replaced by <*>, so you should never consider <*> and missing values as the reason for abnormal log.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. You should generate reasons for your judgment.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""


prompt4 = """Your task is to determine whether the input log template is normal or anomaly. 

Input Log Template: {question}
5 Known Normal Log Templates: {context}

Perform the following steps to check whether the input log template is abnormal:
1. Compare the Input Log Template with each of the Known Normal Log Templates.
2. Determine if the input Log Template is structurally and semantically identical or very similar to the Known Normal Log Templates.
   - If yes, classify the Input Log Template as Normal.
   - If no, proceed to step 3.
3. Analyze the Input Log Template independently:
   - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
   - In the log template, the parameters are replaced by <*>, so you should never consider <*> and missing values as the reason for abnormal log.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. 
Output format: Return back in JSON format, including keys: is_anomaly.
Answer:"""


prompt5 = """Your task is to determine whether the input log template is normal or anomaly. 

Input Log Template: {question}
5 Known Normal Log Templates: {context}

Perform the following steps to check whether the input log template is abnormal:

1. Compare the Input Log Template with each of the Known Normal Log Templates.
2. Determine if the input Log Template is structurally and semantically identical or very similar to the Known Normal Log Templates.
   - If yes, classify the Input Log Template as Normal.
   - If no, proceed to step 3.
3. Analyze the Input Log Template independently and provide an assessment:
   - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
   - In the log template, the parameters are replaced by <*>, so you should never consider <*> and missing values as the reason for abnormal log.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. You should generate reasons for your judgment.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""