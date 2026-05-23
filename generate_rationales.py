import os
import json
import pandas as pd
from tqdm import tqdm

# Prompt 
PROMPT_TWEETEVAL = """
You are an expert sentiment annotator for social media. You will be provided with a Tweet and its human-verified sentiment label (Positive, Negative, Neutral). 
Your task is to generate a single, logical sentence explaining WHY the tweet matches this ground truth sentiment based on linguistic cues.

Strict rules:
1. Lexical Grounding: You MUST quote specific words from the tweet. Even for Neutral tweets, identify the informational tokens (e.g., "The words 'remains out' and 'lineup' provide factual sports reporting without emotional modifiers."). Avoid generic explanations like "states a fact" unless no lexical evidence exists.
2. Sarcasm Handling: If the sentiment relies on sarcasm, explicitly state the contrast between the literal words and the intended meaning.
3. Direct Start: Begin directly with the linguistic evidence (No "This tweet is..." or "The label is...").
4. Length: under 20 words.
5. Anti-Leakage: Do NOT use the label word unless necessary.

Return only valid JSON:
{"rationale": "<Your logical justification>"}
"""

PROMPT_ISARCASM = """
You are an expert sarcasm annotator for social media text. You will be provided with a Tweet and its ground truth sarcasm label (Sarcastic or Non-Sarcastic).
Your task is to generate a single, logical sentence explaining the linguistic mechanics of the tweet.

Strict rules:
1. If Sarcastic: Explicitly explain the semantic contrast. What is the literal meaning vs. the contextual reality? (e.g., "The literal praise 'so thoughtful' is used ironically to describe an inconvenience."). Prefer strongest lexical sarcasm cue (nickname, exaggeration, rhetorical phrase, irony marker, or contrast), not generic contrast if a stronger cue exists.
2. If Non-Sarcastic: Explain how the literal words directly convey the intended meaning without hidden contrast.
3. No Fluff: Start directly with the linguistic breakdown (No "This tweet is..." or "The label is...")
4. Length: under 20 words.
5. Anti-Leakage: Do NOT repeat the label word itself unless necessary.

Return only valid JSON:
{"rationale": "<Your logical justification>"}
"""


def create_batch_line(custom_id, system_prompt, user_content):
    """Formats a single row into the JSONL format required by the OpenAI Batch API """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "response_format": { "type": "json_object" },
            "temperature": 0.1
        }
    }

def merge_logic(original_csv, results_jsonl, final_csv, id_prefix):
    """Merges LLM results back into the original CSV"""
    print(f"Merging results from {results_jsonl} into {final_csv}...")
    df = pd.read_csv(original_csv)
    # Adjust for test runs if you only did .head(20)
    # df = df.head(20) 
    
    rationale_map = {}
    with open(results_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            row_idx = int(data['custom_id'].replace(f"{id_prefix}_", ""))
            try:
                message = data['response']['body']['choices'][0]['message']
                raw_content = message.get('content', '{}')
                content = json.loads(raw_content)
                rationale_map[row_idx] = content.get('rationale', "Missing")
            except Exception as e:
                rationale_map[row_idx] = f"Error: {str(e)}"
    
    df['rationale'] = df.index.map(rationale_map)
    df.to_csv(final_csv, index=False)
    print(f"Saved final file to: {final_csv}")

def run_pipeline(task_name, input_csv, batch_input_jsonl, batch_output_jsonl, final_csv, prompt, label_map):
    # If found output file in the folder, merge the results
    if os.path.exists(batch_output_jsonl):
        print(f"[{task_name}] Found results file. Start merging")
        merge_logic(input_csv, batch_output_jsonl, final_csv, task_name)
        return

    # If batch input exists, wait for result
    if os.path.exists(batch_input_jsonl):
        print(f"[{task_name}] Batch input file found {batch_input_jsonl}. Need the result file")
        return

    # If there is not batch input file, start creating batch input
    print(f"[{task_name}] No files found. Generating new batch input")
    df = pd.read_csv(input_csv)
    
    with open(batch_input_jsonl, 'w', encoding='utf-8') as outfile:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            tweet = str(row['text']).replace("\n", " ").strip()
            label_val = row['label'] if 'label' in df.columns else row['sarcastic']
            true_label = label_map.get(label_val, "Unknown")
            
            user_content = f"Label: {true_label}\nTweet: {tweet}"
            line = create_batch_line(f"{task_name}_{index}", prompt, user_content)
            outfile.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"[{task_name}] Batch input created at {batch_input_jsonl}")

# Run
if __name__ == "__main__":
    # Create folder
    os.makedirs("data/batch_api_calls", exist_ok=True)

    # Process Sentiment Data
    run_pipeline(
        task_name="tweeteval_train",
        input_csv="data/raw/sentiment_train.csv",
        batch_input_jsonl="data/batch_api_calls/sentiment_train_batch_input.jsonl",
        batch_output_jsonl="data/batch_api_calls/sentiment_train_batch_output.jsonl", 
        final_csv="data/processed/sentiment_train_with_rationales.csv",
        prompt=PROMPT_TWEETEVAL,
        label_map={0: "Negative", 1: "Neutral", 2: "Positive"})
    run_pipeline(
        task_name="tweeteval_val",
        input_csv="data/raw/sentiment_validation.csv",
        batch_input_jsonl="data/batch_api_calls/sentiment_val_batch_input.jsonl",
        batch_output_jsonl="data/batch_api_calls/sentiment_val_batch_output.jsonl", 
        final_csv="data/processed/sentiment_validation_with_rationales.csv",
        prompt=PROMPT_TWEETEVAL,
        label_map={0: "Negative", 1: "Neutral", 2: "Positive"})

    # Process Sarcasm Data
    run_pipeline(
        task_name="isarc",
        input_csv="data/processed/sarcasm_train_clean.csv",
        batch_input_jsonl="data/batch_api_calls/sarcasm_batch_input.jsonl",
        batch_output_jsonl="data/batch_api_calls/sarcasm_batch_output.jsonl",
        final_csv="data/processed/sarcasm_train_with_rationales.csv",
        prompt=PROMPT_ISARCASM,
        label_map={1: "Sarcastic", 0: "Non-Sarcastic"})