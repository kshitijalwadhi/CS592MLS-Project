import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re

def get_choices():
    return ["A", "B", "C", "D"]

def format_subject(subject):
    return subject.replace("_", " ")

def format_example(example, include_answer=True):
    # Extract question and choices
    prompt = example['question']
    choices = example['choices']
    
    # Add choices to the prompt
    for j, choice in enumerate(choices):
        prompt += "\n{}. {}".format(get_choices()[j], choice)
    
    prompt += "\nAnswer:"
    
    if include_answer:
        # Get the answer (convert to letter if it's an index)
        answer = example['answer']
        if isinstance(answer, (int, np.integer)):
            answer = get_choices()[answer]
        prompt += " {}\n\n".format(answer)
    
    return prompt

def gen_prompt(train_dataset, num_examples=5):
    prompt = "The following are multiple choice questions (with answers) about professional medicine.\n\n"
    
    for i in range(min(num_examples, len(train_dataset))):
        prompt += format_example(train_dataset[i])
    
    return prompt

def extract_answer(generated_text):
    """
    Extract the first letter that matches A, B, C, or D
    """
    # Convert to uppercase to catch both lower and upper case
    generated_text = generated_text.upper()
    
    # Look for first occurrence of A, B, C, or D
    match = re.search(r'[A-D]', generated_text)
    
    return match.group(0) if match else None

def evaluate_mmlu(model_name, dataset_name="professional_medicine", num_few_shot=5):
    # Load dataset
    ds = load_dataset("cais/mmlu", dataset_name)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side='left'
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare logits processor to constrain generation
    def logits_processor(input_ids, scores):
        # Get indices of A, B, C, D tokens
        choice_token_ids = []
        for choice in ['A', 'B', 'C', 'D']:
            # Try different tokenization approaches
            tokens = tokenizer.encode(choice, add_special_tokens=False)
            if tokens:
                choice_token_ids.append(tokens[0])
        
        # Create a mask that zeros out all tokens except A, B, C, D
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        mask[choice_token_ids] = False
        scores[:, mask] = float('-inf')
        
        return scores

    # Metrics tracking
    # total_questions = len(ds['test'])
    total_questions = 10
    correct_predictions = 0
    
    # Evaluation loop
    for idx in range(total_questions):
        # Prepare few-shot prompt
        few_shot_prompt = gen_prompt(ds['dev'], num_few_shot)
        
        # Current test question
        test_question = format_example(ds['test'][idx], include_answer=False)
        
        # Full prompt
        full_prompt = few_shot_prompt + test_question
        
        # Tokenize
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        ).to(model.device)
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_new_tokens=3,  # Very short generation
            do_sample=False,  # Greedy decoding
            temperature=0.0,
            num_return_sequences=1,
            # Use custom logits processor to constrain to A,B,C,D
            logits_processor=[logits_processor]
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Extract predicted answer
        predicted_answer = extract_answer(generated_text)
        
        # Get ground truth
        ground_truth = ds['test'][idx]['answer']
        
        # Normalize ground truth
        if isinstance(ground_truth, (int, np.integer)):
            ground_truth = get_choices()[ground_truth]
        
        # Compare
        if predicted_answer == ground_truth:
            correct_predictions += 1
        else:
            # Optional: print misclassified examples for debugging
            print(f"Misclassified Question: {ds['test'][idx]['question']}")
            print(f"Predicted: {predicted_answer}, Correct: {ground_truth}")
    
    # Calculate accuracy
    accuracy = correct_predictions / total_questions
    
    return {
        "total_questions": total_questions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    # Replace with the model you want to evaluate
    model_name = "facebook/opt-350m"
    
    results = evaluate_mmlu(model_name)
    print(f"Evaluation Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")