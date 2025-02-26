import os
import ast
import requests
import re
import sys
import io
import tensorflow as tf
from tqdm import tqdm
from datasets import load_dataset
import gpt_2_simple as gpt2

# Set environment variables for TensorFlow
os.environ["OMP_NUM_THREADS"] = "12" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "12"

# Create necessary directories
os.makedirs("training_data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def process_cornell_dialogs():
    """Process Cornell Movie Dialogs dataset"""
    print("Processing Cornell Movie Dialogs dataset...")
    input_path = "training_data/inputs/cornell movie-dialogs corpus/movie_lines.txt"
    output_path = "training_data/outputs/cornell_dialogs.txt"
    
    with open(input_path, "r", encoding="utf-8", errors="ignore") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for row in infile:
            parts = row.strip().split("+++$+++")
            if len(parts) == 5:
                # parts[4] is the raw dialogue text
                dialogue_line = parts[4].strip()
                outfile.write(dialogue_line + "\n")
    
    print("Cornell Movie Dialogs written to:", output_path)

def process_daily_dialog():
    """Process Daily Dialog dataset"""
    print("Processing Daily Dialog dataset...")
    dataset_splits = ["train", "validation", "test"]
    output_path = "training_data/outputs/daily_dialog_full.txt"

    with open(output_path, "w", encoding="utf-8") as outfile:
        for split_name in dataset_splits:
            ds = load_dataset("daily_dialog", split=split_name)
            for conversation in ds:
                for line in conversation["dialog"]:
                    outfile.write(line.strip() + "\n")
                outfile.write("\n")

    print("DailyDialog (train+validation+test) written to:", output_path)

def process_mbpp_data():
    """Process MBPP (Mostly Basic Python Problems) dataset"""
    print("Processing MBPP dataset...")
    mbpp_splits = ["train", "validation", "test"]
    mbpp_output_path = "training_data/outputs/mbpp_data.txt"

    with open(mbpp_output_path, "w", encoding="utf-8") as outfile:
        for split_name in mbpp_splits:
            print(f"Loading MBPP split: {split_name}")
            ds_mbpp = load_dataset("mbpp", split=split_name)
            
            for row in ds_mbpp:
                problem = row["text"].strip()
                code = row["code"].strip()
                # Write Q&A style with a blank line after
                outfile.write(f"Question:\n{problem}\nAnswer:\n{code}\n\n")

    print("MBPP data written to:", mbpp_output_path)

def decode_byte_literal(bstring):
    """
    The dataset strings look like: 
       "b'4+5=\\n'" 
    or 
       "b'Total of 0.06 and -1977321735.\\n'"
    We'll parse out the real text by:
      1) removing the leading "b'"
      2) removing the trailing "'"
      3) unescaping \\n or other escapes
    A robust way is to use ast.literal_eval if it starts with "b'".
    """
    raw = bstring.strip()
    if raw.startswith("b'") or raw.startswith('b"'):
        # parse as a Python byte literal
        try:
            byte_obj = ast.literal_eval(raw)  # this gives us a Python bytes object
            return byte_obj.decode("utf-8", errors="replace").strip()
        except Exception:
            # fallback: just remove b'..'
            raw = raw[2:]
            if raw.endswith("'") or raw.endswith('"'):
                raw = raw[:-1]
            return raw.replace("\\n", "").strip()
    else:
        # If it doesn't start with b', just do a normal strip
        return raw.replace("\\n", "").strip()

def process_math_data():
    """Process math dataset"""
    print("Processing math dataset...")
    math_output_path = "training_data/outputs/math_data.txt"

    print("Loading math_dataset (arithmetic__add_or_sub, 50% of train)...")
    ds_math = load_dataset(
        "math_dataset",
        "arithmetic__add_or_sub",
        split="train[:50%]",
        trust_remote_code=True
    )

    PROBLEM_KEY = "question"
    SOLUTION_KEY = "answer"

    with open(math_output_path, "w", encoding="utf-8") as outfile:
        for example in ds_math:
            raw_problem = str(example[PROBLEM_KEY])
            raw_solution = str(example[SOLUTION_KEY])

            decoded_problem = decode_byte_literal(raw_problem)
            decoded_solution = decode_byte_literal(raw_solution)

            # Now format each Q&A pair
            outfile.write(f"Q: {decoded_problem}\nA: {decoded_solution}\n\n")

    print(f"Math dataset written to: {math_output_path}")

def process_natural_questions():
    """Process Natural Questions dataset"""
    print("Processing Natural Questions dataset...")
    nq_output_path = "training_data/outputs/nq_data.txt"

    print("Loading Natural Questions dataset...")
    nq_dataset = load_dataset("natural_questions", split="train")

    with open(nq_output_path, "w", encoding="utf-8") as outfile:
        for row in nq_dataset:
            question = row["question_text"].strip()
            answers = row["annotations"]["short_answers"]
            answer_text = ", ".join([a["text"] for a in answers if a["text"]])  # Combine multiple answers
            context = row.get("document_text", "").strip()
            outfile.write(f"Question: {question}\nContext: {context}\nAnswer: {answer_text}\n\n")

    print("Natural Questions data written to:", nq_output_path)

def process_pile_dataset():
    """Process The Pile dataset (smaller subset)"""
    print("Processing The Pile dataset...")
    pile_output_path = "training_data/outputs/pile_data.txt"
    
    # The Pile is extremely large, so we'll only use a small subset
    print("Loading a subset of The Pile dataset...")
    try:
        # Load just a small slice of the validation set to keep things manageable
        pile_dataset = load_dataset("EleutherAI/pile", split="validation[:1000]")
        
        with open(pile_output_path, "w", encoding="utf-8") as outfile:
            for row in pile_dataset:
                text = row["text"].strip()
                # Skip empty entries
                if text:
                    outfile.write(text + "\n\n")
        
        print(f"The Pile subset written to: {pile_output_path}")
    except Exception as e:
        print(f"Error loading The Pile dataset: {e}")
        print("Trying alternative approach with specific Pile components...")
        try:
            # Try loading specific components of The Pile that might be more accessible
            pile_subset = load_dataset("EleutherAI/pile-cc", split="validation[:500]")
            
            with open(pile_output_path, "w", encoding="utf-8") as outfile:
                for row in pile_subset:
                    text = row["text"].strip()
                    if text:
                        outfile.write(text + "\n\n")
            
            print(f"The Pile CC subset written to: {pile_output_path}")
        except Exception as e2:
            print(f"Error loading The Pile CC dataset: {e2}")

def process_openbookqa():
    """Process OpenBookQA dataset"""
    print("Processing OpenBookQA dataset...")
    openbookqa_output_path = "training_data/outputs/openbookqa_data.txt"

    print("Loading OpenBookQA dataset...")
    openbookqa_dataset = load_dataset("openbookqa", split="train")

    with open(openbookqa_output_path, "w", encoding="utf-8") as outfile:
        for row in openbookqa_dataset:
            question = row["question_stem"].strip()
            choices = row["choices"]["text"]
            answer = row["answerKey"].strip()
            outfile.write(f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer: {answer}\n\n")

    print("OpenBookQA data written to:", openbookqa_output_path)

def count_tokens():
    """Count tokens in combined dataset"""
    print("Counting tokens in combined dataset...")
    from gpt_2_simple import encoder
    
    checkpoint_path = "models/124M"
    enc = encoder.get_encoder(checkpoint_path)
    
    with open("combined_dataset.txt", "r") as f:
        dataset = f.read()
    
    tokens = enc.encode(dataset)
    num_tokens = len(tokens)
    
    print(f"Total number of tokens in the dataset: {num_tokens}")

class ProgressCapture(io.StringIO):
    """
    Capture output line-by-line, parse 'Step X' from each line,
    and update a TQDM progress bar to reflect how many steps have completed.
    """
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar
        self.buffer = ""
        # Keep track of the last step we saw, so we can increment TQDM
        self.last_step = 0

    def write(self, text):
        # Accumulate writes in our buffer
        self.buffer += text
        
        # Split on newlines to parse complete lines only
        lines = self.buffer.split('\n')
        self.buffer = lines[-1]  # keep the last partial line
        complete_lines = lines[:-1]
        
        for line in complete_lines:
            # Print everything to the real console too (optional)
            original_stdout.write(line + "\n")
            
            # Look for lines like "Step 123"
            match = re.search(r"Step\s+(\d+)", line)
            if match:
                try:
                    current_step = int(match.group(1))
                    # Update TQDM by the difference so TQDM can estimate ETA
                    increment = current_step - self.last_step
                    if increment > 0:
                        self.progress_bar.update(increment)
                        self.last_step = current_step
                except ValueError:
                    # The substring didn't parse cleanly as an int
                    pass

def train_model():
    """Train the GPT-2 model"""
    print("Starting model training...")
    model_name = "124M"
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    file_name = "combined_dataset.txt"
    if not os.path.isfile(file_name):
        url = "combined_dataset.txt"
        data = requests.get(url)
        with open(file_name, 'w') as f:
            f.write(data.text)

    sess = gpt2.start_tf_sess()

    # Total training steps
    steps = 50
    progress_bar = tqdm(total=steps, desc="Training Progress", dynamic_ncols=True)

    # Keep a reference so we can restore original sys.stdout
    original_stdout = sys.stdout

    try:
        # Redirect sys.stdout to capture the training steps
        sys.stdout = ProgressCapture(progress_bar)

        # Increase batch_size to use more CPU/RAM during each step
        gpt2.finetune(
            sess,
            file_name,
            model_name=model_name,
            steps=steps,
            overwrite=True,
            print_every=5,
            run_name="rorchat1",
            batch_size=4
        )
    finally:
        # Always restore original stdout, even if there's an error
        sys.stdout = original_stdout

    progress_bar.close()

    # Once training completes, generate text
    gpt2.generate(sess)
    
    return sess

def generate_text(sess=None):
    """Generate text with the trained model"""
    if sess is None:
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, run_name="rorchat1")
    
    persona = """
    You are RorChat, a helpful and friendly AI assistant. You are not very smart but you try your best. 
    You were made by Rory, who is your creator. You don't know much else about him.
    """

    def generate_response(prompt):
        # Construct a conversation-style prompt
        full_prompt = f"{persona}\nUser: {prompt}\nRorChat:"
        response = gpt2.generate(
            sess,
            run_name="rorchat1",
            prefix=full_prompt,
            length=100,
            temperature=0.7,
            top_p=0.9,
            return_as_list=True
        )[0]

        # Extract only RorChat's response (after "RorChat:")
        if "RorChat:" in response:
            return response.split("RorChat:")[1].strip()
        else:
            return response.strip()

    test_response = generate_response("Who are you?")
    print("Test response to 'Who are you?':")
    print(test_response)
    
    return generate_response

def main():
    """Main function to run all processes"""
    print("=== RORCHAT DATA PROCESSING AND TRAINING ===")
    
    # Process all datasets
    try:
        process_cornell_dialogs()
    except Exception as e:
        print(f"Error processing Cornell dialogs: {e}")
    
    try:
        process_daily_dialog()
    except Exception as e:
        print(f"Error processing Daily Dialog: {e}")
    
    try:
        process_mbpp_data()
    except Exception as e:
        print(f"Error processing MBPP data: {e}")
    
    try:
        process_math_data()
    except Exception as e:
        print(f"Error processing Math data: {e}")
    
    try:
        process_natural_questions()
    except Exception as e:
        print(f"Error processing Natural Questions: {e}")
    
    try:
        process_openbookqa()
    except Exception as e:
        print(f"Error processing OpenBookQA: {e}")
    
    try:
        process_pile_dataset()
    except Exception as e:
        print(f"Error processing The Pile dataset: {e}")
    
    # Count tokens in combined dataset
    try:
        count_tokens()
    except Exception as e:
        print(f"Error counting tokens: {e}")
    
    # Train model
    try:
        sess = train_model()
        generate_text(sess)
    except Exception as e:
        print(f"Error during model training: {e}")
    
    print("=== PROCESSING AND TRAINING COMPLETE ===")

if __name__ == "__main__":
    main() 