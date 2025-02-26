# RorChat: An attempt at building my own LLM in my bedroom

By: Rory Garton-Smith 2025

Please read more about this and my projects here if interested: https://ror.fm/

In early 2025 I trained a custom GPT-2 based AI assistant called "RorChat" cause I wanted to learn how these LLM things work. It was a fun project and I learned a lot - but also surprisingly arduous and took AGES to train (I used colab pro with a T4 GPU to train on 50M tokens).

The results + a little write-up are below. I think Pile was the most impactful dataset for the results - but it's a shame I couldn't get the full Pile dataset to train on (just wayy too large for a hobby project).

I whipped up a little gui in next JS / react to make the I/O a bit more user friendly. It's not very sophisticated but it works.

The model prompt definitely needs a lot of work hahahaha "You were made by Rory, who is your creator. You don't know much else about him." I think Claude's one for comparison is over 200 lines long, this was about all I added - anyway - write up below for anyone interested!

![rorchat](https://github.com/user-attachments/assets/c39cf324-eef4-4112-901e-daa714a2dbfb)


## Overview

The `data_processing_and_training.py` script handles the entire pipeline from dataset preparation to model training and text generation. RorChat uses a fine-tuned 124M parameter GPT-2 model customized with a persona that makes it friendly and approachable.

## Datasets Used

The model was trained on a diverse collection of datasets:

- **Cornell Movie Dialogs Corpus**: Conversational text from movie scripts - not a great dataset tbh
- **DailyDialog**: High-quality multi-turn dialogs for everyday conversation - good dataset for convo bots
- **MBPP (Mostly Basic Python Problems)**: Programming problems and solutions - good dataset for code generation but not great for more nuanced issues, it's not large enough and focusese on competitive coding
- **Math Dataset**: Various arithmetic problems and answers - I didn't see much impact of this dataset tbh
- **Natural Questions**: Real Google search queries and answers - I didn't see much impact of this dataset tbh either, it's not a good fit for this model
- **OpenBookQA**: Question answering dataset focused on elementary science  - this one is pretty cool
- **The Pile**: A curated subset of diverse, high-quality text = imo this is the only one that really mattered for a model this small - it's a good mix of text + code + math + etc - but also totally unpruned so it's like 100x larger than the other datasets combined and took forever to train on + def has security issues in terms of content !

## Training Process on Google Colab

### Setup

Training was performed on Google Colab using a GPU runtime, which was essential for handling the 124M parameter GPT-2 model efficiently.

### Steps Followed

1. Mounted Google Drive to preserve data and model checkpoints between sessions
2. Installed all required dependencies (gpt-2-simple, transformers, datasets, etc.)
3. Downloaded and processed the datasets listed above
4. Combined datasets into a unified training corpus
5. Downloaded the base GPT-2 124M model
6. Fine-tuned the model on the combined dataset
7. Saved the trained model and generated sample responses

### Training Time and Resources

- **Total Training Time**: Approximately 8 hours spread across multiple Colab sessions
- **Hardware Used**: Tesla T4 GPU (16GB VRAM)
- **Steps**: 50 training steps with batch size of 4
- **Token Count**: Approximately 50M tokens processed during training

## Challenges Faced

### Colab Runtime Limitations

The 12-hour runtime limit on Colab Pro was a significant constraint. I had to:
- Implement checkpointing to resume training between sessions
- Use a custom progress capturing mechanism to track training progress
- Save model snapshots to Google Drive to avoid data loss

### Memory Management

- Base GPT-2 (124M) was chosen over larger models to fit within Colab's memory constraints
- Needed to process datasets in smaller chunks to avoid OOM errors
- Used smaller batch sizes (4) to balance between memory usage and training efficiency

### Dataset Processing

- Some datasets (like The Pile) are extremely large and required selecting smaller subsets
- Handling different dataset formats required custom processing for each
- Required fallback mechanisms for datasets that couldn't be loaded directly
- Byte encoding issues in some datasets (like math_dataset) needed special handling

### Connectivity Issues

- Intermittent disconnections from Colab required robust error handling
- Implemented try/except blocks around each dataset processing step to ensure partial progress was maintained

## Using the Trained Model

To use the trained model, follow these steps:

1. Load the model using the `load_model` function
2. Generate text using the `generate_text` function
3. Use the model in a chat interface or other application

## Limitations

- As a 124M parameter model, RorChat has significantly lower capabilities than modern large language models
- The persona makes it clear that it "tries its best" but isn't highly intelligent
- Limited context window compared to modern models
- No knowledge of events after the training cutoff in the datasets
- Tendency to hallucinate information, especially for complex queries
- Best suited for casual conversation rather than factual responses

## Future Improvements

Tbh I'm too busy with my actual projects to do any of this stuff - but if I had time:

- It'd be cool to Implement RLHF
 - The model prompt needs a lot of work hahahaha
- Fine-tune on more specific domains based on intended use cases (I'd love to make this one really good for natural language stat problems)
- Train on a larger GPT-2 model (355M or 774M) with more computational resources = but that's a lot of tokens to train on and a lot of budget for something hobbyist
- Include more programming and technical datasets for better code generation abilities - the pile is not great for this


## Code Structure

The `data_processing_and_training.py` script is organized into several key functions:

- **Dataset Processing Functions**: One function per dataset (e.g., `process_cornell_dialogs()`, `process_pile_dataset()`)
- **Utility Functions**: For tasks like decoding byte literals and counting tokens
- **Training Functions**: For model training and text generation
- **Progress Tracking**: Custom `ProgressCapture` class for monitoring training progress
- **Main Function**: Orchestrates the entire workflow with error handling

## Acknowledgments

OpenAI for the base GPT-2 model
EleutherAI for The Pile dataset
Google Colab for providing the computational resources
The creators of the various datasets used in training
