# GlaggleBot
GlaggleBot is a text-generation bot powered by a character-level GRU language model trained on private Discord message history.
The project includes a full training pipeline for processing Discord message exports, training a PyTorch recurrent neural network, saving the model, and running the model through
Discord slash commands.

The private message dataset and t rained model are intentionally excluded from the repository for user privacy.

The project was built as a personal/class project to experiment with neural text generation and Discord bot deployment. It is not a general-purpose chatbot or LLM.
Glagglebot is designed to generate short, server-style messages based on either a default prompot or recent Discord channel context.

# Features
- Character-level neural text generation
- GRU-based recurrent neural network implemented in PyTorch
- Discord message export preprocessing
- Message cleaning and normalization
- Spam message filtering
- Train/Test/Validation splitting
- Training metrics for loss, accuracy, and perplexity
- Hyperparameter grid search for model tuning
- Saved model checkpoint for later inference
- Discord slash commands for random and context-aware generation
- Temperature and output-length controls

# Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib
- discord.py
- python-dotenv

# How It Works
1. Data Processing
   
The training script loads exported Discord messages, normalizes names/text, removes empty messages, filters repeated spam, and keeps messages only from channels where multiple users participated.
The processed messages are split into training, validation, and test sets using chronological ordering.

2. Model Training

GlaggleBot trains a character-level GRU model using Pytorch. The model learns to predict the next character in a sequence of Discord message style text.

The training pipeline tracks:
- Cross-entropy loss
- Next-character accuracy
- Perplexity
- Training and validation curves
- Hyperparameter search results

After training, the final model is saved as a checkpoint file.

3. Runtime Generation

The runtime module loads the saved checkpoint, rebuilds the GRU model, and generates text from a prompt using temperature-based sampling.

4. Discord Bot

The discord bot exposes slash commands for generation:

- /generate_random generates text from a default prompt ('lol').
- /generate uses recent channel messages as context and asks the model to continue from that context.
