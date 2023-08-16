import os
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Define the model name
model_name = "google/pegasus-cnn_dailymail"

# Download and save the model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save them in the script's directory
model_dir = os.path.join(script_dir, "local_pegasus_model")
tokenizer_dir = os.path.join(script_dir, "local_pegasus_tokenizer")

model.save_pretrained(model_dir)
tokenizer.save_pretrained(tokenizer_dir)
