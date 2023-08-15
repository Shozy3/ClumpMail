from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os


# Define the model name
model_name = "google/pegasus-cnn_dailymail"

# Download and save the model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Save them locally
model_dir = os.path.join(script_dir, "local_pegasus_model")
tokenizer_dir = os.path.join(script_dir, "local_pegasus_tokenizer")

model.save_pretrained(model_dir)
model.save_pretrained(tokenizer_dir)