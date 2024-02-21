from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")

# Define your input sentence
sentence = "<answer> The United States <context> The United States is a federal republic located primarily in North America. It is comprised of 50 states and is known for its diverse culture, economic strength, and global influence in various fields such as politics, technology, and entertainment.."

# Tokenize the input sentence
inputs = tokenizer.encode("translate English to English question: " + sentence, return_tensors="pt")

# Generate question from the input sentence
output = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)

# Decode the generated question
question = tokenizer.decode(output[0], skip_special_tokens=True)

print("Input Sentence:", sentence)
print("Generated Question:", question)