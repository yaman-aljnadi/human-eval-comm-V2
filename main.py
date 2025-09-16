from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
humanevalcomm = load_dataset("jie-jw-wu/HumanEvalComm")
humanevalcomm['train'][0]

# load model and tokenizer from Hugging Face
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (FP16 for GPU; use load_in_8bit=True for 8-bit quantization if needed)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",   # automatically place model layers on GPU(s)
    torch_dtype="auto"   # use float16 if supported
)

# Create a pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


prompt = """Write a Python function that calculates Fibonacci numbers using recursion. but I need the answer to be in a JSON format as the follwing 

[
{answer = ""},
{question_number = "1"},
{score = "0"}
]

"""

# Run inference
output = generator(
    prompt,
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.9,
    do_sample=True
)

print(output[0]["generated_text"])