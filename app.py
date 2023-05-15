import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from flask import Flask, request


app = Flask(__name__)

chkpt = "./deepset/roberta-base-squad2-distilled"
tokenizer = AutoTokenizer.from_pretrained(chkpt)
model = AutoModelForQuestionAnswering.from_pretrained(chkpt)

@app.route("/extract_qa", methods=["POST"])
def extract_qa():
    if request.method == "POST":
        args = request.json
        question = args["question"]
        text = args["text"]
        inputs = tokenizer(
            question, text, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        span = inputs["input_ids"][0][start_idx:end_idx+1]

        answer = tokenizer.decode(span)

        decoded = tokenizer.decode(
            inputs["input_ids"][0]).lstrip('<s>').rstrip('</s>').split('</s></s>')
        question = decoded[0]
        text = decoded[1]

        return json.dumps({'context': text,
                           'question': question,
                           'answer': answer})
