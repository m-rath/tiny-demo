from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering

model_checkpoint = "deepset/roberta-base-squad2-distilled"

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model.save_pretrained(model_checkpoint, from_pt=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.save_pretrained(model_checkpoint, from_pt=True)