from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering


def run(checkpoint):
    """ download model and associated tokenizer from https://huggingface.co/models """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    tokenizer.save_pretrained(checkpoint, from_pt=True)
    model.save_pretrained(checkpoint, from_pt=True)


if __name__ == "__main__":
    run(checkpoint="deepset/roberta-base-squad2-distilled")