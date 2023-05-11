# MVP demo: serving a Hugging Face model

[RoBERTa](https://arxiv.org/abs/1907.11692) is a variant of BERT, with optimized training that dropped the Next Sentence Prediction task but extended the Masked Language Modeling task with more data in bigger batches.

[Deepset's distilled RoBERTa](https://huggingface.co/deepset/roberta-base-squad2-distilled), pre-trained on SQuAD2, is available via Hugging Face.

![Hugging Face logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

Perfect for this quick demo of Extractive Q & A.

# Instructions

1. Clone this repo to your local machine.
2. From the repo directory, download the pre-trained model (~500MB).

        python -m get_roberta
3. Build and run the Docker image.

        docker build -t tiny-demo .
        docker run -dp 8000:8000 tiny-demo
4. Test the endpoint with sample notebook, `post_qa.ipynb`.

# Next steps

## More records
In practice, Extractive Q & A is useful when the context is a database of records, rather than a single record provided with the question. Deepset and the Hugging Face book provide a blueprint: a Retriever and Reader from Deepset's Haystack library, plus an Elasticsearch server.

## More tokens/record
Distilled RoBERTa is impressive and small, but its max sequence length is 384. BigBird, a tranformer model with sparse attention, accepts input sequences up to 4096.

## Optimizations for fast inference
This tiny demo is containerized for stateless serving in real time. Distillation is only one step toward a light-weight model. Converting the PyTorch model to ONNX format would allow for ORT optimizations. Quantization is also proven to decrease latency of predictions with only moderate impact on their quality.