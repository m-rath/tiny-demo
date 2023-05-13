# Serve a model with Flask, Gunicorn and Docker

Real-time predictions depend on stateless serving functions. In this tiny demo, deploy a pre-trained NLP model for extractive QA.

## The Model

[Deepset's distilled RoBERTa](https://huggingface.co/deepset/roberta-base-squad2-distilled), pre-trained on SQuAD2, is available via Hugging Face. [RoBERTa](https://arxiv.org/abs/1907.11692) is a variant of BERT, with optimized training that dropped the Next Sentence Prediction task but extended the Masked Language Modeling task with more data in bigger batches.

![Hugging Face logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

# Instructions

1. Clone this repo to your local machine.
2. From the repo directory, download the pre-trained model (~500MB).

        python -m get_roberta
3. Build and run the Docker image.

        docker build -t tiny-demo .
        docker run -dp 8000:8000 tiny-demo
4. Test the endpoint with sample notebook, `post_qa.ipynb`.

# Next steps

## Increase context
Extractive QA is more useful when the context includes a database of short texts, rather than a single short text packaged alongside the question.

Deepset's [Haystack library](https://haystack.deepset.ai/) provides a blueprint: a Retriever and Reader, plus an Elasticsearch server or other [Document Store](https://docs.haystack.deepset.ai/docs/document_store) for the Retriever's fast vector comparisons.

## Increase efficiency
This tiny demo is containerized for stateless serving in real time. The container could be tinier and faster for more efficient scaling.

First, the model could be smaller and faster. Beyond distillation, *quantization* could decrease prediction latency (with a possible trade-off in quality). Converting the PyTorch model to *ONNX format* would allow for ORT optimizations, too.

Second, the model could be its own container, leaving the serving function very light-weight in case of massive horizontal scaling. This pattern is common to production ML frameworks. For instance, Google's Artifact Registry holds containerized models and other components that comprise or support TFX, Kubeflow, etc. When the pipeline's inputs, outputs, and operations are *isolated containers*, experiments and lineage are easier to track.

## Improve accuracy
[Domain Adaptation](https://docs.haystack.deepset.ai/docs/domain_adaptation) can prepare the model to output better answers. A custom training set could include the original training set -- SQuAD or SQuAD2, e.g. -- plus thousands of labeled examples more representative of the target context.

To build a custom training set, we could use [Haystack's *free* annotation tool](https://www.deepset.ai/annotation-tool-for-labeling-datasets) (more info [here](https://www.deepset.ai/blog/labeling-data-with-haystack-annotation-tool), [here](https://docs.haystack.deepset.ai/docs/annotation), and [here](https://annotate.deepset.ai/index.html)). Other annotation tools, such as spaCy's [Prodigy](https://prodi.gy/), are costly. Thank you, Deepset!

![Deepset logo](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png)

## Push the envelope

AI is moving fast. Less than a year ago (May 2022), O'Reilly released the encore edition of *NLP with HF Transformers*, which presented Haystack's cutting-edge RAGenerator for QA. Today, the RAGenerator is deprecated in favor of *Agent* and *PromptNode* classes, integrating the latest LLMs and generative AI.

Pair an LLM-based [Agent](https://haystack.deepset.ai/tutorials/23_answering_multihop_questions_with_agents) with an extractive QA pipeline for iterative, a.k.a. multi-hop output. Or roll the dice with [generative QA](https://haystack.deepset.ai/tutorials/22_pipeline_with_promptnode). OpenAI models are available only with a paid API key. Google's flan-t5-large model is available for free. Thank you, Google!