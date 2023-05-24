import re
import torch
import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel


def show_size(builder):
    return pd.DataFrame(
        {'size in MB': [builder.info.splits['train'].num_bytes // 2**20,
                        builder.info.splits['test'].num_bytes // 2**20],
         'number of examples': [builder.info.splits['train'].num_examples, 
                                builder.info.splits['test'].num_examples]}, 
                                index=['train', 'test']
    )


def load_yahoo(split='train', num_shards=32, shard_index=0, streaming=False):

    # load training split, drop unimportant column
    ds = datasets.load_dataset("yahoo_answers_topics", split=split, streaming=streaming).shard(
        num_shards=num_shards, index=shard_index)
    ds = ds.remove_columns(['question_content'])

    # simplify column names
    ds = ds.rename_column('question_title', 'question')
    ds = ds.rename_column('best_answer', 'answer')
    ds = ds.rename_column('topic', 'label')

    # add column for topic names
    topics = ds.features['label'].names
    ds = ds.map(lambda x: {'topic': topics[x['label']]})

    ds.set_format('pandas')

    return ds


def q_start(item):
    w0 = str(item['question']).split()[0] if len(str(item['question']).split()) > 0 else ''
    item.update({'q_start': w0.lower()})
    return item


def plot_question_starters(ds):

    # within each topic, what are the most frequent question start words?
    q_start_freq = ds[:].groupby(['topic']).value_counts(['q_start'])

    q_df = pd.DataFrame()
    for topic in ds.unique('topic'):
        q_df[topic] = q_start_freq.loc[topic].index[:10]

    top_question_starts = set(q_df.values.flatten()) # <- union of words in 10x10 q_df above
    q_viz = q_start_freq.to_frame(name="count").query("q_start in @top_question_starts") # filter df to 15 words that capture all topics' top 10

    plt.figure(figsize=(6, 4))
    plt.title('Most common question starts, by topic')
    sns.heatmap(q_viz.reset_index().pivot(index='topic', columns='q_start', values='count'), 
                cmap='Blues', square=True)
    plt.show()


def clean_question(item: dict) -> dict:
    if len(item['question']) > 0:
        item['question'] = str(item['question'])        
        item['question'] = re.sub('<.{,10}>', ' ', item['question']) # remove some html tags
        item['question'] = item['question'].replace("'", '') # remove apostrophes
        item['question'] = re.sub('[^A-Za-z ]', ' ', item['question']) # if punctuation matters, use re.sub(f'[^{string.printable}]', ' ', text)
        item['question'] = re.sub(' {2,}', ' ', item['question']) # remove extra spaces
        item['question'] = item['question'].lower().strip().split()
    return item


def clean_answer(item: dict) -> dict:
    if len(item['answer']) > 0:
        item['answer'] = str(item['answer'])        
        item['answer'] = re.sub('<.{,10}>', ' ', item['answer']) # remove some html tags
        item['answer'] = item['answer'].replace("'", '') # remove apostrophes
        item['answer'] = re.sub('[^A-Za-z ]', ' ', item['answer'])
        item['answer'] = re.sub(' {2,}', ' ', item['answer']) # remove extra spaces
        item['answer'] = item['answer'].lower().strip().split()
    return item


def count_q_words(item):
    item.update({'q_word_count': len(item['question'])})
    return item


def count_ans_words(item):
    item.update({'ans_word_count': len(item['answer'])})
    return item


def word_counts(ds):
    ds.reset_format()
    ds = ds.map(clean_question)
    ds = ds.map(clean_answer)
    ds = ds.map(count_q_words)
    ds = ds.map(count_ans_words)
    ds = ds.filter(lambda x: x['q_word_count'] > 0 and x['ans_word_count'] > 0)
    ds.set_format('pandas')
    return ds


def tokenize(batch: dict)-> dict:
    """ Tokenize a batch of text """

    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2-distilled")

    return tokenizer(
        batch["question"], batch["answer"], padding='max_length', truncation=True, return_tensors='np')


def extract_hidden_states(batch)-> dict:
    """ get feature embeddings from headless model """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('deepset/roberta-base-squad2-distilled').to(device)
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2-distilled")

    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}

    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"feature_embeddings": last_hidden_state[:,0].cpu().numpy()}