# Compare different embedding methods.
import os
import hashlib
import email
import email.policy
import tqdm
import time
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # for testing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np
from itertools import islice
from transformers import T5Tokenizer
import torch
import dbm


CLUSTER_COUNT = 8
EMAIL_DATASET_COUNT = 200
CUDA_SUPPORT = torch.cuda.is_available()
print("CUDA available:", CUDA_SUPPORT)

T5_TOKENIZER = T5Tokenizer.from_pretrained("t5-large")
T5_EMBEDDING_CTX_LENGTH = 512
# %%

_cache_dbm = dbm.open('cache.dbm', 'c')
# %%


def list_disk_cache(namespace):
    """Function decorator to cache function results to disk. Only for list items."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = hashlib.md5(str(args).encode() +
                              str(kwargs).encode()).hexdigest()
            key = namespace + ':' + key
            if key in _cache_dbm:
                return [float(x) for x in str(_cache_dbm[key])[3:-2].split(', ')]
            result = func(*args, **kwargs)
            # Don't be a meanie, I can only do lists!
            assert isinstance(result, list)
            _cache_dbm[key] = str(result)
            return result
        return wrapper
    return decorator


# Helper functions to lazy load various models.
_t5_model = None


def get_t5_model():
    global _t5_model
    if _t5_model is None:
        from transformers import T5Model
        print("Loading T5 model...")
        model_name = "t5-large"
        tokenizer = T5_TOKENIZER
        if CUDA_SUPPORT:
            model = T5Model.from_pretrained(model_name).cuda()
        else:
            model = T5Model.from_pretrained(model_name)

        _t5_model = (tokenizer, model)
    return _t5_model

t5 = get_t5_model()
_, m = t5


_st_model = None


def get_sentence_tranformers(model):
    global _st_model
    if _st_model is None:
        print("Loading SentenceTransformers model %s..." % model)
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model)
    return _st_model


def t5_encode(text):
    tokens = T5_TOKENIZER.encode(
        text, return_tensors="pt", max_length=512, truncation=True)
    return tokens.cuda() if CUDA_SUPPORT else tokens

# Helper functions to chunk larger inputs into smaller ones.

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def chunked_tokens(text, encoder_fn, chunk_length):
    tokens = encoder_fn(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def chunked_text(text, chunk_length, tokens_per_word=2.5):
    words = text.split(' ')
    chunks_iterator = batched(words, int(chunk_length / tokens_per_word))
    # when the we have a chunk of words, we join them back into a string
    yield from map(lambda chunk: ' '.join(chunk), chunks_iterator)


def get_long_embedding(text, embedding_fn, max_tokens=None, encoder_fn=None, average=True):
    assert max_tokens is not None
    assert encoder_fn is not None
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoder_fn=encoder_fn, chunk_length=max_tokens):
        chunk_embeddings.append(embedding_fn(chunk))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(
            chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / \
            np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

# Method 1: Get embeddings using T5 directly. # TODO: max pooling voodoo.

def get_embedding_t5(text):
    tokenizer, model = get_t5_model()
    tokens = t5_encode(text)
    attn = tokens != tokenizer.pad_token_id
    output = model.encoder(
        input_ids=tokens, attention_mask=attn, return_dict=True)
    # Compute the mean of the last hidden state over the non-padded tokens. I think this is what they did in that paper, but I'm not sure...
    embedding = (output.last_hidden_state * attn.unsqueeze(-1)
                 ).sum(dim=-2) / attn.sum(dim=-1)
    return embedding.detach().cpu().numpy()[0]

# Method 2: Use SentenceTransformers.

def get_embedding_st(text, engine):
    model = get_sentence_tranformers(engine)
    if random.random() < 0.01:
        tokens = model.tokenize(text)['input_ids']
        sample_text = text[:100].replace('\n', ' ')
        print(
            f"sample: len={len(text)}, num_tokens={len(tokens)}, max_len={model.max_seq_length}, text={sample_text}")

    return model.encode([text])[0]


# Get embeddings. If "long_mode" is True, then we will chunk the input into smaller pieces and average the embeddings.

def get_embeddings(text, engine, long_mode=False):
    max_tokens = None
    encoder_fn = None
    if engine == "saved":
        return np.load("01-embeddings.npy")

    if not long_mode:
        if engine == "t5":
            return get_embedding_t5(text)
        elif engine.startswith("sentence-transformers/"):
            return get_embedding_st(text, engine)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    else:
        if engine == "t5":
            fn = get_embedding_t5
            max_tokens = T5_EMBEDDING_CTX_LENGTH
            encoder_fn = get_long_embedding(
                text, fn, max_tokens=max_tokens, encoder_fn=encoder_fn)
        elif engine.startswith("sentence-transformers/"):
            raise NotImplementedError(
                "Long mode not implemented for SentenceTransformers")
        else:
            raise ValueError(f"Unknown engine: {engine}")

def download_dataset():
    dataset_link = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
    if not os.path.exists('data/enron_mail_20150507.tar.gz'):
        print("Downloading dataset...")
        os.system("mkdir -p data")
        os.system("wget -P data/ " + dataset_link)
    else:
        print("Dataset already downloaded!")
    if not os.path.exists("data/maildir"):
        print("Extracting dataset...")
        os.system("tar -xzf data/enron_mail_20150507.tar.gz -C data/")
        print("Complete")
    else:
        print("Dataset already extracted!")
download_dataset()

def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        files = [os.path.join(root, name) for name in files]
        all_files.extend(files)
    return all_files


def get_emails(count=EMAIL_DATASET_COUNT):
    emails = []
    email_paths = get_all_files("data/maildir")
    email_paths = email_paths[::len(email_paths)//count]
    for file_name in email_paths:
        with open(file_name, "rb") as fp:
            try:
                msg = email.message_from_binary_file(
                    fp, policy=email.policy.default)
                emails.append(msg)
            except:
                pass
    return emails



def plot_plotly(embeddings_2d, labels, file_name: str, save: bool = False):
    df = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})
    fig = px.scatter(df, x="x", y="y", color="label")
    fig.show()
    # save the image
    if save:
        fig.write_image(file_name, width=1920, height=1080)
    else:
        pass


def run_embedding_test(engine):
    download_dataset()
    print("Getting emails...")
    emails = get_emails()
    # Concat all email IDs and print a hash
    embeddings = []
    print("Getting embeddings...")
    for msg in tqdm.tqdm(emails):
        subject = msg["subject"] or ""
        body = msg.get_body(preferencelist=("plain",))
        body = body.get_content() if body else ""
        if not body:
            continue
        # TODO: are separator tokens good or bad here?
        text = subject + "\n" + body
        # TODO: text = re.sub(r'\s+', ' ', text) # Is this a good idea? Are newlines actually good for embedding performance? Also should probably test.
        embeddings.append(get_embeddings(text, engine))
    embeddings = np.array(embeddings)
    print("Clustering...")
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    # Use t-SNE to reduce the dimensionality and visualize the clusters
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Get the labels for each cluster
    print("Getting labels...")
    email_ids = [msg["message-id"] for msg in emails]
    hashbit = hashlib.sha256("".join(email_ids).encode()).hexdigest()[-5:]
    engine_filename = engine.replace("/", "-")
    file_name = f'{hashbit}-{engine_filename}-cluster{CLUSTER_COUNT}-email{EMAIL_DATASET_COUNT}'
    np.save(file_name + '-embeddings.npy', embeddings)
    plot_plotly(embeddings_2d, labels, file_name + '.png')
    return labels, embeddings, embeddings_2d


def cluster(documents: list[str], engine='t5', run_name="test", k=10, embeddings=None):

    if embeddings is None:
        embeddings = []
        for doc in tqdm.tqdm(documents):
                embeddings.append(get_embeddings(doc, engine))
        embeddings = np.array(embeddings)

    print("Clustering...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")

    labels = kmeans.fit_predict(embeddings)

    # Use t-SNE to reduce the dimensionality and visualize the clusters
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    engine_filename = engine.replace("/", "-")
    file_name = f'{run_name}-{engine_filename}-cluster{k}'

    np.save(file_name + '-embeddings.npy', embeddings)
    return embeddings, labels

# %%

def main():
    start_time = time.time()
# sentence-transformers/all-mpnet-base-v2, sentence-transformers/gtr-t5-large (which should be T5), t5
    labels, embeddings, embeddings_2d = run_embedding_test('t5')

    print("Time taken: ", time.time() - start_time)
    return labels, embeddings, embeddings_2d

if __name__ == "__main__":
    labels, embeddings, embeddings_2d = main()