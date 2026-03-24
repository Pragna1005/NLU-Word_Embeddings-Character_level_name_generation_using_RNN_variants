# -*- coding: utf-8 -*-

!pip install wordcloud

import re, os, time, pickle, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs('outputs', exist_ok=True)

"""## TASK-1: DATASET PREPARATION"""

#Load text file
with open("corpus.txt", "r", encoding="utf-8") as f:
    raw_corpus = f.read()
print(f'   Characters : {len(raw_corpus):,}')
print(f'   Lines      : {raw_corpus.count(chr(10)):,}')

"""Preprocessing"""

#common words to ignore during analysis
STOP_WORDS = { #these carry little meaning and would inflate token counts without adding value
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','up','about','into','is','are','was',
    'were','be','been','being','have','has','had','do','does','did','will',
    'would','could','should','may','might','shall','can','this','that',
    'these','those','it','its','we','our','us','you','your','he','she',
    'his','her','they','their','them','what','which','who','when','where',
    'each','every','both','few','more','other','some','such',
    'than','then','so','if','as','not','no','nor','just','also','etc',
    'i','my','am','ii','iii','iv'
}

#strip common web and PDF clutter before we start tokenizing.
def remove_boilerplate(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text) #remove urls
    text = re.sub(r'\S+@\S+', ' ', text) #remove email addresses
    text = re.sub(r'\[Back to index\]', ' ', text, flags=re.I) #remove nav anchors
    text = re.sub(r'Copyright.*?Reserved.*', ' ', text, flags=re.I|re.S) #remove copyright lines
    text = re.sub(r'Last Updated.*', ' ', text, flags=re.I) #remove last updated stamps
    text = re.sub(r'IIT Jodhpur (Facebook|Twitter|Instagram|LinkedIn|Youtube).*', ' ', text, flags=re.I) #remove social media footers
    return text

#return list of clean tokens from one text string
def tokenise(text: str) -> list:
    text = remove_boilerplate(text)
    text = re.sub(r'\b\d+(\.\d+)?\b', ' ', text) #drop standalone numbers
    text = re.sub(r'[-=*_#|/\\]{2,}', ' ', text) #remove repeated symbols
    text = re.sub(r"[^a-zA-Z\s']", ' ', text) #remove non-alpha
    text = text.lower()
    text = re.sub(r"\b'\b|\s'|'\s", ' ', text) #remove orphan apostrophes
    text = re.sub(r'\s+', ' ', text).strip() #collapse multiple spaces into one
    meaningful_tokens = [
        word for word in text.split()
        if len(word) >= 2  # ignore single-letter tokens
        and word.isalpha() # ignore tokens with leftover non-letter characters
        and word not in STOP_WORDS # ignore filler words
    ]
    return meaningful_tokens


#turn a raw text string into a structured corpus for analysis
def build_corpus(raw: str):
    #Split on end-of-sentence punctuation followed by whitespace, or double newlines
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', raw)
    sentences = []
    all_tokens = []
    for part in parts:
        tokens = tokenise(part)
        if len(tokens) >= 3: #skip parts that are too short to be meaningful
            sentences.append(tokens)
            all_tokens.extend(tokens)
    vocab_counts = Counter(all_tokens)
    return sentences, all_tokens, vocab_counts


sentences, all_tokens, vocab_counts = build_corpus(raw_corpus)

#Save clean corpus
with open('outputs/clean_corpus.txt', 'w', encoding='utf-8') as f:
    for sent in sentences:
        f.write(' '.join(sent) + '\n')

print('  TASK 1 — DATASET STATISTICS')
print('-'*50)
print(f'  Total sentences   : {len(sentences):,}')
print(f'  Total tokens      : {len(all_tokens):,}')
print(f'  Vocabulary (raw)  : {len(vocab_counts):,} unique words')
print(f'\n  Top-20 most frequent words:')
for rank, (w, c) in enumerate(vocab_counts.most_common(20), 1):
    print(f'    {rank:>2}. {w:<25} {c}')

#wordcloud visualization

#join all tokens back into a single string
corpus_as_text = ' '.join(all_tokens)

#configure and generate the word cloud
word_cloud = WordCloud(
    width=1600, height=800,
    background_color='#0D1117', #dark background for visual contrast
    colormap='Set2',
    max_words=100,
    prefer_horizontal=0.8, #80% of words will be horizontal, 20% vertical
    collocations=False,
    min_font_size=9
).generate(corpus_as_text)

#render and save
fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor('#0D1117') #matches figure background to word cloud background

ax.imshow(word_cloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud — IIT Jodhpur',
             color='white',
             fontsize=16,
             fontweight='bold',
             pad=14)
plt.tight_layout()
plt.savefig('outputs/wordcloud.png',
            dpi=150,
            bbox_inches='tight',
            facecolor='#0D1117')
plt.show()

"""## Task 2 : Model Training"""

class Vocabulary:
    #builds a word to index lookup from a list of tokenized sentences, filtering out rare words below a minimum frequency threshold
    #also pre-computes a large negative sampling table using the unigram^0.75 distribution from Word2Vec, this smooths out the frequency imbalance so very common words aren't sampled too often
    def __init__(self, sentences: list, min_count: int = 2):
      #count every word across all sentences
        word_counts = Counter(w for s in sentences for w in s)
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}
        for idx, (word, freq) in enumerate(
                [(w,c) for w,c in word_counts.most_common() if c >= min_count]):
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
            self.word_freq[word]= freq
        self.size = len(self.word2idx)

        print(f'  Vocabulary (min_count={min_count}): {self.size:,} words')

        raw_frequencies  = np.array(
            [self.word_freq[self.idx2word[i]] for i in range(self.size)],
            dtype=np.float64
        )
        smoothed         = raw_frequencies ** 0.75
        sample_probs     = smoothed / smoothed.sum()

        self._neg_table  = np.random.choice(self.size, size=5_000_000, p=sample_probs)
        self._neg_ptr    = 0 #pointer that advances each time we draw negatives

    def get_negatives(self, k: int, exclude: set) -> list:
        out = []
        while len(out) < k:
            idx = int(self._neg_table[self._neg_ptr % len(self._neg_table)])
            self._neg_ptr += 1
            if idx not in exclude: out.append(idx)
        return out

    def encode(self, sentence: list) -> list:
        return [self.word2idx[w] for w in sentence if w in self.word2idx]


vocab = Vocabulary(sentences, min_count=2)

# Numerically stable sigmoid, uses alternate form for negative x to avoid overflow
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class Word2Vec:
    # Supports Skip-gram and CBOW, both trained with negative sampling.
    # W_in  = input embeddings (what we use at inference)
    # W_out = output embeddings (training only)

    def __init__(self, vocab: Vocabulary, embed_dim: int = 100, model_type: str = 'skipgram'):
        self.vocab      = vocab
        self.embed_dim  = embed_dim
        self.model_type = model_type

        V = vocab.size
        self.W_in  = (np.random.randn(V, embed_dim) * 0.01).astype(np.float32)
        self.W_out = (np.random.randn(V, embed_dim) * 0.01).astype(np.float32)


    #Negative sampling loss + gradients for one (center, context) pair

    def _negative_sampling_step(self, center_vector, pos_idx: int, neg_idxs: list):
        positive_vec  = self.W_out[pos_idx]
        negative_vecs = self.W_out[neg_idxs]   # shape (K, D)

        sig_positive  = sigmoid(center_vector @ positive_vec)
        sig_negatives = sigmoid(-negative_vecs @ center_vector)

        loss = (
            -np.log(sig_positive + 1e-10)
            - np.sum(np.log(sig_negatives + 1e-10))
        )

        grad_center = (
            (sig_positive - 1) * positive_vec
            + np.sum((1 - sig_negatives)[:, None] * negative_vecs, axis=0)
        )
        grad_positive  = (sig_positive - 1) * center_vector
        grad_negatives = [
            (noise_idx, (1 - sig_negatives[k]) * center_vector)
            for k, noise_idx in enumerate(neg_idxs)
        ]

        return loss, grad_center, grad_positive, grad_negatives


    #Skip-gram: given a center word, predict each context word

    def train_skipgram(self, sentences: list, window: int, num_neg: int,
                       lr: float, epochs: int) -> list:
        print(f'  [Skip-gram]  embed_dim={self.embed_dim}  window={window}  negatives={num_neg}')

        encoded_sentences = [
            encoded for encoded in
            (self.vocab.encode(sentence) for sentence in sentences)
            if len(encoded) >= 2
        ]

        loss_history = []

        for epoch in range(1, epochs + 1):
            total_loss, num_pairs, t_start = 0.0, 0, time.time()

            for sent in encoded_sentences:
                sent_len = len(sent)

                for center_pos, center_idx in enumerate(sent):
                    window_start = max(0, center_pos - window)
                    window_end   = min(sent_len, center_pos + window + 1)

                    for context_pos in range(window_start, window_end):
                        if context_pos == center_pos:
                            continue

                        pos_idx  = sent[context_pos]
                        neg_idxs = self.vocab.get_negatives(num_neg, exclude={center_idx, pos_idx})

                        loss, grad_center, grad_pos, grad_negs = self._negative_sampling_step(
                            self.W_in[center_idx], pos_idx, neg_idxs
                        )

                        self.W_in[center_idx] -= lr * grad_center
                        self.W_out[pos_idx]   -= lr * grad_pos
                        for noise_idx, grad in grad_negs:
                            self.W_out[noise_idx] -= lr * grad

                        total_loss += loss
                        num_pairs  += 1

            avg_loss = total_loss / max(num_pairs, 1)
            loss_history.append(avg_loss)
            print(f'    Epoch {epoch}/{epochs}  |  loss={avg_loss:.4f}'
                  f'  |  pairs={num_pairs:,}  |  {time.time()-t_start:.1f}s')

        return loss_history


    #CBOW: given averaged context embeddings, predict the center word

    def train_cbow(self, sentences: list, window: int, num_neg: int,
                   lr: float, epochs: int) -> list:
        print(f'  [CBOW]       embed_dim={self.embed_dim}  window={window}  negatives={num_neg}')

        encoded_sentences = [
            encoded for encoded in
            (self.vocab.encode(sentence) for sentence in sentences)
            if len(encoded) >= 2
        ]

        loss_history = []

        for epoch in range(1, epochs + 1):
            total_loss, num_pairs, t_start = 0.0, 0, time.time()

            for sent in encoded_sentences:
                sent_len = len(sent)

                for target_pos, target_idx in enumerate(sent):
                    context_indices = [
                        sent[j]
                        for j in range(max(0, target_pos - window),
                                       min(sent_len, target_pos + window + 1))
                        if j != target_pos
                    ]
                    if not context_indices:
                        continue

                    context_mean = self.W_in[context_indices].mean(axis=0)
                    neg_idxs     = self.vocab.get_negatives(
                        num_neg, exclude=set(context_indices) | {target_idx}
                    )

                    loss, grad_mean, grad_target, grad_negs = self._negative_sampling_step(
                        context_mean, target_idx, neg_idxs
                    )

                    # gradient is split equally across all context words
                    grad_per_context = grad_mean / len(context_indices)
                    for ctx_idx in context_indices:
                        self.W_in[ctx_idx] -= lr * grad_per_context

                    self.W_out[target_idx] -= lr * grad_target
                    for noise_idx, grad in grad_negs:
                        self.W_out[noise_idx] -= lr * grad

                    total_loss += loss
                    num_pairs  += 1

            avg_loss = total_loss / max(num_pairs, 1)
            loss_history.append(avg_loss)
            print(f'    Epoch {epoch}/{epochs}  |  loss={avg_loss:.4f}'
                  f'  |  pairs={num_pairs:,}  |  {time.time()-t_start:.1f}s')

        return loss_history


    #Inference

    # Returns a copy so callers can't accidentally mutate W_in
    def get_embedding(self, word: str) -> np.ndarray | None:
        idx = self.vocab.word2idx.get(word)
        return self.W_in[idx].copy() if idx is not None else None

    def most_similar(self, word: str, topn: int = 5) -> list[tuple]:
        query_vec = self.get_embedding(word)
        if query_vec is None:
            return []

        norms         = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        normed_matrix = self.W_in / norms
        normed_query  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        similarities  = normed_matrix @ normed_query

        similarities[self.vocab.word2idx[word]] = -1.0  # exclude the query word itself
        top_indices = np.argsort(similarities)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(similarities[i])) for i in top_indices]

    # Solves "a is to b as c is to ?" via vector arithmetic: embed(b) - embed(a) + embed(c)
    def analogy(self, a: str, b: str, c: str, topn: int = 5) -> list[tuple]:
        vec_a, vec_b, vec_c = self.get_embedding(a), self.get_embedding(b), self.get_embedding(c)

        oov_words = [word for word, vec in zip([a, b, c], [vec_a, vec_b, vec_c]) if vec is None]
        if oov_words:
            print(f'!Out-of-vocabulary: {oov_words}')
            return []

        target_vec  = vec_b - vec_a + vec_c
        target_vec /= (np.linalg.norm(target_vec) + 1e-10)

        norms         = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        normed_matrix = self.W_in / norms
        similarities  = normed_matrix @ target_vec

        for word in [a, b, c]:
            similarities[self.vocab.word2idx[word]] = -1.0  # exclude input words

        top_indices = np.argsort(similarities)[::-1][:topn]
        return [(self.vocab.idx2word[i], float(similarities[i])) for i in top_indices]

    def save(self, path: str):
        payload = {
            'W_in'       : self.W_in,
            'W_out'      : self.W_out,
            'embed_dim'  : self.embed_dim,
            'model_type' : self.model_type,
            'word2idx'   : self.vocab.word2idx,
            'idx2word'   : self.vocab.idx2word,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)
        print(f'Model saved → {path}')

#experiment configurations
CONFIGS = [
    (50,  2,  5),
    (100, 3,  5),
    (100, 5, 10),
    (200, 3, 10),
]
EPOCHS = 10
LR     = 0.025

sg_histories = []
cb_histories = []
best_sg      = None
best_cbow    = None

print('  TRAINING  —  Skip-gram & CBOW')
print('-' * 60)

for dim, win, neg in CONFIGS:
    label = f'd={dim} w={win} k={neg}'
    print(f'\n── {label} ──')

    # train skip-gram
    sg   = Word2Vec(vocab, embed_dim=dim, model_type='skipgram')
    h_sg = sg.train_skipgram(sentences, window=win, num_neg=neg, lr=LR, epochs=EPOCHS)
    sg_histories.append((f'SG {label}', h_sg))

    # train CBOW
    cb   = Word2Vec(vocab, embed_dim=dim, model_type='cbow')
    h_cb = cb.train_cbow(sentences, window=win, num_neg=neg, lr=LR, epochs=EPOCHS)
    cb_histories.append((f'CB {label}', h_cb))

    if dim == 100 and win == 5:
        best_sg, best_cbow = sg, cb
        print('Best config — selected for downstream tasks')

# fallback: if the preferred config wasn't reached, use whatever ran last
if best_sg is None:
    best_sg, best_cbow = sg, cb

best_sg.save('outputs/skipgram_best.pkl')
best_cbow.save('outputs/cbow_best.pkl')
print('\nTraining completed')

PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Training Loss — All Hyperparameter Configurations',
             color='white', fontsize=13, fontweight='bold')

for ax, (title, histories) in zip(axes, [('Skip-gram', sg_histories), ('CBOW', cb_histories)]):
    ax.set_facecolor('#161B22')

    for spine in ax.spines.values():
        spine.set_color('#30363D')

    for (label, loss_curve), color in zip(histories, PALETTE):
        ax.plot(
            range(1, len(loss_curve) + 1), loss_curve,
            marker='o', color=color,
            linewidth=2.2, markersize=4,
            label=label
        )

    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', color='#8B949E')
    ax.set_ylabel('Average Loss', color='#8B949E')
    ax.tick_params(colors='#8B949E')
    ax.legend(fontsize=7.5, facecolor='#21262D', edgecolor='#30363D', labelcolor='white')
    ax.grid(True, color='#21262D', linewidth=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('outputs/training_loss.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.show()

"""## Task 3 : Semantic Analysis"""

# words we plan to query later — checking they actually made it into the vocabulary
CANDIDATES = [
    'research',    'student',     'students',    'phd',
    'exam',        'examination', 'examinations','programme',
    'academic',    'faculty',     'thesis',      'semester',
    'course',      'engineering', 'department',  'learning',
]

print('Vocabulary presence check (assignment query words):')
print(f'  {"Word":<22} {"In Vocab?"}')
print('  ' + '-' * 35)

for word in CANDIDATES:
    status = 'yes' if word in vocab.word2idx else 'OOV'
    freq   = vocab.word_freq.get(word, 0)
    print(f'  {word:<22} {status}   (freq={freq})')

# picks the first word from a list of alternates that actually exists in our vocabulary
# useful because some forms (e.g. 'exam' vs 'examination') may have been filtered out
def first_in_vocab(candidates: list) -> str | None:
    for word in candidates:
        if word in vocab.word2idx:
            return word
    return None  # none of the alternates survived the min_count filter


# for each concept we care about, we try a few surface forms in order of preference
# the first one found in the vocabulary gets used as the query word
QUERY_WORDS = [
    first_in_vocab(['research']),
    first_in_vocab(['student', 'students']),
    first_in_vocab(['phd']),
    first_in_vocab(['examination', 'examinations', 'exam']),
    first_in_vocab(['programme', 'program', 'academic']),
]

# drop any concepts where none of the alternates made it into the vocabulary
QUERY_WORDS = [word for word in QUERY_WORDS if word is not None]

print('Query words selected from corpus vocabulary:')
print(' ', QUERY_WORDS)

# nn_results stores the full neighbor list for every (model, word) pair
# so we can reuse them later for visualizations without re-querying
nn_results = {}

print('\n' + '-' * 72)
print('  TOP-5 NEAREST NEIGHBORS   (Cosine Similarity)')
print('-' * 72)

for model_name, model in [('Skip-gram', best_sg), ('CBOW', best_cbow)]:
    print(f'\n── {model_name} ──')
    print(f'  {"Word":<18} | Neighbor 1          | Neighbor 2          | Neighbor 3')
    print('  ' + '-' * 75)

    nn_results[model_name] = {}

    for word in QUERY_WORDS:
        neighbors = model.most_similar(word, topn=5)
        nn_results[model_name][word] = neighbors  # save all 5 for later use

        # cosine similarity of 1.0 = identical direction, 0.0 = unrelated
        # so a score of ~0.7+ here generally means a genuinely close semantic neighbor
        top3 = [f'{w}({score:.3f})' for w, score in neighbors[:3]]
        top3 += ['—'] * (3 - len(top3))  # pad to 3 columns if fewer neighbors returned

        print(f'  {word:<18} | {top3[0]:<20} | {top3[1]:<20} | {top3[2]}')

#analogy experiments
def auto_analogy(model, candidates_a, candidates_b, candidates_c, desc, topn=3):
    a = first_in_vocab(candidates_a)
    b = first_in_vocab(candidates_b)
    c = first_in_vocab(candidates_c)
    if None in (a,b,c):
        print(f'  {desc:<50} → OOV ({a},{b},{c})')
        return None
    result = model.analogy(a, b, c, topn=topn)
    ans = '  |  '.join(f'{w}({s:.3f})' for w,s in result) if result else 'N/A'
    print(f'  {desc:<50}')
    print(f'    ({a} : {b} :: {c} : ?)  →  {ans}')
    return result

ANALOGIES = [
    # description, [a candidates], [b candidates], [c candidates]
    ('UG:BTech :: PG:?  (program level analogy)',
     ['undergraduate','btech','ug'],
     ['programme','program'],
     ['postgraduate','mtech','phd']),

    ('student:learning :: faculty:?',
     ['student','students'],
     ['learning','academic'],
     ['faculty','professor']),

    ('research:phd :: teaching:?',
     ['research'],
     ['phd'],
     ['teaching','course','courses']),

    ('exam:semester :: thesis:?',
     ['examination','exam'],
     ['semester'],
     ['thesis']),

    ('engineering:department :: science:?',
     ['engineering'],
     ['department'],
     ['science','academic']),
]

analogy_results = {}
for mname, model in [('Skip-gram', best_sg), ('CBOW', best_cbow)]:
    print(f'\n{'-'*60}')
    print(f'  ANALOGIES : {mname}')
    print('-'*60)
    analogy_results[mname] = {}
    for desc, ca, cb, cc in ANALOGIES:
        r = auto_analogy(model, ca, cb, cc, desc)
        analogy_results[mname][desc] = r

"""## Task 4 : Visualization"""

fig, axes = plt.subplots(2, len(QUERY_WORDS), figsize=(5*len(QUERY_WORDS), 10))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Top-5 Nearest Neighbors — Cosine Similarity',
             color='white', fontsize=14, fontweight='bold')

for row, (mname, model) in enumerate([('Skip-gram',best_sg),('CBOW',best_cbow)]):
    for col, word in enumerate(QUERY_WORDS):
        ax = axes[row,col] if len(QUERY_WORDS)>1 else axes[row]
        ax.set_facecolor('#161B22')
        for sp in ax.spines.values(): sp.set_color('#30363D')
        nn = model.most_similar(word, topn=5)
        if not nn:
            ax.text(0.5,0.5,'OOV',ha='center',va='center',
                    color='white',transform=ax.transAxes); continue
        wds, sims = zip(*nn)
        sims  = [max(0.0,s) for s in sims]
        color = PALETTE[col%len(PALETTE)]
        bars  = ax.barh(range(len(wds)), sims, color=color,
                        alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(len(wds)))
        ax.set_yticklabels(wds, color='#C9D1D9', fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis='x', colors='#8B949E', labelsize=7)
        ax.set_title(f'"{word}"', color='white', fontsize=9, fontweight='bold')
        ax.grid(axis='x', color='#21262D', linewidth=0.5)
        for bar, sim in zip(bars, sims):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f'{sim:.3f}', va='center', color='#8B949E', fontsize=7)
        if col==0:
            ax.set_ylabel(mname, color='#C9D1D9', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('outputs/nearest_neighbors.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()

#cosine similarity heat map
# Auto-select heatmap words from vocabulary
HM_CANDIDATES = ['research','academic','engineering','department','faculty',
                 'phd','thesis','semester','course','examination',
                 'learning','student','students','programme','technology']
HM_WORDS = [w for w in HM_CANDIDATES if w in vocab.word2idx][:12]
print('Heatmap words:', HM_WORDS)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Pairwise Cosine Similarity Heatmap — Key Domain Words',
             color='white', fontsize=14, fontweight='bold')

for ax, (mname, model) in zip(axes,[('Skip-gram',best_sg),('CBOW',best_cbow)]):
    ax.set_facecolor('#161B22')
    present = [w for w in HM_WORDS if model.get_embedding(w) is not None]
    vecs    = np.array([model.get_embedding(w) for w in present])
    norms   = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    sim_m   = (vecs/norms) @ (vecs/norms).T
    im = ax.imshow(sim_m, cmap='RdYlGn', vmin=-0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha='right', color='#C9D1D9', fontsize=9)
    ax.set_yticklabels(present, color='#C9D1D9', fontsize=9)
    for i in range(len(present)):
        for j in range(len(present)):
            v = sim_m[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7.5,
                    color='black' if v>0.4 else 'white', fontweight='bold')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8B949E', fontsize=8)
    ax.set_title(mname, color='white', fontsize=12, fontweight='bold', pad=10)
    for sp in ax.spines.values(): sp.set_color('#30363D')

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig('outputs/cosine_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()

#pca and t-sne projections
#Word clusters for visualization
WORD_CLUSTERS = {
    'Academic Roles'  : ['student','students','faculty','professor','scholar','researcher','advisor'],
    'Degree Programs' : ['btech','mtech','phd','msc','mba','undergraduate','postgraduate','programme'],
    'Departments'     : ['computer','electrical','mechanical','chemical','bioscience','materials'],
    'Research'        : ['research','thesis','publication','paper','project','laboratory'],
    'Evaluation'      : ['examination','grade','cgpa','assignment','semester','credit','course'],
    'AI / CS'         : ['learning','neural','algorithm','data','science','technology','engineering'],
}

def collect_vecs(model, clusters):
    words, vecs, labels, valid = [], [], [], {}
    for ci,(cname,wlist) in enumerate(clusters.items()):
        found = []
        for w in wlist:
            v = model.get_embedding(w)
            if v is not None:
                words.append(w); vecs.append(v); labels.append(ci); found.append(w)
        if found: valid[cname] = (ci,found)
    return words, np.array(vecs,dtype=np.float32), labels, valid


viz_palette = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261','#A8DADC']

fig = plt.figure(figsize=(22, 17))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Word Embeddings — PCA & t-SNE Projections  |  IIT Jodhpur corpus.txt',
             color='white', fontsize=15, fontweight='bold', y=0.98)
gs = GridSpec(2,2,figure=fig,hspace=0.38,wspace=0.28,
              left=0.06,right=0.97,top=0.93,bottom=0.06)

viz_configs = [
    ('Skip-gram', best_sg,   'PCA'),
    ('Skip-gram', best_sg,   't-SNE'),
    ('CBOW',      best_cbow, 'PCA'),
    ('CBOW',      best_cbow, 't-SNE'),
]

for idx,(mname,model,method) in enumerate(viz_configs):
    r,c = divmod(idx,2)
    ax  = fig.add_subplot(gs[r,c])
    ax.set_facecolor('#161B22')
    for sp in ax.spines.values(): sp.set_color('#30363D')

    words, vecs, labels, valid = collect_vecs(model, WORD_CLUSTERS)
    if len(vecs)<4:
        ax.text(0.5,0.5,'Insufficient vocab',ha='center',va='center',
                color='white',transform=ax.transAxes); continue

    if method=='PCA':
        red   = PCA(n_components=2, random_state=42)
        coord = red.fit_transform(vecs)
        var   = red.explained_variance_ratio_
        sub   = f'PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%'
    else:
        perp  = min(30, max(5, len(vecs)//3))
        red   = TSNE(n_components=2, perplexity=perp, random_state=42,
                     max_iter=1000, learning_rate='auto', init='pca')
        coord = red.fit_transform(vecs)
        sub   = f'perplexity={perp}'

    for cname,(ci,_) in valid.items():
        mask = [i for i,l in enumerate(labels) if l==ci]
        col  = viz_palette[ci%len(viz_palette)]
        ax.scatter(coord[mask,0],coord[mask,1],c=col,s=95,alpha=0.85,
                   edgecolors='white',linewidths=0.5,zorder=3)

    for i,(w,pt) in enumerate(zip(words,coord)):
        col = viz_palette[labels[i]%len(viz_palette)]
        ax.annotate(w,pt,xytext=(4,4),textcoords='offset points',
                    fontsize=7.5,color=col,fontweight='semibold',alpha=0.92)

    patches = [mpatches.Patch(color=viz_palette[ci%len(viz_palette)],label=cn)
               for cn,(ci,_) in valid.items()]
    ax.legend(handles=patches,loc='upper left',fontsize=7,
              facecolor='#21262D',edgecolor='#30363D',
              labelcolor='white',framealpha=0.85)
    ax.set_title(f'{mname}  —  {method}  ({sub})',
                 color='white',fontsize=11,fontweight='bold',pad=8)
    ax.tick_params(colors='#8B949E',labelsize=7)
    ax.set_xlabel('Component 1',color='#8B949E',fontsize=8)
    ax.set_ylabel('Component 2',color='#8B949E',fontsize=8)
    ax.grid(True,color='#21262D',linewidth=0.6,alpha=0.6)

plt.savefig('outputs/pca_tsne_projections.png',dpi=150,bbox_inches='tight',
            facecolor='#0D1117')
plt.show()
