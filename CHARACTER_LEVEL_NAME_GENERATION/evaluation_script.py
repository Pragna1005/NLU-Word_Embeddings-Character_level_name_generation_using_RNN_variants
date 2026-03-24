import math
import os
from collections import Counter


# -------- metrics --------

def compute_novelty_rate(generated_names, training_names_lower):
    novel = sum(1 for n in generated_names if n.lower() not in training_names_lower)
    return 100.0 * novel / len(generated_names) if generated_names else 0.0


def compute_diversity(generated_names):
    unique = len(set(n.lower() for n in generated_names))
    return 100.0 * unique / len(generated_names) if generated_names else 0.0


def compute_avg_length(generated_names):
    lengths = [len(n) for n in generated_names]
    return sum(lengths) / len(lengths) if lengths else 0.0


def compute_kl_divergence(generated_names, training_names):
    all_chars = sorted(set("".join(n.lower() for n in training_names)))

    train_text = "".join(n.lower() for n in training_names)
    gen_text = "".join(n.lower() for n in generated_names)

    train_counts = Counter(train_text)
    gen_counts = Counter(gen_text)

    train_total = sum(train_counts.values())
    gen_total = sum(gen_counts.values())

    if gen_total == 0:
        return float('inf')

    eps = 1e-8
    kl = 0.0

    for c in all_chars:
        p = train_counts.get(c, 0) / train_total
        q = gen_counts.get(c, 0) / gen_total
        kl += p * math.log((p + eps) / (q + eps))

    return kl


def compute_bigram_overlap(generated_names, training_names):

    def extract_bigrams(name_list):
        bigrams = set()
        for name in name_list:
            lo = name.lower()
            for i in range(len(lo) - 1):
                bigrams.add(lo[i:i+2])
        return bigrams

    train_bigrams = extract_bigrams(training_names)
    gen_bigrams = extract_bigrams(generated_names)

    if not gen_bigrams:
        return 0.0

    overlap = len(gen_bigrams & train_bigrams)
    return 100.0 * overlap / len(gen_bigrams)


def detect_failure_modes(generated_names):

    vowels = set("aeiou")
    all_letters = set("abcdefghijklmnopqrstuvwxyz")

    too_short = []
    too_long = []
    repetitive = []
    consonant_heavy = []

    for name in generated_names:
        lo = name.lower()

        if len(lo) < 3:
            too_short.append(name)

        if len(lo) > 15:
            too_long.append(name)

        for c in all_letters:
            if c * 3 in lo:
                repetitive.append(name)
                break

        max_consonants = 0
        current_run = 0

        for ch in lo:
            if ch in all_letters and ch not in vowels:
                current_run += 1
                if current_run > max_consonants:
                    max_consonants = current_run
            else:
                current_run = 0

        if max_consonants >= 4:
            consonant_heavy.append(name)

    return {
        "too_short": too_short,
        "too_long": too_long,
        "repetitive": repetitive,
        "consonant_heavy": consonant_heavy
    }


# -------- evaluation --------

def run_full_evaluation(generated_names, training_names, label="Model"):

    training_lower = set(n.lower() for n in training_names)

    novelty = compute_novelty_rate(generated_names, training_lower)
    diversity = compute_diversity(generated_names)
    avg_len = compute_avg_length(generated_names)
    kl_div = compute_kl_divergence(generated_names, training_names)
    bigram_ov = compute_bigram_overlap(generated_names, training_names)
    failures = detect_failure_modes(generated_names)

    lengths = [len(n) for n in generated_names]

    print("\n---", label, "---")
    print("Total:", len(generated_names))
    print("Unique:", len(set(n.lower() for n in generated_names)))

    print("\nMetrics")
    print("Novelty:", round(novelty, 1), "%")
    print("Diversity:", round(diversity, 1), "%")

    print("\nLengths")
    print("Avg:", round(avg_len, 1))
    print("Min:", min(lengths))
    print("Max:", max(lengths))

    print("\nDistribution")
    print("KL:", round(kl_div, 4))
    print("Bigram:", round(bigram_ov, 1), "%")

    print("\nIssues")
    print("Too short:", len(failures['too_short']), failures['too_short'][:5])
    print("Too long:", len(failures['too_long']), failures['too_long'][:5])
    print("Repetitive:", len(failures['repetitive']), failures['repetitive'][:5])
    print("Consonant clusters:", len(failures['consonant_heavy']), failures['consonant_heavy'][:5])

    novel_names = [n for n in generated_names if n.lower() not in training_lower]
    memorised = [n for n in generated_names if n.lower() in training_lower]

    print("\nSamples")
    print("New:", novel_names[:10])
    print("Seen:", memorised[:10])

    return {
        "novelty_rate": novelty,
        "diversity": diversity,
        "avg_length": avg_len,
        "kl_divergence": kl_div,
        "bigram_overlap": bigram_ov,
        "too_short": len(failures["too_short"]),
        "too_long": len(failures["too_long"]),
        "repetitive": len(failures["repetitive"]),
        "consonant_heavy": len(failures["consonant_heavy"]),
    }


# -------- helpers --------

def load_names(filepath):
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return []

    with open(filepath, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    return names


# -------- run (COLAB SAFE) --------

training_path = "TrainingNames.txt"

# choose one option
run_all = True
generated_path = None
# generated_path = "vanilla_rnn_names.txt"

training_names = load_names(training_path)
print("Loaded training:", len(training_names))


if run_all:
    model_files = [
        ("Vanilla RNN", "vanilla_rnn_names.txt"),
        ("BLSTM", "blstm_names.txt"),
        ("RNN + Attention", "attention_rnn_names.txt"),
    ]

    for label, filepath in model_files:
        if os.path.exists(filepath):
            gen_names = load_names(filepath)
            print("\nLoaded:", filepath, len(gen_names))
            run_full_evaluation(gen_names, training_names, label)
        else:
            print("Missing:", filepath)

elif generated_path:
    gen_names = load_names(generated_path)
    print("Loaded:", len(gen_names))

    label = os.path.basename(generated_path).replace("_names.txt", "").replace("_", " ").title()
    run_full_evaluation(gen_names, training_names, label)

else:
    print("Nothing selected")