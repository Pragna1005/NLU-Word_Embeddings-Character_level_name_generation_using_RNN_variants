# NLU: Word Embeddings & Character-Level Name Generation using RNN Variants

---

## QUICK STEPS TO RUN

### Project 1: LEARNING WORD EMBEDDINGS FROM IIT JODHPUR DATA

**Step 1:** Go to the folder
```bash
cd LEARNING_WORD_EMBEDDINGS_FROM_IIT_JODHPUR_DATA
```

**Step 2:** Install dependencies
```bash
pip install numpy matplotlib scikit-learn wordcloud
```

**Step 3:** Run the code
```bash
python source_code.py
```
**Step 4:** Get results

---

### Project 2: CHARACTER-LEVEL NAME GENERATION USING RNN VARIANTS

**Step 1:** Go to the folder
```bash
cd CHARACTER_LEVEL_NAME_GENERATION
```

**Step 2:** Install dependencies
```bash
pip install numpy tensorflow matplotlib
```

**Step 3:** Train & generate names
```bash
python source_code.py
```

**Step 4:** Evaluate models
```bash
python evaluation_script.py
```

**Step 5:** Get results


---

## ✅ System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB
- **All Platforms**: macOS, Linux, Windows

Check your Python version:
```bash
python --version
```

---

## Project Details

### Project 1: Learning Word Embeddings from IIT Jodhpur Data

**What does it do?**
- Learns word meanings using Skip-gram and CBOW models
- Creates beautiful visualizations of word relationships
- Analyzes text patterns in the corpus

**Files you need:**
- `source_code.py` - Main script
- `corpus.txt` - Input text data

**What you get:**
- Word embedding visualizations (PCA & t-SNE)
- Analysis plots and statistics
- Understanding of how words relate to each other

---

### Project 2: Character-Level Name Generation using RNN Variants

**What does it do?**
- Generates Indian names using 3 different AI models:
  1. Vanilla RNN
  2. BLSTM (Bidirectional LSTM)
  3. RNN with Attention
- Evaluates which model generates the best names

**Files you need:**
- `source_code.py` - Training & generation script
- `evaluation_script.py` - Model comparison script
- `TrainingNames.txt` - Training data

**What you get:**
- 3000+ generated Indian names
- Comparison charts showing model performance
- Metrics: novelty, diversity, name patterns

---

## Detailed Guides

### First Time Setup (All Projects)

**Option A: Using Virtual Environment** (Recommended)
```bash
# Create environment
python -m venv my_env

# Activate it
source my_env/bin/activate  # macOS/Linux
# OR
my_env\Scripts\activate  # Windows

# Then install packages as shown in quick steps above
```

**Option B: Direct Installation**
```bash
# Just install directly (skip virtual environment)
pip install numpy matplotlib scikit-learn wordcloud tensorflow
```

---

### GPU Acceleration (Optional - 5-10x Faster)

**For NVIDIA GPU:**
```bash
pip install tensorflow-gpu
```

**For Mac with Apple Silicon:**
```bash
pip install tensorflow-macos tensorflow-metal
```

**Check if GPU is working:**
```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📂 Folder Structure

```
Repository/
├── README.md (this file)
├── LEARNING_WORD_EMBEDDINGS_FROM_IIT_JODHPUR_DATA/
│   ├── source_code.py
│   ├── corpus.txt
│   ├── requirements.txt
│   └── outputs/ (created when you run)
│
└── CHARACTER_LEVEL_NAME_GENERATION/
    ├── source_code.py
    ├── evaluation_script.py
    ├── TrainingNames.txt
    ├── requirements.txt
    └── generated_name_samples/
```
---