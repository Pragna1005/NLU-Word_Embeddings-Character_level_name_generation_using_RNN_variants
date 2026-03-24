# -*- coding: utf-8 -*-

## Task 0 : DATASET


#Generate 1000 Indian names using LLMs
indian_names = [
    #Pan-Indian popular names
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Reyansh",
    "Ayaan","Krishna","Ishaan","Shaurya","Atharva","Advik","Pranav",
    "Advaith","Aarush","Kabir","Ritvik","Anirudh","Dhruv",
    "Saanvi","Aanya","Aadhya","Aarohi","Ananya","Pari","Anika",
    "Navya","Diya","Myra","Sara","Ira","Ahana","Kiara","Prisha",
    "Anvi","Avni","Riya","Zara","Nisha",

    #Traditional Hindu/Sanskrit-origin
    "Achyuth","Akshay","Amarnath","Anand","Arun","Ashok","Balaji",
    "Bharat","Chandra","Damodaran","Deepak","Devendra","Dinesh",
    "Ganesh","Girish","Gopal","Govind","Hari","Hemant","Jagdish",
    "Jayant","Kailash","Kamal","Keshav","Kishore","Lakshman",
    "Madhav","Mahesh","Manoj","Mohan","Mukesh","Naresh","Naveen",
    "Nikhil","Pankaj","Paresh","Prakash","Prem","Raghav","Rajesh",
    "Rakesh","Ramesh","Ravi","Sachin","Sanjay","Shankar","Shiva",
    "Sunil","Suresh","Tushar","Umesh","Varun","Vijay","Vinod",
    "Vishnu","Yash","Yogesh",

    #Traditional female names
    "Asha","Bhavana","Champa","Damini","Durga","Esha","Gauri",
    "Gita","Hema","Indira","Jaya","Kamala","Kanta","Kavita",
    "Lakshmi","Lalita","Lata","Madhu","Mala","Malini","Mangala",
    "Meena","Mira","Mohini","Nandini","Neelam","Nirmala","Padma",
    "Parvati","Pooja","Priya","Pushpa","Radha","Rajni","Rama",
    "Rekha","Rohini","Rukmini","Sangeeta","Sarita","Savitri",
    "Seema","Shakuntala","Shanti","Shobha","Sita","Sudha","Suma",
    "Sunita","Sushila","Swati","Tara","Uma","Usha","Vani",
    "Vasundhara","Vidya","Vimala",

    #South Indian
    "Aravind","Balasubramanian","Chandrasekhar","Dhanush","Ezhil",
    "Gowri","Hariharan","Iniyan","Jagan","Karthik","Kumaran",
    "Lingam","Murugan","Nandha","Oviya","Palani","Rajinikanth",
    "Selvam","Senthil","Subramani","Surya","Tamilselvi","Thirumal",
    "Vaithianathan","Velmurugan","Venkatesh","Vignesh","Yuvan",
    "Anitha","Bhuvaneshwari","Chitra","Deepa","Divya","Gayathri",
    "Gomathi","Janaki","Kanmani","Kavitha","Kalpana","Lalitha",
    "Mahalakshmi","Meenakshi","Mythili","Niranjana","Preethi",
    "Rajalakshmi","Revathi","Saranya","Sharmila","Sowmya",
    "Suganya","Thenmozhi","Vanitha","Vasanthi","Vidhya",

    #North Indian
    "Abhishek","Ajay","Amit","Ankur","Atul","Gaurav","Harsh",
    "Himanshu","Kapil","Kunal","Lalit","Manish","Neeraj","Nitin",
    "Pradeep","Rahul","Rajat","Rohit","Sagar","Sameer","Sandeep",
    "Saurabh","Shekhar","Sumit","Tarun","Vikas","Vikram","Vinay",
    "Alka","Archana","Babita","Chhaya","Deepti","Geeta","Hemlata",
    "Jyoti","Kiran","Komal","Kumari","Mamta","Neha","Nidhi",
    "Pallavi","Poonam","Preeti","Rachna","Rani","Reena","Renu",
    "Sapna","Saroj","Shweta","Smita","Sonal","Suman","Sunaina",
    "Sweta","Tanvi","Vandana",

    #Bengali
    "Abhijit","Amitabha","Anirban","Anurag","Arnab","Debashish",
    "Dipankar","Indrajit","Kaushik","Partha","Prasenjit","Prosenjit",
    "Rajdeep","Sabyasachi","Saugata","Soumitra","Subhash","Subrata",
    "Sukumar","Tapas","Aditi","Amrita","Aparajita","Bratati",
    "Debdutta","Gargi","Ishita","Jayashree","Kalyani","Keya",
    "Mahua","Mitali","Moumita","Nabaneeta","Paromita","Rituparna",
    "Rupa","Sharmistha","Shatabdi","Shreya","Srabanti","Sucharita",
    "Swastika","Tanushree","Titli",

    #Gujarati / Rajasthani
    "Alpesh","Bhavin","Chirag","Darshan","Falgun","Hiren","Jayesh",
    "Ketan","Mehul","Mitesh","Mukund","Nilesh","Parag","Pritesh",
    "Rajiv","Samir","Tejas","Uday","Vipul","Yatin",
    "Asmita","Bhumi","Charmi","Darshana","Falguni","Hetal",
    "Jagruti","Kajal","Mansi","Minal","Namrata","Payal","Purvi",
    "Reshma","Sejal","Shruti","Swara","Tejal","Urvi","Vaishali",

    #Marathi
    "Ajinkya","Aniket","Chetan","Digvijay","Hrushikesh",
    "Jaydeep","Kedar","Milind","Mangesh","Ninad","Omkar","Prasad",
    "Rupesh","Sachet","Shripad","Tanmay","Umakant","Vineet","Yashwant",
    "Aparna","Ashwini","Janhavi","Ketaki","Madhura",
    "Manisha","Mugdha","Pradnya","Prajakta","Rutuja","Saee",
    "Sakshi","Sayali","Sharvari","Sneha","Sonali","Supriya",
    "Tejashri","Vaidehi",

    #Punjabi / Sikh
    "Amritpal","Baldev","Daljit","Gurpreet","Harjinder","Jaspreet",
    "Kuldeep","Lakhvir","Manpreet","Navjot","Paramjit","Rajinder",
    "Satnam","Simran","Sukhvinder","Tejinder","Amarjeet","Bhupinder",
    "Davinder","Gurmeet","Harleen","Jasleen","Kirandeep","Lovleen",
    "Manmeet","Navneet","Pawandeep","Rajveer","Sukhmani","Tavleen",

    #Muslim-Indian
    "Aamir","Adnan","Arshad","Asif","Danish","Faisal","Farhan",
    "Imran","Irfan","Junaid","Khalid","Mansoor","Nadeem","Nasir",
    "Omar","Parvez","Rafiq","Rashid","Salman","Shahid","Tariq",
    "Wasim","Yusuf","Zaheer","Zubair",
    "Aisha","Amina","Farida","Fatima","Hina","Humaira","Iram",
    "Jasmine","Kulsum","Lubna","Meher","Nafisa","Nasreen","Nazia",
    "Noor","Parveen","Razia","Rukhsar","Sabina","Saira","Shabana",
    "Tabassum","Yasmeen","Zahira","Zeenat",

    #Modern/trendy
    "Aarvi","Advika","Amaira","Ansh","Arham","Avyaan","Darsh",
    "Ivaan","Kiaan","Mishka","Nitara","Pihu","Reyanshi","Riaan",
    "Rivaan","Rudra","Samaira","Shanaya","Tiana",
    "Vivan","Yuvaan","Zian","Aadhira","Anaisha","Kashvi",
    "Myrah","Prithvi","Sahil",

    #Assamese/Odia/other
    "Achinta","Bhaskar","Bishnu","Chitrarupa","Debajyoti",
    "Gitartha","Hemanga","Jayanta","Jutika","Kakoli","Mayuri",
    "Nilotpal","Papori","Pranjal","Rajashree","Ranjit","Rituraj",
    "Sandhya","Sarmistha","Uttam",
    "Bijayalaxmi","Debasish","Gagan","Himadri","Jyotirmoy",
    "Lipika","Madhusmita","Monalisa","Nibedita","Pramod",
    "Priyanka","Rajlaxmi","Sagarika","Snigdha","Subhalaxmi",
    "Sucheta","Tapasya","Trishna","Upasana","Yashodhara",

    #Less common but authentic
    "Agastya","Ahilya","Aishani","Akshara","Annapurna","Apurva",
    "Aruni","Avantika","Chahana","Daksha","Devyani","Dhanya",
    "Eesha","Ekta","Garima","Hansika","Hemavati","Ipsita",
    "Jahnavi","Jyotsna","Karuna","Lavanya","Lopamudra","Madhavi",
    "Nayantara","Nivedita","Oorja","Ojaswini","Pallava","Pramila",
    "Radhika","Renuka","Sahana","Sharada","Tarini","Ujjwala",
    "Vaagdevi","Yamuna","Yukta","Zoya",

    #Additional male names
    "Abhinav","Ajit","Akash","Alok","Amar","Amol","Anant",
    "Anil","Arvind","Ashish","Ashwin","Badri",
    "Balu","Barun","Bhavesh","Bipin","Brij","Chaitanya",
    "Chandramouli","Charu","Chinmay","Chiranjeevi","Datta",
    "Devdas","Dhananjay","Dharma","Dilip","Dinanath",
    "Eknath","Gajanan","Ghanshyam","Gopinath",
    "Govardhan","Gulab","Gururaj","Hansraj","Harish",
    "Harshal","Hemendra","Hitesh","Hrithik","Indra",
    "Jagannath","Jaidev","Jatin","Jitendra","Karunesh",
    "Kashyap","Kaushal","Keerthi","Kripa","Krishan",
    "Laxman","Lokesh","Madan","Malhar","Manohar",
    "Mayank","Mrinal","Nagendra","Naman","Narayan",
    "Navin","Nirmal","Nirupam","Onkar","Padmanabh",
    "Parshuram","Phalguni","Prabhakaran","Prahlad","Pratap",
    "Purushottam","Raghunath","Rajaram","Rajkumar","Raman",
    "Ramakrishna","Ramprasad","Ranganath","Ratnesh","Ravindra",
    "Sadashiv","Sahadev","Sambhaji","Sanjiv","Satish",
    "Satyajit","Shailendra","Sharad","Shashank","Shivaji",
    "Shridhar","Shrikant","Siddhant","Srinivas",
    "Sudhir","Surendra","Sushant","Swaroop","Trilok",
    "Upendra","Uttkarsh","Vasant","Venkatraman","Vijaykumar",
    "Vikrant","Vilas","Vinayak","Vishwanath","Vivek",
    "Yashpal","Yogendra","Yugandhar",

    #Additional female names
    "Aarti","Abha","Akanksha","Ambika","Amrapali","Anjali",
    "Ankita","Anusha","Arundhati","Bhairavi","Chandana",
    "Chandrika","Charulata","Damayanti","Devika","Dhara",
    "Gayatri","Gitanjali","Harini",
    "Harshada","Hemali","Himani","Indrani","Jagrati",
    "Jayashri","Jyotika","Kamini","Kanchan",
    "Kashyapi","Kaumudi","Kusum","Latika","Leela",
    "Madhuri","Maithili","Mandira","Manjula","Medha",
    "Meghana","Mithra","Mrinalini","Nalini","Nandita",
    "Niharika","Nishtha","Padmini","Pankaja","Parul",
    "Prabhavati","Prafulla","Prameela","Prasanna","Prativa",
    "Priti","Pushpita","Rajeshwari","Ramaa","Rangita",
    "Rasika","Ratna","Roopali","Sadhana",
    "Samiksha","Sangita","Sarojini","Saudamini",
    "Shaila","Shakti","Shamala","Sharanya","Shefali",
    "Shilpa","Shobhana","Shrutika","Siddhi","Smriti",
    "Sridevi","Subhadra","Sugandhi",
    "Sukanya","Sumitra","Sunanda","Suprabha","Sutapa",
    "Swapna","Trupti","Tulasi","Ujwala","Urmila",
    "Urvashi","Vaani","Vaishnavi","Varsha","Vasumati",
    "Veena","Vidyullata","Vijaya","Vimla","Vinata",
    "Vishakha","Yamini","Yashoda",

    #Final additions to reach 1000
    "Pranay","Ojas","Mihir","Neel","Rishab","Siddharth",
    "Kshitij","Arnav","Parth","Ishan","Vedant","Shivam",
    "Lakshya","Krish","Harsha","Mithun","Suraj","Pavan",
    "Santosh","Ramana","Chandru","Babu","Venkat","Srikanth",
    "Nagesh","Mahendra","Ramaswamy","Govindaraj","Sundaram",
    "Palaniswamy","Sethuraman","Natarajan","Rangarajan",
    "Sivasubramanian","Padmanaban","Sundaresan","Krishnamurthy",
    "Thyagarajan","Seshadri","Vaidyanathan",
    "Padmavathi","Soundarya","Thamarai","Selvi","Manimegalai",
    "Andal","Vasuki","Sivakami","Pattammal","Alamelu",
    "Sakunthala","Seethalakshmi","Gomathy","Bhagyalakshmi",
    "Visalakshi","Vijayalakshmi","Valarmathi","Senthamarai",
    "Thangam","Chellammal",
    "Rithika","Karthika","Nandhini","Ramya","Swathi",
    "Ashvini","Bhavani","Haritha","Madhulika","Chandni",
    "Devaki","Sumathi","Vasundara","Mridula",
    "Ratnamala","Sarada","Swarnamala","Hridaya",
    "Keshavi","Jagadamba","Mahamaya","Bhoomika","Anandi",
    "Prerna","Roshni","Heena","Anuradha","Girija",
    "Durgesh","Eklavya","Thamizh","Rohita",
]

# removing duplicates while keeping order
unique_names = []
for name in indian_names:
    if name not in unique_names:
        unique_names.append(name)

# taking first 1000 names for training
training_names = unique_names[:1000]

print(f"Unique names collected : {len(unique_names)}")
print(f"Names used for training: {len(training_names)}")

#save as .txt file
with open("TrainingNames.txt", "w") as f:
    for n in training_names:
        f.write(n + "\n")

lengths = [len(n) for n in training_names]
print(f"\nShortest : '{min(training_names,key=len)}' ({min(lengths)} chars)")
print(f"Longest  : '{max(training_names,key=len)}' ({max(lengths)} chars)")
print(f"Average  : {sum(lengths)/len(lengths):.1f} chars")
print(f"\nFirst 15 : {training_names[:15]}")

#environment setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random, string, os, math, time
from collections import Counter, defaultdict

# fixing random seed so results are same every time
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

#Load training names and build character vocabulary
with open("TrainingNames.txt") as f:
    raw_names = [l.strip() for l in f if l.strip()]

# convert to lowercase
names = [n.lower() for n in raw_names]

# defining special tokens
SOS_TOKEN = "<SOS>"   # start of sequence
EOS_TOKEN = "<EOS>"   # end of sequence
PAD_TOKEN = "<PAD>"   # padding for equal length

all_chars = sorted(set("".join(names)))
vocab     = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + all_chars
vocab_size = len(vocab)

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
PAD_IDX = char_to_idx[PAD_TOKEN]
SOS_IDX = char_to_idx[SOS_TOKEN]
EOS_IDX = char_to_idx[EOS_TOKEN]

print(f"Training names : {len(names)}")
print(f"Unique chars   : {''.join(all_chars)}  ({len(all_chars)})")
print(f"Vocab size     : {vocab_size} (chars + 3 special tokens)")

#convert names to padded tensor batches(dataset helper)
def name_to_tensor(name):
    # Encode a name as index tensor.
    ids = [SOS_IDX] + [char_to_idx[c] for c in name] + [EOS_IDX]
    return torch.tensor(ids, dtype=torch.long)

def prepare_batches(name_list, batch_size=64):
    # create input-target pairs
    random.shuffle(name_list)
    batches = []
    for i in range(0, len(name_list), batch_size):
        chunk   = name_list[i:i+batch_size]
        tensors = [name_to_tensor(n) for n in chunk]
        max_len = max(t.size(0) for t in tensors)

        # creating padded tensor manually
        padded = []

        for t in tensors:
          temp = t.tolist()   # convert tensor to list

          # add padding manually
          while len(temp) < max_len:
            temp.append(PAD_IDX)

        padded.append(temp)

        # convert back to tensor
        padded = torch.tensor(padded, dtype=torch.long)
        for j, t in enumerate(tensors):
          padded[j, :t.size(0)] = t

          inp = padded[:, :-1].to(device)   # drop last position
          tgt = padded[:, 1:].to(device)    # drop first position
          batches.append((inp, tgt))
    return batches

#check
t = name_to_tensor("priya")
print("Encoded 'priya':", t.tolist())
print("Decoded back   :", [idx_to_char[i] for i in t.tolist()])

#exploratory data analysis - helps understand the statistical landscape of the indian names

#character frequencies and name-length distribution
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

#Character frequency in training data
all_text   = "".join(names)
char_freq  = Counter(all_text)
chars_sorted = sorted(char_freq, key=char_freq.get, reverse=True)

ax1 = fig.add_subplot(gs[0, :])
bars = ax1.bar(chars_sorted, [char_freq[c] for c in chars_sorted],
               color='steelblue', edgecolor='black', linewidth=0.3)
ax1.set_title("Character Frequency in Training Names", fontsize=13)
ax1.set_ylabel("Count")
ax1.set_xlabel("Character")

#Name length distribution
ax2 = fig.add_subplot(gs[1, 0])
lens = [len(n) for n in names]
ax2.hist(lens, bins=range(2, max(lens)+2), color='salmon',
         edgecolor='black', linewidth=0.4, alpha=0.85)
ax2.set_title("Name Length Distribution", fontsize=13)
ax2.set_xlabel("Length (characters)")
ax2.set_ylabel("Count")
ax2.axvline(np.mean(lens), color='darkred', ls='--', label=f'mean={np.mean(lens):.1f}')
ax2.legend()

#Most common starting characters
ax3 = fig.add_subplot(gs[1, 1])
start_freq = Counter(n[0] for n in names)
top_starts = start_freq.most_common(15)
ax3.barh([x[0] for x in top_starts][::-1],
         [x[1] for x in top_starts][::-1],
         color='mediumseagreen', edgecolor='black', linewidth=0.3)
ax3.set_title("Top 15 Starting Characters", fontsize=13)
ax3.set_xlabel("Count")

plt.savefig("eda_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Observations:")
print(f"  Most common char  : '{chars_sorted[0]}' ({char_freq[chars_sorted[0]]} occurrences)")
print(f"  Most common start : '{top_starts[0][0]}' ({top_starts[0][1]} names)")
print(f"  Median name length: {int(np.median(lens))} chars")

"""## Task 1 : MODEL IMPLEMENTATION"""

#Vanilla Recurrent Neural Network (RNN)

class VanillaRNNCell(nn.Module):
    #Takes current input x_t and previous hidden state h_{t-1}, produces new hidden state h_t.
    #Internally: h = tanh(W_ih @ x + W_hh @ h_prev + bias)
    #The two Linear layers each contribute one half of the sum.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size,  hidden_size)   # input  -> hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)   # hidden -> hidden

    def forward(self, x, h_prev):
        # Classic RNN equation
        return torch.tanh(self.W_ih(x) + self.W_hh(h_prev))

#Full Vanilla RNN for character-level generation.
class VanillaRNNModel(nn.Module):
    # Stack: Embedding  -->  RNNCell x num_layers  -->  Linear(hidden -> vocab)
    # At inference we feed one character at a time and sample the next.
    def __init__(self, vs, emb=32, hs=128, nl=2, drop=0.3):
        super().__init__()
        self.hs, self.nl = hs, nl

        self.embedding = nn.Embedding(vs, emb, padding_idx=PAD_IDX)
        self.cells = nn.ModuleList()
        for i in range(nl):
            self.cells.append(VanillaRNNCell(emb if i == 0 else hs, hs))
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hs, vs)

    def forward(self, x, hidden=None):
        B, T = x.size()
        if hidden is None:
            hidden = [torch.zeros(B, self.hs, device=x.device) for _ in range(self.nl)]

        emb = self.drop(self.embedding(x))            # (B, T, emb)
        outputs = []
        for t in range(T):
            inp = emb[:, t, :]
            new_h = []
            for i, cell in enumerate(self.cells):
                h = cell(inp, hidden[i])
                new_h.append(h)
                inp = self.drop(h)                     # next layer input
            hidden = new_h
            outputs.append(inp)

        out = torch.stack(outputs, dim=1)              # (B, T, hs)
        return self.fc(out), hidden                    # logits, final hidden


vanilla_rnn = VanillaRNNModel(vocab_size).to(device)
n_params_vanilla = sum(p.numel() for p in vanilla_rnn.parameters() if p.requires_grad)
print(f"Vanilla RNN - {n_params_vanilla:,} trainable parameters")

#Bidirectional Long Short-Term Memory (BLSTM)

#Single LSTM cell with explicit gate computation.
class LSTMCell(nn.Module):
    #The 'gates' linear layer does all four transformations in one go
    #(more efficient than four separate layers), then we split the output.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hs = hidden_size
        #One big linear: [x ; h_prev] -> 4*hidden  (i, f, g, o stacked)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        raw = self.gates(combined)
        i, f, g, o = raw.chunk(4, dim=1)

        i = torch.sigmoid(i)   # input gate: controls new information flow
        f = torch.sigmoid(f)   # forget gate: controls what to discard
        g = torch.tanh(g)      # candidate values: proposed new content
        o = torch.sigmoid(o)   # output gate: filters what to expose

        c_new = f * c_prev + i * g       # update cell state (the long-term memory)
        h_new = o * torch.tanh(c_new)    # filtered output (short-term state)
        return h_new, c_new

#Bidirectional LSTM for name generation.
class BLSTMModel(nn.Module):
    # Training  : uses both directions for rich representations
    # Generation: forward direction only (cannot peek ahead)
    def __init__(self, vs, emb=32, hs=128, nl=2, drop=0.3):
        super().__init__()
        self.hs, self.nl = hs, nl

        self.embedding = nn.Embedding(vs, emb, padding_idx=PAD_IDX)

        # Forward and backward cell stacks
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for i in range(nl):
            inp = emb if i == 0 else hs
            self.fwd_cells.append(LSTMCell(inp, hs))
            self.bwd_cells.append(LSTMCell(inp, hs))

        self.drop     = nn.Dropout(drop)
        self.compress = nn.Linear(hs * 2, hs)     # merge fwd+bwd back to hs
        self.fc       = nn.Linear(hs, vs)          # bidirectional output head
        self.fc_fwd   = nn.Linear(hs, vs)          # forward-only output head

# Helper to run an LSTM chain in one direction.
    def _run_one_direction(self, emb, cells, reverse=False):
        # Avoids code duplication between forward and backward passes.
        B, T, _ = emb.size()
        h = [torch.zeros(B, self.hs, device=emb.device) for _ in range(self.nl)]
        c = [torch.zeros(B, self.hs, device=emb.device) for _ in range(self.nl)]

        steps = range(T-1, -1, -1) if reverse else range(T)
        outs = []
        for t in steps:
            inp = emb[:, t, :]
            for li, cell in enumerate(cells):
                h[li], c[li] = cell(inp, h[li], c[li])
                inp = self.drop(h[li])
            outs.append(inp)
        if reverse:
            outs = outs[::-1]
        return torch.stack(outs, dim=1), h, c

    def forward(self, x, hidden=None):
        emb = self.drop(self.embedding(x))
        fwd_o, fh, fc = self._run_one_direction(emb, self.fwd_cells, False)
        bwd_o, _,  _  = self._run_one_direction(emb, self.bwd_cells, True)
        combined = torch.cat([fwd_o, bwd_o], dim=2)
        compressed = torch.tanh(self.compress(combined))
        return self.fc(compressed), (fh, fc)

    def forward_only(self, x, hidden=None):
        # Forward-direction pass used during generation.
        B, T = x.size()
        emb = self.embedding(x)
        if hidden is None:
            h = [torch.zeros(B, self.hs, device=x.device) for _ in range(self.nl)]
            c = [torch.zeros(B, self.hs, device=x.device) for _ in range(self.nl)]
        else:
            h, c = hidden

        outs = []
        for t in range(T):
            inp = emb[:, t, :]
            for li, cell in enumerate(self.fwd_cells):
                h[li], c[li] = cell(inp, h[li], c[li])
                inp = self.drop(h[li])
            outs.append(inp)
        outs = torch.stack(outs, dim=1)
        return self.fc_fwd(outs), (h, c)


blstm_model = BLSTMModel(vocab_size).to(device)
n_params_blstm = sum(p.numel() for p in blstm_model.parameters() if p.requires_grad)
print(f"BLSTM - {n_params_blstm:,} trainable parameters")

#RNN with Basic Attention Mechanism

#Additive attention: learns to score how well each past hidden
class BahdanauAttention(nn.Module):
    # state matches the current query via a small feedforward network.

    def __init__(self, hs):
        super().__init__()
        self.Wq = nn.Linear(hs, hs)           # project query
        self.Wk = nn.Linear(hs, hs)           # project keys
        self.v  = nn.Linear(hs, 1)            # score to scalar

    def forward(self, query, keys):
        # query : (B, hs) - current hidden state
        # keys  : (B, T, hs) - all past hidden states
        q = self.Wq(query).unsqueeze(1)        # (B, 1, hs)
        k = self.Wk(keys)                      # (B, T, hs)
        energy  = self.v(torch.tanh(q + k))    # (B, T, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)   # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys)  # (B, 1, hs)
        return context.squeeze(1), weights

#Vanilla RNN backbone + Bahdanau attention over past states.
class AttentionRNNModel(nn.Module):
    #At each timestep the model:
    # - Runs the RNN cell stack
    # - Attends over all previous hidden states
    # - Merges attended context with current state
    # - Projects to vocabulary logits

    def __init__(self, vs, emb=32, hs=128, nl=2, drop=0.3):
        super().__init__()
        self.hs, self.nl = hs, nl

        self.embedding = nn.Embedding(vs, emb, padding_idx=PAD_IDX)
        self.cells = nn.ModuleList()
        for i in range(nl):
            self.cells.append(VanillaRNNCell(emb if i == 0 else hs, hs))
        self.attention = BahdanauAttention(hs)
        self.drop  = nn.Dropout(drop)
        self.merge = nn.Linear(hs * 2, hs)    # concat(hidden, context) -> hs
        self.fc    = nn.Linear(hs, vs)

    def forward(self, x, hidden=None):
        B, T = x.size()
        if hidden is None:
            hidden = [torch.zeros(B, self.hs, device=x.device) for _ in range(self.nl)]

        emb = self.drop(self.embedding(x))
        all_h   = []    # accumulate top-layer hidden states for attention
        outputs = []
        self._last_attn_weights = []   # stash for visualisation later

        for t in range(T):
            inp = emb[:, t, :]
            new_h = []
            for i, cell in enumerate(self.cells):
                h = cell(inp, hidden[i])
                new_h.append(h)
                inp = self.drop(h)
            hidden = new_h
            top_h = hidden[-1]
            all_h.append(top_h)

            # Apply attention over all previously generated hidden states
            if len(all_h) > 1:
                keys = torch.stack(all_h[:-1], dim=1)
                ctx, wts = self.attention(top_h, keys)
                self._last_attn_weights.append(wts.detach().cpu())
                merged = torch.tanh(self.merge(torch.cat([top_h, ctx], dim=1)))
            else:
                # First timestep -- nothing to attend to yet
                merged = top_h
                self._last_attn_weights.append(None)
            outputs.append(self.fc(merged))

        return torch.stack(outputs, dim=1), hidden


attn_rnn = AttentionRNNModel(vocab_size).to(device)
n_params_attn = sum(p.numel() for p in attn_rnn.parameters() if p.requires_grad)
print(f"RNN + Attention - {n_params_attn:,} trainable parameters")

# Parameter comparison
print("-" * 62)
print(f"{'Model':<24}  {'Params':>12}  {'Embed':>5}  {'Hidden':>6}  {'Layers':>6}")
print("-" * 62)
for name, n in [("Vanilla RNN", n_params_vanilla),
                ("Bidirectional LSTM", n_params_blstm),
                ("RNN + Attention", n_params_attn)]:
    print(f"{name:<24}  {n:>12,}  {32:>5}  {128:>6}  {2:>6}")
print("-" * 62)
print()
print("The BLSTM is largest because each LSTM cell has 4x the parameters")
print("of a vanilla RNN cell (four gate projections), and the bidirectional")
print("setup doubles the recurrent layers. Attention adds modest overhead.")

#training loop
def train_model(model, name_list, epochs=100, bs=64, lr=0.003, label="Model"):
    # Returns the per-epoch average loss so we can plot learning curves.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5)

    history = []
    t0 = time.time()
    print(f"\n>>> Training {label}")

    for epoch in range(epochs):
        model.train()
        batches = prepare_batches(name_list, bs)
        total_loss, n_batch = 0.0, 0

        for inp, tgt in batches:
            optimizer.zero_grad()
            logits, _ = model(inp)
            loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        avg = total_loss / n_batch
        history.append(avg)
        scheduler.step(avg)

        if (epoch+1) % 25 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d} | loss {avg:.4f} | lr {lr_now:.5f}")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s -- final loss {history[-1]:.4f}")
    return history

#Train all three models
print("Training started...")

loss_vanilla = train_model(vanilla_rnn, names, label="Vanilla RNN")
loss_blstm   = train_model(blstm_model, names, label="Bidirectional LSTM")
loss_attn    = train_model(attn_rnn,    names, label="RNN + Attention")

#training curves
plt.figure(figsize=(10, 5))
for losses, label, color in [
    (loss_vanilla, 'Vanilla RNN', '#2196F3'),
    (loss_blstm,   'BLSTM',       '#4CAF50'),
    (loss_attn,    'RNN + Attention', '#FF9800')]:
    plt.plot(losses, label=label, color=color, linewidth=2, alpha=0.85)
plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Over Time')
plt.legend(fontsize=11); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

#name generation
def generate_name(model, temperature=0.8, max_len=20, forward_only=False):
    #Sample a single name from a trained model
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([[SOS_IDX]], device=device)
        hidden = None
        chars = []

        for _ in range(max_len):
            if forward_only:
                logits, hidden = model.forward_only(inp, hidden)
            else:
                logits, hidden = model(inp, hidden)

            #Temperature scaling
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx    = torch.multinomial(probs, 1).item()

            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue

            chars.append(idx_to_char[idx])
            inp = torch.tensor([[idx]], device=device)

    return "".join(chars).capitalize()

#Generate n unique names with at least 2 characters
def generate_batch(model, n=200, temperature=0.8, forward_only=False):
    #We over-sample to account for potential empty outputs or duplicates
    results = set()
    for _ in range(n * 3):
        name = generate_name(model, temperature, forward_only=forward_only)
        if len(name) >= 2:
            results.add(name)
        if len(results) >= n:
            break
    return list(results)[:n]

#generate names from every model
N_GEN = 200

print("Generating names (temperature = 0.8) ...\n")

vanilla_names  = generate_batch(vanilla_rnn, N_GEN, 0.8)
blstm_names    = generate_batch(blstm_model, N_GEN, 0.8, forward_only=True)
attn_names     = generate_batch(attn_rnn,    N_GEN, 0.8)

for tag, nms in [("Vanilla RNN", vanilla_names),
                 ("BLSTM", blstm_names),
                 ("RNN+Attn", attn_names)]:
    print(f"{tag:14s}: {len(nms)} names  |  e.g. {nms[:8]}")

# Save to files for later reference
for fn, nms in [("vanilla_rnn_names.txt", vanilla_names),
                ("blstm_names.txt", blstm_names),
                ("attention_rnn_names.txt", attn_names)]:
    with open(fn, "w") as f:
        f.write("\n".join(nms))
print("\nGenerated names saved to text files.")

#effect of temperature on generation
temps = [0.4, 0.6, 0.8, 1.0, 1.2]
print("=" * 80)
print(f"{'Temp':>5}  {'Vanilla RNN':<25}  {'BLSTM':<25}  {'RNN+Attn':<25}")
print("-" * 80)

for temp in temps:
    v_samples = [generate_name(vanilla_rnn, temp) for _ in range(5)]
    b_samples = [generate_name(blstm_model, temp, forward_only=True) for _ in range(5)]
    a_samples = [generate_name(attn_rnn, temp) for _ in range(5)]
    for i in range(5):
        v = v_samples[i] if i < len(v_samples) else ""
        b = b_samples[i] if i < len(b_samples) else ""
        a = a_samples[i] if i < len(a_samples) else ""
        prefix = f"{temp:.1f}" if i == 0 else ""
        print(f"{prefix:>5}  {v:<25}  {b:<25}  {a:<25}")
    print()

print("Low temperature  -> safe, repetitive (sticks to common patterns)")
print("High temperature -> creative but sometimes garbled")

"""## Task 2 : QUANTITATIVE EVALUATION"""

#evaluation metrics
training_lower = set(n.lower() for n in names)

def novelty_rate(gen_names, train_set):
    #Fraction of generated names absent from training data.
    #A model that merely memorises scores 0%. We want high novelty.
    novel = sum(1 for n in gen_names if n.lower() not in train_set)
    return 100.0 * novel / len(gen_names) if gen_names else 0

def diversity_rate(gen_names):
    #Fraction of unique names among all generated.
    #Low diversity = mode collapse (same patterns over and over).
    unique = len(set(n.lower() for n in gen_names))
    return 100.0 * unique / len(gen_names) if gen_names else 0

def avg_name_length(gen_names):
    return np.mean([len(n) for n in gen_names])

# Compute for each model
metrics = {}
for tag, nms in [("Vanilla RNN", vanilla_names),
                 ("BLSTM", blstm_names),
                 ("RNN + Attention", attn_names)]:
    metrics[tag] = {
        "novelty":  novelty_rate(nms, training_lower),
        "diversity": diversity_rate(nms),
        "avg_len":  avg_name_length(nms),
        "count":    len(nms),
    }

print("-" * 68)
print(f"{'Model':<20} {'Novelty':>9} {'Diversity':>11} {'Avg Len':>9} {'Count':>7}")
print("-" * 68)
for tag, m in metrics.items():
    print(f"{tag:<20} {m['novelty']:>8.1f}% {m['diversity']:>10.1f}% {m['avg_len']:>8.1f} {m['count']:>7}")
print("-" * 68)

#evaluation bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = list(metrics.keys())
colors = ['#2196F3', '#4CAF50', '#FF9800']

for ax, key, title in zip(axes,
                          ['novelty', 'diversity', 'avg_len'],
                          ['Novelty Rate (%)', 'Diversity (%)', 'Avg Name Length']):
    vals = [metrics[l][key] for l in labels]
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=13)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.1f}%" if key != 'avg_len' else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                fmt, ha='center', fontweight='bold', fontsize=10)
    if key != 'avg_len':
        ax.set_ylim(0, 110)

plt.suptitle("Quantitative Model Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("evaluation_metrics.png", dpi=150, bbox_inches='tight')
plt.show()

#Character distribution comparison (KL divergence and visual)
def char_distribution(name_list):
    # Return a normalised character frequency dictionary.
    text = "".join(n.lower() for n in name_list)
    counts = Counter(text)
    total = sum(counts.values())
    return {c: counts.get(c, 0) / total for c in all_chars}

train_dist  = char_distribution(names)
van_dist    = char_distribution(vanilla_names)
blstm_dist  = char_distribution(blstm_names)
attn_dist   = char_distribution(attn_names)

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(all_chars))
w = 0.2

ax.bar(x - 1.5*w, [train_dist[c] for c in all_chars], w, label='Training', color='gray', alpha=0.7)
ax.bar(x - 0.5*w, [van_dist[c] for c in all_chars],   w, label='Vanilla RNN', color='#2196F3', alpha=0.7)
ax.bar(x + 0.5*w, [blstm_dist[c] for c in all_chars],  w, label='BLSTM', color='#4CAF50', alpha=0.7)
ax.bar(x + 1.5*w, [attn_dist[c] for c in all_chars],   w, label='RNN+Attn', color='#FF9800', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(all_chars)
ax.set_ylabel("Relative Frequency")
ax.set_title("Character Frequency: Training vs Generated Names")
ax.legend()
plt.tight_layout()
plt.savefig("char_distribution_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# KL-divergence gives a single number summarising distributional distance
def kl_div(p, q, chars):
    # KL(P || Q) with smoothing to avoid log(0).
    eps = 1e-8
    return sum(p[c] * math.log((p[c]+eps)/(q[c]+eps)) for c in chars)

print(f"KL divergence from training distribution:")
print(f"  Vanilla RNN  : {kl_div(train_dist, van_dist, all_chars):.4f}")
print(f"  BLSTM        : {kl_div(train_dist, blstm_dist, all_chars):.4f}")
print(f"  RNN + Attn   : {kl_div(train_dist, attn_dist, all_chars):.4f}")
print("(Lower is better -- means the generated distribution is closer to training)")

"""##Task 3 : QUALITATIVE ANALYSIS"""

def qualitative_report(gen_names, model_name, train_set):
    #We categorise generated names into novel vs memorised,
    #and flag common failure modes like repetition or weird clusters.

    print(f"\n{'-'*60}")
    print(f"  {model_name}")
    print(f"{'-'*60}")

    lengths = [len(n) for n in gen_names]
    novel   = [n for n in gen_names if n.lower() not in train_set]
    copied  = [n for n in gen_names if n.lower() in train_set]

    print(f"\n  Length range: {min(lengths)}-{max(lengths)} chars (mean {np.mean(lengths):.1f})")
    print(f"  Novel: {len(novel)} | Memorised: {len(copied)}")

    #Show representative novel names grouped by length
    short  = sorted([n for n in novel if len(n) <= 5],  key=len)[:6]
    medium = sorted([n for n in novel if 5 < len(n) <= 9], key=len)[:6]
    long_n = sorted([n for n in novel if len(n) > 9],   key=len)[:6]

    print(f"\n  Novel samples:")
    print(f"    Short  (<=5) : {short}")
    print(f"    Medium (6-9) : {medium}")
    print(f"    Long   (10+) : {long_n}")

    print(f"\n  Memorised samples (from training): {copied[:8]}")

    #Failure mode detection

    #Names that are too short to be realistic
    too_short   = [n for n in gen_names if len(n) < 3]

    #Names that are unreasonably long
    too_long    = [n for n in gen_names if len(n) > 15]

    #Repetitive characters
    repetitive  = [n for n in gen_names
                   if any(c*3 in n.lower() for c in all_chars)]

    #Heavy consonant clusters (4+ consonants in a row)
    # detect odd consonant patterns
    vowels = set("aeiou")
    consonant_heavy = []
    for n in gen_names:
        lo = n.lower()
        max_cons = 0
        cur = 0
        for ch in lo:
            if ch not in vowels:
                cur += 1
                max_cons = max(max_cons, cur)
            else:
                cur = 0
        if max_cons >= 4:
            consonant_heavy.append(n)

    print(f"\n  Failure modes:")
    print(f"    Too short (<3 chars)     : {len(too_short):3d}  {too_short[:5]}")
    print(f"    Too long (>15 chars)     : {len(too_long):3d}  {too_long[:5]}")
    print(f"    Repetitive (e.g. 'aaa')  : {len(repetitive):3d}  {repetitive[:5]}")
    print(f"    Heavy consonant clusters : {len(consonant_heavy):3d}  {consonant_heavy[:5]}")


qualitative_report(vanilla_names, "VANILLA RNN",        training_lower)
qualitative_report(blstm_names,   "BIDIRECTIONAL LSTM", training_lower)
qualitative_report(attn_names,    "RNN + ATTENTION",    training_lower)

#Length distribution comparison: training vs all three models
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

for ax, nms, title, col in zip(
    axes,
    [names, vanilla_names, blstm_names, attn_names],
    ['Training Data', 'Vanilla RNN', 'BLSTM', 'RNN + Attention'],
    ['gray', '#2196F3', '#4CAF50', '#FF9800']):

    lens = [len(n) for n in nms]
    ax.hist(lens, bins=range(2, 22), color=col, alpha=0.75,
            edgecolor='black', linewidth=0.4)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Name Length")
    ax.axvline(np.mean(lens), color='red', ls='--', lw=1.2)

axes[0].set_ylabel("Count")
plt.suptitle("Name Length Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("length_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

#attention weight visualization
def generate_with_attention_trace(model, temperature=0.7, max_len=15):
    # Generate a name and record attention weights at each step.
    model.eval()
    with torch.no_grad():
        chars = ['<SOS>']
        all_weights = []

        h_list = [torch.zeros(1, model.hs, device=device) for _ in range(model.nl)]
        past_h = []

        inp_token = SOS_IDX
        for step in range(max_len):
            inp = torch.tensor([[inp_token]], device=device)
            e = model.embedding(inp)[:, 0, :]

            # Run through the RNN cell stack
            new_h = []
            layer_inp = e
            for i, cell in enumerate(model.cells):
                h = cell(layer_inp, h_list[i])
                new_h.append(h)
                layer_inp = h
            h_list = new_h
            top_h = h_list[-1]
            past_h.append(top_h)

            # Apply attention if we have past states to attend over
            if len(past_h) > 1:
                keys = torch.stack(past_h[:-1], dim=1)
                ctx, wts = model.attention(top_h, keys)
                all_weights.append(wts.squeeze(0).cpu().numpy())
                merged = torch.tanh(model.merge(torch.cat([top_h, ctx], dim=1)))
            else:
                all_weights.append(None)
                merged = top_h

            logits = model.fc(merged) / temperature
            probs  = F.softmax(logits, dim=-1)
            idx    = torch.multinomial(probs, 1).item()

            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            chars.append(idx_to_char[idx])
            inp_token = idx

    return chars, all_weights

# Generate a sample and plot its attention pattern
chars, attn_wts = generate_with_attention_trace(attn_rnn, temperature=0.7)
name_str = "".join(c for c in chars if c != '<SOS>')
print(f"Generated name: {name_str.capitalize()}")

# Build attention matrix (rows = generation step, cols = attended position)
valid_steps = [(i, w) for i, w in enumerate(attn_wts) if w is not None]
if valid_steps:
    max_attn_len = max(len(w) for _, w in valid_steps)
    attn_matrix = np.zeros((len(valid_steps), max_attn_len))
    for row_idx, (step_idx, w) in enumerate(valid_steps):
        attn_matrix[row_idx, :len(w)] = w

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap='YlOrRd', aspect='auto')

    y_labels = [chars[i+1] if i+1 < len(chars) else '?' for i, _ in valid_steps]
    x_labels = chars[:max_attn_len]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Attended Position (past characters)")
    ax.set_ylabel("Current Generation Step")
    ax.set_title(f"Attention Heatmap: '{name_str.capitalize()}'")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
  print("Name too short to visualise attention.")
