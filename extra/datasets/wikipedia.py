from multiprocessing import Pool
import sys, pickle, random, unicodedata, functools, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from tinygrad.helpers import diskcache, getenv
from tqdm.contrib.concurrent import process_map

BASEDIR = Path(__file__).parent / "wiki"

################### Tokenization #####################

def _is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  return unicodedata.category(char) == "Zs"

def _is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  return unicodedata.category(char).startswith("C")

def _is_punctuation(char):
  # range(33, 48) -> ! " # $ % & ' ( ) * + , - . /
  # range(58, 65) -> : ; < = > ? @
  # range(91, 97) -> [ \ ] ^ _
  # range(123, 127) -> { | } ~
  if (cp := ord(char)) in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
    return True
  return unicodedata.category(char).startswith("P")

def _run_split_on_punc(text):
  if text in ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"):
    return [text]
  start_new_word = True
  output = []
  for i in range(len(text)):
    if _is_punctuation(char := text[i]):
      output.append([char])
      start_new_word = True
    else:
      if start_new_word:
        output.append([])
      start_new_word = False
      output[-1].append(char)
  return ["".join(x) for x in output]

def _run_strip_accents(text):
  output = []
  for char in unicodedata.normalize("NFD", text):
    if unicodedata.category(char) != "Mn":
      output.append(char)
  return "".join(output)

def _clean_text(text):
  output = []
  for char in text:
    if not ((cp := ord(char)) == 0 or cp == 0xfffd or _is_control(char)):
      output.append(" " if _is_whitespace(char) else char)
  return "".join(output)

def _wordpiece_tokenize(text, vocab):
  text = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
  output_tokens = []
  for token in text.strip().split():
    chars = list(token)
    if len(chars) > 200:
      output_tokens.append("[UNK]")
      continue

    is_bad = False
    start = 0
    sub_tokens = []
    while start < len(chars):
      end = len(chars)
      cur_substr = None
      while start < end:
        substr = "".join(chars[start:end])
        if start > 0: substr = "##" + substr
        if substr in vocab:
          cur_substr = substr
          break
        end -= 1
      if cur_substr is None:
        is_bad = True
        break
      sub_tokens.append(cur_substr)
      start = end

    if is_bad: output_tokens.append("[UNK]")
    else: output_tokens.extend(sub_tokens)
  return output_tokens

class Tokenizer:
  def __init__(self, vocab_file):
    self.vocab = {}
    with open(vocab_file) as f:
      for line in f:
        line = line.decode("utf-8", "ignore") if isinstance(line, bytes) else line
        if (token := line.strip()) and token not in self.vocab: self.vocab[token] = len(self.vocab)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    text = _clean_text(text.decode("utf-8", "ignore") if isinstance(text, bytes) else text)
    # BasicTokenizer
    split_tokens = []
    for token in text.strip().split():
      split_tokens.extend(_run_split_on_punc(_run_strip_accents(token.lower())))
    split_tokens = " ".join(split_tokens).strip().split()
    # WordpieceTokenizer
    tokens = []
    for token in split_tokens:
      tokens.extend(_wordpiece_tokenize(token, self.vocab))
    return tokens

  def convert_tokens_to_ids(self, tokens): return [self.vocab[token] for token in tokens]
  def convert_ids_to_tokens(self, ids): return [self.inv_vocab[id] for id in ids]

##################### Feature transformation #####################

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_masked_lm_predictions(tokens, tokenizer, rng, vocab_words):
  cand_indices = []
  for i, token in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indices.append(i)

  rng.shuffle(cand_indices)
  output_tokens = list(tokens)
  num_to_predict = min(getenv('MAX_PREDICTIONS_PER_SEQ', 20), max(1, int(round(len(tokens) * 0.15))))

  masked_lms = []
  covered_indices = set()
  for index in cand_indices:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indices:
      continue
    covered_indices.add(index)

    masked_token = None
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      if rng.random() < 0.5:
        masked_token = tokens[index]
      else:
        masked_token = vocab_words[rng.randint(0, len(tokenizer.vocab) - 1)]

    output_tokens[index] = masked_token
    masked_lms.append((index, tokens[index]))
  masked_lms = sorted(masked_lms, key=lambda x: x[0])

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p[0])
    masked_lm_labels.append(p[1])

  return output_tokens, masked_lm_positions, masked_lm_labels

def create_instances_from_document(rng, tokenizer, doc, di, documents):
  max_num_tokens = getenv('MAX_SEQ_LENGTH', 128) - 3 # [CLS] + 2 * [SEP]

  target_seq_length = max_num_tokens
  if rng.random() < 0.1:
    target_seq_length = rng.randint(2, max_num_tokens)

  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(doc):
    segment = doc[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(doc) - 1 or current_length >= target_seq_length:
      if current_chunk:
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          for _ in range(10):
            random_document_index = rng.randint(0, len(documents) - 1)
            if random_document_index != di:
              break

          random_document = documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break

          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, tokenizer, rng, list(tokenizer.vocab.keys()))
        instances.append({
          "tokens": tokens,
          "segment_ids": segment_ids,
          "masked_lm_positions": masked_lm_positions,
          "masked_lm_labels": masked_lm_labels,
          "is_random_next": is_random_next
        })
      current_chunk = []
      current_length = 0
    i += 1
  return instances

def get_documents(rng, tokenizer, fn):
  documents = [[]]
  with open(BASEDIR / fn) as f:
    for line in f.readlines():
      if not (line := line.decode("utf-8", "ignore") if isinstance(line, bytes) else line): break
      if not (line := line.strip()): documents.append([])
      if (tokens := tokenizer.tokenize(line)): documents[-1].append(tokens)
  documents = [x for x in documents if x]
  rng.shuffle(documents)
  return documents

def get_instances(rng, tokenizer, documents):
  instances = []
  for i in range(getenv('DUPE_FACTOR', 10)):
    for di, doc in enumerate(documents):
      instances.extend(create_instances_from_document(rng, tokenizer, doc, di, documents))
  rng.shuffle(instances)
  return instances

def instance_to_features(instance, tokenizer):
  input_ids = tokenizer.convert_tokens_to_ids(instance["tokens"])
  input_mask = [1] * len(input_ids)
  segment_ids = instance["segment_ids"]

  max_seq_length = getenv('MAX_SEQ_LENGTH', 128)

  assert len(input_ids) <= max_seq_length
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  masked_lm_positions = instance["masked_lm_positions"]
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance["masked_lm_labels"])
  masked_lm_weights = [1.0] * len(masked_lm_ids)

  while len(masked_lm_positions) < getenv("MAX_PREDICTIONS_PER_SEQ", 20):
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  next_sentence_label = 1 if instance["is_random_next"] else 0

  return {
    "input_ids": np.expand_dims(np.array(input_ids, dtype=np.int32), 0),
    "input_mask": np.expand_dims(np.array(input_mask, dtype=np.int32), 0),
    "segment_ids": np.expand_dims(np.array(segment_ids, dtype=np.int32), 0),
    "masked_lm_positions": np.expand_dims(np.array(masked_lm_positions, dtype=np.int32), 0),
    "masked_lm_ids": np.expand_dims(np.array(masked_lm_ids, dtype=np.int32), 0),
    "masked_lm_weights": np.expand_dims(np.array(masked_lm_weights, dtype=np.float32), 0),
    "next_sentence_labels": np.expand_dims(np.array([next_sentence_label], dtype=np.int32), 0),
  }

def process_part(part):
  tokenizer = Tokenizer(Path(__file__).parent / "wiki" / "vocab.txt")
  os.makedirs(BASEDIR / "train" / str(part), exist_ok=True)
  for i, features in enumerate(process_iterate(tokenizer, val=False, part=part)):
    with open(BASEDIR / f"train/{str(part)}/{part}_{i}.pkl", "wb") as f:
      pickle.dump(features, f)

def process_iterate(tokenizer, val=False, part=0): # Convert raw text to masked NSP samples
  rng = random.Random(getenv('SEED', 12345))

  if val:
    documents = get_documents(rng, tokenizer, "results4/eval.txt")
    instances = get_instances(rng, tokenizer, documents)
    print(f"there are {len(instances)} samples in the dataset")

    print(f"picking 10000 samples")
    pick_ratio = len(instances) // 10000
    for i in range(10000):
      instance = instances[i * pick_ratio]
      features = instance_to_features(instance, tokenizer)
      yield features
  else:
    # part padded to 5 digits
    documents = get_documents(rng, tokenizer, f"results4/part-{part:05d}-of-00500")
    instances = get_instances(rng, tokenizer, documents)

    for instance in instances:
      features = instance_to_features(instance, tokenizer)
      yield features

##################### Data loading #####################

@functools.lru_cache(None)
def get_val_files():
  return sorted(list((BASEDIR / "eval/").glob("*.pkl")))

@diskcache
def get_train_files():
  return sorted(list((BASEDIR / "train/").glob("*/*.pkl")))

def load_bert_file(file_name):
  with open(file_name, "rb") as f:
    features = pickle.load(f)
    return {
        "input_ids": features["input_ids"],
        "input_mask": features["input_mask"],
        "segment_ids": features["segment_ids"],
        "masked_lm_positions": features["masked_lm_positions"],
        "masked_lm_ids": features["masked_lm_ids"],
        "next_sentence_labels": features["next_sentence_labels"],
    }

def iterator(BS:int, val=False, start=0):
  from extra.datasets.wikipedia import get_train_files, get_val_files
  files = get_val_files() if val else get_train_files()
  with Pool() as p:
    i = start
    while True:
      results = p.map(load_bert_file, files[i:i+BS])
      yield {
        "input_ids": np.concatenate([f["input_ids"] for f in results], axis=0),
        "input_mask": np.concatenate([f["input_mask"] for f in results], axis=0),
        "segment_ids": np.concatenate([f["segment_ids"] for f in results], axis=0),
        "masked_lm_positions": np.concatenate([f["masked_lm_positions"] for f in results], axis=0),
        "masked_lm_ids": np.concatenate([f["masked_lm_ids"] for f in results], axis=0),
        "masked_lm_weights": np.concatenate([f["masked_lm_weights"] for f in results], axis=0),
        "next_sentence_labels": np.concatenate([f["next_sentence_labels"] for f in results], axis=0),
      }
      i = (i + BS) % len(files)

if __name__ == "__main__":
  tokenizer = Tokenizer(Path(__file__).parent / "wiki" / "vocab.txt")

  if len(sys.argv) <= 1:
    X, Y = next(iterator(val=False))
    print("Input Ids:\n", X["input_ids"][0])
    print("Tokens:\n", tokenizer.convert_ids_to_tokens(X["input_ids"][0]))
    print("Masked token ids:\n", Y["masked_lm_ids"])
    print("Masked tokens:\n", tokenizer.convert_ids_to_tokens(Y["masked_lm_ids"][0]))

    # fill in the blanks
    for i in range(getenv('MAX_PREDICTIONS_PER_SEQ', 20)):
      X["input_ids"][0][int(X["masked_lm_positions"][0][i])] = Y["masked_lm_ids"][0][i]
    print(" ".join(tokenizer.convert_ids_to_tokens(X["input_ids"][0])))
  else:
    if sys.argv[1] == "pre-eval":
      os.makedirs(BASEDIR / "eval", exist_ok=True)
      for i, features in tqdm(enumerate(process_iterate(tokenizer, val=True)), total=10000):
        with open(BASEDIR / f"eval/{i}.pkl", "wb") as f:
          pickle.dump(features, f)
    elif sys.argv[1] == "pre-train":
      os.makedirs(BASEDIR / "train", exist_ok=True)
      if sys.argv[2] == "all":
        process_map(process_part, [part for part in range(500)], max_workers=getenv('NUM_WORKERS', os.cpu_count()), chunksize=1)
      else:
        part = int(sys.argv[2])
        os.makedirs(BASEDIR / "train" / str(part), exist_ok=True)
        for i, features in tqdm(enumerate(process_iterate(tokenizer, val=False, part=part))):
          with open(BASEDIR / f"train/{str(part)}/{part}_{i}.pkl", "wb") as f:
            pickle.dump(features, f)
