"""Small synthetic dataset in the challenge layout, for end-to-end tests.

Source 1 holds clean entities; Source 2/3 hold noisy copies (abbreviations,
typos, dropped suffixes and postcodes, reordered words) plus distractors that
share words with real entities. The test split adds a country unseen in train.
"""
import random
from pathlib import Path

WORDS = ["sunrise", "global", "apex", "lotus", "metro", "silver", "prime", "royal", "green",
         "star", "delta", "crystal", "golden", "united", "pioneer", "vista", "nova", "urban",
         "shree", "ganesh", "balaji", "bharat", "lumiere", "soleil", "riviera", "eiffel"]
KINDS = ["Traders", "Foods", "Textiles", "Motors", "Pharma", "Logistics", "Electronics",
         "Bakery", "Hardware", "Consulting"]
SUFFIXES = {"US": [("Incorporated", "Inc"), ("Corporation", "Corp"), ("Company", "Co"), ("LLC", "LLC")],
            "India": [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Enterprises", "Ent")],
            "France": [("SARL", "SARL"), ("SAS", "SAS"), ("SA", "SA")]}
STREETS = ["Main", "Oak", "Park", "Station", "Temple", "Church", "Market", "Lake", "Hill", "Mill"]
TYPES = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave")]
CITIES = {"US": ["Austin", "Denver", "Boston"], "India": ["Pune", "Jaipur", "Mysuru"],
          "France": ["Lyon", "Nantes", "Rennes"]}


def _entity(rng, country):
    words = rng.sample(WORDS, 2)
    full, short = rng.choice(SUFFIXES[country])
    street, (typ, typ_short) = rng.choice(STREETS), rng.choice(TYPES)
    code = str(rng.randint(10000, 99999)) if country != "India" else str(rng.randint(110000, 859999))
    return {"words": words, "kind": rng.choice(KINDS), "suffix": (full, short),
            "num": str(rng.randint(1, 999)), "street": street, "type": (typ, typ_short),
            "city": rng.choice(CITIES[country]), "code": code, "country": country}


def _typo(rng, s):
    if len(s) < 4:
        return s
    k = rng.randrange(len(s) - 1)
    return s[:k] + s[k + 1] + s[k] + s[k + 2:]


def _render(rng, e, noisy):
    words = [w.title() for w in e["words"]]
    suffix = e["suffix"][0]
    typ = e["type"][0]
    code = e["code"]
    if noisy:
        if rng.random() < 0.3:
            words[0] = _typo(rng, words[0])
        if rng.random() < 0.3:
            words = words[::-1]
        suffix = rng.choice([e["suffix"][1], "", suffix])
        typ = rng.choice(e["type"])
        if rng.random() < 0.3:
            code = ""
    name = " ".join(words + [e["kind"], suffix]).strip()
    addr = f"{e['num']} {e['street']} {typ}, {e['city']} {code}".strip()
    if noisy and rng.random() < 0.3:
        name = name.upper()
    return name, addr


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def make_split(root: Path, split: str, countries, n_s1: int, seed: int) -> dict:
    """Write {split}_source{1,2,3}.tsv (+ ground truth for train); return the truth."""
    rng = random.Random(seed)
    s1, others, truth = [], [], {}
    for n in range(n_s1):
        e = _entity(rng, rng.choice(countries))
        sid = f"S1-{n:05d}"
        s1.append((sid, *_render(rng, e, noisy=False), e["country"]))
        truth[sid] = []
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
            others.append((sid, rng.choice((2, 3)), *_render(rng, e, noisy=True), e["country"]))
    for _ in range(n_s1 // 2):  # distractors: unrelated businesses in Source 2/3 only
        e = _entity(rng, rng.choice(countries))
        others.append((None, rng.choice((2, 3)), *_render(rng, e, noisy=True), e["country"]))
    rng.shuffle(others)
    rows = {2: [], 3: []}
    for owner, src, name, addr, country in others:
        oid = f"S{src}-{len(rows[src]):05d}"
        rows[src].append((oid, name, addr, country))
        if owner:
            truth[owner].append(oid)
    header = ["entity_id", "business_name", "business_address", "country"]
    base = root / split
    _write(base / f"{split}_source1.tsv", header, s1)
    for src in (2, 3):
        _write(base / f"{split}_source{src}.tsv", header, rows[src])
    if split == "train":
        _write(base / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
               [(k, ",".join(v)) for k, v in truth.items()])
    return truth


def make_dataset(root: Path, n_s1: int = 400, seed: int = 7) -> None:
    make_split(root, "train", ["US", "India"], n_s1, seed)
    make_split(root, "test", ["US", "India", "France"], n_s1 // 2, seed + 1)


def make_tiny_encoder(directory: Path, texts) -> None:
    """A randomly initialised one-layer BERT pair classifier and a word-level tokenizer over
    the words of `texts`, saved like a downloaded checkpoint (stands in for the real base
    encoder in tests; no network)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    split = pre_tokenizers.Whitespace()
    words = sorted({w for t in texts for w, _ in split.pre_tokenize_str(t)})
    vocab = {w: k for k, w in enumerate(specials + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = split
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]").save_pretrained(directory)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=160, num_labels=1)
    BertForSequenceClassification(cfg).save_pretrained(directory)
