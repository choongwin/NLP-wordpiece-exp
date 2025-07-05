from tokenizers import Tokenizer, models, trainers,pre_tokenizers,decoders
import json 

tokenizer = Tokenizer(models.WordPiece(unk_token = "[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

corpus = []
with open(r"pubmed_sampled_corpus.jsonline","r") as f:
    for line in f:
        corpus.append(json.loads(line)["text"])
        
trainer = trainers.WordPieceTrainer(
    vocab_size = 50000,
    special_tokens =["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"],
)
tokenizer.train_from_iterator(corpus, trainer)

tokenizer.save("biometical_tokenizer_50000.json")
