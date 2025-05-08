"""Demo script that prints embeddings for two sample sentences."""

from app.models import BertEmbedder


def main():
    sentences = [
        "There is a stack of papers on the table.",
        "The largest mountain in the world is Mount Everest.",
    ]

    embedder = BertEmbedder()
    embeddings = embedder.encode(sentences)

    for sent, emb in zip(sentences, embeddings):
        print(f"sentence: {sent}\nembedding: {emb.tolist()}\n")


if __name__ == "__main__":
    main() 