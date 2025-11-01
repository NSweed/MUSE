import torch
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import pickle
import os


class SaveSentenceEmbeddings:

    def __init__(self,all_sentences, batch_size):
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        self.sbert_model = SentenceTransformer('stsb-roberta-base', device=device)
        self.all_sentences = all_sentences
        self.batch_size = batch_size

    def add_batch_encodings(self,d, batch, offset):
        embs = self.sbert_model.encode(batch)
        for i in range(len(batch)):
            d[i + offset] =  embs[i]


    def save_embeddings_dict(self,path):
        iterations = len(self.all_sentences) // self.batch_size
        remainder = len(self.all_sentences) % self.batch_size
        print(f"iterations: {iterations}, remainder: {remainder}")
        cur_idx = 0
        d = {}
        offset = 0
        for i in range(iterations):
            print(f"started batch {i}")
            batch = self.all_sentences[cur_idx : cur_idx + self.batch_size]
            cur_idx += self.batch_size
            self.add_batch_encodings(d, batch, offset)
            print(f"current size of d is {len(d.keys())}")
            offset += len(batch)
        #adding the last batch
        if remainder != 0:
            self.add_batch_encodings(d, self.all_sentences[cur_idx:], offset)
        with open(path,"wb") as f:
            pickle.dump(d,f)


class SaveOrLoadEmbeddings:

        def __init__(self, all_sentences, batch_size, purpose_type, save_dir):
            self.all_sentences = all_sentences
            self.batch_size = batch_size
            self.purpose_type = purpose_type
            print(f"the batch size is {self.batch_size}")
            self.save_dir = save_dir

        def save_or_load_embeddings(self, size):
            path = self.create_path(size)
            print(f"the path is {path}")
            if os.path.exists(path):
                d =  self.load_embeddings(path)
                print("loaded found embeddings")
                print(len(d.keys()))
            else:
                print("saving new embeddings")
                sentences = self.all_sentences[:size]
                SSE = SaveSentenceEmbeddings(sentences, self.batch_size)
                SSE.save_embeddings_dict(path)
                d =  self.load_embeddings(path)
            index_to_purpose, value_list = self.convert_dict(d)
            return d, index_to_purpose, value_list

        def convert_dict(self,d):
            index_to_sentence = {}
            value_list = []
            for i, x in enumerate(d):
                index_to_sentence[i] = self.all_sentences[i]
                value_list.append(d[x])
            return index_to_sentence, value_list

        def create_path(self, size):
            path = os.path.join(f"{self.save_dir}",f"sentence_embeddings_{size}_{self.purpose_type}.pkl")
            return path

        def load_embeddings(self, path):
            with open(path, "rb") as f:
                return pickle.load(f)