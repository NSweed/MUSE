#imports
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import gc


class AbstractionBetweenGraphs:

    def __init__(self, threshold, do_reverse = True):

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(f"Device is {self.device}")
        #self.device = "cpu"
        self.get_nli_model()
        self.label_mapping = ["entailment", "neutral", "contradiction"]

        self.threshold = threshold
        self.softmax = torch.nn.Softmax()
        self.do_reverse = do_reverse

    def update_graph_nodes(self,graph1_nodes, graph2_nodes):
        self.graph1_nodes = graph1_nodes
        self.graph2_nodes = graph2_nodes
        self.graph1_sentences = self.form_sentences(self.graph1_nodes)
        self.graph2_sentences = self.form_sentences(self.graph2_nodes)

    def check_entailment_relations(self):
        # Initialize node_to_neighbor dictionary
        node_to_neighbor = {}
        for node in self.graph1_nodes:
            node_to_neighbor[node] = []
        for node in self.graph2_nodes:
            node_to_neighbor[node] = []

        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # # Pre-compute sentences lists
        graph2_sentences = [self.graph2_sentences[x] for x in self.graph2_sentences]
        graph1_sentences = [self.graph1_sentences[x] for x in self.graph1_sentences]

        # Set batch size
        bs = 100   # Adjust this based on your GPU memory
        for node in self.graph1_nodes:
            # Create sentences for current node
            node_sentences = [self.graph1_sentences[node]] * len(graph2_sentences)

            # Calculate number of batches
            n_batches = (len(graph2_sentences) + bs - 1) // bs

            batch_labels = []
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, len(graph2_sentences))

                # Process batch
                with torch.cuda.amp.autocast(enabled=True):  # Enable automatic mixed precision
                    inputs = self.tokenizer(
                        node_sentences[start_idx:end_idx],
                        graph2_sentences[start_idx:end_idx],
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=40
                    ).to(self.device)

                    # Forward pass
                    with torch.no_grad():  # Disable gradient computation for inference
                        output = self.nli_model(inputs["input_ids"])
                        scores = torch.softmax(output["logits"], -1)

                    # Process scores based on threshold
                    if self.threshold:
                        scores_np = np.array(self.softmax(scores.cpu()))
                        max_scores = scores_np.max(axis=1)
                        batch_labels.extend([
                            self.label_mapping[score_max] if max_scores[i] > self.threshold else "neutral"
                            for i, score_max in enumerate(scores_np.argmax(axis=1))
                        ])
                    else:
                        batch_labels.extend([
                            self.label_mapping[score_max.item()]
                            for score_max in scores.argmax(axis=1)
                        ])

                    # Clear memory after each batch
                    del inputs, output, scores
                    if self.threshold:
                        del scores_np, max_scores

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Process entailment relations for the node
            for i, label in enumerate(batch_labels):
                if label == "entailment":
                    node_to_neighbor[node].append(self.graph2_nodes[i])

            # Clear batch labels
            del batch_labels
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return node_to_neighbor

    # def check_entailment_relations(self):
    #     node_to_neighbor = {}
    #     for node in self.graph1_nodes:
    #         node_to_neighbor[node] = []
    #     for node in self.graph2_nodes:
    #         node_to_neighbor[node] = []
    #     graph2_sentences = [self.graph2_sentences[x] for x in self.graph2_sentences]
    #     graph1_sentences = [self.graph1_sentences[x] for x in self.graph1_sentences]
    #     for node in self.graph1_nodes:
    #         node_sentences = [self.graph1_sentences[node] for _ in range(len(self.graph2_sentences))]
    #
    #         inputs = self.tokenizer(node_sentences, graph2_sentences,
    #                                 return_tensors="pt", truncation=True,
    #                                 padding="max_length", max_length=40).to(self.device)
    #         output = self.nli_model(inputs["input_ids"])
    #         scores = torch.softmax(output["logits"], -1)
    #
    #         if self.threshold:
    #             scores = np.array(self.softmax(torch.tensor(scores)).cpu())
    #             max_scores = scores.max(axis=1)
    #             labels = [self.label_mapping[score_max] if max_scores[i] > self.threshold else "netural" for
    #                       i, score_max in enumerate(scores.argmax(axis=1))]
    #         else:
    #             labels = [self.label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    #         for i,label in enumerate(labels):
    #             if label == "entailment":
    #                 node_to_neighbor[node].append(self.graph2_nodes[i])
    #         del inputs, output, scores
    #     if self.do_reverse:
    #         for node in self.graph2_nodes:
    #             node_sentences = [self.graph2_sentences[node] for _ in range(len(self.graph1_sentences))]
    #
    #             inputs = self.tokenizer(node_sentences, graph1_sentences,
    #                                     return_tensors="pt", truncation=True,
    #                                     padding="max_length", max_length=40).to(self.device)
    #             output = self.nli_model(inputs["input_ids"])
    #             scores = torch.softmax(output["logits"], -1)
    #             if self.threshold:
    #                 scores = np.array(self.softmax(torch.tensor(scores)).cpu())
    #                 max_scores = scores.max(axis=1)
    #                 labels = [self.label_mapping[score_max] if max_scores[i] > self.threshold else "netural" for
    #                           i, score_max in enumerate(scores.argmax(axis=1))]
    #             else:
    #                 labels = [self.label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    #             for i,label in enumerate(labels):
    #                 if label == "entailment":
    #                     node_to_neighbor[node].append(self.graph1_nodes[i])
    #     return node_to_neighbor






    def get_nli_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def form_sentences(self, nodes):
        sentences = {}
        for node in nodes:
            sentences[node] = self.form_sentence(node.get_random_point())
        return sentences


    def form_sentence(self, sentence):
        return f"I want {sentence}"

class AbstractionWithClusters:

    def __init__(self, clusters, prefix, threshold  = None):
        self.clusters = clusters
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.get_nli_model()
        self.prefix = prefix
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.form_all_sentences()
        self.threshold = threshold
        self.softmax = torch.nn.Softmax()


    def update_clusters(self, new_clusters):
        self.clusters = new_clusters
        self.form_all_sentences()


    def get_nli_model(self):
        if type == "nli_deberta":
            self.nli_model =  CrossEncoder('cross-encoder/nli-deberta-base', device=torch.cuda.current_device())



    def form_sentence(self, sentence):
        return f"{self.prefix} {sentence}"

    def form_all_sentences(self):
        self.formed_sentences = [self.form_sentence(cluster.get_random_point()) for cluster in self.clusters ]


    def check_entailment_for_cluster(self, cluster, low = 0):
        sentence1 = self.form_sentence(cluster.get_random_point())
        batch_formed_sentences = self.formed_sentences[low:]
        sentences_to_check = [(sentence1, sentence2) for sentence2 in batch_formed_sentences]
        scores = self.nli_model.predict(sentences_to_check)
        scores = np.array(self.softmax(torch.tensor(scores)))
        if self.threshold:
            max_scores = scores.max(axis = 1)
            labels = [(self.label_mapping[score_max], max_scores[i]) if max_scores[i] > self.threshold else ("netural", 0) for i,score_max in enumerate(scores.argmax(axis = 1))]
        else:
            labels = [(self.label_mapping[score_max], score_max) for score_max in scores.argmax(axis = 1)]
        return labels

    def find_abstractions_with_nli(self):
        import gc
        cluster_to_more_abstract = {}
        cluster_to_less_abstract = {}
        edge_to_weight = {}
        for j,cluster in enumerate(self.clusters):
            #print(f"starting with the {j}-th iteration of entailment checking")
            entailment_labels = self.check_entailment_for_cluster(cluster)
            # Clear memory after processing each cluster
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            for i,label in enumerate(entailment_labels):
                if label[0] == "entailment" and i != j:
                    current_cluster = self.clusters[i]
                    if current_cluster not in cluster_to_less_abstract:
                        cluster_to_less_abstract[current_cluster] = []
                    cluster_to_less_abstract[current_cluster].append(cluster)
                    edge_to_weight[(cluster, current_cluster)] = label[1]
                    if cluster not in cluster_to_more_abstract:
                        cluster_to_more_abstract[cluster] = []
                    cluster_to_more_abstract[cluster].append(current_cluster)
        cluster_to_more_abstract = self.add_no_neighbor_clusters(cluster_to_more_abstract)
        cluster_to_less_abstract = self.add_no_neighbor_clusters(cluster_to_less_abstract)
        return cluster_to_more_abstract, cluster_to_less_abstract, edge_to_weight

    def add_no_neighbor_clusters(self, d):
        for cluster in self.clusters:
            if cluster not in d:
                d[cluster] = []
        return d

    def check_clusters(self, cluster1, cluster2):
        sentence1 = self.form_sentence(cluster1.get_random_point())
        sentence2 = self.form_sentence(cluster2.get_random_point())

class AbstractionWithClustersMNLI(AbstractionWithClusters):


    def __init__(self, clusters, prefix, threshold  = None):
        super().__init__(clusters, prefix,threshold)
        self.label_mapping = ["entailment", "neutral", "contradiction"]
    def get_nli_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        #self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(torch.cuda.current_device())



    def check_entailment_for_cluster(self, cluster, low = 0, high = None):
        print("in the right entailment function")
        print(f"the device is really {self.device}")
        sentence1 = self.form_sentence(cluster.get_random_point())
        if high:
            batch_formed_sentences = self.formed_sentences[low:high]
        else:
            batch_formed_sentences = self.formed_sentences[low:]
        first_sentences = [sentence1 for _ in range(len(batch_formed_sentences))]
        labels = []
        bs = 50
        steps = len(batch_formed_sentences) // bs
        remainder = len(batch_formed_sentences) % bs
        #(f"total is {len(batch_formed_sentences)}steps is {steps} and remainder is {remainder}")
        for it in range(steps):
            #print(f"currently in iteration {it+1}")
            inputs = self.tokenizer(first_sentences[it*bs:(it+1)*bs], batch_formed_sentences[it*bs:(it+1)*bs],
                                    return_tensors="pt", truncation = True, padding="max_length", max_length = 40).to(self.device)
            #inputs.to(torch.cuda.current_device())
            # inputs = self.tokenizer(first_sentences, batch_formed_sentences, truncation = True, return_tensors  = "pt", padding = True)\
            #     .to(torch.cuda.current_device())
            output = self.nli_model(inputs["input_ids"])
            scores = torch.softmax(output["logits"], -1)
            max_scores, argmax_scores = scores.max(axis=1)
            # Convert to CPU immediately to free GPU memory
            current_labels = [
                (self.label_mapping[score_max.item()], max_scores[i].item())
                if not self.threshold or max_scores[i] > self.threshold
                else ("neutral", 0)
                for i, score_max in enumerate(argmax_scores)
            ]
            labels.extend(current_labels)

            # Clear memory after each batch
            del inputs, output, scores, max_scores, argmax_scores
            torch.cuda.empty_cache()
            # if self.threshold:
            #
            #     labels.extend([(self.label_mapping[score_max], max_scores[i].item()) if max_scores[i] > self.threshold else ("netural", 0)
            #                    for i,score_max in enumerate(argmax_scores)])
            # else:
            #     labels.extend([(self.label_mapping[score_max], score_max) for score_max in argmax_scores])
        #handle remainder
        if remainder != 0:
            inputs = self.tokenizer(first_sentences[steps * bs:], batch_formed_sentences[steps * bs:], return_tensors="pt",truncation = True,
                                    padding = "max_length", max_length = 40).to(self.device)
            #inputs.to(torch.cuda.current_device())
            # inputs = self.tokenizer(first_sentences, batch_formed_sentences, truncation = True, return_tensors  = "pt", padding = True).to(torch.cuda.current_device())
            output = self.nli_model(inputs["input_ids"])

            scores = torch.softmax(output["logits"], -1)
            max_scores, argmax_scores = scores.max(axis=1)
            current_labels = [
                (self.label_mapping[score_max.item()], max_scores[i].item())
                if not self.threshold or max_scores[i] > self.threshold
                else ("neutral", 0)
                for i, score_max in enumerate(argmax_scores)
            ]
            labels.extend(current_labels)

            # Clear memory after remainder
            del inputs, output, scores, max_scores, argmax_scores
            torch.cuda.empty_cache()
        return labels