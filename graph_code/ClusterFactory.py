import numpy as np
class Cluster:

    def __init__(self, points, point_embeddings, id, pids = None):
        self.points = points
        self.point_embeddings = point_embeddings
        self.id = id
        self.pids = pids

    def get_cluster_points(self):
        return self.points

    def get_id(self):
        return self.id

    def get_point_embeddings(self):
        return self.point_embeddings

    def get_random_point(self):
        #rand_int  = np.random.randint(0,len(self.points))
        return self.points[0]

    def update_id(self, new_id):
        self.id = new_id

    def get_pids(self):
        return self.pids



class ClusterFactory:

    @staticmethod
    def create_problem_clusters(index_to_label, index_to_sentence,index_to_pid, sentence_to_embedding, offset = 0):
        #creating a mapping between labels and sentences
        label_to_sentences = {}
        label_to_embeddings = {}
        label_to_pids = {}
        for index in index_to_sentence:
            label  = index_to_label[index]
            if label != -1:
                if label not in label_to_sentences:
                    label_to_sentences[label] = []
                    label_to_embeddings[label] = []
                    label_to_pids[label] = []
                sentence = index_to_sentence[index]
                label_to_sentences[label].append(sentence)
                label_to_embeddings[label].append(sentence_to_embedding[index])
                label_to_pids[label].append(index_to_pid[index])
        problem_clusters = []
        seen_mechs = set()
        cur_count = 0
        for label in label_to_sentences:
            cur_prob_cluster = Cluster(label_to_sentences[label], label_to_embeddings[label], cur_count + offset, label_to_pids[label] )
            problem_clusters.append(cur_prob_cluster)
            cur_count += 1
            # cur_mech_clusters  = label_to_mechs[label]
            # for mech in cur_mech_clusters:
            #     if mech not in seen_mechs:
            #         mech.update_id(cur_count + offset)
            #         cur_count += 1
            #     problem_cluster_to_mechanism_cluster[cur_prob_cluster].append(mech)



        max_count = cur_count +offset
        return problem_clusters, max_count
