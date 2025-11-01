import json
import os
from ClusterFactory import Cluster
from Serializer import Serializer

class Mechanism:

    def __init__(self, text,id):
        self.text = text
        self.id = id
    def get_text(self):
        return self.text
    def get_id(self):
        return self.id()


class MechanismReader:

    def create_clustered_mechnisms_dict(self, fname):
        with open(fname) as f:
            return_d = {}
            mech_id_to_mech = {}
            patent_id_to_mechanisms = json.load(f)
            for patent_id in patent_id_to_mechanisms:
                return_d[patent_id]  = []
                for d in patent_id_to_mechanisms[patent_id]:
                    mech_str = list(d.keys())[0]
                    id = d[mech_str]
                    if id not in mech_id_to_mech:
                        end_ind = mech_str.find("$")
                        mech_text = mech_str[:end_ind]
                        cur_mech  = Cluster([mech_text], [],id)
                        mech_id_to_mech[id] = cur_mech
                    return_d[patent_id].append(mech_id_to_mech[id])
            return return_d

    def load_raw_mechanism_dict(self, fname):
        from Serializer import Serializer
        return Serializer.load_dict(fname)


class PurposeReader:
    START_REG = "The purpose of the patent is to provide "
    START_IND = len("The purpose of the patent is to provide ")
    NO_PROVIDE_LEN = len("The purpose of the patent is to ")
    NO_PROVIDE_REG = "The purpose of the patent is to "


    def __init__(self, type, path):
        self.type = type
        self.path = path

    def read_json_file(self,fname):
        with open(fname) as f:
            line = f.readlines()[0]
            d = json.loads(line)
            return d

    def extract_purpose_from_text(self,text):
        last_ind = text.find(".")
        count = 0
        if self.START_REG not in text:
            count += 1
            purp = f"to {text[self.NO_PROVIDE_LEN:last_ind]}"
            #print(text)
        else:
            if  self.NO_PROVIDE_REG not in text:
                print("issue")
            purp = text[self.START_IND:last_ind]

        #print(f"missing {count} to provides")
        return purp



    def update_dict_from_dict(self,base, d):
        for x in d:
            id  = x["id"]
            # if id not in base:
            #     base[id] = []
            base[id] = self.extract_purpose_from_text(x["response"])
        return base


    def create_purpose_dict(self):
        purpose_dict = {}
        if self.type == "GPT3":
            for fname in os.listdir(self.path):
                d = self.read_json_file(os.path.join(self.path,fname))
                purpose_dict = self.update_dict_from_dict(purpose_dict, d)
        elif self.type == "CPC":
            purpose_dict = Serializer.load_dict(self.path)
        return purpose_dict




# # text = "The purpose of the patent is to provide a device for inserting an object into a cavity. The context of the patent is medical devices."
# # extract_purpose_from_text(text)
# # fname = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\nwq_patent_tags_0_1000_2021_12_05_06_27_37.json"
# # read_json_file(fname)
# dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
# PR = PurposeReader()
# purpose_dict = PR.create_purpose_dict(dir)
# x = 1

if __name__ == "__main__":
    path = os.path.join("bert_based_info","pid_to_cpc_purpose_clusters_cache_file.json")
    mech_reader = MechanismReader()
    mech_reader.read_json_file(path)
