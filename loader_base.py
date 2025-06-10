import os
import random
import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    def __init__(self, args, logging):
   
        self.args = args
        self.data_name = args.data_name
        self.patient_pretrain = args.patient_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        # Set up data directories and file paths
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train1_patient.txt')
        self.test_file = os.path.join(self.data_dir, 'test1_patient.txt')
        self.kg_file = os.path.join(self.data_dir, "mimic_final_kg.txt")

        # Load training and test interaction data
        self.cf_train_data, self.train_patient_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_patient_dict = self.load_cf(self.test_file)

        self.statistic_cf()

        if self.patient_pretrain == 1:
            self.load_pretrained_data()

    def load_cf(self, filename):
        patient = []
        disease = []
        patient_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                patient_id, disease_ids = inter[0], inter[1:]
                disease_ids = list(set(disease_ids))

                for disease_id in disease_ids:
                    patient.append(patient_id)
                    disease.append(disease_id)
                patient_dict[patient_id] = disease_ids

        patient = np.array(patient, dtype=np.int32)
        disease = np.array(disease, dtype=np.int32)
        return (patient, disease), patient_dict

    def statistic_cf(self):
        self.n_patients = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_diseases = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

    def load_kg(self, filename):

        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def sample_pos_diseases_for_u(self, patient_dict, patient_id, n_sample_pos_diseases):

        pos_diseases = patient_dict[patient_id]
        n_pos_diseases = len(pos_diseases)

        sample_pos_diseases = []
        while True:
            if len(sample_pos_diseases) == n_sample_pos_diseases:
                break

            pos_disease_idx = np.random.randint(low=0, high=n_pos_diseases, size=1)[0]
            pos_disease_id = pos_diseases[pos_disease_idx]
            if pos_disease_id not in sample_pos_diseases:
                sample_pos_diseases.append(pos_disease_id)
        return sample_pos_diseases

    def sample_neg_diseases_for_u(self, patient_dict, patient_id, n_sample_neg_diseases):
        pos_diseases = patient_dict[patient_id]

        sample_neg_diseases = []
        while True:
            if len(sample_neg_diseases) == n_sample_neg_diseases:
                break

            neg_disease_id = np.random.randint(low=0, high=self.n_diseases, size=1)[0]
            if neg_disease_id not in pos_diseases and neg_disease_id not in sample_neg_diseases:
                sample_neg_diseases.append(neg_disease_id)
        return sample_neg_diseases

    def generate_cf_batch(self, patient_dict, batch_size):
        exist_patients = list(patient_dict.keys())
        if batch_size <= len(exist_patients):
            batch_patient = random.sample(exist_patients, batch_size)
        else:
            batch_patient = [random.choice(exist_patients) for _ in range(batch_size)]

        batch_pos_disease, batch_neg_disease = [], []
        for u in batch_patient:
            batch_pos_disease += self.sample_pos_diseases_for_u(patient_dict, u, 1)
            batch_neg_disease += self.sample_neg_diseases_for_u(patient_dict, u, 1)

        batch_patient = torch.LongTensor(batch_patient)
        batch_pos_disease = torch.LongTensor(batch_pos_disease)
        batch_neg_disease = torch.LongTensor(batch_neg_disease)
        return batch_patient, batch_pos_disease, batch_neg_disease

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        head_triples = kg_dict.get(head, [])
        num_triples = len(head_triples)

        if num_triples == 0:
            return [], []

        sample_relations = []
        sample_pos_tails = []

        seen_relations = set()
        seen_tails = set()

        attempts = 0
        max_attempts = num_triples * 2

        while len(sample_relations) < n_sample_pos_triples and attempts < max_attempts:
            random_idx = np.random.randint(0, num_triples)
            tail, relation = head_triples[random_idx]

            if relation not in seen_relations and tail not in seen_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
                seen_relations.add(relation)
                seen_tails.add(tail)

            attempts += 1
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        head_triples = kg_dict.get(head, [])
        forbidden_tails = {triple[0] for triple in head_triples if triple[1] == relation}

        sample_neg_tails = []
        seen_negatives = set()

        attempts = 0
        max_attempts = highest_neg_idx * 2

        while len(sample_neg_tails) < n_sample_neg_triples and attempts < max_attempts:
            random_tail = np.random.randint(0, highest_neg_idx)

            if random_tail not in forbidden_tails and random_tail not in seen_negatives:
                sample_neg_tails.append(random_tail)
                seen_negatives.add(random_tail)

            attempts += 1
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []

        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)

            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 3, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)

        pretrain_data = np.load(pretrain_path)
        self.patient_pre_embed = pretrain_data['patient_embed']
        self.disease_pre_embed = pretrain_data['disease_embed']

        assert self.patient_pre_embed.shape[0] == self.n_patients
        assert self.disease_pre_embed.shape[0] == self.n_diseases
        assert self.patient_pre_embed.shape[1] == self.args.embed_dim
        assert self.disease_pre_embed.shape[1] == self.args.embed_dim