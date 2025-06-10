import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        # W1 in Equation (10)
        self.linear1 = nn.Linear(self.in_dim, self.out_dim)
        # W2 in Equation (10)
        self.linear2 = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)



    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_patients + n_entities, in_dim)
        A_in:            (n_patients + n_entities, n_patients + n_entities), torch.sparse.FloatTensor
        """
        # Equation (7)
        side_embeddings = torch.matmul(A_in, ego_embeddings)
        # Equation (10)
        sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
        bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
        embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)  # (n_patients + n_entities, out_dim)
        return embeddings


class MODEL(nn.Module):
    """
       Main model architecture that combines Patient-Disease Bipartite Graph and Local Knowledge Graph information
       using graph neural networks for recommendation.
    """
    def __init__(self, args,
                 n_patients, n_entities, n_relations, A_in=None,
                 patient_pre_embed=None, disease_pre_embed=None):

        super(MODEL, self).__init__()
        self.patient_pretrain = args.patient_pretrain
        self.adversarial_temperature = args.adversarial_temperature
        self.beta = args.beta
        self.gamma = args.gamma
        self.__softplus = nn.Softplus(beta=self.beta)

        self.__softmax = nn.Softmax(dim=-1)

        self.n_patients = n_patients
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)

        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_patient_embed = nn.Embedding(self.n_patients + self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        if (self.patient_pretrain == 1) and (patient_pre_embed is not None) and (disease_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_patients_entities - disease_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_patient_embed = torch.cat([disease_pre_embed, other_entity_embed, patient_pre_embed], dim=0)
            self.entity_patient_embed.weight = nn.Parameter(entity_patient_embed)

        else:
            nn.init.xavier_uniform_(self.entity_patient_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)


        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_patients + self.n_entities, self.n_patients + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False


    def calc_cf_embeddings(self):
        ego_embed = self.entity_patient_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (13) embeddings concatenation
        all_embed = torch.cat(all_embed, dim=1)
        return all_embed


    def calc_cf_loss(self, patient_ids, disease_pos_ids, disease_neg_ids):
        """
        patient_ids:       (cf_batch_size)
        disease_pos_ids:   (cf_batch_size)
        disease_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()
        patient_embed = all_embed[patient_ids]
        disease_pos_embed = all_embed[disease_pos_ids]
        disease_neg_embed = all_embed[disease_neg_ids]

        # Equation (14) y=u*i
        pos_score = torch.sum(patient_embed * disease_pos_embed, dim=1)
        neg_score = torch.sum(patient_embed * disease_neg_embed, dim=1)

        # Equation (15)   Lcf
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(patient_embed) + _L2_loss_mean(disease_pos_embed) + _L2_loss_mean(disease_neg_embed) #模型参数主要来源于实体嵌入
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """

        r_embed = self.relation_embed(r)  # (kg_batch_size, embed_dim)
        h_embed = self.entity_patient_embed(h)  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_patient_embed(pos_t)  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_patient_embed(neg_t)  # (kg_batch_size, embed_dim)

        # Embedding Model
        # Split embeddings into real and imaginary parts for complex space representation
        h_re, h_im = torch.chunk(h_embed, 2, dim=1)
        pos_t_re, pos_t_im = torch.chunk(pos_t_embed, 2, dim=1)
        neg_t_re, neg_t_im = torch.chunk(neg_t_embed, 2, dim=1)

        # Reshape negative samples for self-adversarial training
        neg_t_re = neg_t_re.view(h_embed.size(0), -1, self.embed_dim // 2)
        neg_t_im = neg_t_im.view(h_embed.size(0), -1, self.embed_dim // 2)

        r_re, r_im = torch.chunk(r_embed, 2, dim=1)

        h_re_rot = h_re * r_re - h_im * r_im
        h_im_rot = h_re * r_im + h_im * r_re

        pos_score = torch.sum((h_re_rot - pos_t_re) ** 2 + (h_im_rot - pos_t_im) ** 2, dim=1)


        h_re_rot_expanded = h_re_rot.unsqueeze(1).expand(-1, neg_t_re.size(1), -1)
        h_im_rot_expanded = h_im_rot.unsqueeze(1).expand(-1, neg_t_im.size(1), -1)

        neg_score = torch.sum((h_re_rot_expanded - neg_t_re) ** 2 + (h_im_rot_expanded - neg_t_im) ** 2, dim=2)

        pos_score = self.gamma - pos_score
        neg_score = neg_score - self.gamma

        # Equation (6) KG loss
        pos_score = self.__softplus(pos_score)
        if self.adversarial_temperature:
            neg_score =  (F.softmax(neg_score * self.adversarial_temperature, dim=1).detach()
                                  * self.__softplus(neg_score)).sum(dim=1)
        else:
            neg_score = self.__softplus(neg_score).mean(dim=1)

        positive_sample_loss = pos_score.mean()
        negative_sample_loss = neg_score.mean()
        kg_loss = (positive_sample_loss + negative_sample_loss)/2

        l2_loss = (
                _L2_loss_mean(h_re_rot) + _L2_loss_mean(h_im_rot) +
                _L2_loss_mean(r_embed) +
                _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        )
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx].unsqueeze(0)
        h_embed = self.entity_patient_embed.weight[h_list]
        t_embed = self.entity_patient_embed.weight[t_list]

        r_re, r_im = torch.chunk(r_embed, 2, dim=1)
        h_re, h_im = torch.chunk(h_embed, 2, dim=1)
        t_re, t_im = torch.chunk(t_embed, 2, dim=1)

        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Equation (8) attention mechanism
        hr_complex = torch.stack((hr_re, hr_im), dim=-1)
        dot_product_real = torch.sum(hr_complex[..., 0] * t_re - hr_complex[..., 1] * t_im, dim=-1)
        v_list = dot_product_real
        return v_list
    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device
        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)
        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (9)  attention normalization
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)



    def calc_score(self, patient_ids, disease_ids):
        """
        patient_ids:  (n_patients)
        disease_ids:  (n_diseases)
        """
        all_embed = self.calc_cf_embeddings()
        patient_embed = all_embed[patient_ids]
        disease_embed = all_embed[disease_ids]

        # Equation (14)  predict
        cf_score = torch.matmul(patient_embed, disease_embed.transpose(0, 1))
        return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


