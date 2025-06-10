import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from MDP_GRL import MODEL
from args import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from loader_MDP_GRL import DataLoaderModel


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_patient_dict = dataloader.train_patient_dict
    test_patient_dict = dataloader.test_patient_dict

    model.eval()


    patient_ids = list(test_patient_dict.keys())
    patient_ids_batches = [patient_ids[i: i + test_batch_size] for i in range(0, len(patient_ids), test_batch_size)]
    patient_ids_batches = [torch.LongTensor(d) for d in patient_ids_batches]

    n_diseases = dataloader.n_diseases
    disease_ids = torch.arange(n_diseases, dtype=torch.long).to(device)


    cf_scores = []
    metric_names = ['precision', 'recall', 'F1', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}


    with tqdm(total=len(patient_ids_batches), desc='Evaluating Iteration') as pbar:

        for batch_patient_ids in patient_ids_batches:
            batch_patient_ids = batch_patient_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_patient_ids, disease_ids, mode='predict')


            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_patient_dict, test_patient_dict, batch_patient_ids.cpu().numpy(), disease_ids.cpu().numpy(), Ks)


            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderModel(args, logging)
    if args.use_pretrain == 1:
        patient_pre_embed = torch.tensor(data.patient_pre_embed)
        disease_pre_embed = torch.tensor(data.disease_pre_embed)
    else:
        patient_pre_embed, disease_pre_embed = None, None


    # construct model & optimizer
    model = MODEL(args, data.n_patients, data.n_entities, data.n_relations, data.A_in, patient_pre_embed, disease_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [],'F1': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_patient, cf_batch_pos_disease, cf_batch_neg_disease = data.generate_cf_batch(data.train_patient_dict, data.cf_batch_size)
            cf_batch_patient = cf_batch_patient.to(device)
            cf_batch_pos_disease = cf_batch_pos_disease.to(device)
            cf_batch_neg_disease = cf_batch_neg_disease.to(device)

            cf_batch_loss = model(cf_batch_patient, cf_batch_pos_disease, cf_batch_neg_disease, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_patients_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict = evaluate(model, data, Ks, device)


            logging.info(
                'CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                    epoch, time() - time6,
                    metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                    metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'],
                    metrics_dict[k_min]['F1'], metrics_dict[k_max]['F1'],
                    metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'F1', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])

            best_ndcg, should_stop = early_stopping(metrics_list[k_min]['ndcg'], args.stopping_steps)

            if should_stop:
                break

            current_ndcg = metrics_list[k_min]['ndcg'][-1]
            if current_ndcg == best_ndcg:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'F1', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(os.path.join(args.save_dir, 'metrics.tsv'), sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()

    output_str = f"Best CF Evaluation: Epoch {int(best_metrics['epoch_idx'])} | "
    for k in sorted(Ks):
        prec = best_metrics[f'precision@{k}']
        rec = best_metrics[f'recall@{k}']
        f1 = best_metrics[f'F1@{k}']
        ndcg = best_metrics[f'ndcg@{k}']
        output_str += f"@{k}: Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, NDCG={ndcg:.4f} | "

    logging.info(output_str)


def predict(args):
    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderModel(args, logging)

    # load model
    model = MODEL(args, data.n_patients, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)

    logging.info('CF Evaluation:')
    print('CF Evaluation:')
    for k in Ks:
        precision = metrics_dict[k]['precision']
        recall = metrics_dict[k]['recall']
        f1 = metrics_dict[k]['F1']
        ndcg = metrics_dict[k]['ndcg']

        line = f"  @{k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, NDCG={ndcg:.4f}"
        logging.info(line)
        print(line)

if __name__ == '__main__':
    args = parse_MDP_GRL_args()
    args.use_pretrain = 2
    args.pretrain_model_path = 'trained_model/MDP_GRL/mimic/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/best_model.pth'
    print(args)
    predict(args)


