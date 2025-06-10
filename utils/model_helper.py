import os
from collections import OrderedDict

import torch


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, current_ndcg, last_best_epoch=None, last_best_ndcg=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_state_file = os.path.join(model_dir, 'best_model.pth')

    if last_best_ndcg is None or current_ndcg > last_best_ndcg:
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': current_epoch,
            'best_ndcg': current_ndcg
        }, model_state_file)

    return current_ndcg 


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


