import torch
import numpy as np
from mount import *
from mount.train_nn_mt_ko import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

import pickle
from mt_config import MT_Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE
print(DEVICE)

opt = MT_Config()

from torch.utils.data import Dataset


meta_path = '/ifs/groups/rosenGrp/zz374/Multi-task_GO/processed_data'
OUTPUT_DIR = '/ifs/groups/rosenGrp/zz374/Multi-task_GO/multi_task_data_exp'
data_path = '/ifs/groups/rosenGrp/zz374/Multi-task_GO/multi_task_data'


fold_dict = pickle.load(open('{}/fold_data.pkl'.format(OUTPUT_DIR), 'rb'))
[go, functions_bp, functions_mf, functions_cc] = pickle.load(open('{}/MT_metadata.pkl'.format(data_path), 'rb'))

functions = functions_bp, functions_mf, functions_cc

def concatenate_arrays(array_list):
    return np.concatenate([array_list[item] for item in range(5)])

X = concatenate_arrays([fold_dict[item][0] for item in fold_dict])
X_length = concatenate_arrays([fold_dict[item][1] for item in fold_dict])
y1 = concatenate_arrays([fold_dict[item][2][0] for item in fold_dict])
y2 = concatenate_arrays([fold_dict[item][2][1] for item in fold_dict])
y3 = concatenate_arrays([fold_dict[item][2][2] for item in fold_dict])
y4 = concatenate_arrays([fold_dict[item][2][3] for item in fold_dict])

[X_test, X_test_length, test_labels, y_FULL_TERM, VALID_INDEX_list] = pickle.load(open('{}/test_data_ko.pkl'.format(OUTPUT_DIR), 'rb'))

print('initializing model...')
best_loss = 100

# epoch
start_epoch = 0
epochs = opt.epochs
epochs_since_improvement = 0  


val_split = 0.05
VAL_SEG_IDX = int(X.shape[0] * (1-val_split))
    
print(X.shape[0])

X_val, X_val_length, y_val_1, y_val_2, y_val_3, y_val_4 = X[VAL_SEG_IDX:], X_length[VAL_SEG_IDX:], y1[VAL_SEG_IDX:], y2[VAL_SEG_IDX:], y3[VAL_SEG_IDX:], y4[VAL_SEG_IDX:]
X, X_length, y1, y2, y3, y4 = X[:VAL_SEG_IDX], X_length[:VAL_SEG_IDX], y1[:VAL_SEG_IDX], y2[:VAL_SEG_IDX], y3[:VAL_SEG_IDX], y4[:VAL_SEG_IDX]

print(X.shape[0], X_val.shape[0])

task_name_list = ['bp', 'mf', 'cc', 'ko']
model_dict = {}
for model_index in range(4):
    FUNCTION = task_name_list[model_index]
    model_dict[model_index] = torch.load('ST_torch_model_{}_all.pkl'.format(FUNCTION), map_location='cpu')


model = CrossStitchModel_KO(in_channels=X.shape[2], out_channels=opt.ResNet_out_channels, kernel_size=opt.ResNet_kernel_size, 
              n_layers=opt.ResNet_n_layers, n_class=[y1.shape[1], y2.shape[1], y3.shape[1], y4.shape[1]], n_hidden_state_list=opt.LSTM_n_hidden_state, 
              use_gru=True, lstm_dropout=0, n_lstm_layers=1, activation='sigmoid', hierarchical=True, functions=functions, go=go)

model = model.to(torch.double)

optimizer = torch.optim.AdamW(params=model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

mt_model_dict_random = model.state_dict()
print('random bias weight at dense_1_608', mt_model_dict_random['final_dense_1.dense_list.608.fc.bias'])

for key in mt_model_dict_random:
    if len(key.split('.')) <= 1:
        continue
    layer_name = key.split('.')[0]
    weight_name = '.'.join(key.split('.')[1:])
    layer_name_simple = '_'.join(layer_name.split('_')[:-1])
    model_index = int(layer_name.split('_')[-1])
    
    layer_name_from_single = '{}_{}.{}'.format(layer_name_simple, 1, weight_name)
    
    mt_model_dict_random[key] = model_dict[model_index-1][layer_name_from_single]

print('weights from pretrained model', model_dict[0]['final_dense_1.dense_list.608.fc.bias'])
model.load_state_dict(mt_model_dict_random, strict=False)
print('weights loaded from pretrained', model.final_dense_1.dense_list[608].fc.bias)
    

def trainable_params(model):
    c = 0
    for param in model.parameters():
        if param.requires_grad:
            c += 1
    return c
def unfreeze_params(submodel):
    for dense_node in submodel:
        for param in dense_node.parameters():
            param.requires_grad = True
def unfreeze_params_dense(dense_node):
    for param in dense_node.parameters():
        param.requires_grad = True
for param in model.parameters():
    param.requires_grad = False
print(trainable_params(model))
model.alpha.requires_grad = True
print(trainable_params(model))
unfreeze_params(model.final_dense_1.dense_list)
print(trainable_params(model))
unfreeze_params(model.final_dense_2.dense_list)
print(trainable_params(model))
unfreeze_params(model.final_dense_3.dense_list)
print(trainable_params(model))
unfreeze_params_dense(model.final_dense_4)
print(trainable_params(model))

print('freezed', model.att_bilstm_1.att.dense.bias.data)
print('unfreezed', model.final_dense_1.dense_list[608].fc.bias.data)


# move to CPU/GPU
model = model.to(opt.device)

# loss function
criterion = nn.BCELoss().to(opt.device)

train_data = dataloader(X, X_length, y1, y2, y3, y4)
val_data = dataloader(X_val, X_val_length, y_val_1, y_val_2, y_val_3, y_val_4)

# Train/valide data

train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers = 1,
                    pin_memory=True)

val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers = 1,
                    pin_memory=True)

# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=opt.weight_decay)
# Epochs
for epoch in range(start_epoch, epochs):

    # decay learning rate
#     if epoch > opt.decay_epoch:
#         adjust_learning_rate(optimizer, epoch)

    # early stopping
    if epochs_since_improvement == opt.improvement_epoch:
        break

    # train on one epoch
    train(train_loader=train_loader, model=model, criterion=criterion, 
          optimizer=optimizer, epoch=epoch, print_freq=opt.print_freq, 
          device=opt.device, grad_clip=opt.grad_clip)

    # validate on one epoch
    recent_loss = validate(val_loader=val_loader, model=model, criterion=criterion, 
                          print_freq=opt.print_freq, device=opt.device)
#     scheduler.step(recent_loss)
    # check improvements
    is_best = recent_loss < best_loss
    best_loss = min(recent_loss, best_loss)
    if not is_best:
        epochs_since_improvement += 1
        print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0

torch.save(model.state_dict(), '{}/MT_trained_model_transfer_all.pkl'.format(opt.save_model_path))

y_test_1, y_test_2, y_test_3, y_test_4 = test_labels

test_data = dataloader(X_test, X_test_length, y_test_1, y_test_2, y_test_3, y_test_4)

model = model.eval()
model = model.to(opt.device)
# Train/valide data

test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size, 
                    shuffle=False,
                    num_workers = 1,
                    pin_memory=True)

res = []
with torch.no_grad():
    for i, (seqs, seqs_len, labels_1, labels_2, labels_3, labels_4) in enumerate(test_loader):

            print('progress: {}/{}'.format(i, np.ceil(X_test.shape[0]/opt.batch_size)))

            index = torch.from_numpy(np.argsort(seqs_len.data.numpy())[::-1].copy())
            seqs = seqs[index]
            seqs_len = seqs_len[index]
            labels_1 = labels_1[index]
            labels_2 = labels_2[index]
            labels_3 = labels_3[index]
            labels_4 = labels_4[index]
            # move to CPU/GPU
            seqs = seqs.to(device)

            logits_1, logits_2, logits_3, logits_4 = model(seqs, seqs_len)
            pred_1 = logits_1.cpu().data.numpy()
            pred_2 = logits_2.cpu().data.numpy()
            pred_3 = logits_3.cpu().data.numpy()
            pred_4 = logits_4.cpu().data.numpy()

            res.append(([pred_1, pred_2, pred_3, pred_4], [labels_1, labels_2, labels_3, labels_4], index))

import pickle
pickle.dump(res, open('/home/zz374/MT_final/mt_predicition_transfer_dropout_all.pkl', 'wb'))

from collections import deque
def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set

def compute_performance(preds, labels, gos):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


import numpy as np
def get_data_for_go_branch(res, branch_id, y_FULL_TERM, VALID_INDEX):
    BATCH_SIZE = 128
    N = y_FULL_TERM.shape[0]
    
    M = res[0][0][branch_id].shape[1]
    pred_all = np.zeros((N, M))
    true_all = np.zeros((N, M))
    VALID_INDEX_sorted = np.zeros(N)
    gos_all = []
    
    for i in range(len(res)):
        pred, true, index = res[i]
        pred_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred[branch_id]
        true_all[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = true[branch_id].data.numpy()
        gos = y_FULL_TERM[i*BATCH_SIZE:(i+1)*BATCH_SIZE][index]
        VALID_INDEX_sorted[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = VALID_INDEX[i*BATCH_SIZE:(i+1)*BATCH_SIZE][index]
        gos_all.extend(gos)
        
    VALID_INDEX_sorted = VALID_INDEX_sorted == 1   
    return pred_all, true_all, gos_all, VALID_INDEX_sorted

FUNCTION_CAND = ['bp', 'mf', 'cc', 'ko']

task_index = 0
from sklearn.metrics import accuracy_score
for task_index in range(4):

    pred_all, true_all, gos_all, mask = get_data_for_go_branch(res, task_index, y_FULL_TERM, VALID_INDEX_list[task_index])
    gos_all = np.array(gos_all)
    gos_all = gos_all[mask]
    pred_all = pred_all[mask]
    true_all = true_all[mask]
    print(true_all.shape)

    if task_index < 3:
        [GO_ID, go, func_set, functions, all_functions, go_indexes] = pickle.load(open('{}/{}_go_meta.pkl'.format(meta_path, FUNCTION_CAND[task_index]), 'rb'))

        f_max, p_max, r_max, t_max, predictions_max = compute_performance(pred_all, true_all, gos_all)

        print('Performance: {}-all, {:.3f}, {:.3f}, {:.3f}, {}'.format(FUNCTION_CAND[task_index], f_max, p_max, r_max, t_max))
    else:
        acc = accuracy_score(np.argmax(true_all, axis=1), np.argmax(pred_all, axis=1))
        print('Performance: {}-all, {}'.format(FUNCTION_CAND[task_index], acc))

