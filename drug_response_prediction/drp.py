import argparse
import random
import os
import numpy as np
import csv
import pandas as pd
from model import MLP, UGCNN, CombinedMLP
from scipy.stats import pearsonr

import hickle as hkl
import argparse
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, DataLoader
import torch.optim as optim
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Drug response prediction.')
parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                    default='0', help='GPU devices. Use -1 for CPU.')
parser.add_argument('--embedding_path', type=str, default=None,
                    help='Path to the gene expression embeddings.')
parser.add_argument('--test_drug', type=str,
                    default=None, help='Hold-out drug.')
parser.add_argument('--val_drug', type=str, default=None,
                    help='Drug used for early stopping.')

args = parser.parse_args()

TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ']
DPATH = './gene_pertubation'
Drug_info_file = f"./{DPATH}/drug/metadata.csv"
# '%s/CCLE/Cell_lines_annotations_20181226.txt' % DPATH
Cell_line_info_file = f"./{DPATH}/cell_line/metadata.txt"
# '%s/GDSC/drug_graph_feat' % DPATH
Drug_feature_file = f"./{DPATH}/drug/graph_feature"
# '%s/CCLE/GDSC_IC50.csv' % DPATH
Cancer_response_exp_file = f"./{DPATH}/ground_truth.csv"
Max_atoms = 100


assert (args.test_drug is None and args.val_drug is None) or (
    args.test_drug != args.val_drug), "test_drug and val_drug must either both be None, or refer to different drugs."


def MetadataGenerate(Drug_info_file, Cell_line_info_file, Drug_feature_file, Gene_expression_file, filtered):
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load(
            '%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # load gene expression faetures
    gexpr_feature = pd.read_csv(
        Gene_expression_file, sep=',', header=0, index_col=[0])

    experiment_data = pd.read_csv(
        Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter experiment gene_pertubation
    drug_match_list = [item for item in experiment_data.index if item.split(':')[
        1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(
                        experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append(
                        (each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' %
          (len(data_idx), nb_celllines, nb_drugs))
    return drug_feature, gexpr_feature, data_idx


def DataSplit(data_idx, ratio=0.95):
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        train_list = random.sample(
            data_subtype_idx, int(ratio*len(data_subtype_idx)))
        test_list = [
            item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx, data_test_idx


def DrugSplit(data_idx, drugtype):
    data_train_idx, data_test_idx = [], []
    data_test_idx = [item for item in data_idx if item[1] == drugtype]
    data_train_idx = [item for item in data_idx if item[1] != drugtype]
    return data_train_idx, data_test_idx


def CalculateGraphFeat(feat_mat, adj_list):
    edge_index = []

    # Convert adjacency list to edge_index format
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edge_index.append([node, neighbor])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a gene_pertubation object
    data = Data(x=torch.tensor(feat_mat).float(),
                edge_index=edge_index)

    return data


def FeatureExtract(data_idx, drug_feature, gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = torch.zeros((nb_instance, nb_gexpr_features)).float()
    target = torch.zeros(nb_instance).float()
    for idx in tqdm(range(nb_instance)):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]

        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]

        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list)

        gexpr_data[idx, :] = torch.tensor(
            gexpr_feature.loc[cell_line_id].values)
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    return drug_data, gexpr_data, target, cancer_type_list


def place_on_device(gpu_id=0, *tensors_or_models):
    # Determine the device based on the gpu_id
    if gpu_id == -1:
        device = torch.device('cpu')
    elif gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU with ID {gpu_id} is not available.")
    else:
        device = torch.device(f'cuda:{gpu_id}')

    # Move each tensor or model to the specified device and return them as a list
    return [tensor_or_model.to(device) for tensor_or_model in tensors_or_models]


def main():
    random.seed(0)
    drug_feature, gexpr_feature, data_idx = MetadataGenerate(
        Drug_info_file, Cell_line_info_file, Drug_feature_file, args.embedding_path, False)

    gexpr_dim = gexpr_feature.shape[-1]

    if args.test_drug is None and args.val_drug is None:
        data_train_idx, data_real_idx = DataSplit(data_idx, ratio=0.95)
        data_train_idx, data_test_idx = DataSplit(
            data_train_idx, ratio=1-0.05/0.95)
    else:
        data_train_idx, data_real_idx = DrugSplit(data_idx, args.test_drug)
        assert len(data_real_idx) > 0, "Test drug doesn't exist."
        data_train_idx, data_test_idx = DrugSplit(
            data_train_idx, args.val_drug)
        assert len(data_real_idx) > 0, "Validation drug doesn't exist."

    # Extract features for training and test
    X_drug_data_train, X_gexpr_data_train, Y_train, _ = FeatureExtract(
        data_train_idx, drug_feature, gexpr_feature)

    batch_size = 32
    train_loader = DataLoader(list(zip(X_drug_data_train, X_gexpr_data_train, Y_train)),
                              batch_size=batch_size, shuffle=True, num_workers=8)

    X_drug_data_test, X_gexpr_data_test, Y_test, _ = FeatureExtract(
        data_test_idx, drug_feature, gexpr_feature)

    test_loader = DataLoader(list(zip(X_drug_data_test, X_gexpr_data_test, Y_test)),
                             batch_size=batch_size, shuffle=False, num_workers=8)

    X_drug_data_real, X_gexpr_data_real, Y_real, _ = FeatureExtract(
        data_real_idx, drug_feature, gexpr_feature)

    real_loader = DataLoader(list(zip(X_drug_data_real, X_gexpr_data_real, Y_real)),
                             batch_size=batch_size, shuffle=False, num_workers=8)

    drug_gcnn = UGCNN(input_dim=75, hidden_dims=[
        256, 256, 256], out_channels=100)
    gexpr_mlp = MLP(out_dim=100, input_dim=gexpr_dim)
    comb_mlp = CombinedMLP(input_dim=200)

    drug_gcnn, gexpr_mlp, comb_mlp = place_on_device(
        args.gpu_id, drug_gcnn, gexpr_mlp, comb_mlp)

    parameters = list(drug_gcnn.parameters()) + \
        list(gexpr_mlp.parameters()) + list(comb_mlp.parameters())

    optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = torch.nn.MSELoss()

    def compute_pcc(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return pearsonr(y_true, y_pred)[0]

    # Training loop
    epochs = 500
    patience = 10
    best_val_loss = -float('inf')
    best_drug_gcnn = None
    best_gexpr_mlp = None
    best_comb_mlp = None
    patience_counter = 0
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)

        drug_gcnn.train()
        gexpr_mlp.train()
        comb_mlp.train()
        for i, (drug, gexpr, target) in enumerate(progress_bar):
            target, drug, gexpr = place_on_device(
                args.gpu_id, target, drug, gexpr)
            optimizer.zero_grad()
            drug = drug_gcnn(drug)
            gexpr = gexpr_mlp(gexpr)
            out = comb_mlp(torch.cat((drug, gexpr), dim=1))

            loss = criterion(out.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=running_loss/(i+1))

        drug_gcnn.eval()
        gexpr_mlp.eval()
        comb_mlp.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for drug, gexpr, target in test_loader:
                target, drug, gexpr = place_on_device(
                    args.gpu_id, target, drug, gexpr)
                drug = drug_gcnn(drug)
                gexpr = gexpr_mlp(gexpr)
                out = comb_mlp(torch.cat((drug, gexpr), dim=1))
                loss = criterion(out.view(-1), target.view(-1))
                val_loss += loss.item()
                all_targets.append(target.view(-1).cpu().numpy())
                all_outputs.append(out.view(-1).cpu().numpy())

        val_loss /= len(test_loader)

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        pcc = compute_pcc(all_targets, all_outputs)

        print(
            f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, PCC: {pcc:.4f}')

        if pcc > best_val_loss:
            best_drug_gcnn = copy.deepcopy(drug_gcnn)
            best_gexpr_mlp = copy.deepcopy(gexpr_mlp)
            best_comb_mlp = copy.deepcopy(comb_mlp)
            best_val_loss = pcc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

    drug_gcnn.eval()
    gexpr_mlp.eval()
    comb_mlp.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for drug, gexpr, target in real_loader:
            target, drug, gexpr = place_on_device(
                args.gpu_id, target, drug, gexpr)
            drug = best_drug_gcnn(drug)
            gexpr = best_gexpr_mlp(gexpr)
            out = best_comb_mlp(torch.cat((drug, gexpr), dim=1))
            loss = criterion(out.view(-1), target.view(-1))
            val_loss += loss.item()
            all_targets.append(target.view(-1).cpu().numpy())
            all_outputs.append(out.view(-1).cpu().numpy())

    pcc_real = compute_pcc(np.concatenate(all_targets),
                           np.concatenate(all_outputs))
    print(f"PCC on hold-out gene_pertubation: {pcc_real}.")

    embedding_identifier = args.embedding_path[args.embedding_path.rfind(
        "/")+1: args.embedding_path.rfind(".csv")]

    os.makedirs("./results", exist_ok=True)

    task = "cell_line" if (
        args.val_drug is None and args.test_drug is None) else f"drug_{args.test_drug}"

    data = {
        'cell_line': [entry[0] for entry in data_real_idx],
        'pubchem_cid': [entry[1] for entry in data_real_idx],
        'cancer_type': [entry[3] for entry in data_real_idx],
        'prediction': np.concatenate(all_outputs).tolist(),
        'ground_truth': np.concatenate(all_targets).tolist(),
    }

    df = pd.DataFrame(data)
    df.to_csv(
        f"./results/hold_out_{task}_{embedding_identifier}.csv", index=False)
    print(
        f"Results saved to ./results/hold_out_{task}_{embedding_identifier}.csv")


if __name__ == '__main__':
    main()
