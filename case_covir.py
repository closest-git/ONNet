'''
@Author: Yingshi Chen
https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx_v2.ipynb
@Date: 2020-04-06 15:50:21
@
# Description: 
'''
import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from PIL import Image
import logging
import sys 
import time
ONNET_DIR = os.path.abspath("./python-package/")
sys.path.append(ONNET_DIR)  # To find local version of the onnet
#sys.path.append(os.path.abspath("./python-package/cnn_models/")) 
from cnn_models.COVIDNext50 import COVIDNext50 
from onnet import *
import torch
from torch.optim import Adam
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,classification_report

isONN=True
class COVID_set(Dataset):
    def __init__(self, config,img_dir, labels_file, transforms):
        self.config = config
        self.img_pths, self.labels = self._prepare_data(img_dir, labels_file)
        self.transforms = transforms
        

    def _prepare_data(self, img_dir, labels_file):
        with open(labels_file, 'r') as f:
            labels_raw = f.readlines()

        labels, img_pths = [], []
        for i in range(len(labels_raw)):
            data = labels_raw[i].split()
            img_pth = data[1] 
            #img_name = data[1]
            #img_pth = os.path.join(img_dir, img_name)
            img_pths.append(img_pth)
            labels.append(self.config.mapping[data[2]])

        return img_pths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_pths[idx]).convert("RGB")
        img_tensor = self.transforms(img)

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

    def train_test_split():
        seed = 0
        np.random.seed(seed) # Reset the seed so all runs are the same.
        random.seed(seed)
        MAXVAL = 255  # Range [0 255]

        # path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
        imgpath = 'E:/Insegment/covid-chestxray-dataset-master/images' 
        csvpath = 'E:/Insegment/covid-chestxray-dataset-master/metadata.csv'

        # path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
        kaggle_datapath = 'F:/Datasets/rsna-pneumonia-detection-challenge/'
        kaggle_csvname = 'stage_2_detailed_class_info.csv' # get all the normal from here
        kaggle_csvname2 = 'stage_2_train_labels.csv' # get all the 1s from here since 1 indicate pneumonia
        kaggle_imgpath = 'stage_2_train_images'

        # parameters for COVIDx dataset
        train = []
        test = []
        test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

        mapping = dict()
        mapping['COVID-19'] = 'COVID-19'
        mapping['SARS'] = 'pneumonia'
        mapping['MERS'] = 'pneumonia'
        mapping['Streptococcus'] = 'pneumonia'
        mapping['Normal'] = 'normal'
        mapping['Lung Opacity'] = 'pneumonia'
        mapping['1'] = 'pneumonia'

        train_file = open("train_split_v2.txt","a") 
        test_file = open("test_split_v2.txt", "a")
        # train/test split
        split = 0.1
        csv = pd.read_csv(csvpath, nrows=None)
        idx_pa = csv["view"] == "PA"  # Keep only the PA view
        csv = csv[idx_pa]

        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
        pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
        pathologies = sorted(pathologies)

        filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
        for index, row in csv.iterrows():
            f = row['finding']
            if f in mapping:
                count[mapping[f]] += 1
                entry = [int(row['patientid']), row['filename'], mapping[f]]
                filename_label[mapping[f]].append(entry)

        print('Data distribution from covid-chestxray-dataset:')
        print(count)

        for key in filename_label.keys():
            arr = np.array(filename_label[key])
            if arr.size == 0:
                continue
            # split by patients
            # num_diff_patients = len(np.unique(arr[:,0]))
            # num_test = max(1, round(split*num_diff_patients))
            # select num_test number of random patients
            if key == 'pneumonia':
                test_patients = ['8', '31']
            elif key == 'COVID-19':
                test_patients = ['19', '20', '36', '42', '86'] # random.sample(list(arr[:,0]), num_test)
            else: 
                test_patients = []
            print('Key: ', key)
            print('Test patients: ', test_patients)
            # go through all the patients
            for patient in arr:    
                info = f"{str(patient[0])} {imgpath}\{patient[1]} {patient[2]}\n" 
                if patient[0] in test_patients:
                    #copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
                    test.append(patient);            test_count[patient[2]] += 1
                    train_file.write(info)
                else:
                    #copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))            
                    train.append(patient);            train_count[patient[2]] += 1
                    test_file.write(info)


        csv_normal = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname), nrows=None)
        csv_pneu = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname2), nrows=None)
        patients = {'normal': [], 'pneumonia': []}

        for index, row in csv_normal.iterrows():
            if row['class'] == 'Normal':
                patients['normal'].append(row['patientId'])

        for index, row in csv_pneu.iterrows():
            if int(row['Target']) == 1:
                patients['pneumonia'].append(row['patientId'])

        for key in patients.keys():
            arr = np.array(patients[key])
            if arr.size == 0:
                continue
            # split by patients 
            num_diff_patients = len(np.unique(arr))
            num_test = max(1, round(split*num_diff_patients))
            #test_patients = np.load('rsna_test_patients_{}.npy'.format(key)) # 
            test_patients = random.sample(list(arr), num_test)  #, download the .npy files from the repo.
            np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
            for patient in arr:
                ds = dicom.dcmread(os.path.join(kaggle_datapath, kaggle_imgpath, patient + '.dcm'))
                pixel_array_numpy = ds.pixel_array
                imgname = patient + '.png'
                if patient in test_patients:
                    path = os.path.join(kaggle_datapath, 'test', imgname)
                    cv2.imwrite(path, pixel_array_numpy)
                    test.append([patient, imgname, key]);                test_count[key] += 1
                    test_file.write(f"{patient} {path} {key}\n" )
                    if test_count[key]%50==0:
                        test_file.flush()
                else:
                    path = os.path.join(kaggle_datapath, 'train', imgname)
                    cv2.imwrite(path, pixel_array_numpy)
                    train_file.write(f"{patient} {path} {key}\n")
                    if train_count[key]%20==0:
                        train_file.flush()
                    train.append([patient, imgname, key]);                train_count[key] += 1
                print(f"\r@{path}",end="")                

        print('Final stats')
        print('Train count: ', train_count)
        print('Test count: ', test_count)
        print('Total length of train: ', len(train))
        print('Total length of test: ', len(test))

        train_file.close()
        test_file.close()



log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_model(model, config):
    if isinstance(model, torch.nn.DataParallel):
        # Save without the DataParallel module
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    state = {
        "state_dict": model_dict,
        "global_step": config['global_step'],
        "clf_report": config['clf_report']
    }
    f1_macro = config['clf_report']['macro avg']['f1-score'] * 100
    name = "{}_F1_{:.2f}_step_{}.pth".format(config['name'],
                                             f1_macro,
                                             config['global_step'])
    model_path = os.path.join(config['save_dir'], name)
    torch.save(state, model_path)
    log.info("Saved model to {}".format(model_path))


def validate(data_loader, model, best_score, global_step, cfg):
    model.eval()
    gts, predictions = [], []

    log.info("Validation started...")
    for data in data_loader:
        imgs, labels = data
        imgs = to_device(imgs, gpu=cfg.gpu)

        with torch.no_grad():
            logits = model(imgs)
            if isONN:
                preds = net.predict(logits).cpu().numpy()
            else:
                probs = model.module.probability(logits)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

        labels = labels.cpu().detach().numpy()
        predictions.extend(preds)
        gts.extend(labels)

    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)
    acc, f1, prec, rec = clf_metrics(predictions=predictions,targets=gts,average="macro")
    report = classification_report(gts, predictions, output_dict=True)
    log.info("\n====== VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | Recall {:.4f}".format(acc, f1, prec, rec))

    if f1 > best_score:
        save_config = {
                    'name': config.name,
                    'save_dir': config.ckpts_dir,
                    'global_step': global_step,
                    'clf_report': report
                }
        #save_model(model=model, config=save_config)
        best_score = f1
    #log.info("Validation end")
    model.train()
    return best_score

def train_transforms(width, height):
    trans_list = [
        transforms.Resize((height, width)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=20,
                                    translate=(0.15, 0.15),
                                    scale=(0.8, 1.2),
                                    shear=5)], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)


def val_transforms(width, height):
    trans_list = [
        transforms.Resize((height, width)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)

def to_device(tensor, gpu=False):
    return tensor.cuda() if gpu else tensor.cpu()

def clf_metrics(predictions, targets, average='macro'):
    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    acc = accuracy_score(targets, predictions)

    return acc, f1, precision, recall

def main(model):
    if config.gpu and not torch.cuda.is_available():
        raise ValueError("GPU not supported or enabled on this system.")
    use_gpu = config.gpu

    log.info("Loading train dataset")
    train_dataset = COVID_set(config,config.train_imgs, config.train_labels,train_transforms(config.width,config.height))
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,shuffle=True,drop_last=True, num_workers=config.n_threads,pin_memory=use_gpu)
    log.info("Number of training examples {}".format(len(train_dataset)))

    log.info("Loading val dataset")
    val_dataset = COVID_set(config,config.val_imgs, config.val_labels,val_transforms(config.width,config.height))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.n_threads,
                            pin_memory=use_gpu)
    log.info("Number of validation examples {}".format(len(val_dataset)))

    if use_gpu:
        model.cuda()
        #model = torch.nn.DataParallel(model)
    optim_layers = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer and lr scheduler
    optimizer = Adam(optim_layers,
                     lr=config.lr,
                     weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  factor=config.lr_reduce_factor,
                                  patience=config.lr_reduce_patience,
                                  mode='max',
                                  min_lr=1e-7)

    # Load the last global_step from the checkpoint if existing
    global_step = 0 if state is None else state['global_step'] + 1

    class_weights = to_device(torch.FloatTensor(config.loss_weights),gpu=use_gpu)
    loss_fn = CrossEntropyLoss(reduction='mean', weight=class_weights)

    # Reset the best metric score
    best_score = -1
    t0=time.time()
    for epoch in range(config.epochs):
        log.info("\nStarted epoch {}/{}".format(epoch + 1,config.epochs))
        for data in train_loader:
            imgs, labels = data
            imgs = to_device(imgs, gpu=use_gpu)
            labels = to_device(labels, gpu=use_gpu)

            logits = model(imgs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % config.log_steps == 0 and global_step > 0:
                if isONN:
                    preds = net.predict(logits).cpu().numpy()
                else:
                    probs = model.module.probability(logits)
                    preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
                labels = labels.cpu().detach().numpy()
                acc, f1, _, _ = clf_metrics(preds, labels)
                lr = optimizer.param_groups[0]['lr']    #get_learning_rate(optimizer)
                print(f"\r{global_step} | batch: Loss={loss.item():.3f} | F1={f1:.3f} | Accuracy={acc:.4f} | LR={lr:.2e}\tT={time.time()-t0:.4f}",end="")


            if global_step % config.eval_steps == 0 and global_step > 0: 
                best_score = validate(val_loader, model,best_score=best_score,global_step=global_step,cfg=config)
                scheduler.step(best_score)
            global_step += 1

def UpdateConfig(config):
    config.name = "COVIDNext50_NewData"
    config.gpu = True
    config.batch_size = 16
    config.n_threads = 4
    config.random_seed = 1337
    config.weights = "E:/Insegment/COVID-Next-Pytorch-master/COVIDNext50_NewData_F1_92.98_step_10800.pth"
    config.lr = 1e-4
    config.weight_decay = 1e-3
    config.lr_reduce_factor = 0.7
    config.lr_reduce_patience = 5
    # Data
    config.train_imgs = None#"/data/ssd/datasets/covid/COVIDxV2/data/train"
    config.train_labels = "E:/ONNet/data/covid_train_split_v2.txt"   #"/data/ssd/datasets/covid/COVIDxV2/data/train_COVIDx.txt"
    config.val_imgs = None#"/data/ssd/datasets/covid/COVIDxV2/data/test"
    config.val_labels = "E:/ONNet/data/covid_test_split_v2.txt"    #"/data/ssd/datasets/covid/COVIDxV2/data/test_COVIDx.txt"
    # Categories mapping
    config.mapping = {
        'normal': 0,
        'pneumonia': 1,
        'COVID-19': 2
    }
    # Loss weigths order follows the order in the category mapping dict
    config.loss_weights = [0.05, 0.05, 1.0]

    config.width = 256
    config.height = 256
    config.n_classes = len(config.mapping)
    # Training
    config.epochs = 300
    config.log_steps = 5
    config.eval_steps = 400
    config.ckpts_dir = "./experiments/ckpts"
    return config

IMG_size =  (256, 256)
if __name__ == '__main__':
    config_0 = NET_config("DNet",'covid',IMG_size,0.01,batch_size=16, nClass=3, nLayer=5)
    #config_0 = RGBO_CNN_config("RGBO_CNN",'covid',IMG_size,0.01,batch_size=16, nClass=3, nLayer=5)
    if isONN:
        env_title, net = DNet_instance(config_0)  
        #env_title, net = RGBO_CNN_instance(config_0)  
        config = net.config
        config = UpdateConfig(config)
        config.batch_size = 64
        config.log_steps = 10
        config.lr = 0.001
        state = None
    else:
        config = UpdateConfig(config_0)
        if config.weights:
            state = torch.load(config.weights)
            log.info("Loaded model weights from: {}".format(config.weights))
        else:
            state = None

        state_dict = state["state_dict"] if state else None
        net = COVIDNext50(n_classes=config.n_classes)
        if state_dict:
            net = load_model_weights(model=net, state_dict=state_dict,log=log)
    print(net)
    Net_dump(net)
    seed_everything(config.random_seed)    
    main(net)
