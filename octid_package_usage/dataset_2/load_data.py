import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from PIL import Image

base_dir = "/home/lingfeng/Downloads/tumor_dataset"

ADIMUC_dir = os.path.join(base_dir, "ADIMUC") # adipose tissue and mucus, i.e. loose non-tumor tissue
STRMUS_dir = os.path.join(base_dir, "STRMUS") # stroma and muscle, i.e. dense non-tumor tissue
TUMSTU_dir = os.path.join(base_dir, "TUMSTU") # colorectal cancer epithelial tissue and stomach cancer epithelial tissue, i.e. tumor tissue

print("ADIMUC length:", len(os.listdir(ADIMUC_dir))) # 3997
print("STRMUS length:", len(os.listdir(STRMUS_dir))) # 4000
print("TUMSTU length:", len(os.listdir(TUMSTU_dir))) # 4000

def dataset_collection_func(abnormal_ratio):

    normal_filepaths = []
    normal_labels = []

    abnormal_filepaths = []
    abnormal_labels = []

    train_filepaths=[]
    train_labels=[]
    test_filepaths=[]
    test_labels=[]

    np.random.seed(1234) # set seed

    for tissue_slice in tqdm(natsorted(os.listdir(ADIMUC_dir))):
        tissue_slice_path = os.path.join(ADIMUC_dir, tissue_slice)
        normal_filepaths.append(tissue_slice_path)
        normal_labels.append('No')
    
    for tissue_slice in tqdm(natsorted(os.listdir(STRMUS_dir))):
        tissue_slice_path = os.path.join(STRMUS_dir, tissue_slice)
        normal_filepaths.append(tissue_slice_path)
        normal_labels.append('No')

    for tissue_slice in tqdm(natsorted(os.listdir(TUMSTU_dir))):
        tissue_slice_path = os.path.join(TUMSTU_dir, tissue_slice)
        abnormal_filepaths.append(tissue_slice_path)
        abnormal_labels.append('Yes')

    normal_filepaths = np.array(normal_filepaths)
    normal_labels = np.array(normal_labels)

    abnormal_filepaths = np.array(abnormal_filepaths)
    abnormal_labels = np.array(abnormal_labels)
    
    # normal
    idx = np.random.permutation(len(normal_labels))
    normal_filepaths = normal_filepaths[idx]
    normal_labels = normal_labels[idx]

    validation_normal_filepaths = normal_filepaths[:1000]
    validation_normal_labels = normal_labels[:1000]

    test_normal_filepaths = normal_filepaths[1000:2000]
    test_normal_labels = normal_labels[1000:2000]

    train_normal_filepaths = normal_filepaths[2000:3000]
    train_normal_labels = normal_labels[2000:3000]

     # abnormal
    idx = np.random.permutation(len(abnormal_labels))
    abnormal_filepaths = abnormal_filepaths[idx]
    abnormal_labels = abnormal_labels[idx]

    validation_abnormal_filepaths = abnormal_filepaths[:1000]
    validation_abnormal_labels = abnormal_labels[:1000]

    test_abnormal_filepaths = abnormal_filepaths[1000:2000]
    test_abnormal_labels = abnormal_labels[1000:2000]

    train_abnormal_filepaths = abnormal_filepaths[2000:]
    train_abnormal_labels = abnormal_labels[2000:]

    # abnormal reduce
    train_abnormal_filepaths = train_abnormal_filepaths[:int(abnormal_ratio*len(train_normal_filepaths))]
    train_abnormal_labels = train_abnormal_labels[:int(abnormal_ratio*len(train_normal_labels))]

    print("train_normal_labels:",len(train_normal_labels))
    print("train_abnormal_labels:",len(train_abnormal_labels))

    print("validation_normal_labels:",len(validation_normal_labels))
    print("validation_abnormal_labels:",len(validation_abnormal_labels))

    print("test_normal_labels:",len(test_normal_labels))
    print("test_abnormal_labels:",len(test_abnormal_labels))

    train_filepaths = np.concatenate((train_normal_filepaths,train_abnormal_filepaths),axis=0)
    train_labels = np.concatenate((train_normal_labels,train_abnormal_labels),axis=0)

    validation_filepaths = np.concatenate((validation_normal_filepaths,validation_abnormal_filepaths),axis=0)
    validation_labels = np.concatenate((validation_normal_labels,validation_abnormal_labels),axis=0)

    test_filepaths = np.concatenate((test_normal_filepaths,test_abnormal_filepaths),axis=0)
    test_labels = np.concatenate((test_normal_labels,test_abnormal_labels),axis=0)

    train_labels = np.where(train_labels=='No', 0, 1)
    validation_labels = np.where(validation_labels=='No', 0, 1)
    test_labels = np.where(test_labels=='No', 0, 1)

    return train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels

train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels = dataset_collection_func(0.01)

image_shape = (256,256)

train_normal_filepaths = train_filepaths[np.where(train_labels==0)]

print("len(train_normal_filepaths):", len(train_normal_filepaths))

validation_normal_filepaths = validation_filepaths[np.where(validation_labels==0)]
validation_abnormal_filepaths = validation_filepaths[np.where(validation_labels==1)]

def get_image(img_path, save_path):
    im = Image.open(img_path)
    im = im.resize(image_shape)
    im.save(os.path.join(save_path, img_path.split('/')[-1]))

for idx in tqdm(range(len(train_normal_filepaths))):
    save_path = 'small_samples/training_dataset/template'
    img_path = train_normal_filepaths[idx]
    get_image(img_path, save_path)

for idx in tqdm(range(len(validation_normal_filepaths))):
    save_path = 'small_samples/validation_dataset/0_positive'
    img_path = validation_normal_filepaths[idx]
    get_image(img_path, save_path)

for idx in tqdm(range(len(validation_abnormal_filepaths))):
    save_path = 'small_samples/validation_dataset/1_negative'
    img_path = validation_abnormal_filepaths[idx]
    get_image(img_path, save_path)

# test_normal_filepaths = test_filepaths[np.where(test_labels==0)]
# test_abnormal_filepaths = test_filepaths[np.where(test_labels==1)]

# for idx in tqdm(range(len(test_normal_filepaths))):
#     save_path = 'small_samples/testing_dataset/0_positive'
#     img_path = test_normal_filepaths[idx]
#     get_image(img_path, save_path)

# for idx in tqdm(range(len(test_abnormal_filepaths))):
#     save_path = 'small_samples/testing_dataset/1_negative'
#     img_path = test_abnormal_filepaths[idx]
#     get_image(img_path, save_path)
