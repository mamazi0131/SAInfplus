
import os,pickle

path = './dataset'
for region in ['A','B','C']:
    dataset = 'region_{}'.format(region)
    dataset_path = os.path.join(path,dataset)
    data = {}
    for _,_,filename_list in os.walk(dataset_path):
        for filename in filename_list:
            with open(os.path.join(dataset_path,filename),'rb') as f:
                one_data = pickle.load(f)
            data[filename] = one_data

    data['sequence_data.pkl'] = data['sequence_data.pkl'].head(1000)
    
    for filename in ['label_with_duration.pkl','label.pkl']:
        new_one_data = {}
        for key,value in data[filename].items():
            if key[0] < 1000:
                new_one_data[key] = value
        data[filename] = new_one_data
    
    for filename in ['sequence_data.pkl','label_with_duration.pkl','label.pkl']:
        with open(os.path.join(dataset_path,filename),'wb') as f:
            pickle.dump(data[filename],f)