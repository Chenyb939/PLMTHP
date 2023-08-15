import torch

#Combine THPs' ESM features(Please change your paths)
print("pos-pep")
lt = []
for i in range(1,652):
    ESM_dict = torch.load("C:\\Windows\\System32\\PLMTHP\\data\\Feature\\ESM3B\\pos-pep%d.pt"%(i))
    pos = ESM_dict['representations'][36].mean(axis=0)
    tensor_list = pos.numpy()
    lt.append(tensor_list)
ESM_pos = torch.tensor(lt)
torch.save(ESM_pos, 'C:\\Windows\\System32\\PLMTHP\\data\\Feature\\pos_esm.pt')

#Combine non_THPs' ESM features(Please change your paths)
print("neg-pep")
lt = []
for i in range(1,652):
    ESM_dict = torch.load("C:\\Windows\\System32\\PLMTHP\\data\\Feature\\ESM3B\\neg-pep%d.pt"%(i))
    neg = ESM_dict['representations'][36].mean(axis=0)
    tensor_list = neg.numpy()
    lt.append(tensor_list)
ESM_neg = torch.tensor(lt)
torch.save(ESM_neg, 'C:\\Windows\\System32\\PLMTHP\\data\\Feature\\neg_esm.pt')

#Check the shape of feature ESM( * ,2560)
ESM_pos.shape
ESM_neg.shape