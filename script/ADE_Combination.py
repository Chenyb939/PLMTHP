import torch

def combine_train():
    # Load AAC,DPC and ESM features of THPs and combine them
    aac = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aac_pos.pt'))
    dpc = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpc_pos.pt'))
    esm = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pos_esm.pt'))
    pos_ad = torch.cat((aac, dpc), 1)
    pos_ade = torch.cat((pos_ad, esm), 1)
    torch.save(pos_ade, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pos_ade.pt'))

    # Load AAC,DPC and ESM features of non_THPs and combine them
    aac = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aac_neg.pt'))
    dpc = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpc_neg.pt'))
    esm = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neg_esm.pt'))
    neg_ad = torch.cat((aac, dpc), 1)
    neg_ade = torch.cat((neg_ad, esm), 1)
    torch.save(neg_ade, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neg_ade.pt'))

def combine_test():
    # Load AAC,DPC and ESM features of test and combine them
    aac = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aac.pt'))
    dpc = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dpc.pt'))
    esm = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'esm.pt'))
    pos_ad = torch.cat((aac, dpc), 1)
    pos_ade = torch.cat((pos_ad, esm), 1)
    torch.save(pos_ade, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ade.pt'))





