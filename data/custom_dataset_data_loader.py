import torch.utils.data
from data.base_data_loader import BaseDataLoader
from nn_temp.dataset import *

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'unaligned_triplet':
        from data.unaligned_triplet_dataset import UnalignedTripletDataset
        dataset = UnalignedTripletDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        # self.dataset = CreateDataset(opt)
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batchSize,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.nThreads))

        # self.dataset = combo_dataset()
        # self.dataloader = torch.utils.data.DataLoader(
        #     combo_dataset(),
        #     batch_size=opt.batchSize,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.nThreads))

        self.dataset = Custom_MNIST()
        self.dataloader = torch.utils.data.DataLoader(
            Custom_MNIST(),
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))


    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            return data
