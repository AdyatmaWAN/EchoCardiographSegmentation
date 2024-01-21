import math 
import os 
import time 
import numpy as np 
import torch 
import tqdm 
import echonet

def run_epoch(model, dataloader, train, optim, device):
        total = 0.
        n = 0

        pos = 0
        neg = 0
        pos_pix = 0
        neg_pix = 0

        model.train(train)

        large_inter = 0
        large_union = 0
        small_inter = 0
        small_union = 0
        large_inter_list = []
        large_union_list = []
        small_inter_list = []
        small_union_list = []

        with torch.set_grad_enabled(train):
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                        for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                                # Count number of pixels in/out of human segmentation
                                pos += (large_trace == 1).sum().item()
                                pos += (small_trace == 1).sum().item()
                                neg += (large_trace == 0).sum().item()
                                neg += (small_trace == 0).sum().item()

                                # Count number of pixels in/out of computer segmentation
                                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                                # Run prediction for diastolic frames and compute loss
                                large_frame = large_frame.to(device)
                                large_trace = large_trace.to(device)
                                y_large = model(large_frame)
                                #print(y_large.size())
                                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                                # Compute pixel intersection and union between human and computer segmentations
                                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                                
                                # Run prediction for systolic frames and compute loss
                                small_frame = small_frame.to(device)
                                small_trace = small_trace.to(device)
                                y_small = model(small_frame)
                                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                                # Compute pixel intersection and union between human and computer segmentations
                                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                                # Take gradient step if training
                                loss = (loss_large + loss_small) / 2
                                if train:
                                        optim.zero_grad()
                                        loss.backward()
                                        optim.step()

                                # Accumulate losses and compute baselines
                                total += loss.item()
                                n += large_trace.size(0)
                                p = pos / (pos + neg)
                                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)
                                
                                # Show info on process bar
                                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                                pbar.update()

        large_inter_list = np.array(large_inter_list)
        large_union_list = np.array(large_union_list)
        small_inter_list = np.array(small_inter_list)
        small_union_list = np.array(small_union_list)

        return (total / n / 112 / 112,
                        large_inter_list,
                        large_union_list,
                        small_inter_list,
                        small_union_list,
                        )

batch_size=6
num_epochs=50
# Seed RNGs
seed = 8
np.random.seed(seed)
torch.manual_seed(seed)

data_dir = "../EchoNet-Dynamic"
base_model = "models"

model_name="proposed"

# Set default output directory
output = os.path.join("output", "segmentation", "{}".format(model_name))
os.makedirs(output, exist_ok=True)

# Set device for computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set up model
#from Models.DeepLabV3Plus.Dv3plusCoord1024.deeplabv3plus import DeepLab
#model = DeepLab()
#model.load_state_dict(torch.load("./CAMUS_deeplabfiz-coordatt1024-gtx1650.pth"))
from Models.DeepLabV3Plus.Dv3plusCoord.deeplabv3plus import DeepLab
model = DeepLab()
model.to(device)
lr=1e-4
optim = torch.optim.Adam(model.parameters(), lr=lr)
#optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Compute mean and std
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
kwargs = {"target_type": tasks, "mean": mean, "std": std}

# Set up datasets and dataloaders
dataset = {}
num_train_patients=5
dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
#if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
#       indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
#       dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
#if num_train_patients is not None and len(dataset["val"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
#       indices = np.random.choice(len(dataset["val"]), num_train_patients, replace=False)
#       dataset["val"] = torch.utils.data.Subset(dataset["val"], indices)

# Run on validation and test
with open(os.path.join(output, "log.csv"), "a") as f:
        # Train
        print("Running Train")
        f.write("Train\n")
        for epoch in range(num_epochs):
                print("Epoch #{}".format(epoch), flush=True)
                for phase in ['train', 'val']:
                        start_time = time.time()
                        ds = dataset[phase]
                        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
                        loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, phase == "train", optim, device)
                        overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                        large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                        small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                        f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch, phase, loss, overall_dice, large_dice, small_dice, time.time() - start_time, large_inter.size, batch_size))
                        f.flush()

        # save model
        print('saving model')
        torch.save(model.state_dict(), "./proposed.pth")
        # Test
        print("Running Test")
        f.write("Val and Test\n")
        for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                        g.write("Filename, Overall, Large, Small\n")
                        for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                                g.write("{},{},{},{}\n".format(filename, overall, large, small))
        
        f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
        f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
        f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
        f.flush()
