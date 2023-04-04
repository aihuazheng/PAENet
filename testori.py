from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine import embed_net 
from utils import *
import pdb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE; HAS_SK = True

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network newnew: resnet50')
parser.add_argument('--resume', '-r', default='margin_mmd1.40_sysu_c_tri_pcb_off_w_tri_2.0_share_net3_base_gm10_k4_p4_lr_0.1_seed_0_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='/data/panpeng/A/MMD-ReID-main/save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='/data/panpeng/A/MMD-ReID-main/log/log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='/data/panpeng/A/MMD-ReID-main/log/log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')

parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')
parser.add_argument('--run_name', type=str,
                    help='Run Name for following experiment', default='test_run')

parser.add_argument('--m', default=0.4, type=float, help='Value of Additive Margin')
parser.add_argument('--num_trials', type=int, default=10 ,help='Number of Trials for Averaging')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/data/panpeng/'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = '/data/panpeng/RegDB/'
    n_class = 206
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048
print('==> Building model..')
if args.method =='base':
    #net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
    net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def rank_visual(query_i, qf, ql, qc, qi, gf, gl, gc, gi):
    index = sort_img(qf[query_i], ql[query_i], qc[query_i], gf, gl, gc)
    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(qi[query_i][0], 'query')
        for i in range(10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path = gi[index[i]][0]
            label = gl[index[i]]
            imshow(img_path)
            if label == ql[query_i]:
                ax.set_title('%d' % (i + 1), color='green')
            else:
                ax.set_title('%d' % (i + 1), color='red')
            # print(img_path)
    except RuntimeError:
        for i in range(10):
            img_path = gi[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    name = str(query_i) +'rank.jpg'
    save_path = '/data/panpeng/A/MMD-ReID-main/'
    path = os.path.join(save_path,name)
    fig.savefig(path)

def sort_img(qf, ql, qc, gf, gl, gc):

    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index
    
def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label,atr_label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            atr_label = atr_label.float()
            atr_label =atr_label.cuda()
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, atr_label,test_mode[0])
            gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label,atr_label ) in enumerate(query_loader):
            batch_num = input.size(0)
            atr_label = atr_label.float()
            atr_label =atr_label.cuda()
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, atr_label,test_mode[1])
            query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc


def showPointSingleModal(features1, label1, features2,label2,save_path):
    # label = self.relabel(label)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    features_tsne = tsne.fit_transform(features1)
    COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
    MARKS = ['x', 'o', '+', '^', 's']
    features_min, features_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - features_min) / (features_max - features_min)
    plt.figure(figsize=(20, 20))
    for i in range(features_norm.shape[0]):
        plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[label1[i] % 6],
                    marker='s')
                    
    features_tsne = tsne.fit_transform(features2)
    COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
    MARKS = ['x', 'o', '+', '^', 's']
    features_min, features_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - features_min) / (features_max - features_min)
    for i in range(features_norm.shape[0]):
        plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[label2[i] % 6],
                    marker='o')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    
if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
       # model_path = checkpoint_path + args.resume
        model_path = '/data/panpeng/A/MMD-ReID-main/save_model/margin_mmd1.40_sysu_c_tri_pcb_off_w_tri_2.0_share_net3_base_gm10_k4_p4_lr_0.1_seed_0_best.t'
                                        #sysu_agw_p4_n8_lr_0.1_seed_0_best.t
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam ,query_atr= process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam ,gall_atr= process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label,query_atr, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
    query_feat_pool, query_feat_fc= extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam,gall_atr = process_gallery_sysu(data_path, mode=args.mode, trial=trial)
        trial_gallset = TestData(gall_img, gall_label, gall_atr,transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool, gall_feat_fc= extract_gall_feat(trial_gall_loader)
        #rank_visual(1, query_feat, query_label, query_cam, query_img, gall_feat, gall_label, gall_cam, gall_img)
        #showPointSingleModal(query_feat_pool,query_label,gall_feat_pool,gall_label,'/data/panpeng/A/MMD-ReID-main/')
        #pdb.set_trace()
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        #model_path = checkpoint_path +  args.resume
        model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
        gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

        if args.tvsearch:
            # pool5 feature
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
        else:
            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

cmc = all_cmc / 10
mAP = all_mAP / 10
mINP = all_mINP / 10

cmc_pool = all_cmc_pool / 10
mAP_pool = all_mAP_pool / 10
mINP_pool = all_mINP_pool / 10
print('All Average:')
print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))