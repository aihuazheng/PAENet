import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)



class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.visible,'layer'+str(i), getattr(model_v,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer'+str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net
        
        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.thermal,'layer'+str(i), getattr(model_t,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):           
                    x = getattr(self.thermal, 'layer'+str(i))(x)             
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net       
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base,'layer'+str(i), getattr(model_base,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer'+str(i))(x)
            return x


class ASENets(nn.Module):
    def __init__(self):
        super(ASENets, self).__init__()

        self.mask_fc1 = nn.Linear(8, 2048, bias=False)
        self.mask_fc2 = nn.Linear(8, 2048, bias=False)
    
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
    
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        #self.fc3 = nn.Linear(128, 512)
    
        self.drop2d = torch.nn.Dropout2d(0.1)
    
    def forward(self, f, attribute):
        image_attetion = self.ATT(f, attribute)  # b * 2048 * 18 * 9
        image_embedding = f * image_attetion  # 50 * 128 * 4 * 4
        image_embedding = image_embedding.view(image_embedding.size(0), image_embedding.size(1),
                                               image_embedding.size(2) * image_embedding.size(3))  # b * 2048 * 16
        image_embedding = image_embedding.sum(dim=2)  # b * 2048
    
        mask = self.ACA(image_embedding, attribute)  # b * 2048
        #pdb.set_trace()
        #image_embedding = image_embedding * mask  # 50 * 128
        #image_embedding = self.fc3(image_embedding)
        embedding = l2norm(mask)  # 50 * 512
        return embedding
    
    def ATT(self, x, c):
        img_embedding = self.conv1(x)  # b * 2048 * 1 * 1
        img_embedding = self.tanh(img_embedding)  # 50 * 128 * 1 * 1
        c = c.float()
        mask_fc_input = c.view(c.size(0), -1)  # 8
    
        mask = self.mask_fc1(mask_fc_input)  # 2048
        mask = self.tanh(mask)  # 2048
    
        mask = mask.view(mask.size(0), mask.size(1), 1, 1)  # b * 2048 * 1 * 1
        mask = mask.expand(mask.size(0), mask.size(1), 18, 9)  # b * 2048 * h * w
    
        attmap = mask * img_embedding  # 50 * 128 * 4 * 4
        attmap = self.conv2(attmap)  # 50 * 128 * 4 * 4
        attmap = self.tanh(attmap)  # 50 * 128 * 4 * 4
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)  # 50 * 128 * 16
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 18, 9)  # 50 * 128 * 4 * 4
    
        return attmap
    
    def ACA(self, x, c):
        c = c.float()
        c = c.view(c.size(0), -1)  # 8
        attr = self.mask_fc2(c)  # 50 * 512
        attr = self.relu(attr)  # 50 * 512
        img_attr = torch.cat((x, attr), dim=1)  # 50 * 640
        mask = self.fc1(img_attr)  # 50 * 128
        mask = self.relu(mask)  # 50 * 128
        mask = self.fc2(mask)  # 50 * 128
        mask = self.sigmoid(mask)
        return mask




class Region(nn.Module):
    def __init__(self):
        super(Region, self).__init__()
        pool_dim=2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        ######################################################
        self.bn = nn.BatchNorm2d(pool_dim)
        self.bn.bias.requires_grad_(False)
        
        
        self.bn.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.sig = nn.Sigmoid()
        

    def forward(self, id_feat_map,region_score_map):
        region_score_map = self.bn(region_score_map)
        region_score_map = self.sig(region_score_map)
        region_fea = id_feat_map * region_score_map
        region_fea = self.sig(region_fea)
        b, c, h, w = region_fea.shape
        region_fea = region_fea.view(b, c, -1)
        p = 3.0
        region_fea_pool = (torch.mean(region_fea**p, dim=-1) + 1e-12)**(1/p)
        region_fea_feat = self.bottleneck(region_fea_pool)
        return region_fea_pool,region_fea_feat
        
class Attrfea(nn.Module):
    def __init__(self):
        super(Attrfea, self).__init__()
        pool_dim=2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        ######################################################
        
        self.fc1 =  nn.Linear(pool_dim *8 ,2048 )
        self.bottl = nn.BatchNorm1d(pool_dim *8)
        self.bottl.bias.requires_grad_(False)  # no shift
        
        self.bottleneck.apply(weights_init_kaiming)
        self.bottl.apply(weights_init_kaiming)
        
        self.sig = nn.Sigmoid()

    def forward(self, attr_scores,attr_fea):
        attr_scores = self.bottl(attr_scores)
        attr_scores = self.sig(attr_scores)
        #attr_fea = torch.cat(attr_feat_list, 1)  #atr_feat
        #attr_fea = self.fc1(attr_fea)
        attr_fea = attr_fea*attr_scores
        #pdb.set_trace()
        attr_fea = self.fc1(attr_fea)
        attr_fea = self.bottleneck(attr_fea)
        return attr_fea

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self):
        super(MultiHeadCrossAttention, self).__init__()
        self.scale = 1 ** -0.5
        self.to_q = nn.Linear(2048, 2048, bias=False)
        self.to_kv = nn.Linear(2056, 2056 * 2, bias=False)

        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(2048, 2048)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, complement):
        # x [50, 128]
        
        B_x, N_x = x.shape  # 50, 128

        x_copy = x  # 50 * 128
        #pdb.set_trace()
        complement = torch.cat([x.float(), complement.float()], 1)  # 50 * 512

        B_c, N_c = complement.shape  # 50, 512

        # q [50, 1, 128, 1]
        #pdb.set_trace()
        q = self.to_q(x).reshape(B_x, N_x, 1, 1).permute(0, 2, 1, 3)
        # kv [2, 50, 1, 512, 1]
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, 1, 1).permute(2, 0, 3, 1, 4)
#kv = kv.reshape(B_c, N_c, 2, 1, 1)
        # 50 * 1 * 512 * 1
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 50 * 1 * 128 * 512
        attn = attn.softmax(dim=-1)  # 50 * 1 * 128 * 512
        attn = self.attn_drop(attn)  # 50 * 1 * 128 * 512

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x)  # 50 * 128

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)  # 50 * 128
        return x


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, mlp_ratio=1., proj_drop=0.3, act_layer=nn.GELU, norm_layer1=nn.LayerNorm,
                 norm_layer2=SwitchNorm1d):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer1(2048)
        self.c_norm1 = norm_layer2(8)

        self.attn = MultiHeadCrossAttention()

        self.x_norm2 = norm_layer1(2048)

        mlp_hidden_dim = int(2048 * mlp_ratio)
        self.mlp = Mlp(in_features=2048, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, complement):
        # x: 50 * 128
        # complement: 50 * 512
        x = self.x_norm1(x)  # 50 * 128
        complement = self.c_norm1(complement)  # 50 * 384
        #pdb.set_trace()
        x = x + self.drop1(self.attn(x, complement))  # 50 * 128
        x = x + self.drop2(self.mlp(self.x_norm2(x)))  # 50 * 128
        #pdb.set_trace()
        return x

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        #add_block = []
        #add_block += [nn.Linear(input_dim, num_bottleneck)]
        #add_block += [nn.BatchNorm1d(num_bottleneck)]
       # add_block += [nn.LeakyReLU(0.1)]
       # add_block += [nn.Dropout(p=0.5)]

       # add_block = nn.Sequential(*add_block)
        #add_block.bias.requires_grad_(False)
        #add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(input_dim, class_num,bias=False)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

       # self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
      #  x = self.add_block(x)
        x = self.classifier(x)
        return x

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X              
class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'off', gm_pool = 'on', arch='resnet50', share_net=1, pcb='on',local_feat_dim=256, num_strips=6):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.base_resnet = base_resnet(arch=arch, share_net=share_net)

        self.non_local = no_local
        self.pcb = pcb
        if self.non_local =='on':
            pass

        self.cross_transformer1 = CrossTransformerEncoderLayer(mlp_ratio=1., proj_drop=0.3, act_layer=nn.GELU, norm_layer1=nn.LayerNorm,norm_layer2=SwitchNorm1d)
        self.region = Region()
        self.ASENet=ASENets()
        self.attrfea = Attrfea()
        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool
        self.atr_num=8
        self.class_num=class_num
        ##########################################################
        for c in range(self.atr_num+1):
           if c == self.atr_num:
               self.__setattr__('class_%d' % c, ClassBlock(pool_dim, class_num=self.class_num, activ='none'))
           else:
               self.__setattr__('class_%d' % c, ClassBlock(pool_dim, class_num=1, activ='sigmoid'))
               ########################################################################
        if self.pcb == 'on':
            self.num_stripes=num_strips
            local_conv_out_channels=local_feat_dim

            self.local_conv_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
            
            self.fc_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)
            
        
        else:
            self.bottleneck = nn.BatchNorm1d(pool_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(pool_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
            

        

    def forward(self, x1, x2, attribute,modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            pass
        else:
            x = self.base_resnet(x)
        id_feat_map=x
        if self.pcb == 'on':
            feat = x
            assert feat.size(2) % self.num_stripes == 0
            stripe_h = int(feat.size(2) / self.num_stripes)
            local_feat_list = []
            logits_list = []
            for i in range(self.num_stripes):
                # shape [N, C, 1, 1]
                
                # average pool
                #local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                if self.gm_pool  == 'on':
                    # gm pool
                    local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b,c,-1)
                    p = 10.0    # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat**p, dim=-1) + 1e-12)**(1/p)
                else:
                    # average pool
                    #local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                

                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[i](local_feat.view(feat.size(0),feat.size(1),1,1))
               

                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)


                if hasattr(self, 'fc_list'):
                    logits_list.append(self.fc_list[i](local_feat))

            feat_all = [lf for lf in local_feat_list]
            feat_all = torch.cat(feat_all, dim=1)
            

            if self.training:
                return local_feat_list, logits_list, feat_all 
            else:
                return self.l2norm(feat_all)
        else:    
            if self.gm_pool  == 'on':
                b, c, h, w = x.shape
                x = x.view(b, c, -1)
                p = 3.0
                x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
            else:
                x_pool = self.avgpool(x)
                x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            feat = self.bottleneck(x_pool)
            
            ###################################################################################################
            #PAE：embedding-I
            feat= self.cross_transformer1(feat, attribute)
            f1 = torch.unsqueeze(feat, -1)
            f1 = torch.unsqueeze(f1, -1)
            #pdb.set_trace()
            featfeat2=f1*id_feat_map
            #atr_feat=feat
            #PAE：embedding-II and embedding-III
            atr_feat=self.ASENet(f1,attribute)


            #AAL MODEL
            attr_feat_list = []
            for i in range(8):
                attr_pool = atr_feat
                attr_feat_list.append(attr_pool)
            #pdb.set_trace()
            # attention generation
            region_score_map_list = []
            attr_score_list = []
            for i in range(8):
                attn1 = id_feat_map * attr_feat_list[i].unsqueeze(2).unsqueeze(2)
                fea_score = x_pool * attr_feat_list[i]
                region_score_map_list.append(attn1)
                attr_score_list.append(fea_score)
            # regional feature fusion
            #pdb.set_trace()#b 2048 18 9
            region_score_map = torch.mean(torch.stack(region_score_map_list), 0)
            
            
            region_fea_pool,region_fea_feat = self.region(id_feat_map,region_score_map)
            
            f1 = torch.unsqueeze(region_fea_pool, -1)
            f1 = torch.unsqueeze(f1, -1)
            region1 = f1*id_feat_map
            
            f1 = torch.unsqueeze(region_fea_feat, -1)
            f1 = torch.unsqueeze(f1, -1)
            region2 = f1*id_feat_map
            
            
            # attribute feature fusion
            attr_scores = torch.cat(attr_score_list,1) #b 2048
            
            attr_fea = torch.cat(attr_feat_list, 1)  #atr_feat
            
            attr_fea = self.attrfea(attr_scores,attr_fea)
            
            f1 = torch.unsqueeze(attr_fea, -1)
            f1 = torch.unsqueeze(f1, -1)
            atr = f1*id_feat_map
            
            
            #featfeat1=id_feat_map*attr_fea.unsqueeze(2).unsqueeze(2)
            #print(feat.shape)
            #pdb.set_trace()
            pred_label = [self.__getattr__('class_%d' % c)(atr_feat) for c in range(8)]
            pred_label = torch.cat(pred_label, dim=1)
            pred_id = self.__getattr__('class_%d' % 8)(region_fea_feat)
            #print(pred_id.shape)
            #print(pred_label.shape)
            if self.training:
                #print(x_pool.shape)
                return region_fea_pool, pred_id , pred_label
            else:
                #return region_fea_pool, pred_id , pred_label
                return self.l2norm(region_fea_pool), self.l2norm(region_fea_feat)