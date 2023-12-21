import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
from torchvision.models.resnet import ResNet, BasicBlock
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel
import torchvision.transforms as transforms


class ResNet18_conv(ResNet):
    def __init__(self):
        super(ResNet18_conv, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
def unnormalize(img):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                    std = [ 1/0.5, 1/0.5, 1/0.5]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5],
                                    std = [ 1., 1., 1. ]),
                                    ])
    return invTrans(img)

class transformer_patch_feature_extractor(nn.Module):
    def __init__(self, visualize_attention=False):
        super(transformer_patch_feature_extractor, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(torch.device("cuda:0"))
        self.model.classifier=nn.Sequential()
        self.visualize=visualize_attention
    def forward(self,x):
        # inputs = self.feature_extractor(images=x, return_tensors="pt")
        dict_input={'pixel_values':x}
        outputs = self.model(**dict_input, output_attentions=self.visualize)
        att_mat1= outputs.attentions
        results=[]
        if self.visualize:
            att_mat2 = torch.stack(att_mat1).squeeze(1)
            imgs=unnormalize(x)*255


            for i in range(0,att_mat2.shape[1]):
                im=imgs[i]
                im=im.unsqueeze(-1).transpose(0,-1).squeeze(0)
                im= im.data.cpu().numpy()
                att_mat=att_mat2[:,i,:,:,:]
                # att_mat = torch.stack(att_mat)

                # print(att_mat.shape)
                # Average the attention weights across all heads.
                att_mat = torch.mean(att_mat, dim=1)
                # print(att_mat.shape)

                # To account for residual connections, we add an identity matrix to the
                # attention matrix and re-normalize the weights.
                residual_att = torch.eye(att_mat.size(1)).cuda()
                aug_att_mat = att_mat + residual_att.cuda()
                aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1).cuda()

                # Recursively multiply the weight matrices
                joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
                joint_attentions[0] = aug_att_mat[0]

                for n in range(1, aug_att_mat.size(0)):
                    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

                # Attention from the output token to the input space.
                v = joint_attentions[-1].cuda()
                grid_size = int(np.sqrt(aug_att_mat.size(-1)))
                mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
                mask = cv2.resize(mask / mask.max(), (im.shape[0],im.shape[1]))[..., np.newaxis]
                result = (mask * im).astype("uint8")
                results.append(result)
            # results=torch.stack(results)
            return outputs.last_hidden_state,results
        else:
            return outputs.last_hidden_state


class transformer_feature_extractor(nn.Module):
    def __init__(self,visualize_attention=False):
        super(transformer_feature_extractor, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').to(torch.device("cuda:0"))
        self.visualize=visualize_attention
        self.model.classifier=nn.Sequential()
    def forward(self,x):
        # inputs = self.feature_extractor(images=x, return_tensors="pt")
        dict_input={'pixel_values':x}

        outputs = self.model(**dict_input)
        if self.visualize:
            att_mat= outputs.att_mat

            att_mat = torch.stack(att_mat).squeeze(1)
            im=unnormalize(x)
            # Average the attention weights across all heads.
            att_mat = torch.mean(att_mat, dim=1)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat.size(1))
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

            # Attention from the output token to the input space.
            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
            result = (mask * im).astype("uint8")
            return outputs.logits,result
        else:
            return outputs.logits
class transformer_feature_extractor_double(nn.Module):
    def __init__(self):
        super(transformer_feature_extractor_double, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').to(torch.device("cuda:0"))
        self.model2 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').to(torch.device("cuda:0"))
        self.model.classifier=nn.Sequential()
        self.model2.classifier=nn.Sequential()

    def forward(self,x):
        # inputs = self.feature_extractor(images=x, return_tensors="pt")
        dict_input={'pixel_values':x}
        outputs = self.model(**dict_input)
        outputs2=self.model(**dict_input)
        return torch.cat([outputs.logits,outputs2.logits],dim=-1)
class resnet18_double(nn.Module):
    def __init__(self):
        super(transformer_feature_extractor, self).__init__()
        model = models.resnet18(pretrained = pretrained)
        model2=models.resnet18(pretrained=pretrained)
        self.model.fc=nn.Sequential()
        self.model2.fc=nn.Sequential()
    def forward(self,x):
        # inputs = self.feature_extractor(images=x, return_tensors="pt")

        return torch.cat([model(x),model2(x)],dim=-1)

def get_image_extractor(arch = 'resnet18', pretrained = True, feature_dim = None, checkpoint = '', visualize_attention=False):
    '''
    Inputs
        arch: Base architecture
        pretrained: Bool, Imagenet weights
        feature_dim: Int, output feature dimension
        checkpoint: String, not implemented
    Returns
        Pytorch model
    '''

    if arch == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(512, feature_dim)
    if arch == 'resnet18_double':
        model=resnet18_double()

    if arch == 'resnet18_conv':
        model = ResNet18_conv()
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())

    elif arch == 'resnet50':
        model = models.resnet50(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim)

    elif arch == 'resnet50_cutmix':
        model = models.resnet50(pretrained = pretrained)
        checkpoint = torch.load('/home/ubuntu/workspace/pretrained/resnet50_cutmix.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim)

    elif arch == 'resnet152':
        model = models.resnet152(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim)
    elif arch == 'resnet101':
        model = models.resnet101(pretrained = pretrained)
        if feature_dim is None:
            model.fc = nn.Sequential()
        else:
            model.fc = nn.Linear(2048, feature_dim)

    elif arch == 'vgg16':
        model = models.vgg16(pretrained = pretrained)
        modules = list(model.classifier.children())[:-3]
        model.classifier=torch.nn.Sequential(*modules)
        if feature_dim is not None:
            model.classifier[3]=torch.nn.Linear(4096,feature_dim)
    elif arch == "resnet18_intermediate":
        # get intermediate features
        model_ft=models.resnet18(pretrained = pretrained)
        model= torch.nn.Sequential(*list(model_ft.children())[:-2])
    elif arch == "resnet50_intermediate":
        # get intermediate features
        model_ft=models.resnet50(pretrained = pretrained)
        model= torch.nn.Sequential(*list(model_ft.children())[:-2])
    elif arch == "resnet101_intermediate":
        # get intermediate features
        model_ft=models.resnet101(pretrained = pretrained)
        model= torch.nn.Sequential(*list(model_ft.children())[:-2])
    elif arch== "transformer":
         model= transformer_feature_extractor(visualize_attention=visualize_attention)
    elif arch== "transformer_double":
         model= transformer_feature_extractor_double()
    elif arch== "transformer_patch":
        model= transformer_patch_feature_extractor(visualize_attention=visualize_attention)


    return model
