import torch as torch
import copy
from linear_solver import linear_solver
from torch.autograd import Variable
from utils import clip_image_values
from deepfool import deepfool


def sparsefool(x_0, net, lb, ub, lambda_=3., max_iter=20, epsilon=0.02, device='cuda',fixed_iter=False):
    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()

    x_i = copy.deepcopy(x_0)
    fool_im = copy.deepcopy(x_i)
    fool_label = pred_label
    loops = 0
    if fixed_iter:
        while loops < max_iter:
            normal, x_adv = deepfool(x_i, net, lambda_, device=device)

            x_i = linear_solver(x_i, normal, x_adv, lb, ub)

            fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
            fool_im = clip_image_values(fool_im, lb, ub)
            fool_label = torch.argmax(net.forward(Variable(fool_im, requires_grad=True)).data).item()

            loops += 1

        r = fool_im - x_0
        print('Number of loops: ', loops)
        return fool_im, r, pred_label, fool_label, loops
    else:
        while fool_label == pred_label and loops < max_iter:
            normal, x_adv = deepfool(x_i, net, lambda_, device=device)

            x_i = linear_solver(x_i, normal, x_adv, lb, ub)

            fool_im = x_0 + (1 + epsilon) * (x_i - x_0)
            fool_im = clip_image_values(fool_im, lb, ub)
            fool_label = torch.argmax(net.forward(Variable(fool_im, requires_grad=True)).data).item()

            loops += 1

        r = fool_im - x_0
        return fool_im, r, pred_label, fool_label, loops


def sparsefool_1(x_0, net, lb, ub, lambda_=3., max_iter=20, epsilon=0.1, device='cuda',fixed_iter=False):

    pred_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
    x_i = copy.deepcopy(x_0)
    fool_im_100 = copy.deepcopy(x_i)
    fool_im_40 = copy.deepcopy(x_i)
    fool_im_70 = copy.deepcopy(x_i)
    fool_label = pred_label
    loops = 0
    while loops < max_iter:
        normal, x_adv = deepfool(x_i, net, lambda_, device=device)

        x_i = linear_solver(x_i, normal, x_adv, lb, ub)

        fool_im_100 = x_0 + (1 + epsilon) * (x_i - x_0)
        fool_im_100 = clip_image_values(fool_im_100, lb, ub)
        fool_label = torch.argmax(net.forward(Variable(fool_im_100, requires_grad=True)).data).item()

        loops += 1
        if loops == 30:
            fool_im_40 = copy.deepcopy(fool_im_100)
        elif loops == 45:
            fool_im_70 = copy.deepcopy(fool_im_100)
    r_40 = fool_im_40 -x_0
    r_70 = fool_im_70 - x_0
    r_100 = fool_im_100 - x_0

    return fool_im_40, fool_im_70, fool_im_100, r_40, r_70, r_100, pred_label, fool_label, loops
