import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsefool import sparsefool
from utils import valid_bounds
from PIL import Image
import os
from torchvision.utils import  save_image
import time
from utils import nnz_pixels


def sparsefool_generate(input_path,output_path,num=100,delta=100 ,max_iter =25, fixed_iter=False):

    # Check for cuda devices (GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load a pretrained model
    # Here we use resnet18 as target model
    net = torch.load('./Model_trained/animals10_resnet18.pth', map_location=torch.device('cpu'))

    net = net.to(device)
    net.eval()

    listing = os.listdir(input_path)
    existed_file_list = os.listdir(output_path)
    print(existed_file_list)
    cnt = 0
    cnt_correct = 0
    num_pixel_modified = []
    for image_name in listing:

        print(image_name)
        # Generate output file name and path for later the image saving
        if image_name[-3:] == 'jpg' or image_name[-3:] == 'bmp':
            output_file_path = (output_path+str(delta)+'_'+image_name)[:-3]+'bmp'
            output_file_path_diff = (output_path+'diff_'+str(delta)+'_'+image_name)[:-3] + 'bmp'
            output_file_name = str(delta)+'_'+image_name[:-3]+'bmp'
        elif image_name[-4:] == 'jpeg':
            output_file_path = (output_path + str(delta)+'_'+image_name)[:-4] + 'bmp'
            output_file_path_diff = (output_path+'diff_'+str(delta)+'_'+image_name)[:-4] + 'bmp'
            output_file_name = str(delta)+'_'+image_name[:-4]+'bmp'
        else:
            print('Only .jpeg of .jpg is supported')
            continue

        # Check if an image is already processed
        # This condition permits the scripts to stop half way and continue after
        if output_file_name in existed_file_list:
            print('File already processed')
            continue

        # Count of images process, used to calculate accuracy after attack
        cnt += 1

        # Load Image and Resize
        im_orig = Image.open(input_path + '/' + image_name)
        im_sz = 224
        im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)

        # Bounds for Validity and Perceptibility
        lb, ub = valid_bounds(im_orig, delta)

        # Transform image, ub and lb to PyTorch tensors for calculation
        im = transforms.Compose([transforms.ToTensor()])(im_orig)
        lb = transforms.Compose([transforms.ToTensor()])(lb)
        ub = transforms.Compose([transforms.ToTensor()])(ub)
        im = im[None, :, :, :].to(device)
        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)

        # Params
        lambda_ = 3

        # Execute SparseFool
        x_adv, r, pred_label, fool_label, loops = sparsefool(im, net, lb, ub, lambda_, max_iter,
                                                             device=device, fixed_iter=fixed_iter)

        # count the number of pixel modified and store in an array
        num_pixel_modified.append(nnz_pixels(r.cpu().numpy().squeeze()))

        # Save image generated and the noise
        save_image(x_adv, output_file_path)
        save_image(r, output_file_path_diff)

        # Print result
        if fool_label == int(image_name[0]):
            cnt_correct += 1
        print('True labels: ', image_name[0])
        print('After adding noise, classified as: ', fool_label)
        print('Num_pixel_modified: ', num_pixel_modified[-1])
        print('\n')

    # Summarize results for all images processed
    if len(num_pixel_modified) > 0:
        print("Accuracy of Network after attack: " + str(100*cnt_correct/num) + " %")
        print('Average number of pixel modified : ', sum(num_pixel_modified)/len(num_pixel_modified))
    else:
        print('No images is processed.')


if __name__ == "__main__":
    sparsefool_generate("./data/demo", "./data_output_animals10/demo/", num=1000, delta=25, max_iter=50)