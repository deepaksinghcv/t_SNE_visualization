import time
import multiprocessing
from PIL import Image
from PIL import ImageFilter
import torch
import torchvision.models as models

def custom_image_processor(img_path):
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval()
    with torch.no_grad():
        test_tensor = torch.ones(1,3,512,512)
        z = resnet50(test_tensor)
        print(z)
#     im = Image.open(img_path)
#     out = im.filter(ImageFilter.BLUR)
# #     out = im.filter(ImageFilter.CONTOUR)
# #     out = im.filter(ImageFilter.DETAIL)
# #     out = im.filter(ImageFilter.EDGE_ENHANCE)
# #     out = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
# #     out = im.filter(ImageFilter.EMBOSS)
# #     out = im.filter(ImageFilter.FIND_EDGES)
# #     out = im.filter(ImageFilter.SHARPEN)
# #     out = im.filter(ImageFilter.SMOOTH)
# #     out = im.filter(ImageFilter.SMOOTH_MORE)
#     print(f'Processed image: {img_path}\n')
    return 1

def multiprocessing_func(x):
    output_img = custom_image_processor(x)



train_file = open("./train_file_list.txt", 'r')
file_path_list = [line.rstrip() for line in train_file.readlines()]

starttime = time.time()
pool = multiprocessing.Pool(processes=40)
pool.map(multiprocessing_func,file_path_list)
pool.close()
print('Time taken = {} seconds'.format(time.time() - starttime))
