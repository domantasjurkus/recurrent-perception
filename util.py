import numpy as np
# import cv2
from matplotlib import pyplot as plt

def load_or_train():
    if os.path.exists('saved_models/model'):
        print('loading existing model')
        model.load_state_dict(torch.load('saved_models/model'))
    else:
        train()

def save_model():
    try:
        os.stat('saved_models')
    except:
        os.makedirs('saved_models')
    if not os.path.exists('saved_models/model'):
        torch.save(model.state_dict(), './saved_models/model')
        print("model saved")

def draw_result(lst_iter, lst_loss, lst_acc, title):
    plt.plot(lst_iter, lst_loss, '-b', label='loss')
    plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()

def plot_1st_layer_weights():
    # for plotting conv responses (for SimpleNetwork)
    ws = list(filter(lambda l: isinstance(l, nn.modules.conv.Conv2d), model.conv1.children()))
    ws = [w.weight.data for w in ws]
    grid = vutils.make_grid(ws[0], nrow=8)
    transposed = grid.permute(1, 2, 0)
    plt.imshow(transposed)
    plt.colorbar()
    plt.show()

def show_grid(minibatch):
    grid = vutils.make_grid(minibatch, nrow=6)
    transposed = grid.permute(1, 2, 0)
    plt.imshow(transposed)
    plt.colorbar()
    plt.show()

def show_tensor(tensor, layer=0):
    # print(tensor.shape)
    # x = tensor.view(-1, HEIGHT, WIDTH)
    x = tensor.detach().numpy()
    plt.imshow(x[layer, ...])
    plt.colorbar()
    plt.show()

def show_image(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()

# hardcore numpy normalisation because opencv sucks
# def normalise_histogram(img):
#     # print(img[150])
#     # print(type(img))
#     hist,bins = np.histogram(img.flatten(),256, [1, img.max()])
#     cdf = hist.cumsum()
#     # print(hist)
#     cdf_normalized = cdf * hist.max()/ cdf.max()

#     # plt.plot(cdf_normalized, color = 'b')
#     plt.hist(img.flatten(),256,[1,img.max()])
#     plt.xlim([0,256])
#     plt.legend(('cdf','histogram'), loc = 'upper left')
#     plt.show()

#     cdf_m = np.ma.masked_equal(cdf,0)
#     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#     cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
#     img2 = cdf[img]
#     hist,bins = np.histogram(img2.flatten(),256,[0,256])
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max()/ cdf.max()

#     # plt.plot(cdf_normalized, color = 'b')
#     plt.hist(img2.flatten(),256,[0,256], color = 'r')
#     plt.xlim([0,256])
#     plt.legend(('cdf','histogram'), loc = 'upper left')
#     # plt.show()
