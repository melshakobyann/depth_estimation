import torch
import numpy as np
import depth_estimation_model
import depth_estimation_vgg_model
import cv2
import os
import glob
import random
from PIL import Image
from random import sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eigen_loss(y_pred, y):
	"""
	This loss was proposed by David Eigen and Rob Fergus in their paper - 

	Predicting Depth, Surface Normals and Semantic Labels
	with a Common Multi-Scale Convolutional Architecture
	"""
	sobel_x_kernel = torch.tensor([[[[2., 2., 4., 2., 2.],
						           [1., 1., 2., 1., 1.],
						           [0., 0., 0., 0., 0.],
						           [-1., -1., -2., -1., -1.],
						           [-2., -2., -4., -2., -2.]]]]).to(device)

	sobel_y_kernel = torch.tensor([[[[2., 1., 0., -1., -2.],
						           [2., 1., 0., -1., -2.],
						           [4., 2., 0., -2., -4.],
						           [2., 1., 0., -1., -2.],
						           [2., 1., 0., -1., -2.]]]]).to(device)




	D = y.unsqueeze(1)
	D_star = y_pred

	d = D - D_star

	sobel_x = torch.nn.functional.conv2d(d, sobel_x_kernel)
	sobel_y = torch.nn.functional.conv2d(d, sobel_y_kernel)
	n = d.numel()

	loss = ((d**2).sum()/n) - ((d.sum()**2)/(2*n**2)) + (((sobel_x**2 + sobel_y**2).sum())/n)

	return loss.to(device)



def huber_loss(y_pred, y):

    loss = torch.zeros(y.size()).to(device)

    y_sqz = torch.squeeze(y).to(device)
    y_pred_sqz = torch.squeeze(y_pred).to(device)

    c = ((torch.abs(y_sqz - y_pred_sqz))/5).max().item()
    difference = torch.abs(y_sqz - y_pred_sqz)
    mask = difference.data <= c
    loss[mask] = 0.5*(y_sqz[mask]-y_pred_sqz[mask])**2
    loss[~mask] = c * (torch.abs(y_sqz[~mask] - y_pred_sqz[~mask]) - 0.5 * c) 
     
    return loss.to(device)


def batch_load(batch_size, stereo='add'):

	depth_img_dir = '/media/vache/Data/_out/DEPTH_320'
	depth_data_path = os.path.join(depth_img_dir, '*g')
	depth_files = glob.glob(depth_data_path)
	batch_depth = random.sample(depth_files, batch_size)

	batch_left = []
	batch_right = []
	batch_main = []

	for img in batch_depth:
		batch_left.append(img[:23] + 'STEREO_LEFT_320' + img[32:])
		batch_right.append(img[:23] + 'STEREO_RIGHT_320' + img[32:])
		batch_main.append(img[:23] + 'RGB_320' + img[32:])


	if stereo == 'add':
		data_left = []
		for f1 in batch_left:
			img = cv2.imread(f1)
			img_arr = np.array(img)
			data_left.append(img_arr)


		data_stereo = []
		for i, f1 in enumerate(batch_right):
		    img = cv2.imread(f1)
		    img_arr = np.array(img)
		    left_img = np.floor(np.true_divide(data_left[i], 2)).astype(np.uint8)
		    right_img = np.floor(np.true_divide(img_arr, 2)).astype(np.uint8)
		    stereo_arr = np.array(left_img) + np.array(right_img)
		    data_stereo.append(stereo_arr.reshape(3, 180, 320))

	elif stereo == 'cat':
		data_left = []
		for f1 in batch_left:
			img = cv2.imread(f1)
			img_arr = np.array(img)
			data_left.append(img_arr)


		data_stereo = []
		for i, f1 in enumerate(batch_right):
		    img = cv2.imread(f1)
		    img_arr = np.array(img)
		    left_img = data_left[i]
		    right_img = img_arr
		    stereo_arr = np.concatenate((left_img, right_img), axis=0)
		    data_stereo.append(stereo_arr.reshape(6, 180, 320))


	data_main = []
	for f1 in batch_main:
	    img = cv2.imread(f1)
	    img_arr = np.array(img)
	    data_main.append(img_arr.reshape(3, 180, 320))


	data_depth = []
	for f1 in batch_depth:
	    img = cv2.imread(f1)
	    img_arr = np.array(img)
	    data_depth.append(img_arr[:, :, 0]/255)


	if stereo =='cat' or stereo == 'add':
		return data_stereo, data_depth

	else:
		return data_main, data_depth
	    




def train(epoch_num, model):

	optimizer = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999))

	batch, label = batch_load(batch_size=10, stereo=False)
	label_torch = torch.tensor(label).float().to(device)
	batch_torch = torch.tensor(batch).float().to(device)

	for epoch in range(epoch_num):

		preds = model.forward(batch_torch)
		print('PREDICTED')

#		loss = torch.mean((preds - label_torch)**2)
#		loss = torch.mean(huber_loss(preds, label_torch))
		loss = eigen_loss(preds, label_torch)

		print('Epoch - ', epoch, ': Loss - ', loss)

		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		torch.cuda.empty_cache()


	torch.save(model.state_dict(), '/home/vache/ML_projects/HDmap/Code/depth_NN/saved_models/depth_model_320.pth')


def pred_show(model, stereo=False):

	inp, label = batch_load(batch_size=1, stereo=stereo)
	pred = model.forward(torch.tensor(inp).float().to(device))
	label = label[0]*255
	pred_sqz = pred.squeeze().cpu()
	pred = (pred_sqz*255).detach().numpy()
	inp = inp[0]

	label_resized = cv2.resize(label, (640, 360), interpolation = cv2.INTER_AREA)
	pred_resized = cv2.resize(pred, (640, 360), interpolation = cv2.INTER_AREA)
	

	label = Image.fromarray(label_resized)
	pred = Image.fromarray(pred_resized)
	

	if stereo == 'cat':

		inp_L = np.reshape(inp[:3, :, :], (inp.shape[1], inp.shape[2], 3))
		inp_R = np.reshape(inp[3:, :, :], (inp.shape[1], inp.shape[2], 3))

		inp_resized_L = cv2.resize(inp_L, (640, 360), interpolation = cv2.INTER_AREA)
		inp_resized_R = cv2.resize(inp_R, (640, 360), interpolation = cv2.INTER_AREA)
		inp_show_L = Image.fromarray(inp_resized_L)
		inp_show_R = Image.fromarray(inp_resized_R)
		#inp_show = Image.fromarray(inp[:, :, 3:])

		label.show()
		pred.show()
		inp_show_L.show()
		inp_show_R.show()
		#inp_show.show()

	else:

		inp = np.reshape(inp, (inp.shape[1], inp.shape[2], inp.shape[0]))

		inp_resized = cv2.resize(inp, (640, 360), interpolation = cv2.INTER_AREA)
		inp_show = Image.fromarray(inp_resized)

		label.show()
		pred.show()
		inp_show.show()


def main(training=False, continue_train=True):

	print('Using', torch.cuda.get_device_name(0))
	torch.backends.cudnn.benchmark = True

	# model = depth_estimation_model.FCN(height=720, width=1280, in_channels=6, out_channels=5, kernel_size=5)
	vgg_L = depth_estimation_vgg_model.VGGNet(requires_grad=True)
	vgg_L.cuda()
	vgg_R = depth_estimation_vgg_model.VGGNet(requires_grad=True)
	vgg_R.cuda()
	vgg = depth_estimation_vgg_model.VGGNet(requires_grad=True)
	vgg.cuda()
	model = depth_estimation_vgg_model.FCNs(pretrained_net=vgg, n_class=1)
	model.cuda()

	if training:

		if continue_train:

			model.load_state_dict(torch.load('/home/vache/ML_projects/HDmap/Code/depth_NN/saved_models/depth_model_320.pth'))

			for i in range(12000):

				print('Iteration - ', i)

				train(22, model)

			pred_show(model)


		else:

			for i in range(2000):

				print('Iteration - ', i)

				train(22, model)

			pred_show(model)

	else:

		model.load_state_dict(torch.load('/home/vache/ML_projects/HDmap/Code/depth_NN/saved_models/depth_model_320.pth'))

		pred_show(model, stereo=False)




if __name__ == '__main__':

	main()






