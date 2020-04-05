import torch
import torch.nn as nn


class FCN(nn.Module):

	def __init__(self, height, width, in_channels, out_channels, kernel_size):
		super(FCN, self).__init__()

		self.height = height
		self.width = width
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size

		self.relu = nn.ReLU(inplace=True)

		self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
		self.bn_1 = nn.BatchNorm2d(out_channels)
		self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*4, kernel_size=kernel_size)
		self.bn_2 = nn.BatchNorm2d(out_channels*4)
		self.conv_3 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*8, kernel_size=kernel_size)
		self.bn_3 = nn.BatchNorm2d(out_channels*8)
		self.conv_4 = nn.Conv2d(in_channels=out_channels*8, out_channels=out_channels*16, kernel_size=kernel_size)
		self.bn_4 = nn.BatchNorm2d(out_channels*16)
		self.conv_5 = nn.Conv2d(in_channels=out_channels*16, out_channels=out_channels*32, kernel_size=kernel_size)
		self.bn_5 = nn.BatchNorm2d(out_channels*32)

		self.not_deconv_1 = nn.ConvTranspose2d(in_channels=out_channels*32, out_channels=out_channels*16, kernel_size=kernel_size)
		self.not_deconv_2 = nn.ConvTranspose2d(in_channels=out_channels*16, out_channels=out_channels*8, kernel_size=kernel_size)
		self.not_deconv_3 = nn.ConvTranspose2d(in_channels=out_channels*8, out_channels=out_channels*4, kernel_size=kernel_size)
		self.not_deconv_4 = nn.ConvTranspose2d(in_channels=out_channels*4, out_channels=out_channels, kernel_size=kernel_size)
		self.not_deconv_5 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=1, kernel_size=kernel_size)
		self.bn_0 = nn.BatchNorm2d(1)

		self.sigmoid = nn.Sigmoid()


	def forward(self, input_img):

		conv_1_out = self.bn_1(self.relu(self.conv_1(input_img)))
		conv_2_out = self.bn_2(self.relu(self.conv_2(conv_1_out)))
		conv_3_out = self.bn_3(self.relu(self.conv_3(conv_2_out)))
		conv_4_out = self.bn_4(self.relu(self.conv_4(conv_3_out)))
		conv_5_out = self.bn_5(self.relu(self.conv_5(conv_4_out)))

		not_deconv_1_out = self.bn_4(self.relu(self.not_deconv_1(conv_5_out)))
		not_deconv_2_out = self.bn_3(self.relu(self.not_deconv_2(not_deconv_1_out)))
		not_deconv_3_out = self.bn_2(self.relu(self.not_deconv_3(not_deconv_2_out)))
		not_deconv_4_out = self.bn_1(self.relu(self.not_deconv_4(not_deconv_3_out)))
		not_deconv_5_out = self.bn_0(self.relu(self.not_deconv_5(not_deconv_4_out)))

		out = self.sigmoid(not_deconv_5_out)

		return out


# model = FCN(720, 1280, 6, 6, 3)
# inp = torch.randn((1, 6, 720, 1280))

# print(model.forward(inp))