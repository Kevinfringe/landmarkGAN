class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        """
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_blocks = []
        unet_blocks.append(self.build_unet_block(ngf * 8, ngf * 8, input_nc=input_nc, innermost=True, norm_layer=norm_layer))
        for i in range(num_downs - 5):
            unet_blocks.append(self.build_unet_block(ngf * 8, ngf * 8, submodule=unet_blocks[-1], norm_layer=norm_layer, use_dropout=use_dropout))
        unet_blocks.append(self.build_unet_block(ngf * 4, ngf * 8, submodule=unet_blocks[-1], norm_layer=norm_layer))
        unet_blocks.append(self.build_unet_block(ngf * 2, ngf * 4, submodule=unet_blocks[-1], norm_layer=norm_layer))
        unet_blocks.append(self.build_unet_block(ngf, ngf * 2, submodule=unet_blocks[-1], norm_layer=norm_layer))
        self.model = self.build_unet_block(output_nc, ngf, submodule=unet_blocks[-1], outermost=True, norm_layer=norm_layer)

        # register unet blocks
        self.unet_blocks = nn.ModuleList(unet_blocks)

    def forward(self, input):
        """Standard forward"""
        x = input
        for unet_block in self.unet_blocks:
            x = unet_block(x)
        return self.model(x)

    def build_unet_block(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Builds a UnetSkipConnectionBlock"""
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=not norm_layer)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)
            return nn.Sequential(downconv, uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=not)
