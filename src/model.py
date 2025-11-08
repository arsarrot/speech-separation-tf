@tf.keras.utils.register_keras_serializable()
class SeparationNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [16, 32, 64, 128, 256]
        self.f = f
        self.downsample = [
            DownSample(f[0]),
            DownSample(f[1]),
            DownSample(f[2]),
            DownSample(f[3]),
        ]
        self.bottle_neck = BottleNeck(f[4])
        self.mha = MultiHeadBlock(64, 4, 1500, 64)
        self.mhb = MultiHeadBlock(32, 4, 3000, 32)
        self.upsamp = [
            UpSample(f[3]),
            UpSample(f[2]),
            UpSample(f[1]),
            UpSample(f[0]),
        ]
        self.output_conv1 = layers.Conv1D(filters = 256,
                     kernel_size=1, activation='relu', use_bias = False)
        self.output_conv2 = layers.Conv1D(filters = 256,
                     kernel_size=1, activation='relu', use_bias = False)
        self.trsp = TransposeLayer
        self.conv1g = layers.Conv1D(filters =256, kernel_size = 1, activation='relu', padding = 'same', dilation_rate = 2,  kernel_initializer='he_normal')
        self.conv2g = layers.Conv1D(filters = 256, kernel_size = 1, activation='sigmoid', dilation_rate = 2, padding = 'same', kernel_initializer='he_normal')
        self.conv1d =  layers.Conv1DTranspose(filters =2, kernel_size = 1, use_bias=False, dilation_rate = 1)
        self.drop = tf.keras.layers.Dropout(0.1)


    def call(self, x):
        c1, p1 = self.downsample[0](x)
        c2, p2 = self.downsample[1](p1)
        c3, p3 = self.downsample[2](p2)
        c4, p4 = self.downsample[3](p3)
        bn = self.bottle_neck(p4)
        mbn = self.drop(self.mha(bn))
        mc4 = self.drop(self.mhb(c4))
        u1 = self.upsamp[0](mbn, mc4)
        u2 = self.upsamp[1](u1, c3)
        u3 = self.upsamp[2](u2, c2)
        u4 = self.upsamp[3](u3, c1)
        out1, out2 = self.output_conv1(u4), self.output_conv2(u4)
        out = tf.concat([out1, out2], axis = -1)
        op = self.conv1g(out)
        og = self.conv2g(out)
        out = op * og
        out = self.conv1d(out)
        return self.trsp(perm=[0, 2, 1])(out), bn, mbn, mc4
    def get_config(self):
        config = super().get_config()
        config.update({
            'f': self.f,
        })
        return config
    @classmethod
    def from_config(cls, config):
        model = cls()
        return model

model = SeparationNetwork()
input = tf.zeros(shape=(1, 24000, 1), dtype=tf.float32)
_ = model(input)
model.summary()