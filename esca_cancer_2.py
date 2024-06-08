from keras.engine.topology import Layer
import keras.backend as K
if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional
    Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun'''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, \
                   self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, \
                   self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert len(x) == 2, 'Input must be a list of two tensors'
        img, rois = x[0], x[1]
        input_shape = K.shape(img)

        outputs = []
        for roi_idx in range(self.num_rois):
            x1, y1, x2, y2 = rois[0, roi_idx, :]
            x1, y1, x2, y2 = K.cast(x1, 'int32'), K.cast(y1, 'int32'), K.cast(x2, 'int32'), K.cast(y2, 'int32')

            # Crop region of interest from the input image
            rs = tf.image.resize_images(img[:, y1:y2, x1:x2, :], (self.pool_size, self.pool_size))
            outputs.append(K.max(rs, axis=(1, 2)))

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
