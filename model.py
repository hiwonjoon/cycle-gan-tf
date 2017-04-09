from ops import *

def build_enc_dec(source,reuse=False) :
    #TODO: transposed conv weights cannot accept dynamic shape?
    batch_size, channels, image_size, _ = source.get_shape().as_list()

    with tf.variable_scope('encoder') as s:
        encoder_spec = [
            Conv2d('conv2d_1',channels,32,7,7,1,1),
            Lrelu(), #64,SEQ_LEN//2
        ]
        for l,(in_,out_) in enumerate([(32,64),(64,128)]):
            encoder_spec +=[
                Conv2d('conv2d_%d'%(l+2),in_,out_,3,3,2,2),
                InstanceNorm('conv2d_in_%d'%(l+2)),
                Lrelu(), #64,SEQ_LEN//2
            ]
        for l in xrange(9) :
            encoder_spec +=[
                ResidualBlock('res_%d'%(l+1),128)
            ]
    with tf.variable_scope('decoder') as s:
        decoder_spec = []
        for l,(in_,out_,size_) in enumerate([(128,64,image_size//2),(64,32,image_size)]):
            decoder_spec += [
                TransposedConv2d('tconv_%d'%(l+1),in_,[batch_size,out_,size_,size_],3,3,2,2),
                InstanceNorm('tconv_in_%d'%(l+1)),
                Lrelu()
            ]
        decoder_spec += [
            Conv2d('conv2d_1',32,3,7,7,1,1),
            lambda t : tf.nn.tanh(t,name='b_gen'),
        ]

    _t = source
    for block in encoder_spec+decoder_spec :
        if( type(block) == BatchNorm ) :
            _t = block(_t,reuse=reuse)
        else :
            _t = block(_t)
    target = _t

    return target

def build_critic(_t) :
    _, channels, _, _ = _t.get_shape().as_list()

    c_spec = []
    for l,(in_,out_) in enumerate(
        [(channels,64),(64,128),(128,256),(256,512),(512,512)]):
        c_spec +=[
            Conv2d('conv2d_%d'%(l+1),in_,out_,4,4,2,2),
            #InstanceNorm('conv2d_in_%d'%(l+1)),
            Lrelu(), #64,SEQ_LEN//2
        ]
    c_spec += [
        Linear('linear_1',512,512),
        Lrelu(),
        Linear('linear_2',512,512),
        Lrelu(),
        Linear('linear_3',512,1),
    ]

    for block in c_spec :
        _t = block(_t)
    return _t
