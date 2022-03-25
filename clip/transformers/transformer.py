from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Conv1D, Dense, concatenate, Input, add, Dropout, LayerNormalization, Embedding
import tensorflow as tf
import numpy as np
from . import attention


class EncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, att_module="MHA", **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.p_d = embed_dim
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self.att_module = att_module
        
        if (att_module == "MHA"):
            self.att = attention.MHAttention(num_heads=num_heads, key_dim=embed_dim)
        elif (att_module == "x-linear"):
            self.att = attention.X_Linear_Attention(num_heads=num_heads, key_dim=embed_dim)
        
        self.ffn = Sequential(
            #[Conv1D(ff_dim, kernel_size=1, activation=tf.nn.gelu), 
            # Conv1D(embed_dim, kernel_size=1),]
            [Dense(ff_dim, activation=tf.nn.gelu), 
             Dense(embed_dim),]
        )
        #self.layernorm0 = LayerNormalization(epsilon=1e-6)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({"embed_dim":self.p_d, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config

        
    def call(self, inputs, training=False):
        #inputs = self.layernorm0(inputs)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(add([inputs, attn_output]))
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(add([out1, ffn_output]))
    

class DecoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, att_module="MHA", **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.e_d = embed_dim
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self.att_module = att_module
        
        if (att_module == "MHA"):
            self.att1 = attention.MHAttention(num_heads=num_heads, key_dim=embed_dim)
            self.att2 = attention.MHAttention(num_heads=num_heads, key_dim=embed_dim)
        elif (att_module == "x-linear"): 
            self.att1 = attention.X_Linear_Attention(num_heads=num_heads, key_dim=embed_dim)
            self.att2 = attention.X_Linear_Attention(num_heads=num_heads, key_dim=embed_dim)
            
        self.ffn = Sequential(
            #[Conv1D(ff_dim, kernel_size=1, activation=tf.nn.gelu), 
            # Conv1D(embed_dim, kernel_size=1),]
            [Dense(ff_dim, activation=tf.nn.gelu), 
             Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({"embed_dim":self.e_d, "num_heads":self.n_h, "ff_dim":self.f_d, "rate":self.rate})
        return config

    def call(self, inputs, encoder_output, look_ahead_mask, padding_mask, training=None):
        y, attn_output1 = self.att1(inputs, inputs, attention_mask=look_ahead_mask, return_attention_scores=True)
        y = self.dropout1(y, training=training)
        y = add([inputs, y])                
        out1 = self.layernorm1(y)
        
        y, attn_encoder = self.att2(out1, encoder_output, attention_mask=padding_mask, return_attention_scores=True)
        y = self.dropout2(y, training=training)
        y = add([out1, y])                
        out2 = self.layernorm1(y)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        final_output =  self.layernorm2(out2 + ffn_output)
        
        return final_output, attn_output1, attn_encoder

    
"""
Encoder
    n: number of encoder blocks
    embed_dim: dimension of the embeddings
    max_length: max length of the sequences
    num_heads: number of heads of multihead attention
    ff_dim: size of the feedforward layers
    rate: Dropout rate
    pos_encodding: Include or not a positional encoding
    att_module: "MHA" for MultiHeadAttention, "x-linear" for X-Linear attention
    
"""
class Encoder(Layer):
    def __init__(self, n, embed_dim, max_length, num_heads, ff_dim, rate=0.1, pos_encoding=False, att_module="MHA", **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.n = n        
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self._layers = [EncoderBlock(embed_dim, num_heads, ff_dim, rate=0.1, att_module=att_module) for _ in range(n)]
        self.pos_encoding = pos_encoding
        self.att_module = att_module
        self.pe = positional_encoding(self.max_length, self.embed_dim)
        
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"n": self.n, "embed_dim":self.embed_dim, "max_length": self.max_length, "num_heads":self.n_h, "ff_dim":self.f_d, "pos_encoding":self.pos_encoding, "att_module":self.att_module, "rate":self.rate})
        return config
    
    def call(self, x, training=False):
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))    
        if (self.pos_encoding == True):
            x = x + self.pe[:, :tf.shape(x)[1], :]
        
        for layer in self._layers:
            x = layer(x, training)
        return x

    
"""
Decoder
    n: number of decoder blocks
    embed_dim: dimension of the embeddings
    max_length: max length of the sequences
    num_heads: number of heads of multihead attention
    ff_dim: size of the feedforward layers
    rate: Dropout rate
    att_module: "MHA" for MultiHeadAttention, "x-linear" for X-Linear attention
    
"""
class Decoder(Layer):
    def __init__(self, n, embed_dim, max_length, num_heads, ff_dim, rate=0.1, att_module="MHA", **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.n = n
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.n_h = num_heads
        self.f_d = ff_dim
        self.rate = rate
        self.att_module = att_module
        self._layers = [DecoderBlock(embed_dim, num_heads, ff_dim, rate=0.1, att_module=att_module) for _ in range(n)]
        self.pe = positional_encoding(self.max_length, self.embed_dim)
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"n": self.n, "embed_dim":self.embed_dim, "max_length": self.max_length, "num_heads":self.n_h, "ff_dim":self.f_d, "att_module":self.att_module, "rate":self.rate})
        return config
    
    def call(self, x, encoder_output, look_ahead_mask, padding_mask, training):      
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = x + self.pe[:, :tf.shape(x)[1], :]
        
        for layer in self._layers:
            x, self_att, enc_att = layer(x, encoder_output, look_ahead_mask, padding_mask, training)

        return x, self_att, enc_att

    
def get_transformer(n=6, embed_dim=300, max_length=49, roi_dim=512, roi_num=30, num_heads=6, ff_dim=512, vocab_size=10000):
    """
    n: number of encoders/decoders
    embed_dim: dimension of the embeddings
    max_length: max length of the sequences
    roi_dim: dimension od the regions of interest of the images
    roi_num: number of the regions of interest per image
    num_heads: number of heads of multihead attention
    ff_dim: size of the feedforward layers
    vocab_size: vocab size
    
    Returns the model of the transformer
    
    """
    enc = Encoder(n, roi_dim, num_heads, ff_dim)
    dec = Decoder(n, embed_dim, max_length, num_heads, ff_dim)

    # image input
    enc_input = Input(shape=(roi_num, roi_dim,), name="source_input")
    # Caption input
    target_input = Input(shape=(max_length, embed_dim,), name="target_input")

    decoder_mask, cross_att_mask = create_masks(enc_input, target_input)

    enc_output = enc(enc_input)
    dec_output = dec(target_input, enc_output, decoder_mask, cross_att_mask)

    fin_output = TimeDistributed(Dense(vocab_size, activation='softmax', use_bias=False), name="output")(dec_output)

    train_model = Model(inputs=[enc_input, target_input], outputs=fin_output)
    return train_model





# =========================================
#   M A S K S 
# =========================================
def create_padding_mask(seq):
    """
    For self-attention
    seq shape(bs, max_length, emb_dim)
    output shape (bs, max_length, max_length)
    """
    mask = tf.cast(tf.not_equal(seq, 0), tf.bool)
    mask = tf.reduce_any(mask, 2)
    mask = tf.repeat(mask, tf.shape(seq)[1], 0)
    mask = tf.reshape(mask, (-1, tf.shape(seq)[1], tf.shape(seq)[1]))
    return tf.cast(mask, tf.float32)


def create_cross_padding_mask(seq, target_seq):
    """
    For cross-attention
    seq shape(bs, k, image_features)
    target_seq(bs, max_length, emb_dim)
    output shape (bs, max_length, k)
    """
    mask = tf.cast(tf.not_equal(target_seq, 0), tf.bool)
    mask = tf.reduce_any(mask, 2)
    #mask = tf.repeat(mask, seq.shape[1], 0)
    mask = tf.repeat(mask, tf.shape(seq)[1], 0)
    mask = tf.reshape(mask, (-1, tf.shape(seq)[1], tf.shape(target_seq)[1]))
    mask = tf.transpose(mask, [0, 2, 1])
    return mask


def create_look_ahead_mask(seq):
    """
    seq shape(bs, max_length, emb_dim)
    output 2D matrix of shape (bs, max_length, max_length) with ones on the diagonal and below.
    """
    size = tf.shape(seq)[1]
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.repeat(mask, tf.shape(seq)[0], 0)
    return mask


def create_masks(seq, target_seq):
    """
    Create the masks for the decoder where it can not see the future words.
    """
    decoder_mask = create_padding_mask(target_seq)
    decoder_mask *= create_look_ahead_mask(target_seq)
    cross_att_mask = create_cross_padding_mask(seq, target_seq)
    return decoder_mask, cross_att_mask
        
    
def create_masks_looking_ahead(seq, target_seq):
    """
    Create the masks for the decoder but the decoder can see the full caption.
    """
    decoder_mask = create_padding_mask(target_seq)
    cross_att_mask = create_cross_padding_mask(seq, target_seq)
    return decoder_mask, cross_att_mask
    
# =========================================
#   P O S I T I O N A L   E N C O D I N G
# =========================================
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

@tf.autograph.experimental.do_not_convert
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches, "projection_dim":self.projection_dim})
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded