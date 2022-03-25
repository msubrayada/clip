from clip.transformers import transformer
from tensorflow.keras.layers import  Input, Dense, BatchNormalization, Embedding, Conv2D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

"""
class GetEmbeddingPositions(Layer):
    def __init__(self, end_token, **kwargs):
        super(GetEmbeddingPositions, self).__init__(**kwargs)
        self.end_token = tf.cast(end_token, tf.float32)
    
    def get_config(self):
        config = super(GetEmbeddingPositions, self).get_config()        
        config.update({"end_token": self.end_token.numpy()})
        return config
    
    def call(self, inputs):                
        end_token_pos = tf.where(tf.reduce_all(tf.equal(inputs, self.end_token), axis=[2])==True)
        return end_token_pos
"""

def get_clip(img_input_shape, num_enc, num_dec, heads, out_dim, embed_dim, dec_max_length, img_patch_dim, ff_dim, rate):
    im_enc, txt_enc = get_encoders(img_input_shape, out_dim=out_dim, n1=num_enc, n2=num_dec, embed_dim=embed_dim, max_length=dec_max_length, ff_dim=ff_dim, num_heads=heads, rate=rate)
    model = clip(im_enc, txt_enc, temperature=.001)
    return model

def load_encoder(enc_file):
    custom_objects={"EncoderBlock": transformer.EncoderBlock, 
                    "Encoder": transformer.Encoder,
                    "PatchEncoder": transformer.PatchEncoder}
    return load_model(enc_file, custom_objects=custom_objects)
    
def get_encoders(img_input_shape, out_dim=512, n1=1, n2=6, embed_dim=300, max_length=50, img_patch_dim=1024, num_heads=6, ff_dim=512, rate=0.1):
    # images
    inception = InceptionV3(include_top=False, weights="imagenet", input_shape=img_input_shape)
    inception.trainable = False
        
    img_inputs = Input(shape=img_input_shape)
    img_feat = inception(img_inputs)
    img_feat = Conv2D(img_patch_dim, (1,1), activation="relu")(img_feat)
    img_feat = tf.reshape(img_feat, (-1, 64, img_patch_dim))
    # Encode patches.
    encoded_patches = transformer.PatchEncoder(64, img_patch_dim)(img_feat)
    
    img_enc = transformer.Encoder(n1, img_patch_dim, 64, num_heads, ff_dim, rate=rate)
    im = img_enc(encoded_patches)
    im = GlobalAveragePooling1D()(im)
    im = Dense(out_dim, use_bias=True)(im)
    
    # text
    
    capt_inputs = Input(shape=(max_length, embed_dim,), name='capt_input')
    
    txt_enc = transformer.Encoder(n2, embed_dim, max_length, num_heads, ff_dim, pos_encoding=True, rate=rate)
    txt = txt_enc(capt_inputs) # (bs, max_length, embed_dim)
    txt = txt[:, 0, :]
    txt = Dense(out_dim, use_bias=True)(txt)
                        
    img_model = Model(inputs=img_inputs, outputs=im)
    txt_model = Model(inputs=capt_inputs, outputs=txt)
    
    return img_model, txt_model
    

def get_dense_encoders(img_input_shape, out_dim=512, embed_dim=300, max_length=50, ff_dim=1024, rate=0.1):
    # images
    inception = InceptionV3(include_top=False, weights="imagenet", pooling="avg", input_shape=img_input_shape)
    inception.trainable = False
        
    img_inputs = Input(shape=img_input_shape)
    im = inception(img_inputs)
    im_embed = im
    
    im = Dense(ff_dim, activation="relu")(im)
    im = Dropout(rate)(im)
    im = Dense(out_dim, activation="relu")(im)
    
    
    # text
    
    capt_inputs = Input(shape=(max_length, embed_dim,), name='capt_input')
    
    txt = Dense(ff_dim, activation="relu")(capt_inputs)
    txt = Dropout(rate)(txt)
    txt = GlobalMaxPooling1D()(txt)
    txt = Dense(out_dim, activation="relu")(txt)
    
                        
    img_model = Model(inputs=img_inputs, outputs=im)
    txt_model = Model(inputs=capt_inputs, outputs=txt)
    
    return img_model, txt_model
    
    
class clip(keras.Model):
    def __init__(self, img_encoder, txt_encoder, temperature=0.001, **kwargs):
        super(clip, self).__init__(**kwargs)
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder
        self.temperature = tf.Variable(temperature, trainable=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    def get_config():
        config = super(clip, self).get_config()
        config.update({"img_encoder": self.img_encoder, "txt_encoder": self.txt_encoder, "temperature": self.temperature})
        return config
    
    def call(self, inputs, training=False):
        img_features = self.img_encoder(inputs[0], training=training)
        txt_features = self.txt_encoder(inputs[1], training=training)
        return img_features, txt_features
        
    def compute_loss(self, image_embeddings, caption_embeddings):
        v1 = tf.linalg.normalize(image_embeddings, axis=1)[0]
        v2 = tf.linalg.normalize(caption_embeddings, axis=1)[0]
        logits = (
            tf.matmul(v1, v2, transpose_b=True)
            / self.temperature
        )
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = tf.matmul(
            v1, v1, transpose_b=True
        )
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = tf.matmul(
            v2, v2, transpose_b=True
        )
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the captions using crossentropy
        captions_loss = keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the images using crossentropy
        images_loss = keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2
    
    def train_step(self, features):
        [im, cap], _ = features
        with tf.GradientTape() as tape:
            # Forward pass
            image_embeddings, caption_embeddings = self([im, cap], training=True)
            loss = self.compute_loss(image_embeddings, caption_embeddings)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
    def test_step(self, features):
        [im, cap], _ = features
        image_embeddings, caption_embeddings = self([im, cap], training=False)
        loss = self.compute_loss(image_embeddings, caption_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        

class clip2(clip):        
    """
    Clip with original loss
    """
    def compute_loss(self, image_embeddings, caption_embeddings):
        n = tf.shape(image_embeddings)[0]
        v1 = tf.linalg.normalize(image_embeddings, axis=1)[0]
        v2 = tf.linalg.normalize(caption_embeddings, axis=1)[0]
        scores = tf.matmul(v1, tf.transpose(v2)) / self.temperature
        
        labels = tf.range(n)
        scxe = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)    
        loss1 = scxe(labels, scores)
        loss2 = scxe(labels, tf.transpose(scores))
        
        return (loss1 + loss2) / 2

