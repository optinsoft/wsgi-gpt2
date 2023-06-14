import json
import os
import numpy as np
import tensorflow as tf

from gpt2 import model, encoder, sample

class GPT2Application(object):
    def __init__(self,
        model_name='124M',
        seed=None,
        nsamples=1,
        batch_size=1,
        length=None,
        temperature=1,
        top_k=40,
        top_p=1,
        models_dir='models'):

        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        self.model_name = model_name
        self.models_dir = models_dir
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size

        self.enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        self.hparams = hparams

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
        
        self.length = length        
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def __call__(self, environ, start_response):
        try:
            request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = environ['wsgi.input'].read(request_body_size)

        raw_text = request_body.decode('utf-8')

        if len(raw_text) > 0:
            context_tokens = self.enc.encode(raw_text)
            generated = 0
            with tf.Session(graph=tf.Graph()) as sess:
                context = tf.placeholder(tf.int32, [self.batch_size, None])
                np.random.seed(self.seed)
                tf.set_random_seed(self.seed)
                output = sample.sample_sequence(
                    hparams=self.hparams, length=self.length,
                    context=context,
                    batch_size=self.batch_size,
                    temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
                )
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(os.path.join(self.models_dir, self.model_name))
                saver.restore(sess, ckpt)

                response: str = ''

                for _ in range(self.nsamples // self.batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(self.batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(self.batch_size):
                        generated += 1
                        text = self.enc.decode(out[i])
                        if self.batch_size > 1:
                            response += "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40
                            response += "\n"
                        response += text
                        if self.batch_size > 1:
                            response += "\n"
                if self.batch_size > 1:
                    response += "=" * 80
                    response += "\n"

                response_body = response.encode('utf=8')

        else:
            response_body = f"No prompt".encode('utf-8')
        
        status = '200 OK'
        response_headers = [
            ('Content-Type', 'text/plain'),
            ('Content-Length', str(len(response_body))),
        ]

        start_response(status, response_headers)
        yield response_body
