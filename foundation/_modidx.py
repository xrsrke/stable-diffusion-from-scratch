# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/stable-diffusion-from-scratch',
                'doc_host': 'https://xrsrke.github.io',
                'git_url': 'https://github.com/xrsrke/stable-diffusion-from-scratch',
                'lib_path': 'foundation'},
  'syms': { 'foundation.attention': {},
            'foundation.core': { 'foundation.core.A': ('core.html#a', 'foundation/core.py'),
                                 'foundation.core.A.__init__': ('core.html#a.__init__', 'foundation/core.py'),
                                 'foundation.core.B': ('core.html#b', 'foundation/core.py'),
                                 'foundation.core.B.__init__': ('core.html#b.__init__', 'foundation/core.py'),
                                 'foundation.core.foo': ('core.html#foo', 'foundation/core.py')},
            'foundation.diffusion': {'foundation.diffusion.corrupt': ('diffusion.html#corrupt', 'foundation/diffusion.py')},
            'foundation.transformer': {},
            'foundation.transformer.all': { 'foundation.transformer.all.Transformer': ( 'transformer/transformer.html#transformer',
                                                                                        'foundation/transformer/all.py'),
                                            'foundation.transformer.all.Transformer.__init__': ( 'transformer/transformer.html#transformer.__init__',
                                                                                                 'foundation/transformer/all.py')},
            'foundation.transformer.attention': { 'foundation.transformer.attention.A': ( 'transformer/transformer.attention.html#a',
                                                                                          'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.A.__init__': ( 'transformer/transformer.attention.html#a.__init__',
                                                                                                   'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.MultiHeadAttention': ( 'transformer/transformer.attention.html#multiheadattention',
                                                                                                           'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.MultiHeadAttention.__init__': ( 'transformer/transformer.attention.html#multiheadattention.__init__',
                                                                                                                    'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.MultiHeadAttention.forward': ( 'transformer/transformer.attention.html#multiheadattention.forward',
                                                                                                                   'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.MultiHeadAttention.scaled_dot_product_attention': ( 'transformer/transformer.attention.html#multiheadattention.scaled_dot_product_attention',
                                                                                                                                        'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.PrepareForMultiHeadAttention': ( 'transformer/transformer.attention.html#prepareformultiheadattention',
                                                                                                                     'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.PrepareForMultiHeadAttention.__init__': ( 'transformer/transformer.attention.html#prepareformultiheadattention.__init__',
                                                                                                                              'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.PrepareForMultiHeadAttention.forward': ( 'transformer/transformer.attention.html#prepareformultiheadattention.forward',
                                                                                                                             'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.ScaleDotProductAttention': ( 'transformer/transformer.attention.html#scaledotproductattention',
                                                                                                                 'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.ScaleDotProductAttention.__init__': ( 'transformer/transformer.attention.html#scaledotproductattention.__init__',
                                                                                                                          'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention.ScaleDotProductAttention.forward': ( 'transformer/transformer.attention.html#scaledotproductattention.forward',
                                                                                                                         'foundation/transformer/attention.py'),
                                                  'foundation.transformer.attention._calculate_attention': ( 'transformer/transformer.attention.html#_calculate_attention',
                                                                                                             'foundation/transformer/attention.py')},
            'foundation.transformer.decoder': { 'foundation.transformer.decoder.Decoder': ( 'transformer/transformer.decoder.html#decoder',
                                                                                            'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.Decoder.__init__': ( 'transformer/transformer.decoder.html#decoder.__init__',
                                                                                                     'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.Decoder.forward': ( 'transformer/transformer.decoder.html#decoder.forward',
                                                                                                    'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.DecoderLayer': ( 'transformer/transformer.decoder.html#decoderlayer',
                                                                                                 'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.DecoderLayer.__init__': ( 'transformer/transformer.decoder.html#decoderlayer.__init__',
                                                                                                          'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.DecoderLayer.forward': ( 'transformer/transformer.decoder.html#decoderlayer.forward',
                                                                                                         'foundation/transformer/decoder.py'),
                                                'foundation.transformer.decoder.create_mask': ( 'transformer/transformer.decoder.html#create_mask',
                                                                                                'foundation/transformer/decoder.py')},
            'foundation.transformer.efficient_attention': { 'foundation.transformer.efficient_attention.MultiHeadAttention': ( 'transformer/transformer.efficient_attention.html#multiheadattention',
                                                                                                                               'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.MultiHeadAttention.__init__': ( 'transformer/transformer.efficient_attention.html#multiheadattention.__init__',
                                                                                                                                        'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.MultiHeadAttention.concat': ( 'transformer/transformer.efficient_attention.html#multiheadattention.concat',
                                                                                                                                      'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.MultiHeadAttention.forward': ( 'transformer/transformer.efficient_attention.html#multiheadattention.forward',
                                                                                                                                       'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.MultiHeadAttention.split_head': ( 'transformer/transformer.efficient_attention.html#multiheadattention.split_head',
                                                                                                                                          'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.ScaleDotProductAttention': ( 'transformer/transformer.efficient_attention.html#scaledotproductattention',
                                                                                                                                     'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.ScaleDotProductAttention.__init__': ( 'transformer/transformer.efficient_attention.html#scaledotproductattention.__init__',
                                                                                                                                              'foundation/transformer/efficient_attention.py'),
                                                            'foundation.transformer.efficient_attention.ScaleDotProductAttention.forward': ( 'transformer/transformer.efficient_attention.html#scaledotproductattention.forward',
                                                                                                                                             'foundation/transformer/efficient_attention.py')},
            'foundation.transformer.embedding': { 'foundation.transformer.embedding.PositionalEncoding': ( 'transformer/transformer.embedding.html#positionalencoding',
                                                                                                           'foundation/transformer/embedding.py'),
                                                  'foundation.transformer.embedding.PositionalEncoding.__init__': ( 'transformer/transformer.embedding.html#positionalencoding.__init__',
                                                                                                                    'foundation/transformer/embedding.py'),
                                                  'foundation.transformer.embedding.PositionalEncoding.forward': ( 'transformer/transformer.embedding.html#positionalencoding.forward',
                                                                                                                   'foundation/transformer/embedding.py'),
                                                  'foundation.transformer.embedding.TextEmbedding': ( 'transformer/transformer.embedding.html#textembedding',
                                                                                                      'foundation/transformer/embedding.py'),
                                                  'foundation.transformer.embedding.TextEmbedding.__init__': ( 'transformer/transformer.embedding.html#textembedding.__init__',
                                                                                                               'foundation/transformer/embedding.py'),
                                                  'foundation.transformer.embedding.TextEmbedding.forward': ( 'transformer/transformer.embedding.html#textembedding.forward',
                                                                                                              'foundation/transformer/embedding.py')},
            'foundation.transformer.encoder': { 'foundation.transformer.encoder.Encoder': ( 'transformer/transformer.encoder.html#encoder',
                                                                                            'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.Encoder.__init__': ( 'transformer/transformer.encoder.html#encoder.__init__',
                                                                                                     'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.Encoder.forward': ( 'transformer/transformer.encoder.html#encoder.forward',
                                                                                                    'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.EncoderLayer': ( 'transformer/transformer.encoder.html#encoderlayer',
                                                                                                 'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.EncoderLayer.__init__': ( 'transformer/transformer.encoder.html#encoderlayer.__init__',
                                                                                                          'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.EncoderLayer.forward': ( 'transformer/transformer.encoder.html#encoderlayer.forward',
                                                                                                         'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.PostionWiseFeedForward': ( 'transformer/transformer.encoder.html#postionwisefeedforward',
                                                                                                           'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.PostionWiseFeedForward.__init__': ( 'transformer/transformer.encoder.html#postionwisefeedforward.__init__',
                                                                                                                    'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.PostionWiseFeedForward.forward': ( 'transformer/transformer.encoder.html#postionwisefeedforward.forward',
                                                                                                                   'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.ResidualLayerNorm': ( 'transformer/transformer.encoder.html#residuallayernorm',
                                                                                                      'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.ResidualLayerNorm.__init__': ( 'transformer/transformer.encoder.html#residuallayernorm.__init__',
                                                                                                               'foundation/transformer/encoder.py'),
                                                'foundation.transformer.encoder.ResidualLayerNorm.forward': ( 'transformer/transformer.encoder.html#residuallayernorm.forward',
                                                                                                              'foundation/transformer/encoder.py')},
            'foundation.transformer.layers': { 'foundation.transformer.layers.EncoderLayer': ( 'transformer.encoder.html#encoderlayer',
                                                                                               'foundation/transformer/layers.py'),
                                               'foundation.transformer.layers.EncoderLayer.__init__': ( 'transformer.encoder.html#encoderlayer.__init__',
                                                                                                        'foundation/transformer/layers.py'),
                                               'foundation.transformer.layers.EncoderLayer.forward': ( 'transformer.encoder.html#encoderlayer.forward',
                                                                                                       'foundation/transformer/layers.py'),
                                               'foundation.transformer.layers.ResidualLayer': ( 'transformer.encoder.html#residuallayer',
                                                                                                'foundation/transformer/layers.py'),
                                               'foundation.transformer.layers.ResidualLayer.__init__': ( 'transformer.encoder.html#residuallayer.__init__',
                                                                                                         'foundation/transformer/layers.py'),
                                               'foundation.transformer.layers.ResidualLayer.forward': ( 'transformer.encoder.html#residuallayer.forward',
                                                                                                        'foundation/transformer/layers.py')},
            'foundation.transformer.positional_encoding': {},
            'foundation.unet': { 'foundation.unet.BasicUNet': ('unet.html#basicunet', 'foundation/unet.py'),
                                 'foundation.unet.BasicUNet.__init__': ('unet.html#basicunet.__init__', 'foundation/unet.py'),
                                 'foundation.unet.BasicUNet.forward': ('unet.html#basicunet.forward', 'foundation/unet.py')}}}