import numpy as np
from gnuradio import gr

class blk(gr.sync_block):

    def __init__(self, a=1.0, b=0):
        gr.sync_block.__init__(self, name='Example Block',
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.a, self.b = a, b
    
    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0] * self.a + self.b
        return len(output_items[0])

