import numpy as np
from gnuradio import gr

class Keep_1_in_N(gr.decim_block):

    def __init__(self, N=4):
        gr.decim_block.__init__(
            self, name='Decimation Block',
            in_sig=[np.complex64],
            out_sig=[np.complex64],
            decim=N)
        self.N = N

    def work(self, input_items, output_items):
        output_items[0][:] = input_items[0][::self.N] 
        return len(output_items[0])
