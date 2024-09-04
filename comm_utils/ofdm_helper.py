"""
This script is adapted from the starter code published by the RF Challenge organizers at
https://github.com/RFChallenge/icassp2024rfchallenge
"""

import sionna as sn
import numpy as np
import tensorflow as tf

nfft = 64  # fft size for OFDM generation
cp_len = 16  # cyclic prefix len
ofdm_len = nfft + cp_len  # total length of OFDM symbol
code_rate = 1  # channel coding rate (1 corresponds to no coding)
n_streams_per_tx = 1

# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

# QPSK (4-QAM) constellation
bits_per_symbol = 2
constellation = sn.mapping.Constellation(
    "qam", bits_per_symbol, trainable=False)
stream_manager = sn.mimo.StreamManagement(np.array([[1]]), 1)

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.channel.AWGN()


def create_resource_grid(num_ofdm_symbols):
    return sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                fft_size=nfft,
                                subcarrier_spacing=20e6/nfft,
                                num_tx=1,
                                num_streams_per_tx=n_streams_per_tx,
                                num_guard_carriers=[4, 3],
                                dc_null=True,
                                cyclic_prefix_length=cp_len,
                                pilot_pattern=None,
                                pilot_ofdm_symbol_indices=[])


def modulate_ofdm_signal(msg_bits, resource_grid, ebno_db=None):
    # codewords = encoder(info_bits) # using uncoded bits for now
    codewords = msg_bits
    rg_mapper = sn.ofdm.ResourceGridMapper(resource_grid)
    ofdm_mod = sn.ofdm.OFDMModulator(resource_grid.cyclic_prefix_length)

    x = mapper(codewords)
    x_rg = rg_mapper(x)
    x_ofdm = ofdm_mod(x_rg)

    if ebno_db is None:
        y = x_ofdm
    else:
        no = sn.utils.ebnodb2no(ebno_db=10.0,
                                num_bits_per_symbol=bits_per_symbol,
                                coderate=code_rate,
                                resource_grid=resource_grid)
        y = awgn_channel([x_ofdm, no])
    # squeeze axis corresponding to num_tx, num_streams_per_tx (assumed to be 1)
    y = tf.squeeze(y, axis=[1, 2])
    msg_bits = tf.squeeze(msg_bits, axis=[1, 2])
    return y, msg_bits


def generate_ofdm_signal(batch_size, num_ofdm_symbols, ebno_db=None):
    resource_grid = create_resource_grid(num_ofdm_symbols)

    # Number of coded bits in a resource grid
    num_coded_bits = int(resource_grid.num_data_symbols * bits_per_symbol)
    # Number of information bits in a resource grid
    k = int(num_coded_bits * code_rate)

    bits = binary_source([batch_size, 1, n_streams_per_tx, k])
    return modulate_ofdm_signal(bits, resource_grid, ebno_db)


def demodulate_ofdm_signal(sig, resource_grid, no=1e-4):
    rg_demapper = sn.ofdm.ResourceGridDemapper(resource_grid, stream_manager)
    ofdm_demodulator = sn.ofdm.OFDMDemodulator(nfft, 0, cp_len)
    x_ofdm_demod = ofdm_demodulator(sig)
    x_demod = rg_demapper(tf.reshape(
        x_ofdm_demod, (sig.shape[0], 1, 1, -1, nfft)))
    likelihoods = demapper([x_demod, no])
    likelihoods = tf.squeeze(likelihoods, axis=[1, 2])
    return tf.cast(likelihoods > 0, tf.float32).numpy(), x_ofdm_demod.numpy()
