"""
This script is adapted from the starter code published by the RF Challenge organizers at
https://github.com/RFChallenge/icassp2024rfchallenge
"""

import sionna as sn
import tensorflow as tf
from .rrc_helper import apply_matched_rrc_filter


# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()
# rrc filter params
span_in_symbols = 11
beta = 0.5

# QPSK (4-QAM) Constellation
bits_per_symbol = 2
constellation = sn.mapping.Constellation(
    "qam", bits_per_symbol, trainable=False)

# Mapper and Demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.channel.AWGN()


def generate_qpsk_signal(batch_size, samples_per_symbol, num_symbols=16, ebno_db=None):
    bits = binary_source(
        [batch_size, num_symbols * bits_per_symbol])  # block length
    return modulate_qpsk_signal(bits, samples_per_symbol, ebno_db)


def modulate_qpsk_signal(msg_bits, samples_per_symbol, ebno_db=None):
    x = mapper(msg_bits)
    us = sn.signal.Upsampling(samples_per_symbol)
    x_us = us(x)
    x_us = tf.pad(x_us, tf.constant(
        [[0, 0,], [samples_per_symbol//2, 0]]), "CONSTANT")
    x_us = x_us[:, :-samples_per_symbol//2]
    x_matched = apply_matched_rrc_filter(
        x_us, span_in_symbols, samples_per_symbol, beta)
    if ebno_db is None:
        y = x_matched
    else:
        no = sn.utils.ebnodb2no(
            ebno_db=ebno_db, num_bits_per_symbol=bits_per_symbol, coderate=1.0)
        y = awgn_channel([x_matched, no])
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    return y, msg_bits


def demodulate_qpsk_signal(sig, samples_per_symbol, no=1e-4, soft=False):
    x_matched = apply_matched_rrc_filter(
        sig, span_in_symbols, samples_per_symbol, beta)
    num_symbols = sig.shape[-1] // samples_per_symbol
    ds = sn.signal.Downsampling(
        samples_per_symbol, samples_per_symbol // 2, num_symbols)
    x_hat = ds(x_matched)
    x_hat /= tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    likelihood_ratios = demapper([x_hat, no])
    if soft:
        return likelihood_ratios, x_hat
    return tf.cast(likelihood_ratios > 0, tf.float32).numpy(), x_hat.numpy()
