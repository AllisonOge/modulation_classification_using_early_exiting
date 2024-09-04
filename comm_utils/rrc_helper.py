"""
This script is adapted from the starter code published by the RF Challenge organizers at
https://github.com/RFChallenge/icassp2024rfchallenge
"""

import sionna as sn


def get_rrc_filter(span_in_symbols, samples_per_symbol, beta):
    return sn.signal.RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)


def apply_matched_rrc_filter(sig, span_in_symbols, samples_per_symbol, beta):
    rrc_filter = get_rrc_filter(span_in_symbols, samples_per_symbol, beta)
    return rrc_filter(sig, padding="same")
