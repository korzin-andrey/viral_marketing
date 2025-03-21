import numpy as np


class SIRModelOutput:
    def __init__(self, t, S, I, R):
        self.t = t
        self.S = S
        self.I = I
        self.R = R
        self.daily_incidence = None
        self.weekly_incidence = None
        self.calculate_incidence()

    def pad_array_to_multiple_of_seven(self, arr):
        '''
        Auxiliary function used for padding array of daily data by zeroes for converting
        to weekly data
        '''
        current_size = len(arr)
        new_size = (current_size + 6) // 7 * 7
        padding_needed = new_size - current_size
        padded_array = np.pad(arr, (0, padding_needed),
                              mode='constant', constant_values=0)
        return padded_array

    def calculate_incidence(self):
        self.incidence = [0 if index == 0 else (
            self.S[index-1] - self.S[index]) for index in range(len(self.S))]
        daily_incidence_padded = self.pad_array_to_multiple_of_seven(
            self.incidence)
        self.weekly_incidence = daily_incidence_padded.reshape(
            -1, 7).sum(axis=1)
