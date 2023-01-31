import numpy as np

class HyperOptimizer():
    def __init__(self, init_value, final_value, method, rate=0, decay_value=0):
        self.init_value = init_value
        self.final_value = final_value

        self.decay_value = decay_value
        self.rate = rate

        if method == "step":
            self.method = self.step_decay
        if method == "exponantial":
            self.method = self.exponantial_decay
        if method == "exponantial_reverse":
            self.method = self.exponantial_reverse
        if method == "constant":
            self.method = self.constant

    def constant(self, generation):
        return self.init_value

    def step_decay(self, generation):
        v = self.init_value - self.decay_value * generation
        return v

    def exponantial_decay(self, generation):
        # if generation % 15 == 0: 
        #     self.init_value+=0.1
            # self.final_value-=0.1

        v = (self.init_value-self.final_value) * \
            np.exp(-self.rate * (generation % 40)) + self.final_value

        return v

    def exponantial_reverse(self, generation):
        # if generation % 15 == 0: 
        #     self.init_value+=0.1
        #     self.rate-=0.01

        v = self.init_value + (self.rate*(generation % 40))**np.exp(0.5)
        
        return min(v, self.final_value)

    def update(self, generation):
        return self.method(generation)