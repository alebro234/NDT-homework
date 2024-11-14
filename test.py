import numpy as np


class Fuel():

    def __init__(self, x=0.1, p=0, wt_Pu=0, BU=0):
        self.x = x
        self.p = p
        self.wt_Pu = wt_Pu
        self.BU = BU

    def tc(self, T):
        print(f"K calcolato con {T}, @ burnup {self.BU}")
        self.k = 0.1
        return self.k

class Cladding():
    pass

class Coolant():
    pass

class NuclearChannel():

    def __init__(self):
        mox = Fuel()
        clad = Cladding()
        cool = Coolant()

    def thermal_analysis(self):

        self.mox.tc()



channel = NuclearChannel()