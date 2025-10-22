########################################################################################
##
##                    EMBEDDING ENVIRONMENT FOR BLOCK TESTING
##
##                              Milan Rother 2025
##
########################################################################################


class Embedding:
    """
    This is an auxiliary class for testing the algebraic components of blocks 
    by wrapping the blocks to be easily evaluated and checked.
    """

    def __init__(self, block, source, expected):
        self.block = block
        self.source = source
        self.expected = expected


    def check_SISO(self, t, dt=1.0):

        u = self.source(t)

        self.block.inputs[0] = u
        self.block.update(t)
        self.block.sample(t, dt)

        return self.block.outputs[0], self.expected(t)


    def check_MIMO(self, t, dt=1.0):

        U = self.source(t)

        for i, u in enumerate(U):
            self.block.inputs[i] = u
        self.block.update(t)
        self.block.sample(t, dt)

        _1, Y, _2 = self.block.get_all()

        return Y, self.expected(t)
