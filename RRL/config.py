class Config:
    def __init__(self):
        self.output_size = 1
        self.num_layers = 1
        self.batch_size = 64
        self.num_step = 3
        self.input_num = 10
        self.input_size = 13
        self.hidden_size = 64
        self.tc = 0  # 거래 비용
        self.lr = 0.0001
        self.steps = 10000
