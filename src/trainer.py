import os


class Trainer:
    def __init__(self, args, model, train_set=None, tester=None):
        self.args = args
        self.model = model
        self.train_set = train_set
        self.set_tester(tester)
        self.save_path = args.save_path if args.save_path is not None else None
        if self.save_path is None:
            self.save_path = os.path.join(os.getcwd(), 'best_model.pt')
            pass
            #TODO: logging warning, set to default 

        print()
        #TODO: optimizer

    def set_tester(self, tester):
        self.tester = tester

    def set_data(self, train_set=None):
        self.train_set = train_set if train_set is not None

    def prepare_batch_train(self, batch_size)
        
    def one_epoch_train(self, batch_size):
        self.model.train()
        self.optimizer.zero_grad()

        for idx_batch, batch_input in enumerate(self.prepare_batch_train(batch_size)):






    def train(self, num_epoches, batch_size, train_set=None):
        set_data(train_set)


        best_score, best_epoch = 0, 0
        for idx_epoch in range(num_epoches):

            self.one_epoch_train(batch_size)
            score = self.tester.test_dev(model)

            if score >= best_score:
                best_score, best_epoch = score, idx_epoch
                torch.save(model.state_dict(), self.save_path)

        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        self.tester.test(model)
                






        

        
