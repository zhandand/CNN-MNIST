from Semi_Supervised.Pseudo_Label import PLModel
import Semi_Supervised.DataMgr as DataMgr
from Semi_Supervised.Net import Model

if __name__ == "__main__":
    Model = Model().cuda()
    PL = PLModel(_Model=Model)

    # 加载第一轮dataloader
    round1_train_dataloader, round1_validation_dataloader = DataMgr.get_round1_dataloader()
    PL.train_round1(round1_train_dataloader, round1_validation_dataloader)

    # 加载mark轮dataloader
    unlabelled_dataloader = DataMgr.get_unlablled_dataloader()
    PL.mark(unlabelled_dataloader)

    # 加载第二轮dataloader
    round2_train_dataloader, round2_validation_dataloader = DataMgr.get_round2_dataloader()
    PL.train_round2(round2_train_dataloader, round2_validation_dataloader)
