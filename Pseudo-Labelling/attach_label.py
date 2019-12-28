from utils.Net import *
from utils.DataMgr import *

model = Model()
model.cuda()
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(torch.load('D:\study\Code\python_codes\CNN\\utils\model_parameter.pkl'))

testing_correct = (0.0)
generate_label = []

test_size = 0

# for data in unsup_loader:
#     test_size+=batch_size
#     x_test, y_test = data
#     outputs = model(x_test)
#     pred = torch.max(outputs.data, 1)[1].cuda().squeeze()
#     testing_correct += torch.sum(pred == y_test)
#     print("Test Accuracy is:{:.4f}%".format(100.0 * testing_correct.item() /test_size))

# for i in range(batch_size):


# img, true_label = data
# outputs = model(img)
# pred = torch.max(outputs.data, 1)[1].cuda().squeeze().cpu()
# generate_label.extend(pred)
# true_l = true_label[i].item()
# print(pred[i].item(),true_l)

print("generate labels finished")

generate_label = [label.numpy().item() for label in generate_label]
original_img = dataset.getImg()

# label_to_file(generate_label,original_img,generate_file="mnist_generate.csv")
print("write to file finished")
