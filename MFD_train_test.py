# -*- coding: utf-8 -*-


################################################
########        IMPORT LIBARIES         ########
################################################

from LibConfig import *

DatasetType = "Simulate"
ModelName = "MFD"
Pretrained = True
if Pretrained:
    reuse = 'pred_'
else:
    reuse = ''
DataDim = [1800, 301]  # Dimension of original one-shot seismic data
data_dsp_blk = (5, 1)  # Downsampling ratio of input
ModelDim = [201, 301]  # Dimension of one velocity model
label_dsp_blk = (1, 1)  # Downsampling ratio of output
dh = 10  # Space interval
BatchSize = 20  # Number of batch size
LearnRate = 1e-3  # Learning rate
Nclasses = 1  # Number of output channels
Inchannels = 29  # Number of input channels, i.e. the number of shots
SaveEpoch = 1
DisplayStep = 1  # Number of steps till outputting stats
Epochs = 200
TestBatchSize = 1
StartTestNumber = 1
dataname = 'time_record_acoustic'

if DatasetType == "Simulate":
    TrainSize = 1200
    TestSize = 200
    datafilename = 'georec_vmodel'
    truthfilename = 'vmodel'
    truthname = 'vmodel'
    train_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/train_data/SimulateData/'
    test_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/test_data/SimulateData/'
    t_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/val_data/SimulateData/'

elif DatasetType == "SEGSalt":
    TrainSize = 120
    TestSize = 20
    BatchSize = 2  # Number of batch size
    datafilename = 'georec_svmodel'
    truthfilename = 'svmodel'
    truthname = 'svmodel'
    train_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/train_data/SEGSaltData/'
    test_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/test_data/SEGSaltData/'
    t_data_dir = '../../project/2022geo/FCNVMB/FCNVMB-data/data/val_data/SEGSaltData/'

Tag_name = DatasetType + '_' + ModelName + '_' + reuse
save_path = 'Experiment_Results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
save_path = save_path + time_now + '/'
results_dir = save_path + Tag_name + 'Results/'
models_dir = save_path + Tag_name + 'Model/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
txt_path = models_dir + Tag_name + 'train_loss_log.txt'
test_txt_path = results_dir + Tag_name + 'test_loss_log.txt'
train_exel_path = models_dir + Tag_name + "train_para_loss.xlsx"  # 将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
train_epoch_exel_path = models_dir + Tag_name + "train_para_Epoch_loss.xlsx"  # 将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
test_exel_path = results_dir + Tag_name + "test_para_loss.xlsx"  # 将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
wb = openpyxl.Workbook()  # 创建Workbook()对象
ws = wb.active  # 获取默认工作薄
name = ['DatasetType', 'Pretrained', 'ModelName', 'DataDim', 'data_dsp_blk', 'ModelDim', 'label_dsp_blk', 'dh',
        'Epochs', 'TrainSize', 'TestSize', 'TestBatchSize', 'BatchSize', 'LearnRate', 'Nclasses', 'Inchannels',
        'SaveEpoch', 'DisplayStep']
data = [DatasetType, Pretrained, ModelName, str(DataDim), str(data_dsp_blk), str(ModelDim), str(label_dsp_blk), dh,
        Epochs, TrainSize, TestSize, TestBatchSize, BatchSize, LearnRate, Nclasses, Inchannels, SaveEpoch, DisplayStep]
ws.append(name)  # 往文件中写入数据
ws.append(data)
wb.save(train_exel_path)  # 保存
wb.save(test_exel_path)  # 保存
wb.close()

################################################
########             NETWORK            ########
################################################

cuda_available = torch.cuda.is_available()
Cuda = True
device = torch.device("cuda" if cuda_available else "cpu")

Tmodel = MFDTeacher(n_classes=Nclasses, in_channels=Inchannels, is_deconv=True, is_batchnorm=True)
Smodel = MFDStudent(n_classes=Nclasses, in_channels=Inchannels, is_deconv=True, is_batchnorm=True)
# model = MFNSR(n_classes=Nclasses, in_channels=Inchannels, is_deconv=True, is_batchnorm=True)
# Optimizer we want to use
optimizer = torch.optim.Adam(Smodel.parameters(), lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train
ReUse = False
if ReUse:
    print('***************** Loading the pre-trained model *****************')
    print('')
    premodel_file = 'Simulate_TrainSize1600_Epoch100_BatchSize4_LR0.001_epoch100.pkl'
    model_dict = Tmodel.state_dict()
    pretrained_dict = torch.load(premodel_file, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    Tmodel.load_state_dict(model_dict)
    print('Finished!')
    print('Finish downloading:', str(premodel_file))
    with open(txt_path, "w") as f:
        f.write('**********Loading the pre-trained model*********\n')
        f.write('Finish downloading: %s \n' % str(premodel_file))
        f.write('*******************************************\n')
if Cuda:
    net = torch.nn.DataParallel(Smodel)
    net = net.cuda()

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading Training DataSet *****************')
train_set, label_set, data_dsp_dim, label_dsp_dim = DataLoad_Train(train_size=TrainSize, train_data_dir=train_data_dir, \
                                                                   data_dim=DataDim, in_channels=Inchannels, \
                                                                   model_dim=ModelDim, data_dsp_blk=data_dsp_blk, \
                                                                   label_dsp_blk=label_dsp_blk, start=1, \
                                                                   datafilename=datafilename, dataname=dataname, \
                                                                   truthfilename=truthfilename, truthname=truthname)
# Change data type (numpy --> tensor)
train = data_utils.TensorDataset(torch.from_numpy(train_set), torch.from_numpy(label_set))
train_loader = data_utils.DataLoader(train, batch_size=BatchSize, shuffle=True)

################################################
########            TRAINING            ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('           START TRAINING                  ')
print('*******************************************')
print('*******************************************')
print()

with open(txt_path, "w") as f:
    f.write('*******************************************\n')
    f.write('           START TRAINING                  \n')
    f.write('*******************************************\n')
    f.write('Original data dimention:%s\n' % str(DataDim))
    f.write('Downsampled data dimention:%s \n' % str(data_dsp_dim))
    f.write('Original label dimention:%s\n' % str(ModelDim))
    f.write('Downsampled label dimention:%s\n' % str(label_dsp_dim))
    f.write('Training size:%d\n' % int(TrainSize))
    f.write('Traning batch size:%d\n' % int(BatchSize))
    f.write('Number of epochs:%d\n' % int(Epochs))
    f.write('Learning rate:%.5f\n' % float(LearnRate))
    f.close()

print('Original data dimention:%s' % str(DataDim))
print('Downsampled data dimention:%s ' % str(data_dsp_dim))
print('Original label dimention:%s' % str(ModelDim))
print('Downsampled label dimention:%s' % str(label_dsp_dim))
print('Training size:%d' % int(TrainSize))
print('Traning batch size:%d' % int(BatchSize))
print('Number of epochs:%d' % int(Epochs))
print('Learning rate:%.5f' % float(LearnRate))

# Initialization
loss1 = 0.0
step = np.int(TrainSize / BatchSize)
start = time.time()

for epoch in range(Epochs):
    epoch_loss = 0.0
    total_loss = 0
    total_f_score = 0
    T_loss = 0

    since = time.time()
    for i, (images, labels) in enumerate(train_loader):
        iteration = epoch * step + i + 1
        # Set Net with train condition
        Smodel.train()

        # Reshape data size
        images = images.view(BatchSize, Inchannels, data_dsp_dim[0], data_dsp_dim[1])
        labels = labels.view(BatchSize, Nclasses, label_dsp_dim[0], label_dsp_dim[1])
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        Toutputs = Tmodel(images, label_dsp_dim)
        Soutputs = Smodel(images, label_dsp_dim)
        

        # outputs = model(images, label_dsp_dim)
        CLoss=Cosine_similarity_loss(Soutputs,Toutputs,0.6)
        Tloss=F.mse_loss(Toutputs, labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * BatchSize)
        SLoss=Mean_absolute_error(Toutputs,labels)
        mseloss=0.2*CLoss+0.5*TLoss+0.3*SLoss

        # mseloss = F.mse_loss(outputs, labels, reduction='sum') / (label_dsp_dim[0] * label_dsp_dim[1] * BatchSize)
        # loss = diceloss + mseloss
        if np.isnan(float(mseloss.item())):
            raise ValueError('loss is nan while training')

        epoch_loss += mseloss.item()
        # T_loss += loss.item()
        # Loss backward propagation
        mseloss.backward()

        # Optimize
        optimizer.step()

        # Print loss
        if iteration % DisplayStep == 0:
            print(
                'Epoch: {}/{}, Iteration: {}/{} --- Training DiceLoss:{:.6f}--- Training MSELoss:{:.6f}--- Training Loss:{:.6f}'.format(
                    epoch + 1, \
                    Epochs, iteration, \
                    step * Epochs, 0, mseloss.item(), 0))
            with open(txt_path, "a") as f:
                f.write(
                    'Epoch: {}/{}, Iteration: {}/{} --- Training DiceLoss:{:.6f}--- Training MSELoss:{:.6f}--- Training Loss:{:.6f}\n'.format(
                        epoch + 1, \
                        Epochs, iteration, \
                        step * Epochs, 0, mseloss.item(), 0))
                f.close()

    print('Epoch: {:d} finished ! DiceLoss: {:.5f} ----MSELoss: {:.5f} ----TLoss: {:.5f} ----F_score: {:.5f}'.format(
        epoch + 1, total_loss / i, epoch_loss / i, T_loss / i, total_f_score / i))
    with open(txt_path, "a") as f:
        f.write(
            'Epoch: {:d} finished ! DiceLoss: {:.5f} ----MSELoss: {:.5f} ----TLoss: {:.5f}  ----F_score: {:.5f}\n'.format(
                epoch + 1, total_loss / i, epoch_loss / i, T_loss / i, total_f_score / i))
        f.close()
    loss1 = np.append(loss1, epoch_loss / i)
    time_elapsed = time.time() - since
    print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    with open(txt_path, "a") as f:
        f.write('Epoch consuming time: {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        f.close()

    # Save net parameters every 10 epochs
    # if (epoch+1) % SaveEpoch == 0:
    torch.save(Smodel.state_dict(), models_dir + Tag_name + '_epoch' + str(epoch + 1) + '.pkl')
    print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))
    with open(txt_path, "a") as f:
        f.write('Trained model saved: %d percent completed\n' % int((epoch + 1) * 100 / Epochs))
        f.close()

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
with open(txt_path, "a") as f:
    f.write('Training complete in {:.0f}m  {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    f.close()

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }
SaveTrainResults(loss=loss1, SavePath=results_dir, font2=font2, font3=font3, name=Tag_name)

test_txt_path = results_dir + Tag_name + 'test_loss_log.txt'

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
model_file = models_dir + Tag_name + '_epoch' + str(Epochs) + '.pkl'
net = MFDStudent(n_classes=Nclasses, in_channels=Inchannels, \
                is_deconv=True, is_batchnorm=True)
net.load_state_dict(torch.load(model_file))
if torch.cuda.is_available():
    net.cuda()

################################################
########    LOADING TESTING DATA       ########
################################################
print('***************** Loading Testing DataSet *****************')

test_set, label_set, data_dsp_dim, label_dsp_dim = DataLoad_Test(test_size=TestSize, test_data_dir=test_data_dir, \
                                                                 data_dim=DataDim, in_channels=Inchannels, \
                                                                 model_dim=ModelDim, data_dsp_blk=data_dsp_blk, \
                                                                 label_dsp_blk=label_dsp_blk, start=StartTestNumber, \
                                                                 datafilename=datafilename, dataname=dataname, \
                                                                 truthfilename=truthfilename, truthname=truthname)

test = data_utils.TensorDataset(torch.from_numpy(test_set), torch.from_numpy(label_set))
test_loader = data_utils.DataLoader(test, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('            START TESTING                  ')
print('*******************************************')
print('*******************************************')
print()
with open(test_txt_path, "w") as f:
    f.write('model:%s\n' % str(model_file))
    f.write('*******************************************\n')
    f.write('           START TRAINING                  \n')
    f.write('*******************************************\n')
    f.close()
# Initialization
since = time.time()
TotPSNR = np.zeros((1, TestSize), dtype=float)
TotSSIM = np.zeros((1, TestSize), dtype=float)
Prediction = np.zeros((TestSize, label_dsp_dim[0], label_dsp_dim[1]), dtype=float)
GT = np.zeros((TestSize, label_dsp_dim[0], label_dsp_dim[1]), dtype=float)
total = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.view(TestBatchSize, Inchannels, data_dsp_dim[0], data_dsp_dim[1])
    labels = labels.view(TestBatchSize, Nclasses, label_dsp_dim[0], label_dsp_dim[1])
    images = images.to(device)
    labels = labels.to(device)

    # Predictions
    net.eval()
    outputs = net(images, label_dsp_dim)
    outputs = outputs.view(TestBatchSize, label_dsp_dim[0], label_dsp_dim[1])
    outputs = outputs.data.cpu().numpy()
    gts = labels.data.cpu().numpy()

    # Calculate the PSNR, SSIM
    for k in range(TestBatchSize):
        pd = outputs[k, :, :].reshape(label_dsp_dim[0], label_dsp_dim[1])
        gt = gts[k, :, :].reshape(label_dsp_dim[0], label_dsp_dim[1])
        pd = turn(pd)
        gt = turn(gt)
        Prediction[i * TestBatchSize + k, :, :] = pd
        GT[i * TestBatchSize + k, :, :] = gt
        psnr = PSNR(pd, gt)
        TotPSNR[0, total] = psnr
        ssim = SSIM(pd.reshape(-1, 1, label_dsp_dim[0], label_dsp_dim[1]),
                    gt.reshape(-1, 1, label_dsp_dim[0], label_dsp_dim[1]))
        TotSSIM[0, total] = ssim
        print('The %d testing psnr: %.2f, SSIM: %.4f ' % (total, psnr, ssim))
        with open(test_txt_path, "a") as f:
            f.write('The %d testing psnr: %.2f, SSIM: %.4f \n' % (total, psnr, ssim))
            f.close()
        total = total + 1

# Save Results
SaveTestResults(TotPSNR, TotSSIM, Prediction, GT, results_dir, Tag_name)

# Plot one prediction and ground truth
num = 0
if DatasetType == "Simulate":
    minvalue = 2000
elif DatasetType == "SEGSalt":
    minvalue = 1500
maxvalue = 4500
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }
PlotComparison(Prediction[num, :, :], GT[num, :, :], label_dsp_dim, label_dsp_blk, dh, minvalue, maxvalue, font2, font3,
               SavePath=results_dir, name=Tag_name)

# Record the consuming time
time_elapsed = time.time() - since
print('Testing complete in  {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
with open(test_txt_path, "a") as f:
    f.write('Testing complete in  {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    f.close()
