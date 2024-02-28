# -*- coding: utf-8 -*-
"""
Path setting


"""

from LibConfig import *
from ParamConfig import *
import time
import openpyxl

####################################################
####                   FILENAMES               ####
####################################################

# Data filename
if SimulateData:
    tagD0 = 'georec_vmodel'
    tagV0 = 'vmodel'
    # tagD1 = 'Rec'
    # tagD1='time_record_viscoaco_noise'
    tagD1 = 'time_record_acoustic'
    tagV1 = 'vmodel'

else:
    tagD0 = 'georec_svmodel'
    tagV0 = 'svmodel'
    # tagD1 = 'Rec'
    tagD1 = 'time_record_acoustic'
    tagV1 = 'svmodel'

# if SimulateData:
#     if Train:
#         tagD0 = 'georec'
#         tagV0 = 'vmodel'
#         # tagD1 = 'Rec'
#         tagD1 = 'time_record_acoustic'
#         tagV1 = 'vmodel'
#     else:
#         tagD0 = 'georec'
#         tagV0 = 'vmodel'
#         # tagD1 = 'Rec'
#         tagD1 = 'time_record_acoustic'
#         tagV1 = 'vmodel'
# else:
#     if Train:
#         tagD0 = 'georec_svmodel'
#         tagV0 = 'svmodel'
#         # tagD1 = 'Rec'
#         tagD1 = 'time_record_acoustic'
#         tagV1 = 'svmodel'
#     else:
#         tagD0 = 'georec_saltv'
#         tagV0 = 'saltv'
#         # tagD1 = 'Rec'
#         tagD1 = 'time_record_acoustic'
#         tagV1 = 'svmodel'
if ReUse:
    reuse='_pred_P'
else:
    reuse='_no_pred_P'
# if Baseline:
#     baseline='Baseline_'
# else:
#     baseline=''

datafilename  = tagD0
dataname      = tagD1
truthfilename = tagV0
truthname     = tagV1

if SimulateData:
    Tag_name = 'SimulateData_'+model_name+reuse + str(Inchannels) + '_' + str(Epochs) + '_'
else:
    Tag_name = 'SEGSaltData_'+model_name+reuse + str(Inchannels) + '_' + str(Epochs) + '_'


print("model_name!!!!!!",model_name)
###################################################
####                   PATHS                  #####
###################################################
 
main_dir   = 'FCNVMB-data/'     # Replace your main path here
# main_dir   = ''     # Replace your main path here
    
## Data path
data_dir    = main_dir + 'data/'
    
# Define training/testing data directory

train_data_dir  = data_dir  + 'train_data/'        # Replace your training data path here
test_data_dir   = data_dir  + 'test_data/'         # Replace your testing data path here
t_data_dir =data_dir  + 'val_data/'
noise_data_dir =data_dir+'noise_data/'
    
# Define directory for simulate data and SEG data respectively
if SimulateData:
    train_data_dir  = train_data_dir + 'SimulateData/'
    test_data_dir   = test_data_dir  + 'SimulateData/'
    t_data_dir = t_data_dir + 'SimulateData/'
    noise_data_dir =noise_data_dir+'SimulateData/'
else:
    train_data_dir  = train_data_dir + 'SEGSaltData/'
    test_data_dir   = test_data_dir  + 'SEGSaltData/'
    t_data_dir = t_data_dir + 'SEGSaltData/'
    noise_data_dir = noise_data_dir +'SEGSaltData/'
    

    
    
## Create Results and Models path

results_dir     = main_dir + '202311/11_results/'
models_dir      = main_dir + '202311/11_models/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if SimulateData:
    results_dir = results_dir +Tag_name + 'Results/'
    models_dir = models_dir + Tag_name + 'Model/'
else:
    results_dir = results_dir + Tag_name + 'Results/'
    models_dir = models_dir + Tag_name + 'Model/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
models_dir = models_dir + time_now+'/'
results_dir = results_dir + time_now+'/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
txt_path = models_dir + Tag_name+'train_loss_log.txt'
test_txt_path = results_dir +Tag_name+ 'test_loss_log.txt'
train_exel_path = models_dir+Tag_name+ "train_para_loss.xlsx" #将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
train_epoch_exel_path = models_dir+Tag_name+ "train_para_Epoch_loss.xlsx" #将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
test_exel_path = results_dir +Tag_name+ "test_para_loss.xlsx" #将Index输出的表格；自己起名字，修改“Output_Index.xlsx”即可，记住后缀为‘.xlsx'或者'.xls'
wb = openpyxl.Workbook()  # 创建Workbook()对象
ws = wb.active  # 获取默认工作薄
name=['SimulateData', 'ReUse(pretrained)', 'model_name', 'DataDim', 'data_dsp_blk', 'ModelDim', 'label_dsp_blk', 'dh', 'Epochs','TrainSize','TestSize','TestBatchSize','BatchSize','LearnRate','Nclasses','Inchannels','SaveEpoch','DisplayStep']
data=[SimulateData,ReUse,model_name,str(DataDim),str(data_dsp_blk),str(ModelDim),str(label_dsp_blk),dh,Epochs,TrainSize,TestSize,TestBatchSize,BatchSize,LearnRate,Nclasses,Inchannels,SaveEpoch,DisplayStep]
ws.append(name)  # 往文件中写入数据
ws.append(data)
wb.save(train_exel_path)  # 保存
wb.save(test_exel_path)  # 保存
wb.close()


# Create Model name
if SimulateData:
    tagM = 'Simulate'
else:
    tagM = 'SEGSalt'
tagM0 = '_'+model_name+'Unet'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch'     + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR'        + str(LearnRate)

modelname = tagM+tagM0+tagM1+tagM2+tagM3+tagM4
# Change here to set the model as the pre-trained initialization
premodelname = 'Simulate_FCNVMBModel_TrainSize1600_Epoch100_BatchSize1_LR0.001'

