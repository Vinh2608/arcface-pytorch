class Config(object):
    env = 'default'
    backbone = 'mobilefacenet'
    classify = 'softmax'
    num_classes = 1021
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False
    load_optimizer = False
    load_model = True

    train_root = '/content/arcface-pytorch/data/Datasets/VN-celeb_align_frontal_full'
    train_list = '/content/arcface-pytorch/label_train.txt'
    #val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/content/arcface-pytorch/data/Datasets/VN-celeb_align_frontal_full'
    test_list = '/content/arcface-pytorch/label_test.txt'

    lfw_root = '/content/arcface-pytorch/data/Datasets/VN-celeb_align_frontal_full'
    lfw_test_list = '/content/arcface-pytorch/data/Datasets/pairs_new_test_VNCeleb_align_frontal.txt'

    checkpoints_path = 'checkpoints/'
    checkpoints_optimizer_path = 'checkpoints_optimizer_path/sgd_s=72_m=0.2_60.pth'
    checkpoints_optimizer_save_path = 'checkpoints_optimizer_path'
    load_model_path = 'checkpoints/mobilefacenet_s=64_m=0.2batch_size=200_align_frontal__70_acc905.pth'
    test_model_path = 'checkpoints/mobilefacenet_s=72_m=0.2_60.pth'
    save_interval = 10

    train_batch_size = 200  # batch size
    test_batch_size = 100

    input_shape = (3, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
