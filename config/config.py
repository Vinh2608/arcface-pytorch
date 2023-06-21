class Config(object):
    env = 'default'
    backbone = 'mobilefacenet'
    classify = 'softmax'
    num_classes = 716
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = '/content/arcface-pytorch/data/Datasets/VN-celeb1/train'
    train_list = '/content/arcface-pytorch/data/Datasets/label_train.txt'
    #val_list = '/data/Datasets/webface/val_data_13938.txt'

    #test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    #test_list = 'test.txt'

    lfw_root = '/content/arcface-pytorch/data/Datasets/VN-celeb1/test'
    lfw_test_list = '/content/arcface-pytorch/data/Datasets/pairs_new_test_VNCeleb.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'checkpoints/mobilefacenet_10.pth'
    test_model_path = 'checkpoints/mobilefacenet_10.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 20
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
