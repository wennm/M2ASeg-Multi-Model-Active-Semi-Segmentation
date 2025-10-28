
import numpy as np

import os
import time
from torchsummary import summary

import random

from torchvision.transforms import InterpolationMode

# *Personal file library
import arguments
from query_strategies import *
from function import get_results_dir, get_hms_time, get_init_seg
from tools import Timer, csv_results
from torchvision import transforms
from log import Logger
from dataset import get_dataset, get_handler
from model import get_net
import torch


def set_seed(seed):
    # Fix Python random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Fix NumPy random seed
    np.random.seed(seed)
    # Fix PyTorch random seed
    torch.manual_seed(seed)
    # Fix random seeds for all GPU devices
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fix randomness for specific operations to ensure experimental reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # *Loop multiple times using different initial seeds, complete framework
    # -Initial parameter processing part, start timing with Timer
    time_start = time.time()
    T = Timer()  # Program start time
    args.timer = T
    args.repeat_times = 1
    repeat_times = args.repeat_times  # Total number of loop iterations

    # ~Three lists store corresponding classes after each AL learning process
    list_train_loss = []
    list_train_acc = []

    # Some content used later
    samples_count_all = []
    samples_count_total_all = []
    str_train_result = []
    times_sampling_all = []
    n_budget_used_all = []
    dice_fin_all = []
    dice_fin = []
    iou_fin = []
    f1_fin = []
    hausdorff95_fin = []
    sensitivity_fin = []
    specificity_fin = []
    iou_fin_all = []
    hausdorff95_fin_all = []
    sensitivity_fin_all = []
    specificity_fin_all = []
    f1_fin_all = []
    dice_fin_np = []
    dice_rep_fin = []

    # DATA_NAME = args.chose_dataset
    DATA_NAME = "ISIC2018"

    DATASET_NAME = DATA_NAME

    # *Initial parameter settings
    use_args_default = not args.no_argsd

    # args.model_name = 'Unet'
    args.model_name = 'TransAtt_Unet'
    # args.lr = 0.001
    args.lr = 0.01
    args.opt = 'sgd'
    # args.opt = 'adam'
    args.sch = 'cos'
    args.tmax = 80
    args.epochs = 100
    args.method_budget = "budget"
    # args.method_budget = "num"
    # args.method_budget = "prop"
    args.prop_init = 0.01
    args.prop_budget = 0.05
    args.budget_init = 10
    args.budget_once = 10
    # args.times = 1
    args.times = 4
    args.change_line = 5000
    args.data_path = r'.\datasets\isic2018_segment'
    args.P_masks_path = r'.\P_masks\ISIC2018'

    args.batch_size = 4

    args.test_batch_size = 4
    args.dataset = 'ISIC2018'
    args.acc_expected = 1
    args.method = "area_seg"

    # args.method_init = "RS"

    args.method_init = "area_seg"
    args.method_seed = "const"
    args.memory_size = 0.1
    args.save_model = True
    args.use_csv = True

    args.SSL = True
    args.SSL_Large_model = True
    args.epochs_SSL = 100

    if args.save_model:
        args.model_save_path = os.path.join(
            "./output_model/",
            f"{args.dataset}_{args.method}_init{args.prop_init:.3f}_{args.model_name}_{args.SSL_Large_model}_IM{args.method_init}.pth"
        )

    prop_lb_once = float((args.prop_budget - args.prop_init) / args.times)

    args.log_name = f"{args.dataset}_{args.method}_init{args.prop_init:.3f}_once{prop_lb_once:.3f}_{args.model_name}_{args.SSL_Large_model}_IM{args.method_init}_{args.seed}"
    log_run = Logger(args, level=args.log_level)
    args.log_run = log_run

    # Process storage folder, args.out_path represents the output location of results
    get_results_dir(args)

    MODEL_NAME = args.model_name

    # Random seed settings
    # ~ Need 3 seeds in total, first an initial seed, then randomly generate five times based on this as the seed for each loop
    method_seed = args.method_seed
    if method_seed == 'time':
        seed_global = int(time_start)
    elif method_seed == 'const':
        seed_global = args.seed

    tmp_t = T.stop()
    log_run.logger.info('Program started, basic parameter preprocessing completed, time used {:.4f} s'.format(tmp_t))
    log_run.logger.info('''Dataset used: {}, Network model: {}, Epochs: {}, Batch size: {}, Learning rate: {}, If using proportion mode, annotation budget: {},\n
                        Number of experiments to perform: {}, Global seed: {}'''.
                        format(DATA_NAME, MODEL_NAME, args.epochs, args.batch_size, args.lr, args.prop_budget,
                               repeat_times, seed_global))

    T.start()

    # -Calculate a transform list
    transforms_test_list = {
        'hyper_kvasir_seg':
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
        'ISIC2017':
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
        'ISIC2018':
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])],
    }
    transform_train_list = {

        'hyper_kvasir_seg':
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ],
        'ISIC2017':
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ],
        'ISIC2018':
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ],

    }

    tmp_transform_train_list = transform_train_list[DATASET_NAME]
    tmp_transform_test_list = transforms_test_list[DATASET_NAME]

    test_transform = transforms.Compose(tmp_transform_test_list)
    train_transform = transforms.Compose(tmp_transform_train_list)

    args.test_transform = test_transform
    args.train_transform = train_transform

    # Whether to use GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device

    X_tr, Y_tr, Y_Ptr, X_te, Y_te, X_val, Y_val = get_dataset(DATA_NAME, args.data_path, args.P_masks_path,
                                                              args.model_name, seed=seed_global)

    args.n_pool_class = None

    # Training, testing parameters
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Additional parameters
    args.train_kwargs = train_kwargs
    args.test_kwargs = test_kwargs

    tmp_t = T.stop()
    log_run.logger.info('Processing transform, cuda parameters, reading dataset, time used {:.4f} s'.format(tmp_t))

    T.start()

    # Numerical calculation part

    # -About screening budget, actually only need to know: initial budget, number of samples screened each time, final limited number of samples
    method_budget = args.method_budget
    n_pool = len(Y_tr)

    # -Given budget proportions for each part
    if method_budget == 'prop':
        times = args.times  # Sampling times
        n_init_pool = int(n_pool * args.prop_init)
        if n_init_pool == 0:
            n_init_pool = 1
        # n_init_pool = args.budget_init
        n_budget = int(n_pool * args.prop_budget)
        n_lb_once = (n_budget - n_init_pool) // times
        # acc_expected = 0.95
        acc_expected = 1  # ~Budget mode uses all budget until exhausted, regardless of acc
    # -Given initial budget, single sampling, target accuracy, no limit on sampling times;
    elif method_budget == 'num':
        n_init_pool = args.budget_init
        n_lb_once = args.budget_once
        n_budget = n_pool  # ~This mode has no budget limit, only checks if accuracy meets standard
        acc_expected = args.acc_expected
        times = (n_budget - n_init_pool) // n_lb_once
        if (n_budget - n_init_pool) % n_lb_once != 0:
            times += 1  # Consider last sampling
    # -Given numerical budget for each part
    elif method_budget == 'budget':
        times = args.times
        n_init_pool = args.budget_init
        n_lb_once = args.budget_once
        n_budget = n_init_pool + times * n_lb_once  # ~Budget depends on expected sampling times
        acc_expected = 1  # ~No expected accuracy limit, budget used until exhausted
        # acc_expected = 0.58  # ~No expected accuracy limit, budget used until exhausted

    # args.log_name = f"{args.dataset}_{args.method}_init{args.prop_init:.3f}_once{prop_lb_once:.3f}"

    method_init = args.method_init
    log_run.logger.info(
        '''In this experiment, training set sample count: {}; Initial labeled count: {}; Initial sample screening method: {}; Total budget: {}; Single sampling labeled count: {}; Single proportion: {}; Expected accuracy: {}'''.format(
            n_pool, n_init_pool, method_init, n_budget, n_lb_once, prop_lb_once, acc_expected))

    for repeat_id in range(repeat_times):
        n_budget_used = 0
        repeat_round = repeat_id + 1
        time_start_round = time.time()
        # -Start a complete AL iteration from scratch
        # Add csv classes to args, recording loss changes etc. during training process iteration;
        csv_tmp_trloss = 'train_loss_round{}.csv'.format(repeat_round)
        csv_tmp_tracc = 'train_acc_round{}.csv'.format(repeat_round)
        csv_record_trloss = csv_results(args, csv_tmp_trloss)
        csv_record_tracc = csv_results(args, csv_tmp_tracc)
        args.csv_record_trloss = csv_record_trloss
        args.csv_record_tracc = csv_record_tracc
        csv_record_trloss.write_title(['sampling_time', 'epoch', 'loss'])
        csv_record_tracc.write_title(
            ['sampling_time', 'sampled_count', 'dice', 'IOU', 'hd95', 'Sensitivity', 'Specificity', 'f1_score'])

        # -Current experiment seed
        SEED = args.seed

        # Initialize labeled set
        idxs_lb = np.zeros(n_pool, dtype=bool)
        idxs_tmp = np.arange(n_pool)
        np.random.shuffle(idxs_tmp)
        smp_idxs = get_init_seg(args, idxs_tmp, n_init_pool, X_tr, Y_Ptr, SEED)
        idxs_lb[smp_idxs] = True
        # args.class_weight =create_class_weights(n_pool)

        # Load network model etc.
        handler = get_handler(DATASET_NAME, args.model_name)
        handler_SSL = get_handler("SSL_Dataset", args.model_name)
        net = get_net(args.model_name)

        # net.to(device)
        # summary(net, input_size=(3, 224, 224), batch_size=args.batch_size)
        net = net.to(device)

        idxs_real = [i for i, val in enumerate(idxs_lb) if val]
        log_run.logger.info(f'Selected sample init indices: {smp_idxs}')
        # Strategy selection

        strategy = Area_Seg(X_tr, Y_tr, Y_Ptr, X_val, Y_val, idxs_lb, net, handler, args, device)

        # *Training start
        log_run.logger.info(
            'dataset is {},\n seed is {}, \nstrategy is {}\n'.format(DATA_NAME, SEED, type(strategy).__name__))
        # Some parameters for counting, first initialize

        # *First training
        rd = 0  # Record loop sampling times, round
        args.sampling_time = rd
        n_budget_used += n_init_pool
        args.n_budget_used = n_budget_used

        output_model = strategy.train_segmentation(idxs_real=idxs_real)
        if args.save_model:
            strategy.update_model(args.model_save_path)
        dice_tmp, iou_tmp, hausdorff95_tmp, sensitivity_tmp, specificity_tmp, f1_tmp = strategy.predict_segmentation(
            X_te, Y_te)

        dice = []
        iou = []
        hausdorff95 = []
        sensitivity = []
        specificity = []
        f1 = []
        dice.append(dice_tmp)
        iou.append(iou_tmp)
        hausdorff95.append(hausdorff95_tmp)
        sensitivity.append(sensitivity_tmp)
        specificity.append(specificity_tmp)
        f1.append(f1_tmp)

        samples_count = []
        samples_count_total = []
        samples_count_total.append(n_budget_used)
        samples_count.append(args.budget_once)
        # log_run.logger.info(
        #     'Sampling loop: {}, samples selected this loop by class: {}, overall proportion: {}'.format(rd, n_lb_once, tmp_total_props))


        # Initialize counter for consecutive accuracy non-improvement
        no_improvement_count = 0
        best_dice = float('-inf')  # Initialize best accuracy as negative infinity

        while n_budget_used < n_budget and dice_tmp < acc_expected and no_improvement_count < 20:
            rd = rd + 1
            # *First sample according to screening strategy, modify labels
            # ~ n_lb_use represents budget to use in current iteration, consider each budget and remaining
            if rd != times:
                n_lb_use = n_lb_once
            else:
                n_lb_use = n_budget - n_budget_used
            n_budget_used += n_lb_use
            args.sampling_time = rd
            args.n_budget_used = n_budget_used

            smp_idxs = strategy.query(n_lb_use, idxs_real)
            idxs_lb[smp_idxs] = True

            log_run.logger.info(f"Selected samples: {smp_idxs}")
            idxs_lb[smp_idxs] = True
            idxs_real = [i for i, val in enumerate(idxs_lb) if val]

            log_run.logger.info(
                'Sampling loop: {}, samples selected this loop: {}'.format(rd, n_lb_once))

            # *Oracle labeling phase, and training
            strategy.update(idxs_lb)
            # strategy.update(idxs_lb)
            if args.save_model:
                strategy.update_model(args.model_save_path)
            output_model = strategy.train_segmentation(idxs_real=idxs_real)
            if args.save_model:
                strategy.update_model(args.model_save_path)
            # *Test results
            # acc_tmp = strategy.predict(X_te, Y_te)
            dice_tmp, iou_tmp, hausdorff95_tmp, sensitivity_tmp, specificity_tmp, f1_tmp = strategy.predict_segmentation(
                X_te, Y_te)
            dice.append(dice_tmp)
            iou.append(iou_tmp)
            hausdorff95.append(hausdorff95_tmp)
            sensitivity.append(sensitivity_tmp)
            specificity.append(specificity_tmp)
            f1.append(f1_tmp)



        # * Calculate representativeness of current labeled set, process prediction results
        idxs_untrain = np.arange(n_pool)[~idxs_lb]
        if len(X_tr[idxs_untrain]) == 0 or len(Y_tr[idxs_untrain]) == 0:
            dice_rep_tmp = 0
            iou_rep_tmp = 0
            hausdorff95_rep_tmp = 0
            sensitivity_rep_tmp = 0
            specificity_rep_tmp = 0
            f1_rep_tmp = 0
        else:
            if args.save_model:
                strategy.update_model(args.model_save_path)
            dice_rep_tmp, iou_rep_tmp, hausdorff95_rep_tmp, sensitivity_rep_tmp, specificity_rep_tmp, f1_rep_tmp = strategy.predict_segmentation(
                X_tr[idxs_untrain], Y_tr[idxs_untrain])

        if args.SSL:
            # last_pseudo_label_idxs=strategy.SSL(idxs_untrain)
            log_run.logger.info(
                "Before labeling pseudo-labels, dice on remaining training set: {}".format(dice_rep_tmp))
            if args.SSL_Large_model:
                last_pseudo_label_idxs = strategy.SSL_large_model(idxs_untrain)
            else:
                last_pseudo_label_idxs = strategy.SSL(idxs_untrain)
            idxs_real = [i for i, val in enumerate(idxs_lb) if val]
            idxs_lb[last_pseudo_label_idxs] = True
            # *Oracle labeling phase, and training
            strategy.update(idxs_lb)
            if args.save_model:
                strategy.update_model(args.model_save_path)
            output_model = strategy.train_segmentation(idxs_real=idxs_real)
            if args.save_model:
                strategy.update_model(args.model_save_path)
            dice_tmp, iou_tmp, hausdorff95_tmp, sensitivity_tmp, specificity_tmp, f1_tmp = strategy.predict_segmentation(
                X_te, Y_te)
            dice.append(dice_tmp)
            iou.append(iou_tmp)
            hausdorff95.append(hausdorff95_tmp)
            sensitivity.append(sensitivity_tmp)
            specificity.append(specificity_tmp)
            f1.append(f1_tmp)

        dice_rep_fin.append(dice_rep_tmp)

        dice_fin.append(dice)
        iou_fin.append(iou)
        hausdorff95_fin.append(hausdorff95)
        sensitivity_fin.append(sensitivity)
        specificity_fin.append(specificity)
        f1_fin.append(f1)

        csv_record_trloss.close()
        csv_record_tracc.close()

        samples_count_all.append(samples_count)
        samples_count_total_all.append(samples_count_total)
        times_sampling_all.append(rd + 1)  # ~Record total sampling times, initialization also counts
        n_budget_used_all.append(n_budget_used)

        list_train_loss.append(csv_record_trloss)
        list_train_acc.append(csv_record_tracc)
        dice_fin_all.append(dice_tmp)
        iou_fin_all.append(iou_tmp)
        hausdorff95_fin_all.append(hausdorff95_tmp)
        sensitivity_fin_all.append(sensitivity_tmp)
        specificity_fin_all.append(specificity_tmp)
        f1_fin_all.append(f1_tmp)

        time_use_round = time.time() - time_start_round
        h, m, s = get_hms_time(time_use_round)

        str_train_reslut_tmp = 'Experiment {} (out of {}) completed, time used: {}h {}min {:.4f}s, seed used this experiment: {}, final budget used: {}, prediction accuracy on remaining training set: {},\nExperimental prediction dice coefficient: {},\nExperimental iou: {},\nExperimental hausdorff95: {},\nExperimental sensitivity: {},\nExperimental specificity: {},\nExperimental f1: {}'.format(
            repeat_round, repeat_times, h, m, s, SEED, n_budget_used, dice_rep_tmp, dice, iou, hausdorff95, sensitivity,
            specificity, f1)
        str_train_result.append(str_train_reslut_tmp)
        log_run.logger.info(str_train_reslut_tmp)

    # * Plotting, display results section
    T.start()


    # * Process data, get some final results
    n_times_fin = min(times_sampling_all)

    for i in range(repeat_times):
        dice_fin_np.append(dice_fin[i][:n_times_fin])
    dice_fin_np = np.array(dice_fin_np)
    acc_avg_fin = np.average(dice_fin_np, axis=0)

    tmp_t = T.stop()
    log_run.logger.info('Plotting time: {:.4f} s'.format(tmp_t))
    time_used = time.time() - time_start
    h, m, s = get_hms_time(time_used)
    log_run.logger.info(
        'Log storage path: {}; Experimental results storage path: {}'.format(args.log_run.filename, args.out_path))
    log_run.logger.info(
        'Training completed, sampling method used: {};\nExperimental results:'.format(type(strategy).__name__))
    for str in str_train_result:
        log_run.logger.info(str)
    log_run.logger.info(
        'Initial experiment seed: {}, initial annotation budget: {}, initial sampling method: {}; single sampling budget: {};\nMinimum sampling times: {}; final average budget: {}; average prediction accuracy: {}; average accuracy on remaining training set: {}\nAverage prediction results: {};\nTotal time used: {}h {}min {:.4f}s'.format(
            seed_global, n_init_pool, method_init, n_lb_once, n_times_fin, np.mean(n_budget_used_all),
            np.mean(dice_fin_all), np.mean(dice_rep_fin), acc_avg_fin.tolist(), h, m, s))





if __name__ == '__main__':
    args = arguments.get_args()

    args.seed = 42
    set_seed(args.seed)
    main(args)