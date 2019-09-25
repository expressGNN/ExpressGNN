import argparse
from os.path import join as joinpath
from common.utils import get_time_stamp


cmd_opt = argparse.ArgumentParser(description='argparser')

cmd_opt.add_argument('-embedding_size', default=128, type=int, help='embedding size')
cmd_opt.add_argument('-gcn_free_size', default=64, type=int, help='embedding size of GCN concat params')
cmd_opt.add_argument('-slice_dim', default=32, type=int, help='slice dimension of posterior params')
cmd_opt.add_argument('-data_root', default='../data/kinship', help='root of data_process')
cmd_opt.add_argument('-rule_filename', default='cleaned_rules_weight_larger_than_0.9.txt', help='rule file name')
cmd_opt.add_argument('-exp_folder', default='../exp', help='folder for experiment')
cmd_opt.add_argument('-exp_name', default='default_exp', help='name of the experiment')
cmd_opt.add_argument('-model_load_path', default=None, help='path to load the trained model')
cmd_opt.add_argument('-no_train', default=0, type=int, help='set to 1 for evaluation only')
cmd_opt.add_argument('-transe_data_root', default='baselines/KB2E/data', type=str, help='path to TransE data')
cmd_opt.add_argument('-batchsize', default=32, type=int, help='batch size for training')

cmd_opt.add_argument('-trans', default=0, type=int, help='GCN transductive or inductive')
cmd_opt.add_argument('-num_hops', default=3, type=int, help='num of hops in GCN')
cmd_opt.add_argument('-num_mlp_layers', default=2, type=int, help='num of MLP layers in GCN')
cmd_opt.add_argument('-num_epochs', default=100, type=int, help='num epochs')
cmd_opt.add_argument('-num_batches', default=100, type=int, help='num batches per epoch')

cmd_opt.add_argument('-learning_rate', default=0.0005, type=float, help='learning rate')
cmd_opt.add_argument('-lr_decay_factor', default=0.5, type=float, help='learning rate decay factor')
cmd_opt.add_argument('-lr_decay_patience', default=10, type=float, help='learning rate decay patience')
cmd_opt.add_argument('-lr_decay_min', default=0.00001, type=float, help='learning rate decay min')
cmd_opt.add_argument('-patience', default=10, type=int, help='patience for early stopping')
cmd_opt.add_argument('-l2_coef', default=0.0, type=float, help='L2 coefficient for weight decay')
cmd_opt.add_argument('-observed_prob', default=0.9, type=float, help='prob for sampling observed fact')
cmd_opt.add_argument('-entropy_temp', default=1, type=float, help='temperature for entropy term')
cmd_opt.add_argument('-no_entropy', default=0, type=int, help='no entropy term in ELBO')

cmd_opt.add_argument('-rule_weights_learning', default=1, type=int, help='set 1 to learn rule weights')
cmd_opt.add_argument('-learning_rate_rule_weights', default=0.001, type=float, help='learning rate of rule weights')
cmd_opt.add_argument('-epoch_mode', default=0, type=int, help='set 1 to run in epoch mode')
cmd_opt.add_argument('-shuffle_sampling', default=1, type=int, help='set 1 to shuffle formula when sampling')
cmd_opt.add_argument('-seed', default=10, type=int, help='random seed')

cmd_opt.add_argument('-load_method', default=1, type=int, help='set 1 to load FBWN dataset')
cmd_opt.add_argument('-use_gcn', default=1, type=int, help='set 1 to use gcn')
cmd_opt.add_argument('-filter_latent', default=0, type=int, help='set 1 to filter full latent formula')
cmd_opt.add_argument('-closed_world', default=0, type=int, help='set 1 to consider facts not in fact_dict '
                                                                'as observed neg facts')

cmd_opt.add_argument('-device', default='cpu', type=str, help='run on cpu or cuda')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.exp_path = joinpath(cmd_args.exp_folder, cmd_args.exp_name, get_time_stamp())
