def config(parser):

    # Exp info
    parser.add_argument('--exp_name', type=str, default='None', help='Name of the experiment')

    # Paths
    parser.add_argument('--root', type=str, default='../', help='Path to current directory')
    parser.add_argument('--data_root', type=str, default='../input_data/', help='Path to data directory')
    parser.add_argument('--mask_root', type=str, default='../input_data/masks/', help='Path to mask directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset for which compute the metrics: train, valid, test')

    parser.add_argument('--dataset_name', type=str, help='Ground truth dataset name')
    parser.add_argument('--reconstructed_file', type=str, help='Reconstructed file')

    parser.add_argument('--save_pred_dir', type=str, default='None', help='Path to the saved model directory')
    parser.add_argument('--save_metrics_dir', type=str, help='Path to the saved metrics directory')
    parser.add_argument('--metrics', nargs='+', type=str, default=None, help='Metrics to be used')   

    parser.add_argument('--device',default='cuda:0')

        # Spain mask
    parser.add_argument('--spain_mask_bool', action='store_false', help='Do you want to use the spain mask for the metrics?')
    parser.add_argument('--spain_mask', type=str, default='mask_spain_with_0.nc') 

    parser.add_argument('--print_info', action='store_true')

    