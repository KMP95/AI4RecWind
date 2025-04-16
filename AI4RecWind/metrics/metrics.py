import warnings
import os 
import xarray as xr
import torch 
import torch.nn as nn 
import torchmetrics.image as t_metrics 
import argparse

from config_parser_metrics import config
from mask_spain import load_mask

metric_settings = {
    'l1Error': {},
    'l2Error': {},
    'Bias_time': {},
    'Bias_space': {},
    'Corr_time': {},
    'Corr_space': {},
    'RMSE_time': {},
    'RMSE_space': {},
    'StructuralSimilarityIndexMeasure': {'torchmetric_settings': {}},
    'UniversalImageQualityIndex': {'torchmetric_settings': {}},#{'reset_real_features': True}
    'FrechetInceptionDistance': {'torchmetric_settings': {}},
                                #{'requires_image': True, 'feature': 64, 'reset_real_features': True,
                                #'subset_size': args.batch_size}
    'KernelInceptionDistance': {'torchmetric_settings': {}}
                                #{'requires_image': True, 'feature': 64, 'reset_real_features': True,
                                #'subset_size': args.batch_size},
                                #'outputs': ['mu', 'std']
    }

def check_metrics(metrics):

    metric_list = metric_settings.keys()

    invalid_metrics = [m for m in metrics if m not in metric_list]
    if invalid_metrics:
        raise ValueError('The given metric(s) ({}) not defined'.format(invalid_metrics))
        
# When used a pretrained model to further training, this function is used to remove the metrics of the epochs which is going to recalculate 
def remove_repeated_lines_metrics_csv(args, cur_epoch):

    for split in ['train', 'valid']:

        with open(os.path.join(args.save_metrics_dir, split, 'metrics_{:s}.csv'.format(split)), 'r') as f:
            lines = f.readlines()

        header = lines[0]
        
        filtered_lines = [header]

        for line in lines[1:]:
            epoch = int(line.split(',')[0])
            if epoch <= cur_epoch:
                filtered_lines.append(line)
        
        with open(os.path.join(args.save_metrics_dir, split, 'metrics_{:s}.csv'.format(split)), 'w') as f:
            f.writelines(filtered_lines)

def metrics(preds, targets, spain_mask, split, args):#, epoch):
    
    metric_dict = get_metrics(preds, targets, spain_mask, split, args)

    if args.print_info:
        print()
        print('Metrics: ')
                
        for k, metric in enumerate(metric_dict):
            if 'map' not in metric.split("/")[1]:
                print(k, '.....', metric, '.....', metric_dict[metric])
        print()

    
    if not os.path.exists(os.path.join(args.save_metrics_dir, split)):
        os.makedirs(os.path.join(args.save_metrics_dir, split))

    # Check if file exists to write header only once
    file_path = os.path.join(args.save_metrics_dir, split, f'metrics_{split}.csv')
    file_exists = os.path.exists(file_path)          

    with open(os.path.join(args.save_metrics_dir, split, 'metrics_{:s}.csv'.format(split)), 'a') as f:
            # Filter metrics to exclude map metrics
            metrics_to_write = [(metric.split("/")[1], str(metric_dict[metric])) for metric in metric_dict if 'map' not in metric.split("/")[1]]
            
            # Write header if file doesn't exist
            if not file_exists:
                f.write('experiment,' + ','.join([name for name, _ in metrics_to_write]))
                f.write('\n')
            
            # Always write the experiment name in the first column, followed by metric values
            f.write(args.exp_name + ',' + ','.join([value for _, value in metrics_to_write]))
            f.write('\n')

    return metric_dict

@torch.no_grad()
def get_metrics(preds, targets, spain_mask, split, args):
    
    metric_dict = {}

    for metric in args.metrics:

        print('Calculating {}...'.format(metric))

        settings = metric_settings[metric]

        if 'l1Error' in metric:
            
            l1_loss_t, l1_loss_spain = compute_loss(preds, targets, nn.L1Loss(reduction='mean'), spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/l1Error_total'] = l1_loss_t
            if args.spain_mask_bool:
                metric_dict[f'{split}/l1Error_spain'] = l1_loss_spain

        elif 'l2Error' in metric:
            
            l2_loss_t, l2_loss_spain = compute_loss(preds, targets, nn.MSELoss(reduction='mean'), spain_mask, args.spain_mask_bool)
            metric_dict[f'{split}/l2Error_total'] = l2_loss_t
            if args.spain_mask_bool:
                metric_dict[f'{split}/l2Error_spain'] = l2_loss_spain

        elif 'Bias_time' in metric:
            
            bias_map_time, bias_map_time_spain, bias_time, bias_time_spain = compute_bias_time(preds, targets, spain_mask, args.spain_mask_bool)
            metric_dict[f'{split}/Bias_map_time'] = bias_map_time
            metric_dict[f'{split}/Bias_time'] = bias_time
            if args.spain_mask_bool:
                metric_dict[f'{split}/Bias_map_time_spain'] = bias_map_time_spain
                metric_dict[f'{split}/Bias_time_spain'] = bias_time_spain

        elif 'Bias_space' in metric:
            
            bias_map_space, bias_map_space_spain, bias_space, bias_space_spain = compute_bias_space(preds, targets, spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/Bias_map_space'] = bias_map_space
            metric_dict[f'{split}/Bias_space'] = bias_space
            if args.spain_mask_bool:
                metric_dict[f'{split}/Bias_map_space_spain'] = bias_map_space_spain
                metric_dict[f'{split}/Bias_space_spain'] = bias_space_spain

        elif 'Corr_time' in metric:
            
            correlation_map_time, correlation_map_time_spain, correlation_time, correlation_time_spain = compute_corr_time(preds, targets, spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/Corr_map_time'] = correlation_map_time
            metric_dict[f'{split}/Corr_time'] = correlation_time
            if args.spain_mask_bool:
                metric_dict[f'{split}/Corr_map_time_spain'] = correlation_map_time_spain
                metric_dict[f'{split}/Corr_time_spain'] = correlation_time_spain

        elif 'Corr_space' in metric:
            
            correlation_map_space, correlation_map_space_spain, correlation_space, correlation_space_spain = compute_corr_space(preds, targets, spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/Corr_map_space'] = correlation_map_space
            metric_dict[f'{split}/Corr_space'] = correlation_space
            if args.spain_mask_bool:
                metric_dict[f'{split}/Corr_map_space_spain'] = correlation_map_space_spain
                metric_dict[f'{split}/Corr_space_spain'] = correlation_space_spain


        elif 'RMSE_time' in metric:
            
            rmse_map_time, rmse_map_time_spain, rmse_time, rmse_time_spain = compute_rmse_time(preds, targets, spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/RMSE_map_time'] = rmse_map_time
            metric_dict[f'{split}/RMSE_time'] = rmse_time
            if args.spain_mask_bool:
                metric_dict[f'{split}/RMSE_map_time_spain'] = rmse_map_time_spain
                metric_dict[f'{split}/RMSE_time_spain'] = rmse_time_spain

        elif 'RMSE_space' in metric:
            
            rmse_map_space, rmse_map_space_spain, rmse_space, rmse_space_spain = compute_rmse_space(preds, targets, spain_mask, args.spain_mask_bool)

            metric_dict[f'{split}/RMSE_map_space'] = rmse_map_space
            metric_dict[f'{split}/RMSE_space'] = rmse_space
            if args.spain_mask_bool:
                metric_dict[f'{split}/RMSE_map_space_spain'] = rmse_map_space_spain
                metric_dict[f'{split}/RMSE_space_spain'] = rmse_space_spain


        else:
            metric_outputs = calculate_metric(metric, preds, targets, torchmetrics_settings=settings['torchmetric_settings'])
            metric_dict[f'{split}/{metric}'] = metric_outputs

            #if len(metric_outputs) > 1:
            #    for k, metric_name in enumerate(settings['outputs']):
            #        metric_dict[f'{split}/{metric}_{metric_name}'] = metric_outputs[k]
            #else:
            #    metric_dict[f'{split}/{metric}'] = metric_outputs[0]

    return metric_dict

def compute_loss(pred, target, loss_fn, spain_mask, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    
    loss_total = loss_fn(pred, target).item()

    if spain_bool:
        # Create a mask to ignore NaN values
        # mask = ~torch.isnan(spain_mask.expand_as(pred))
        mask = spain_mask.expand_as(pred) != 0

        # Apply masks to vectors
        pred_spain = pred[mask]
        target_spain = target[mask]

        loss_spain = loss_fn(pred_spain, target_spain).item()

    return loss_total, loss_spain

def compute_bias_time(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    bias_map_time = torch.mean(pred - target, dim=0)

    # Máscara para España
    if spain_bool:
        mask_spain = mask_spain.squeeze()
        bias_map_time_spain = torch.where(mask_spain == 0, float('nan'), bias_map_time)
    else:
        bias_map_time_spain = torch.zeros_like(bias_map_time)

    return (bias_map_time, bias_map_time_spain, torch.nanmean(bias_map_time).item(), torch.nanmean(bias_map_time_spain).item())

def compute_bias_space(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the bias
    bias_map_space = torch.mean(pred_flattened - target_flattened, dim=1)

    # Spain_mask
    if spain_bool:
        mask = mask_spain.flatten() != 0
        pred_spain = pred_flattened[:, mask]
        target_spain = target_flattened[:, mask]  

        bias_map_space_spain = torch.mean(pred_spain - target_spain, dim=1)
                
    return bias_map_space, bias_map_space_spain, torch.nanmean(bias_map_space).item(), torch.nanmean(bias_map_space_spain).item()
  
def compute_corr_time(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # Calculation of the Correlation: 
    # r = Cov(X, Y)/(sigma_x * sigma_y)
    
    # Calcule the means
    pred_mean  = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)

    # Calcule the Covariance and standard deviations
    covariance = torch.mean(pred * target, dim=0) - pred_mean * target_mean
    pred_var = torch.mean(pred ** 2, dim=0) - pred_mean ** 2
    target_var = torch.mean(target ** 2, dim=0) - target_mean ** 2

    # Correlation
    corr_map_time = covariance / (torch.sqrt(pred_var) * torch.sqrt(target_var))

    # Check for pixels with standard variance approximately 0, i.e., correlation values >1 or <-1
    corr_map_time[corr_map_time>1] = float('nan')
    corr_map_time[corr_map_time<-1] = float('nan')

    # Spain_mask
    
    if spain_bool:
        mask_spain = mask_spain.squeeze()
        corr_map_time_spain = torch.where(mask_spain == 0, float('nan'), corr_map_time)
    else:
        corr_map_time_spain = torch.zeros_like(corr_map_time)

    return corr_map_time, corr_map_time_spain, torch.nanmean(corr_map_time).item(), torch.nanmean(corr_map_time_spain).item()

def compute_corr_space(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the Means
    pred_mean = torch.mean(pred_flattened, dim=1)
    target_mean = torch.mean(target_flattened, dim=1)

    # Calculation of covariance and standard deviations
    covariance = torch.mean(pred_flattened * target_flattened, dim=1) - pred_mean * target_mean
    pred_var = torch.mean(pred_flattened ** 2, dim=1) - pred_mean ** 2
    target_var = torch.mean(target_flattened ** 2, dim=1) - target_mean ** 2

    # Correlation
    corr_map_space = covariance / (torch.sqrt(pred_var) * torch.sqrt(target_var))
    corr_map_space[torch.isnan(corr_map_space)] = 0

    # Spain_mask
    if spain_bool:
        mask = mask_spain.flatten() != 0
        pred_spain = pred_flattened[:, mask]
        target_spain = target_flattened[:, mask]

        # Calculation of the Means
        pred_mean_spain = torch.mean(pred_spain, dim=1)
        target_mean_spain = torch.mean(target_spain, dim=1)

        # Calculation of covariance and standard deviations
        covariance_spain = torch.mean(pred_spain * target_spain, dim=1) - pred_mean_spain * target_mean_spain
        pred_var_spain = torch.mean(pred_spain ** 2, dim=1) - pred_mean_spain ** 2
        target_var_spain = torch.mean(target_spain ** 2, dim=1) - target_mean_spain ** 2

        # Correlation
        corr_map_space_spain = covariance_spain / (torch.sqrt(pred_var_spain) * torch.sqrt(target_var_spain))
        corr_map_space_spain[torch.isnan(corr_map_space_spain)] = 0    
                
    return corr_map_space, corr_map_space_spain, torch.nanmean(corr_map_space).item(), torch.nanmean(corr_map_space_spain).item()

def compute_rmse_time(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # Calculation of the Full RMSE
    rmse_map_time = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))

    # Spain_mask
    if spain_bool:
        mask_spain = mask_spain.squeeze()
        rmse_map_time_spain = torch.where(mask_spain == 0, float('nan'), rmse_map_time)
    else:
        rmse_map_time_spain = torch.zeros_like(rmse_map_time)

    return rmse_map_time, rmse_map_time_spain, torch.nanmean(rmse_map_time).item(), torch.nanmean(rmse_map_time_spain).item() 

def compute_rmse_space(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the RMSE
    rmse_map_space = torch.sqrt(torch.mean((pred_flattened - target_flattened) ** 2, dim=1))

    # Spain_mask
    if spain_bool:
        mask = mask_spain.flatten() != 0
        pred_spain = pred_flattened[:, mask]
        target_spain = target_flattened[:, mask]

        rmse_map_space_spain = torch.sqrt(torch.mean((pred_spain - target_spain) ** 2, dim=1))
                
    return rmse_map_space, rmse_map_space_spain, torch.nanmean(rmse_map_space).item(), torch.nanmean(rmse_map_space_spain).item()

# The old way to calcule the corr map along the time axis
def compute_corr_time_old(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    corr_pixel_time = torch.zeros((pred.shape[1], pred.shape[2]))
    corr_pixel_time_spain = torch.zeros((pred.shape[1], pred.shape[2]))

    for i in range(pred.shape[1]):
        for j in range(pred.shape[2]):

            pred_ij = pred[:, i, j]
            target_ij = target[:, i, j]

            combined_tensor = torch.stack([pred_ij, target_ij], dim=0)

            corr_pixel_time[i, j] = torch.corrcoef(combined_tensor)[0, 1].item()
            
            if spain_bool:
                if torch.isnan(mask_spain[i, j]):
                    corr_pixel_time_spain[i, j] = float('nan')
                else:
                    corr_pixel_time_spain[i, j] = corr_pixel_time[i, j]

            else:
                corr_pixel_time_spain[i, j] = 0
                
    return corr_pixel_time, corr_pixel_time_spain, torch.nanmean(corr_pixel_time).item(), torch.nanmean(corr_pixel_time_spain).item() 

# The old way to calcule the corr map along the space axis
def compute_corr_space_old(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    corr_pixel_space = torch.zeros((pred.shape[0]))
    corr_pixel_space_spain = torch.zeros((pred.shape[0]))

    if spain_bool:
        mask = ~torch.isnan(mask_spain.flatten())

    for i in range(pred.shape[0]):
            # I take the one temporal frame and flate the wind speed map
            pred_i = pred[i].flatten()
            target_i = target[i].flatten()

            combined_tensor = torch.stack([pred_i, target_i], dim=0)

            corr_pixel_space[i] = torch.corrcoef(combined_tensor)[0, 1].item()

            if spain_bool:
                pred_i_spain = pred_i[mask]
                target_i_spain = target_i[mask]

                combined_tensor = torch.stack([pred_i_spain, target_i_spain], dim=0)

                corr_pixel_space_spain[i] = torch.corrcoef(combined_tensor)[0, 1].item()

            else:
                corr_pixel_space_spain[i] = 0
                
    return corr_pixel_space, corr_pixel_space_spain, torch.nanmean(corr_pixel_space).item(), torch.nanmean(corr_pixel_space_spain).item()

# The old way to calcule the rmse map along the time axis
def compute_rmse_time_old(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    rmse_pixel_time = torch.zeros((pred.shape[1], pred.shape[2]))
    rmse_pixel_time_spain = torch.zeros((pred.shape[1], pred.shape[2]))

    for i in range(pred.shape[1]):
        for j in range(pred.shape[2]):

            pred_ij = pred[:, i, j]
            target_ij = target[:, i, j]

            rmse_pixel_time[i, j] = torch.sqrt(torch.mean((pred_ij - target_ij) ** 2)).item()
            
            if spain_bool:
                if torch.isnan(mask_spain[i, j]):
                    rmse_pixel_time_spain[i, j] = float('nan')
                else:
                    rmse_pixel_time_spain[i, j] = rmse_pixel_time[i, j]

            else:
                rmse_pixel_time_spain[i, j] = 0
                
    return rmse_pixel_time, rmse_pixel_time_spain, torch.nanmean(rmse_pixel_time).item(), torch.nanmean(rmse_pixel_time_spain).item() 

# The old way to calcule the rmse map along the space axis
def compute_rmse_space_old(pred, target, mask_spain, spain_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    rmse_pixel_space = torch.zeros((pred.shape[0]))
    rmse_pixel_space_spain = torch.zeros((pred.shape[0]))

    if spain_bool:
        mask = ~torch.isnan(mask_spain.flatten())

    for i in range(pred.shape[0]):
            # I take the one temporal frame and flate the wind speed map
            pred_i = pred[i].flatten()
            target_i = target[i].flatten()

            rmse_pixel_space[i] = torch.sqrt(torch.mean((pred_i - target_i) ** 2)).item()

            if spain_bool:
                pred_i_spain = pred_i[mask]
                target_i_spain = target_i[mask]

                combined_tensor = torch.stack([pred_i_spain, target_i_spain], dim=0)

                rmse_pixel_space_spain[i] = torch.sqrt(torch.mean((pred_i_spain - target_i_spain) ** 2)).item()

            else:
                rmse_pixel_space_spain[i] = 0
                
    return rmse_pixel_space, rmse_pixel_space_spain, torch.nanmean(rmse_pixel_space).item(), torch.nanmean(rmse_pixel_space_spain).item()

def calculate_metric(name_expr, pred, target, torchmetrics_settings={}, part=5000):

    metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr == m)]

    if len(metric_str) == 0:
        metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr in m)]
        if len(metric_str) > 1:
            warnings.warn('found multiple hits for metric name {}. Will use {}'.format(name_expr, metric_str[0]))

    assert len(metric_str) > 0, 'metric {} not found in torchmetrics.image. Maybe torch-fidelity is missing.'.format(name_expr)

    metric = t_metrics.__dict__[metric_str[0]](**torchmetrics_settings)

    total_sum = 0.0
    total_batches = 0

    for i in range(0, pred.size(0), part):

        batch_preds = pred[i:min(i + part, pred.size(0))]
        batch_targets = target[i:min(i + part, pred.size(0))]
    
        with torch.no_grad():
            value = metric(batch_preds.unsqueeze(1), batch_targets.unsqueeze(1)).item()
            total_sum += value
            total_batches += 1
    
    del batch_preds, batch_targets

    return total_sum/total_batches

if __name__ == '__main__':

    # First of all I call to the parser
    parser = argparse.ArgumentParser()

    config(parser=parser)

    args = parser.parse_args()

    # I set the root and root_data paths
    args.root = os.path.join(os.getcwd())
    if args.data_root == 'None':
        args.data_root = os.path.join(args.root, 'input_data/')#, args.time_res)

    # Set the path to save/find (trained) model
    if args.save_pred_dir == 'None':
        args.save_pred_dir = os.path.join(args.root, 'execution/evaluation/') # path to the predictions of the model (so after the evaluation of CRAI)

    args.save_metrics_dir = args.root   

    # I read the spain mask for map correlations and RMSE
    if (args.metrics is not None) and (args.spain_mask_bool):
        print('Reading spain mask data...')
        print()
        spain_mask = load_mask(args)
    else:
        spain_mask = 0

    # Create the folder
    #if not os.path.isdir(os.path.join(args.save_metrics_dir, 'single')):
        #os.makedirs(os.path.join(args.save_metrics_dir, 'single'))

    # I read the data:
    print('Loading data...')
    print()

    split = args.split #'test' #'train' #'valid'
    # epoch = 500
        
    if split == 'train':
        target_data = xr.open_dataset(os.path.join(args.data_root, 'train', args.dataset_name))
    elif split == 'valid':
        target_data = xr.open_dataset(os.path.join(args.data_root, 'val', args.dataset_name))
    elif split == 'test':
        target_data = xr.open_dataset(os.path.join(args.data_root, 'test', args.dataset_name))

    output_data = xr.open_dataset(os.path.join(args.save_pred_dir, args.reconstructed_file))
    print('output data:', output_data)
    
    print('Transforming to PyTorch tensors...')
    preds = torch.from_numpy(output_data["ws"].values)
    preds = torch.nan_to_num(preds, nan=0.0)
    targets = torch.from_numpy(target_data["ws"].values)
    targets = torch.nan_to_num(targets, nan=0.0)
    
        
    print('Calculating metrics...')
    print('preds shape: ', preds.shape)
    print('targets shape: ', targets.shape)
    print('spain_mask shape: ', spain_mask.shape)
    print('split: ', split)
    print('args: ', args)
    
    metric_dict = metrics(preds, targets, spain_mask, split, args) #,epoch)
    # metric_dict = get_metrics(preds, targets, spain_mask, split, args)

    print()
    print('Metrics: ')

                        
    for k, metric in enumerate(metric_dict):
        if 'map' not in metric.split("/")[1]:
            print(k, '.....', metric, '.....', metric_dict[metric])
        else:
            # For map metrics, convert to xarray and save as netCDF
            map_name = metric.split("/")[1]
            print(f"Converting {map_name} to xarray and saving as netCDF...")
            
            # Get the map data
            map_data = metric_dict[metric]
            
            print(f'map_data {map_name} shape: ', map_data.shape)
            # Create an xarray DataArray with the same coordinates as output_data
            if 'time' in map_name:
                # For time maps (maps over lat/lon)
                map_da = xr.DataArray(
                    data=map_data.cpu().numpy(),
                    dims=['latitude', 'longitude'],
                    coords={
                        'latitude': output_data.latitude,
                        'longitude': output_data.longitude
                    },
                    name=map_name
                )
            elif 'space' in map_name:
                # For space maps (maps over time)
                map_da = xr.DataArray(
                    data=map_data.cpu().numpy(),
                    dims=['time'],
                    coords={
                        'time': output_data.time
                    },
                    name=map_name
                )
            
            # Convert to dataset
            map_ds = map_da.to_dataset()
            
            # Create directory if it doesn't exist
            map_dir = os.path.join(args.save_metrics_dir, split, 'maps')
            os.makedirs(map_dir, exist_ok=True)
            
            # Save as netCDF
            map_file = os.path.join(map_dir, f"{map_name}.nc")
            map_ds.to_netcdf(map_file)
            print(f"Saved {map_name} to {map_file}")