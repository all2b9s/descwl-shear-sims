import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
import os
from scipy.spatial import cKDTree

# Functions to read predicted shapes and truth values from files.
def read_single_file(args):
    f_name, model_name, img_head, gt_match = args
    try:
        temp = np.load(f'{f_name}{model_name}_{img_head}_measured_shape.npy')
    except FileNotFoundError:
        print(f"File {f_name}{model_name}_{img_head}_measured_shape.npy not found.")
        return None # Return an empty array if the file is not found
    
    if not gt_match:
        return temp
    else:
        pred_center = temp[:,:2]
        pred_shape = temp[:,2:4]
        truth_cat = pd.read_csv(f_name+'sim_mdet_cat.csv')
        truth_center = truth_cat[['new_x', 'new_y']].values
        tree = cKDTree(pred_center)
        dist, indices = tree.query(truth_center, distance_upper_bound=5.0)

        # Handle unmatched entries (those with no neighbor within 5 pixels)
        # cKDTree returns len(pred_center) if no neighbor is found
        matched = indices < len(pred_center)
        matched_pred_pos = pred_center[indices[matched]]
        matched_pred_shape = pred_shape[indices[matched]]
        return np.concatenate((matched_pred_pos, matched_pred_shape), axis=1)

def read_pred(direct_name, 
              model_name, 
              img_head, 
              img_range,
              gt_match=False, 
              max_workers=8):
    '''
    direct_name: str
        The directory where the images are stored.
    img_head: str
        The prefix of the image filenames.
    img_range: tuple
        A tuple specifying the range of image indices to read (start, end).
    max_workers: int
        The maximum number of worker threads to use for parallel processing.

    Returns:
        np.ndarray
            A numpy array containing the stacked results from all specified image files.
    '''
    file_args = [(f'{direct_name}{img_index}/', model_name, img_head, gt_match) for img_index in range(img_range[0], img_range[1])]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(read_single_file, file_args), total=len(file_args)))
    results = list(result for result in results if result is not None)  # Filter out empty results
    return results

def read_single_truth(f_name):
    cols = ['ellipticity_1_true', 'ellipticity_2_true', 'shear_1', 'shear_2']
    df = pd.read_csv(f'{f_name}sim_mdet_cat.csv', usecols=cols)
    return df[cols].values  # Explicitly reorders to ensure column order

def read_truth(direct_name, img_range, max_workers=8):
    """
    Read truth data from multiple files in parallel.

    Args:
        direct_name (str): The directory name where the files are located.
        img_range (tuple): The range of image indices to read.
        max_workers (int): The maximum number of worker processes to use.

    Returns:
        numpy.ndarray: The stacked truth data from all the files.

    """
    # Create a list of file names based on the image range
    file_names = [f'{direct_name}{img_index}/' for img_index in range(img_range[0], img_range[1])]

    # Read the truth data from each file in parallel using multiple processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar while reading the files
        results = list(tqdm(executor.map(read_single_truth, file_names), total=len(file_names)))

    # Stack the results into a single numpy array
    return np.vstack(results)


# Class to get the calibration bias and error.
def _shape_bootstrap(cali_result, size, datasets):
    np.random.seed()
    rand_ints = np.random.choice(len(cali_result[datasets[0]]), size=size, replace=True)
    temp_dict = {dataset: np.vstack([cali_result[dataset][rand_index] 
                                    for rand_index in rand_ints])[:,2:4] for dataset in datasets}
    return temp_dict

class DD_calibration():
    datasets =  ['1+', '1-', '2+', '2-']
    mdet_sets = ['1p', '1m', '2p', '2m']
    max_workers = 64
    gt_match = False
    def __init__(self, model_name, mdet_head, cali_head, mdet_img_range, cali_img_range, shear_step = 1e-3):
        self.base_folder = os.environ.get('DD_WL_SIMS', 'Model path is not defined')
        self.mdet_head = mdet_head
        self.cali_head = cali_head
        self.model_name = model_name
        self.mdet_img_range = mdet_img_range
        self.cali_img_range = cali_img_range
        self.shear_step = shear_step
        self.response = None
        self.cali_result = None


    def __call__(self, size = 100, bs_times = 100):
        if self.response is None:
            self.mdet_result = self._do_mdet()
        if self.cali_result is None:
            self.cali_result = self._do_cali()
        mdet_dict = {dataset: np.vstack(self.mdet_result[dataset])[:,2:] for dataset in self.mdet_sets}
        cali_dict = {dataset: np.vstack(self.cali_result[dataset])[:,2:] for dataset in self.datasets}
        self.response = self._mdet_response(mdet_dict)
        self.cali_bias = self._cali(self._get_shear(cali_dict), self.response)
        bootstrap_result = self.bootstrap_error(size = size, bs_times = bs_times)
        return self.cali_bias, bootstrap_result

    def bootstrap_error(self, size=100, bs_times=100):
        shears = []
        responses = []
        
        with ProcessPoolExecutor(64) as executor:
            cali_futures = [executor.submit(_shape_bootstrap, self.cali_result, size, self.datasets) for _ in range(bs_times)]
            mdet_futures = [executor.submit(_shape_bootstrap, self.mdet_result, size, self.mdet_sets) for _ in range(bs_times)]

        with ProcessPoolExecutor(64) as executor:
            temp_shears = [executor.submit(self._get_shear, future.result()) for future in cali_futures]
            temp_response = [executor.submit(self._mdet_response, future.result()) for future in mdet_futures]

        for shear in temp_shears:
            shears.append(shear.result())
        for response in temp_response:
            responses.append(response.result())
        responses = np.array(responses)
        biases = np.array([self._cali(shear, resp) for shear, resp in zip(shears, responses)]) # [n, 2, 2]
        return np.array(biases[:,0]).std(axis=0), np.array(biases[:,1]).std(axis=0), shears, responses
        

    def _get_fnames(self, mode):
        if mode == 'mdet':
            return f'{self.base_folder}{self.mdet_head}/'
        elif mode == 'cali':
            return [f'{self.base_folder}{self.cali_head}_{img_head}/' for img_head in ['1+', '1-', '2+', '2-']]
    
    def _do_mdet(self):
        fname = self._get_fnames('mdet')
        img_heads = ['1p', '1m', '2p', '2m']
        mdet_dict = {img_head: read_pred(fname, self.model_name, img_head, self.mdet_img_range, self.gt_match, max_workers=self.max_workers) for img_head in img_heads} 
        #response = self._mdet_response(mdet_dict)
        return mdet_dict
    
    def _do_cali(self):
        fnames = self._get_fnames('cali')
        img_head = 'ori'
        cali_result =  {self.datasets[i]: read_pred(fname, self.model_name, img_head, self.cali_img_range, self.gt_match, max_workers=self.max_workers) 
                        for (i, fname) in enumerate(fnames)}
        return cali_result

    def _mdet_response(self, pred_dict):
        r11 = (pred_dict['1p'][:,0].mean()-pred_dict['1m'][:,0].mean())/0.02
        r12 = (pred_dict['1p'][:,1].mean()-pred_dict['1m'][:,1].mean())/0.02
        r21 = (pred_dict['2p'][:,0].mean()-pred_dict['2m'][:,0].mean())/0.02
        r22 = (pred_dict['2p'][:,1].mean()-pred_dict['2m'][:,1].mean())/0.02
        return [r11, r12, r21, r22]
    
    def _get_shear(self, cali_dict):
        shears = {dataset: [cali_dict[dataset][:,i].mean() for i in range(0,2)] for dataset in self.datasets}
        return shears
    

    def _cali(self, shears, response):
        m1 = (shears['1+'][0]-shears['1-'][0])/response[0]/(2*self.shear_step)-1
        c2 = (shears['2+'][1]+shears['2-'][1])/response[3]/2
        m2 = (shears['2+'][1]-shears['2-'][1])/response[3]/(2*self.shear_step)-1
        c1 = (shears['1+'][0]+shears['1-'][0])/response[0]/2
        return [[m1, m2],[c1, c2]]
    

    

    
    


