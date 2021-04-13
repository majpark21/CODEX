##############################################################################################
# Utils to read ZIP archive that contains data, some preprocessing before Pytorch DataLoader #
##############################################################################################

import pandas as pd
import warnings
import zipfile
import numpy as np
from copy import copy
from re import search
from os import remove, chdir
from os.path import split, splitext
from functools import reduce
from collections import OrderedDict


class DataProcesser:
    """
    Class for reading, subsetting and processing time-series data before passing to network
    Attributes:
        * archive_path :str: path to the data archive
        * archive :zipfile: object containing the archive
        * col_id :str: name of the column containing the series ID in .dataset and .id_set
        * col_class :str: name of the column containing the series class (dummy coded as integer starting at 0) in .dataset and .classes
        * col_classname :str: name of the column containing the full name (i.e. not dummy coded) of the class in .classes
        * col_set :str: name of the column containing the series set (training|validation|test) in .id_set
        * dataset :DataFrame: observations (series) in rows, measurements in columns. Names of columns must have the format:
         A_1, A_2, A_3,..., C_1, C_2,... where A and C are groups (sensors) and 1,2,3... measurement time
        * dataset_cropped :DataFrame: cropped dataset if crop_random() method is called
        * id_set :DataFrame: IDs of training/validation/test.
        * classes :DataFrame: Conversion table to go from dummy class name to actual one.
        * logs :list: operations that were performed on the archive and stats for further preprocessing
        * stats :dictionary: contains various stats in different sets (including mean of training set)
        * train(validation|test)_set :DataFrame:
        * flag_subset :boolean: whether .dataset has undergone subset operation
        * flag_process :boolean: whether .dataset has undergone normalization
        * flag_split :boolean: whether dataset was split
    Example:
        tmp=DataProcesser('/path/to/archive/myarchive.zip')
        tmp.subset(['groupA', 'groupB'], 3 ,7)
        tmp.process('center_train', independent_groups=True)
        tmp.split_sets()
        tmp.export_processed(compress=True)
    """

    def __init__(self, archive_path, col_id='ID', col_class='class', col_classname='class_name', col_set='set', read_on_init=True, **kwargs):
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(self.archive_path, 'r')
        self.col_id = col_id
        self.col_class = col_class
        self.col_classname = col_classname
        self.col_set = col_set
        self.dataset = None
        self.dataset_cropped = None
        self.id_set = None
        self.classes = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None
        self.logs = []
        self.stats = None
        self.flag_subset = False
        self.flag_process = False
        self.flag_split = False
        if read_on_init:
            self.read_archive(**kwargs)


    def read_archive(self, datatable=True, **kwargs):
        """
        Read a zip archive, without extraction, than contains:

        * data as .csv, observations in rows, measurements in columns. Names of columns must have the format:
         A_1, A_2, A_3,..., C_1, C_2,... where A and C are groups (sensors) and 1,2,3... measurement time

        * IDs of training/validation/test as .csv

        * Explicit name of classes as .csv
        :return: 2 pandas, one with raw data, one with IDs
        """
        if datatable:
            try:
                from datatable import fread
                self.dataset = fread(self.archive.open('dataset.csv'), **kwargs).to_pandas()
                self.id_set = fread(self.archive.open('id_set.csv'), **kwargs).to_pandas()
                self.classes = fread(self.archive.open('classes.csv'), **kwargs).to_pandas()
            except ModuleNotFoundError:
                warnings.warn('datatable module not found, using pandas instead. To prevent this message from appearing'
                              ' use "datatable = False" when reading the archive.')
                self.dataset = pd.read_csv(self.archive.open('dataset.csv'))
                self.id_set = pd.read_csv(self.archive.open('id_set.csv'))
                self.classes = pd.read_csv(self.archive.open('classes.csv'))
        else:
            self.dataset = pd.read_csv(self.archive.open('dataset.csv'))
            self.id_set = pd.read_csv(self.archive.open('id_set.csv'))
            self.classes = pd.read_csv(self.archive.open('classes.csv'))
        self.check_datasets()
        self.logs.append('Read archive: {0}'.format(self.archive_path))
        return None


    def check_datasets(self):
        """
        Check that there is at least one correct measurement in dataset, and values in set_ids.
        :return:
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        colnames_dataset = list(self.dataset.columns.values)
        colnames_dataset.remove(self.col_id)
        colnames_dataset.remove(self.col_class)

        if not self.col_id in self.dataset.columns.values:
            warnings.warn('ID column "{}" is missing in dataset.'.format(self.col_id))
        if not self.col_id in self.id_set.columns.values:
            warnings.warn('ID column "{}" is missing in id_set.'.format(self.col_id))
        if not self.col_class in self.dataset.columns.values:
            warnings.warn('Class column "{}" not present in dataset.'.format(self.col_class))
        if not self.col_class in self.classes.columns.values:
            warnings.warn('Class column "{}" not present in classes.'.format(self.col_class))
        if not self.col_classname in self.classes.columns.values:
            warnings.warn('Class name column "{}" not present in classes.'.format(self.col_classname))
        if not self.col_set in self.id_set.columns.values:
            warnings.warn('Set column "{}" not present in id_set.'.format(self.col_set))
        if self.dataset.select_dtypes(numerics).empty:
            warnings.warn('No numerical columns in dataset.')
        if len(list(set(self.dataset[self.col_id]) - set(self.id_set[self.col_id]))) != 0 or len(list(set(self.id_set[self.col_id]) - set(self.dataset[self.col_id]))) != 0:
            warnings.warn('ID list is different between dataset and id_set.')
        if (not all([search('^\w+_', i) for i in colnames_dataset])) or (not all([search('_\d+$', i) for i in colnames_dataset])):
            ill_col = [i for i in colnames_dataset if not search('^\w+_', i)]
            ill_col += [i for i in colnames_dataset if not search('_\d+$', i)]
            warnings.warn('At least some column names of dataset are ill-formatted. Should follow "Group_Time" format. '
                          'List of ill-formatted: {0}'.format(ill_col))
        if any(self.dataset[self.col_id].duplicated()):
            warnings.warn('Found duplicated ID in "dataset".')
        if any(self.id_set[self.col_id].duplicated()):
            warnings.warn('Found duplicated ID in "id_set".')
        return None


    def detect_groups_times(self, return_groups=True, return_times=True, return_times_pergroup=True):
        """
        Detect the measurement groups and the time range spanned by these measurements.
        :param return_groups: bool, whether to return the unique groups names.
        :param return_times: bool, whether to return the global time range.
        :param return_times_pergroup: bool, whether to return the time range per group.
        :return: dict, with keys groups, times, times_pergroup.
        """
        if return_times_pergroup and not return_times:
            return_times = True
            warnings.warn('"return_times" is False but "return_times_pergroup" is True. "return_times" will be set to True.')
        out = {}
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        if return_groups:
            out['groups'] = groups
        if return_times:
            times = [int(i.split('_')[1]) for i in colnames]
            out['times'] = [min(times), max(times)]
            if return_times_pergroup:
                out['times_pergroup'] = {}
                for group in groups:
                    group_columns = [i for i in colnames if search('^{0}_'.format(group), i)]
                    group_times = [int(i.split('_')[1]) for i in group_columns]
                    out['times_pergroup'][group] = [min(group_times), max(group_times)]
        return out

    def get_max_common_length(self, return_group=False):
        """
        Get the maximum common length of all trajectories, without the NAs tails. Optionally returns the limiting measurement group.
        """
        def get_length_without_na(series, cols):
            series = np.array(series.loc[cols]).astype('float')
            len_wihtout_na = np.where(~np.isnan(series))[0].size
            return len_wihtout_na
        # Get measurement columns, make a distinction between each measurement group
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        # Determine the max common length independently for each measurement group
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        groups_min = {}
        for group in groups:
            group_cols = [i for i in colnames if search('^{0}_'.format(group), i)]
            df_len = self.dataset.apply(get_length_without_na, args=(group_cols,), axis = 1)
            groups_min[group] = df_len.min()
        mini_v = min(groups_min.values()) 
        mini_k = [k for k in groups_min if groups_min[k] == mini_v]
        if return_group:
            return (mini_k, mini_v)
        else:
            return int(mini_v)


    def subset(self, sel_groups=None, start_time=None, end_time=None):
        """
        Select only columns whose group matches and keep times within boundary. Use None for auto detection
        :return: replace self.dataset with the subset, along with ID column
        """
        if sel_groups is None:
            sel_groups = self.detect_groups_times()['groups']
        if start_time is None:
            start_time = self.detect_groups_times()['times'][0]
        if end_time is None:
            end_time = self.detect_groups_times()['times'][1]
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        col_dict = {col:col.split(sep='_') for col in colnames}
        # Subset if col prefix is a selected group and if col suffix is a selected time
        sel_col = [col for col in col_dict
                   if (col_dict[col][0] in sel_groups and
                       int(col_dict[col][1]) in range(start_time, end_time+1))]
        if len(sel_col) == 0:
            raise ValueError('Empty columns subset')
        self.dataset = self.dataset.loc[:, [self.col_id, self.col_class] + sel_col]
        self.logs.append('Subset: groups:{0}, start_time:{1}, end_time:{2}'.format(sel_groups, start_time, end_time))
        self.flag_subset = True
        return None


    def get_stats(self):
        """
        Get means, mins, maxs, sds... in different subset, different groups
        :return:
        """
        if self.flag_process:
            warnings.warn('data were already processed. Stats will be extracted from the processed data, not the '
                          'original data.')
        if not self.flag_subset:
            warnings.warn('Stats are extracted before subset. Subset groups or times will affect the stats.')

        # Different groups of measurements
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))

        # Initialize all dictionaries;
        # {'mu':{'groups_combined', 'groupA', 'groupB'}, 'sd':{'groups_combined', 'groupA', 'groupB'}, ...}
        stats = ['mu', 'sd', 'mini', 'maxi']
        self.stats = {k1:{k2:{k3:{} for k3 in ['global', 'train']} for k2 in groups+['groups_combined']} for k1 in stats}

        # Consider all channels as one single measurement
        ## Global
        # Convert to array otherwise mean can only be row or column-wise
        groups_combined = np.array(self.dataset.drop([self.col_id, self.col_class], axis=1))
        self.stats['mu']['groups_combined']['global'] = np.nanmean(groups_combined)
        self.stats['sd']['groups_combined']['global'] = np.nanstd(groups_combined)
        self.stats['mini']['groups_combined']['global'] = np.nanmin(groups_combined)
        self.stats['maxi']['groups_combined']['global'] = np.nanmax(groups_combined)
        del groups_combined
        ## Training set only
        groups_combined_train = pd.merge(self.dataset, self.id_set, on=self.col_id)
        groups_combined_train = np.array(
            groups_combined_train[groups_combined_train[self.col_set] == 'train'].drop([self.col_id, self.col_class, self.col_set], axis=1))
        self.stats['mu']['groups_combined']['train'] = np.nanmean(groups_combined_train)
        self.stats['sd']['groups_combined']['train'] = np.nanstd(groups_combined_train)
        self.stats['mini']['groups_combined']['train'] = np.nanmin(groups_combined_train)
        self.stats['maxi']['groups_combined']['train'] = np.nanmax(groups_combined_train)

        # Extract statistics independently for each channel
        for group in groups:
            group_columns = [i for i in colnames if search('^{0}_'.format(group), i)]
            ## Global
            # Class and ID columns are already excluded here
            group_array = np.array(self.dataset[group_columns])
            self.stats['mu'][group]['global'] = np.nanmean(group_array)
            self.stats['sd'][group]['global'] = np.nanstd(group_array)
            self.stats['mini'][group]['global'] = np.nanmin(group_array)
            self.stats['maxi'][group]['global'] = np.nanmax(group_array)
            del group_array
            ## Training set only
            group_array_train = pd.merge(self.dataset, self.id_set, on=self.col_id)
            group_array_train = group_array_train[group_array_train[self.col_set] == 'train']
            group_array_train = np.array(group_array_train[group_columns])
            self.stats['mu'][group]['train'] = np.nanmean(group_array_train)
            self.stats['sd'][group]['train'] = np.nanstd(group_array_train)
            self.stats['mini'][group]['train'] = np.nanmin(group_array_train)
            self.stats['maxi'][group]['train'] = np.nanmax(group_array_train)
            del group_array_train

        return None


    def process(self, method, independent_groups=True):
        """
        Process data for neural network.
        :param independent_groups :boolean: Whether operations should be applied independantly on each group.
        :param method :str:
          * center: x - mean(x)
          * zscore: (x - mean(x))/sd(x)
          * squeeze01: (x-min(x))/(max(x)-min(x))
          The second part of the method determines where statistics are taken from:

          * train: use all training data
          * global: use all data (training+validation+test)
          * individual: perform operation series by series
        :return: Modifies .dataset in-place and returns info about process (mean train...)
        """
        methods = ['center_train', 'center_global', 'center_individual',
                   'zscore_train', 'zscore_global', 'zscore_individual',
                   'squeeze01_train', 'squeeze01_global', 'squeeze01_individual']

        if not method in methods:
            raise ValueError('Invalid method, must be one of: {0}'.format(methods))
        if not self.flag_subset:
            warnings.warn('Data were not subset.')
        if self.flag_split:
            warnings.warn(
                'Data were already split, split will not contain process. Call .split_sets() again to overwrite.')
        if self.stats is None:
            raise ValueError('Statistics were not extracted. Use get_stats() prior to process().')

        # Build unique groups dictionary {'A':['A_1', 'A_2], 'B':['B_1', 'B_2']}
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        groups_dict = {}
        for group in groups:
            groups_dict[group] = [i for i in colnames if search('^{0}_'.format(group), i)]

        operation, second_group = method.split('_')
        # Temporary add set split for grouping operations that involve extracting training set statistics
        if second_group == 'train':
            self.dataset = pd.merge(self.dataset, self.id_set, on=self.col_id)

        # Extract statistics independently for each channel
        if independent_groups:
            # Store after each group has been processed and stitch back together to obtain dataset of original size
            processed_groups = []
            for group in groups_dict.keys():
                # Dataframe with measurements (numerical) only for the current group
                current_group = self.dataset[groups_dict[group]]

                if second_group == 'global':
                    mu = self.stats['mu'][group]['global']
                    sd = self.stats['sd'][group]['global']
                    mini = self.stats['mini'][group]['global']
                    maxi = self.stats['maxi'][group]['global']
                    self.logs.append(
                        'Global stats group {}; mu:{}; sd:{}; mini:{}; maxi:{}'.format(group, mu, sd, mini, maxi))

                if second_group == 'train':
                    mu = self.stats['mu'][group]['train']
                    sd = self.stats['sd'][group]['train']
                    mini = self.stats['mini'][group]['train']
                    maxi = self.stats['maxi'][group]['train']
                    self.logs.append(
                        'Train set stats group {}; mu:{}; sd:{}; mini:{}; maxi:{}'.format(group, mu, sd, mini, maxi))

                if second_group == 'individual':
                    h, w = current_group.shape
                    mu = np.array(current_group.apply(np.mean, axis=1))
                    mu = np.tile(mu.reshape((h, 1)), (1, w))
                    sd = np.array(current_group.apply(np.std, axis=1))
                    sd = np.tile(sd.reshape((h, 1)), (1, w))
                    mini = np.array(current_group.apply(np.min, axis=1))
                    mini = np.tile(mini.reshape((h, 1)), (1, w))
                    maxi = np.array(current_group.apply(np.max, axis=1))
                    maxi = np.tile(maxi.reshape((h, 1)), (1, w))

                if operation == 'center':
                    current_group -= mu
                elif operation == 'zscore':
                    current_group = (current_group - mu) / sd
                elif operation == 'squeeze01':
                    current_group = (current_group - mini) / (maxi - mini)

                processed_groups.append(current_group)
            # Stitch back together the processed groups with ID and class
            processed_groups = [pd.DataFrame(self.dataset[[self.col_id, self.col_class]])] + processed_groups
            self.dataset = pd.concat(processed_groups, axis=1)

        # Consider all channels as one single measurement
        else:
            if second_group == 'global':
                mu = self.stats['mu']['groups_combined']['global']
                sd = self.stats['sd']['groups_combined']['global']
                mini = self.stats['mini']['groups_combined']['global']
                maxi = self.stats['maxi']['groups_combined']['global']
                self.logs.append(
                    'Global stats; mu:{}; sd:{}; mini:{}; maxi:{}'.format(mu, sd, mini, maxi))

            if second_group == 'train':
                mu = self.stats['mu']['groups_combined']['train']
                sd = self.stats['sd']['groups_combined']['train']
                mini = self.stats['mini']['groups_combined']['train']
                maxi = self.stats['maxi']['groups_combined']['train']
                self.logs.append(
                    'Global stats; mu:{}; sd:{}; mini:{}; maxi:{}'.format(mu, sd, mini, maxi))

            if second_group == 'individual':
                # Dataframe with measurements (numerical) only
                measurements_df = self.dataset.drop([self.col_id, self.col_class], axis=1)
                h,w = measurements_df.shape
                mu = np.array(measurements_df.apply(np.mean, axis=1))
                mu = np.tile(mu.reshape((h, 1)), (1, w))
                sd = np.array(measurements_df.apply(np.std, axis=1))
                sd = np.tile(sd.reshape((h, 1)), (1, w))
                mini = np.array(measurements_df.apply(np.min, axis=1))
                mini = np.tile(mini.reshape((h, 1)), (1, w))
                maxi = np.array(measurements_df.apply(np.max, axis=1))
                maxi = np.tile(maxi.reshape((h, 1)), (1, w))

            if operation == 'center':
                self.dataset.loc[:, list(set(self.dataset.columns) - set([self.col_id, self.col_class]))] -= mu
            elif operation == 'zscore':
                self.dataset.loc[:, list(set(self.dataset.columns) - set([self.col_id, self.col_class]))] -= mu
                self.dataset.loc[:, list(set(self.dataset.columns) - set([self.col_id, self.col_class]))] /= sd
            elif operation == 'squeeze01':
                self.dataset.loc[:, list(set(self.dataset.columns) - set([self.col_id, self.col_class]))] -= mini
                self.dataset.loc[:, list(set(self.dataset.columns) - set([self.col_id, self.col_class]))] /= (maxi-mini)

        self.logs.append('Process: method:{0}, indep_groups:{1}'.format(method, independent_groups))
        self.flag_process = True
        return None


    def crop_random(self, output_length,  group_crop=None, ignore_na_tails=True):
        """
        Returns a random subset of each row. Useful to get rid of NA tails. If "dataset" contains several groups, only one
        will be used to determine the range of cropping. So caution if NA tails are not strictly aligned across the groups.
        :param output_size: length of each new series
        :param ignore_na_tails: whether to ignore NA tails from subset
        :param group_crop: name of the measurement used to determine the crop position. If None, use first available
        group.
        :return: Creates .dataset_cropped
        """
        # Get a random range to crop for each row
        def get_range_crop(series, cols, output_length, ignore_na_tails):
            series = np.array(series.loc[cols]).astype('float')
            if ignore_na_tails:
                pos_non_na = np.where(~np.isnan(series))
                start = pos_non_na[0][0]
                end = pos_non_na[0][-1]
                left = np.random.randint(start,
                                         end - output_length + 2)  # +1 to include last in randint; +1 for slction span
            else:
                length = len(series)
                left = np.random.randint(0, length - output_length)
            right = left + output_length
            return left, right

        # Use only the provided group to determine range of cropping
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        groups_dict = {}
        for group in groups:
            groups_dict[group] = [i for i in colnames if search('^{0}_'.format(group), i)]

        if group_crop is None:
            group_crop = groups[0]
        sel_col = groups_dict[group_crop]
        range_subset = self.dataset.apply(get_range_crop, args=(sel_col, output_length, ignore_na_tails,), axis = 1)

        l_group_cropped = []
        for group in groups:
            # Crop the rows to random range, reset_index to do concat without recreating new columns
            group_dt = self.dataset[groups_dict[group]]
            new_rows = [group_dt.iloc[irow, range_subset[irow][0]: range_subset[irow][1]]
                        for irow in range(self.dataset.shape[0])]
            for row in new_rows:
                row.reset_index(drop=True, inplace=True)

            # Concatenate all rows, add ID and class column
            col_class, col_id = self.col_class, self.col_id
            group_cropped = pd.concat(new_rows, axis=1).T
            group_cropped.columns = ['{}_{}'.format(group,i) for i in range(group_cropped.shape[1])]
            group_cropped[col_id] = self.dataset[col_id]
            group_cropped[col_class] = self.dataset[col_class]

            l_group_cropped.append(group_cropped)

        # Stitch individual groups back together and reorder columns
        dataset_cropped = reduce(lambda left, right: pd.merge(left, right, on=[col_id, col_class]), l_group_cropped)
        newcols = list(dataset_cropped.columns.values)
        newcols.remove(col_id)
        newcols.remove(col_class)
        dataset_cropped = dataset_cropped[[col_id, col_class] + newcols]

        self.dataset_cropped = dataset_cropped
        return None


    def split_sets(self, which='dataset'):
        """
        Split dataset in train, validation, test according to id_split. Not memory efficient because Copies not views!

        :return: 3 pandas, one for each set
        """
        if not self.flag_subset:
            warnings.warn('Data were not subset.')
        if self.flag_process:
            warnings.warn('Data were already preprocessed, be careful not to process again with dataloaders.')
        if not which in ['dataset', 'dataset_cropped']:
            raise ValueError('which must be one of ["dataset", "dataset_cropped]')
        ids_train = list(self.id_set[self.id_set[self.col_set] == 'train'][self.col_id])
        ids_validation = list(self.id_set[self.id_set[self.col_set] == 'validation'][self.col_id])
        ids_test = list(self.id_set[self.id_set[self.col_set] == 'test'][self.col_id])

        if which == 'dataset':
            self.train_set = self.dataset[self.dataset[self.col_id].isin(ids_train)]
            self.validation_set = self.dataset[self.dataset[self.col_id].isin(ids_validation)]
            self.test_set = self.dataset[self.dataset[self.col_id].isin(ids_test)]
        elif which == 'dataset_cropped':
            self.train_set = self.dataset_cropped[self.dataset_cropped[self.col_id].isin(ids_train)]
            self.validation_set = self.dataset_cropped[self.dataset_cropped[self.col_id].isin(ids_validation)]
            self.test_set = self.dataset_cropped[self.dataset_cropped[self.col_id].isin(ids_test)]

        self.flag_split = True
        return None


    def export_processed(self, compress=True, out_path=None, export_process_info=True):
        """
        Export the processed data, either as zipped or 3 csv files

        :param compressed: boolean, if True return a zip archive, else returns 3 csv files.
        :param out_path: string, default to archive_path.
        :param export_process_info: boolean, whether to export processing info in .txt file
        :return:
        """
        if not self.flag_split:
            raise AttributeError('Data were not split. Run first self.split_sets()')
        if out_path is None:
            out_path = splitext(self.archive_path)[0] # split extension

        # If compressed, write csv, add to zip archive and delete
        self.train_set.to_csv(out_path + '_train.csv', index=False)
        self.validation_set.to_csv(out_path + '_validation.csv', index=False)
        self.test_set.to_csv(out_path + '_test.csv', index=False)
        if export_process_info:
            with open(out_path + '_process_info.txt', 'w') as f:
                for log in self.logs:
                    f.write(log + '\n')

        if compress:
            with zipfile.ZipFile(out_path + "_DataProcesser.zip", mode='w') as zipMe:
                lfile = [out_path + i for i in ['_train.csv', '_validation.csv', '_test.csv']]
                if export_process_info:
                    lfile.append(out_path + '_process_info.txt')
                for file in lfile:
                    # Remove all path before file to avoid it being exported in the archive
                    path, filename = split(file)
                    chdir(path)
                    zipMe.write(filename, compress_type=zipfile.ZIP_DEFLATED)
                    remove(file)
        return None

# Example
# myProc = DataProcesser('/home/marc/Dropbox/Work/TSclass/data/paolo/KTR_FOX_DMSO_noEGF_len120_7repl.zip')