import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime as dt
from seren.utils.functions import reindex
from ..config import DATA_PATH

class Interactions(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.user_key = config['user_key']
        self.item_key = config['item_key']
        self.session_key = config['session_key']
        self.time_key = config['time_key']
        self.category_key = config['category_key']

        self._process_flow()

    def _process_flow(self):
        self._load_data()
        #self._make_sessions()
        #self._core_filter()
        self._filter()
        # self._reindex()

        #self.user_num = self.df[self.user_key].nunique()
        self.item_num = self.df[self.item_key].nunique()

        #self.df.sort_values([self.item_key, self.user_key, self.item_key], inplace=True)

        self.logger.info(f'Finish loading {self.dataset_name} data, current length is: {len(self.df)}, item number: {self.item_num}')

    def _load_data(self):
        '''
        load raw data to dataframe and rename columns as required

        Parameters
        ----------
        user_key : str, optional
            column to present users, by default 'user_id'
        item_key : str, optional
            column to present items, by default 'item_id'
        time_key : str, optional
            column to present timestamp, by default 'ts'
        '''
        dataset_name = self.config['dataset']
        self.dataset_name = dataset_name
        
        if not os.path.exists(f'{DATA_PATH}{dataset_name}/'):
            self.logger.error('unexisted dataset...')
        if dataset_name == 'ml-100k':
            self.df = pd.read_csv(
                f'{DATA_PATH}ml-100k/u.data',
                delimiter='\t',
                names=[self.user_key, self.item_key, 'rating', self.time_key]
            )
            self._make_sessions()
        elif dataset_name == 'yoochoose':
            df = pd.read_csv(
                f'{DATA_PATH}yoochoose/yoochoose-clicks.dat',
                sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64},
                names=[self.session_key, 'TimeStr', self.item_key] #'SessionId', 'TimeStr', 'ItemId'
            )
            df[self.time_key] = df.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
            del(df['TimeStr'])
            self.df = df
            self.n_days = 1
        elif dataset_name == 'diginetica':
            df = pd.read_csv(f'{DATA_PATH}diginetica/train-item-views.csv', sep=';') # sessionId, userId, itemId, timeframe, eventdate
            df.rename({'sessionId':self.session_key, 'itemId':self.item_key}, axis='columns', inplace=True)
            df[self.time_key] = df.eventdate.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))#.timestamp()) #This is not UTC. It does not really matter.
            del(df['eventdate'])
            self.df = df
            self.n_days = 7
        elif dataset_name == 'retailrocket':
            df = pd.read_csv( f'{DATA_PATH}retailrocket/events.csv', sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int64, 2:str, 3:np.int64})
            #specify header names
            df.columns = ['Time', self.user_key,'Type',self.item_key]
            df['Time'] = (df.Time / 1000).astype( int )
            df.sort_values( [self.user_key,'Time'], ascending=True, inplace=True )
            df = df[df.Type == 'view']
            df[self.time_key] = pd.to_datetime(df.Time, unit='s')
            df.sort_values( [self.user_key,self.time_key], ascending=True, inplace=True )
            del df['Type']
            del df['Time']
            self.df = df
            self.n_days = 7
            self._make_sessions(idle_time = 30*60) # 30 minutes
        elif dataset_name == 'tmall':
#            df = pd.read_csv( f'{DATA_PATH}tmall/user_log_format1.csv', sep=',', header=0, usecols=[0,1,2,5,6], dtype={0:np.int32, 1:np.int32, 5:str, 6:np.int32} )
#            df.columns = [self.user_key, self.item_key, self.category_key, 'Time', 'ActionType']
#            df = df[df.ActionType.isin([0,2])] #0 click, 2 buy
#            df[self.time_key] = df['Time'].apply(lambda x: dt.datetime.strptime('2015-'+x, '%Y-%m%d'))
#            df = df.sort_values(by=[self.user_key, self.time_key])
#            #convert time string to timestamp and remove the original column
#            df[self.session_key] = df.groupby( [df[self.user_key], df[self.time_key]] ).grouper.group_info[0]
#            df = df.drop_duplicates(subset=[self.session_key, self.item_key])
#            print(df[self.session_key].nunique())
#            df = df[df[self.session_key]<=df[self.session_key].nunique()/16]
#            del df['Time']
#            del df['ActionType']
            df = pd.read_csv( f'{DATA_PATH}tmall/tmall_1_16.csv')
            df.columns = [self.user_key, self.item_key, self.category_key, self.time_key, self.session_key]
            df[self.time_key] = pd.to_datetime(df[self.time_key])
            self.df = df
            self.n_days = 7
        elif dataset_name == 'taobao':
            df = pd.read_csv( f'{DATA_PATH}taobao/UserBehavior.csv')
            df.columns = [self.user_key, self.item_key,'CatID', 'ActionType', 'Time']
            df_pv = df[df.ActionType == 'pv'] # pv, buy
            df_pv[self.time_key] = pd.to_datetime(df_pv.Time, unit='s')
            df_pv.sort_values( [self.user_key,self.time_key], ascending=True, inplace=True )
            del df_pv['ActionType']
            del df_pv['Time']
            self.df = df_pv
            self.n_days = 7
            self._make_sessions()
        elif dataset_name == 'globo':
            CLICK_FILES_PATH = f'{DATA_PATH}globo/clicks/'
            click_files = [CLICK_FILES_PATH + x for x in os.listdir(CLICK_FILES_PATH)]
            click_files.sort()
            list_click_file_to_df = [pd.read_csv(x, index_col=None, header=0) for x in click_files]
            list_click_file_to_df_cleaned = [x.drop(columns = ['session_start',
                                                               'session_size',
                                                               'click_environment',
                                                               'click_deviceGroup',
                                                               'click_os',
                                                               'click_country',
                                                               'click_region',
                                                               'click_referrer_type']) for x in list_click_file_to_df]
            all_clicks_df = pd.concat(list_click_file_to_df_cleaned, axis=0, ignore_index=True)
            # user_id    session_id    click_article_id    click_timestamp
            all_clicks_df.columns = [self.user_key, self.session_key, self.item_key, 'click_timestamp'] #13digits for milliseconds
            all_clicks_df[self.time_key] = all_clicks_df['click_timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x/1000.0))
            all_clicks_df, sess_id_map, id_sess_map = reindex(all_clicks_df, self.session_key, start_from_zero=False)
            del all_clicks_df['click_timestamp']
            self.df = all_clicks_df[all_clicks_df[self.session_key]<=all_clicks_df[self.session_key].nunique()/4]
            self.n_days = 7
        elif dataset_name in ['aotm', '30music', 'nowplaying']:
            if dataset_name == 'aotm':
                df = pd.read_csv( f'{DATA_PATH}aotm/raw/playlists-aotm.csv', sep='\t') # UserId    SessionId    ItemId    Time    ArtistId
            elif dataset_name == '30music':
                df = pd.read_csv( f'{DATA_PATH}30music/raw/30music-200ks.csv', sep='\t') # UserId    SessionId    ItemId    Time    ArtistId
            else:
                df = pd.read_csv( f'{DATA_PATH}nowplaying/raw/nowplaying.csv', sep='\t') # UserId    SessionId    ItemId    Time    Artist
            df.columns = [self.user_key, self.session_key, self.item_key, 'Time', self.category_key]
            df[self.time_key] = df['Time'].apply(lambda x: dt.datetime.fromtimestamp(x))
            del df['Time']
            self.df = df
            self.n_days = 7
        else:
            self.logger.error(f'cannot load data: {dataset_name}')
            raise ValueError(f'cannot load data: {dataset_name}')

        #self.df = df


    def _set_map(self, df, key):
        codes = pd.Categorical(df[key]).codes + 1
        res = dict(zip(df[key], codes))
        return res, codes

    def _make_sessions(self, is_ordered=True, idle_time=None):
        if is_ordered:
            self.df.sort_values(
                by=[self.user_key, self.time_key],
                ascending=True,
                inplace=True
            )
        if idle_time:
            self.df['TimeShift'] = self.df[self.time_key].shift(1)
            self.df['TimeDiff'] = (self.df[self.time_key] - self.df['TimeShift']).dt.total_seconds().abs()
            self.df['SessionIdTmp'] = (self.df['TimeDiff'] > idle_time).astype( int )
            self.df[self.session_key] = self.df['SessionIdTmp'].cumsum( skipna=False )
            del self.df['SessionIdTmp'], self.df['TimeShift'], self.df['TimeDiff']
            self.df.sort_values( [self.session_key, self.time_key], ascending=True, inplace=True )
            self.logger.info(f'Finish making {self.session_key} for data by idle time')
        else:
            self.df['date'] = pd.to_datetime(self.df[self.time_key], unit='s').dt.date

            # check whether the day changes
            split_session = self.df['date'].values[1:] != self.df['date'].values[:-1]
            split_session = np.r_[True, split_session]
            # check whether the user changes
            new_user = self.df[self.user_key].values[1:] != self.df[self.user_key].values[:-1]
            new_user = np.r_[True, new_user]
            # a new sessions stars when at least one of the two conditions is verified
            new_session = np.logical_or(new_user, split_session)
            # compute the session ids
            session_ids = np.cumsum(new_session)
            self.df[self.session_key] = session_ids
            self.df.sort_values( [self.session_key, self.time_key], ascending=True, inplace=True )
            self.logger.info(f'Finish making {self.session_key} for data by one day')

    def _core_filter(self, pop_num=5, bad_sess_len=1, user_sess_num=5, user_num_good_sess=200):
        # drop duplicate interactions within the same session
        self.df.drop_duplicates(
            subset=[self.user_key, self.item_key, self.time_key],
            keep='first',
            inplace=True
        )
        # TODO this is totally different from daisy filter-core
        # keep items with >=pop_num interactions
        item_pop = self.df[self.item_key].value_counts()
        good_items = item_pop[item_pop >= pop_num].index
        self.df = self.df[self.df[self.item_key].isin(good_items)].reset_index(drop=True)

        # remove sessions with length < bad_sess_len
        session_length = self.df[self.session_key].value_counts()
        good_sessions = session_length[session_length > bad_sess_len].index
        self.df = self.df[self.df.session_id.isin(good_sessions)]

        # let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
        sess_per_user = self.df.groupby(self.user_key)[self.session_key].nunique()
        good_users = sess_per_user[(sess_per_user >= user_sess_num) & (sess_per_user < user_num_good_sess)].index
        self.df = self.df[self.df[self.user_key].isin(good_users)]

        self.user_num = self.df[self.user_key].nunique()
        self.item_num = self.df[self.item_key].nunique()

        self.logger.info(f'Finish filtering data, current length is: {len(self.df)}, user number: {self.user_num}, item number: {self.item_num}')
        
    def _filter(self, pop_num=5, bad_sess_len=1):
        data = self.df
        session_lengths = data.groupby(self.session_key).size()
#        print('#clicks: ', data.shape[0])
#        print('#sessions: ', data[self.session_key].nunique())
#        assert 1!=1
        data = data[np.in1d(data[self.session_key], session_lengths[(session_lengths>bad_sess_len)&(session_lengths<200)].index)]
        item_supports = data.groupby(self.item_key).size()
        data = data[np.in1d(data[self.item_key], item_supports[item_supports>=pop_num].index)]
        session_lengths = data.groupby(self.session_key).size()
        data = data[np.in1d(data[self.session_key], session_lengths[session_lengths>bad_sess_len].index)]
        self.df = data
        
        self.item_num = self.df[self.item_key].nunique()

        self.logger.info(f'Finish filtering data, current length is: {len(self.df)}, item number: {self.item_num}')
        
    def _reindex(self):
        self.used_items = self.df[self.item_key].unique()
        #self.user_map, self.df[self.user_key] = self._set_map(self.df, self.user_key)
        self.item_map, self.df[self.item_key] = self._set_map(self.df, self.item_key)
  
    
    def get_seq_from_df(self):
        dic = self.df[[self.user_key, self.session_key]].drop_duplicates()
        seq = []
        for u, s in dic.values:
            items = self.df.query(f'{self.user_key} == {u} and {self.session_key} == {s}')[self.item_key].tolist()
            seq.append([u, s, items])

        return seq


class Categories(object):
    # TODO read category info, used for IDLS
    def __init__(self, item_map, config, logger):
        self.config = config
        self.logger = logger
        self.item_key = self.config['item_key']
        self.category_key = self.config['category_key']

        self.item_map = item_map
        self.item_set = list(item_map.keys())
        self.item_num = max(item_map.values()) + 1 # pad 0
        self._process_flow()

    def _process_flow(self):
        self._load_data()
        self._reindex()
        self._one_hot()
        #self._generate_cat_mat()

    def _load_data(self):
        dataset_name = self.config['dataset']
        self.dataset_name = dataset_name
        
        if not os.path.exists(f'{DATA_PATH}{dataset_name}/'):
            self.logger.error('unexisted dataset...')
        if dataset_name == 'ml-100k':
            df = pd.read_csv(
                f'{DATA_PATH}ml-100k/u.item',
                delimiter='|',
                header=None,
                encoding="ISO-8859-1"
            )

            genres = df.iloc[:,6:].values  # TODO not consider 'unknown'
            df[self.category_key] = pd.Series(genres.tolist())
            df.rename(columns={0: self.item_key}, inplace=True)
            df = df[[self.item_key, self.category_key]].copy()
            df = df[df[self.item_key].isin(self.item_set)].reset_index()
            
            self.n_cates = len(df[self.category_key][0]) + 1
        elif dataset_name == 'diginetica':
            df = pd.read_csv(
                f'{DATA_PATH}diginetica/product-categories.csv', sep=';'
            )
            df.columns = [self.item_key, self.category_key]
            df = df[df[self.item_key].isin(self.item_set)].reset_index()
            #df[self.item_key] = df[self.item_key].map(self.item_map)
            #print(df[self.item_key].nunique())
            #print(df.shape)
            
        elif dataset_name == 'retailrocket':
            cate = pd.read_csv(
                f'{DATA_PATH}retailrocket/category_tree.csv')
            item1 = pd.read_csv(
                f'{DATA_PATH}retailrocket/item_properties_part1.csv')
            item2 = pd.read_csv(
                f'{DATA_PATH}retailrocket/item_properties_part2.csv')
            item1 = item1[item1.property=='categoryid']
            item1.reset_index(drop=True, inplace=True)
            item2 = item2[item2.property=='categoryid']
            item2.reset_index(drop=True, inplace=True)
            # drop column, timestamp and property
            item1 = item1.drop(columns=['timestamp','property'])
            item2 = item2.drop(columns=['timestamp','property'])
            # stack two item dataframes
            item = pd.concat([item1, item2], ignore_index=True)
            # reset index
            item.reset_index(drop=True, inplace=True)
            # drop duplicates
            msk = item.duplicated()
            item = item[~msk]
            # rename column name 'value' to 'categoryid'
            item.rename(columns={'value':'categoryid'}, inplace=True)
            item.categoryid = item.categoryid.astype(int)
            # choose only one category to one item
            numClass_peritem = item[['itemid','categoryid']].groupby('itemid')['categoryid'].nunique()
            multi_class_item = numClass_peritem[numClass_peritem>1].index
            for i in multi_class_item:
                item.loc[item.itemid==i, 'categoryid'] = item[item.itemid==i]['categoryid'].value_counts().argmax()
            # drop duplicates
            msk = item.duplicated()
            item = item[~msk]
            item.columns = [self.item_key, self.category_key]
            df = item
            df = df[df[self.item_key].isin(self.item_set)].reset_index()
        elif dataset_name in ['tmall', 'taobao', 'aotm', '30music', 'nowplaying']:
            if dataset_name == 'tmall':
                df = pd.read_csv( f'{DATA_PATH}tmall/user_log_format1.csv', sep=',', header=0, usecols=[1,2])
                df.columns = [self.item_key, self.category_key]
            elif dataset_name == 'taobao':
                df = pd.read_csv( f'{DATA_PATH}taobao/UserBehavior.csv')
                df.columns = ['userId', self.item_key, self.category_key, 'ActionType', 'Time']
                df = df[[self.item_key, self.category_key]]
            elif dataset_name == 'aotm':
                df = pd.read_csv( f'{DATA_PATH}aotm/raw/playlists-aotm.csv', sep='\t', usecols=[2,4]) # UserId    SessionId    ItemId    Time    ArtistId
                df.columns = [self.item_key, self.category_key]
            elif dataset_name == '30music':
                df = pd.read_csv( f'{DATA_PATH}30music/raw/30music-200ks.csv', sep='\t', usecols=[2,4]) # UserId    SessionId    ItemId    Time    ArtistId
                df.columns = [self.item_key, self.category_key]
            else:
                df = pd.read_csv( f'{DATA_PATH}nowplaying/raw/nowplaying.csv', sep='\t', usecols=[2,4]) # UserId    SessionId    ItemId    Time    Artist
                df.columns = [self.item_key, self.category_key]
            
            # remain items in interactions and map to new index
            df = df[df[self.item_key].isin(self.item_set)].reset_index(drop=True)
            #df[self.item_key] = df[self.item_key].map(self.item_map)
            # choose only one category to one item
            numClass_peritem = df[[self.item_key, self.category_key]].groupby(self.item_key)[self.category_key].nunique()
            multi_class_item = numClass_peritem[numClass_peritem>1].index
            for i in multi_class_item:
                df.loc[df[self.item_key]==i, self.category_key] = df[df[self.item_key]==i][self.category_key].value_counts().argmax()
            # drop duplicates
            msk = df.duplicated()
            df = df[~msk]
        elif dataset_name == 'globo':
            df = pd.read_csv(f'{DATA_PATH}globo/articles_metadata.csv', usecols=[0,1])
            df.columns = [self.item_key, self.category_key]
            df = df[df[self.item_key].isin(self.item_set)].reset_index()
        else:
            self.logger.error(f'cannot load item information data: {dataset_name}')
            raise ValueError(f'cannot load item information data: {dataset_name}')
        self.df = df

    def _reindex(self):
        self.df[self.item_key] = self.df[self.item_key].map(self.item_map)
        # add category reindex
        self.df, cate_id_map, id_cate_map = reindex(self.df, self.category_key, start_from_zero=False)
        self.n_cates = max(cate_id_map.values()) + 1 # pad 0
        
    def _generate_cat_mat(self):
        item_cate_matrix = torch.zeros(self.item_num, self.n_cates)
        self.df.sort_values(by=self.item_key, inplace=True)
        item_cate = torch.tensor(self.df[self.category_key])
        item_cate_matrix[1:, 1:] = item_cate

        self.item_cate_matrix = item_cate_matrix

    def _one_hot(self):
        '''
        TODO
        some dataset categories are not like ml-100k,
        so we need to process these data to one-hot vector
        '''
        # item_cate_matrix[1:, :] = F.one_hot(item_cate, num_classes=self.n_cates)
        item_cate_matrix = torch.zeros(self.item_num, self.n_cates)
        # if item has no categoryid, set default vector all 0
#        items_usable_sorted = self.df.sort_values(by=self.item_key)
#        item_cate = torch.tensor(items_usable_sorted[self.category_key])
#        item_cate_matrix[1:, :] = F.one_hot(item_cate, num_classes=self.n_cates)
        items_available = self.df[self.item_key].values
        cates_respect = self.df[self.category_key].values
        num = len(items_available)
        mod = int(num//10000) + 1
        for i in range(mod):
#            print(torch.LongTensor(cates_respect[i*10000:min((i+1)*10000, num)]))
#            print(self.n_cates)
            item_cate_matrix[torch.LongTensor(items_available[i*10000:min((i+1)*10000, num)]), :] = F.one_hot(torch.LongTensor(cates_respect[i*10000:min((i+1)*10000, num)]), num_classes=int(self.n_cates)).float()
        self.item_cate_matrix = item_cate_matrix
