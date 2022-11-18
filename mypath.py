class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return 'C:/Users/oem/Desktop/citydata'

        elif dataset == 'nightcity':
            return 'C:/Users/oem/Desktop/nightcitydata'

        elif dataset == 'video':
            return 'C:/Users/oem/Desktop/video_data'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError