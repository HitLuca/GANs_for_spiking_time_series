import logging
import os
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('setup')

berka_dataset_folder = 'datasets/berka_dataset/'

if not os.path.exists(berka_dataset_folder + 'original'):
    if not os.path.isfile(berka_dataset_folder + 'data_berka.zip'):
        logger.info('downloading Berka dataset')
        urllib.request.urlretrieve('http://lisp.vse.cz/pkdd99/DATA/data_berka.zip',
                                   berka_dataset_folder + 'data_berka.zip')
        logger.info('done')

    logger.info('extracting dataset')
    zip_ref = zipfile.ZipFile(berka_dataset_folder + 'data_berka.zip', 'r')
    zip_ref.extractall(berka_dataset_folder + 'original')
    zip_ref.close()
    logger.info('done')

logger.info('converting and parsing dataset')
os.chdir('datasets/berka_dataset')
os.system('python3 dataset_creation.py')
logger.info('done')
