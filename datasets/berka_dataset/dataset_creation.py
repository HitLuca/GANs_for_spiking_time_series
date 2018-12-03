import datetime
import logging
import os
from datetime import date

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dataset_creation')

TOTAL_DAYS = 2190


def parse_dataset():
    for filename in os.listdir('original'):
        logger.debug(filename)
        with open('original/' + filename, 'r') as input_file, open('parsed/' + filename.split('.')[0] + '.csv',
                                                                   'w') as output_file:
            for line in input_file.readlines():
                print(line.replace('"', '').replace('\n', '').replace(';', ', ').replace(' ', ''), file=output_file)


def split_date(input_date):
    year = 1900 + int(str(input_date)[0:2])
    month = int(str(input_date)[2:4])
    day = int(str(input_date)[4:6])
    return year, month, day


def date_to_index(input_date):
    input_year, input_month, input_day = split_date(input_date)
    delta = date(input_year, input_month, input_day) - date(1993, 1, 1)
    return delta.days


def generate_transactions(account_id_list):
    transactions = np.zeros((len(account_id_list), TOTAL_DAYS), dtype=np.float32)
    for account_index, account_id in enumerate(account_id_list):
        if account_index % 100 == 0:
            logger.info(str(account_index) + '/' + str(len(account_id_list)))

        df_account = df.query('account_id == ' + str(account_id))
        for _, row in df_account.iterrows():
            transaction_date = row['date']
            amount = row['amount']
            if row['type'] == 'VYDAJ':
                amount = -amount
            try:
                transactions[account_index, date_to_index(transaction_date)] = amount
            except:
                pass

    logger.info(str(len(account_id_list)) + '/' + str(len(account_id_list)))
    return transactions


def generate_normalized_transactions(transactions):
    normalized_transactions = np.array(transactions)

    percentile_1 = np.percentile(normalized_transactions, 1)
    percentile_99 = np.percentile(normalized_transactions, 99)
    normalized_transactions[normalized_transactions < percentile_1] = percentile_1
    normalized_transactions[normalized_transactions > percentile_99] = percentile_99

    transactions_min = np.min(normalized_transactions)
    transactions_max = np.max(normalized_transactions)
    normalized_transactions = 2 * (
        (normalized_transactions - transactions_min) / (transactions_max - transactions_min)) - 1

    return normalized_transactions


def generate_normalized_transactions_months(transactions):
    d0 = date(1993, 1, 1)
    d1 = date(1998, 12, 31)
    new_transactions = np.zeros((4500, 30 * 12 * 6))

    current_date = d0
    current_month = 1
    row_index = 0

    while (d1 - current_date).days >= 0:
        if current_date.day <= 30:
            transaction_index = current_date.day - 1
            transaction_index += (current_date.month - 1) * 30
            transaction_index += (current_date.year - 1993) * 360
            new_transactions[:, transaction_index] = transactions[:, (current_date - d0).days]
        if current_date.month != current_month:
            row_index += 1
            current_month += 1
            if current_month == 13:
                current_month = 1
        current_date += datetime.timedelta(days=1)
    return new_transactions


def split_data(dataset, timesteps):
    D = dataset.shape[1]
    if D < timesteps:
        return None
    elif D == timesteps:
        return dataset
    else:
        splitted_data, remaining_data = np.hsplit(dataset, [timesteps])
        remaining_data = split_data(remaining_data, timesteps)
        if remaining_data is not None:
            return np.vstack([splitted_data, remaining_data])
        return splitted_data


if not os.path.exists('parsed'):
    logger.info('parsing dataset')
    os.mkdir('parsed')
    parse_dataset()
    logger.info('done')

if os.path.exists('usable'):
    os.rmdir('usable')

os.mkdir('usable')

df = pd.read_csv('parsed/trans.csv', low_memory=False)
account_id_list = df.account_id.unique()

logger.info('generating transactions')
transactions = generate_transactions(account_id_list)
np.save('usable/transactions.npy', transactions)
logger.info('done')

logger.info('generating normalized transactions')
normalized_transactions = generate_normalized_transactions(transactions)
np.save('usable/normalized_transactions.npy', normalized_transactions)
logger.info('done')

logger.info('generating normalized monthly transactions')
normalized_transactions_months = generate_normalized_transactions_months(normalized_transactions)
np.random.shuffle(normalized_transactions_months)
np.save('usable/normalized_transactions_months.npy', normalized_transactions_months)
logger.info('done')
