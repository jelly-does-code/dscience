from pandas import DataFrame
# Logging function
def log(msg, fname='log.txt'):
    if isinstance(msg, str):
        with open(fname, 'a') as file:
            file.write(msg + '\n')
            file.write('' + '\n')
    elif isinstance(msg, DataFrame):
        with open(fname, 'a') as file:
            msg.to_csv(fname, index=False)

