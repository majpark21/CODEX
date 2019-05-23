from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from functools import reduce
import pandas as pd

#Todo: function to get representation and probability of each trajectory


def tensorboard_to_df(path_to_logs, tags):
    """
    Read tensorboard logs and return a DataFrame with values. Useful for further plotting.
    :param path_to_logs: path to directory with log files
    :param tag: string name of the tag to read and export. Can pass a string or a list of strings. Tags must have
    common step vector in the logs.
    :return: A pandas Dataframe in wide format with one column for each tag
    """

    if isinstance(tags, str):
        tags = [tags]
    dfList = []
    event_acc = EventAccumulator(path_to_logs)
    event_acc.Reload()
    for tag in tags:
        w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
        df = pd.DataFrame(data={'step': step_nums, tag: vals})
        dfList.append(df)

    # Merge all dataframes
    out = reduce(lambda x, y: pd.merge(x, y, on='step'), dfList)
    return out