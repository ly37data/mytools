import logging
import os
import math
import cloudpickle

from feature_extraction.entityset.entity import Entity
from feature_extraction.computational_backends.feature_set import FeatureSet
from feature_extraction.computational_backends.feature_set_calculator import (
    FeatureSetCalculator
)
from feature_extraction.feature_base import FeatureBase
from feature_extraction.computational_backends.utils import (
    bin_cutoff_times,
    create_client_and_cluster,
    gather_approximate_features,
    gen_empty_approx_features_df,
    save_csv_decorator
)
from feature_extraction.utils.gen_utils import make_tqdm_iterator
from feature_extraction.utils.wrangle import _check_time_type
from feature_extraction.variable_types import (
    DatetimeTimeIndex,
    NumericTimeIndex,
    PandasTypes
)

logger = logging.getLogger('feature_extraction.computational_backend')

PBAR_FORMAT = "Elapsed: {elapsed} | Progress: {l_bar}{bar}"
FEATURE_CALCULATION_PERCENTAGE = .95  # make total 5% higher to allot time for wrapping up at end


def calculate_feature_matrix(features, entity=None, verbose=False,
                             chunk_size=None, n_jobs=1):
    """Calculates a matrix for a given set of instance ids and calculation times.

    Args:
        features (list[:class:`.FeatureBase`]): Feature definitions to be calculated.

        entity (Entity): An already initialized entity. 

        verbose (bool, optional): Print progress info. The time granularity is
            per chunk.

        chunk_size (int or float or None): maximum number of rows of
            output feature matrix to calculate at time. If passed an integer
            greater than 0, will try to use that many rows per chunk. If passed
            a float value between 0 and 1 sets the chunk size to that
            percentage of all rows. if None, and n_jobs > 1 it will be set to 1/n_jobs

        n_jobs (int, optional): number of parallel processes to use when
            calculating feature matrix.

        dask_kwargs (dict, optional): Dictionary of keyword arguments to be
            passed when creating the dask client and scheduler. Even if n_jobs
            is not set, using `dask_kwargs` will enable multiprocessing.
            Main parameters:

            cluster (str or dask.distributed.LocalCluster):
                cluster or address of cluster to send tasks to. If unspecified,
                a cluster will be created.
            diagnostics port (int):
                port number to use for web dashboard.  If left unspecified, web
                interface will not be enabled.

            Valid keyword arguments for LocalCluster will also be accepted.

        progress_callback (callable): function to be called with incremental progress updates.
            Has the following parameters:

                update: percentage change (float between 0 and 100) in progress since last call
                progress_percent: percentage (float between 0 and 100) of total computation completed
                time_elapsed: total time in seconds that has elapsed since start of call

    """
    assert (isinstance(features, list) and features != [] and
            all([isinstance(feature, FeatureBase) for feature in features])), \
        "features must be a non-empty list of features"

    # handle loading entityset
    if not isinstance(entity, Entity):
        raise TypeError( )


    groupby_var = entity.index
    frame = entity.df.drop_duplicates(groupby_var)[groupby_var].set_index(groupby_var)

    chunk_size = _handle_chunk_size(chunk_size, frame.shape[0])
    
    feature_set = FeatureSet(features)
    tqdm_options = {'total': (len(feature_set.features_by_name.values()) / FEATURE_CALCULATION_PERCENTAGE),
                    'bar_format': PBAR_FORMAT,
                    'disable': True}

    if verbose:
        tqdm_options.update({'disable': False})

    progress_bar = make_tqdm_iterator(**tqdm_options)

    feature_matrix = calculate_chunk(chunk_size=chunk_size,
                                         features=features,
                                         entity=entity,
                                         frame=frame,
                                         progress_bar=progress_bar,
                                         )

    # ensure rows are sorted by input order

    # force to 100% since we saved last 5 percent
    progress_bar.update(progress_bar.total - progress_bar.n)

    progress_bar.refresh()
    progress_bar.close()

    return feature_matrix


def calculate_chunk(chunk_size, features, entity, frame, progress_bar=None):
    feature_set = FeatureSet(features)
    if not isinstance(feature_set, FeatureSet):
        feature_set = cloudpickle.loads(feature_set)

    update_progress_callback = None

    if progress_bar is not None:
        def update_progress_callback(done):
            progress_bar.update(done)
    calculator = FeatureSetCalculator(entity,feature_set,features)
    matrix = calculator.run(frame, progress_callback=update_progress_callback)
    return matrix


def _handle_chunk_size(chunk_size, total_size):
    if chunk_size is not None:
        assert chunk_size > 0, "Chunk size must be greater than 0"

        if chunk_size < 1:
            chunk_size = math.ceil(chunk_size * total_size)

        chunk_size = int(chunk_size)

    return chunk_size