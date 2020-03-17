import itertools
import logging
from collections import defaultdict

from feature_extraction.feature_base import (
    AggregationFeature,
    FeatureOutputSlice,
    GroupByTransformFeature,
    TransformFeature
)

logger = logging.getLogger('feature_extraction.computational_backend')


class FeatureSet(object):
    """
    Represents an immutable set of features to be calculated for a single entity, and their
    dependencies.
    """

    def __init__(self, features):
        """
        Args:
            features (list[Feature]): Features of the target entity.
        """
        self.target_eid = features[0].entity.index
        self.target_features = features
        self.target_feature_names = {f.unique_name() for f in features}

        # Maps the unique name of each feature to the actual feature. This is necessary
        # because features do not support equality and so cannot be used as
        # dictionary keys. The equality operator on features produces a new
        # feature (which will always be truthy).
        self.features_by_name = {f.unique_name(): f for f in features}

        feature_dependents = defaultdict(set)
        for f in features:
            deps = f.get_dependencies(deep=True)
            for dep in deps:
                feature_dependents[dep.unique_name()].add(f.unique_name())
                self.features_by_name[dep.unique_name()] = dep
                subdeps = dep.get_dependencies(deep=True)
                for sd in subdeps:
                    feature_dependents[sd.unique_name()].add(dep.unique_name())

        # feature names (keys) and the features that rely on them (values).
        self.feature_dependents = {
            fname: [self.features_by_name[dname] for dname in feature_dependents[fname]]
            for fname, f in self.features_by_name.items()}

    def group_features(self):
        """
        Topologically sort the given features, then group by path,
        feature type, use_previous, and where.
        """
        features = [value for value in self.features_by_name.values()]
        depths = self._get_feature_depths(features)

        def key_func(f):
            return (depths[f.unique_name()],
                    str(f.__class__),
                    _get_time_window(f),
                    _get_where(f))

        # Sort the list of features by the complex key function above, then
        # group them by the same key
        sort_feats = sorted(features, key=key_func)
        feature_groups = [list(g) for _, g in
                          itertools.groupby(sort_feats, key=key_func)]

        return feature_groups

    def _get_feature_depths(self, features):
        """
        Generate and return a mapping of {feature name -> depth} in the
        feature DAG for the given entity.
        """
        order = defaultdict(int)
        depths = {}
        queue = features[:]
        while queue:
            # Get the next feature.
            f = queue.pop(0)

            depths[f.unique_name()] = order[f.unique_name()]

            dependencies = f.get_dependencies()
            for dep in dependencies:
                order[dep.unique_name()] = \
                    min(order[f.unique_name()] - 1, order[dep.unique_name()])
                queue.append(dep)

        return depths

    def uses_full_entity(self, feature, check_dependents=False):
        if isinstance(feature, TransformFeature) and feature.primitive.uses_full_entity:
            return True
        return check_dependents and self._dependent_uses_full_entity(feature)

    def _dependent_uses_full_entity(self, feature):
        for d in self.feature_dependents[feature.unique_name()]:
            if isinstance(d, TransformFeature) and d.primitive.uses_full_entity:
                return True
        return False

# These functions are used for sorting and grouping features


def _get_use_previous(f):  # TODO Sort and group features for DateOffset with two different temporal values
    if isinstance(f, AggregationFeature) and f.use_previous is not None:
        if len(f.use_previous.times.keys()) > 1:
            return ("", -1)
        else:
            unit = list(f.use_previous.times.keys())[0]
            value = f.use_previous.times[unit]
            return (unit, value)
    else:
        return ("", -1)


def _get_where(f):
    if isinstance(f, AggregationFeature) and f.where is not None:
        return f.where.get_name()
    else:
        return ''

def _get_time_window(f):
    if isinstance(f, AggregationFeature) and f.time_window is not None:
        return f._time_window_str()
    else:
        return ''




def _get_groupby(f):
    if isinstance(f, GroupByTransformFeature):
        return f.groupby.unique_name()
    else:
        return ''
