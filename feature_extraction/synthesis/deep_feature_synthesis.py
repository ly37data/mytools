import logging
import pandas as pd
from feature_extraction.feature_base import (
    AggregationFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)

logger = logging.getLogger('feature_extraction')


class DeepFeatureSynthesis(object):
    """ 
    produce features for an Entity.

    """

    def __init__(self, entity):

        self.es = entity
       
    def build_features(self,verbose=False):
        """ builds feature definitions for config 

        Args:

            verbose (bool, optional): If True, print progress.

        Returns:
            list[BaseFeature]: Returns a list of
                features for target entity, sorted by feature depth
                (shallow first).
            
            pd.DataFrame: Returns a dataframe of feature names and feature cn names.
        """
        all_features = {}

        self._run_dfs(self.es,all_features)
                
        new_features = list(all_features.values())

        feature_cn_map = [new_feature.get_name_map() for new_feature in new_features]
        feature_cn_map = pd.concat(feature_cn_map,sort=False,ignore_index=True)
        feature_cn_map.columns = ['feature_name','cn_desc']

        if verbose:
            logger.info("Built {} features".format(feature_cn_map.shape[0]))

        return new_features,feature_cn_map

    def _filter_features(self, new_features):
        if self.es.time_index:
            new_features = [new_feature for new_feature in new_features for time_w in self.es.time_window]
                
        return new_features

    def _run_dfs(self, entity,all_features):
        """
        create features for the provided entity

        Args:
            entityset (EntitySet): Entity for which to create features.
            relationship_path (RelationshipPath): The path to this entity.
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
            max_depth (int) : Maximum allowed depth of features.
        """

        """
        Create features
        """        
        for feature_ent in entity.feature_ents_base:
            feature_names = feature_ent.feature

            features = [feature for feature_name in 
                       feature_names for feature in entity.variables  if feature.id == feature_name]
            
            new_features = [IdentityFeature(feature) for feature in features]

            if len(feature_ent.trans_primitives)>0:
                new_features = self._build_transform_features(new_features,feature_ent)
            else:
                new_features = new_features

            if len(feature_ent.agg_primitives)>0:
                new_features = [AggregationFeature(feature,primitive=aggs,entity=entity,
                                                   time_window=time_w,where = where,
                                                   name = feature_ent.feature_name,
                                                   cn_name = feature_ent.feature_cn_name)
                                for feature in new_features                                
                                for aggs in feature_ent.agg_primitives
                                for time_w in self.es.time_window
                                for where in self.es.where]
                
            for new_feature in new_features:
                self._handle_new_feature(new_feature,all_features)
                
                
    def _handle_new_feature(self, new_feature, all_features):
        """Adds new feature to the dict

        Args:
            new_feature (:class:`.FeatureBase`): New feature being
                checked.
            all_features (dict):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.

        Returns:
            dict: Dict of features with any new features.

        Raises:
            Exception: Attempted to add a single feature multiple times
        """

        if self.es.prefix_name is not None:
            new_feature._prefix_name = self.es.prefix_name+'.'
        if self.es.prefix_cn_name is not None:
            new_feature._prefix_cn_name = self.es.prefix_cn_name+'.'
        new_feature.is_return_f = True
        name = new_feature.unique_name()
        # Warn if this feature is already present.
        if name in all_features :
            logger.warning('Attempting to add feature %s which is already '
                           'present. This is likely a bug.' % new_feature)
            return
        
        all_features[name] = new_feature

    def _build_transform_features(self, features, feature_ent):
        """Creates trans_features for all the variables in an entity

        Args:
            features：
            
            feature_ent:
                
        """
        for trans in feature_ent.trans_primitives:
            input_types = trans.input_types
            if type(input_types[0]) == list:
                features = [TransformFeature(features,
                                     primitive=trans,
                                     name = feature_ent.feature_name,
                                     cn_name = feature_ent.feature_cn_name)]
            else:
                features = [TransformFeature(feature,
                                     primitive=trans,
                                     name = feature_ent.feature_name,
                                     cn_name = feature_ent.feature_cn_name)
                            for feature in features]                
        return features
