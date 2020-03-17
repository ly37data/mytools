import pandas as pd
from feature_extraction import primitives
from feature_extraction.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from feature_extraction.primitives.utils import serialize_primitive
from feature_extraction.utils.wrangle import (
    _check_time_against_column,
    _check_timedelta
)
from feature_extraction.variable_types import (
    Boolean,
    Categorical,
    Datetime,
    DatetimeTimeIndex,
    Discrete,
    Id,
    Index,
    Numeric,
    NumericTimeIndex,
    Variable
)


class FeatureBase(object):
    def __init__(self, entity, base_features, primitive, name=None, names=None,time_widow=None,
                 filters = None,prefix_name=None,prefix_cn_name=None):
        """Base class for all features

        Args:
            entity (Entity): entity this feature is being calculated for
            base_features (list[FeatureBase]): list of base features for primitive
            primitive (:class:`.PrimitiveBase`): primitive to calculate. if not initialized when passed, gets initialized with no arguments
        """
        assert all(isinstance(f, FeatureBase) for f in base_features), \
            "All base features must be features"
        self.entity = entity
        self.index = entity.index
        self.base_features = base_features
 
        # initialize if not already initialized
        if not isinstance(primitive, PrimitiveBase):
            primitive = primitive()
        self.primitive = primitive

        self._name = name
        self._cn_name = None
        self._names = names
        self._cn_names = None
        self._prefix_name = prefix_name if prefix_name is not None else ''
        self._prefix_cn_name = prefix_cn_name if prefix_cn_name is not None else ''
        self.is_return_f = False
        
        assert self._check_input_types(), ("Provided inputs don't match input "
                                           "type requirements")

    def __getitem__(self, key):
        assert self.number_output_features > 1, \
            'can only access slice of multi-output feature'
        assert self.number_output_features > key, \
            'index is higher than the number of outputs'
        return FeatureOutputSlice(self, key)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        raise NotImplementedError("Must define from_dictionary on FeatureBase subclass")

    def rename(self, name):
        """Rename Feature, returns copy"""
        feature_copy = self.copy()
        feature_copy._name = name
        return feature_copy

    def copy(self):
        raise NotImplementedError("Must define copy on FeatureBase subclass")

    def get_name(self):

        self._name = "%s%s"%(self._prefix_name,
                              self.generate_name())
        return self._name

    def get_cn_name(self):

        self._cn_name = u"%s%s"%(self._prefix_cn_name,
                              self.generate_cn_name())
        return self._cn_name
    
    def get_name_map(self):
        n = self.number_output_features
        if n == 1:
            return pd.DataFrame([[self.get_name(),self.get_cn_name()]])
        else:
            return pd.DataFrame([[name,cnname] for name,cnname in zip(self.get_names(),self.get_cn_names())])
    
    def get_names(self):
        if not self._names:
            self._names = ["%s%s"%(self._prefix_name,
                              generate_name) for generate_name in  self.generate_names()]
        return self._names

    def get_cn_names(self):
        if not self._cn_names:
            self._cn_names = ["%s%s"%(self._prefix_cn_name,
                              generate_cn_name) for generate_cn_name in  self.generate_cn_names()]
        return self._cn_names

    def get_feature_names(self):
        n = self.number_output_features
        if n == 1:
            names = [self.get_name()]
        else:
            names = self.get_names()
        return names

    def get_function(self):
        return self.primitive.get_function()

    def get_dependencies(self, deep=False, copy=True):
        """Returns features that are used to calculate this feature

        ..note::

            If you only want the features that make up the input to the feature
            function use the base_features attribute instead.

        """
        deps = []

        for d in self.base_features[:]:
            deps += [d]

        if deep:
            for dep in deps[:]:  # copy so we don't modify list we iterate over
                deep_deps = dep.get_dependencies(deep)
                deps += deep_deps

        return deps

    def get_depth(self, stop_at=None):
        """Returns depth of feature"""
        max_depth = 0
        stop_at_set = set()
        if stop_at is not None:
            stop_at_set = set([i.unique_name() for i in stop_at])
            if self.unique_name() in stop_at_set:
                return 0
        for dep in self.get_dependencies(deep=True, ignored=stop_at_set):
            max_depth = max(dep.get_depth(stop_at=stop_at),
                            max_depth)
        return max_depth + 1

    def _check_input_types(self):
        if len(self.base_features) == 0:
            return True

        input_types = self.primitive.input_types
        if input_types is not None:
            if type(input_types[0]) != list:
                input_types = [input_types]

            for t in input_types:
                zipped = list(zip(t, self.base_features))
                if all([issubclass(f.variable_type, v) for v, f in zipped]):
                    return True
        else:
            return True

        return False

    @property
    def number_output_features(self):
        return self.primitive.number_output_features

    def __repr__(self):
        return "<Feature: %s , %s>" % (self.get_name(),self.get_cn_name())

    def hash(self):
        return hash(self.get_name())

    def __hash__(self):
        # logger.warning("To hash a feature, use feature.hash()")
        return self.hash()

    @property
    def variable_type(self):
        feature = self
        variable_type = self.primitive.return_type

        while variable_type is None:
            # get variable_type of first base feature
            base_feature = feature.base_features[0]
            variable_type = base_feature.variable_type

            # only the original time index should exist
            # so make this feature's return type just a Datetime
            if variable_type == DatetimeTimeIndex:
                variable_type = Datetime
            elif variable_type == NumericTimeIndex:
                variable_type = Numeric
            elif variable_type == Index:
                variable_type = Categorical

            # direct features should keep the Id return type, but all other features should get
            # converted to Categorical

            feature = base_feature

        return variable_type

    @property
    def default_value(self):
        return self.primitive.default_value

    def get_arguments(self):
        raise NotImplementedError("Must define get_arguments on FeatureBase subclass")

    def to_dictionary(self):
        return {
            'type': type(self).__name__,
            'dependencies': [dep.unique_name() for dep in self.get_dependencies()],
            'arguments': self.get_arguments(),
        }

    def _handle_binary_comparision(self, other, Primitive, PrimitiveScalar):
        if isinstance(other, FeatureBase):
            return Feature([self, other], primitive=Primitive)

        return Feature([self], primitive=PrimitiveScalar(other))

    def __eq__(self, other):
        """Compares to other by equality"""
        return self._handle_binary_comparision(other, primitives.Equal, primitives.EqualScalar)

    def __ne__(self, other):
        """Compares to other by non-equality"""
        return self._handle_binary_comparision(other, primitives.NotEqual, primitives.NotEqualScalar)

    def __gt__(self, other):
        """Compares if greater than other"""
        return self._handle_binary_comparision(other, primitives.GreaterThan, primitives.GreaterThanScalar)

    def __ge__(self, other):
        """Compares if greater than or equal to other"""
        return self._handle_binary_comparision(other, primitives.GreaterThanEqualTo, primitives.GreaterThanEqualToScalar)

    def __lt__(self, other):
        """Compares if less than other"""
        return self._handle_binary_comparision(other, primitives.LessThan, primitives.LessThanScalar)

    def __le__(self, other):
        """Compares if less than or equal to other"""
        return self._handle_binary_comparision(other, primitives.LessThanEqualTo, primitives.LessThanEqualToScalar)

    def __add__(self, other):
        """Add other"""
        return self._handle_binary_comparision(other, primitives.AddNumeric, primitives.AddNumericScalar)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract other"""
        return self._handle_binary_comparision(other, primitives.SubtractNumeric, primitives.SubtractNumericScalar)

    def __rsub__(self, other):
        return Feature([self], primitive=primitives.ScalarSubtractNumericFeature(other))

    def __div__(self, other):
        """Divide by other"""
        return self._handle_binary_comparision(other, primitives.DivideNumeric, primitives.DivideNumericScalar)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rdiv__(self, other):
        return Feature([self], primitive=primitives.DivideByFeature(other))

    def __mul__(self, other):
        """Multiply by other"""
        if isinstance(other, FeatureBase):
            if self.variable_type == Boolean and other.variable_type == Boolean:
                return Feature([self, other], primitive=primitives.MultiplyBoolean)
        return self._handle_binary_comparision(other, primitives.MultiplyNumeric, primitives.MultiplyNumericScalar)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        """Take modulus of other"""
        return self._handle_binary_comparision(other, primitives.ModuloNumeric, primitives.ModuloNumericScalar)

    def __rmod__(self, other):
        return Feature([self], primitive=primitives.ModuloByFeature(other))

    def __and__(self, other):
        return self.AND(other)

    def __rand__(self, other):
        return Feature([other, self], primitive=primitives.And)

    def __or__(self, other):
        return self.OR(other)

    def __ror__(self, other):
        return Feature([other, self], primitive=primitives.Or)

    def __not__(self, other):
        return self.NOT(other)

    def __abs__(self):
        return Feature([self], primitive=primitives.Absolute)

    def __neg__(self):
        return Feature([self], primitive=primitives.Negate)

    def AND(self, other_feature):
        """Logical AND with other_feature"""
        return Feature([self, other_feature], primitive=primitives.And)

    def OR(self, other_feature):
        """Logical OR with other_feature"""
        return Feature([self, other_feature], primitive=primitives.Or)

    def NOT(self):
        """Creates inverse of feature"""
        return Feature([self], primitive=primitives.Not)

    def isin(self, list_of_output):
        return Feature([self], primitive=primitives.IsIn(list_of_outputs=list_of_output))

    def is_null(self):
        """Compares feature to null by equality"""
        return Feature([self], primitive=primitives.IsNull)

    def __invert__(self):
        return self.NOT()

    def unique_name(self):
        return u"%s" % (self.get_name())


class IdentityFeature(FeatureBase):
    """Feature for entity that is equivalent to underlying variable"""

    def __init__(self, variable, name=None):
        self.variable = variable
        self.return_type = type(variable)
        super(IdentityFeature, self).__init__(entity=variable.entity,
                                              base_features=[],
                                              primitive=PrimitiveBase,
                                              name=name)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        entity_id = arguments['entity_id']
        variable_id = arguments['variable_id']
        variable = entityset[entity_id][variable_id]
        return cls(variable=variable, name=arguments['name'])

    def copy(self):
        """Return copy of feature"""
        return IdentityFeature(self.variable)

    def generate_name(self):
        return self.variable.name

    def generate_cn_name(self):
        return self.variable.name


    def get_depth(self, stop_at=None):
        return 0

    def get_arguments(self):
        return {
            'name': self._name,
            'variable_id': self.variable.id,
            'entity_id': self.variable.entity_id,
        }

    @property
    def variable_type(self):
        return type(self.variable)

class AggregationFeature(FeatureBase):

    def __init__(self, base_features, primitive,entity, where=None, name=None,cn_name=None,time_window=None):
        if hasattr(base_features, '__iter__'):
            base_features = [_check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        else:
            base_features = [_check_feature(base_features)]

        for bf in base_features:
            if bf.number_output_features > 1:
                raise ValueError("Cannot stack on whole multi-output feature.")
                
        self.time_window = time_window
        self.where = where
        self.name = name
        self.cn_name = cn_name
        
        super(AggregationFeature, self).__init__(entity=entity,
                                                 base_features=base_features,
                                                 primitive=primitive,
                                                 name=name)
    def copy(self):
        return AggregationFeature(self.base_features,
                                  primitive=self.primitive,
                                  entity = self.entity,
                                  where=self.where,
                                  name = self.name,
                                  cn_name = self.cn_name,
                                  time_window = self.time_window
                                  )

    def _where_str(self,cn=False):
        if cn:
            if self.where is not None:
                where_str = self.where.get_cn_name()
            else:
                where_str = ''
            return where_str
        else:
            if self.where is not None:
                where_str = self.where.get_name()
            else:
                where_str = ''
            return where_str

    def _time_window_str(self,cn=False):
        if cn:
            if self.time_window is not None :
                time_window_str = "近{}{}_".format(self.time_window,self.entity.time_filter_cn_name)
            else:
                time_window_str = ''
            return time_window_str
        else:
            if self.time_window is not None :
                time_window_str = "Last{}{}_".format(self.time_window,self.entity.time_filter_name)
            else:
                time_window_str = ''
            return time_window_str
    
    def generate_name(self):
        if self.name is not None:
            return self.primitive.generate_name(base_feature_names=[self.name],
                                            where_str=self._where_str(),
                                            time_window_str=self._time_window_str())

        
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features],
                                            where_str=self._where_str(),
                                            time_window_str=self._time_window_str())

    def generate_cn_name(self):
        if self.cn_name is not None:
            return self.primitive.generate_name(base_feature_names=[self.cn_name],
                                            where_str=self._where_str(cn=True),
                                            time_window_str=self._time_window_str(cn=True),
                                            cn=True)

        
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features],
                                            where_str=self._where_str(cn=True),
                                            time_window_str=self._time_window_str(cn=True),
                                            cn=True)

    def generate_names(self):
        if self.name is not None:
            return self.primitive.generate_names(base_feature_names=[self.name],
                                            where_str=self._where_str(),
                                            time_window_str=self._time_window_str())


        return self.primitive.generate_names(base_feature_names=[bf.get_name() for bf in self.base_features],
                                             where_str=self._where_str(),
                                             time_window_str=self._time_window_str())

    def generate_cn_names(self):
        if self.name is not None:
            return self.primitive.generate_names(base_feature_names=[self.cn_name],
                                            where_str=self._where_str(cn=True),
                                            time_window_str=self._time_window_str(cn=True),
                                            cn=True)


        return self.primitive.generate_names(base_feature_names=[bf.get_name() for bf in self.base_features],
                                             where_str=self._where_str(cn=True),
                                             time_window_str=self._time_window_str(cn=True),
                                             cn=True)





class TransformFeature(FeatureBase):
    def __init__(self, base_features, primitive, name=None,cn_name=None):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_trans_primitive
        if hasattr(base_features, '__iter__'):
            base_features = [_check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        else:
            base_features = [_check_feature(base_features)]
        
        for bf in base_features:
            if bf.number_output_features > 1:
                raise ValueError("Cannot stack on whole multi-output feature.")
        self.name = name
        self.cn_name = cn_name
        super(TransformFeature, self).__init__(entity=base_features[0].entity,
                                               base_features=base_features,
                                               primitive=primitive,
                                               name=name)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_features = [dependencies[name] for name in arguments['base_features']]
        primitive = primitives_deserializer.deserialize_primitive(arguments['primitive'])
        return cls(base_features=base_features, primitive=primitive, name=arguments['name'])

    def copy(self):
        return TransformFeature(self.base_features, self.primitive,name = self.name,cn_name = self.cn_name)

    def generate_name(self):
        if self.is_return_f :
            return self.primitive.generate_name(base_feature_names=[self.name],is_return = self.is_return_f)
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features])

    def generate_cn_name(self):
        if self.is_return_f:
            return self.primitive.generate_name(base_feature_names=[self.cn_name],cn=True,is_return = self.is_return_f)
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features],
                                                                cn=True)

    def generate_names(self):
        return self.primitive.generate_names(base_feature_names=[bf.get_name() for bf in self.base_features])

    def get_arguments(self):
        return {
            'name': self._name,
            'base_features': [feat.unique_name() for feat in self.base_features],
            'primitive': serialize_primitive(self.primitive)
        }


class GroupByTransformFeature(TransformFeature):
    def __init__(self, base_features, primitive, groupby, name=None):
        if not isinstance(groupby, FeatureBase):
            groupby = IdentityFeature(groupby)
        assert issubclass(groupby.variable_type, Discrete)
        self.groupby = groupby

        if hasattr(base_features, '__iter__'):
            base_features.append(groupby)
        else:
            base_features = [base_features, groupby]

        super(GroupByTransformFeature, self).__init__(base_features=base_features,
                                                      primitive=primitive,
                                                      name=name)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_features = [dependencies[name] for name in arguments['base_features']]
        primitive = primitives_deserializer.deserialize_primitive(arguments['primitive'])
        groupby = dependencies[arguments['groupby']]
        return cls(base_features=base_features, primitive=primitive, groupby=groupby, name=arguments['name'])

    def copy(self):
        # the groupby feature is appended to base_features in the __init__
        # so here we separate them again
        return GroupByTransformFeature(self.base_features[:-1],
                                       self.primitive,
                                       self.groupby)

    def generate_name(self):
        # exclude the groupby feature from base_names since it has a special
        # place in the feature name
        base_names = [bf.get_name() for bf in self.base_features[:-1]]
        _name = self.primitive.generate_name(base_names)
        return u"{} by {}".format(_name, self.groupby.get_name())

    def generate_names(self):
        base_names = [bf.get_name() for bf in self.base_features[:-1]]
        _names = self.primitive.generate_names(base_names)
        names = [name + " by {}".format(self.groupby.get_name()) for name in _names]
        return names

    def get_arguments(self):
        # Do not include groupby in base_features.
        feature_names = [feat.unique_name() for feat in self.base_features
                         if feat.unique_name() != self.groupby.unique_name()]
        return {
            'name': self._name,
            'base_features': feature_names,
            'primitive': serialize_primitive(self.primitive),
            'groupby': self.groupby.unique_name(),
        }


class Feature(object):
    """
    Alias to create feature. Infers the feature type based on init parameters.
    """

    def __new__(self, base, entity=None, groupby=None, parent_entity=None,
                primitive=None, use_previous=None, where=None):
        # either direct or indentity
        if primitive is None and entity is None:
            return IdentityFeature(base)

        elif primitive is not None and parent_entity is not None:
            assert isinstance(primitive, AggregationPrimitive) or issubclass(primitive, AggregationPrimitive)
            return AggregationFeature(base, parent_entity=parent_entity,
                                      use_previous=use_previous, where=where,
                                      primitive=primitive)
        elif primitive is not None:
            assert (isinstance(primitive, TransformPrimitive) or
                    issubclass(primitive, TransformPrimitive))
            if groupby is not None:
                return GroupByTransformFeature(base,
                                               primitive=primitive,
                                               groupby=groupby)
            return TransformFeature(base, primitive=primitive)

        raise Exception("Unrecognized feature initialization")


class FeatureOutputSlice(FeatureBase):
    """
    Class to access specific multi output feature column
    """

    def __init__(self, base_feature, n, name=None):
        base_features = [base_feature]
        self.num_output_parent = base_feature.number_output_features

        msg = "cannot access slice from single output feature"
        assert(self.num_output_parent > 1), msg
        msg = "cannot access column that is not between 0 and " + str(self.num_output_parent - 1)
        assert(n < self.num_output_parent), msg

        self.n = n
        self._name = name
        self.base_features = base_features
        self.base_feature = base_features[0]

        self.entity_id = base_feature.entity_id
        self.entityset = base_feature.entityset
        self.primitive = base_feature.primitive

        self.relationship_path = base_feature.relationship_path

    def __getitem__(self, key):
        raise ValueError("Cannot get item from slice of multi output feature")

    def generate_name(self):
        return self.base_feature.get_names()[self.n]

    @property
    def number_output_features(self):
        return 1

    def get_arguments(self):
        return {
            'name': self._name,
            'base_feature': self.base_feature,
            'n': self.n
        }

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_feature = arguments['base_feature']
        n = arguments['n']
        name = arguments['name']
        return cls(base_feature=base_feature, n=n, name=name)


def _check_feature(feature):
    if isinstance(feature, Variable):
        return IdentityFeature(feature)
    elif isinstance(feature, FeatureBase):
        return feature

    raise Exception("Not a feature")
