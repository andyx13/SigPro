# -*- coding: utf-8 -*-
"""Process Signals core functionality."""

from collections import Counter
from copy import deepcopy
from itertools import product
from abc import ABC

import pandas as pd
from mlblocks import MLPipeline, load_primitive
from mlblocks.mlblock import import_object


from sigpro import contributing, primitive
from sigpro.primitive import Primitive
from sigpro.basic_primitives import Identity
import json
import inspect






DEFAULT_INPUT = [
    {
        'name': 'readings',
        'keyword': 'data',
        'type': 'pandas.DataFrame'
    },
    {
        'name': 'feature_columns',
        'default': None,
        'type': 'list'
    }
]

DEFAULT_OUTPUT = [
    {
        'name': 'readings',
        'type': 'pandas.DataFrame'
    },
    {
        'name': 'feature_columns',
        'type': 'list'
    }
]

class Pipeline(ABC):

    def __init__(self):
        self.values_column_name = 'values'
        self.input_is_dataframe = True
        self.pipeline = None


    def set_values_column_name(values_column_name):
        self.values_column_name = values_column_name

    def accept_dataframe_input(self, input_is_dataframe):
        self.input_is_dataframe = input_is_dataframe

    def _apply_pipeline(self, window, is_series=False):
        """Apply a ``mlblocks.MLPipeline`` to a row.

        Apply a ``MLPipeline`` to a window of a ``pd.DataFrame``, this function can
        be combined with the ``pd.DataFrame.apply`` method to be applied to the
        entire data frame.

        Args:
            window (pd.Series):
                Row or multiple rows (window) used to apply the pipeline to.
            is_series (bool):
                Indicator whether window is formated as a series or dataframe.
        """
        if is_series:
            context = window.to_dict()
            amplitude_values = context.pop(self.values_column_name)
        else:
            context = {} if window.empty else {
                k: v for k, v in window.iloc[0].to_dict().items() if k != self.values_column_name
            }
            amplitude_values = list(window[self.values_column_name])

        output = self.pipeline.predict(
            amplitude_values=amplitude_values,
            **context,
        )
        output_names = self.pipeline.get_output_names()

        # ensure that we can iterate over output
        output = output if isinstance(output, tuple) else (output, )

        return pd.Series(dict(zip(output_names, output)))

    def process_signal(self, data=None, window=None, time_index=None, groupby_index=None, 
                       feature_columns=None, values_column_name = 'values', keep_columns = False, **kwargs):
        """Apply multiple transformation and aggregation primitives.

        Args:
            data (pandas.DataFrame):
                Dataframe with a column that contains signal values.
            window (str):
                Duration of window size, e.g. ('1h').
            time_index (str):
                Column in ``data`` that represents the time index.
            groupby_index (str or list[str]):
                Column(s) to group together and take the window over.
            feature_columns (list):
                List of column names from the input data frame that must be considered as
                features and should not be dropped.            
            keep_columns (Union[bool, list]):
                Whether to keep non-feature columns in the output DataFrame or not.
                If a list of column names are passed, those columns are kept.

        Returns:
            tuple:
                pandas.DataFrame:
                    A data frame with new feature columns by applying the previous primitives. If
                    ``keep_values`` is ``True`` the original signal values will be conserved in the
                    data frame, otherwise the original signal values will be deleted.
                list:
                    A list with the feature names generated.
        """
        self.values_column_name = values_column_name

        if data is None:
            window = pd.Series(kwargs)
            values = self._apply_pipeline(window, is_series=True).values
            return values if len(values) > 1 else values[0]

        data = data.copy()
        if window is not None and groupby_index is not None:
            features = data.set_index(time_index).groupby(groupby_index).resample(
                rule=window, **kwargs).apply(
                self._apply_pipeline
            ).reset_index()
            data = features

        else:
            features = data.apply(
                self._apply_pipeline,
                axis=1,
                is_series=True
            )
            data = pd.concat([data, features], axis=1)

        if feature_columns:
            feature_columns = feature_columns + list(features.columns)
        else:
            feature_columns = list(features.columns)

        if isinstance(keep_columns, list):
            data = data[keep_columns + feature_columns]
        elif not keep_columns:
            data = data[feature_columns]

        return data, feature_columns

    def get_input_args(self):
        """Return the pipeline input args."""
        if self.input_is_dataframe:
            return deepcopy(DEFAULT_INPUT)

        return self.pipeline.get_predict_args()

    def get_output_args(self):
        """Return the pipeline output args."""
        if self.input_is_dataframe:
            return deepcopy(DEFAULT_OUTPUT)

        return self.pipeline.get_outputs()


    
"""
Analogue of sigpro.SigPro object in current use, takes in same arguments.
Only distinction is that we accept primitive objects, rather than dict inputs.
"""
class LinearPipeline(Pipeline): 

    def __init__(self, transformations, aggregations): 

        super().__init__()
        self.primitive = 'sigpro.SigPro' #change later.

        self.transformations = transformations
        self.aggregations = aggregations

        #     pass
        primitives = []
        init_params = {}
        prefix = []
        outputs = []
        counter = Counter()

        for transformation_ in self.transformations:
            transformation_._validate_primitive_spec()
            transformation = transformation_.get_hyperparam_dict()

            name = transformation.get('name')
            if name is None:
                name = transformation['primitive'].split('.')[-1]

            prefix.append(name)
            primitive = transformation['primitive']
            counter[primitive] += 1
            primitive_name = f'{primitive}#{counter[primitive]}'
            primitives.append(primitive)
            params = transformation.get('init_params')
            if params:
                init_params[primitive_name] = params

        prefix = '.'.join(prefix) if prefix else ''

        for aggregation_ in self.aggregations:
            aggregation_._validate_primitive_spec()
            aggregation = aggregation_.get_hyperparam_dict()

            name = aggregation.get('name')
            if name is None:
                name = aggregation['primitive'].split('.')[-1]

            aggregation_name = f'{prefix}.{name}' if prefix else name

            primitive = aggregation['primitive']
            counter[primitive] += 1
            primitive_name = f'{primitive}#{counter[primitive]}'
            primitives.append(primitive)

            primitive = aggregation_.make_primitive_json()
            primitive_outputs = primitive['produce']['output']

            params = aggregation.get('init_params')
            if params:
                init_params[primitive_name] = params

            if name.lower() == 'sigpro':
                primitive = MLPipeline([primitive], init_params={'sigpro.SigPro#1': params})
                primitive_outputs = primitive.get_outputs()

            # primitive_outputs = getattr(self, primitive_outputs)()
            if not isinstance(primitive_outputs, str):
                for output in primitive_outputs:
                    output = output['name']
                    outputs.append({
                        'name': f'{aggregation_name}.{output}',
                        'variable': f'{primitive_name}.{output}'
                    })

        outputs = {'default': outputs} if outputs else None

        self.pipeline = MLPipeline(
            primitives,
            init_params=init_params,
            outputs=outputs)
            


"""
Layer pipelines: interleave primitives with 'column renamer primitives' that take existing ft columns,
and set/add duplicates w/ the appropriate column names. 
"""

def build_linear_pipeline(self, transformations, aggregations, values_column_name='values', input_is_dataframe=True):
    pipeline_object = LinearPipeline(transformations, aggregations)
    pipeline_object.set_values_column_name(values_column_name)
    pipeline_object.accept_dataframe_input(input_is_dataframe)
    return pipeline_object


class LayerPipeline(Pipeline):

    def __init__(self, primitives, primitive_combinations):
        """
        Initialize a Layer pipeline with a list of primitive lists (layers, aggregation layer) and/or list of primitives w/ names, and the primitive combinations
        
        Args:
            primitives (Union[List, dict]): Dictionary mapping user-given string names of primitives to Primitive objects. 
                
            primitive_combinations (iterator):
                List of output features to be generated. Each combination in primitive_combinations should be a list or tuple of name strings found as keys in primitives.
                All lists should be of the same length and end with a single aggregation primitive. 

        Returns:
            LayerPipeline that generates the primitives in primitive_combinations.
        """
        super().__init__()
        if isinstance(primitives, list):
            primitives_dict = {}
            for primitive in primitives:
                if not(isinstance(primitive, Primitive)):
                    raise ValueError('Non-primitive specified in list primitives')

                if primitive.get_tag() in primitives_dict:
                    raise ValueError(f'Tag {primitive.get_tag()} duplicated in list primitives. When primitives is given as a list, all primitives must have distinct tags.')

                primitives_dict[primitive.get_tag()] = primitive
            self.primitives = primitives_dict
        elif isinstance(primitives, dict):
            for primitive_str in primitives:
                if not (isinstance(primitive_str, str)):
                    raise ValueError('Primitive names must be strings')
                elif not(isinstance(primitives[primitive_str], Primitive)):
                    raise ValueError('Non-primitive specified in dict primitives')
            self.primitives = primitives
        else:
            raise ValueError('primitives must be a dict or a list')

        self.primitive_combinations = [tuple(combination) for combination in primitive_combinations]

        length = len(primitive_combinations[0])
        for combination in primitive_combinations:
            if len(combination) != length:
                raise ValueError('Length of all feature sequences must be the same. Consider padding with an identity primitive.')

        self.num_layers = length
        
        used_primitives = {}
        for combination in self.primitive_combinations:
            for primitive_name in combination:
                if primitive_name not in self.primitives:
                    raise ValueError(f'Primitive {primitive_name} not found in given primitives')
                else:
                    used_primitives[primitive_name] = self.primitives[primitive_name]

        self.used_primitives = used_primitives
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """
        Segment of code that actually builds the pipeline.
        """
        prefixes = {}
        primitive_counter = Counter()
        final_primitives_list = []
        final_init_params = {}
        final_primitive_inputs = {}
        final_primitive_outputs = {}


        for layer in range(1, 1 + self.num_layers):
            for combination in self.primitive_combinations:
                if tuple(combination[:layer]) not in prefixes:
                    
                    final_primitve_str = combination[layer - 1]
                    final_primitive = self.used_primitives[final_primitive_str]

                    prefixes[(tuple(combination[:layer]))] =  final_primitive_str

                    final_primitive_name = final_primitive.get_name()
                    primitives_list.append(final_primitive.get_name()) #build the primitives list
                    primitive_counter[final_primitive_name] += 1
                    numbered_primitive_name = f'{final_primitive_name}#{primitive_counter[final_primitive_name]}'

                    final_init_params[numbered_primitive_name] = final_primitive.get_hyperparam_dict()['init_params']

                    final_primitive_inputs[numbered_primitive_name] = {}
                    final_primitive_outputs[numbered_primitive_name] = {}

                    for input_dict in final_primitive.get_inputs():
                        final_primitive_inputs[numbered_primitive_name][input_dict['name']] = f'{final_primitive_str}.{input_dict['name']}'
                    if layer == 1:
                        pass  #Don't pass in the name of the input column for the first-level primitives just yet.
                    else:
                        #Get input name
                        input_column_name = '.'.join(combination[:layer-1]) + f'.{layer-1}' #.amplitude_values'
                        final_primitive_inputs[numbered_primitive_name]['amplitude_values'] = input_column_name + '.amplitude_values'

                    output_column_name = '.'.join(combination[:layer])

                    if layer <= self.num_layers - 1:
                        output_column_name +=  + f'.{layer}' #don't include the layer number in the final output.

                    for output_dict in final_primitive.get_outputs():
                        final_primitive_outputs[numbered_primitive_name][output_dict['name']] = f'{output_column_name}.{output_dict['name']}'

        return MLPipeline( 
            primitives = final_primitives_list,
            init_params = final_init_params,
            input_names = final_primitive_inputs,
            output_names = final_primitive_outputs,
        )
    def get_primitives(self):
        return self.primitives.copy()
    def get_used_primitives(self):
        return self.used_primitives
    def get_primitive_combinations(self):
        return self.primitive_combinations.copy()

    def rename_primitives(self, renaming): 
        """
        Renames the specified primitives as given in renaming.

        """
        primitives_dict, primitive_combinations = self.get_primitives(), self.get_primitive_combinations()

        new_names = set()
        for old_name in renaming:
            new_name = renaming[old_name]
            if new_name in new_names:
                raise ValueError('Duplicate new names encountered in renaming')
            new_names.add(new_name)

            if new_name in primitives_dict and new_name not in renaming:
                raise ValueError(f'A primitive named {new_name} already exists in the pipeline')

        new_primitives = {}
        new_combinations = []

        for primitive_str in primitives_dict:
            if primitive_str in renaming:
                new_primitives[renaming[primitive_str]] = primitives_dict[primitive_str]
            else:
                new_primitives[primitive_str] = primitives_dict[primitive_str]
        for combination in primitive_combinations:
            new_combinations.append(   tuple(renaming[name] if name in renaming else name for name in combination)    )
        
        return LayerPipeline(primitives = new_primitives, primitive_combinations = new_combinations)

    def identity_pad(self, padding_length):

        pass

def build_tree_pipeline(transformation_layers, aggregation_layer):

    primitives_dict = {}
    all_layers = []

    if not(isinstance(transformation_layers, list)):
        raise ValueError('transformation_layers must be a list')
    for layer in transformation_layers:
        if isinstance(layer, dict):
            for primitive_str in layer:
                if not(isinstance(layer[primitives_str], Primitive)):
                    raise ValueError('Non-primitive specified in transformation_layers')
                else:
                    primitives_dict[primitive_str] = layer[primitive_str]
            all_layers.append([ps for ps in layer])
        elif isinstance(layer, list):
            for primitive in layer:
                if isinstance(primitive, Primitive):
                    primitives_dict[primitive.get_tag()] = primitive
                else:
                    raise ValueError('Non-primitive specified in transformation_layers')
            all_layers.append([pr.get_tag() for pr in layer])
        else:
            raise ValueError('Each layer in transformation_layers must be a list or dict')
        

    if isinstance(aggregation_layer, dict):
        for primitive_str in aggregation_layer:
            if not(isinstance(aggregation_layer[primitives_str], Primitive)):
                raise ValueError('Non-primitive specified in aggregation_layer')
            else:
                primitives_dict[primitive_str] = aggregation_layer[primitive_str]
        all_layers.append([ps for ps in aggregation_layer])
    elif isinstance(aggregation_layer, list):
        for primitive in aggregation_layer:
            if isinstance(primitive, Primitive):
                primitives_dict[primitive.get_tag()] = primitive
            else:
                raise ValueError('Non-primitive specified in aggregation_layer')
        all_layers.append([pr.get_tag() for pr in aggregation_layer])
    else:
        raise ValueError('aggregation_layer must be a list or a dict')

    primitive_combinations = list(product(*all_layers))
    return LayerPipeline(primitives = primitives_dict, primitive_combinations = primitive_combinations)

def build_layer_pipeline(primitives, primitive_combinations):
    return LayerPipeline(primitives = primitives, primitive_combinations = primitive_combinations)

def merge_pipelines(*pipelines, overwrite_labels = True, given_primitives = None):

    """
    Create a single pipeline that is the 'union' of several other pipelines; that is, the pipeline generates all features generated by at least one input pipeline.

    Important note: pipeline merges are done by merging the feature tuples of the primitive names, which are user-assigned strings or primitive tags. 
    If two primitives in distinct pipelines are given the same label, the default behavior is to pick the primitive object in the earliest pipeline with that particular label,
    and override all later objects. If this behavior is not desired, set overwrite_labels to False and explicitly specify all primitives as a dictionary in given_primitives.
    Alternatively, redefine pipelines and/or rename primitives so that all primitives that are not meant to be overloaded are given different string names.

    One thing to consider is that different pipelines may have different lengths. It may therefore be useful to construct 'identity-padded pipelines' to ensure same length.

    This identity padding can be done by inserting the identity primitive with a default name 'identity_padding' 
    (if this is found, repeatedly try identity_padding1, 2, etc until no duplicate exists. warn the user for each of padding, padding1, etc. is already existing in the pipeline.)

    We can also allow the user to automatically pad inputs. put the padded primitives at the end (do we even need to enforce this?)
    """

    if given_primitives is None:
        given_primitives = {}

    primitives_dict = {}
    primitive_combinations = set()

    if overwrite_labels:
        for pipeline in (pipelines)[::-1]: 

            primitives = pipeline.get_primitives()
            combinations = pipeline.get_primitive_combinations()

            for primitive_str in primitives:
                primitives_dict[primitive_str] = primitives[primitive_str]

            primitive_combinations.update(combinations)

        for primitive in given_primitives:
            primitives_dict[primitive] = given_primitives[primitive]

        return LayerPipeline(primitives = primitives_dict, primitive_combinations = list(primitive_combinations))

    else:
        for pipeline in reversed(pipelines): 
            combinations = pipeline.get_primitive_combinations()
            primitive_combinations.update(combinations)

        return LayerPipeline(primitives = given_primitives, primitive_combinations = list(primitive_combinations))
