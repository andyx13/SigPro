# -*- coding: utf-8 -*-
"""Process Signals core functionality."""

from collections import Counter
from copy import deepcopy

import pandas as pd
from mlblocks import MLPipeline, load_primitive

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

class Pipeline:

    def __init__(self):
        self.values_column_name = 'values'
        self.input_is_dataframe = True

    def get_input_args(self):
        raise NotImplementedError

    def get_output_args(self):
        raise NotImplementedError

    def process_signal(self, data=None, window=None, time_index=None, groupby_index=None, feature_columns=None, values_column_name = 'values', keep_columns = False, **kwargs):
        raise NotImplementedError

    pass

    
"""
Analogue of sigpro.SigPro object in current use, takes in same arguments.
Only distinction is that we accept primitive objects, rather than dict inputs.
"""
class LinearPipeline(Pipeline): 

    def __init__(self, transformations, aggregations): 
        self.primitive = 'sigpro.SigPro' #change later.

        self.transformations = transformations
        self.aggregations = aggregations
        self.values_column_name = 'values' #values_column_name
        self.input_is_dataframe = True #input_is_dataframe

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
Layer pipelines: interleave primitives with 'column renamer primitives' that take existing ft columns,
and set/add duplicates w/ the appropriate column names. 
"""

def build_linear_pipeline(self, transformations, aggregations, values_column_name='values', input_is_dataframe=True):
    pipeline_object = LinearPipeline(transformations, aggregations)
    pipeline_object.set_values_column_name(values_column_name)
    pipeline_object.accept_dataframe_input(input_is_dataframe)
    return pipeline_object
