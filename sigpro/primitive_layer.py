from sigpro import contributing, primitive, linear_pipeline
import json
import inspect
from mlblocks.discovery import load_primitive
from mlblocks.mlblock import import_object


class TransformationLayer:


    """
    #NOTE: CAN PASS IN INPUT NAMES WHEN CREATING THE MLPIPELINE!

        def _get_block_args(self, block_name, block_args, context):
            Get the arguments expected by the block method from the context.

            The arguments will be taken from the context using both the method
            arguments specification and the ``input_names`` given when the pipeline
            was created.

            ...


    input_names (dict) – dictionary that maps input variable names with the actual names expected by each primitive. This allows reusing the same input argument for multiple primitives that name it differently, as well as passing different values to primitives that expect arguments named similary.

    output_names (dict) – dictionary that maps output variable names with the name these variables will be given when stored in the context dictionary. This allows storing the output of different primitives in different variables, even if the primitive output name is the same one.

    """
    def __init__(self, transformations, input_columns, output_columns):

        """
        transformations is a list of transformations

        input_columns is a list of dictionaries mapping the basic primitive inputs as keys to the data-format primitive input strings as values
        output_columns is similar
        """
        if len(transformations) != len(input_columns):
            raise ValueError('transformations and input_columns should have the same length')

        if len(transformations) != len(output_columns):
            raise ValueError('transformations and output_columns should have the same length')

        self.primitive_inputs = {}
        self.primitive_outputs = {}
        self.layer_index = 0 #will be essential for uniquely naming each primitive
        self.transformations = transformations.copy()

        for i, primitive in enumerate(self.transformations):
            self.primitive_inputs[primitive] = input_columns[i]
            self.primitive_outputs[primitive] = output_columns[i]



    pass

    def set_layer_index(self, layer_index):
        self.layer_index = layer_index

    def change_primitive_inputs(self, new_inputs):

        """

        """
        if not isinstance(new_inputs, dict):
            raise ValueError('new_inputs should be a dict')
        for primitive in new_inputs:
            if primitive not in self.transformations:
                raise ValueError('Attempted to update a primitive input not found in the layer')
            else:
                self.primitive_inputs[primitive] = new_inputs[primitive].copy()


    def change_primitive_outputs(self, new_outputs):

        """

        """
        if not isinstance(new_outputs, dict):
            raise ValueError('new_outputs should be a dict')
        for primitive in new_outputs:
            if primitive not in self.transformations:
                raise ValueError('Attempted to update a primitive output not found in the layer')
            else:
                self.primitive_outputs[primitive] = new_outputs[primitive].copy()

    

    def _produce_dict_outputs(self):

        outputs = []

        for i, primitive in enumerate(self.transformations):

            primitive_name = str(primitive.primitive)
            primitive_outputs  = primitive.outputs.copy()

            for output_dict in primitive_outputs:
                output_name = self.primitive_outputs[i][output_dict['name']].copy()
                output_str = primitive_name + f'_Layer#{self.layer_index}_#{i}.' + output_dict['name']
                outputs.append([{'name': str(input_name), 'variable': str(output_str) }])
        pass



    def get_input_args(self):
        pass

    def get_output_args(self):
        pass

    def make_primitive_json(self):

        
        primitive_dict = {
            'name': primitive,
            'primitive': primitive,
            'classifiers': {
                'type': primitive_type,
                'subtype': primitive_subtype
            },
            'produce': {
                'args': primitive_args,
                'output': [
                    {
                        'name': primitive_output['name'],
                        'type': primitive_output['type'],
                    }
                    for primitive_output in primitive_outputs
                ],
            },
            'hyperparameters': {
                'fixed': fixed_hyperparameters,
                'tunable': tunable_hyperparameters
            }
        }


        {
            "name": "sigpro.TransformationLayer",
            "primitive": "sigpro.TransformationLayer",
            "classifiers": {
                "type": "",
                "subtype": ""
            },
            "produce": {
                "method": "process_signal",
                "args": "get_input_args",
                "output": "get_output_args"
            },
            "hyperparameters": {
                "fixed": {
                    "keep_columns": {
                        "type": "bool or list",
                        "default": false
                    },
                    "values_column_name": {
                        "type": "str",
                        "default": "values"
                    },
                    "transformations": {
                        "type": "list"
                    },
                    "aggregations": {
                        "type": "list"
                    },
                    "input_is_dataframe": {
                        "type": "bool",
                        "default": true
                    }
                }
            }
        }
        raise NotImplementedError
        return primitive_dict

    def process_signal(self, **args):
        pass

class AggregationLayer:

    pass


class LayerPipeline(linear_pipeline.Pipeline):


    def __init__(self, transformation_layers, aggregation_layer):
        self.transformation_layers = transformation_layers
        self.aggregation_layer = aggregation_layer

    def _build_pipeline(self): #do some dict manipulation here.
        pass

    pass